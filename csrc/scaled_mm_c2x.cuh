#pragma once
#include <stddef.h>
#include "paddle/extension.h"
#include "paddle_compat.h"

// clang-format will break include orders
// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
// clang-format on

using namespace cute;

/*
   Epilogues defined in,
   csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp
   must contain a public type named EVTCompute of type Sm80EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

namespace vllm {
template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_,
          template <typename, typename> typename Epilogue_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages,
          typename FP8MathOperator = cutlass::arch::OpMultiplyAdd>
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                FP8MathOperator>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // These are the minimum alignments needed for the kernels to compile
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType =
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp,
      Arch,
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

template <typename Gemm, typename... EpilogueArgs>
inline void cutlass_gemm_caller(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.shape()[0];
  int32_t n = b.shape()[0];  // b is (N, K) row-major = column-major (K, N)
  int32_t k = a.shape()[1];
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  // For A (row-major): lda = K
  // For B (column-major stored as row-major (N,K)): ldb = N (leading dimension of column-major)
  // For C (row-major): ldc = N
  int64_t lda = a.shape()[1];  // K
  int64_t ldb = b.shape()[0];  // N (column-major leading dim)
  int64_t ldc = out.shape()[1];  // N

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data());
  auto b_ptr = static_cast<ElementAB const*>(b.data());
  auto c_ptr = static_cast<ElementD*>(const_cast<void*>(out.data()));

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,  // universal mode
      problem_size,                                           // problem size
      1,                                                      // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  // Launch the CUTLASS GEMM kernel.
  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = paddle::empty({(int64_t)workspace_size}, paddle::DataType::UINT8, a.place());

  auto stream = a.stream();

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, const_cast<void*>(workspace.data()), stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
inline void fallback_cutlass_gemm_caller(paddle::Tensor& out,
                                         paddle::Tensor const& a,
                                         paddle::Tensor const& b,
                                         EpilogueArgs&&... args) {
  // In some cases, the GPU isn't able to accommodate the
  // shared memory requirements of the Gemm. In such cases, use
  // the FallbackGemm instead.
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= max_shared_mem_per_block_opt_in) {
    return cutlass_gemm_caller<Gemm>(out, a, b,
                                     std::forward<EpilogueArgs>(args)...);
  } else {
    PD_CHECK(fallback_gemm_shared_mem_size <=
                max_shared_mem_per_block_opt_in);
    return cutlass_gemm_caller<FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
