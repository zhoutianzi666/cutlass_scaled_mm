#pragma once

// clang-format will break include orders
// clang-format off
#include "paddle/extension.h"
#include "paddle_compat.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
// clang-format on

namespace vllm::c3x {

static inline cute::Shape<int, int, int, int> get_problem_shape(
    paddle::Tensor const& a, paddle::Tensor const& b) {
  // a is row-major (M, K), b is row-major (N, K) which has same memory as
  // column-major (K, N) that CUTLASS expects
  int32_t m = a.shape()[0], n = b.shape()[0], k = a.shape()[1];
  return {m, n, k, 1};
}

template <typename GemmKernel>
void cutlass_gemm_caller(
    paddle::Tensor const& useless_tensor, cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info,
                                      scheduler};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = paddle::empty({(int64_t)workspace_size}, paddle::DataType::UINT8, useless_tensor.place());

  // Use default stream for the current device
  cudaStream_t stream = useless_tensor.stream();

  auto* workspace_ptr = const_cast<void*>(workspace.data());
  cutlass::Status status = gemm_op.run(args, workspace_ptr, stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(paddle::Tensor& out, paddle::Tensor const& a,
                         paddle::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = StrideC;
  using StrideAux = StrideC;

  typename GemmKernel::ProblemShape prob_shape = get_problem_shape(a, b);
  auto [M, N, K, L] = prob_shape;

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  StrideD d_stride =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
  StrideAux aux_stride = d_stride;

  auto a_ptr = static_cast<ElementAB*>(const_cast<void*>(a.data()));
  auto b_ptr = static_cast<ElementAB*>(const_cast<void*>(b.data()));
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(const_cast<void*>(out.data()));
  // auto d_ptr = static_cast<ElementC*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, d_stride};

  cutlass_gemm_caller<GemmKernel>(a, prob_shape, mainloop_args,
                                  epilogue_args);
}

}  // namespace vllm::c3x
