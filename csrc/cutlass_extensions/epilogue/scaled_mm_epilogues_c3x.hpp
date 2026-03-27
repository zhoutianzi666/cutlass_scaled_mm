#pragma once

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp"
#include "paddle_compat.h"

namespace vllm::c3x {

using namespace cute;

template <typename T>
struct identity {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const { return lhs; }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct TrivialEpilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using Compute = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::Identity, ElementD, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute, Accum>;
  using ArgumentType = typename EVTCompute::Arguments;
  template <typename... Args>
  static ArgumentType prepare_args(Args... args) { return {}; }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  template <typename T>
  using ColOrScalarLoad = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      0, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T, bool EnableNullPtr = false>
  using ColLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  template <typename T, bool EnableNullPtr = false>
  using RowLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  template <typename T>
  using ColOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90ColOrScalarBroadcastArray<
          0, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90RowOrScalarBroadcastArray<
          0, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const paddle::Tensor& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = const_cast<T*>(reinterpret_cast<const T*>(tensor.data()));
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, paddle_numel(tensor) != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, ColLoad<T, true>> &&
                    !std::is_same_v<Descriptor, RowLoad<T, true>>);
      return Arguments{data_ptr};
    }
  }

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const paddle::optional<paddle::Tensor>& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? const_cast<T*>(reinterpret_cast<const T*>(tensor->data())) : nullptr;
    static_assert(std::is_same_v<Descriptor, ColLoad<T, true>> ||
                  std::is_same_v<Descriptor, RowLoad<T, true>>);
    return Arguments{data_ptr};
  }

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const T* const* data_ptr, bool do_broadcast) {
    using Arguments = typename Descriptor::Arguments;
    static_assert(std::is_same_v<Descriptor, ColOrScalarLoadArray<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoadArray<T>>);
    return Arguments{data_ptr, do_broadcast};
  }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales,
                                   const paddle::Tensor& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueColumnBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template ColLoad<ElementD>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales,
                                   const paddle::Tensor& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzp
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;
  using AzpWithAdj = typename SUPER::template RowLoad<int32_t>;
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Accum, AzpWithAdj>;
  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAzp>;
  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales,
                                   const paddle::Tensor& azp_adj,
                                   const paddle::optional<paddle::Tensor>& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_adj_args = SUPER::template args_from_tensor<AzpWithAdj, int32_t>(azp_adj);
    typename EVTComputeAzp::Arguments evt_azp_args{{}, azp_adj_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{b_args, evt_azp_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzpToken
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;
  using Azp = typename SUPER::template ColLoad<int32_t>;
  using AzpAdj = typename SUPER::template RowLoad<int32_t>;
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, int32_t, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Azp, AzpAdj>;
  using ComputeAcc = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAcc =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAcc, Accum, EVTComputeAzp>;
  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAcc>;
  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales,
                                   const paddle::Tensor& azp_adj,
                                   const paddle::Tensor& azp,
                                   const paddle::optional<paddle::Tensor>& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_args = SUPER::template args_from_tensor<Azp, int32_t>(azp);
    auto azp_adj_args = SUPER::template args_from_tensor<AzpAdj, int32_t>(azp_adj);
    typename EVTComputeAzp::Arguments evt_azp_args{azp_args, azp_adj_args, {}};
    typename EVTComputeAcc::Arguments evt_acc_args{{}, evt_azp_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{b_args, evt_acc_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogueArray
    : private ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoadArray<float>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;
  using ScaleAArray = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleBArray = typename SUPER::template RowOrScalarLoadArray<float>;
  static ArgumentType prepare_args(float const* const* a_scales_ptr,
                                   float const* const* b_scales_ptr,
                                   bool a_col_broadcast, bool b_row_broadcast) {
    auto a_args = SUPER::template args_from_tensor<ScaleAArray, float>(
        a_scales_ptr, a_col_broadcast);
    auto b_args = SUPER::template args_from_tensor<ScaleBArray, float>(
        b_scales_ptr, b_row_broadcast);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

};  // namespace vllm::c3x
