#pragma once

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp"
#include "paddle_compat.h"

namespace vllm::c2x {

using namespace cute;

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  template <typename T>
  using ColOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using ColLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using RowOrZeroLoad =
      cutlass::epilogue::threadblock::VisitorRowOrZeroBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const paddle::Tensor& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = const_cast<T*>(reinterpret_cast<const T*>(tensor.data()));
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, paddle_numel(tensor) != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
      return Arguments{data_ptr};
    }
  }

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const paddle::optional<paddle::Tensor>& tensor) {
    static_assert(std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? const_cast<T*>(reinterpret_cast<const T*>(tensor->data())) : nullptr;
    return Arguments{data_ptr};
  }
};

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(const paddle::Tensor& a_scales,
                                   const paddle::Tensor& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBias
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 protected:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA,
                                                             EVTCompute0, Bias>;
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

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBiasAzp
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowOrZeroLoad<ElementD>;
  using AzpWithAdj = typename SUPER::template RowLoad<int32_t>;
  using ComputeAzp = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAzp =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAzp, Accum, AzpWithAdj>;
  using ComputeScaleB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeScaleB =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleB, ScaleB,
                                              EVTComputeAzp>;
  using ComputeScaleBiasA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleBiasA, ScaleA,
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

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBiasAzpToken
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowOrZeroLoad<ElementD>;
  using Azp = typename SUPER::template ColLoad<int32_t>;
  using AzpAdj = typename SUPER::template RowLoad<int32_t>;
  using ComputeAzp = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, int32_t, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAzp =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAzp, Azp, AzpAdj>;
  using ComputeAcc = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeAcc =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAcc, Accum, EVTComputeAzp>;
  using ComputeScaleB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeScaleB =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleB, ScaleB,
                                              EVTComputeAcc>;
  using ComputeScaleBiasA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleBiasA, ScaleA,
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

};  // namespace vllm::c2x
