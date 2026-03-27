#pragma once

#include "paddle/extension.h"
#include "paddle_compat.h"

namespace vllm {

void cutlass_scaled_mm_sm90_fp8(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_sm90_int8(paddle::Tensor& out, paddle::Tensor const& a,
                                 paddle::Tensor const& b,
                                 paddle::Tensor const& a_scales,
                                 paddle::Tensor const& b_scales,
                                 paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm90_int8(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& b,
                                     paddle::Tensor const& a_scales,
                                     paddle::Tensor const& b_scales,
                                     paddle::Tensor const& azp_adj,
                                     paddle::optional<paddle::Tensor> const& azp,
                                     paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm90_fp8(paddle::Tensor& out,
                                          paddle::Tensor const& a,
                                          paddle::Tensor const& b,
                                          paddle::Tensor const& a_scales,
                                          paddle::Tensor const& b_scales);

void cutlass_scaled_mm_sm100_fp8(paddle::Tensor& out, paddle::Tensor const& a,
                                 paddle::Tensor const& b,
                                 paddle::Tensor const& a_scales,
                                 paddle::Tensor const& b_scales,
                                 paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_sm120_fp8(paddle::Tensor& out, paddle::Tensor const& a,
                                 paddle::Tensor const& b,
                                 paddle::Tensor const& a_scales,
                                 paddle::Tensor const& b_scales,
                                 paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm100_fp8(paddle::Tensor& out,
                                           paddle::Tensor const& a,
                                           paddle::Tensor const& b,
                                           paddle::Tensor const& a_scales,
                                           paddle::Tensor const& b_scales);

void cutlass_scaled_mm_blockwise_sm120_fp8(paddle::Tensor& out,
                                           paddle::Tensor const& a,
                                           paddle::Tensor const& b,
                                           paddle::Tensor const& a_scales,
                                           paddle::Tensor const& b_scales);
}  // namespace vllm
