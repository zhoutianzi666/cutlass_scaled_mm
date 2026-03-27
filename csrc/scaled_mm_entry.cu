#include <cudaTypedefs.h>
#include "paddle/extension.h"
#include "paddle_compat.h"
#include "cutlass_extensions/common.hpp"

// Forward declarations for SM-specific kernels
void cutlass_scaled_mm_sm75(paddle::Tensor& c, const paddle::Tensor& a,
                            const paddle::Tensor& b,
                            const paddle::Tensor& a_scales,
                            const paddle::Tensor& b_scales,
                            const paddle::optional<paddle::Tensor>& bias);

void cutlass_scaled_mm_sm80(paddle::Tensor& c, const paddle::Tensor& a,
                            const paddle::Tensor& b,
                            const paddle::Tensor& a_scales,
                            const paddle::Tensor& b_scales,
                            const paddle::optional<paddle::Tensor>& bias);

void cutlass_scaled_mm_sm89(paddle::Tensor& c, const paddle::Tensor& a,
                            const paddle::Tensor& b,
                            const paddle::Tensor& a_scales,
                            const paddle::Tensor& b_scales,
                            const paddle::optional<paddle::Tensor>& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_sm90(paddle::Tensor& c, const paddle::Tensor& a,
                            const paddle::Tensor& b,
                            const paddle::Tensor& a_scales,
                            const paddle::Tensor& b_scales,
                            const paddle::optional<paddle::Tensor>& bias);
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
void cutlass_scaled_mm_sm100(paddle::Tensor& c, const paddle::Tensor& a,
                             const paddle::Tensor& b,
                             const paddle::Tensor& a_scales,
                             const paddle::Tensor& b_scales,
                             const paddle::optional<paddle::Tensor>& bias);
#endif

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
void cutlass_scaled_mm_sm120(paddle::Tensor& c, const paddle::Tensor& a,
                             const paddle::Tensor& b,
                             const paddle::Tensor& a_scales,
                             const paddle::Tensor& b_scales,
                             const paddle::optional<paddle::Tensor>& bias);
#endif

void cutlass_scaled_mm_azp_sm75(paddle::Tensor& c, const paddle::Tensor& a,
                                const paddle::Tensor& b,
                                const paddle::Tensor& a_scales,
                                const paddle::Tensor& b_scales,
                                const paddle::Tensor& azp_adj,
                                const paddle::optional<paddle::Tensor>& azp,
                                const paddle::optional<paddle::Tensor>& bias);

void cutlass_scaled_mm_azp_sm80(paddle::Tensor& c, const paddle::Tensor& a,
                                const paddle::Tensor& b,
                                const paddle::Tensor& a_scales,
                                const paddle::Tensor& b_scales,
                                const paddle::Tensor& azp_adj,
                                const paddle::optional<paddle::Tensor>& azp,
                                const paddle::optional<paddle::Tensor>& bias);

void cutlass_scaled_mm_azp_sm89(paddle::Tensor& c, const paddle::Tensor& a,
                                const paddle::Tensor& b,
                                const paddle::Tensor& a_scales,
                                const paddle::Tensor& b_scales,
                                const paddle::Tensor& azp_adj,
                                const paddle::optional<paddle::Tensor>& azp,
                                const paddle::optional<paddle::Tensor>& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_azp_sm90(paddle::Tensor& c, const paddle::Tensor& a,
                                const paddle::Tensor& b,
                                const paddle::Tensor& a_scales,
                                const paddle::Tensor& b_scales,
                                const paddle::Tensor& azp_adj,
                                const paddle::optional<paddle::Tensor>& azp,
                                const paddle::optional<paddle::Tensor>& bias);
#endif


// Main dispatch function - this is the entry point for the Paddle custom op
void cutlass_scaled_mm(paddle::Tensor& c, const paddle::Tensor& a,
                       const paddle::Tensor& b, const paddle::Tensor& a_scales,
                       const paddle::Tensor& b_scales,
                       const paddle::optional<paddle::Tensor>& bias) {
  // Checks for conformality
  // Note: In Paddle, B is stored as row-major (N, K), which has the same memory
  // layout as column-major (K, N) that CUTLASS expects.
  // So b.shape() = {N, K}, a.shape() = {M, K}, c.shape() = {M, N}
  PD_CHECK((int)a.shape().size() == 2 && (int)b.shape().size() == 2 && (int)c.shape().size() == 2);
  PD_CHECK(c.shape()[0] == a.shape()[0] && a.shape()[1] == b.shape()[1] &&
           b.shape()[0] == c.shape()[1]);

  // Check for strides and alignment
  // In Paddle, tensors from Python are always row-major contiguous
  // For column-major b, we rely on the caller to ensure correct layout
  // (Paddle doesn't expose strides directly, so we check shapes)

  if (bias) {
    PD_CHECK(paddle_numel(*bias) == b.shape()[0] && (int)bias->shape().size() == 1);
  }

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
  if (version_num >= 120) {
    cutlass_scaled_mm_sm120(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    cutlass_scaled_mm_sm100(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90 && version_num < 100) {
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }
  if (version_num >= 80) {
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }
  if (version_num >= 75) {
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  PD_CHECK(false,
           "No compiled cutlass_scaled_mm for CUDA device capability: ",
           version_num);
}


// Paddle custom op wrapper - takes tensors and returns output
std::vector<paddle::Tensor> CutlassScaledMM(
    const paddle::Tensor& a,
    const paddle::Tensor& b,
    const paddle::Tensor& a_scales,
    const paddle::Tensor& b_scales,
    const paddle::optional<paddle::Tensor>& bias) {
  // Determine output dtype: bf16 or fp16 based on scales dtype or explicit choice
  // The output is created here with same dtype as what the kernel expects
  paddle::DataType out_dtype;
  if (a.dtype() == paddle::DataType::INT8) {
    out_dtype = paddle::DataType::BFLOAT16;  // default for int8
  } else {
    out_dtype = paddle::DataType::BFLOAT16;  // default for fp8
  }

  // If bias is provided, match its dtype
  if (bias) {
    out_dtype = bias->dtype();
  }

  // b.shape() = {N, K} in Paddle (row-major, same memory as column-major (K,N))
  int64_t m = a.shape()[0];
  int64_t n = b.shape()[0];
  auto c = paddle::empty({m, n}, out_dtype, a.place());

  cutlass_scaled_mm(c, a, b, a_scales, b_scales, bias);
  return {c};
}

// Shape inference
std::vector<std::vector<int64_t>> CutlassScaledMMInferShape(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape,
    const std::vector<int64_t>& a_scales_shape,
    const std::vector<int64_t>& b_scales_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape) {
  return {{a_shape[0], b_shape[0]}};
}

// Dtype inference
std::vector<paddle::DataType> CutlassScaledMMInferDtype(
    const paddle::DataType& a_dtype,
    const paddle::DataType& b_dtype,
    const paddle::DataType& a_scales_dtype,
    const paddle::DataType& b_scales_dtype,
    const paddle::optional<paddle::DataType>& bias_dtype) {
  if (bias_dtype) {
    return {*bias_dtype};
  }
  return {paddle::DataType::BFLOAT16};
}

// Register the Paddle custom op
PD_BUILD_OP(cutlass_scaled_mm)
    .Inputs({"a", "b", "a_scales", "b_scales", paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(CutlassScaledMM))
    .SetInferShapeFn(PD_INFER_SHAPE(CutlassScaledMMInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CutlassScaledMMInferDtype));
