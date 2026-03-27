#include "paddle/extension.h"
#include "paddle_compat.h"
#include "cuda_utils.h"
#include "cutlass_extensions/common.hpp"

template <typename Fp8Func, typename Int8Func, typename BlockwiseFunc>
void dispatch_scaled_mm(paddle::Tensor& c, paddle::Tensor const& a,
                        paddle::Tensor const& b, paddle::Tensor const& a_scales,
                        paddle::Tensor const& b_scales,
                        paddle::optional<paddle::Tensor> const& bias,
                        Fp8Func fp8_func, Int8Func int8_func,
                        BlockwiseFunc blockwise_func) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

  int M = a.shape()[0], N = b.shape()[0], K = a.shape()[1];

  if ((paddle_numel(a_scales) == 1 || paddle_numel(a_scales) == a.shape()[0]) &&
      (paddle_numel(b_scales) == 1 || paddle_numel(b_scales) == b.shape()[0])) {
    // Standard per-tensor/per-token/per-channel scaling
    PD_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      fp8_func(c, a, b, a_scales, b_scales, bias);
    } else {
      PD_CHECK(a.dtype() == paddle::DataType::INT8);
      if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
        int8_func(c, a, b, a_scales, b_scales, bias);
      } else {
        int32_t version_num = get_sm_version_num();
        PD_CHECK(
            false, "Int8 not supported on SM", version_num,
            ". Use FP8 quantization instead, or run on older arch (SM < 100).");
      }
    }
  } else {
    PD_CHECK((int)a_scales.shape().size() == 2, "a scale must be 2d tensor.");
    PD_CHECK((int)b_scales.shape().size() == 2, "b scale must be 2d tensor.");
    int32_t version_num = get_sm_version_num();
    if (version_num >= 90) {
      PD_CHECK(
          a.shape()[0] == a_scales.shape()[0] &&
              cuda_utils::ceil_div(a.shape()[1], int64_t(128)) == a_scales.shape()[1],
          "a_scale_group_shape must be [1, 128].");
      PD_CHECK(
          cuda_utils::ceil_div(b.shape()[1], int64_t(128)) == b_scales.shape()[0] &&
              cuda_utils::ceil_div(b.shape()[0], int64_t(128)) == b_scales.shape()[1],
          "b_scale_group_shape must be [128, 128].");
    }

    PD_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");
    blockwise_func(c, a, b, a_scales, b_scales);
  }
}
