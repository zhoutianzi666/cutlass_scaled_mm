"""Test script for cutlass_scaled_mm Paddle custom op."""
import numpy as np
import paddle
import cutlass_scaled_mm_paddle


def quantize_to_fp8(x_fp32, scale):
    """Quantize float32 array to FP8 e4m3fn via Paddle."""
    x_scaled = np.clip(x_fp32 / scale, -448.0, 448.0)
    t = paddle.to_tensor(x_scaled, dtype=paddle.float32).cast(paddle.float8_e4m3fn).cuda()
    return t


def call_scaled_mm(a, b, a_scales, b_scales, bias=None):
    """Call the custom op and return the result as float32 numpy array."""
    result = cutlass_scaled_mm_paddle.cutlass_scaled_mm(a, b, a_scales, b_scales, bias)
    return result.cast(paddle.float32).numpy()


def test_fp8_per_tensor():
    """Test FP8 scaled matmul with per-tensor scales."""
    print("=" * 60)
    print("Test: FP8 per-tensor scaled matmul")
    print("=" * 60)

    M, N, K = 128, 256, 512
    np.random.seed(42)
    a_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_fp32 = np.random.randn(N, K).astype(np.float32) * 0.1

    # Reference: A @ B^T
    ref = a_fp32 @ b_fp32.T

    # Quantize
    sa = np.abs(a_fp32).max() / 448.0
    sb = np.abs(b_fp32).max() / 448.0
    a_fp8 = quantize_to_fp8(a_fp32, sa)
    b_fp8 = quantize_to_fp8(b_fp32, sb)

    # The kernel computes: out = a_scales * (A_fp8 @ B_fp8^T) * b_scales
    # A_fp8 ≈ A_fp32 / sa, so out ≈ a_scales * b_scales / (sa * sb) * (A_fp32 @ B_fp32^T)
    # For out ≈ ref, set a_scales = sa, b_scales = sb
    a_scales = paddle.to_tensor([sa], dtype=paddle.float32).cuda()
    b_scales = paddle.to_tensor([sb], dtype=paddle.float32).cuda()

    result = call_scaled_mm(a_fp8, b_fp8, a_scales, b_scales)

    abs_diff = np.abs(result - ref)
    # Use relative error only where |ref| > threshold to avoid near-zero noise
    mask = np.abs(ref) > 0.01
    rel_err_filtered = np.abs(result[mask] - ref[mask]) / np.abs(ref[mask])
    corr = np.corrcoef(ref.flatten(), result.flatten())[0, 1]

    print(f"  Shapes: a={a_fp8.shape}, b={b_fp8.shape}, out={result.shape}")
    print(f"  Ref range: [{ref.min():.4f}, {ref.max():.4f}]")
    print(f"  Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean abs error: {abs_diff.mean():.6f}")
    print(f"  Mean rel error (|ref|>0.01): {rel_err_filtered.mean():.6f}")

    # FP8 e4m3fn has ~8% relative error, correlation should be > 0.999
    passed = corr > 0.998 and rel_err_filtered.mean() < 0.15
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


def test_fp8_large():
    """Test with larger matrices."""
    print("\n" + "=" * 60)
    print("Test: FP8 per-tensor scaled matmul (large)")
    print("=" * 60)

    M, N, K = 1024, 2048, 1024
    np.random.seed(123)
    a_fp32 = np.random.randn(M, K).astype(np.float32) * 0.05
    b_fp32 = np.random.randn(N, K).astype(np.float32) * 0.05

    ref = a_fp32 @ b_fp32.T

    sa = np.abs(a_fp32).max() / 448.0
    sb = np.abs(b_fp32).max() / 448.0
    a_fp8 = quantize_to_fp8(a_fp32, sa)
    b_fp8 = quantize_to_fp8(b_fp32, sb)

    a_scales = paddle.to_tensor([sa], dtype=paddle.float32).cuda()
    b_scales = paddle.to_tensor([sb], dtype=paddle.float32).cuda()

    result = call_scaled_mm(a_fp8, b_fp8, a_scales, b_scales)

    abs_diff = np.abs(result - ref)
    mask = np.abs(ref) > 0.01
    rel_err_filtered = np.abs(result[mask] - ref[mask]) / np.abs(ref[mask])
    corr = np.corrcoef(ref.flatten(), result.flatten())[0, 1]

    print(f"  Shapes: a={a_fp8.shape}, b={b_fp8.shape}, out={result.shape}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean abs error: {abs_diff.mean():.6f}")
    print(f"  Mean rel error (|ref|>0.01): {rel_err_filtered.mean():.6f}")

    passed = corr > 0.998 and rel_err_filtered.mean() < 0.15
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


def test_fp8_with_bias():
    """Test FP8 scaled matmul with bias."""
    print("\n" + "=" * 60)
    print("Test: FP8 per-tensor scaled matmul with bias")
    print("=" * 60)

    M, N, K = 128, 256, 512
    np.random.seed(42)
    a_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_fp32 = np.random.randn(N, K).astype(np.float32) * 0.1
    bias_fp32 = np.random.randn(N).astype(np.float32) * 0.5

    ref = a_fp32 @ b_fp32.T + bias_fp32

    sa = np.abs(a_fp32).max() / 448.0
    sb = np.abs(b_fp32).max() / 448.0
    a_fp8 = quantize_to_fp8(a_fp32, sa)
    b_fp8 = quantize_to_fp8(b_fp32, sb)

    a_scales = paddle.to_tensor([sa], dtype=paddle.float32).cuda()
    b_scales = paddle.to_tensor([sb], dtype=paddle.float32).cuda()
    bias = paddle.to_tensor(bias_fp32, dtype=paddle.bfloat16).cuda()

    result = call_scaled_mm(a_fp8, b_fp8, a_scales, b_scales, bias)

    abs_diff = np.abs(result - ref)
    mask = np.abs(ref) > 0.01
    rel_err_filtered = np.abs(result[mask] - ref[mask]) / np.abs(ref[mask])
    corr = np.corrcoef(ref.flatten(), result.flatten())[0, 1]

    print(f"  Shapes: a={a_fp8.shape}, b={b_fp8.shape}, bias={bias.shape}, out={result.shape}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean abs error: {abs_diff.mean():.6f}")
    print(f"  Mean rel error (|ref|>0.01): {rel_err_filtered.mean():.6f}")

    passed = corr > 0.998 and rel_err_filtered.mean() < 0.15
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    paddle.set_device("gpu:0")

    results = []
    results.append(("FP8 per-tensor", test_fp8_per_tensor()))
    results.append(("FP8 large", test_fp8_large()))
    results.append(("FP8 with bias", test_fp8_with_bias()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n{'All tests PASSED!' if all_pass else 'Some tests FAILED!'}")
