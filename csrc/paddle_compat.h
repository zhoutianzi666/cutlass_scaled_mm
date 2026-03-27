// Paddle compatibility header for porting from torch
#pragma once

#include "paddle/extension.h"
#include <cuda_runtime.h>
#include <optional>

// Helper to get raw data pointer from paddle tensor
template <typename T>
inline T* paddle_data_ptr(paddle::Tensor& tensor) {
  return const_cast<T*>(tensor.data<T>());
}

template <typename T>
inline const T* paddle_data_ptr(const paddle::Tensor& tensor) {
  return tensor.data<T>();
}

inline void* paddle_data_ptr_void(paddle::Tensor& tensor) {
  return const_cast<void*>(tensor.data());
}

inline const void* paddle_data_ptr_void(const paddle::Tensor& tensor) {
  return tensor.data();
}

// Get CUDA stream from paddle tensor
inline cudaStream_t paddle_get_cuda_stream(const paddle::Tensor& tensor) {
  return tensor.stream();
}

// Get device index
inline int paddle_get_device(const paddle::Tensor& tensor) {
  return tensor.place().GetDeviceId();
}

// Numel
inline int64_t paddle_numel(const paddle::Tensor& tensor) {
  int64_t n = 1;
  for (auto d : tensor.shape()) n *= d;
  return n;
}

// Check if contiguous (paddle tensors are always contiguous in practice)
inline bool paddle_is_contiguous(const paddle::Tensor& tensor) {
  return true;  // Paddle tensors from Python are contiguous
}

// Allocate empty GPU tensor (uint8)
inline paddle::Tensor paddle_empty_cuda(int64_t size, const paddle::Tensor& ref) {
  return paddle::empty({size}, paddle::DataType::UINT8, ref.place());
}
