#pragma once

#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>
#include "paddle/extension.h"

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    PD_CHECK(error == cutlass::Status::kSuccess,    \
             cutlassGetStatusString(error));         \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

inline int32_t get_sm_version_num() {
  int device = -1;
  cudaGetDevice(&device);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major * 10 + minor;
}

template <typename Kernel>
struct enable_sm75_to_sm80 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[75, 80).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[80, 89).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[89, 90).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm >= 90.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm90_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm90.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm100f_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm100f.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm100a_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 1000
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm100a.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm120_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ == 1200
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
