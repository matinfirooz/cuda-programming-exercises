#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

inline void check_kernel(const char* name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed [" << name << "]: "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline bool nearly_equal(float a, float b, float atol=1e-4f, float rtol=1e-4f) {
  return std::fabs(a-b) <= atol + rtol*std::fabs(b);
}
