#include "cuda_utils.cuh"
#include <iostream>

// Exercise: Fused Scaled Dot-Product Attention
// Objective: Implement a small fused attention kernel: score QK^T, stable softmax, and weighted V without materializing the full score matrix in global memory.
// TODO: implement the CUDA kernel(s), host memory management, launch, and validation.
// Reference solution: ../../solutions/20_fused_attention.cu

int main() {
  std::cout << "TODO: solve Fused Scaled Dot-Product Attention\n";
  return 0;
}
