#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void write_indices(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

int main() {
  const int N = 1000;
  std::vector<int> h(N, -1);
  int* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d, N*sizeof(int)));
  int block=256, grid=(N+block-1)/block;
  write_indices<<<grid,block>>>(d,N); check_kernel("write_indices");
  CUDA_CHECK(cudaMemcpy(h.data(),d,N*sizeof(int),cudaMemcpyDeviceToHost));
  for(int i=0;i<N;i++) if(h[i]!=i){ std::cerr<<"Mismatch at "<<i<<"\n"; return 1; }
  CUDA_CHECK(cudaFree(d));
  std::cout<<"PASS: indexed "<<N<<" elements\n";
}
