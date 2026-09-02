#include "cuda_utils.cuh"
#include <iostream>
__global__ void transform(float* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=sqrtf(x[i]+1.0f);}
int main(){int dev=0;CUDA_CHECK(cudaSetDevice(dev));int N=1<<22;float* x;CUDA_CHECK(cudaMallocManaged(&x,N*4));for(int i=0;i<N;i++)x[i]=(float)(i%1000);CUDA_CHECK(cudaMemPrefetchAsync(x,N*4,dev));transform<<<(N+255)/256,256>>>(x,N);check_kernel("transform");CUDA_CHECK(cudaMemPrefetchAsync(x,N*4,cudaCpuDeviceId));CUDA_CHECK(cudaDeviceSynchronize());for(int i=0;i<N;i+=4093){float ref=sqrtf((float)(i%1000)+1);if(!nearly_equal(x[i],ref))return 1;}CUDA_CHECK(cudaFree(x));std::cout<<"PASS\n";}
