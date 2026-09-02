#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void reduce_sum(const float* in,float* out,int n){
  extern __shared__ float s[]; unsigned t=threadIdx.x; unsigned i=blockIdx.x*(blockDim.x*2)+t;
  float x=0.0f;if(i<n)x+=in[i];if(i+blockDim.x<n)x+=in[i+blockDim.x];s[t]=x;__syncthreads();
  for(unsigned stride=blockDim.x/2;stride>0;stride>>=1){if(t<stride)s[t]+=s[t+stride];__syncthreads();}if(t==0)out[blockIdx.x]=s[0];
}
int main(){
  int N=1<<20;std::vector<float>h(N);double ref=0;for(int i=0;i<N;i++){h[i]=(i%11)*0.1f;ref+=h[i];}
  float *dA,*dB;CUDA_CHECK(cudaMalloc(&dA,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dB,N*sizeof(float)));CUDA_CHECK(cudaMemcpy(dA,h.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  int n=N,block=256;float* in=dA;float* out=dB;while(n>1){int grid=(n+block*2-1)/(block*2);reduce_sum<<<grid,block,block*sizeof(float)>>>(in,out,n);check_kernel("reduce_sum");n=grid;std::swap(in,out);}float result;CUDA_CHECK(cudaMemcpy(&result,in,sizeof(float),cudaMemcpyDeviceToHost));
  std::cout<<"GPU="<<result<<" CPU="<<ref<<"\n";CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));if(std::fabs(result-ref)>0.01*std::fabs(ref))return 1;std::cout<<"PASS\n";
}
