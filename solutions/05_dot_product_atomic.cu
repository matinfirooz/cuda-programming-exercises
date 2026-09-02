#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void dot_atomic(const float* a,const float* b,float* out,int n){
  extern __shared__ float s[]; unsigned t=threadIdx.x; int i=blockIdx.x*blockDim.x+t;
  float v=0.0f; for(int j=i;j<n;j+=blockDim.x*gridDim.x) v+=a[j]*b[j]; s[t]=v; __syncthreads();
  for(unsigned stride=blockDim.x/2;stride>0;stride>>=1){if(t<stride)s[t]+=s[t+stride];__syncthreads();}
  if(t==0) atomicAdd(out,s[0]);
}
int main(){
  int N=1<<20;size_t bytes=N*sizeof(float);std::vector<float>a(N),b(N);double ref=0;for(int i=0;i<N;i++){a[i]=(i%100)*0.01f;b[i]=(i%37)*0.02f;ref+=(double)a[i]*b[i];}
  float *da,*db,*dout;CUDA_CHECK(cudaMalloc(&da,bytes));CUDA_CHECK(cudaMalloc(&db,bytes));CUDA_CHECK(cudaMalloc(&dout,sizeof(float)));CUDA_CHECK(cudaMemset(dout,0,sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
  int block=256,grid=256;dot_atomic<<<grid,block,block*sizeof(float)>>>(da,db,dout,N);check_kernel("dot_atomic");float out;CUDA_CHECK(cudaMemcpy(&out,dout,sizeof(float),cudaMemcpyDeviceToHost));
  std::cout<<"GPU="<<out<<" CPU="<<ref<<"\n"; if(std::fabs(out-ref)>0.01*std::fabs(ref))return 1; CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dout));std::cout<<"PASS\n";
}
