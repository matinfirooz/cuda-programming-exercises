#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void saxpy(float a,const float* x,float* y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) y[i]=a*x[i]+y[i];
}
int main(){
  const int N=1<<20; const float a=2.5f; size_t bytes=N*sizeof(float);
  std::vector<float>x(N),y(N),ref(N); for(int i=0;i<N;i++){x[i]=i*0.001f;y[i]=1.0f-i*0.0001f;ref[i]=a*x[i]+y[i];}
  float *dx,*dy; CUDA_CHECK(cudaMalloc(&dx,bytes));CUDA_CHECK(cudaMalloc(&dy,bytes));
  CUDA_CHECK(cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dy,y.data(),bytes,cudaMemcpyHostToDevice));
  saxpy<<<(N+255)/256,256>>>(a,dx,dy,N);check_kernel("saxpy"); CUDA_CHECK(cudaMemcpy(y.data(),dy,bytes,cudaMemcpyDeviceToHost));
  for(int i=0;i<N;i++) if(!nearly_equal(y[i],ref[i],1e-4f,1e-4f)){std::cerr<<"FAIL\n";return 1;}
  CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));std::cout<<"PASS\n";
}
