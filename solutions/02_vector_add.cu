#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void vector_add(const float* a,const float* b,float* c,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) c[i]=a[i]+b[i];
}
int main(){
  const int N=1<<20; size_t bytes=N*sizeof(float);
  std::vector<float>a(N),b(N),c(N);
  for(int i=0;i<N;i++){a[i]=0.5f*i;b[i]=2.0f-i*0.25f;}
  float *da,*db,*dc; CUDA_CHECK(cudaMalloc(&da,bytes)); CUDA_CHECK(cudaMalloc(&db,bytes)); CUDA_CHECK(cudaMalloc(&dc,bytes));
  CUDA_CHECK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
  int block=256,grid=(N+block-1)/block; vector_add<<<grid,block>>>(da,db,dc,N); check_kernel("vector_add");
  CUDA_CHECK(cudaMemcpy(c.data(),dc,bytes,cudaMemcpyDeviceToHost));
  for(int i=0;i<N;i++) if(!nearly_equal(c[i],a[i]+b[i])){std::cerr<<"FAIL at "<<i<<"\n";return 1;}
  CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc)); std::cout<<"PASS\n";
}
