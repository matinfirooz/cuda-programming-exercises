#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

__global__ void matrix_add(const float* a,const float* b,float* c,int rows,int cols){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<cols&&y<rows){int i=y*cols+x;c[i]=a[i]+b[i];}
}
int main(){
  int rows=513,cols=777,n=rows*cols;size_t bytes=n*sizeof(float);std::vector<float>a(n),b(n),c(n);
  for(int i=0;i<n;i++){a[i]=i%97;b[i]=-(i%53);}
  float *da,*db,*dc;CUDA_CHECK(cudaMalloc(&da,bytes));CUDA_CHECK(cudaMalloc(&db,bytes));CUDA_CHECK(cudaMalloc(&dc,bytes));
  CUDA_CHECK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
  dim3 block(16,16),grid((cols+15)/16,(rows+15)/16);matrix_add<<<grid,block>>>(da,db,dc,rows,cols);check_kernel("matrix_add");
  CUDA_CHECK(cudaMemcpy(c.data(),dc,bytes,cudaMemcpyDeviceToHost));for(int i=0;i<n;i++)if(!nearly_equal(c[i],a[i]+b[i]))return 1;
  CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));std::cout<<"PASS\n";
}
