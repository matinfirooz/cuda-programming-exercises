#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#define TILE 32
#define BLOCK_ROWS 8
__global__ void transpose(const float* in,float* out,int rows,int cols){
  __shared__ float tile[TILE][TILE+1];int x=blockIdx.x*TILE+threadIdx.x,y=blockIdx.y*TILE+threadIdx.y;
  for(int j=0;j<TILE;j+=BLOCK_ROWS)if(x<cols&&y+j<rows)tile[threadIdx.y+j][threadIdx.x]=in[(y+j)*cols+x];__syncthreads();
  x=blockIdx.y*TILE+threadIdx.x;y=blockIdx.x*TILE+threadIdx.y;
  for(int j=0;j<TILE;j+=BLOCK_ROWS)if(x<rows&&y+j<cols)out[(y+j)*rows+x]=tile[threadIdx.x][threadIdx.y+j];
}
int main(){int rows=511,cols=769,n=rows*cols;std::vector<float>a(n),b(n);for(int i=0;i<n;i++)a[i]=i;float *da,*db;CUDA_CHECK(cudaMalloc(&da,n*sizeof(float)));CUDA_CHECK(cudaMalloc(&db,n*sizeof(float)));CUDA_CHECK(cudaMemcpy(da,a.data(),n*sizeof(float),cudaMemcpyHostToDevice));dim3 block(TILE,BLOCK_ROWS),grid((cols+TILE-1)/TILE,(rows+TILE-1)/TILE);transpose<<<grid,block>>>(da,db,rows,cols);check_kernel("transpose");CUDA_CHECK(cudaMemcpy(b.data(),db,n*sizeof(float),cudaMemcpyDeviceToHost));for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)if(b[c*rows+r]!=a[r*cols+c]){std::cerr<<"FAIL\n";return 1;}CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));std::cout<<"PASS\n";}
