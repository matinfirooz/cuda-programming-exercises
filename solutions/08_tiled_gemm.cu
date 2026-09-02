#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#define TILE 16
__global__ void gemm(const float* A,const float* B,float* C,int M,int K,int N){
  __shared__ float As[TILE][TILE],Bs[TILE][TILE];int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;float acc=0;
  for(int t=0;t<(K+TILE-1)/TILE;t++){int aCol=t*TILE+threadIdx.x,bRow=t*TILE+threadIdx.y;As[threadIdx.y][threadIdx.x]=(row<M&&aCol<K)?A[row*K+aCol]:0;Bs[threadIdx.y][threadIdx.x]=(bRow<K&&col<N)?B[bRow*N+col]:0;__syncthreads();for(int k=0;k<TILE;k++)acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];__syncthreads();}if(row<M&&col<N)C[row*N+col]=acc;
}
int main(){int M=129,K=193,N=117;std::vector<float>A(M*K),B(K*N),C(M*N);for(int i=0;i<M*K;i++)A[i]=(i%13-6)*.1f;for(int i=0;i<K*N;i++)B[i]=(i%17-8)*.05f;float *dA,*dB,*dC;CUDA_CHECK(cudaMalloc(&dA,A.size()*4));CUDA_CHECK(cudaMalloc(&dB,B.size()*4));CUDA_CHECK(cudaMalloc(&dC,C.size()*4));CUDA_CHECK(cudaMemcpy(dA,A.data(),A.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB,B.data(),B.size()*4,cudaMemcpyHostToDevice));dim3 block(TILE,TILE),grid((N+TILE-1)/TILE,(M+TILE-1)/TILE);gemm<<<grid,block>>>(dA,dB,dC,M,K,N);check_kernel("gemm");CUDA_CHECK(cudaMemcpy(C.data(),dC,C.size()*4,cudaMemcpyDeviceToHost));for(int r=0;r<M;r+=17)for(int c=0;c<N;c+=19){float ref=0;for(int k=0;k<K;k++)ref+=A[r*K+k]*B[k*N+c];if(!nearly_equal(C[r*N+c],ref,1e-3f,1e-3f)){std::cerr<<"FAIL\n";return 1;}}CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));std::cout<<"PASS\n";}
