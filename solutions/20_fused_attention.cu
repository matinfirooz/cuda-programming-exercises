#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <cmath>
// Educational fused attention for N<=128 and D<=64. One CUDA block computes one query row.
__global__ void fused_attention(const float* Q,const float* K,const float* V,float* O,int N,int D){
  extern __shared__ float sm[]; float* scores=sm; float* red=sm+N; int q=blockIdx.x,t=threadIdx.x;if(q>=N)return;
  // Score each key: dot(Q[q],K[k]) / sqrt(D)
  for(int k=t;k<N;k+=blockDim.x){float s=0;for(int d=0;d<D;d++)s+=Q[q*D+d]*K[k*D+d];scores[k]=s*rsqrtf((float)D);}__syncthreads();
  // Stable max reduction over scores.
  float local=-CUDART_INF_F;for(int k=t;k<N;k+=blockDim.x)local=fmaxf(local,scores[k]);red[t]=local;__syncthreads();for(int st=blockDim.x/2;st;st>>=1){if(t<st)red[t]=fmaxf(red[t],red[t+st]);__syncthreads();}float m=red[0];
  // Exponentiate in-place and sum.
  float sum=0;for(int k=t;k<N;k+=blockDim.x){float e=expf(scores[k]-m);scores[k]=e;sum+=e;}red[t]=sum;__syncthreads();for(int st=blockDim.x/2;st;st>>=1){if(t<st)red[t]+=red[t+st];__syncthreads();}float inv=1.0f/red[0];
  // Weighted value accumulation: one or more output dimensions per thread.
  for(int d=t;d<D;d+=blockDim.x){float acc=0;for(int k=0;k<N;k++)acc+=(scores[k]*inv)*V[k*D+d];O[q*D+d]=acc;}
}
int main(){const int N=64,D=32;std::vector<float>Q(N*D),K(N*D),V(N*D),O(N*D),R(N*D);for(int i=0;i<N*D;i++){Q[i]=sinf(i*.017f);K[i]=cosf(i*.013f);V[i]=sinf(i*.009f)*.5f;}for(int q=0;q<N;q++){std::vector<double>s(N);double m=-1e300;for(int k=0;k<N;k++){double v=0;for(int d=0;d<D;d++)v+=(double)Q[q*D+d]*K[k*D+d];v/=std::sqrt((double)D);s[k]=v;m=std::max(m,v);}double z=0;for(double&v:s){v=std::exp(v-m);z+=v;}for(int d=0;d<D;d++){double a=0;for(int k=0;k<N;k++)a+=(s[k]/z)*V[k*D+d];R[q*D+d]=(float)a;}}
  float *dQ,*dK,*dV,*dO;size_t bytes=N*D*4;CUDA_CHECK(cudaMalloc(&dQ,bytes));CUDA_CHECK(cudaMalloc(&dK,bytes));CUDA_CHECK(cudaMalloc(&dV,bytes));CUDA_CHECK(cudaMalloc(&dO,bytes));CUDA_CHECK(cudaMemcpy(dQ,Q.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dK,K.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dV,V.data(),bytes,cudaMemcpyHostToDevice));int block=128;fused_attention<<<N,block,(N+block)*sizeof(float)>>>(dQ,dK,dV,dO,N,D);check_kernel("fused_attention");CUDA_CHECK(cudaMemcpy(O.data(),dO,bytes,cudaMemcpyDeviceToHost));for(int i=0;i<N*D;i++)if(!nearly_equal(O[i],R[i],2e-4f,2e-4f)){std::cerr<<"FAIL at "<<i<<" gpu="<<O[i]<<" cpu="<<R[i]<<"\n";return 1;}CUDA_CHECK(cudaFree(dQ));CUDA_CHECK(cudaFree(dK));CUDA_CHECK(cudaFree(dV));CUDA_CHECK(cudaFree(dO));std::cout<<"PASS: fused attention N="<<N<<" D="<<D<<"\n";}
