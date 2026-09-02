#include "cuda_utils.cuh"
#include <vector>
#include <iostream>

// Work-efficient Blelloch scan. This teaching version uses one 1024-thread block,
// so N must be <= 1024. Padding entries are initialized to zero.
__global__ void inclusive_scan(const float* in,float* out,int n){
  extern __shared__ float s[];
  int t=threadIdx.x;
  float original=(t<n)?in[t]:0.0f;
  s[t]=original;
  __syncthreads();

  // Upsweep (reduce) phase.
  for(int offset=1; offset<blockDim.x; offset<<=1){
    int idx=(t+1)*offset*2-1;
    if(idx<blockDim.x) s[idx]+=s[idx-offset];
    __syncthreads();
  }

  // Convert total sum to an exclusive scan seed.
  if(t==0) s[blockDim.x-1]=0.0f;
  __syncthreads();

  // Downsweep phase.
  for(int offset=blockDim.x>>1; offset>0; offset>>=1){
    int idx=(t+1)*offset*2-1;
    if(idx<blockDim.x){
      float left=s[idx-offset];
      s[idx-offset]=s[idx];
      s[idx]+=left;
    }
    __syncthreads();
  }

  // Exclusive + original value = inclusive scan.
  if(t<n) out[t]=s[t]+original;
}

int main(){
  const int N=1000;
  std::vector<float>x(N),y(N),ref(N);
  float acc=0;
  for(int i=0;i<N;i++){x[i]=(i%5)+1;acc+=x[i];ref[i]=acc;}
  float *dx,*dy;
  CUDA_CHECK(cudaMalloc(&dx,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy,N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx,x.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  inclusive_scan<<<1,1024,1024*sizeof(float)>>>(dx,dy,N);
  check_kernel("inclusive_scan");
  CUDA_CHECK(cudaMemcpy(y.data(),dy,N*sizeof(float),cudaMemcpyDeviceToHost));
  for(int i=0;i<N;i++) if(!nearly_equal(y[i],ref[i])){std::cerr<<"FAIL at "<<i<<"\n";return 1;}
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
  std::cout<<"PASS\n";
}
