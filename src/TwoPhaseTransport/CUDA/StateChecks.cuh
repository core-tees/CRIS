#pragma once
// Device reductions only. The evaluator and physics kernels remain unchanged.
namespace transport_gpu {
__global__ void check_trial_kernel(int n,const double* u,const double* r,const double* physical,unsigned long long* maxima,int* bad){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  if(!isfinite(u[i])||!isfinite(r[i])||!isfinite(physical[i]))atomicExch(bad,1);
  if(isfinite(r[i]))atomicMax(maxima,__double_as_longlong(fabs(r[i])));
  if(isfinite(physical[i]))atomicMax(maxima+1,__double_as_longlong(fabs(physical[i])));
}
struct TrialCheck {
  Buffer<double> physical;
  Buffer<unsigned long long> maxima;
  Buffer<int> bad;
  explicit TrialCheck(int n){physical.allocate(n);maxima.allocate(2);bad.allocate(1);}
  void evaluate(Evaluator& e,const double* u,double& residual_inf,double& physical_inf,bool& finite,cudaStream_t stream){
    e.evaluate(u,e.r.p,false,stream);
    original_kernel<<<(e.g.n+255)/256,256,0,stream>>>(e.g,u,e.old.p,e.f.p,e.cpu.dt,physical.p);
    cuda_check(cudaMemsetAsync(maxima.p,0,2*sizeof(unsigned long long),stream));
    cuda_check(cudaMemsetAsync(bad.p,0,sizeof(int),stream));
    check_trial_kernel<<<(e.g.n+255)/256,256,0,stream>>>(e.g.n,u,e.r.p,physical.p,maxima.p,bad.p);
    cuda_check(cudaGetLastError());
    double norms[2];int invalid;
    cuda_check(cudaMemcpyAsync(norms,maxima.p,2*sizeof(double),cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(&invalid,bad.p,sizeof(int),cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaStreamSynchronize(stream));
    residual_inf=norms[0];physical_inf=norms[1];finite=invalid==0;
  }
};
}
