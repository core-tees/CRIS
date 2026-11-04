#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include "CORE_SCL/DiscreteProblem/TransportProblem.hpp"

namespace transport_gpu {
inline void cuda_check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
inline void blas_check(cublasStatus_t e) {if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error "+std::to_string(int(e)));}
template<class T> struct Buffer {
  T* p=nullptr; size_t count=0;
  Buffer()=default; Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
  ~Buffer(){if(p)cudaFree(p);}
  void allocate(size_t n){if(p)cuda_check(cudaFree(p));count=n;cuda_check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));}
  void upload(const std::vector<T>& a){allocate(a.size());cuda_check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
struct GraphView {int n; const long long* ptr;const int* up;const double *vol,*out,*inj,*q;};
struct Scaling {double lo[3],scale[3],outlo,outscale;int tr[3];};
__device__ inline void quadratic(double u,double& f,double& df){
  if(u<0){f=df=0;return;} if(u>1){f=1;df=0;return;}
  double w=u*u,o=(1-u)*(1-u),dw=2*u,doo=-2*(1-u),d=w+.4*o;
  f=w/d;df=(dw*d-w*(dw+.4*doo))/(d*d);
}
// Match the archived CPU endpoint convention exactly, including its zero
// derivative at u==1. The one-sided mathematical derivative is singular.
__device__ inline void power_flux(double u,double exponent,double& f,double& df){
  if(u<0){f=df=0;return;} if(u>1){f=1;df=0;return;}
  double w=pow(u,2.),o=pow(1.-u,exponent);
  double dw=2.*w/(u+DBL_MIN),doo=-exponent*o/(1.-u+DBL_MIN),d=w+.4*o;
  f=w/d;df=(dw*d-w*(dw+.4*doo))/(d*d);
}
__global__ void flux_kernel(int n,const double* u,double* f,double* df,double exponent){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){if(exponent!=2.)power_flux(u[i],exponent,f[i],df[i]);else quadratic(u[i],f[i],df[i]);}}
__global__ void fixed_kernel(GraphView g,Scaling s,const double* old,double dt,double* beta,double* fixed){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;
  double b=dt*g.out[i]/g.vol[i],o=fmin(1.,fmax(0.,old[i]));beta[i]=b;
  fixed[i]=((s.tr[1]?log1p(b):b)-s.lo[1])*s.scale[1]-1.;
  fixed[g.n+i]=s.tr[0]==2?o:((s.tr[2]?log1p(o):o)-s.lo[2])*s.scale[2]-1.;
}
__global__ void inputs_kernel(GraphView g,Scaling s,const double* f,const double* fixed,const double* beta,double* x,double* dx,bool grad){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;double a=g.inj[i];
  for(long long k=g.ptr[i];k<g.ptr[i+1];k++)a+=g.q[k]*f[g.up[k]];
  a/=g.out[i]+DBL_MIN;
  double chain=1.;
  if(s.tr[0]==2){chain=beta[i]/(1.+beta[i]);a=fixed[g.n+i]/(1.+beta[i])+chain*a;}
  x[i]=((s.tr[0]==1?log1p(a):a)-s.lo[0])*s.scale[0]-1.;
  x[g.n+i]=fixed[i];x[2*g.n+i]=fixed[g.n+i];
  if(grad){dx[i]=chain*s.scale[0]/(s.tr[0]==1?1+a:1.);dx[g.n+i]=dx[2*g.n+i]=0;}
}
__global__ void activate_kernel(int n,int width,int act,const double* bias,double* y,double* dy,bool grad){
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(n)*width)return;
  double z=y[k]+bias[k/n];double v=act==2?tanh(z):(act==1?fmax(0.,z):z);y[k]=v;
  if(grad)dy[k]*=act==2?1-v*v:(act==1?(z>0?1.:0.):1.);
}
__global__ void output_kernel(GraphView g,Scaling s,const double* u,const double* y,const double* dy,double* r,double* da,bool grad){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;
  r[i]=u[i]-(s.outlo+s.outscale*(y[i]+1.));if(grad)da[i]=s.outscale*dy[i]/(g.out[i]+DBL_MIN);
}
__global__ void original_kernel(GraphView g,const double* u,const double* old,const double* f,double dt,double rate,double equilibrium,double width,int reaction_type,double* r){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;double incoming=g.inj[i];
  for(long long k=g.ptr[i];k<g.ptr[i+1];k++)incoming+=g.q[k]*f[g.up[k]];
  r[i]=u[i]-old[i]+dt/g.vol[i]*(g.out[i]*f[i]-incoming)+(rate==0?0.:dt*rate*(reaction_type==2?-u[i]*(1.-u[i]):tanh((u[i]-equilibrium)/width)));
}
__global__ void matrix_kernel(GraphView g,const double* u,const double* df,const double* da,double dt,bool learned,double rate,double equilibrium,double width,int reaction_type,double* values){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;long long start=i+g.ptr[i];
  double t=rate==0?0.:tanh((u[i]-equilibrium)/width);
  values[start]=learned?1.:1.+dt/g.vol[i]*g.out[i]*df[i]+dt*rate*(reaction_type==2?2.*u[i]-1.:(1.-t*t)/width);
  for(long long k=g.ptr[i];k<g.ptr[i+1];k++)values[i+1+k]=-(learned?da[i]:dt/g.vol[i])*g.q[k]*df[g.up[k]];
}
// Matrices are batch-by-feature, column major. Dense layers are FP64 cuBLAS GEMMs.
// No Python, host packing, or per-query allocation occurs in an evaluation.

__global__ void cheb_kernel(GraphView g,const double* tree,double exponent,const double* old,const double* beta,const double* f,const double* u,double* r,double* da,bool grad){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n)return;double a=g.inj[i];
 for(long long k=g.ptr[i];k<g.ptr[i+1];k++)a+=g.q[k]*f[g.up[k]];
 a/=g.out[i]+DBL_MIN;double v,d;cheb_value(tree,a,beta[i],fmin(1.,fmax(0.,old[i])),exponent,v,d);
 r[i]=u[i]-v;if(grad)da[i]=d/(g.out[i]+DBL_MIN);
}

struct Evaluator {
  Buffer<double> tree;
  transport::TransportProblem& cpu;GraphView g{};Scaling s{};cublasHandle_t handle{};
  Buffer<long long> ptr;Buffer<int> up;Buffer<double> vol,out,inj,q,old,beta,fixed,f,df,da,r,coo;
  Buffer<double> x0,x1,d0,d1;std::vector<Buffer<double>*> w,b;
  explicit Evaluator(transport::TransportProblem& p):cpu(p){
    if(!p.model.tree.empty())tree.upload(p.model.tree);
    ptr.upload(std::vector<long long>(p.g.ptr.begin(),p.g.ptr.end()));up.upload(p.g.upstream);
    vol.upload(p.g.volume);out.upload(p.g.outgoing);inj.upload(p.g.injection);q.upload(p.g.q);
    g={p.g.n,ptr.p,up.p,vol.p,out.p,inj.p,q.p};
    for(int j=0;j<3;j++){s.lo[j]=p.model.low[j];s.scale[j]=p.model.scale[j];s.tr[j]=p.model.transform[j];}
    s.outlo=p.model.outlow;s.outscale=p.model.outscale;
    int n=g.n,width=*std::max_element(p.model.width.begin(),p.model.width.end());
    old.allocate(n);beta.allocate(n);fixed.allocate(2*n);f.allocate(n);df.allocate(n);da.allocate(n);r.allocate(n);coo.allocate(n+p.g.q.size());
    x0.allocate(size_t(n)*width);x1.allocate(size_t(n)*width);d0.allocate(size_t(n)*width);d1.allocate(size_t(n)*width);
    for(size_t l=0;l<p.model.activation.size();l++){w.push_back(new Buffer<double>);b.push_back(new Buffer<double>);w.back()->upload(p.model.weights[l]);b.back()->upload(p.model.bias[l]);}
    blas_check(cublasCreate(&handle));blas_check(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
  }
  ~Evaluator(){if(handle)cublasDestroy(handle);for(auto a:w)delete a;for(auto a:b)delete a;}
  void timestep(cudaStream_t stream=0){
    cuda_check(cudaMemcpyAsync(old.p,cpu.old.data(),g.n*sizeof(double),cudaMemcpyHostToDevice,stream));
    fixed_kernel<<<(g.n+255)/256,256,0,stream>>>(g,s,old.p,cpu.dt,beta.p,fixed.p);cuda_check(cudaGetLastError());
  }
  void evaluate(const double* u,double* residual,bool grad,cudaStream_t stream=0){
    int blocks=(g.n+255)/256;flux_kernel<<<blocks,256,0,stream>>>(g.n,u,f.p,df.p,cpu.flux_exponent);
    if(cpu.learned && tree.count){
      cheb_kernel<<<blocks,256,0,stream>>>(g,tree.p,cpu.model.exponent,old.p,beta.p,f.p,u,residual,da.p,grad);
    }else if(cpu.learned){
      inputs_kernel<<<blocks,256,0,stream>>>(g,s,f.p,fixed.p,beta.p,x0.p,d0.p,grad);
      blas_check(cublasSetStream(handle,stream));double one=1,zero=0;double *x=x0.p,*y=x1.p,*dx=d0.p,*dy=d1.p;
      for(size_t l=0;l<w.size();l++){
        int ni=cpu.model.width[l],no=cpu.model.width[l+1];
        blas_check(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,g.n,no,ni,&one,x,g.n,w[l]->p,ni,&zero,y,g.n));
        if(grad)blas_check(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,g.n,no,ni,&one,dx,g.n,w[l]->p,ni,&zero,dy,g.n));
        activate_kernel<<<(size_t(g.n)*no+255)/256,256,0,stream>>>(g.n,no,cpu.model.activation[l],b[l]->p,y,dy,grad);
        std::swap(x,y);std::swap(dx,dy);
      }
      output_kernel<<<blocks,256,0,stream>>>(g,s,u,x,dx,residual,da.p,grad);
    }else original_kernel<<<blocks,256,0,stream>>>(g,u,old.p,f.p,cpu.dt,cpu.reaction_rate,cpu.reaction_equilibrium,cpu.reaction_width,cpu.reaction_type,residual);
    if(grad)matrix_kernel<<<blocks,256,0,stream>>>(g,u,df.p,da.p,cpu.dt,cpu.learned,cpu.reaction_rate,cpu.reaction_equilibrium,cpu.reaction_width,cpu.reaction_type,coo.p);
    cuda_check(cudaGetLastError());
  }
};
}
