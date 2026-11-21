#pragma once
#include <petscsnes.h>
#include <omp.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include "CORE_SCL/DiscreteProblem/TransportProblem.hpp"
#include "CRIS/ChebyshevVerification.hpp"
#ifdef CRIS_PETSC_CUDA
#include <petscdevice.h>
#include <memory>
#include "TwoPhaseTransport/CUDA/TransportEvaluator.cuh"
#include "TwoPhaseTransport/CUDA/StateChecks.cuh"
#endif

using transport::TransportProblem;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct Context {
  TransportProblem p; std::vector<double> scratch; long long feval=0,jeval=0;
  double fseconds=0,jseconds=0;
#ifdef CRIS_PETSC_CUDA
  std::unique_ptr<transport_gpu::Evaluator> gpu;
#endif
  Context(std::string g,std::string m,bool l):p(g,m,l),scratch(p.g.n){}
};
#ifdef CRIS_PETSC_CUDA
PetscErrorCode current_stream(cudaStream_t& stream) {
  PetscDeviceContext ctx;void* handle;
  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetCurrentContext(&ctx));
  PetscCall(PetscDeviceContextGetStreamHandle(ctx,&handle));
  stream=*static_cast<cudaStream_t*>(handle);
  transport_gpu::cuda_check(cudaStreamSynchronize(stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
PetscErrorCode residual(SNES,Vec x,Vec r,void* raw) {
  auto& c=*static_cast<Context*>(raw); auto start=Clock::now();
  const PetscScalar* u; PetscScalar* y;
  PetscFunctionBeginUser;
#ifdef CRIS_PETSC_CUDA
  cudaStream_t stream;PetscCall(current_stream(stream));
  PetscCall(VecCUDAGetArrayRead(x,&u));PetscCall(VecCUDAGetArrayWrite(r,&y));
  c.gpu->evaluate(u,y,false,stream);
  transport_gpu::cuda_check(cudaStreamSynchronize(stream));
  PetscCall(VecCUDARestoreArrayRead(x,&u));PetscCall(VecCUDARestoreArrayWrite(r,&y));
#else
  PetscCall(VecGetArrayRead(x,&u)); PetscCall(VecGetArray(r,&y));
  c.p.evaluate(u,y,false);
  PetscCall(VecRestoreArrayRead(x,&u)); PetscCall(VecRestoreArray(r,&y));
#endif
  c.feval++; c.fseconds+=seconds(start); PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode jacobian(SNES,Vec x,Mat A,Mat P,void* raw) {
  auto& c=*static_cast<Context*>(raw); auto start=Clock::now(); const PetscScalar* u;
  PetscFunctionBeginUser;
#ifdef CRIS_PETSC_CUDA
  cudaStream_t stream;PetscCall(current_stream(stream));
  PetscCall(VecCUDAGetArrayRead(x,&u));c.gpu->evaluate(u,c.gpu->r.p,true,stream);
  transport_gpu::cuda_check(cudaStreamSynchronize(stream));
  PetscCall(VecCUDARestoreArrayRead(x,&u));
  PetscCall(MatSetValuesCOO(P,c.gpu->coo.p,INSERT_VALUES));
  transport_gpu::cuda_check(cudaStreamSynchronize(stream));
#else
  PetscCall(VecGetArrayRead(x,&u)); c.p.evaluate(u,c.scratch.data(),true);
  PetscCall(VecRestoreArrayRead(x,&u)); PetscCall(MatZeroEntries(P));
  for(PetscInt i=0;i<c.p.g.n;i++) {
    PetscCall(MatSetValue(P,i,i,c.p.diagonal(i),ADD_VALUES));
    for(auto k=c.p.g.ptr[i];k<c.p.g.ptr[i+1];k++)
      PetscCall(MatSetValue(P,i,c.p.g.upstream[k],c.p.offdiagonal(i,k),ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
  if(A!=P){PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));}
#endif
  c.jeval++; c.jseconds+=seconds(start); PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode verify(SNES snes,Vec x,Vec r,Mat J,Context& c,const std::string& prefix) {
  PetscFunctionBeginUser;
  int n=c.p.g.n; std::vector<double> state(n),direction(n),analytic(n);
  for(int i=0;i<n;i++){state[i]=.1+.7*((i*37)%997)/997.;c.p.old[i]=.2+.5*((i*17)%991)/991.;direction[i]=std::sin(i+1.);}
#ifdef CRIS_PETSC_CUDA
  c.gpu->timestep();transport_gpu::cuda_check(cudaDeviceSynchronize());
#endif
  PetscScalar* data; PetscCall(VecGetArray(x,&data)); std::copy(state.begin(),state.end(),data);PetscCall(VecRestoreArray(x,&data));
  PetscCall(residual(snes,x,r,&c)); PetscCall(jacobian(snes,x,J,J,&c));
  c.p.evaluate(state.data(),c.scratch.data(),true);
  const PetscScalar* checked;PetscCall(VecGetArrayRead(r,&checked));
  double re=0,rs=1;for(int i=0;i<n;i++){re=std::max(re,std::abs(checked[i]-c.scratch[i]));rs=std::max(rs,std::abs(c.scratch[i]));}
  PetscCall(VecRestoreArrayRead(r,&checked));
  PetscCheck(re/rs<2e-11,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Residual CPU parity failed: %g",re/rs);
  Vec v,jv;PetscCall(VecDuplicate(x,&v));PetscCall(VecDuplicate(x,&jv));
  PetscCall(VecGetArray(v,&data));std::copy(direction.begin(),direction.end(),data);PetscCall(VecRestoreArray(v,&data));
  PetscCall(MatMult(J,v,jv));const PetscScalar* ad;PetscCall(VecGetArrayRead(jv,&ad));std::copy(ad,ad+n,analytic.begin());PetscCall(VecRestoreArrayRead(jv,&ad));
  double je=0,js=1;for(int i=0;i<n;i++){double expected=c.p.diagonal(i)*direction[i];for(auto k=c.p.g.ptr[i];k<c.p.g.ptr[i+1];k++)expected+=c.p.offdiagonal(i,k)*direction[c.p.g.upstream[k]];je=std::max(je,std::abs(expected-analytic[i]));js=std::max(js,std::abs(expected));}
  PetscCheck(je/js<2e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Jacobian CPU parity failed: %g",je/js);
  std::cout<<"{\"check\":\"CPU_residual_Jacobian_parity\",\"residual\":"<<re/rs<<",\"jacobian\":"<<je/js<<"}"<<std::endl;
  double best=1e100;
  for(double eps:{1e-4,1e-5,1e-6}) {
    std::vector<double> plus(n),minus(n),rp(n),rm(n);
    for(int i=0;i<n;i++){plus[i]=state[i]+eps*direction[i];minus[i]=state[i]-eps*direction[i];}
    c.p.evaluate(plus.data(),rp.data(),false);c.p.evaluate(minus.data(),rm.data(),false);
    double err=0,scale=0;for(int i=0;i<n;i++){err=std::max(err,std::abs((rp[i]-rm[i])/(2*eps)-analytic[i]));scale=std::max(scale,std::abs(analytic[i]));}
    best=std::min(best,err/std::max(1.,scale));
  }
  if(c.p.learned && !c.p.model.tree.empty()) {
    std::cout<<"{\"check\":\"Chebyshev_unrestricted_global_secant\",\"relative_error\":"<<best<<",\"derivative_gate\":false}"<<std::endl;
    best=transport::verify_cheb_direction(c.p,state,direction,analytic);
  }
  c.p.evaluate(state.data(),c.scratch.data(),false);
  std::ofstream probes(prefix+".verification.csv");probes<<std::setprecision(17)<<"cell,u,old,r,jv\n";
  for(int i=0;i<n;i++)probes<<i<<','<<state[i]<<','<<c.p.old[i]<<','<<c.scratch[i]<<','<<analytic[i]<<'\n';
  std::cout<<"{\"check\":\"PETSc_Jv_finite_difference\",\"scope\":\""<<(c.p.learned&&!c.p.model.tree.empty()?"center_selected_polynomial_all_cells":"full_smooth_evaluator")<<"\",\"relative_error\":"<<best<<"}"<<std::endl;
  PetscCheck(best<2e-6,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Jacobian check failed: %g",best);
  PetscCall(VecDestroy(&v));PetscCall(VecDestroy(&jv));PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc vectors, matrix structure, callbacks and solver options.
PetscErrorCode create_solver(Context& c, Vec& x, Vec& r, Mat& J, SNES& snes) {
  PetscFunctionBeginUser;
  PetscInt n=c.p.g.n;
#ifdef CRIS_PETSC_CUDA
  c.gpu=std::make_unique<transport_gpu::Evaluator>(c.p);
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF,n,&x));
#else
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&x));
#endif
  PetscCall(VecDuplicate(x,&r));
  std::vector<PetscInt> nnz(n);for(int i=0;i<n;i++)nnz[i]=1+c.p.g.ptr[i+1]-c.p.g.ptr[i];
#ifdef CRIS_PETSC_CUDA
  PetscCall(MatCreate(PETSC_COMM_SELF,&J));PetscCall(MatSetSizes(J,n,n,n,n));PetscCall(MatSetType(J,MATSEQAIJCUSPARSE));
  std::vector<PetscInt> rows,cols;rows.reserve(n+c.p.g.q.size());cols.reserve(n+c.p.g.q.size());
  for(PetscInt i=0;i<n;i++){rows.push_back(i);cols.push_back(i);for(auto k=c.p.g.ptr[i];k<c.p.g.ptr[i+1];k++){rows.push_back(i);cols.push_back(c.p.g.upstream[k]);}}
  PetscCall(MatSetPreallocationCOO(J,rows.size(),rows.data(),cols.data()));
#else
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,0,nnz.data(),&J));
#endif
  PetscCall(SNESCreate(PETSC_COMM_SELF,&snes));PetscCall(SNESSetFunction(snes,r,residual,&c));PetscCall(SNESSetJacobian(snes,J,J,jacobian,&c));
  PetscCall(SNESSetType(snes,SNESNEWTONLS));SNESLineSearch ls;PetscCall(SNESGetLineSearch(snes,&ls));PetscCall(SNESLineSearchSetType(ls,SNESLINESEARCHBT));
  PetscCall(SNESSetTolerances(snes,1e-8,1e-10,0.,100,10000));
  KSP ksp;PC pc;PetscCall(SNESGetKSP(snes,&ksp));PetscCall(KSPSetType(ksp,KSPGMRES));PetscCall(KSPGMRESSetRestart(ksp,80));
  PetscCall(KSPSetTolerances(ksp,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,100));PetscCall(KSPGetPC(ksp,&pc));PetscCall(PCSetType(pc,PCILU));
#ifdef CRIS_PETSC_CUDA
  PetscCall(PCFactorSetMatSolverType(pc,MATSOLVERCUSPARSE));
#endif
  // Retain genuine linear failures. Set -ksp_converged_maxits true to accept
  // iteration-budget exhaustion separately through PETSc's options.
  PetscCall(SNESSetMaxLinearSolveFailures(snes,1));
  PetscCall(SNESSetFromOptions(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}
