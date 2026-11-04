#include <petscsnes.h>
#include <omp.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include "transport.hpp"
#include "cheb_verification.hpp"
#ifdef CRIS_PETSC_CUDA
#include <petscdevice.h>
#include <memory>
#include "gpu_evaluator.cuh"
#include "device_check.cuh"
#endif

using transport::Problem;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct Context {
  Problem p; std::vector<double> scratch; long long feval=0,jeval=0;
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
PetscErrorCode run() {
  PetscFunctionBeginUser;
  PetscMPIInt ranks;PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
  PetscCheck(ranks==1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Initial verification driver requires one MPI rank");
  char graph[4096]="",model[4096]="",method[32]="srdm",prefix[4096]="transport",initialfile[4096]="";
  PetscBool check=PETSC_FALSE;PetscInt threads=8,maxsteps=100000,maxcuts=12;
  PetscReal factor=1,initial=0,flux_exponent=2.;PetscBool exponent_set=PETSC_FALSE;
  PetscReal reaction_rate=0,reaction_equilibrium=.5,reaction_width=.01;
  PetscInt reaction_type=1;
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-graph",graph,sizeof(graph),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-model",model,sizeof(model),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-method",method,sizeof(method),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-output",prefix,sizeof(prefix),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-initial_file",initialfile,sizeof(initialfile),nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-verify",&check,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-threads",&threads,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-steps",&maxsteps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-max_cuts",&maxcuts,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-dt_scale",&factor,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-initial",&initial,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-flux_exponent",&flux_exponent,&exponent_set));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-reaction_rate",&reaction_rate,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-reaction_equilibrium",&reaction_equilibrium,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-reaction_width",&reaction_width,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-reaction_type",&reaction_type,nullptr));
  PetscCheck(reaction_type==1||reaction_type==2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown reaction type");
  PetscCheck(std::string(method)=="srdm"||std::string(method)=="learned",PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown method");
  PetscCheck(threads>0 && factor>0 && maxsteps>=0 && maxcuts>=0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid execution controls");
  omp_set_num_threads(threads); Context c(graph,model,std::string(method)=="learned");
  if(exponent_set)c.p.set_flux_exponent(flux_exponent);
  c.p.set_reaction(reaction_rate,reaction_equilibrium,reaction_width);
  c.p.reaction_type=reaction_type;
  Vec x,r;Mat J;SNES snes;PetscInt n=c.p.g.n;
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
  // Retain genuine linear failures. Iteration-budget exhaustion is separately
  // accepted through -ksp_converged_maxits true in every final command.
  PetscCall(SNESSetMaxLinearSolveFailures(snes,1));
  PetscCall(SNESSetFromOptions(snes));
  std::string tag(prefix);PetscViewer viewer;PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,(tag+".options.txt").c_str(),&viewer));
  PetscCall(SNESView(snes,viewer));PetscCall(PetscOptionsView(nullptr,viewer));PetscCall(PetscViewerDestroy(&viewer));
  c.p.dt=c.p.g.steps.front()*factor;if(check)PetscCall(verify(snes,x,r,J,c,tag));
  const std::string base_tag(tag);
  PetscInt repeats=3;PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-trajectory_repeats",&repeats,nullptr));
  PetscCheck(repeats>=1&&repeats<=3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid repeats");
  struct SavedAttempt {int schedule;double time,dt,inf,physical,snes,core;int reason,nit,lit;bool finite,accepted;std::vector<double> u;};
  transport_gpu::TrialCheck trialcheck(n);
  for(int repeat=1;repeat<=repeats;repeat++) {
    tag=base_tag+"_r"+std::to_string(repeat);
    std::fill(c.p.old.begin(),c.p.old.end(),initial);
    if(initialfile[0]){std::ifstream file(initialfile,std::ios::binary);c.p.old=transport::read<double>(file,n);}
    const auto initial_state=c.p.old;
    c.feval=c.jeval=0;c.fseconds=c.jseconds=0;
    std::vector<SavedAttempt> attempts;
    double time=0,solve_seconds=0,advance_seconds=0,snapshot_seconds=0,device_check_seconds=0;
    int accepted=0,cuts=0;long long nonlinear=0,linear=0;bool complete=true;
    std::vector<double> solution(n,initial);
    transport_gpu::cuda_check(cudaDeviceSynchronize());auto wall_start=Clock::now();
    for(int step=0;step<std::min<int>(maxsteps,c.p.g.steps.size())&&complete;step++){
      double remaining=c.p.g.steps[step]*factor,trial=remaining;int localcuts=0;
      while(remaining>1e-12*c.p.g.steps[step]*factor){
        auto core_start=Clock::now();
        c.p.dt=std::min(trial,remaining);
        PetscScalar* data;PetscCall(VecGetArray(x,&data));std::copy(c.p.old.begin(),c.p.old.end(),data);PetscCall(VecRestoreArray(x,&data));
        c.gpu->timestep();transport_gpu::cuda_check(cudaDeviceSynchronize());
        auto solve_start=Clock::now();PetscCall(SNESSolve(snes,nullptr,x));
        transport_gpu::cuda_check(cudaDeviceSynchronize());double solve_elapsed=seconds(solve_start);solve_seconds+=solve_elapsed;
        SNESConvergedReason reason;PetscCall(SNESGetConvergedReason(snes,&reason));
        PetscInt nit,lit;PetscCall(SNESGetIterationNumber(snes,&nit));PetscCall(SNESGetLinearSolveIterations(snes,&lit));nonlinear+=nit;linear+=lit;
        const PetscScalar* device_u;cudaStream_t stream;PetscCall(current_stream(stream));PetscCall(VecCUDAGetArrayRead(x,&device_u));
        auto device_check_start=Clock::now();
        double inf,physical;bool finite;trialcheck.evaluate(*c.gpu,device_u,inf,physical,finite,stream);
        device_check_seconds+=seconds(device_check_start);
        PetscCall(VecCUDARestoreArrayRead(x,&device_u));
        bool ok=reason>0&&finite;
        // Transfer is retained and charged: the existing timestep setup uses host old state.
        const PetscScalar* u;PetscCall(VecGetArrayRead(x,&u));std::copy(u,u+n,solution.begin());PetscCall(VecRestoreArrayRead(x,&u));
        double before=time,dt=c.p.dt;
        if(ok){time+=dt;remaining-=dt;accepted++;c.p.old=solution;trial=std::min(2*trial,remaining);}
        else{cuts++;localcuts++;trial*=.5;if(localcuts>maxcuts)complete=false;}
        transport_gpu::cuda_check(cudaDeviceSynchronize());double core_elapsed=seconds(core_start);advance_seconds+=core_elapsed;
        auto snapshot_start=Clock::now();
        attempts.push_back({step,before,dt,inf,physical,solve_elapsed,core_elapsed,int(reason),int(nit),int(lit),finite,ok,solution});
        snapshot_seconds+=seconds(snapshot_start);
        if(!complete)break;
      }
    }
    double trajectory_wall=seconds(wall_start);auto final_state=c.p.old;
    auto output_start=Clock::now();std::ofstream events(tag+".jsonl");events<<std::setprecision(17);int saved=0,rejected_saved=0;
    for(const auto& a:attempts){
      events<<"{\"schedule_step\":"<<a.schedule<<",\"time_before\":"<<a.time<<",\"dt\":"<<a.dt<<",\"accepted\":"<<(a.accepted?"true":"false")<<",\"reason\":"<<a.reason<<",\"newton\":"<<a.nit<<",\"krylov\":"<<a.lit<<",\"residual_inf\":";
      if(a.finite)events<<a.inf;else events<<"null";
      events<<",\"physical_inf\":";if(a.finite)events<<a.physical;else events<<"null";
      events<<",\"snes_seconds\":"<<a.snes<<",\"advance_seconds\":"<<a.core<<"}\n";
      if(a.accepted){std::ofstream field(tag+".step"+std::to_string(++saved)+".field",std::ios::binary);field.write(reinterpret_cast<const char*>(a.u.data()),n*sizeof(double));}
      else{std::ofstream field(tag+".rejected"+std::to_string(++rejected_saved)+".field",std::ios::binary);field.write(reinterpret_cast<const char*>(a.u.data()),n*sizeof(double));}
    }
    events.close();{std::ofstream field(tag+".field",std::ios::binary);field.write(reinterpret_cast<char*>(final_state.data()),n*sizeof(double));}
    double output_seconds=seconds(output_start);
    // Deferred CPU audit on EVERY trial, advancing its old state only on acceptance.
    auto audit_start=Clock::now();c.p.old=initial_state;double worst_r=0,worst_p=0;int audit_warnings=0,acceptance_mismatches=0;
    std::ofstream diagnostics(tag+".audit.jsonl");diagnostics<<std::setprecision(17);
    for(const auto& a:attempts){
      c.p.dt=a.dt;c.p.evaluate(a.u.data(),c.scratch.data(),false);
      double inf=0;bool finite=true;for(int i=0;i<n;i++){finite=finite&&std::isfinite(a.u[i])&&std::isfinite(c.scratch[i]);inf=std::max(inf,std::abs(c.scratch[i]));}
      double physical=finite?c.p.physical(a.u.data()):std::numeric_limits<double>::infinity();
      bool cpu_ok=a.reason>0&&finite&&std::isfinite(physical);
      if(cpu_ok!=a.accepted){
        acceptance_mismatches++;
        diagnostics<<"{\"kind\":\"acceptance_mismatch\",\"schedule_step\":"<<a.schedule<<",\"time_before\":"<<a.time<<",\"dt\":"<<a.dt<<"}\n";
      }
      if(finite&&a.finite){
        double re=std::abs(inf-a.inf)/std::max(1.,inf),pe=std::abs(physical-a.physical)/std::max(1.,physical);
        worst_r=std::max(worst_r,re);worst_p=std::max(worst_p,pe);
        if(!(re<2e-11&&pe<2e-11)){
          audit_warnings++;
          diagnostics<<"{\"kind\":\"residual_discrepancy\",\"schedule_step\":"<<a.schedule<<",\"time_before\":"<<a.time<<",\"dt\":"<<a.dt<<",\"residual_relative\":"<<re<<",\"physical_relative\":"<<pe<<",\"cpu_physical_inf\":"<<physical<<",\"gpu_physical_inf\":"<<a.physical<<"}\n";
        }
      }
      if(a.accepted)c.p.old=a.u;
    }
    diagnostics.close();double audit_seconds=seconds(audit_start);c.p.old=final_state;
    std::ofstream summary(tag+".summary.json");summary<<std::setprecision(17)
      <<"{\"complete_requested_steps\":"<<(complete?"true":"false")<<",\"time\":"<<time<<",\"accepted\":"<<accepted<<",\"cuts\":"<<cuts<<",\"newton\":"<<nonlinear<<",\"krylov\":"<<linear
      <<",\"snes_seconds\":"<<solve_seconds<<",\"advance_seconds\":"<<advance_seconds<<",\"trajectory_wall_seconds\":"<<trajectory_wall<<",\"snapshot_seconds\":"<<snapshot_seconds<<",\"cpu_audit_seconds\":"<<audit_seconds<<",\"field_output_seconds\":"<<output_seconds
      <<",\"device_check_seconds\":"<<device_check_seconds<<",\"residual_calls\":"<<c.feval<<",\"jacobian_calls\":"<<c.jeval<<",\"residual_seconds\":"<<c.fseconds<<",\"jacobian_seconds\":"<<c.jseconds<<",\"audit_residual_relative\":"<<worst_r<<",\"audit_physical_relative\":"<<worst_p<<",\"audit_warning_count\":"<<audit_warnings<<",\"acceptance_mismatches\":"<<acceptance_mismatches<<",\"all_trial_acceptance_parity\":"<<(acceptance_mismatches==0?"true":"false")<<"}\n";
    std::cout<<tag<<" complete="<<complete<<" accepted="<<accepted<<" advance_seconds="<<advance_seconds<<" snes_seconds="<<solve_seconds<<std::endl;
    if(!complete||acceptance_mismatches)break; // A changed model may stop; do not repeat a stopped trajectory for timings.
  }
  PetscCall(SNESDestroy(&snes));PetscCall(MatDestroy(&J));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&r));PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc,char** argv){PetscInitialize(&argc,&argv,nullptr,nullptr);std::cout<<std::setprecision(17);int code=0;try{code=run();}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;code=1;}PetscFinalize();return code;}
