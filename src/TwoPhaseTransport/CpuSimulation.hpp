#pragma once
#include "CORE_SCL/Solvers/Nonlinear/PetscTransportSolver.hpp"

// Read a case, advance its schedule and write simulation results.
PetscErrorCode run() {
  PetscFunctionBeginUser;
  PetscMPIInt ranks;PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ranks));
  PetscCheck(ranks==1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Transport driver requires one MPI rank");
  char graph[4096]="",model[4096]="",method[32]="srdm",prefix[4096]="transport",initialfile[4096]="";
  PetscBool check=PETSC_FALSE;PetscInt threads=8,maxsteps=100000,maxcuts=12;
  PetscReal factor=1,initial=0,flux_exponent=2.;PetscBool exponent_set=PETSC_FALSE;
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
  PetscCheck(std::string(method)=="srdm"||std::string(method)=="learned",PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown method");
  PetscCheck(threads>0 && factor>0 && maxsteps>=0 && maxcuts>=0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid execution controls");
  omp_set_num_threads(threads); Context c(graph,model,std::string(method)=="learned");
  if(exponent_set)c.p.set_flux_exponent(flux_exponent);
  Vec x,r;Mat J;SNES snes;PetscInt n=c.p.g.n;
  PetscCall(create_solver(c,x,r,J,snes));
  std::string tag(prefix);PetscViewer viewer;PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,(tag+".options.txt").c_str(),&viewer));
  PetscCall(SNESView(snes,viewer));PetscCall(PetscOptionsView(nullptr,viewer));PetscCall(PetscViewerDestroy(&viewer));
  c.p.dt=c.p.g.steps.front()*factor;if(check)PetscCall(verify(snes,x,r,J,c,tag));
  std::fill(c.p.old.begin(),c.p.old.end(),initial);c.feval=c.jeval=0;c.fseconds=c.jseconds=0;
  if(initialfile[0]) {std::ifstream file(initialfile,std::ios::binary);c.p.old=transport::read<double>(file,n);}
  std::ofstream events(tag+".jsonl");events<<std::setprecision(17);auto start=Clock::now();
  double time=0,solve_seconds=0;int accepted=0,cuts=0;long long nonlinear=0,linear=0;bool complete=true;
  std::vector<double> solution(n,initial);
  for(int s=0;s<std::min<int>(maxsteps,c.p.g.steps.size()) && complete;s++) {
    double remaining=c.p.g.steps[s]*factor,trial=remaining;int localcuts=0;
    while(remaining>1e-12*c.p.g.steps[s]*factor) {
      c.p.dt=std::min(trial,remaining);PetscScalar* data;PetscCall(VecGetArray(x,&data));std::copy(c.p.old.begin(),c.p.old.end(),data);PetscCall(VecRestoreArray(x,&data));
#ifdef CRIS_PETSC_CUDA
      c.gpu->timestep();transport_gpu::cuda_check(cudaDeviceSynchronize());
#endif
      auto stepstart=Clock::now();PetscCall(SNESSolve(snes,nullptr,x));double solve_elapsed=seconds(stepstart);solve_seconds+=solve_elapsed;SNESConvergedReason reason;PetscCall(SNESGetConvergedReason(snes,&reason));
      PetscInt nit,lit;PetscCall(SNESGetIterationNumber(snes,&nit));PetscCall(SNESGetLinearSolveIterations(snes,&lit));nonlinear+=nit;linear+=lit;
      const PetscScalar* u;PetscCall(VecGetArrayRead(x,&u));std::copy(u,u+n,solution.begin());PetscCall(VecRestoreArrayRead(x,&u));
      c.p.evaluate(solution.data(),c.scratch.data(),false);double inf=0;bool finite=true;
      for(int i=0;i<n;i++){finite=finite&&std::isfinite(solution[i])&&std::isfinite(c.scratch[i]);inf=std::max(inf,std::abs(c.scratch[i]));}
      double physical=finite?c.p.physical(solution.data()):std::numeric_limits<double>::infinity();bool ok=reason>0&&finite&&std::isfinite(physical);
      events<<"{\"schedule_step\":"<<s<<",\"time_before\":"<<time<<",\"dt\":"<<c.p.dt<<",\"accepted\":"<<(ok?"true":"false")<<",\"reason\":"<<int(reason)<<",\"newton\":"<<nit<<",\"krylov\":"<<lit<<",\"residual_inf\":";
      if(finite)events<<inf;else events<<"null";
      events<<",\"physical_inf\":";if(std::isfinite(physical))events<<physical;else events<<"null";
      events<<",\"snes_seconds\":"<<solve_elapsed<<",\"seconds\":"<<seconds(stepstart)<<"}"<<std::endl;
      if(ok){time+=c.p.dt;remaining-=c.p.dt;accepted++;c.p.old=solution;std::ofstream field(tag+".step"+std::to_string(accepted)+".field",std::ios::binary);field.write(reinterpret_cast<char*>(solution.data()),n*sizeof(double));trial=std::min(2*trial,remaining);}
      else {cuts++;localcuts++;trial*=.5;if(localcuts>maxcuts){complete=false;break;}}
    }
  }
  double total=seconds(start);std::ofstream field(tag+".field",std::ios::binary);field.write(reinterpret_cast<char*>(c.p.old.data()),n*sizeof(double));
  std::ofstream summary(tag+".summary.json");summary<<std::setprecision(17)<<"{\"method\":\""<<method<<"\",\"complete_requested_steps\":"<<(complete?"true":"false")<<",\"time\":"<<time<<",\"cells\":"<<n<<",\"accepted\":"<<accepted<<",\"cuts\":"<<cuts<<",\"newton\":"<<nonlinear<<",\"krylov\":"<<linear<<",\"residual_calls\":"<<c.feval<<",\"jacobian_calls\":"<<c.jeval<<",\"residual_seconds\":"<<c.fseconds<<",\"jacobian_seconds\":"<<c.jseconds<<",\"snes_seconds\":"<<solve_seconds<<",\"transport_seconds_including_step_output\":"<<total<<"}\n";
  std::cout<<method<<" accepted="<<accepted<<" cuts="<<cuts<<" time="<<time<<" complete="<<complete<<std::endl;
  PetscCall(SNESDestroy(&snes));PetscCall(MatDestroy(&J));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&r));PetscFunctionReturn(PETSC_SUCCESS);
}
