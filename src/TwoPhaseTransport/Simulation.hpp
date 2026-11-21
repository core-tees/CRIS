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
  const std::string base_tag(tag);
  PetscInt repeats=3;PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-trajectory_repeats",&repeats,nullptr));
  PetscCheck(repeats>=1&&repeats<=3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid repeats");
  struct SavedAttempt {int schedule;double time,dt,inf,physical,snes,core;int reason,nit,lit;bool finite,accepted;std::vector<double> u;};
  transport_gpu::TrialCheck trialcheck(n);
  for(int repeat=1;repeat<=repeats;repeat++) {
    tag=base_tag+"_r"+std::to_string(repeat);
    std::fill(c.p.old.begin(),c.p.old.end(),initial);
    if(initialfile[0]){std::ifstream file(initialfile,std::ios::binary);c.p.old=transport::read<double>(file,n);}

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
        // Copy the host state needed for the next timestep; this transfer is included in advance timing.
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
    std::ofstream summary(tag+".summary.json");summary<<std::setprecision(17)
      <<"{\"complete_requested_steps\":"<<(complete?"true":"false")<<",\"time\":"<<time<<",\"accepted\":"<<accepted<<",\"cuts\":"<<cuts<<",\"newton\":"<<nonlinear<<",\"krylov\":"<<linear
      <<",\"snes_seconds\":"<<solve_seconds<<",\"advance_seconds\":"<<advance_seconds<<",\"trajectory_wall_seconds\":"<<trajectory_wall<<",\"snapshot_seconds\":"<<snapshot_seconds<<",\"field_output_seconds\":"<<output_seconds
      <<",\"device_check_seconds\":"<<device_check_seconds<<",\"residual_calls\":"<<c.feval<<",\"jacobian_calls\":"<<c.jeval<<",\"residual_seconds\":"<<c.fseconds<<",\"jacobian_seconds\":"<<c.jseconds<<"}\n";
    std::cout<<tag<<" complete="<<complete<<" accepted="<<accepted<<" advance_seconds="<<advance_seconds<<" snes_seconds="<<solve_seconds<<std::endl;
    if(!complete)break; // Do not repeat incomplete trajectories for timing.
  }
  PetscCall(SNESDestroy(&snes));PetscCall(MatDestroy(&J));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&r));PetscFunctionReturn(PETSC_SUCCESS);
}
