// PETSc Bratu comparisons. The launcher supplies the grid, method, cases,
// output directory and shared local-inverse model; PETSc options follow.
#include <petscsnes.h>
#include <petscdmshell.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <iomanip>
using Array=std::vector<double>;
template<class T>void read(std::ifstream&f,std::vector<T>&a,size_t n){a.resize(n);f.read((char*)a.data(),n*sizeof(T));if(!f)throw std::runtime_error("read failed");}
struct Problem{int64_t n,m;std::vector<int64_t>ptr,col;Array w,rate,src,vol,inc,old,beta;double inj=0,q=3;};
struct Cell{Problem*p;PetscInt i;};
std::vector<Cell> cells;
long long local_evals=0,local_updates=0;
PetscErrorCode globalFunction(SNES snes,Vec x,Vec f,void*ctx){
 auto&p=*(Problem*)ctx;const PetscScalar*u;PetscScalar*r;PetscCall(VecGetArrayRead(x,&u));PetscCall(VecGetArray(f,&r));

 for(int64_t i=0;i<p.n;i++){double a=p.old[i];for(auto k=p.ptr[i];k<p.ptr[i+1];k++)a+=p.w[k]*u[p.col[k]];r[i]=u[i]-a-p.beta[i]*std::exp(u[i]);if(!std::isfinite(r[i]))PetscCall(SNESSetFunctionDomainError(snes));}
 PetscCall(VecRestoreArrayRead(x,&u));PetscCall(VecRestoreArray(f,&r));return 0;
}
PetscErrorCode globalJacobian(SNES,Vec x,Mat A,Mat B,void*ctx){
 auto&p=*(Problem*)ctx;const PetscScalar*u;PetscCall(VecGetArrayRead(x,&u));Array du(p.n);for(int64_t i=0;i<p.n;i++)du[i]=std::exp(u[i]);
 std::vector<PetscInt> js;Array vs;
 for(PetscInt i=0;i<p.n;i++){js.clear();vs.clear();js.push_back(i);vs.push_back(1-p.beta[i]*du[i]);for(auto k=p.ptr[i];k<p.ptr[i+1];k++){js.push_back(p.col[k]);vs.push_back(-p.w[k]);}PetscCall(MatSetValues(B,1,&i,js.size(),js.data(),vs.data(),INSERT_VALUES));}
 PetscCall(VecRestoreArrayRead(x,&u));PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));return 0;
}
// NASM installs a hook which copies its frozen subdomain boundary into l.
// Only the one owned unknown is replaced during a local Newton evaluation.
PetscErrorCode toLocal(DM,Vec x,InsertMode,Vec l){const PetscScalar*u;PetscScalar*v;PetscCall(VecGetArrayRead(x,&u));PetscCall(VecGetArray(l,&v));v[0]=u[0];PetscCall(VecRestoreArrayRead(x,&u));PetscCall(VecRestoreArray(l,&v));return 0;}
PetscErrorCode noOp(DM,Vec,InsertMode,Vec){return 0;}
PetscErrorCode cellFunction(SNES s,Vec x,Vec f,void*ctx){
 auto&c=*(Cell*)ctx;auto&p=*c.p;auto i=c.i;DM dm;Vec l;PetscCall(SNESGetDM(s,&dm));PetscCall(DMGetLocalVector(dm,&l));PetscCall(DMGlobalToLocalBegin(dm,x,INSERT_VALUES,l));PetscCall(DMGlobalToLocalEnd(dm,x,INSERT_VALUES,l));
 const PetscScalar*u;PetscScalar*r;PetscCall(VecGetArrayRead(l,&u));PetscCall(VecGetArray(f,&r));double a=p.old[i];for(auto k=p.ptr[i];k<p.ptr[i+1];k++)a+=p.w[k]*u[1+k-p.ptr[i]];r[0]=u[0]-a-p.beta[i]*std::exp(u[0]);if(!std::isfinite(r[0]))PetscCall(SNESSetFunctionDomainError(s));local_evals++;
 PetscCall(VecRestoreArrayRead(l,&u));PetscCall(VecRestoreArray(f,&r));PetscCall(DMRestoreLocalVector(dm,&l));return 0;
}
PetscErrorCode cellJacobian(SNES,Vec x,Mat A,Mat B,void*ctx){auto&c=*(Cell*)ctx;const PetscScalar*u;PetscCall(VecGetArrayRead(x,&u));PetscInt z=0;PetscScalar v=1-c.p->beta[c.i]*std::exp(u[0]);PetscCall(MatSetValues(B,1,&z,1,&z,&v,INSERT_VALUES));PetscCall(VecRestoreArrayRead(x,&u));PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));return 0;}
PetscErrorCode decomposition(DM dm,PetscInt*n,char***names,IS**inner,IS**outer,DM**sub){
 Problem*p;PetscCall(DMGetApplicationContext(dm,&p));*n=p->n;if(names)*names=nullptr;if(inner)*inner=nullptr;if(outer)*outer=nullptr;PetscCall(PetscMalloc1(*n,sub));cells.resize(*n);
 for(PetscInt i=0;i<*n;i++){
  cells[i]={p,i};DM d;Vec x,l;Mat J;PetscCall(DMShellCreate(PETSC_COMM_SELF,&d));PetscCall(DMSetApplicationContext(d,&cells[i]));PetscCall(DMShellSetContext(d,&cells[i]));PetscCall(VecCreateSeq(PETSC_COMM_SELF,1,&x));PetscCall(VecCreateSeq(PETSC_COMM_SELF,1+p->ptr[i+1]-p->ptr[i],&l));PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,1,1,1,nullptr,&J));
  PetscCall(MatSetValue(J,0,0,1.,INSERT_VALUES));PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscCall(DMShellSetGlobalVector(d,x));PetscCall(DMShellSetLocalVector(d,l));PetscCall(DMShellSetMatrix(d,J));PetscCall(DMShellSetGlobalToLocal(d,toLocal,noOp));PetscCall(DMShellSetLocalToGlobal(d,toLocal,noOp));PetscCall(DMSNESSetFunction(d,cellFunction,&cells[i]));PetscCall(DMSNESSetJacobian(d,cellJacobian,&cells[i]));(*sub)[i]=d;
  PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&l));PetscCall(MatDestroy(&J));if(i%10000==0)std::cout<<"SETUP cells="<<i<<"/"<<*n<<std::endl;
 }return 0;
}
PetscErrorCode configureSub(DM,DM sub,void*){Cell*c;PetscCall(DMShellGetContext(sub,&c));PetscCall(DMSNESSetFunction(sub,cellFunction,c));PetscCall(DMSNESSetJacobian(sub,cellJacobian,c));return 0;}
PetscErrorCode scatters(DM dm,PetscInt n,DM*sub,VecScatter**is,VecScatter**os,VecScatter**gs){
 Problem*p;PetscCall(DMGetApplicationContext(dm,&p));Vec global;PetscCall(DMCreateGlobalVector(dm,&global));PetscCall(PetscMalloc1(n,is));PetscCall(PetscMalloc1(n,os));PetscCall(PetscMalloc1(n,gs));
 for(PetscInt i=0;i<n;i++){Vec x,l;IS own,all;PetscCall(DMCreateGlobalVector(sub[i],&x));PetscCall(DMCreateLocalVector(sub[i],&l));PetscCall(ISCreateGeneral(PETSC_COMM_SELF,1,&i,PETSC_COPY_VALUES,&own));std::vector<PetscInt> ids{i};for(auto k=p->ptr[i];k<p->ptr[i+1];k++)ids.push_back(p->col[k]);PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ids.size(),ids.data(),PETSC_COPY_VALUES,&all));
 PetscCall(VecScatterCreate(global,own,x,nullptr,&(*os)[i]));PetscCall(VecScatterCopy((*os)[i],&(*is)[i]));PetscCall(VecScatterCreate(global,all,l,nullptr,&(*gs)[i]));PetscCall(ISDestroy(&own));PetscCall(ISDestroy(&all));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&l));}
 PetscCall(VecDestroy(&global));return 0;
}
struct Monitor{std::ofstream*log;int attempt=0;std::chrono::steady_clock::time_point begin;};
PetscErrorCode monitor(SNES s,PetscInt it,PetscReal norm,void*ctx){auto&m=*(Monitor*)ctx;PetscInt k;PetscCall(SNESGetLinearSolveIterations(s,&k));*m.log<<m.attempt<<','<<it<<','<<k<<','<<norm<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-m.begin).count()<<std::endl;return 0;}
void save(const std::string&path,const double*u,int64_t n){std::ofstream f(path,std::ios::binary);f.write((char*)&n,8);f.write((const char*)u,n*8);}
struct Fit{
 Array w;long long outside=0;bool numerical=false;
 Fit(const std::string&file){std::ifstream f(file,std::ios::binary);read(f,w,1425);}
 std::pair<double,double>eval(double a,double b){
  if(numerical){if(b==0)return {a,1};if(!std::isfinite(a)||a>=-1-std::log(b))return {NAN,NAN};double q=std::exp(a+std::log(b)),s=0,lo=0,hi=1;for(int it=0;it<=80;it++){double r=q*std::exp(s),g=s-r;if(std::abs(g)<=1e-14)return {a+s,1/(1-r)};if(g<0)lo=s;else hi=s;double v=s-g/(1-r);if(v<=lo||v>=hi||!std::isfinite(v))v=.5*(lo+hi);s=v;}return {NAN,NAN};}
  if(!std::isfinite(a)||!std::isfinite(b))return {NAN,NAN};
  if(a < -90 || a > 8 || b < .015 || b > .025)outside++;
  double chain=1.;if(a>8.){double t=std::tanh((a-8.)/5.);a=8.+5.*t;chain=1.-t*t;}else if(a < -90.){double t=std::tanh((a+90.)/5.);a=-90.+5.*t;chain=1.-t*t;}
  double x[16]={},dx[16]={},y[16]={},dy[16]={};x[0]=std::asinh(a/5.);x[1]=(b-.02)/.005;dx[0]=1/std::hypot(5.,a);
  int off=0;
  for(int l=0;l<7;l++){int ni=l==0?2:16,no=l==6?1:16;
   for(int j=0;j<no;j++){double z=w[off+ni*no+j],dz=0;for(int k=0;k<ni;k++){z+=w[off+j*ni+k]*x[k];dz+=w[off+j*ni+k]*dx[k];}
    if(l<6){y[j]=std::tanh(z);dy[j]=dz*(1-y[j]*y[j]);}else{y[j]=z;dy[j]=dz;}}
   off+=ni*no+no;for(int j=0;j<no;j++){x[j]=y[j];dx[j]=dy[j];}
  }
  return {a+x[0],(1+dx[0])*chain};
 }
};
struct Context{Problem*p;Fit*fit;bool learned;};
PetscErrorCode learnedFunction(SNES s,Vec x,Vec f,void*ctx){auto&c=*(Context*)ctx;auto&p=*c.p;const PetscScalar*u;PetscScalar*r;PetscCall(VecGetArrayRead(x,&u));PetscCall(VecGetArray(f,&r));for(int64_t i=0;i<p.n;i++){double a=p.old[i];for(auto k=p.ptr[i];k<p.ptr[i+1];k++)a+=p.w[k]*u[p.col[k]];r[i]=u[i]-c.fit->eval(a,p.beta[i]).first;if(!std::isfinite(r[i]))PetscCall(SNESSetFunctionDomainError(s));}PetscCall(VecRestoreArrayRead(x,&u));PetscCall(VecRestoreArray(f,&r));return 0;}
PetscErrorCode learnedJacobian(SNES,Vec x,Mat A,Mat B,void*ctx){auto&c=*(Context*)ctx;auto&p=*c.p;const PetscScalar*u;PetscCall(VecGetArrayRead(x,&u));std::vector<PetscInt>js;Array vs;for(PetscInt i=0;i<p.n;i++){double a=p.old[i];for(auto k=p.ptr[i];k<p.ptr[i+1];k++)a+=p.w[k]*u[p.col[k]];double slope=c.fit->eval(a,p.beta[i]).second;js={i};vs={1};for(auto k=p.ptr[i];k<p.ptr[i+1];k++){js.push_back(p.col[k]);vs.push_back(-slope*p.w[k]);}PetscCall(MatSetValues(B,1,&i,js.size(),js.data(),vs.data(),INSERT_VALUES));}PetscCall(VecRestoreArrayRead(x,&u));PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));return 0;}
std::pair<double,double>norms(Vec x,Context&c){const PetscScalar*u;VecGetArrayRead(x,&u);double physical=0,map=0;auto&p=*c.p;for(int64_t i=0;i<p.n;i++){double a=p.old[i];for(auto k=p.ptr[i];k<p.ptr[i+1];k++)a+=p.w[k]*u[p.col[k]];double r=u[i]-a-p.beta[i]*std::exp(u[i]),g=c.learned?u[i]-c.fit->eval(a,p.beta[i]).first:r;physical=std::isfinite(r)?std::max(physical,std::abs(r)):INFINITY;map=std::isfinite(g)?std::max(map,std::abs(g)):INFINITY;}VecRestoreArrayRead(x,&u);return {physical,map};}
PetscErrorCode convergence(SNES s,PetscInt it,PetscReal xn,PetscReal sn,PetscReal fn,SNESConvergedReason*reason,void*ctx){PetscCall(SNESConvergedDefault(s,it,xn,sn,fn,reason,nullptr));Vec x;PetscCall(SNESGetSolution(s,&x));auto [p,m]=norms(x,*(Context*)ctx);if(p<=1e-6&&m<=1e-6)*reason=SNES_CONVERGED_FNORM_ABS;else if(*reason>0)*reason=SNES_CONVERGED_ITERATING;return 0;}
int main(int argc,char**argv){
 PetscCall(PetscInitialize(&argc,&argv,nullptr,nullptr));if(argc<6)return 2;std::string input=argv[1],graph=argv[2],method=argv[3],casesfile=argv[4],out=argv[5];std::filesystem::create_directories(out);Problem p;
 {std::ifstream f(input+"/"+graph+".bin",std::ios::binary);f.read((char*)&p.n,8);f.read((char*)&p.m,8);read(f,p.ptr,p.n+1);read(f,p.col,p.m);read(f,p.w,p.m);}p.old.resize(p.n);p.beta.resize(p.n);char model[4096];PetscBool hasmodel=PETSC_FALSE;PetscCall(PetscOptionsGetString(nullptr,nullptr,"-bratu_model",model,sizeof(model),&hasmodel));if(!hasmodel)throw std::runtime_error("-bratu_model is required");Fit fit(model);fit.numerical=method=="selected";Context ctx{&p,&fit,method=="learned"||method=="selected"};
 // An optional read-only evaluator validates the exported spline without training.
 char query[4096];PetscBool hasquery=PETSC_FALSE;PetscCall(PetscOptionsGetString(nullptr,nullptr,"-fit_queries",query,sizeof(query),&hasquery));if(hasquery){std::ifstream f(query);double a,b;std::ofstream z(out+"/fit_values.csv");z<<std::setprecision(17);while(f>>a>>b){auto [v,d]=fit.eval(a,b);z<<v<<','<<d<<'\n';}PetscCall(PetscFinalize());return 0;}
 auto begin=std::chrono::steady_clock::now();Vec x,f;Mat J;DM dm;SNES s;PetscCall(VecCreateSeq(PETSC_COMM_SELF,p.n,&x));PetscCall(VecDuplicate(x,&f));std::vector<PetscInt>nnz(p.n);for(int64_t i=0;i<p.n;i++)nnz[i]=1+p.ptr[i+1]-p.ptr[i];PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,p.n,p.n,0,nnz.data(),&J));PetscCall(VecSet(x,0));PetscCall(globalJacobian(nullptr,x,J,J,&p));PetscCall(SNESCreate(PETSC_COMM_SELF,&s));
 if(method=="aspin"){
  PetscCall(DMShellCreate(PETSC_COMM_SELF,&dm));PetscCall(DMSetApplicationContext(dm,&p));PetscCall(DMShellSetGlobalVector(dm,x));PetscCall(DMShellSetMatrix(dm,J));PetscCall(DMSNESSetFunction(dm,globalFunction,&p));PetscCall(DMSNESSetJacobian(dm,globalJacobian,&p));PetscCall(DMSubDomainHookAdd(dm,configureSub,nullptr,nullptr));PetscCall(DMShellSetCreateDomainDecomposition(dm,decomposition));PetscCall(DMShellSetCreateDomainDecompositionScatters(dm,scatters));PetscCall(SNESSetDM(s,dm));PetscCall(SNESSetType(s,SNESASPIN));
 }else{
  PetscCall(SNESSetType(s,SNESNEWTONLS));if(ctx.learned){PetscCall(SNESSetFunction(s,f,learnedFunction,&ctx));PetscCall(SNESSetJacobian(s,J,J,learnedJacobian,&ctx));}else{PetscCall(SNESSetFunction(s,f,globalFunction,&p));PetscCall(SNESSetJacobian(s,J,J,globalJacobian,&p));}
  KSP k;PC pc;PetscCall(SNESGetKSP(s,&k));PetscCall(KSPSetType(k,KSPGMRES));PetscCall(KSPGetPC(k,&pc));PetscCall(PCSetType(pc,PCILU));PetscCall(KSPSetTolerances(k,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10000));PetscCall(SNESSetConvergenceTest(s,convergence,&ctx,nullptr));
 }
 PetscCall(SNESSetTolerances(s,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,1000,PETSC_DEFAULT));PetscCall(SNESSetFromOptions(s));PetscCall(SNESSetUp(s));if(method=="aspin"){SNES npc;PetscCall(SNESGetNPC(s,&npc));PetscCall(SNESSetUp(npc));}
 std::ofstream progress(out+"/updates.csv");progress<<"case,newton,krylov,norm,elapsed\n";Monitor mon{&progress,0,begin};PetscCall(SNESMonitorSet(s,monitor,&mon,nullptr));
 PetscViewer viewer;PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,(out+"/solver.txt").c_str(),&viewer));PetscCall(SNESView(s,viewer));PetscCall(PetscViewerDestroy(&viewer));
 std::ofstream log(out+"/cases.csv");log<<std::setprecision(17)<<"case,lambda,Q,initial1,initial2,accepted,newton,krylov,physical_residual,map_residual,reason,seconds,min_u,max_u,min_local_derivative,outside_fit_queries,final1,final2\n";
 std::ifstream cases(casesfile);int id;double lam,Q,u1,u2;std::string initial;int done=0;double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 while(cases>>id>>lam>>Q>>u1>>u2>>initial){
  // Independent basin trials must not inherit native NPC work state.
  if(done && method=="aspin"){
   PetscCall(SNESDestroy(&s));PetscCall(DMDestroy(&dm));PetscCall(SNESCreate(PETSC_COMM_SELF,&s));

  PetscCall(DMShellCreate(PETSC_COMM_SELF,&dm));PetscCall(DMSetApplicationContext(dm,&p));PetscCall(DMShellSetGlobalVector(dm,x));PetscCall(DMShellSetMatrix(dm,J));PetscCall(DMSNESSetFunction(dm,globalFunction,&p));PetscCall(DMSNESSetJacobian(dm,globalJacobian,&p));PetscCall(DMSubDomainHookAdd(dm,configureSub,nullptr,nullptr));PetscCall(DMShellSetCreateDomainDecomposition(dm,decomposition));PetscCall(DMShellSetCreateDomainDecompositionScatters(dm,scatters));PetscCall(SNESSetDM(s,dm));PetscCall(SNESSetType(s,SNESASPIN));

   PetscCall(SNESSetTolerances(s,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,1000,PETSC_DEFAULT));PetscCall(SNESSetFromOptions(s));PetscCall(SNESSetUp(s));SNES npc;PetscCall(SNESGetNPC(s,&npc));PetscCall(SNESSetUp(npc));PetscCall(SNESMonitorSet(s,monitor,&mon,nullptr));
  }
  std::fill(p.old.begin(),p.old.end(),-Q/1800);std::fill(p.beta.begin(),p.beta.end(),lam/1800);PetscScalar*a;PetscCall(VecGetArray(x,&a));std::fill(a,a+p.n,u1);if(p.n==2)a[1]=u2;if(initial!="-"){std::ifstream in((std::filesystem::path(input).parent_path().parent_path()/initial).string(),std::ios::binary);int64_t n;in.read((char*)&n,8);in.read((char*)a,p.n*8);if(!in||n!=p.n)throw std::runtime_error("initial field");}PetscCall(VecRestoreArray(x,&a));fit.outside=0;mon.attempt=id;auto tick=std::chrono::steady_clock::now();PetscCall(SNESSolve(s,nullptr,x));auto [physical,map]=norms(x,ctx);SNESConvergedReason reason;PetscInt ni,ki;PetscCall(SNESGetConvergedReason(s,&reason));PetscCall(SNESGetIterationNumber(s,&ni));PetscCall(SNESGetLinearSolveIterations(s,&ki));const PetscScalar*u;PetscCall(VecGetArrayRead(x,&u));double mind=INFINITY;for(int64_t i=0;i<p.n;i++)mind=std::min(mind,1-p.beta[i]*std::exp(u[i]));double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();bool ok=physical<=1e-6&&map<=1e-6;
  log<<id<<','<<lam<<','<<Q<<','<<u1<<','<<u2<<','<<ok<<','<<ni<<','<<ki<<','<<physical<<','<<map<<','<<reason<<','<<sec<<','<<*std::min_element(u,u+p.n)<<','<<*std::max_element(u,u+p.n)<<','<<mind<<','<<fit.outside<<','<<u[0]<<','<<u[p.n-1]<<std::endl;
  if(p.n>2)save(out+"/state_"+std::to_string(id)+".bin",u,p.n);PetscCall(VecRestoreArrayRead(x,&u));done++;if(p.n>2||done%500==0)std::cout<<"CASE "<<id<<" accepted="<<ok<<" Newton="<<ni<<" physical="<<physical<<" map="<<map<<" seconds="<<sec<<std::endl;
 }
 std::ofstream result(out+"/result.json");result<<"{\"cases\":"<<done<<",\"setup_seconds\":"<<setup<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"}";result.close();PetscCall(SNESDestroy(&s));if(method=="aspin")PetscCall(DMDestroy(&dm));PetscCall(MatDestroy(&J));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&f));PetscCall(PetscFinalize());return 0;
}
