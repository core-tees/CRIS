#include <petscsnes.h>
#include <petscdevice.h>
#include <cstring>
using std::isfinite;
#define __host__
#define __device__
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double sec(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
__host__ __device__ double logistic(double t){return t>=0?1/(1+exp(-t)):exp(t)/(1+exp(t));}
__host__ __device__ void rd(int kind,const double*q,double z,double lambda,double&r,double&d){
 if(kind==0){
  double b=expm1(lambda*log1p(q[1])),s=logistic(z),c=logistic(-z);
  double ls=z>=0?-log1p(exp(-z)):z-log1p(exp(z));
  double lc=z>=0?-z-log1p(exp(-z)):-log1p(exp(z));
  double v=q[2]*ls-q[3]*lc-log(q[4]),f=logistic(v),fc=logistic(-v),w=b/(1+b);
  r=q[0]>.5?(1-q[0])-(c/(1+b)+w*fc):s/(1+b)+w*f-q[0];
  d=s*c/(1+b)+w*f*fc*(q[2]*c+q[3]*s);
 }else{
  if(q[1]==0||lambda==0){r=z;d=1;return;}
  double v=log(q[1])+q[0]+log(lambda)+z;
  r=z>.5?(z-1)-expm1(v):z-exp(v);d=-expm1(v);
 }
}
__host__ __device__ bool certified(int kind,const double*q,double z,double lambda){
 double r,d;rd(kind,q,z,lambda,r,d);
 bool domain=kind==0?(z>=-256&&z<=256):(z>=0&&z<=1);
 return domain&&isfinite(r)&&isfinite(d)&&d>0&&fabs(r)<=(kind==0?2e-13:2e-14)&&fabs(r/d)<=(kind==0?2e-12:1e-9);
}
struct Ctx{int n,kind;double lambda;double*q,*fixed,*diag;int*active,*bad;};
void eval(Ctx c,const double*z,double*r,double*d){for(int i=0;i<c.n;i++){
 double rr,dd;if(!c.active[i]){rr=z[i]-c.fixed[i];dd=1;}else{
 rd(c.kind,c.q+5*i,z[i],c.lambda,rr,dd);
 if(c.kind==0){double scale=fmin(c.q[5*i],1-c.q[5*i]);rr/=scale;dd/=scale;}
 bool domain=c.kind==0?(z[i]>=-256&&z[i]<=256):(z[i]>=0&&z[i]<=1);
 if(!domain||!isfinite(rr)||!isfinite(dd)||dd<=0)*c.bad=1;}
 if(r)r[i]=rr;if(d)d[i]=dd;}}
void fallback(Ctx c,double*z,int*which){for(int i=0;i<c.n;i++){if(!which[i])continue;
 double lo=c.kind==0?-256:0,hi=c.kind==0?256:1;
 for(int k=0;k<(c.kind==0?100:80);k++){double m=(lo+hi)/2,r,d;rd(c.kind,c.q+5*i,m,1,r,d);if(r<0)lo=m;else hi=m;}
 double r1,d1,r2,d2;rd(c.kind,c.q+5*i,lo,1,r1,d1);rd(c.kind,c.q+5*i,hi,1,r2,d2);z[i]=fabs(r1)<fabs(r2)?lo:hi;}}
PetscErrorCode fun(SNES sn,Vec x,Vec f,void*p){PetscFunctionBeginUser;auto&c=*(Ctx*)p;const double*z;double*r;
 PetscCall(VecGetArrayRead(x,&z));PetscCall(VecGetArray(f,&r));*c.bad=0;eval(c,z,r,nullptr);
 PetscCall(VecRestoreArrayRead(x,&z));PetscCall(VecRestoreArray(f,&r));if(*c.bad)PetscCall(SNESSetFunctionDomainError(sn));PetscFunctionReturn(PETSC_SUCCESS);}
PetscErrorCode jac(SNES,Vec x,Mat A,Mat P,void*p){PetscFunctionBeginUser;auto&c=*(Ctx*)p;const double*z;PetscCall(VecGetArrayRead(x,&z));eval(c,z,nullptr,c.diag);PetscCall(VecRestoreArrayRead(x,&z));PetscCall(MatSetValuesCOO(P,c.diag,INSERT_VALUES));if(A!=P){PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));}PetscFunctionReturn(PETSC_SUCCESS);}
void pc(PetscErrorCode e){if(e)throw std::runtime_error("PETSc error "+std::to_string(e));}
struct Solver{
 Ctx c{};Vec x,f;Mat J;SNES sn;int n;long its=0,calls=0,negative=0;double setup=0;
 Solver(int nn,int kind):n(nn){auto t=Clock::now();c.n=n;c.kind=kind;c.lambda=1;
  c.q=(decltype(c.q))malloc(n*5*sizeof(double));c.fixed=(decltype(c.fixed))malloc(n*sizeof(double));c.diag=(decltype(c.diag))malloc(n*sizeof(double));c.active=(decltype(c.active))malloc(n*sizeof(int));c.bad=(decltype(c.bad))malloc(sizeof(int));
  pc(VecCreateSeq(PETSC_COMM_SELF,n,&x));pc(VecDuplicate(x,&f));pc(MatCreate(PETSC_COMM_SELF,&J));pc(MatSetSizes(J,n,n,n,n));pc(MatSetType(J,MATSEQAIJ));std::vector<PetscInt>ii(n);for(int i=0;i<n;i++)ii[i]=i;pc(MatSetPreallocationCOO(J,n,ii.data(),ii.data()));
  pc(SNESCreate(PETSC_COMM_SELF,&sn));pc(SNESSetType(sn,SNESNEWTONLS));pc(SNESSetFunction(sn,f,fun,&c));pc(SNESSetJacobian(sn,J,J,jac,&c));SNESLineSearch ls;pc(SNESGetLineSearch(sn,&ls));pc(SNESLineSearchSetType(ls,SNESLINESEARCHBT));pc(SNESSetTolerances(sn,1e-14,0,0,50,10000));
  KSP k;PC p;pc(SNESGetKSP(sn,&k));pc(KSPSetType(k,KSPPREONLY));pc(KSPGetPC(k,&p));pc(PCSetType(p,PCJACOBI));setup=sec(t);
 }
 void put(const std::vector<double>&z){double*d;pc(VecGetArray(x,&d));memcpy(d,z.data(),n*sizeof(double));pc(VecRestoreArray(x,&d));}
 void get(std::vector<double>&z){const double*d;pc(VecGetArrayRead(x,&d));memcpy(z.data(),d,n*sizeof(double));pc(VecRestoreArrayRead(x,&d));}
 void mask(const std::vector<int>&a,const std::vector<double>&z){memcpy(c.active,a.data(),n*sizeof(int));memcpy(c.fixed,z.data(),n*sizeof(double));}
 int solve(std::vector<double>&z,double lambda){c.lambda=lambda;put(z);pc(SNESSolve(sn,nullptr,x));get(z);PetscInt it;SNESConvergedReason reason;pc(SNESGetIterationNumber(sn,&it));pc(SNESGetConvergedReason(sn,&reason));its+=it;calls++;if(reason<0)negative++;return reason;}
 ~Solver(){SNESDestroy(&sn);MatDestroy(&J);VecDestroy(&f);VecDestroy(&x);free(c.q);free(c.fixed);free(c.diag);free(c.active);free(c.bad);}
};
void preflight(Solver&s,const std::vector<double>&q){
 std::vector<double>z(s.n),out(s.n),der(s.n);std::vector<int>a(s.n,1);for(int i=0;i<s.n;i++)z[i]=s.c.kind==0?log(q[5*i]/(1-q[5*i]))*.3:.25;
 s.mask(a,z);s.put(z);s.c.lambda=.5;pc(fun(s.sn,s.x,s.f,&s.c));pc(jac(s.sn,s.x,s.J,s.J,&s.c));const double*d;pc(VecGetArrayRead(s.f,&d));memcpy(out.data(),d,s.n*sizeof(double));pc(VecRestoreArrayRead(s.f,&d));memcpy(der.data(),s.c.diag,s.n*sizeof(double));
 double er=0,ed=0,fd=0;for(int i=0;i<s.n;i++){double r,j,rp,dp,rm,dm;rd(s.c.kind,&q[5*i],z[i],.5,r,j);rd(s.c.kind,&q[5*i],z[i]+1e-5,.5,rp,dp);rd(s.c.kind,&q[5*i],z[i]-1e-5,.5,rm,dm);double scale=s.c.kind==0?std::min(q[5*i],1-q[5*i]):1;r/=scale;j/=scale;rp/=scale;rm/=scale;er=std::max(er,fabs(out[i]-r)/(1+fabs(r)));ed=std::max(ed,fabs(der[i]-j)/(1+fabs(j)));fd=std::max(fd,fabs((rp-rm)/2e-5-j)/(1+fabs(j)));}
 std::cout<<"{\"preflight\":true,\"kind\":"<<s.c.kind<<",\"cpu_callback_value\":"<<er<<",\"cpu_callback_jacobian\":"<<ed<<",\"directional_error\":"<<fd<<"}"<<std::endl;
 if(er>1e-12||ed>1e-12||fd>1e-5)throw std::runtime_error("CPU callback preflight failed");
}
int main(int argc,char**argv){PetscInitialize(&argc,&argv,nullptr,nullptr);int result=0;try{
 if(argc<5)throw std::runtime_error("labels kind input.raw output repeats");int kind=std::stoi(argv[1]),reps=std::stoi(argv[4]);std::string path=argv[2],prefix=argv[3];std::ifstream in(path,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("input missing");size_t bytes=in.tellg();if(bytes%40)throw std::runtime_error("input size");size_t N=bytes/40;in.seekg(0);auto readt=Clock::now();std::vector<double>raw(5*N);in.read((char*)raw.data(),bytes);double readseconds=sec(readt);int n=std::min<size_t>(4096,N);Solver s(n,kind);std::cout<<std::setprecision(17);std::ofstream history(prefix+".jsonl");
 for(int rep=0;rep<reps;rep++){
  auto all=Clock::now();double transfer=0,directtime=0,conttime=0,fallbacktime=0,checktime=0,writetime=0,maxr=0,maxcorr=0;long direct=0,continued=0,fb=0,failed=0,stages=0,reject=0,oldits=s.its,oldcalls=s.calls,oldneg=s.negative;std::ofstream labels(prefix+".repeat"+std::to_string(rep+1)+".labels",std::ios::binary);
  for(size_t off=0;off<N;off+=n){int m=std::min<size_t>(n,N-off);std::vector<double>q(5*n),z(n),reference(n);std::vector<int>a(n,0),need(n,0);
   for(int i=0;i<n;i++){int j=std::min(i,m-1);std::copy_n(&raw[5*(off+j)],5,&q[5*i]);z[i]=reference[i]=kind==0?log(q[5*i]/(1-q[5*i])):0;a[i]=i<m;}
   auto t=Clock::now();memcpy(s.c.q,q.data(),5*n*sizeof(double));s.mask(a,z);transfer+=sec(t);if(off==0&&rep==0)preflight(s,q);s.mask(a,z);
   t=Clock::now();s.solve(z,1);directtime+=sec(t);t=Clock::now();for(int i=0;i<m;i++){need[i]=!certified(kind,&q[5*i],z[i],1);direct+=!need[i];}checktime+=sec(t);
   if(std::count(need.begin(),need.end(),1)){
    t=Clock::now();std::vector<double>accepted=z;for(int i=0;i<n;i++)if(need[i])accepted[i]=reference[i];s.mask(need,accepted);double lambda=0,h=.125;int attempts=0;
    while(lambda<1&&h>=1e-10&&attempts<10000){z=accepted;double target=std::min(1.,lambda+h);s.solve(z,target);bool ok=true;for(int i=0;i<m;i++)if(need[i]&&!certified(kind,&q[5*i],z[i],target))ok=false;stages++;attempts++;if(ok){accepted=z;lambda=target;h=std::min(.5,1.5*h);}else{h*=.5;reject++;}}
    z=accepted;conttime+=sec(t);for(int i=0;i<m;i++)if(need[i]&&lambda==1&&certified(kind,&q[5*i],z[i],1)){continued++;need[i]=0;}
   }
   if(std::count(need.begin(),need.end(),1)){t=Clock::now();s.mask(need,z);s.put(z);double*d;pc(VecGetArray(s.x,&d));fallback(s.c,d,s.c.active);pc(VecRestoreArray(s.x,&d));s.get(z);fallbacktime+=sec(t);fb+=std::count(need.begin(),need.end(),1);}
   t=Clock::now();std::vector<double>values(4*m);for(int i=0;i<m;i++){double r,d;rd(kind,&q[5*i],z[i],1,r,d);maxr=std::max(maxr,fabs(r));maxcorr=std::max(maxcorr,fabs(r/d));if(!certified(kind,&q[5*i],z[i],1))failed++;values[4*i]=kind==0?logistic(z[i]):q[5*i]+z[i];values[4*i+1]=z[i];values[4*i+2]=r;values[4*i+3]=d;}checktime+=sec(t);t=Clock::now();labels.write((char*)values.data(),values.size()*sizeof(double));writetime+=sec(t);
   if(failed)throw std::runtime_error("uncertified labels in batch at "+std::to_string(off));
   if(off==0||(off+n)>=N||((off/n)%32==0)){std::cout<<"{\"progress\":true,\"repeat\":"<<rep+1<<",\"processed\":"<<std::min(off+n,N)<<",\"count\":"<<N<<",\"seconds\":"<<sec(all)<<"}"<<std::endl;}
  }
  labels.close();double total=sec(all);std::string record="";std::ostringstream os;os<<std::setprecision(17)<<"{\"repeat\":"<<rep+1<<",\"count\":"<<N<<",\"workflow_seconds\":"<<total<<",\"setup_seconds\":"<<(rep==0?s.setup:0)<<",\"input_read_seconds\":"<<(rep==0?readseconds:0)<<",\"input_transfer_seconds\":"<<transfer<<",\"direct_seconds\":"<<directtime<<",\"continuation_seconds\":"<<conttime<<",\"fallback_seconds\":"<<fallbacktime<<",\"certification_seconds\":"<<checktime<<",\"output_write_seconds\":"<<writetime<<",\"direct_queries\":"<<direct<<",\"continuation_queries\":"<<continued<<",\"fallback_queries\":"<<fb<<",\"failed\":"<<failed<<",\"stage_attempts\":"<<stages<<",\"rejected_stages\":"<<reject<<",\"batched_newton_updates\":"<<s.its-oldits<<",\"snes_calls\":"<<s.calls-oldcalls<<",\"negative_snes_reasons\":"<<s.negative-oldneg<<",\"max_residual\":"<<maxr<<",\"max_correction\":"<<maxcorr<<"}";history<<os.str()<<std::endl;std::cout<<os.str()<<std::endl;
 }
 }catch(const std::exception&e){std::cerr<<"FAILED: "<<e.what()<<std::endl;result=1;}PetscFinalize();return result;}
