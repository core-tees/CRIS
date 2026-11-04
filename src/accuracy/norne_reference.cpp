// Accuracy-only original-balance Newton. No nonlinear local inversion.
#include "CORE_SCL/DiscreteProblem/TransportProblem.hpp"
#include <queue>
#include <iostream>
#include <iomanip>
#include <omp.h>
int main(int argc,char** argv){try{
 if(argc!=6)throw std::runtime_error("graph initial model max_dt output_prefix");
 omp_set_num_threads(4);transport::TransportProblem p(argv[1],argv[3],false);p.set_flux_exponent(.8);
 int n=p.g.n;double cap=std::stod(argv[4]);std::string out=argv[5];
 std::ifstream input(argv[2],std::ios::binary);p.old=transport::read<double>(input,n);
 // Permute the GLOBAL linear system to triangular form. Only the linear
 // Newton correction is eliminated; all nonlinear residuals are global.
 std::vector<int> indegree(n),order;std::vector<std::vector<int>> children(n);
 for(int i=0;i<n;i++)for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];k++)if(p.g.q[k]>0){indegree[i]++;children[p.g.upstream[k]].push_back(i);}
 std::queue<int> queue;for(int i=0;i<n;i++)if(!indegree[i])queue.push(i);
 while(!queue.empty()){int i=queue.front();queue.pop();order.push_back(i);for(int j:children[i])if(--indegree[j]==0)queue.push(j);}
 if(order.size()!=size_t(n))throw std::runtime_error("Graph is cyclic: direct triangular Newton solve is not applicable");
 std::vector<double> u(n),r(n),trial(n),delta(n);double time=0;int accepted=0,cuts=0;bool complete=true;
 std::ofstream events(out+".jsonl");events<<std::setprecision(17);
 auto norm=[&](const std::vector<double>& x,bool jac){p.evaluate(x.data(),r.data(),jac);double s=0;for(double v:r){if(!std::isfinite(v))return double(INFINITY);s+=v*v;}return std::sqrt(s);};
 for(size_t report=0;report<p.g.steps.size()&&complete;report++){
  double end=time+p.g.steps[report],trialdt=cap;int localcuts=0;
  while(end-time>1e-9){p.dt=std::min(trialdt,end-time);u=p.old;double error=norm(u,true);bool failed=false;int it=0;
   for(;error>1e-6&&it<1000;it++){
    for(int i:order){double rhs=-r[i];for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];k++)rhs-=p.offdiagonal(i,k)*delta[p.g.upstream[k]];delta[i]=rhs/p.diagonal(i);}
    if(accepted==0&&cuts==0&&it==0){double e=0,scale=1;for(int i=0;i<n;i++){double v=p.diagonal(i)*delta[i]+r[i];for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];k++)v+=p.offdiagonal(i,k)*delta[p.g.upstream[k]];e=std::max(e,std::abs(v));scale=std::max(scale,std::abs(r[i]));}if(e/scale>1e-9)throw std::runtime_error("Global Jacobian solve failed its algebraic check");std::cout<<"global_linear_solve_relative_defect "<<e/scale<<std::endl;}
    double eta=1;bool found=false;
    for(int ls=0;ls<40;ls++){for(int i=0;i<n;i++)trial[i]=u[i]+eta*delta[i];double next=norm(trial,false);
     if(std::isfinite(next)&&(next<=1e-6||next*next<=(1-1e-4*eta)*error*error)){u.swap(trial);found=true;break;}eta*=.5;}
    if(!found){failed=true;break;}error=norm(u,true);
   }
   bool ok=!failed&&error<=1e-6;events<<"{\"time_before\":"<<time<<",\"dt\":"<<p.dt<<",\"accepted\":"<<(ok?"true":"false")<<",\"original_residual_l2\":"<<error<<"}\n";events.flush();
   if(ok){time+=p.dt;p.old=u;accepted++;trialdt=std::min(cap,2*p.dt);}
   else{cuts++;localcuts++;trialdt=p.dt/2;if(localcuts>20){complete=false;break;}}
   if(accepted%25==0)std::cout<<"time "<<time<<" accepted "<<accepted<<" cuts "<<cuts<<std::endl;
  }
  if(complete){std::ofstream field(out+".report"+std::to_string(report+1)+".field",std::ios::binary);field.write((char*)p.old.data(),n*8);}
 }
 std::ofstream summary(out+".summary.json");summary<<std::setprecision(17)<<"{\"complete\":"<<(complete?"true":"false")<<",\"time\":"<<time<<",\"accepted\":"<<accepted<<",\"cuts\":"<<cuts<<",\"max_dt\":"<<cap<<",\"atol\":1e-6,\"accuracy_only\":true}";
 return complete?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
