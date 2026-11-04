// Accuracy-only CPU global Newton solver for the archived implicit upwind column.
// Uses the deployed Chebyshev value/derivative implementation unchanged.
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdint>
#include "../transport/cheb.hpp"
double flux(double u){u=std::min(1.,std::max(0.,u));return u*u/(u*u+.4*(1-u)*(1-u));}
double df(double u){if(u<=0||u>=1)return 0;double d=u*u+.4*(1-u)*(1-u);return .8*u*(1-u)/(d*d);}
void field(const std::string& p,const std::vector<double>& u){std::ofstream f(p,std::ios::binary);f.write((char*)u.data(),u.size()*8);}
int main(int argc,char** argv){try{
 if(argc!=6&&argc!=7)throw std::runtime_error("tree-or-original N beta prefix max_newton [front]");
 bool front=argc==7&&std::string(argv[6])=="front";
 bool original=std::string(argv[1])=="original";int n=std::stoi(argv[2]),cap=std::stoi(argv[5]);double beta=std::stod(argv[3]);std::string prefix=argv[4];
 std::vector<double> tree;
 if(!original){std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missing tree");auto bytes=f.tellg();if(bytes<=0||bytes%(74*8))throw std::runtime_error("invalid tree");tree.resize(size_t(bytes)/8);f.seekg(0);f.read((char*)tree.data(),bytes);}
 const double h=front?1:1000./n,dt=front?50:beta*h,atol=front?1e-6:1e-10,left=front?1:.25;int steps=front?1:int(std::llround(200./dt));if(!front&&(n%4||std::abs(steps*dt-200)>1e-8||steps%2))throw std::runtime_error("unaligned archived grid");
 std::vector<double> u(n),old(n),r(n),diag(n),sub(n),delta(n),trial(n);for(int i=0;i<n;i++)u[i]=front?0:(i<n/4?.25:.05);
 field(prefix+".initial.field",u);std::ofstream log(prefix+".jsonl");log<<std::setprecision(17);double budget=0;bool complete=true;int accepted=0;
 auto residual=[&](const std::vector<double>& v,bool grad){double norm=0;for(int i=0;i<n;i++){
   double upstream=i?v[i-1]:left,a=flux(upstream);
   if(original){r[i]=v[i]+beta*flux(v[i])-old[i]-beta*a;if(grad){diag[i]=1+beta*df(v[i]);sub[i]=i?-beta*df(upstream):0;}}
   else{double value,d;cheb_value(tree.data(),a,beta,std::min(1.,std::max(0.,old[i])),2.,value,d);r[i]=v[i]-value;if(grad){diag[i]=1;sub[i]=i?-d*df(upstream):0;}}
   if(!std::isfinite(r[i]))return double(INFINITY);norm=std::max(norm,std::abs(r[i]));}return norm;};
 for(int step=1;step<=steps;step++){
   old=u;double norm=residual(u,true);int it=0;bool failure=false;
   while(norm>atol && it<cap){
     // Exact solve of the assembled lower-bidiagonal GLOBAL Newton Jacobian.
     for(int i=0;i<n;i++)delta[i]=(-r[i]-(i?sub[i]*delta[i-1]:0))/diag[i];
     double eta=1,base=0;for(double x:r)base+=x*x;
     for(;;){for(int i=0;i<n;i++)trial[i]=u[i]+eta*delta[i];double next=residual(trial,false),merit=0;for(double x:r)merit+=x*x;
       if((!original&&!front)||(std::isfinite(next)&&merit<=(1-1e-4*eta)*base)||next<=atol){u.swap(trial);break;}
       eta*=.5;if(eta<1e-12){failure=true;break;}}
     ++it;if(failure)break;norm=residual(u,true);if(!std::isfinite(norm)){failure=true;break;}
   }
   bool ok=!failure&&norm<=atol;log<<"{\"step\":"<<step<<",\"time\":"<<step*dt<<",\"residual_inf\":"<<norm<<",\"accepted\":"<<(ok?"true":"false")<<"}\n";log.flush();
   if(!ok){complete=false;field(prefix+".failed_trial.field",u);field(prefix+".last_accepted.field",old);u=old;break;}accepted++;
   double defect=0;for(int i=0;i<n;i++)defect+=std::abs(u[i]+beta*flux(u[i])-old[i]-beta*flux(i?u[i-1]:left));budget+=defect/n;
   if(step==steps/2||step==steps)field(prefix+(front?".t50.field":step==steps?".t200.field":".t100.field"),u);
 }
 std::ofstream out(prefix+".summary.json");out<<std::setprecision(17)<<"{\"complete\":"<<(complete?"true":"false")<<",\"N\":"<<n<<",\"beta\":"<<beta<<",\"accepted\":"<<accepted<<",\"time\":"<<accepted*dt<<",\"accumulated_mean_absolute_defect\":"<<budget<<",\"accuracy_only\":true,\"atol\":"<<atol<<"}";
 return complete?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
