// Bratu model and objective adapter for the shared native LM optimizer.
#include <native_lm/lm_optimizer.hpp>
#include <native_lm/recovery.hpp>
#include <torch/torch.h>
#include <ATen/Parallel.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

namespace bratu_regularized {
constexpr int P=1425, H=16, L=7;
struct Row { double a,b,y,w; };
struct Jet {double v=0,d=0,dd=0;};
Jet operator+(Jet x,Jet y){return {x.v+y.v,x.d+y.d,x.dd+y.dd};}
Jet operator-(Jet x,Jet y){return {x.v-y.v,x.d-y.d,x.dd-y.dd};}
Jet operator*(Jet x,Jet y){return {x.v*y.v,x.d*y.v+x.v*y.d,x.dd*y.v+2*x.d*y.d+x.v*y.dd};}
Jet operator*(double x,Jet y){return {x*y.v,x*y.d,x*y.dd};}
Jet jtanh(Jet x){double t=std::tanh(x.v),q=1-t*t;return {t,q*x.d,q*x.dd-2*t*q*x.d*x.d};}
struct Net {
 std::vector<double> v; int off[L],nin[L],nout[L];
 Net():v(P){int k=0;for(int l=0;l<L;l++){nin[l]=l==0?2:16;nout[l]=l==6?1:16;off[l]=k;k+=nin[l]*nout[l]+nout[l];}if(k!=P)throw std::runtime_error("size");}
 void derivs(double a,double b,double& value,double& slope,double& curvature,double* gv=nullptr,double* gs=nullptr,double* gc=nullptr) const {
  Jet x[7][16]={};double z=25+a*a;x[0][0]={std::asinh(a/5),1/std::sqrt(z),-a/std::pow(z,1.5)};x[0][1]={(b-.02)/.005,0,0};Jet out;
  for(int l=0;l<L;l++){int ni=nin[l],no=nout[l],o=off[l];for(int r=0;r<no;r++){Jet q={v[o+ni*no+r],0,0};for(int c=0;c<ni;c++)q=q+v[o+r*ni+c]*x[l][c];if(l<6)x[l+1][r]=jtanh(q);else out=q;}}
  value=a+out.v;slope=1+out.d;curvature=out.dd;
  if(!gv)return;Jet delta[16]={},prev[16]={};delta[0]={1,0,0};Jet one={1,0,0};
  for(int l=6;l>=0;l--){int ni=nin[l],no=nout[l],o=off[l];for(int r=0;r<no;r++){for(int c=0;c<ni;c++){Jet j=delta[r]*x[l][c];int ix=o+r*ni+c;gv[ix]=j.v;gs[ix]=j.d;gc[ix]=j.dd;}int ix=o+ni*no+r;gv[ix]=delta[r].v;gs[ix]=delta[r].d;gc[ix]=delta[r].dd;}if(l){for(int c=0;c<ni;c++){prev[c]={};for(int r=0;r<no;r++)prev[c]=prev[c]+v[o+r*ni+c]*delta[r];prev[c]=prev[c]*(one-x[l][c]*x[l][c]);}std::copy(prev,prev+16,delta);}}
 }
 double eval(double a,double b,double* grad=nullptr) const {
  double x[7][16]={};x[0][0]=std::asinh(a/5);x[0][1]=(b-.02)/.005;double out=0;
  for(int l=0;l<L;l++){int ni=nin[l],no=nout[l],o=off[l];for(int r=0;r<no;r++){double z=v[o+ni*no+r];for(int c=0;c<ni;c++)z+=v[o+r*ni+c]*x[l][c];if(l<6)x[l+1][r]=std::tanh(z);else out=z;}}
  if(grad){double delta[16]={1},prev[16]={};for(int l=6;l>=0;l--){int ni=nin[l],no=nout[l],o=off[l];for(int r=0;r<no;r++){for(int c=0;c<ni;c++)grad[o+r*ni+c]=delta[r]*x[l][c];grad[o+ni*no+r]=delta[r];}if(l){std::fill(prev,prev+16,0.);for(int c=0;c<ni;c++){for(int r=0;r<no;r++)prev[c]+=v[o+r*ni+c]*delta[r];prev[c]*=1-x[l][c]*x[l][c];}std::copy(prev,prev+16,delta);}}}
  return a+out;
 }
};
std::vector<Row> readrows(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Cannot open training data: "+p);auto size=static_cast<std::streamoff>(f.tellg());if(size<0 || size%sizeof(Row)!=0)throw std::runtime_error("Invalid training data: "+p);size_t n=static_cast<size_t>(size)/sizeof(Row);std::vector<Row> a(n);f.seekg(0);f.read((char*)a.data(),n*sizeof(Row));if(!f)throw std::runtime_error("Cannot read training data: "+p);return a;}
constexpr double h=.001;
struct Metrics {double loss=0, rmse=0, maxerr=0;};
Metrics loss(const Net& net,const std::vector<Row>& in,const std::vector<Row>& reg,int ni,int nr,double lambda,double slope_min,double slope_max,double slope_weight,double curvature_weight){
 double s=0,err=0,mx=0;for(int i=0;i<ni;i++){auto q=in[i];double r=net.eval(q.a,q.b)-q.y;s+=q.w*r*r;err+=r*r;mx=std::max(mx,std::abs(r));}
 const bool active=lambda>0&&(slope_weight>0||curvature_weight>0);
 double t=0;if(active)for(int i=0;i<nr;i++){auto q=reg[i];double p,d,c;net.derivs(q.a,q.b,p,d,c);double excess=d<slope_min?d-slope_min:(d>slope_max?d-slope_max:0.);t+=slope_weight*excess*excess+curvature_weight*c*c;}
 return {.5*(s/ni+(active?lambda*t/nr:0)),std::sqrt(err/ni),mx};
}

using Args = std::map<std::string,std::string>;
std::string option(const Args& args,const std::string& name,const std::string& fallback) {
 auto it=args.find(name);return it==args.end()?fallback:it->second;
}
void save_weights(const std::filesystem::path& path,const Net& model) {
 std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(model.v.data()),P*sizeof(double));
 if(!out)throw std::runtime_error("Cannot write checkpoint");
}
int run(const Args& args) {
 namespace fs=std::filesystem;
 const fs::path dir=args.at("--data-dir"),out=args.at("--run-dir");
 const double lambda=std::stod(option(args,"--regularization","1e-8"));
 const double slope_min=std::stod(option(args,"--slope-min","0"));
 const double slope_max=std::stod(option(args,"--slope-max","1.015"));
 const double slope_weight=std::stod(option(args,"--slope-weight","1"));
 const double curvature_weight=std::stod(option(args,"--curvature-weight","0.01"));
 const int iters=std::stoi(option(args,"--max-proposals","800"));
 const int maxaccepted=std::stoi(option(args,"--max-accepted",std::to_string(iters)));
 const int threads=std::stoi(option(args,"--threads","4"));
 const int block=std::stoi(option(args,"--block-size","256"));
 const int patience=std::stoi(option(args,"--patience","0"));
 const auto resume=option(args,"--resume-state","");
 const auto initial=option(args,"--initial-checkpoint","");
 if(!std::isfinite(lambda)||lambda<0||!std::isfinite(slope_min)||!std::isfinite(slope_max)||slope_min>slope_max||
    !std::isfinite(slope_weight)||slope_weight<0||!std::isfinite(curvature_weight)||curvature_weight<0||iters<1||maxaccepted<1||threads<1||block<1||patience<0)
   throw std::runtime_error("Invalid regularized LM settings");
 if(!resume.empty()&&!initial.empty())throw std::runtime_error("Cannot mix recovery and an initial checkpoint");
 torch::set_num_threads(threads);at::set_num_interop_threads(1);omp_set_dynamic(0);omp_set_num_threads(threads);
 const bool reg_active=lambda>0&&(slope_weight>0||curvature_weight>0);
 auto in=readrows((dir/"interior.bin").string()),val=readrows((dir/"validation.bin").string());
 std::vector<Row> reg,vr;
 if(reg_active) {reg=readrows((dir/"regularity.bin").string());vr=readrows((dir/"validation_reg.bin").string());}
 if(in.empty()||val.empty()||(reg_active&&(reg.empty()||vr.empty())))throw std::runtime_error("Training data are incomplete");
 const int ni=static_cast<int>(in.size()),nr=static_cast<int>(reg.size());
 auto metrics=[&](const Net& n,const std::vector<Row>& x,const std::vector<Row>& r) {
   return loss(n,x,r,static_cast<int>(x.size()),static_cast<int>(r.size()),lambda,slope_min,slope_max,slope_weight,curvature_weight);
 };
 std::ostringstream settings;
 settings<<std::setprecision(17)<<fs::weakly_canonical(dir).string()<<'\n'<<lambda<<'\n'<<slope_min<<'\n'<<slope_max<<'\n'<<slope_weight<<'\n'<<curvature_weight<<'\n';
 if(!resume.empty()) {
   std::ifstream saved_settings(fs::path(resume).parent_path()/"settings.txt");
   std::ostringstream saved;saved<<saved_settings.rdbuf();
   if(!saved_settings || saved.str()!=settings.str())throw std::runtime_error("Recovery settings or training data directory do not match");
 }
 fs::create_directories(out);
 if(resume.empty()&&fs::exists(out/"optimizer_trace.csv"))throw std::runtime_error("Refusing to overwrite existing trace");
 {std::ofstream saved_settings(out/"settings.txt");saved_settings<<settings.str();if(!saved_settings)throw std::runtime_error("Cannot write training settings");}
 Net net,best;native_lm::LmState state;state.damping=.001;double elapsed=0;
 if(!resume.empty()) {
   auto saved=native_lm::read_recovery(resume,P);net.v=saved.current;best.v=saved.best;state=saved.state;elapsed=saved.phase_seconds;
 } else {
   std::ifstream f(initial.empty()?dir/"initial.bin":fs::path(initial),std::ios::binary);
   f.read(reinterpret_cast<char*>(net.v.data()),P*sizeof(double));
   if(!f||f.peek()!=EOF)throw std::runtime_error("Invalid initial checkpoint");
   auto train=metrics(net,in,reg),valid=metrics(net,val,vr);
   state.objective_loss=train.loss;state.training_mse=train.rmse*train.rmse;
   state.validation_mse=valid.loss;state.best_validation_mse=valid.loss;best=net;
   save_weights(out/"checkpoint_000000.bin",net);
 }
 const bool trace_exists=fs::exists(out/"optimizer_trace.csv")&&fs::file_size(out/"optimizer_trace.csv")>0;
 std::ofstream trace(out/"optimizer_trace.csv",std::ios::app);trace<<std::setprecision(17);
 if(!trace_exists)trace<<"iteration,accepted_updates,status,damping,objective,train_mse,validation_objective,best_validation_objective\n";
 native_lm::LmControl control;control.max_iterations=std::max(0,iters-state.iteration);
 control.max_accepted_updates=maxaccepted;control.max_validation_failures=patience;
 control.scaled_cholesky=true;control.max_damping_trials=15;control.max_step_trials=7;
 control.damping_decrease=1.0/.7;control.damping_max=1e15;control.damping_min=1e-12;control.stop_on_damping_max=true;
 double current_objective=state.objective_loss;
 const auto start_time=std::chrono::steady_clock::now();
 const auto opts=torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
 auto result=native_lm::optimize(net,control,state,
  [&](const Net& model){
   native_lm::LmLinearization normal{torch::zeros({P,P},opts),torch::zeros({P,1},opts),1,{}};
   for(int typ=0;typ<2;++typ) {
    if(typ&&!reg_active)continue;
    int n=typ?nr:ni;const auto& rows=typ?reg:in;
    for(int start=0;start<n;start+=block) {
     int count=std::min(block,n-start),terms=typ?2:1;
     auto J=torch::empty({count*terms,P},opts),r=torch::empty({count*terms,1},opts);
     double* jj=J.data_ptr<double>();double* rr=r.data_ptr<double>();
#pragma omp parallel for schedule(static)
     for(int k=0;k<count;k++) {
      Row q=rows[start+k];double* j=jj+size_t(k*terms)*P;double base[P],slope_grad[P],curv_grad[P];
      double y=model.eval(q.a,q.b,base);
      if(!typ){double scale=std::sqrt(q.w/ni);rr[k]=scale*(y-q.y);for(int t=0;t<P;t++)j[t]=scale*base[t];}
      else {
       double slope,curv;model.derivs(q.a,q.b,y,slope,curv,base,slope_grad,curv_grad);
       double excess=slope<slope_min?slope-slope_min:(slope>slope_max?slope-slope_max:0.);
       double ss=std::sqrt(lambda*slope_weight/nr),cs=std::sqrt(lambda*curvature_weight/nr);
       rr[2*k]=ss*excess;rr[2*k+1]=cs*curv;double active=(slope>slope_max||slope<slope_min)?1:0;
       for(int t=0;t<P;t++){j[t]=ss*active*slope_grad[t];j[P+t]=cs*curv_grad[t];}
      }
     }
     normal.hessian.addmm_(J.t(),J);normal.gradient.addmm_(J.t(),r);
    }
   }
   return normal;
  },
  [](const Net& model,const torch::Tensor& step){Net candidate=model;auto v=step.contiguous();const auto* p=v.data_ptr<double>();for(int i=0;i<P;i++)candidate.v[i]+=p[i];return candidate;},
  [&](const Net& candidate){
   native_lm::CandidateEvaluation e;auto train=metrics(candidate,in,reg);e.objective_loss=train.loss;
   if(std::isfinite(train.loss)&&train.loss<current_objective) {
    auto validation=metrics(candidate,val,vr);e.training_mse=train.rmse*train.rmse;e.validation_mse=validation.loss;
   }
   return e;
  },
  [&](const Net& model,const native_lm::LmIteration& event){
   const auto& st=event.state;current_objective=st.objective_loss;
   if(event.validation_improved)best=model;
   trace<<st.iteration<<','<<st.accepted_updates<<','<<(event.status==native_lm::LmIteration::Status::Accepted?"accepted":"rejected")<<','<<st.damping<<','<<st.objective_loss<<','<<st.training_mse<<','<<st.validation_mse<<','<<st.best_validation_mse<<'\n';trace.flush();
   const double seconds=elapsed+std::chrono::duration<double>(std::chrono::steady_clock::now()-start_time).count();
   native_lm::write_recovery(out/("optimizer_state_"+std::to_string(st.iteration%2)+".bin"),model.v.data(),best.v.data(),P,st,seconds);
   if(st.iteration==1||st.iteration%5==0||event.status==native_lm::LmIteration::Status::Rejected) {
    auto validation=metrics(model,val,vr);
    std::cout<<std::setprecision(12)<<"iter "<<st.iteration<<" accepted "<<st.accepted_updates<<" n "<<ni<<" lambda "<<lambda<<" validation "<<st.validation_mse<<" rmse "<<validation.rmse<<" mu "<<st.damping<<std::endl;
   }
  });
 save_weights(out/"final_checkpoint.bin",best);save_weights(out/"latest_checkpoint.bin",result.model);
 std::ofstream result_file(out/"result.json");result_file<<std::setprecision(17)<<"{\n  \"stopping_reason\": \""<<result.stopping_reason<<"\",\n  \"iterations\": "<<result.state.iteration<<",\n  \"accepted_updates\": "<<result.state.accepted_updates<<",\n  \"best_validation_objective\": "<<result.state.best_validation_mse<<"\n}\n";
 return 0;
}
} // namespace bratu_regularized
int train_regularized_bratu(const std::map<std::string,std::string>& args) { return bratu_regularized::run(args); }
