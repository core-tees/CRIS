#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "FluxFunctors.hpp"

#include "cheb.hpp"
namespace transport {
using Index = std::int64_t;
template<class T> std::vector<T> read(std::ifstream& f, std::size_t n) {
  std::vector<T> x(n); f.read(reinterpret_cast<char*>(x.data()), n*sizeof(T));
  if (!f) throw std::runtime_error("truncated transport package"); return x;
}
struct Graph {
  int n=0; bool singular=false;
  std::vector<double> volume, outgoing, injection, q, steps;
  std::vector<Index> ptr;
  std::vector<int> upstream;
  explicit Graph(const std::string& path) {
    std::ifstream f(path,std::ios::binary); auto h=read<Index>(f,5);
    if(h[0]!=0x4352495354505431LL || h[1]<1 || h[1]>20000000 || h[2]<0)
      throw std::runtime_error("invalid graph header");
    n=static_cast<int>(h[1]); singular=h[4]!=0;
    volume=read<double>(f,n); outgoing=read<double>(f,n); injection=read<double>(f,n);
    ptr=read<Index>(f,n+1); upstream=read<int>(f,h[2]); q=read<double>(f,h[2]);
    steps=read<double>(f,h[3]);
    if(ptr[0]!=0 || ptr.back()!=h[2]) throw std::runtime_error("invalid graph pointers");
    for(int i=0;i<n;i++) {
      if(!(volume[i]>0) || outgoing[i]<0 || ptr[i]>ptr[i+1]) throw std::runtime_error("invalid cell");
      for(Index k=ptr[i];k<ptr[i+1];k++)
        if(upstream[k]<0 || upstream[k]>=n || upstream[k]==i || !(q[k]>=0))
          throw std::runtime_error("invalid connection");
    }
  }
};
// The archive uses row-major layer weights followed by biases, and [-1,1]
// normalization after any log1p transformation. Forward derivatives include it.
struct Model {
  std::vector<double> tree;double exponent=2;
  std::vector<int> width, activation, transform;
  std::vector<std::vector<double>> weights,bias;
  std::array<double,3> low{},scale{}; double outlow=0,outscale=1;
  explicit Model(const std::string& path) {
    std::ifstream f(path,std::ios::binary); int layers=read<int>(f,1)[0];if(layers==-1024){int n=read<int>(f,1)[0];exponent=read<double>(f,1)[0];tree=read<double>(f,size_t(n)*74);width={3,1};transform={2,1,0};low={0,0,0};scale={2,2/log1p(2500000.),2};return;}
    if(layers<1 || layers>32) throw std::runtime_error("invalid network depth");
    width=read<int>(f,layers+1); activation=read<int>(f,layers); transform=read<int>(f,3);
    auto lo=read<double>(f,3),hi=read<double>(f,3);
    outlow=read<double>(f,1)[0]; outscale=(read<double>(f,1)[0]-outlow)/2.;
    if(width.front()!=3 || width.back()!=1) throw std::runtime_error("expected three-input scalar inverse");
    for(int c=0;c<3;c++) {
      if(transform[c]!=0 && transform[c]!=1 && !(c==0 && transform[c]==2)) throw std::runtime_error("unknown transform");
      low[c]=transform[c]==1?std::log1p(lo[c]):lo[c];
      scale[c]=2./((transform[c]==1?std::log1p(hi[c]):hi[c])-low[c]);
    }
    for(int l=0;l<layers;l++) {
      if(width[l]<1 || width[l]>256 || width[l+1]<1 || width[l+1]>256 ||
         activation[l]<1 || activation[l]>3) throw std::runtime_error("unsupported network");
      weights.push_back(read<double>(f,width[l]*width[l+1])); bias.push_back(read<double>(f,width[l+1]));
    }
  }
  // Return value and derivative with respect to incoming fractional-flow average.
  std::array<double,2> evaluate(double a,double beta,double old,bool grad) const {
    if(!tree.empty()){double v,d;cheb_value(tree.data(),a,beta,old,exponent,v,d);return {v,d};}
    double x[256],y[256],dx[256],dy[256]; double raw[3]={a,beta,old};
    // Mode 2 is an exact fixed-law export of the three-input network. Its
    // constant constitutive inputs are folded into the first-layer biases.
    if(transform[0]==2)raw[0]=old/(1.+beta)+(beta/(1.+beta))*a;
    for(int c=0;c<3;c++) {
      x[c]=((transform[c]==1?std::log1p(raw[c]):raw[c])-low[c])*scale[c]-1.;
      dx[c]=c==0?scale[c]*(transform[0]==2?beta/(1.+beta):1.)/(transform[c]==1?1.+raw[c]:1.):0.;
    }
    for(std::size_t l=0;l<activation.size();l++) {
      for(int j=0;j<width[l+1];j++) {
        double z=bias[l][j],dz=0;
        for(int k=0;k<width[l];k++) {double w=weights[l][j*width[l]+k];z+=w*x[k];if(grad)dz+=w*dx[k];}
        y[j]=activation[l]==2?std::tanh(z):(activation[l]==1?std::max(0.,z):z);
        if(grad)dy[j]=dz*(activation[l]==2?1.-y[j]*y[j]:(activation[l]==1?(z>0?1.:0.):1.));
      }
      std::copy(y,y+width[l+1],x); if(grad)std::copy(dy,dy+width[l+1],dx);
    }
    return {outlow+outscale*(x[0]+1.),grad?outscale*dx[0]:0.};
  }
};
struct Problem {
  Graph g; Model model; CocurrentFracFlux flux;
  bool learned=false; double dt=0, flux_exponent=2.; std::vector<double> old,f,df,da,dr;
  double reaction_rate=0.,reaction_equilibrium=.5,reaction_width=.01;
  int reaction_type=1; // 1: relaxation; 2: autocatalytic A+B -> 2A.
  Problem(const std::string& graph,const std::string& net,bool learned_)
    :g(graph),model(net),flux(.4,2.,g.singular?.2:2.,0.),learned(learned_),
     old(g.n,0),f(g.n),df(g.n),da(g.n),dr(g.n) {flux_exponent=g.singular?.2:2.;}
  void set_reaction(double rate,double equilibrium,double width) {
    if(!std::isfinite(rate)||rate<0||!(equilibrium>0&&equilibrium<1)||!std::isfinite(width)||width<=0)
      throw std::runtime_error("invalid reaction parameters");
    if(learned&&rate!=0)throw std::runtime_error("network is not trained for reaction");
    reaction_rate=rate;reaction_equilibrium=equilibrium;reaction_width=width;
  }
  // Positive loss term: the physical source is its negative. Complementary
  // species receive opposite sources, conserving their total concentration.
  double reaction(double u) const {
    return reaction_rate==0?0.:(reaction_type==2?-reaction_rate*u*(1.-u):reaction_rate*std::tanh((u-reaction_equilibrium)/reaction_width));
  }
  void set_flux_exponent(double exponent) {
    if(!(exponent>0) || !std::isfinite(exponent))throw std::runtime_error("invalid flux exponent");
    if(learned && exponent!=flux_exponent && model.transform[0]!=2)throw std::runtime_error("network is not trained for the requested flux exponent");
    flux_exponent=exponent;flux=CocurrentFracFlux(.4,2.,exponent,0.);
  }
  void evaluate(const double* u,double* r,bool jac) {
    #pragma omp parallel for schedule(static)
    for(int i=0;i<g.n;i++) {auto v=flux.val_grad(u[i]);f[i]=std::get<0>(v);df[i]=std::get<1>(v);
      double t=reaction_rate==0?0.:std::tanh((u[i]-reaction_equilibrium)/reaction_width);
      dr[i]=reaction_type==2?reaction_rate*(2.*u[i]-1.):reaction_rate/reaction_width*(1.-t*t);}
    #pragma omp parallel for schedule(static)
    for(int i=0;i<g.n;i++) {
      double incoming=g.injection[i];
      for(Index k=g.ptr[i];k<g.ptr[i+1];k++) incoming+=g.q[k]*f[g.upstream[k]];
      if(learned) {
        auto z=model.evaluate(incoming/(g.outgoing[i]+DBL_MIN),dt*g.outgoing[i]/g.volume[i],std::clamp(old[i],0.,1.),jac);
        r[i]=u[i]-z[0]; da[i]=z[1]/(g.outgoing[i]+DBL_MIN);
      } else r[i]=u[i]-old[i]+dt/g.volume[i]*(g.outgoing[i]*f[i]-incoming)+dt*reaction(u[i]);
    }
  }
  double diagonal(int i) const {return learned?1.:1.+dt/g.volume[i]*g.outgoing[i]*df[i]+dt*dr[i];}
  double offdiagonal(int i,Index k) const {
    return -(learned?da[i]:dt/g.volume[i])*g.q[k]*df[g.upstream[k]];
  }
  double physical(const double* u) const {
    double result=0;
    for(int i=0;i<g.n;i++) {
      double incoming=g.injection[i];
      for(Index k=g.ptr[i];k<g.ptr[i+1];k++)incoming+=g.q[k]*flux(u[g.upstream[k]]);
      result=std::max(result,std::abs(u[i]-old[i]+dt/g.volume[i]*(g.outgoing[i]*flux(u[i])-incoming)+dt*reaction(u[i])));
    }return result;
  }
};
}
