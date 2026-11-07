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
#include "CORE_SCL/MeshRepresentations/TransportGraph.hpp"
#include "CRIS/LocalInverse.hpp"
#include "TwoPhaseTransport/FluxFunctors.hpp"

namespace transport {
// Backward-Euler cell balances and the corresponding CRIS residual/Jacobian.
struct TransportProblem {
  TransportGraph g; LocalInverse model; CocurrentFracFlux flux;
  bool learned=false; double dt=0, flux_exponent=2.; std::vector<double> old,f,df,da;
  TransportProblem(const std::string& graph,const std::string& net,bool learned_)
    :g(graph),model(net),flux(.4,2.,g.singular?.2:2.,0.),learned(learned_),
     old(g.n,0),f(g.n),df(g.n),da(g.n) {flux_exponent=g.singular?.2:2.;}
  void set_flux_exponent(double exponent) {
    if(!(exponent>0) || !std::isfinite(exponent))throw std::runtime_error("invalid flux exponent");
    if(learned && exponent!=flux_exponent && model.transform[0]!=2)throw std::runtime_error("network is not trained for the requested flux exponent");
    flux_exponent=exponent;flux=CocurrentFracFlux(.4,2.,exponent,0.);
  }
  void evaluate(const double* u,double* r,bool jac) {
    #pragma omp parallel for schedule(static)
    for(int i=0;i<g.n;i++) {auto v=flux.val_grad(u[i]);f[i]=std::get<0>(v);df[i]=std::get<1>(v);}
    #pragma omp parallel for schedule(static)
    for(int i=0;i<g.n;i++) {
      double incoming=g.injection[i];
      for(Index k=g.ptr[i];k<g.ptr[i+1];k++) incoming+=g.q[k]*f[g.upstream[k]];
      if(learned) {
        auto z=model.evaluate(incoming/(g.outgoing[i]+DBL_MIN),dt*g.outgoing[i]/g.volume[i],std::clamp(old[i],0.,1.),jac);
        r[i]=u[i]-z[0]; da[i]=z[1]/(g.outgoing[i]+DBL_MIN);
      } else r[i]=u[i]-old[i]+dt/g.volume[i]*(g.outgoing[i]*f[i]-incoming);
    }
  }
  double diagonal(int i) const {return learned?1.:1.+dt/g.volume[i]*g.outgoing[i]*df[i];}
  double offdiagonal(int i,Index k) const {
    return -(learned?da[i]:dt/g.volume[i])*g.q[k]*df[g.upstream[k]];
  }
  double physical(const double* u) const {
    double result=0;
    for(int i=0;i<g.n;i++) {
      double incoming=g.injection[i];
      for(Index k=g.ptr[i];k<g.ptr[i+1];k++)incoming+=g.q[k]*flux(u[g.upstream[k]]);
      result=std::max(result,std::abs(u[i]-old[i]+dt/g.volume[i]*(g.outgoing[i]*flux(u[i])-incoming)));
    }return result;
  }
};
}
