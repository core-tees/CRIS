#pragma once
#include "transport.hpp"
#include <iostream>
namespace transport {
// Checks the derivative actually returned by the piecewise evaluator: the
// polynomial selected at the center. It also exposes finite-difference probes
// crossing a partition, whose secant includes unconstrained leaf-value mismatch.
// No cells are omitted from the selected-polynomial derivative check.
inline double verify_cheb_direction(const Problem& p,const std::vector<double>& state,
 const std::vector<double>& direction,const std::vector<double>& analytic) {
 const int n=p.g.n;const double* tree=p.model.tree.data();double best=1e100;
 std::vector<double> center_a(n),beta(n),old(n),f0(n);std::vector<int> leaf(n);
 for(int i=0;i<n;i++)f0[i]=p.flux(state[i]);
 for(int i=0;i<n;i++){
  double incoming=p.g.injection[i];for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];k++)incoming+=p.g.q[k]*f0[p.g.upstream[k]];
  center_a[i]=incoming/(p.g.outgoing[i]+DBL_MIN);beta[i]=p.dt*p.g.outgoing[i]/p.g.volume[i];old[i]=std::clamp(p.old[i],0.,1.);
  leaf[i]=cheb_leaf(tree,center_a[i],beta[i],old[i],p.model.exponent);
 }
 for(double eps:{1e-4,1e-5,1e-6}){
  std::vector<double> up(n),um(n),fp(n),fm(n);
  for(int i=0;i<n;i++){up[i]=state[i]+eps*direction[i];um[i]=state[i]-eps*direction[i];fp[i]=p.flux(up[i]);fm[i]=p.flux(um[i]);}
  double err=0,rawerr=0,stableerr=0,scale=1,mismatch=0;int crossed=0;
  for(int i=0;i<n;i++){
   double ap=p.g.injection[i],am=ap;
   for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];k++){ap+=p.g.q[k]*fp[p.g.upstream[k]];am+=p.g.q[k]*fm[p.g.upstream[k]];}
   ap/=p.g.outgoing[i]+DBL_MIN;am/=p.g.outgoing[i]+DBL_MIN;
   int lp=cheb_leaf(tree,ap,beta[i],old[i],p.model.exponent),lm=cheb_leaf(tree,am,beta[i],old[i],p.model.exponent);
   double vp,vm,rp,rm,dummy;
   cheb_value(tree,ap,beta[i],old[i],p.model.exponent,vp,dummy,leaf[i]);
   cheb_value(tree,am,beta[i],old[i],p.model.exponent,vm,dummy,leaf[i]);
   cheb_value(tree,ap,beta[i],old[i],p.model.exponent,rp,dummy);
   cheb_value(tree,am,beta[i],old[i],p.model.exponent,rm,dummy);
   double fixed_fd=((up[i]-vp)-(um[i]-vm))/(2*eps);
   double raw_fd=((up[i]-rp)-(um[i]-rm))/(2*eps);
   err=std::max(err,std::abs(fixed_fd-analytic[i]));rawerr=std::max(rawerr,std::abs(raw_fd-analytic[i]));scale=std::max(scale,std::abs(analytic[i]));
   if(lp!=leaf[i]||lm!=leaf[i]){
    crossed++;double vc,other;cheb_value(tree,center_a[i],beta[i],old[i],p.model.exponent,vc,dummy,leaf[i]);
    for(int l:{lp,lm})if(l!=leaf[i]){cheb_value(tree,center_a[i],beta[i],old[i],p.model.exponent,other,dummy,l);mismatch=std::max(mismatch,std::abs(other-vc));}
   }else stableerr=std::max(stableerr,std::abs(raw_fd-analytic[i]));
  }
  std::cout<<"{\"check\":\"Chebyshev_partition_directional_probe\",\"epsilon\":"<<eps
   <<",\"cells_checked\":"<<n<<",\"crossed_leaf_cells\":"<<crossed
   <<",\"selected_polynomial_relative_error\":"<<err/scale
   <<",\"unrestricted_relative_error\":"<<rawerr/scale
   <<",\"unchanged_leaf_relative_error\":"<<stableerr/scale
   <<",\"crossed_polynomial_center_value_mismatch\":"<<mismatch<<"}"<<std::endl;
  best=std::min(best,err/scale);
 }
 return best;
}
}
