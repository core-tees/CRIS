#ifndef __FLUXFUNCTORSHPP_
#define __FLUXFUNCTORSHPP_

#include <cmath>
#include <cfloat>
#include <tuple>

struct CocurrentFracFlux
{
  CocurrentFracFlux(double _M, double _nw, double _nnw, double _Ng)
    : M(_M), nw(_nw), nnw(_nnw), Ng(_Ng) {  }
  std::tuple<double,double> val_grad( double u) const {
    const double krw = std::pow(u, nw); 
	const double krw_grad = nw * krw/(u + DBL_MIN);
    const double krnw = std::pow(1.0 - u, nnw); 
	const double krnw_grad = -nnw * krnw/(1 - u + DBL_MIN);
    const double N  = krw * (1.0 - Ng * krnw);
    const double D  = krw + M * krnw;
    const double Np = krw_grad * (1.0 - Ng * krnw) - krw * (Ng * krnw_grad);
    const double Dp = krw_grad + M *krnw_grad;

    if (u < 0)
        return { 0.0, 0.0 };
    if (u > 1.0)
        return { 1.0, 0.0 };
    return { N / D, (Np * D - N * Dp) / (D * D) };
  }
  template <typename Ty>
  Ty operator () (const Ty& u) const {
      return pow(u, nw) * (1.0 - Ng * pow(1.0 - u, nnw))/(pow(u, nw) + M * pow(1.0 - u, nnw));
  }
  
  double operator()(double u) const { return std::get<0>(val_grad(u)); }
  double grad(double u) const       { return std::get<1>(val_grad(u)); }

private:
  double M, nw, nnw, Ng;
};


#endif // __FLUXFUNCTORSHPP_ included
