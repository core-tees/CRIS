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
#include "CORE_SCL/BinaryIO.hpp"
#include "CRIS/ChebyshevInverse.hpp"

namespace transport {
// Neural archives use row-major layer weights followed by biases, and [-1,1]
// normalization after any log1p transformation. Forward derivatives include it.
struct LocalInverse {
  std::vector<double> tree;double exponent=2;
  std::vector<int> width, activation, transform;
  std::vector<std::vector<double>> weights,bias;
  std::array<double,3> low{},scale{}; double outlow=0,outscale=1;
  explicit LocalInverse(const std::string& path) {
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
}
