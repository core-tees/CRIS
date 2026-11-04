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

namespace transport {
using Index = std::int64_t;
// Frozen control-volume storage, directed flow connections and schedule.
struct TransportGraph {
  int n=0; bool singular=false;
  std::vector<double> volume, outgoing, injection, q, steps;
  std::vector<Index> ptr;
  std::vector<int> upstream;
  explicit TransportGraph(const std::string& path) {
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
}
