#pragma once
#include <fstream>
#include <stdexcept>
#include <vector>

namespace transport {
template<class T> std::vector<T> read(std::ifstream& f, std::size_t n) {
  std::vector<T> x(n); f.read(reinterpret_cast<char*>(x.data()), n*sizeof(T));
  if (!f) throw std::runtime_error("truncated transport package"); return x;
}
}
