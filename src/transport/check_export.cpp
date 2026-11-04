#include "transport.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("model queries required");transport::Model m(argv[1]);std::ifstream f(argv[2],std::ios::binary);double q[5],ve=0,de=0;int n=0;while(f.read(reinterpret_cast<char*>(q),sizeof(q))){auto y=m.evaluate(q[0],q[1],q[2],true);ve=std::max(ve,std::abs(y[0]-q[3]));de=std::max(de,std::abs(y[1]-q[4]));n++;}std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"value_max\":"<<ve<<",\"derivative_max\":"<<de<<"}\n";if(n!=2000||ve>1e-11||de>1e-8)throw std::runtime_error("export mismatch");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
