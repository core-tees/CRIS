#include "CpuSimulation.hpp"

int main(int argc,char** argv){PetscInitialize(&argc,&argv,nullptr,nullptr);std::cout<<std::setprecision(17);int code=0;try{code=run();}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;code=1;}PetscFinalize();return code;}
