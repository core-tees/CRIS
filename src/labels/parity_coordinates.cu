#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "CRIS/ChebyshevInverse.hpp"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<double> load(const char* f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=size_t(s.tellg())/8;s.seekg(0);std::vector<double> v(n);s.read((char*)v.data(),n*8);return v;}
CHD inline void record(const double* tree,const double* q,double* o){
 double a=q[0],beta=q[1],old=q[2],exponent=q[3];
 cheb_value(tree,a,beta,old,exponent,o[0],o[1]);
 double x[3]={2*(old/(1+beta)+beta/(1+beta)*a)-1,2*log1p(beta)/log1p(2500000.)-1,2*(exponent-.2)/1.8-1};
 int node=0;const double* r=tree;while(r[0]>=0){node=int(x[int(r[0])]<=r[3]?r[1]:r[2]);r=tree+74*node;}
 o[2]=node;for(int j=0;j<3;j++){o[3+j]=x[j];o[6+j]=(x[j]-r[4+j])*r[7+j]-1;}o[9]=r[7]*2*beta/(1+beta);
}
__global__ void eval(const double* t,const double* q,double* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)record(t,q+6*i,out+10*i);}
int main(int argc,char**argv){if(argc!=4)return 2;auto tree=load(argv[1]),q=load(argv[2]);int n=q.size()/6;double *t,*x,*y;
 check(cudaMalloc(&t,tree.size()*8));check(cudaMalloc(&x,q.size()*8));check(cudaMalloc(&y,n*80));check(cudaMemcpy(t,tree.data(),tree.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(x,q.data(),q.size()*8,cudaMemcpyHostToDevice));
 eval<<<(n+255)/256,256>>>(t,x,y,n);check(cudaGetLastError());check(cudaDeviceSynchronize());std::vector<double> out(n*20);check(cudaMemcpy(out.data()+n*10,y,n*80,cudaMemcpyDeviceToHost));for(int i=0;i<n;i++)record(tree.data(),q.data()+6*i,out.data()+10*i);
 std::ofstream f(argv[3],std::ios::binary);f.write((char*)out.data(),out.size()*8);check(cudaFree(t));check(cudaFree(x));check(cudaFree(y));return 0;}
