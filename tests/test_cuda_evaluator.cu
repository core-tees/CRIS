#include "TwoPhaseTransport/CUDA/TransportEvaluator.cuh"
#include <iostream>

void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}
void compare(double actual,double expected) {
  require(std::isfinite(actual)&&std::isfinite(expected),"nonfinite evaluator result");
  require(std::abs(actual-expected)<=1e-9*std::max(1.,std::abs(expected)),"CPU/CUDA evaluator mismatch");
}
void scenario(const char* graph,const char* model,bool learned) {
  transport::TransportProblem p(graph,model,learned);
  p.dt=.5;
  std::vector<double> state(p.g.n),residual(p.g.n),actual(p.g.n),matrix(p.g.n+p.g.q.size());
  for(int i=0;i<p.g.n;++i){state[i]=.2+.1*std::sin(double(i));p.old[i]=.15;}
  transport_gpu::Evaluator gpu(p);
  const bool neural=learned&&p.model.tree.empty();
  for(const auto* buffer:{&gpu.x0,&gpu.x1,&gpu.d0,&gpu.d1}) {
    if(neural)require(buffer->p&&buffer->count>0,"missing neural workspace");
    else require(!buffer->p&&buffer->count==0,"unused neural workspace allocated");
  }
  require((gpu.handle!=nullptr)==neural,"cuBLAS handle does not match evaluator mode");
  require(gpu.w.empty()!=neural&&gpu.b.empty()!=neural,"neural weights do not match evaluator mode");
  transport_gpu::Buffer<double> device_state;
  device_state.upload(state);
  gpu.timestep();
  for(bool gradient:{false,true}) {
    p.evaluate(state.data(),residual.data(),gradient);
    gpu.evaluate(device_state.p,gpu.r.p,gradient);
    transport_gpu::cuda_check(cudaMemcpy(actual.data(),gpu.r.p,actual.size()*sizeof(double),cudaMemcpyDeviceToHost));
    for(int i=0;i<p.g.n;++i)compare(actual[i],residual[i]);
    if(gradient) {
      transport_gpu::cuda_check(cudaMemcpy(matrix.data(),gpu.coo.p,matrix.size()*sizeof(double),cudaMemcpyDeviceToHost));
      for(int i=0;i<p.g.n;++i) {
        compare(matrix[i+p.g.ptr[i]],p.diagonal(i));
        for(auto k=p.g.ptr[i];k<p.g.ptr[i+1];++k)compare(matrix[i+1+k],p.offdiagonal(i,k));
      }
    }
  }
}
int main(int argc,char** argv) {
  try {
    require(argc==4,"graph, neural model and Chebyshev model are required");
    for(bool learned:{false,true})for(const char* model:{argv[2],argv[3]})scenario(argv[1],model,learned);
    std::cout<<"PASS: evaluator allocations and CPU/CUDA residual/Jacobian parity in all modes\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
