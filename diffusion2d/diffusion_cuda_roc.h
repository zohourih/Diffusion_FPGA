#ifndef DIFFUSION_DIFFUSION_CUDA_ROH_H_
#define DIFFUSION_DIFFUSION_CUDA_ROH_H_

#include "diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDAROC: public DiffusionCUDA {
 public:
  DiffusionCUDAROC(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_roc");
  }
  virtual void RunKernel(int count);

};

}

#endif // DIFFUSION_DIFFUSION_CUDA_ROH_H_ 
