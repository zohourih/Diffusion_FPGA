#ifndef DIFFUSION_DIFFUSION_CUDA_COPY_H_
#define DIFFUSION_DIFFUSION_CUDA_COPY_H_

#include "diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDACopy: public DiffusionCUDA {
 public:
  DiffusionCUDACopy(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // This version optimizes 3D stencils. No 2D version is
    // available.
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_copy");
  }
  virtual std::string GetDescription() const {
    return std::string("Dummy version for measuring memory bandwidht");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_COPY_H_
