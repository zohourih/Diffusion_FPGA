#ifndef DIFFUSION_DIFFUSION_CUDA_SHFL_H_
#define DIFFUSION_DIFFUSION_CUDA_SHFL_H_

#include "diffusion_cuda.h"

#define WARP_SIZE (32)
#define WARP_MASK (WARP_SIZE-1)
#define NUM_WB_X (BLOCK_X / WARP_SIZE)

namespace diffusion {

class DiffusionCUDASHFL1: public DiffusionCUDA {
 public:
  DiffusionCUDASHFL1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_shfl1");
  }
  virtual std::string GetDescription() const {
    return std::string("Data sharing with register shuffling");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
  
};

class DiffusionCUDASHFL2: public DiffusionCUDA {
 public:
  DiffusionCUDASHFL2(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_shfl2");
  }
  virtual std::string GetDescription() const {
    return std::string("Data sharing with register shuffling");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
  
};


class DiffusionCUDASHFL3: public DiffusionCUDA {
 public:
  DiffusionCUDASHFL3(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_shfl3");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_shfl1 + vector loads and stores");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
  
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_SHFL_H_
