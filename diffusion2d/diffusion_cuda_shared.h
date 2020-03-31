#ifndef DIFFUSION_DIFFUSION_CUDA_SHARED_H_
#define DIFFUSION_DIFFUSION_CUDA_SHARED_H_

#include "diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDAShared1: public DiffusionCUDA {
 public:
  DiffusionCUDAShared1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared1");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_opt1 + shared memory caching with w/o halo");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared2: public DiffusionCUDA {
 public:
  DiffusionCUDAShared2(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared2");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_opt1 + shared memory caching with w/ halo");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared3: public DiffusionCUDA {
 public:
  DiffusionCUDAShared3(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // 2D is not yet implemented 
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared3");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_opt1 + shared memory caching w/ x-dir halo + branch hoisting");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared3Prefetch: public DiffusionCUDA {
 public:
  DiffusionCUDAShared3Prefetch(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // 2D is not yet implemented 
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared3_prefetch");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_shared3 + vertical prefetch");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared4: public DiffusionCUDA {
 public:
  DiffusionCUDAShared4(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // 2D is not yet implemented 
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared4");
  }
  virtual std::string GetDescription() const {
    return std::string("2-stage temporal blocking w/o z-blocking");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared5: public DiffusionCUDA {
 public:
  DiffusionCUDAShared5(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // 2D is not yet implemented 
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared5");
  }
  virtual std::string GetDescription() const {
    return std::string("2-stage temporal blocking w z-blocking");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared6: public DiffusionCUDA {
 public:
  DiffusionCUDAShared6(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // 2D is not yet implemented 
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared6");
  }
  virtual std::string GetDescription() const {
    return std::string("cuda_shared5 + separate warp for loading diagonal points");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_OPT_H_
