#ifndef DIFFUSION_DIFFUSION_CUDA_H_
#define DIFFUSION_DIFFUSION_CUDA_H_

#include "diffusion.h"
#include "baseline.h"

#include <cuda_runtime.h>
#ifndef BLOCK_X
#define BLOCK_X (64)
#endif
#ifndef BLOCK_Y
#define BLOCK_Y (4)
#endif
#ifndef GRID_Z
#define GRID_Z (4)
#endif

#define F1_DECL const REAL * __restrict__
#define F2_DECL REAL * __restrict__

#ifdef USE_LDG
#define LDG(x) __ldg(x)
#else
#define LDG(x) (*(x))
#endif


namespace diffusion {

class DiffusionCUDA: public Baseline {
 public:
  DiffusionCUDA(int nd, const int *dims):
      Baseline(nd, dims), f1_d_(NULL), f2_d_(NULL),
      block_x_(BLOCK_X), block_y_(BLOCK_Y), block_z_(1), grid_z_(GRID_Z)
  {
    //assert(nx_ % block_x_ == 0);
    //assert(ny_ % block_y_ == 0);
    //assert(nz_ % block_z_ == 0);
  }
  virtual std::string GetName() const {
    return std::string("cuda");
  }
  virtual std::string GetDescription() const {
    return std::string("baseline CUDA implementation");
  }
  virtual void Setup();
  virtual void WarmingUp() {
    // more iterations of warming up runs
    RunKernel(50);
  }
  virtual void RunKernel(int count);
  virtual void FinalizeBenchmark();
  virtual void DisplayResult(int count, float time);
  
 protected:
  REAL *f1_d_, *f2_d_;
  int block_x_, block_y_, block_z_;
  int grid_z_;
  cudaEvent_t ev1_, ev2_;

};

class DiffusionCUDARestrict: public DiffusionCUDA {
 public:
  DiffusionCUDARestrict(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_restrict");
  }
  virtual std::string GetDescription() const {
    return std::string("baseline + restrict annotation");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAZBlock: public DiffusionCUDA {
 public:
  DiffusionCUDAZBlock(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_zblock");
  }
  virtual std::string GetDescription() const {
    return std::string("baseline + z-direction blocking");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
};

} // namespace diffusion

#endif /* DIFFUSION_DIFFUSION_CUDA_H_ */
