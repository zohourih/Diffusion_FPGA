#ifndef DIFFUSION3D_DIFFUSION3D_CUDA_H_
#define DIFFUSION3D_DIFFUSION3D_CUDA_H_

#include "diffusion3d.h"
#include "baseline.h"
#include "common/power_gpu.h"

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

#if defined(ENABLE_ROC)
#define F1_DECL const REAL * __restrict__ f1
#else
#define F1_DECL REAL *f1
#endif

namespace diffusion3d {

class Diffusion3DCUDA: public Baseline {
 public:
  Diffusion3DCUDA(int nx, int ny, int nz):
      Baseline(nx, ny, nz), f1_d_(NULL), f2_d_(NULL),
      block_x_(BLOCK_X), block_y_(BLOCK_Y), block_z_(1), grid_z_(GRID_Z),
	 power(0), energy(0)
  {
    //assert(nx_ % block_x_ == 0);
    //assert(ny_ % block_y_ == 0);
    //assert(nz_ % block_z_ == 0);
  }
  virtual std::string GetName() const {
    return std::string("cuda");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
  virtual void FinalizeBenchmark();
  virtual void DisplayResult(int count, float time);
 protected:
  REAL *f1_d_, *f2_d_;
  int block_x_, block_y_, block_z_;
  int grid_z_;
  cudaEvent_t ev1_, ev2_;
  double power, energy;

};

class Diffusion3DCUDAZBlock: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAZBlock(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_zblock");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAOpt0: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt0(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_opt0");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAOpt1: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt1(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_opt1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAOpt2: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt2(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    // block_x_ = 128;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt2");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAOpt3: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt3(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    // block_x_ = 128;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt3");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAXY: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAXY(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    // block_x_ = 32;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_xy");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAShared(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared1: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared1(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared2: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared2(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared2");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared3: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared3(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
    // block_x_ = 32;
    // block_y_ = 4;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared3");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared4: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared4(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared4");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared5: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared5(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared5");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared6: public Diffusion3DCUDAShared {
 public:
  Diffusion3DCUDAShared6(int nx, int ny, int nz):
      Diffusion3DCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared6");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

}

#endif /* DIFFUSION3D_DIFFUSION3D_CUDA_H_ */
