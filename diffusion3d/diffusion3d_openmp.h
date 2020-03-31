#ifndef DIFFUSION3D_DIFFUSION3D_OPENMP_H_
#define DIFFUSION3D_DIFFUSION3D_OPENMP_H_

#include "diffusion3d.h"
#include "baseline.h"
#include "common/power_cpu.h"

namespace diffusion3d {

class Diffusion3DOpenMP: public Baseline {
 public:
  Diffusion3DOpenMP(int nx, int ny, int nz):
      Baseline(nx, ny, nz), energy(0) {}
  virtual std::string GetName() const {
    return std::string("openmp");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
  virtual void InitializeOMP(
      REAL *buff, const int nx, const int ny, const int nz,
      const REAL kx, const REAL ky, const REAL kz,
      const REAL dx, const REAL dy, const REAL dz,
      const REAL kappa, const REAL time);
  virtual void DisplayResult(int count, float time);
protected:
  double energy;
  
};

}

#endif /* DIFFUSION3D_DIFFUSION3D_OPENMP_H_ */
