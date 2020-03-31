#ifndef DIFFUSION3D_DIFFUSION3D_FORTRAN_H_
#define DIFFUSION3D_DIFFUSION3D_FORTRAN_H_

#include "diffusion3d.h"
#include "baseline.h"

namespace diffusion3d {

extern "C" {
  void diffusion_fortran_(REAL*, REAL*, int*, int*, int*,
                          REAL*, REAL*, REAL*, REAL*, REAL*, REAL*, REAL*, int*);
}

class Diffusion3DFortran : public Baseline {
 public:
  Diffusion3DFortran(int nx, int ny, int nz):
      Baseline(nx, ny, nz) {
  }
  virtual ~Diffusion3DFortran() {}
  virtual std::string GetName() const {
    return std::string("fortran");
  }

  virtual void RunKernel(int count) {
    diffusion_fortran_(f1_, f2_, &nx_, &ny_, &nz_,
                       &ce_, &cw_, &cn_, &cs_, &ct_, &cb_, &cc_,
                       &count);
  }

};

} // namespace diffusion3d

#endif /* DIFFUSION3D_DIFFUSION3D_FORTRAN_H_ */
