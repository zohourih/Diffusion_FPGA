#include "diffusion3d.h"

#include <getopt.h>

#include <string>
#include <vector>
#include <map>

using std::vector;
using std::string;

namespace diffusion3d {

void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time) {
  REAL ax = exp(-kappa*time*(kx*kx));
  REAL ay = exp(-kappa*time*(ky*ky));
  REAL az = exp(-kappa*time*(kz*kz));
  int jz;  
  for (jz = 0; jz < nz; jz++) {
    int jy;
    for (jy = 0; jy < ny; jy++) {
      int jx;
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
          *(1.0 - ay*cos(ky*y))
          *(1.0 - az*cos(kz*z));
        buff[j] = f0;
      }
    }
  }
}

#define BX 64
#define BY 64

void BaselineTest(REAL *f1_, REAL *f2_,
                const int nx_, const int ny_, const int nz_,
                const REAL cc_, const REAL cw_, const REAL ce_,
                const REAL cs_, const REAL cn_, const REAL cb_,
                const REAL ct_, const REAL count) {
  int i;
  for (i = 0; i < count; ++i) {
  #pragma omp parallel for firstprivate(i)
    for (int z = 0; z < nz_; z++) {
      for (int y = 0; y < ny_; y+=BY) {
        for (int x = 0; x < nx_; x+=BX) {
          for (int by = y; by < ((y+BY < ny_) ? y+BY : ny_); by++) {
            for (int bx = x; bx < ((x+BX < nx_) ? x+BX : nx_); bx++) {

              int c, w[RAD], e[RAD], n[RAD], s[RAD], b[RAD], t[RAD];
              c = bx + by * nx_ + z * nx_ * ny_;
          
              // address on boundary
              int wb =    0    +     by    * nx_ +     z     * nx_ * ny_;
              int eb = nx_ - 1 +     by    * nx_ +     z     * nx_ * ny_;
              int nb =    bx   +     0     * nx_ +     z     * nx_ * ny_;
              int sb =    bx   + (ny_ - 1) * nx_ +     z     * nx_ * ny_;
              int bb =    bx   +     by    * nx_ +     0     * nx_ * ny_;
              int tb =    bx   +     by    * nx_ + (nz_ - 1) * nx_ * ny_;

              for (int j = 0; j < RAD; j++) {
                w[j] = (bx <= j)       ? wb : c - (j + 1);
                e[j] = (bx >= nx_-1-j) ? eb : c + (j + 1);
                n[j] = (by <= j)       ? nb : c - nx_ * (j + 1);
                s[j] = (by >= ny_-1-j) ? sb : c + nx_ * (j + 1);
                b[j] = (z  <= j)       ? bb : c - nx_ * ny_ * (j + 1);
                t[j] = (z  >= nz_-1-j) ? tb : c + nx_ * ny_ * (j + 1);
              }
               
              f2_[c] = cc_ * f1_[c];
              for (int j = 0; j < RAD; j++) {
                f2_[c] = f2_[c]          +
                         cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                         cs_ * f1_[s[j]] + cn_ * f1_[n[j]] +
                         cb_ * f1_[b[j]] + ct_ * f1_[t[j]];
              }
            }
          }
        }
      }
    }

    REAL *t = f1_;
    f1_ = f2_;
    f2_ = t;
  }
}

}
