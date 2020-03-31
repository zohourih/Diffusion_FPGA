#include "diffusion_openmp.h"

namespace diffusion {

void DiffusionOpenMP::Initialize2D() {
  REAL time = 0.0;
  REAL ax = exp(-kappa_*time*(kx_*kx_));
  REAL ay = exp(-kappa_*time*(ky_*ky_));
  int jy;
#pragma omp parallel for        
  for (jy = 0; jy < ny_; jy++) {
    int jx;
    for (jx = 0; jx < nx_; jx++) {
      int j = jy*nx_ + jx;
      REAL x = dx_*((REAL)(jx + 0.5));
      REAL y = dy_*((REAL)(jy + 0.5));
      REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx_*x))
          *(1.0 - ay*cos(ky_*y));
      f1_[j] = f0;
    }
  }
}

void DiffusionOpenMP::Initialize3D() {
  REAL time = 0.0;
  REAL ax = exp(-kappa_*time*(kx_*kx_));
  REAL ay = exp(-kappa_*time*(ky_*ky_));
  REAL az = exp(-kappa_*time*(kz_*kz_));
  int jz;
#pragma omp parallel for    
  for (jz = 0; jz < nz_; jz++) {
    int jy;
    for (jy = 0; jy < ny_; jy++) {
      int jx;
      for (jx = 0; jx < nx_; jx++) {
        int j = jz*nx_*ny_ + jy*nx_ + jx;
        REAL x = dx_*((REAL)(jx + 0.5));
        REAL y = dy_*((REAL)(jy + 0.5));
        REAL z = dz_*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx_*x))
          *(1.0 - ay*cos(ky_*y))
          *(1.0 - az*cos(kz_*z));
        f1_[j] = f0;
      }
    }
  }
}

void DiffusionOpenMP::InitializeInput() {
  if (ndim_ == 2) {
    Initialize2D();
  } else if (ndim_ == 3) {
    Initialize3D();
  }
}

void DiffusionOpenMP::RunKernel2D(int count) {
  int i;
  for (i = 0; i < count; ++i) {
    int y;
#pragma omp parallel for              
    for (y = 0; y < ny_; y++) {
      int x;
      for (x = 0; x < nx_; x++) {
        int c, w, e, n, s;
        c =  x + y * nx_;
        w = (x == 0)    ? c : c - 1;
        e = (x == nx_-1) ? c : c + 1;
        s = (y == 0)    ? c : c - nx_;
        n = (y == ny_-1) ? c : c + nx_;
        f2_[c] = cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
            + cs_ * f1_[s] + cn_ * f1_[n];
      }
    }
    REAL *t = f1_;
    f1_ = f2_;
    f2_ = t;
  }
  return;
}

void DiffusionOpenMP::RunKernel3D(int count) {
  int i;
  for (i = 0; i < count; ++i) {
    int z;
#pragma omp parallel for        
    for (z = 0; z < nz_; z++) {
      int y;
      for (y = 0; y < ny_; y++) {
        int x;
        for (x = 0; x < nx_; x++) {
          int c, w, e, n, s, b, t;
          c =  x + y * nx_ + z * nx_ * ny_;
          w = (x == 0)    ? c : c - 1;
          e = (x == nx_-1) ? c : c + 1;
          s = (y == 0)    ? c : c - nx_;
          n = (y == ny_-1) ? c : c + nx_;
          b = (z == 0)    ? c : c - nx_ * ny_;
          t = (z == nz_-1) ? c : c + nx_ * ny_;
          f2_[c] = cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
              + cs_ * f1_[s] + cn_ * f1_[n] + cb_ * f1_[b] + ct_ * f1_[t];
        }
      }
    }
    REAL *t = f1_;
    f1_ = f2_;
    f2_ = t;
  }
  return;
}

void DiffusionOpenMP::RunKernel(int count) {
  if (ndim_ == 2) {
    return RunKernel2D(count);
  } else if (ndim_ == 3) {
    return RunKernel3D(count);
  }
  return;
}
    
}
