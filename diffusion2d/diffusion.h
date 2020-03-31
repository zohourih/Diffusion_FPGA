#ifndef DIFFUSION_DIFFUSION_H_
#define DIFFUSION_DIFFUSION_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>

// Radius of stencil, e.g 5-point stencil => 1
#ifndef RAD
  #define RAD  1
#endif

#define REAL float
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#include "common/stopwatch.h"

#define OFFSET2D(i, j, nx) \
  ((i) + (j) * (nx))
#define OFFSET3D(i, j, k, nx, ny) \
  ((i) + (j) * (nx) + (k) * (nx) * (ny))

#define SHIFT3(x, y, z) x = y; y = z
#define SHIFT4(x, y, z, k) x = y; y = z; z = k

#define STRINGIFY(x) #x
#ifndef UNROLL
//#define PRAGMA_UNROLL(x)
#define PRAGMA_UNROLL
#elif UNROLL == 0
//#define PRAGMA_UNROLL(x) _Pragma("unroll")
#define PRAGMA_UNROLL _Pragma("unroll")
#elif UNROLL > 0
#define PRAGMA_UNROLL__(x, y) STRINGIFY(x y)
#define PRAGMA_UNROLL_(x) PRAGMA_UNROLL__(unroll, x)
//#define PRAGMA_UNROLL(x) _Pragma(PRAGMA_UNROLL_(x))
#define PRAGMA_UNROLL _Pragma(PRAGMA_UNROLL_(UNROLL))
#else
#error Invalid macro definition
#endif



namespace diffusion {

inline
void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time,
                int ndim) {
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
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
            *(1.0 - ay*cos(ky*y));
        if (ndim == 3) {
          REAL z = dz*((REAL)(jz + 0.5));          
          f0 *= (1.0 - az*cos(kz*z));
        }
        buff[j] = f0;
      }
    }
  }
}

#define BX 64
#define BY 64

inline
void BaselineTest(REAL *f1_, REAL *f2_,
                const int nx_, const int ny_, const int nz_,
                const REAL cc_, const REAL cw_, const REAL ce_,
                const REAL cs_, const REAL cn_, const REAL cb_,
                const REAL ct_, const REAL count, int ndim) {
  int i;
  for (i = 0; i < count; ++i) {
  #pragma omp parallel for firstprivate(i)
    for (int z = 0; z < nz_; z++) {
      for (int y = 0; y < ny_; y+=BY) {
        for (int x = 0; x < nx_; x+=BX) {
	     for (int by = y; by < ((y+BY < ny_) ? y+BY : ny_); by++) {
	       for (int bx = x; bx < ((x+BX < nx_) ? x+BX : nx_); bx++) {

              int c, w[RAD], e[RAD], n[RAD], s[RAD], b[RAD], t[RAD];
              c =  bx + by * nx_ + z * nx_ * ny_;

              // address on boundary
              int wb =    0    +     by    * nx_ +     z     * nx_ * ny_;
              int eb = nx_ - 1 +     by    * nx_ +     z     * nx_ * ny_;
              int sb =    bx   +     0     * nx_ +     z     * nx_ * ny_;
              int nb =    bx   + (ny_ - 1) * nx_ +     z     * nx_ * ny_;
              int bb =    bx   +     by    * nx_ +     0     * nx_ * ny_;
              int tb =    bx   +     by    * nx_ + (nz_ - 1) * nx_ * ny_;

              for (int j = 0; j < RAD; j++)
              {
                w[j] = (bx <= j)       ? wb : c - (j + 1);
                e[j] = (bx >= nx_-1-j) ? eb : c + (j + 1);
                s[j] = (by <= j)       ? sb : c - nx_ * (j + 1);
                n[j] = (by >= ny_-1-j) ? nb : c + nx_ * (j + 1);
                b[j] = (z <= j)        ? bb : c - nx_ * ny_ * (j + 1);
                t[j] = (z >= nz_-1-j)  ? tb : c + nx_ * ny_ * (j + 1);
              }

              REAL f = 0;

              f = cc_ * f1_[c];
              if (ndim == 2)
              {
                for (int j = 0; j < RAD; j++)
                {
                  f = f           +
                  cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                  cs_ * f1_[s[j]] + cn_ * f1_[n[j]];
                }
              }
              else if (ndim == 3)
              {
                for (int j = 0; j < RAD; j++)
                {
                  f = f           +
                  cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                  cs_ * f1_[s[j]] + cn_ * f1_[n[j]] +
                  cb_ * f1_[b[j]] + ct_ * f1_[t[j]];
                }
              }

              f2_[c] = f;
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
  

class Diffusion {
 protected:
  int ndim_;  
  int nx_;
  int ny_;
  int nz_;
  REAL kappa_;
  REAL dx_, dy_, dz_;
  REAL kx_, ky_, kz_;
  REAL dt_;
  REAL ce_, cw_, cn_, cs_, ct_, cb_, cc_;
  
 public:
  bool alt_flag;
  Diffusion(int ndim, const int *dims):
      ndim_(ndim), kappa_(0.1) {
    nx_ = dims[0];
    ny_ = (ndim > 1) ? dims[1] : 1;
    nz_ = (ndim > 2) ? dims[2] : 1;
    REAL l = 1.0;
    REAL scaling_factor = pow(10, 1 - RAD);
    dx_ = l / nx_;
    dy_ = l / ny_;
    dz_ = l / nz_;
    kx_ = ky_ = kz_ = 2.0 * M_PI;
    dt_ = 0.1 * dx_ * dx_ / kappa_;
    ce_ = cw_ = kappa_*dt_/(dx_*dx_) * scaling_factor;
    cn_ = cs_ = kappa_*dt_/(dy_*dy_) * scaling_factor;
    ct_ = cb_ = kappa_*dt_/(dz_*dz_) * scaling_factor;
    cc_ = 1.0 - (ce_ + cw_ + cn_ + cs_ + (ndim_ == 3 ? (ct_ + cb_) : 0));
  }
  
  virtual std::string GetName() const = 0;
  virtual std::string GetDescription() const = 0;  
  void RunBenchmark(int count, bool dump, bool warmup) {
    std::cout << "*** Diffusion Benchmark ***\n";
    std::cout << "Benchmark: " << GetName()
              << " (" << GetDescription() << ")\n";
    Setup();    
    std::cout << "Initializing benchmark input...\n";
    InitializeInput();
    std::cout << "Iteration count: " << count << "\n";
    std::cout << "Grid size: ";
    if (ndim_ == 2) {
      std::cout << nx_ << "x" << ny_ << "\n";
    } else if (ndim_ == 3) {
      std::cout << nx_ << "x" << ny_ << "x" << nz_ << "\n";
    }
    if (warmup) {
      std::cout << "Warming up the kernel...\n";
      WarmingUp();
      std::cout << "Reinitializing benchmark input...\n";    
      InitializeInput();
    }
    std::cout << "Running the kernel...\n";    
    Stopwatch st;
    StopwatchStart(&st);
    RunKernel(count);
    float elapsed_time = StopwatchStop(&st);
    std::cout << "Benchmarking finished.\n";
    DisplayResult(count, elapsed_time);
    if (dump) Dump();
    FinalizeBenchmark();
  }

 protected:
  std::string GetDumpPath() const {
    return std::string("diffusion_result.")
        + GetName() + std::string(".out");
  }
  virtual void Setup() = 0;  
  virtual void InitializeInput() = 0;  
  virtual void RunKernel(int count) = 0;
  virtual void WarmingUp() = 0;  
  virtual void Dump() const = 0;
  virtual REAL GetAccuracy(int count) = 0;
  virtual void FinalizeBenchmark() = 0;    
  
  float GetThroughput(int count, float time) {
    float out = 0;
    if (ndim_ == 2) {
      out = ((nx_ * ny_) / (time * 1.0e9)) * sizeof(REAL) * 2.0 * count;
    } else if (ndim_ == 3) {
      out = ((nx_ * ny_ * nz_) / (time * 1.0e9)) * sizeof(REAL) * 2.0 * count;
    }

    return out;
  }
  float GetGFLOPS(int count, float time) {
    float out = 0;
    if (ndim_ == 2) {
      out = ((nx_ * ny_) / (time * 1.0e9)) * (1 + 8 * RAD) * count;
    } else if (ndim_ == 3) {
      out = ((nx_ * ny_ * nz_) / (time * 1.0e9)) * (1 + 12 * RAD) * count;
    }

    return out;
  }
  virtual void DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
           GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
           GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  }
  REAL *GetCorrectAnswerOriginal(int count) const {
    REAL *f = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f);
    Initialize(f, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, count * dt_, ndim_);
    return f;
  }
  REAL *GetCorrectAnswerAlternate(int count) const {
    REAL *f1 = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f1);    
    REAL *f2 = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f2);
    Initialize(f1, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, 0.0, ndim_);
    BaselineTest(f1, f2,
                 nx_, ny_, nz_,
                 cc_, cw_, ce_,
                 cs_, cn_, cb_,
                 ct_, count, ndim_);
    return (count % 2 == 0) ? f1 : f2;
  }
  REAL *GetCorrectAnswer(int count) const {
    if (!alt_flag) {
      return GetCorrectAnswerOriginal(count);
    }
    else {
      return GetCorrectAnswerAlternate(count);
    }
  }
};

}

#endif /* DIFFUSION_DIFFUSION_H_ */
