#ifndef DIFFUSION_BASELINE_H_
#define DIFFUSION_BASELINE_H_

#include "diffusion.h"

#include <string>

namespace diffusion {

class Baseline: public Diffusion {
 protected:
  REAL *f1_, *f2_;
 public:
  Baseline(int ndim, const int *dims):
      Diffusion(ndim, dims), f1_(NULL), f2_(NULL) {
  }
  
  virtual std::string GetName() const {
    return std::string("baseline") + std::to_string(ndim_) + std::string("d");
  }

  virtual std::string GetDescription() const {
    return std::string("baseline serial implementation");
  }
  
  virtual void Setup() {
    f1_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f1_);    
    f2_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f2_);
  }
  
  virtual void InitializeInput() {
    Initialize(f1_, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, 0.0, ndim_);
  }
  
  virtual void FinalizeBenchmark() {
    assert(f1_);
    free(f1_);
    assert(f2_);
    free(f2_);
  }
    
  virtual void RunKernel(int count) {
    int i, j;
    for (i = 0; i < count; ++i) {
      int z;
      for (z = 0; z < nz_; z++) {
        int y;
        for (y = 0; y < ny_; y++) {
          int x;
          for (x = 0; x < nx_; x++) {
            int c, w[RAD], e[RAD], n[RAD], s[RAD], b[RAD], t[RAD];
            c =  x + y * nx_ + z * nx_ * ny_;

            // address on boundary
            int wb =    0    +     y     * nx_ +     z     * nx_ * ny_;
            int eb = nx_ - 1 +     y     * nx_ +     z     * nx_ * ny_;
            int sb =    x    +     0     * nx_ +     z     * nx_ * ny_;
            int nb =    x    + (ny_ - 1) * nx_ +     z     * nx_ * ny_;
            int bb =    x    +     y     * nx_ +     0     * nx_ * ny_;
            int tb =    x    +     y     * nx_ + (nz_ - 1) * nx_ * ny_;

            for (j = 0; j < RAD; j++)
            {
              w[j] = (x <= j)       ? wb : c - (j + 1);
              e[j] = (x >= nx_-1-j) ? eb : c + (j + 1);
              s[j] = (y <= j)       ? sb : c - nx_ * (j + 1);
              n[j] = (y >= ny_-1-j) ? nb : c + nx_ * (j + 1);
              b[j] = (z <= j)       ? bb : c - nx_ * ny_ * (j + 1);
              t[j] = (z >= nz_-1-j) ? tb : c + nx_ * ny_ * (j + 1);
            }

            REAL f = 0;

            f = cc_ * f1_[c];
            if (ndim_ == 2)
            {
            for (j = 0; j < RAD; j++)
              {
                f = f           +
                cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                cs_ * f1_[s[j]] + cn_ * f1_[n[j]];
              }
            }
            else if (ndim_ == 3)
            {
            for (j = 0; j < RAD; j++)
              {
                f = f           +
                cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                cs_ * f1_[s[j]] + cn_ * f1_[n[j]] +
                cb_ * f1_[b[j]] + ct_ * f1_[t[j]];
              }
            }

            f2_[c] = f;

            //if (y >=0 && y < 1 && x >=0 && x < nx_)
            //  printf("row: %04d, col: %04d, current: %f, north: %f, south: %f, left: %f, right: %f, out: %f\n", y, x, f1_[c], f1_[n], f1_[s], f1_[w], f1_[e], f);
          }
        }
      }
      REAL *t = f1_;
      f1_ = f2_;
      f2_ = t;
    }
    return;
  }

  virtual void WarmingUp() {
    RunKernel(1);
  }
    
  virtual REAL GetAccuracy(int count) {
    REAL *ref = GetCorrectAnswer(count);
    REAL err = 0.0;
    long len = nx_*ny_*nz_;
    for (long i = 0; i < len; i++) {
      REAL diff = ref[i] - f1_[i];
      err +=  diff * diff;
    }
    return (REAL)sqrt(err/len);
  }
    
  virtual void Dump() const {
    FILE *out = fopen(GetDumpPath().c_str(), "w");
    assert(out);
    long nitems = nx_ * ny_ * nz_;
    for (long i = 0; i < nitems; ++i) {
      fprintf(out, "%.7E\n", f1_[i]);
    }
    fclose(out);
  }
    
};


}

#endif /* DIFFUSION_BASELINE_H_ */
