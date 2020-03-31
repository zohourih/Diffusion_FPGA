#include "baseline.h"

namespace diffusion3d {

void Baseline::InitializeBenchmark() {
  f1_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f1_);    
  f2_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f2_);
  Initialize(f1_, nx_, ny_, nz_,
             kx_, ky_, kz_, dx_, dy_, dz_,
             kappa_, 0.0);
}

void Baseline::FinalizeBenchmark() {
  assert(f1_);
  free(f1_);
  assert(f2_);
  free(f2_);
}

void Baseline::RunKernel(int count) {
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
          int nb =    x    +     0     * nx_ +     z     * nx_ * ny_;
          int sb =    x    + (ny_ - 1) * nx_ +     z     * nx_ * ny_;
          int bb =    x    +     y     * nx_ +     0     * nx_ * ny_;
          int tb =    x    +     y     * nx_ + (nz_ - 1) * nx_ * ny_;
          
          for (j = 0; j < RAD; j++) {
            w[j] = (x <= j)       ? wb : c - (j + 1);
            e[j] = (x >= nx_-1-j) ? eb : c + (j + 1);
            n[j] = (y <= j)       ? nb : c - nx_ * (j + 1);
            s[j] = (y >= ny_-1-j) ? sb : c + nx_ * (j + 1);
            b[j] = (z <= j)       ? bb : c - nx_ * ny_ * (j + 1);
            t[j] = (z >= nz_-1-j) ? tb : c + nx_ * ny_ * (j + 1);
          }

          f2_[c] = cc_ * f1_[c];
          for (j = 0; j < RAD; j++) {
            f2_[c] = f2_[c]          +
                     cw_ * f1_[w[j]] + ce_ * f1_[e[j]] +
                     cs_ * f1_[s[j]] + cn_ * f1_[n[j]] +
                     cb_ * f1_[b[j]] + ct_ * f1_[t[j]];
          }
		//if (c < 10)printf("x: %02d, y: %02d, z: %02d, index: %04d, current: %.7E, north: %.7E,%.7E, south: %.7E.%.7E, west: %.7E,%.7E, east: %.7E,%.7E, above: %.7E,%.7E, below: %.7E,%.7E, out: %.7E\n", x, y, z, c, f1_[c], f1_[n[0]], f1_[n[1]], f1_[s[0]], f1_[s[1]], f1_[w[0]], f1_[w[1]], f1_[e[0]], f1_[e[1]], f1_[t[0]], f1_[t[1]], f1_[b[0]], f1_[b[1]], f2_[c]);
        }
      }
    }
    REAL *t = f1_;
    f1_ = f2_;
    f2_ = t;
  }
  return;
}

REAL Baseline::GetAccuracy(int count) {
  REAL *ref = GetCorrectAnswer(count);
  REAL err = 0.0;
  long len = nx_ * ny_ * nz_;
  for (long i = 0; i < len; i++) {
    REAL diff = ref[i] - f1_[i];
    err += diff * diff;
  }
  return (REAL)sqrt(err/len);
}

void Baseline::Dump() const {
  FILE *out = fopen(GetDumpPath().c_str(), "w");
  assert(out);
  long nitems = nx_ * ny_ * nz_;
  for (long i = 0; i < nitems; ++i) {
    fprintf(out, "%ld: %.7E\n", i, f1_[i]);
  }
  fclose(out);
}


}
