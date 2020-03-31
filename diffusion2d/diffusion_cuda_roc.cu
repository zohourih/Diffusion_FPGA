#include "diffusion_cuda_roc.h"
#include "common/cuda_util.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace diffusion {
namespace cuda_roc {

__global__ void kernel2d(const REAL * __restrict__ f1,
                         REAL * __restrict__ f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int s = (j == 0)        ? c : c - nx;
  int n = (j == ny-1)     ? c : c + nx;
  f2[c] = cc * __ldg(f1+c) + cw * __ldg(f1+w) + ce * __ldg(f1+e)
      + cs * __ldg(f1+s) + cn * __ldg(f1+n);
  return;
}

__global__ void kernel3d(const REAL * __restrict__ f1,
                         REAL * __restrict__ f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int s = (j == 0)        ? c : c - nx;
    int n = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * __ldg(f1+c) + cw * __ldg(f1+w) + ce * __ldg(f1+e)
        + cs * __ldg(f1+s) + cn * __ldg(f1+n) + cb * __ldg(f1+b)
        + ct * __ldg(f1+t);
    c += xy;
  }
  return;
}

} // namespace cuda_roc

void DiffusionCUDAROC::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, 1);

  assert(nx_ % block_x_ == 0);
  assert(ny_ % block_y_ == 0);
  assert(nz_ % block_z_ == 0);

  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_roc::kernel2d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_roc::kernel3d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  FORCE_CHECK_CUDA(cudaDeviceSynchronize());
  return;
}

}

