#include "diffusion_cuda_copy.h"
#include "common/cuda_util.h"

namespace diffusion {

namespace cuda_copy {

__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int xy = nx * ny;    
  int c = i + j * nx + k * xy;
  for (; k < k_end; ++k) {
    f2[c] = cc * f1[c];
    c += xy;
  }
  return;
}

} // namespace cuda_copy

void DiffusionCUDACopy::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      assert(0);
    } else if (ndim_ == 3) {
      cuda_copy::kernel3d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_,
           nx_, ny_, nz_, ce_, cw_, cn_, cs_,
           ct_, cb_, cc_);
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDACopy::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_copy::kernel3d,
                                          cudaFuncCachePreferL1));
}

}

