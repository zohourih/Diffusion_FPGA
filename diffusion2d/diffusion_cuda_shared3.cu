#define USE_LDG
#include "diffusion_cuda_shared.h"
#include "common/cuda_util.h"

namespace diffusion {
namespace cuda_shared3 {

#define GET(x) (x)

/*
  Hoists boundary conditions out of the z-direction loop. Three
  top-level conditional blocks, one corresponding to the horizontal
  row at y == 0, another to the horizontal row at y == dimy-1, and the
  other to the rest. The first section takes care of loading halos at
  the x-direction. The y-direction halos are not cached for
  simplicity, and it is expected not to have much performance
  difference. 
 */
__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  // shared memory shape is (dimx+2) * dimy. Halo for y dir is not
  // cached. 
  extern __shared__ REAL sb[];
  const int sbx = blockDim.x+2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int xy = nx * ny;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int p = OFFSET3D(i, j, k, nx, ny);
  int ps = threadIdx.x+1 + threadIdx.y * sbx;  
  float t1, t2, t3;

  int s = (j == 0)        ? 0 : -nx;
  int n = (j == ny-1)     ? 0 : nx;

  t3 = GET(f1[p]);
  t2 = (k == 0) ? t3 : GET(f1[p-xy]);

  // Move out the boundary conditions from the loop body
  if (threadIdx.y == 0) {
    // the threads at row y == 0 also take care of loading vertical
    // halows at x == 0 and x == blockDIm.x - 1
    
    int w = (blockIdx.x == 0)        ? 0 : -1;
    int e = (blockIdx.x == gridDim.x-1)     ? 0 : 1;

    int h = (threadIdx.x < blockDim.y) ? w : (blockDim.x - 1 + e);
    h = - threadIdx.x + h + (threadIdx.x & (blockDim.y-1)) * nx;
    int sbt = (threadIdx.x & (blockDim.y-1)) * sbx;
    // the latter half takes care of the east boundary
    if (threadIdx.x >= blockDim.y) sbt += sbx-1;
    for (; k < k_end-1; ++k) {
      t1 = t2;
      t2 = t3;
      t3 = GET(f1[p+xy]);
      sb[ps] = t2;
      if (threadIdx.x < blockDim.y*2) {
        sb[sbt] = LDG(f1 + p+h);
      }
      __syncthreads();
      
      f2[p] = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs*GET(f1[p+s]) + cn*sb[ps+sbx] + cb*t1 + ct*t3;
      p += xy;
      __syncthreads();      
    }

    t1 = t2;
    t2 = t3;
    t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;
    sb[ps] = t2;
    if (threadIdx.x < blockDim.y*2) {
      sb[sbt] = LDG(f1 + p+h);
    }
    __syncthreads();

    f2[p] = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * GET(f1[p+s]) + cn * sb[ps+sbx] + cb * t1 + ct * t3;
  } else if (threadIdx.y == blockDim.y - 1) {
    for (; k < k_end-1; ++k) {
      t1 = t2;
      t2 = t3;
      //t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;
      t3 = GET(f1[p+xy]);
      sb[ps] = t2;
      __syncthreads();

      f2[p] = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]
          + cn * GET(f1[p+n])
          + cb * t1 + ct * t3;
      p += xy;
      __syncthreads();      
    }
    t1 = t2;
    t2 = t3;
    t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;
    sb[ps] = t2;
    __syncthreads();
    
    f2[p] = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx]+ cn * GET(f1[p+n]) + cb * t1 + ct * t3;
  } else {
    for (; k < k_end-1; ++k) {
      t1 = t2;
      t2 = t3;
      t3 = GET(f1[p+xy]);
      sb[ps] = t2;
      __syncthreads();
    
      f2[p] = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
          + cb * t1 + ct * t3;
      p += xy;
      __syncthreads();      
    }
    t1 = t2;
    t2 = t3;
    t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;
    sb[ps] = t2;
    __syncthreads();
    
    f2[p] = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx]+ cn * sb[ps+sbx] + cb * t1 + ct * t3;
  }
  return;
}

} // namespace cuda_shared3

void DiffusionCUDAShared3::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    cuda_shared3::kernel3d<<<grid_dim, block_dim,
        (block_x_+2)*(block_y_)*sizeof(float)>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDAShared3::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared3::kernel3d,
                                          cudaFuncCachePreferShared));
}

} // namespace diffusion

