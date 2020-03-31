#include "diffusion_cuda_shared.h"
#include "common/cuda_util.h"

namespace diffusion {
namespace cuda_shared2 {

__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  // x dimension of the shared memory
  const int sbx = blockDim.x+2;
  extern __shared__ REAL sb[];
  // offset in the global memory
  int c = OFFSET2D(i, j, nx);
  // offset in the shared memory
  const int c1 = OFFSET2D(tid_x + 1, tid_y + 1, sbx);

  // load the element this thread is responsible for
  REAL fc = f1[c];
  sb[c1] = fc;

  // boundary threads load boundary elements to shared 
  int w = (i == 0)        ? 0 : - 1;
  int e = (i == nx-1)     ? 0 : 1;
  int s = (j == 0)        ? 0 : -nx;
  int n = (j == ny-1)     ? 0 : nx;
  if (tid_x == 0) {
    sb[c1-1]= f1[c+w];
  } else if (tid_x == blockDim.x-1) {
    sb[c1+1]= f1[c+e];
  }
  if (tid_y == 0) {
    sb[c1-sbx]= f1[c+s];
  } else if (tid_y == blockDim.y-1) {
    sb[c1+sbx]= f1[c+n];
  }
  
  __syncthreads();
  
  f2[c] = cc * fc
      + cw * sb[c1-1] + ce * sb[c1+1]
      + cs * sb[c1-sbx] + cn * sb[c1+sbx];
  return;
}


__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int sbx = blockDim.x+2;
  const int xy = nx * ny;
  extern __shared__ REAL sb[];
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int c = i + j * nx + k * xy;
  const int c1 = tid_x + 1 + (tid_y + 1) * sbx;
  REAL t1, t2, t3;
  t3 = f1[c];
  t2 = (k == 0) ? t3 : f1[c-xy];
  int w = (i == 0)        ? 0 : - 1;
  int e = (i == nx-1)     ? 0 : 1;
  int s = (j == 0)        ? 0 : -nx;
  int n = (j == ny-1)     ? 0 : nx;

  PRAGMA_UNROLL  
  for (; k < k_end-1; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = f1[c+xy];
    if (tid_y == 0) {
      sb[c1-sbx]= f1[c+s];
    } else if (tid_y == blockDim.y-1) {
      sb[c1+sbx]= f1[c+n];
    }
    // west
    if (tid_x == 0) {
      sb[c1-1]= f1[c+w];
    } else if (tid_x == blockDim.x-1) {
      sb[c1+1]= f1[c+e];
    }
    __syncthreads();
   f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1]
      + cs * sb[c1-sbx] + cn * sb[c1+sbx]
      + cb * t1 + ct * t3;
    c += xy;
    __syncthreads();
  }

  t1 = t2;
  t2 = t3;
  sb[c1] = t2;    
  t3 = (k < nz-1) ? f1[c+xy] : t3;
  if (tid_y == 0) {
    sb[c1-sbx]= f1[c+s];
  } else if (tid_y == blockDim.y-1) {
    sb[c1+sbx]= f1[c+n];
  }
  // west
  if (tid_x == 0) {
    sb[c1-1]= f1[c+w];
  } else if (tid_x == blockDim.x-1) {
    sb[c1+1]= f1[c+e];
  }
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1]
      + cs * sb[c1-sbx] + cn * sb[c1+sbx]
      + cb * t1 + ct * t3;
  c += xy;
  return;
}

} // namespace cuda_shared2

void DiffusionCUDAShared2::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared2::kernel2d,
                                          cudaFuncCachePreferShared));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared2::kernel3d,
                                          cudaFuncCachePreferShared));
}

void DiffusionCUDAShared2::RunKernel(int count) {
  assert(nx_ % block_x_ == 0);
  assert(ny_ % block_y_ == 0);
  assert(nx_ / block_x_ > 0);
  assert(ny_ / block_y_ > 0);
  
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, 1);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_);
  if (ndim_ == 3 ) grid_dim.z = grid_z_;  

  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_shared2::kernel2d<<<grid_dim, block_dim,
          (block_x_+2)*(block_y_+2)*sizeof(REAL)>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_shared2::kernel3d<<<grid_dim, block_dim,
          (block_x_+2)*(block_y_+2)*sizeof(REAL)>>>
          (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

} // namespace diffusion

