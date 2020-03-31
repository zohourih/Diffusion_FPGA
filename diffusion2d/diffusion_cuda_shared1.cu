#include "diffusion_cuda_shared.h"
#include "common/cuda_util.h"

namespace diffusion {
namespace cuda_shared1 {

__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  extern __shared__ REAL sb[];
  // offset in the global memory
  int offset_g = OFFSET2D(i, j, nx);
  // offset in the shared memory
  const int offset_s = OFFSET2D(tid_x, tid_y, blockDim.x);
  REAL fc = f1[offset_g];
  sb[offset_s] = fc;
  __syncthreads();
  // load each neighbor
  REAL fw;
  if (i == 0) {
    // global boundary
    fw = fc;
  } else if (tid_x == 0) {
    // block boundary. Not loaded to the shared memory of this thread
    // block, so needs to access global memory
    fw = f1[offset_g-1];
  } else {
    // already loaded by neighbor thread
    fw = sb[offset_s-1];
  }
  REAL fe;
  if (i == nx - 1) {
    // global boundary
    fe = fc;
  } else if (tid_x == blockDim.x - 1) {
    // block boundary
    fe = f1[offset_g+1];
  } else {
    fe = sb[offset_s+1];
  }
  REAL fs;
  if (j == 0) {
    // global boundary
    fs = fc;
  } else if (tid_y == 0) {
    // block boundary
    fs = f1[offset_g-nx];
  } else {
    fs = sb[offset_s-blockDim.x];
  }
  REAL fn;
  if (j == ny - 1) {
    // global boundary
    fn = fc;
  } else if (tid_y == blockDim.y - 1) {
    // block boundary
    fn = f1[offset_g+nx];
  } else {
    fn = sb[offset_s+blockDim.x];
  }
  REAL t = cc * fc + cw * fw + ce * fe + cs * fs 
      + cn * fn;
  f2[offset_g] = t;
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
  const int xy = nx * ny;
  extern __shared__ REAL sb[];
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  // offset in the global memory  
  int c = OFFSET3D(i, j, k, nx, ny);
  // offset in the shared memory
  const int c1 = OFFSET2D(tid_x, tid_y, blockDim.x);
  REAL t1, t2, t3;
  t3 = f1[c];
  t2 = (k == 0) ? t3 : f1[c-xy];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int s = (j == 0)        ? c1 : c1 - blockDim.x;
  int n = (j == ny-1)     ? c1 : c1 + blockDim.x;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == blockDim.x-1 && i != nx - 1;
  int bs = tid_y == 0 && j != 0;
  int bn = tid_y == blockDim.y-1 && j != ny - 1;

  PRAGMA_UNROLL  
  for (; k < k_end-1; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = f1[c+xy];
    __syncthreads();
    REAL fw = bw ? f1[c-1] : sb[w];
    REAL fe = be ? f1[c+1] : sb[e];
    REAL fs = bs ? f1[c-nx] : sb[s];
    REAL fn = bn ? f1[c+nx] : sb[n];
    REAL t = cc * t2 + cw * fw + ce * fe + cs * fs + cn * fn
        + cb * t1 + ct * t3;
    f2[c] = t;
    c += xy;
    __syncthreads();
  }
  t1 = t2;
  t2 = t3;
  sb[c1] = t2;    
  t3 = (k < nz-1) ? f1[c+xy] : t3;
  __syncthreads();
  REAL fw = bw ? f1[c-1] : sb[w];
  REAL fe = be ? f1[c+1] : sb[e];
  REAL fs = bs ? f1[c-nx] : sb[s];
  REAL fn = bn ? f1[c+nx] : sb[n];
  REAL t = cc * t2 + cw * fw + ce * fe + cs * fs + cn * fn
      + cb * t1 + ct * t3;
  f2[c] = t;
  return;
}

} // namespace cuda_shared1

void DiffusionCUDAShared1::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared1::kernel2d,
                                          cudaFuncCachePreferShared));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared1::kernel3d,
                                          cudaFuncCachePreferShared));
}

void DiffusionCUDAShared1::RunKernel(int count) {
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
      cuda_shared1::kernel2d<<<grid_dim, block_dim,
          block_x_ * block_y_ * sizeof(REAL)>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_shared1::kernel3d<<<grid_dim, block_dim,
          block_x_ * block_y_ * sizeof(REAL)>>>
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

