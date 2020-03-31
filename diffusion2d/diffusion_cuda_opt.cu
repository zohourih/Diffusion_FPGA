#include "diffusion_cuda_opt.h"
#include "common/cuda_util.h"

namespace diffusion {
namespace cuda_opt {

// loop peeling, register blocking, z-dim blocking, unrolling
__global__ void kernel3d_opt1(F1_DECL f1,
                              F2_DECL f2,
                              int nx, int ny, int nz,
                              REAL ce, REAL cw, REAL cn, REAL cs,
                              REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int xy = nx * ny;  
  int c = OFFSET3D(i, j, k, nx, ny);
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int s = (j == 0)        ? c : c - nx;
  int n = (j == ny-1)     ? c : c + nx;
  REAL t1, t2, t3;
  t2 = f1[c];
  t1 = (k == 0) ? t2 : f1[c-xy];
  t3 = f1[c+xy];
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  ++k;

  PRAGMA_UNROLL
  for (; k < k_end-1; ++k) {
    t1 = t2;
    t2 = t3;
    t3 = f1[c+xy];
    f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * t1 + ct * t3;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
  }
  t1 = t2;
  t2 = t3;
  t3 = (k < nz-1) ? f1[c+xy] : t3;
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  return;
}


// opt1 + prefetch
__global__ void kernel3d_opt2(F1_DECL f1,
                              F2_DECL f2,
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
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int s = (j == 0)        ? c : c - nx;
  int n = (j == ny-1)     ? c : c + nx;
  REAL t1, t2, t3, t4;
  t2 = f1[c];
  t1 = (k == 0) ? t2 : f1[c-xy];
  t3 = f1[c+xy];
  t4 = f1[c+xy*2];
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  ++k;
  
  PRAGMA_UNROLL
  for (; k < k_end-2; ++k) {
    t1 = t2;
    t2 = t3;
    t3 = t4;
    t4 = f1[c+xy*2];
    f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * t1 + ct * t3;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
  }

  t1 = t2;
  t2 = t3;
  t3 = t4;
  t4 = (k < nz-2) ? f1[c+xy*2] : t4;
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  ++k;
  
  t1 = t2;
  t2 = t3;
  t3 = t4;
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  return;
}

} // namespace cuda_opt

void DiffusionCUDAOpt1::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_opt::kernel3d_opt1,
                                          cudaFuncCachePreferL1));
}

void DiffusionCUDAOpt1::RunKernel(int count) {
  assert(ndim_ == 3);
      
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, grid_z_);


  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    cuda_opt::kernel3d_opt1<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDAOpt2::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_opt::kernel3d_opt2,
                                          cudaFuncCachePreferL1));
}

void DiffusionCUDAOpt2::RunKernel(int count) {
  assert(ndim_ == 3);
      
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, grid_z_);


  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    cuda_opt::kernel3d_opt2<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

}

