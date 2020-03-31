#include "diffusion_cuda.h"
#include "common/cuda_util.h"

namespace diffusion {

namespace cuda_baseline {

__global__ void kernel2d(const REAL *f1, REAL *f2,
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
  f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n];
  return;
}

__global__ void kernel3d(const REAL *f1, REAL *f2,
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
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

__global__ void kernel2d_restrict(F1_DECL f1, F2_DECL f2,
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
  f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n];
  return;
}

__global__ void kernel3d_restrict(F1_DECL f1, F2_DECL f2,
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
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

__global__ void kernel3d_zblock(const REAL *f1, REAL *f2,
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
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int s = (j == 0)        ? c : c - nx;
    int n = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

} // namespace CUDABaseline

void DiffusionCUDA::Setup() {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;
  FORCE_CHECK_CUDA(cudaMallocHost((void**)&f1_, s));
  FORCE_CHECK_CUDA(cudaMalloc((void**)&f1_d_, s));
  FORCE_CHECK_CUDA(cudaMalloc((void**)&f2_d_, s));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_baseline::kernel2d,
                                          cudaFuncCachePreferL1));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_baseline::kernel3d,
                                          cudaFuncCachePreferL1));
  FORCE_CHECK_CUDA(cudaEventCreate(&ev1_));
  FORCE_CHECK_CUDA(cudaEventCreate(&ev2_));
}

void DiffusionCUDA::FinalizeBenchmark() {
  assert(f1_);
  FORCE_CHECK_CUDA(cudaFreeHost(f1_));
  assert(f1_d_);
  FORCE_CHECK_CUDA(cudaFree(f1_d_));
  assert(f2_d_);
  FORCE_CHECK_CUDA(cudaFree(f2_d_));
}


void DiffusionCUDA::RunKernel(int count) {
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
      cuda_baseline::kernel2d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_baseline::kernel3d<<<grid_dim, block_dim>>>
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

void DiffusionCUDA::DisplayResult(int count, float time) {
  Baseline::DisplayResult(count, time);
  float time_wo_pci;
  cudaEventElapsedTime(&time_wo_pci, ev1_, ev2_);
  time_wo_pci *= 1.0e-03;
  printf("Kernel-only performance:\n");
  printf("Elapsed time : %.3f (s)\n", time_wo_pci);
  printf("FLOPS        : %.3f (GFLOPS)\n",
         GetGFLOPS(count, time_wo_pci));
  printf("Throughput   : %.3f (GB/s)\n",
         GetThroughput(count ,time_wo_pci));
}

void DiffusionCUDARestrict::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_baseline::kernel2d_restrict,
                                          cudaFuncCachePreferL1));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_baseline::kernel3d_restrict,
                                          cudaFuncCachePreferL1));
}

void DiffusionCUDARestrict::RunKernel(int count) {
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
      cuda_baseline::kernel2d_restrict<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_baseline::kernel3d_restrict<<<grid_dim, block_dim>>>
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

void DiffusionCUDAZBlock::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_baseline::kernel3d_zblock,
                                          cudaFuncCachePreferL1));
}

void DiffusionCUDAZBlock::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, grid_z_);

  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    cuda_baseline::kernel3d_zblock<<<grid_dim, block_dim>>>
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

