#include "diffusion3d_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

namespace diffusion3d {

// prefetch
__global__ void diffusion_kernel_opt3(const REAL * __restrict__ f1,
                                      REAL * __restrict__ f2,
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
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
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
  
#pragma unroll 
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

void Diffusion3DCUDAOpt3::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_opt3,
                                        cudaFuncCachePreferL1));
}
void Diffusion3DCUDAOpt3::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, grid_z_);

  #pragma omp parallel num_threads(2) shared(flag)
  {
    if (omp_get_thread_num() == 0)
    {
      power = GetPowerGPU(&flag, 0);
    }
    else
    {
      #pragma omp barrier
      CUDA_SAFE_CALL(cudaEventRecord(ev1_));
      for (int i = 0; i < count; ++i) {
        diffusion_kernel_opt3<<<grid_dim, block_dim>>>
            (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
        REAL *t = f1_d_;
        f1_d_ = f2_d_;
        f2_d_ = t;
      }
      CUDA_SAFE_CALL(cudaEventRecord(ev2_));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      flag = 1;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

}

