#include "diffusion3d_cuda.h"

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

namespace diffusion3d {

__global__ void diffusion_kernel_shared(F1_DECL, REAL *__restrict f2,
                                        int nx, int ny, int nz,
                                        REAL ce, REAL cw, REAL cn, REAL cs,
                                        REAL ct, REAL cb, REAL cc) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;
  __shared__ REAL sb[BLOCK_X * BLOCK_Y];
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int c = i + j * nx + k * xy;
  const int c1 = tid_x + tid_y * blockDim.x;
  REAL t1, t2, t3;
  t3 = f1[c];
  t2 = (k == 0) ? t3 : f1[c-xy];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int n = (j == 0)        ? c1 : c1 - blockDim.x;
  int s = (j == ny-1)     ? c1 : c1 + blockDim.x;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == blockDim.x-1 && i != nx - 1;
  int bn = tid_y == 0 && j != 0;
  int bs = tid_y == blockDim.y-1 && j != ny - 1;

#pragma unroll
  for (; k < k_end-1; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = f1[c+xy];
    REAL t = cc * t2 + cb * t1 + ct * t3;    
    __syncthreads();
    t += cw * (bw ? f1[c-1] : sb[w]);
    t += ce * (be ? f1[c+1] : sb[e]);
    t += cs * (bs ? f1[c+nx] : sb[s]);
    t += cn * (bn ? f1[c-nx] : sb[n]);
    f2[c] = t;
    c += xy;
    __syncthreads();
  }
  t1 = t2;
  t2 = t3;
  sb[c1] = t2;    
  t3 = (k < nz-1) ? f1[c+xy] : t3;
  REAL t = cc * t2 + cb * t1 + ct * t3;    
  __syncthreads();
  t += cw * (bw ? f1[c-1] : sb[w]);
  t += ce * (be ? f1[c+1] : sb[e]);
  t += cs * (bs ? f1[c+nx] : sb[s]);
  t += cn * (bn ? f1[c-nx] : sb[n]);
  f2[c] = t;
  return;
}

void Diffusion3DCUDAShared::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared,
                                        cudaFuncCachePreferShared));
}

void Diffusion3DCUDAShared::RunKernel(int count) {
  int flag = 0;
  assert(nx_ % block_x_ == 0);
  assert(ny_ % block_y_ == 0);
  assert(nx_ / block_x_ > 0);
  assert(ny_ / block_y_ > 0);
  
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, 1);
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
        diffusion_kernel_shared<<<grid_dim, block_dim>>>
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

