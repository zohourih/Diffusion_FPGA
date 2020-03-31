#include "diffusion3d_cuda.h"
#include <assert.h>

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

namespace diffusion3d {

#if __CUDA_ARCH__ >= 350
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif

//#define GET(x) LDG(x)
#define GET(x) (x)

__global__ void diffusion_kernel_shared3(const REAL *f1,
                                         REAL * f2,
                                         int nx, int ny, int nz,
                                         REAL ce, REAL cw, REAL cn, REAL cs,
                                         REAL ct, REAL cb, REAL cc) {
  extern __shared__ REAL sb[];
  const int sbx = blockDim.x+2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int xy = nx * ny;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int p = i + j * nx + k *xy;
  int ps = threadIdx.x+1 + threadIdx.y * sbx;  
  float t1, t2, t3;


  int s = (j == 0)        ? 0 : -nx;
  int n = (j == ny-1)     ? 0 : nx;

  t3 = GET(f1[p]);
  t2 = (k == 0) ? t3 : GET(f1[p-xy]);

  if (threadIdx.y == 0) {
    int w = (blockIdx.x == 0)        ? 0 : -1;
    int e = (blockIdx.x == gridDim.x-1)     ? 0 : 1;

    // assume blockDim.y == 4
    int h = (threadIdx.x < blockDim.y) ? w : (blockDim.x - 1 + e);
    h = - threadIdx.x + h + (threadIdx.x & 3) * nx;
    int sbt = (threadIdx.x & 3) * sbx + ((threadIdx.x & 4) >> 2) * (sbx-1);
    for (; k < k_end-1; ++k) {
      t1 = t2;
      t2 = t3;
      t3 = GET(f1[p+xy]);
      sb[ps] = t2;
      if (threadIdx.x < blockDim.y*2) {
        sb[sbt] = LDG(f1[p+h]);
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
      sb[sbt] = LDG(f1[p+h]);
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

void Diffusion3DCUDAShared3::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, 1);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, 4);

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
        diffusion_kernel_shared3<<<grid_dim, block_dim,
            (block_x_+2)*(block_y_)*sizeof(float)>>>
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

void Diffusion3DCUDAShared3::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared3,
                                        cudaFuncCachePreferShared));
}

}

