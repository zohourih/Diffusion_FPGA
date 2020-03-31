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
#define bdimx (64)
#define bdimy (4)

#define SHIFT3(x, y, z) x = y; y = z

#define diffusion_backward()                                            \
  do {                                                                  \
    sb[ps] = s2;                                                        \
    __syncthreads();                                                    \
    f2[p-xy] = cc * s2                                                  \
        + cw * sb[ps+sb_w] + ce * sb[ps+sb_e]                           \
        + cs * sb[ps+sb_s] + cn * sb[ps+sb_n] + cb*s1 + ct*s3;          \
  } while (0)

// Temporal blocking
// no z blocking
__global__ void diffusion_kernel_shared4(REAL *f1, REAL *f2,
                                         int nx, int ny, int nz,
                                         REAL ce, REAL cw, REAL cn, REAL cs,
                                         REAL ct, REAL cb, REAL cc) {
  extern __shared__ REAL sb[];
  const int sbx = bdimx+4;
  const int tidx = threadIdx.x % bdimx;
  const int tidy = threadIdx.x / bdimx - 1;
  int i = bdimx * blockIdx.x + tidx;
  int j = bdimy * blockIdx.y + tidy;
  j = (j < 0) ? 0 : j;      // max(j, 0)
  j = (j == ny) ? ny - 1 : j; // min(j, ny-1)

  int xy = nx * ny;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int p = i + j * nx + k *xy;
  int ps = tidx+2 + (tidy+1) * sbx;  

  if (tidy == -1) {
    int s = (j == 0)        ? 0 : -nx;

    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;            
    s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * GET(f1[p+s])
        + cn * sb[ps+sbx] + cb*t1 + ct*t3;
    p += xy;
    __syncthreads();

    for (k = 1; k < k_end; ++k) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;

      s2 = s3;
      __syncthreads();
      
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * GET(f1[p+s])
          + cn * sb[ps+sbx] + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();       
      p += xy;      
    }

    s2 = s3;
    sb[ps] = s2;
    __syncthreads();    
  } else if (tidy == bdimy) {
    int n = (j == ny-1)     ? 0 : nx;

    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;      
    s2 = s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
    p += xy;
    __syncthreads();
    for (k = 1; k < k_end; ++k) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();      
      p += xy;      
    }
    s2 = s3;
    sb[ps] = s2;
    __syncthreads();
  } else if (tidy >= 0 && tidy < bdimy) {
    int sb_s = (j == 0)    ? 0: -sbx;
    int sb_n = (j == ny-1) ? 0:  sbx; 
    int sb_w = (i == 0)    ? 0: -1;
    int sb_e = (i == nx-1) ? 0:  1;

    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s1, s2, s3;
    s2 = s3 = cc * t2
        + cw * sb[ps-1] + ce * sb[ps+1]
        + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
        + cb * t1 + ct * t3;
    p += xy;
    __syncthreads();
    for (k = 1; k < k_end; ++k) {          
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      SHIFT3(s1, s2, s3);      
      __syncthreads();
    
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
          + cb * t1 + ct * t3;
      __syncthreads();
      diffusion_backward();
      __syncthreads();
      p += xy;            
    }
    SHIFT3(s1, s2, s3);
    diffusion_backward();
  } else {
    // horizontal halo
    int xoffset = (tidx & 1) + ((tidx & 2) >> 1) * (bdimx + 2);
    int yoffset = tidx >> 2;
    yoffset = (yoffset >= (bdimy + 2)) ? bdimy+1 : yoffset;
    i = bdimx * blockIdx.x - 2 + xoffset;
    i = (i < 0) ? 0 : i;
    i = (i >= nx) ? nx - 1 : i;
    j = bdimy * blockIdx.y -1 + yoffset;
    j = (j < 0) ? 0 : j;      // max(j, 0)
    j = (j >= ny) ? ny - 1 : j; // min(j, ny-1)

    int s = (yoffset == 0)  ? 0 : -sbx;
    int n = (yoffset == bdimy+1) ? 0 : sbx;
    int w = (xoffset == 0) ? 0 : -1;
    int e = (xoffset == sbx-1) ? 0 : 1;
    
    p = i + j * nx + k * xy;
    ps = xoffset + yoffset * sbx;
    
    float t2 = GET(f1[p]);
    float t1 = (k == 0) ? t2 : GET(f1[p-xy]);
    float t3 = (k < nz-1) ? GET(f1[p+xy]) : t2;
    sb[ps] = t2;
    __syncthreads();
    float s2, s3;
    s2 = s3 = cc * t2
        + cw * sb[ps+w] + ce * sb[ps+e]
        + cs * sb[ps+s] + cn * sb[ps+n]
        + cb*t1 + ct*t3;
    __syncthreads();
    p += xy;      

    for (k = 1; k < k_end-1; ++k) {
      SHIFT3(t1, t2, t3);
      t3 = GET(f1[p+xy]);
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      s3 = cc * t2
          + cw * sb[ps+w] + ce * sb[ps+e]
          + cs * sb[ps+s] + cn * sb[ps+n]
          + cb*t1 + ct*t3;
      __syncthreads();
      sb[ps] = s2;
      __syncthreads();
      __syncthreads();      
      p += xy;      
    }

    SHIFT3(t1, t2, t3);
    t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
    sb[ps] = t2;
    s2 = s3;
    __syncthreads();
    s3 = cc * t2
        + cw * sb[ps+w] + ce * sb[ps+e]
        + cs * sb[ps+s] + cn * sb[ps+n]
        + cb*t1 + ct*t3;
    __syncthreads();
    sb[ps] = s2;
    __syncthreads();
    __syncthreads();      
    p += xy;      
    
    s2 = s3;
    sb[ps] = s2;
    __syncthreads();
  }
  return;
}

void Diffusion3DCUDAShared4::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));
  assert(count % 2 == 0);
  //dim3 block_dim(bdimx * bdimy + 32); // + 1 warp
  dim3 block_dim(bdimx * (bdimy+2) + 32);
  dim3 grid_dim(nx_ / bdimx, ny_ / bdimy, 1);

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
      for (int i = 0; i < count; i+=2) {
        diffusion_kernel_shared4<<<grid_dim, block_dim,
            (bdimx+4)*(bdimy+2)*sizeof(float)>>>
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

void Diffusion3DCUDAShared4::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared4,
                                        cudaFuncCachePreferShared));
}

}

