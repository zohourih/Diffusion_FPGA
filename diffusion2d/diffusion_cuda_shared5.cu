#define USE_LDG
#include "diffusion_cuda_shared.h"
#include "common/cuda_util.h"

namespace diffusion {
namespace cuda_shared5 {

#define GET(x) (x)

#define diffusion_backward()                                            \
  do {                                                                  \
    sb[ps] = s2;                                                        \
    __syncthreads();                                                    \
    f2[p-xy] = cc * s2                                                  \
        + cw * sb[ps+sb_w] + ce * sb[ps+sb_e]                           \
        + cs * sb[ps+sb_s] + cn * sb[ps+sb_n] + cb*s1 + ct*s3;          \
  } while (0)

// Temporal blocking
// z blocking
// the diagonal points are loaded by the vertical warp
__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  extern __shared__ REAL sb[];
  const int sbx = BLOCK_X+4;
  const int tidx = threadIdx.x % BLOCK_X;
  const int tidy = threadIdx.x / BLOCK_X - 1;
  int i = BLOCK_X * blockIdx.x + tidx;
  int j = BLOCK_Y * blockIdx.y + tidy;
  j = (j < 0) ? 0 : j;      // max(j, 0)
  j = (j == ny) ? ny - 1 : j; // min(j, ny-1)

  int xy = nx * ny;
  const int block_z = nz / gridDim.z;
  int k = (blockIdx.z == 0) ? 0:
      block_z * blockIdx.z - 1;
  const int k_end = (blockIdx.z == gridDim.z-1) ? nz:
      block_z * (blockIdx.z + 1) + 1;
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
    ++k;

    if (k != 1) {
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
      p += xy;
      ++k;
    }

    for (; k < k_end; ++k) {
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

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();
    }
  } else if (tidy == BLOCK_Y) {
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
    ++k;

    if (k != 1) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      s2 = s3;
      __syncthreads();
      
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx] + cn * GET(f1[p+n]) + cb*t1 + ct*t3;
      p += xy;
      __syncthreads();
      ++k;
    }
    
    for (; k < k_end; ++k) {
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

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();      
    }
  } else if (tidy >= 0 && tidy < BLOCK_Y) {
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
    ++k;

    if (k != 1) {
      SHIFT3(t1, t2, t3);
      t3 = (k < nz-1) ? GET(f1[p+xy]) : t3;      
      sb[ps] = t2;
      SHIFT3(s1, s2, s3);      
      __syncthreads();
    
      s3 = cc * t2
          + cw * sb[ps-1] + ce * sb[ps+1]
          + cs * sb[ps-sbx]+ cn * sb[ps+sbx]
          + cb * t1 + ct * t3;
      p += xy;      
      __syncthreads();
      ++k;
    }
    
    for (; k < k_end; ++k) {          
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
    
    if (k == nz) {
      SHIFT3(s1, s2, s3);
      diffusion_backward();
    }
  } else {
    // horizontal halo
    int xoffset = (tidx & 1) + ((tidx & 2) >> 1) * (BLOCK_X + 2);
    int yoffset = tidx >> 2;
    yoffset = (yoffset >= (BLOCK_Y + 2)) ? BLOCK_Y+1 : yoffset;
    i = BLOCK_X * blockIdx.x - 2 + xoffset;
    i = (i < 0) ? 0 : i;
    i = (i >= nx) ? nx - 1 : i;
    j = BLOCK_Y * blockIdx.y -1 + yoffset;
    j = (j < 0) ? 0 : j;      // max(j, 0)
    j = (j >= ny) ? ny - 1 : j; // min(j, ny-1)

    int s = (yoffset == 0)  ? 0 : -sbx;
    int n = (yoffset == BLOCK_Y+1) ? 0 : sbx;
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
    ++k;

    if (k != 1) {
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
      p += xy;
      ++k;
    }

    for (; k < k_end-1; ++k) {
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
    ++k;

    if (k == nz) {
      s2 = s3;
      sb[ps] = s2;
      __syncthreads();
    }
  }
  return;
}

} // namespace cuda_shared5

void DiffusionCUDAShared5::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));
  assert(count % 2 == 0);
  //dim3 block_dim(BLOCK_X * BLOCK_Y + 32); // + 1 warp
  dim3 block_dim(BLOCK_X * (BLOCK_Y+2) + 32);
  dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y, grid_z_);
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; i+=2) {
    cuda_shared5::kernel3d<<<grid_dim, block_dim,
        (BLOCK_X+4)*(BLOCK_Y+2)*sizeof(float)>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDAShared5::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shared5::kernel3d,
                                          cudaFuncCachePreferShared));
}

}

