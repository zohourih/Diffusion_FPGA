#include "diffusion3d_cuda.h"

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)


#define USE_TEX
//#define USE_ROC

namespace diffusion3d {

#ifndef USE_TEX

#if defined(USE_ROC) && __CUDA_ARCH__ >= 350
#define getf1(offset) __ldg(&f1[offset])
#else 
#define getf1(offset) (f1[offset])
#endif

__global__ void diffusion_kernel_shared2(REAL *f1, REAL *f2,
                                         int nx, int ny, int nz,
                                         REAL ce, REAL cw, REAL cn, REAL cs,
                                         REAL ct, REAL cb, REAL cc) {
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  const int xy = nx * ny;
  extern __shared__ REAL sb[];
  int c = i + j * nx + k * xy;
  const int c1 = tid_x + 1 + (tid_y + 1) * (blockDim.x+2);
  REAL t1, t2, t3;
  t2 = getf1(c);
  t1 = (k == 0) ? t2 : getf1(c-xy);
  int offset_s = (j == 0)    ? 0 : -nx;
  int offset_n = (j == ny-1) ? 0 : nx;
  int offset_w = (i == 0)    ? 0 : -1;
  int offset_e = (i == nx-1) ? 0 : 1;

  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  t3 = getf1(c+xy);
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  c += xy;
  __syncthreads();
  ++k;
  
  // manual unrolling; unroll pragma doesn't work
#pragma unroll 2
  for (; k < k_end-1; k+=1) {
    // 1st iteration
    t1 = t2;
    t2 = t3;
    // sw
    sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
    // se
    sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
    // nw
    sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
    // ne
    sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
    // Results do not match with this. Why?
    // Not happen with debug option
    // Without debug option the compiler acutally removes the
    // conditional operation, so this doesn't have any penalty.
    // compiler bug?
    t3 = (k < nz-1) ? getf1(c+xy) : t3; 
    //t3 = getf1(c+xy); //                             
    __syncthreads();
    f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
        + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
    c += xy;
    __syncthreads();
  }
  t1 = t2;
  t2 = t3;
  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  t3 = (k < nz-1) ? getf1(c+xy) : t3; 
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  return;
}

#elif defined(USE_TEX)

texture<float, cudaTextureType1D, cudaReadModeElementType> f1_tex;
texture<float, cudaTextureType1D, cudaReadModeElementType> f2_tex;

#define getf1(offset) tex1Dfetch(f1_tex, offset)
//#define getf1(offset) (f1[offset])

__global__ void diffusion_kernel_shared2_t1(REAL *f1, REAL *f2,
                                            int nx, int ny, int nz,
                                            REAL ce, REAL cw, REAL cn, REAL cs,
                                            REAL ct, REAL cb, REAL cc) {
  
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;
  extern __shared__ REAL sb[];
  int c = i + j * nx;
  const int c1 = tid_x + 1 + (tid_y + 1) * (blockDim.x+2);
  REAL t1, t2, t3;
  t1 = t2 = f1[c];
  int offset_s = (j == 0)    ? 0 : -nx;
  int offset_n = (j == ny-1) ? 0 : nx;
  int offset_w = (i == 0)    ? 0 : -1;
  int offset_e = (i == nx-1) ? 0 : 1;

  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  t3 = f1[c+xy];
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  c += xy;
  __syncthreads();
  
  #pragma unroll
  for (int k = 1; k < nz-1; k+=1) {
    t1 = t2;
    t2 = t3;
    // sw
    sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
    // se
    sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
    // nw
    sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
    // ne
    sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
    // Results do not match with this. Why?
    // Not happen with debug option
    // Without debug option the compiler acutally removes the
    // conditional operation, so this doesn't have any penalty.
    // compiler bug?
    t3 = (k < nz-1) ? f1[c+xy] : t3; 
    //t3 = f1[c+xy]; //                             
    __syncthreads();
    f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
        + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
    c += xy;
    __syncthreads();
  }
  t1 = t2;
  t2 = t3;
  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  return;
}

#undef getf1
#define getf1(offset) tex1Dfetch(f2_tex, offset)
//#define getf1(offset) (f1[offset])

__global__ void diffusion_kernel_shared2_t2(REAL *f1, REAL *f2,
                                            int nx, int ny, int nz,
                                            REAL ce, REAL cw, REAL cn, REAL cs,
                                            REAL ct, REAL cb, REAL cc) {
  
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;
  extern __shared__ REAL sb[];
  int c = i + j * nx;
  const int c1 = tid_x + 1 + (tid_y + 1) * (blockDim.x+2);
  REAL t1, t2, t3;
  t1 = t2 = f1[c];
  int offset_s = (j == 0)    ? 0 : -nx;
  int offset_n = (j == ny-1) ? 0 : nx;
  int offset_w = (i == 0)    ? 0 : -1;
  int offset_e = (i == nx-1) ? 0 : 1;

  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  t3 = f1[c+xy];
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  c += xy;
  __syncthreads();
  
  #pragma unroll
  for (int k = 1; k < nz-1; k+=1) {
    // 1st iteration
    t1 = t2;
    t2 = t3;
    // sw
    sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
    // se
    sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
    // nw
    sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
    // ne
    sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
    // Results do not match with this. Why?
    // Not happen with debug option
    // Without debug option the compiler acutally removes the
    // conditional operation, so this doesn't have any penalty.
    // compiler bug?
    t3 = (k < nz-1) ? f1[c+xy] : t3; 
    //t3 = f1[c+xy]; //                             
    __syncthreads();
    f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
        + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
    c += xy;
    __syncthreads();
  }
  t1 = t2;
  t2 = t3;
  // sw
  sb[c1-(blockDim.x+2)-1] = getf1(c+offset_s+offset_w);
  // se
  sb[c1-(blockDim.x+2)+1] = getf1(c+offset_s+offset_e);
  // nw
  sb[c1+(blockDim.x+2)-1] = getf1(c+offset_n+offset_w);
  // ne
  sb[c1+(blockDim.x+2)+1] = getf1(c+offset_n+offset_e);
  __syncthreads();
  f2[c] = cc * t2 + cw * sb[c1-1] + ce * sb[c1+1] + cs * sb[c1-(blockDim.x+2)]
      + cn * sb[c1+(blockDim.x+2)] + cb * t1 + ct * t3;
  return;
}

#endif

void Diffusion3DCUDAShared2::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

#ifdef USE_TEX  
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  CUDA_SAFE_CALL(cudaBindTexture(NULL, f1_tex, f1_d_, desc, s));
  CUDA_SAFE_CALL(cudaBindTexture(NULL, f2_tex, f2_d_, desc, s));
#endif  
  
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
#ifndef USE_TEX
        diffusion_kernel_shared2<<<grid_dim, block_dim,
            (block_x_+2)*(block_y_+2)*sizeof(float)>>>
            (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
        REAL *t = f1_d_;
        f1_d_ = f2_d_;
        f2_d_ = t;
#else
        diffusion_kernel_shared2_t1<<<grid_dim, block_dim,
            (block_x_+2)*(block_y_+2)*sizeof(float)>>>
            (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
        diffusion_kernel_shared2_t2<<<grid_dim, block_dim,
            (block_x_+2)*(block_y_+2)*sizeof(float)>>>
            (f2_d_, f1_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
        ++i;
#endif
      }
      CUDA_SAFE_CALL(cudaEventRecord(ev2_));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      flag = 1;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void Diffusion3DCUDAShared2::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
#ifdef USE_TEX  
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared2_t1,
                                        cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared2_t2,
                                        cudaFuncCachePreferShared));
#else
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared2,
                                        cudaFuncCachePreferShared));
#endif  
}

}

