#include "diffusion3d_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

namespace diffusion3d {

__global__ void diffusion_kernel(F1_DECL,
                                 REAL * __restrict__ f2,
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
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
#if 1
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
#else
    // simulating the ordering of shared memory version
    REAL v = cc * f1[c];
    v += cw * f1[w];
    v += ce * f1[e];
    v += cs * f1[s];
    v += cn * f1[n];
    v += cb * f1[b] + ct * f1[t];
    f2[c] = v;
#endif    
    c += xy;
  }
  return;
}

__global__ void diffusion_kernel_zblock(F1_DECL,
                                        REAL * __restrict__ f2,
                                        int nx, int ny, int nz,
                                        REAL ce, REAL cw, REAL cn, REAL cs,
                                        REAL ct, REAL cb, REAL cc) {
  int xy = nx * ny;  
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  int c = i + j * nx + k * xy;
#pragma unroll
  for (; k < k_end; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

// minimal optimization
__global__ void diffusion_kernel_opt0(const REAL * __restrict__ f1,
                                      REAL * __restrict__ f2,
                                      int nx, int ny, int nz,
                                      REAL ce, REAL cw, REAL cn, REAL cs,
                                      REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
  for (int k = 0; k < nz; ++k) {
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
  }
  return;
}

void Diffusion3DCUDAOpt0::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_opt0,
                                        cudaFuncCachePreferL1));
}
void Diffusion3DCUDAOpt0::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, 1);

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
        diffusion_kernel_opt0<<<grid_dim, block_dim>>>
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

// register blocking
// loop peeling
__global__ void diffusion_kernel_opt1(F1_DECL,
                                      REAL * __restrict__ f2,
                                      int nx, int ny, int nz,
                                      REAL ce, REAL cw, REAL cn, REAL cs,
                                      REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
  REAL t1, t2, t3;
  t1 = t2 = f1[c];
  t3 = f1[c+xy];
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
#pragma unroll 8
  for (int k = 1; k < nz-1; ++k) {
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
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  return;
}

// opt1 + z-dim blocking
__global__ void diffusion_kernel_opt2(F1_DECL,
                                      REAL * f2,
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
  
#pragma unroll 
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

#if 0
__global__ void diffusion_kernel_xy(REAL *f1, REAL *f2,
                                    int nx, int ny, int nz,
                                    REAL ce, REAL cw, REAL cn, REAL cs,
                                    REAL ct, REAL cb, REAL cc) {
  int bdimx = blockDim.x;
  int i = bdimx * blockIdx.x * 2 + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += bdimx;
    w = c - 1;
    e = (i+bdimx == nx-1)     ? c : c + 1;
    n += bdimx;
    s += bdimx;
    b += bdimx;
    t += bdimx;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy-bdimx;
  }
  return;
}
#else
__global__ void diffusion_kernel_xy(REAL *f1, REAL *f2,
                                    int nx, int ny, int nz,
                                    REAL ce, REAL cw, REAL cn, REAL cs,
                                    REAL ct, REAL cb, REAL cc) {
  int bdimx = blockDim.x;  
  int i = blockDim.x * blockIdx.x *2 + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
  REAL t1, t2, t3;
  REAL t1_2, t2_2, t3_2;  
  t1 = t2 = f1[c];
  t3 = f1[c+xy];
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  //
  c += bdimx;
  int w2 = c - 1;
  int e2 = ((i+bdimx) == nx-1)     ? c : c + 1;
  n += bdimx;
  s += bdimx;
  t1_2 = t2_2 = f1[c];
  t3_2 = f1[c+xy];
  f2[c] = cc * t2_2 + cw * f1[w2] + ce * f1[e2] + cs * f1[s]
      + cn * f1[n] + cb * t1_2 + ct * t3_2;
  //
  c += xy-bdimx;
  w += xy; w2 += xy;
  e += xy; e2 += xy;
  n += xy-bdimx;
  s += xy-bdimx;
#pragma unroll 8
  for (int k = 1; k < nz-1; ++k) {
    t1 = t2;
    t2 = t3;
    t3 = f1[c+xy];
    f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * t1 + ct * t3;
    c += bdimx;
    w += xy;
    e += xy; 
    n += bdimx;
    s += bdimx;
    t1_2 = t2_2;
    t2_2 = t3_2;
    t3_2 = f1[c+xy];
    f2[c] = cc * t2_2 + cw * f1[w2] + ce * f1[e2] + cs * f1[s]
        + cn * f1[n] + cb * t1_2 + ct * t3_2;
    c += xy-bdimx;
    w2 += xy;
    e2 += xy;
    n += xy-bdimx;
    s += xy-bdimx;
  }
  t1 = t2;
  t2 = t3;
  f2[c] = cc * t2 + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * t1 + ct * t3;
  c += bdimx;
  n += bdimx;
  s += bdimx;
  t1_2 = t2_2;
  t2_2 = t3_2;
  f2[c] = cc * t2_2 + cw * f1[w2] + ce * f1[e2] + cs * f1[s]
      + cn * f1[n] + cb * t1_2 + ct * t3_2;
  return;
}
#endif

void Diffusion3DCUDA::InitializeBenchmark() {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&f1_, s));
  Initialize(f1_, nx_, ny_, nz_,
             kx_, ky_, kz_, dx_, dy_, dz_,
             kappa_, 0.0);
  CUDA_SAFE_CALL(cudaMalloc((void**)&f1_d_, s));
  CUDA_SAFE_CALL(cudaMalloc((void**)&f2_d_, s));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel,
                                        cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1_));
  CUDA_SAFE_CALL(cudaEventCreate(&ev2_));
}

void Diffusion3DCUDA::FinalizeBenchmark() {
  assert(f1_);
  CUDA_SAFE_CALL(cudaFreeHost(f1_));
  assert(f1_d_);
  CUDA_SAFE_CALL(cudaFree(f1_d_));
  assert(f2_d_);
  CUDA_SAFE_CALL(cudaFree(f2_d_));
}


void Diffusion3DCUDA::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, 1);

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
        diffusion_kernel<<<grid_dim, block_dim>>>
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

void Diffusion3DCUDA::DisplayResult(int count, float time) {
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
  energy = GetEnergyGPU(power, time_wo_pci);
  if (power != -1) // -1 --> sensor read failure
  {
    printf("Energy usage : %.3lf (J)\n", energy);
    printf("Average power: %.3lf (Watt)\n", power);
  }
  else
  {
    printf("Failed to read power values from the sensor!\n");
  }
}

void Diffusion3DCUDAZBlock::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_zblock,
                                        cudaFuncCachePreferL1));
}
void Diffusion3DCUDAZBlock::RunKernel(int count) {
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
        diffusion_kernel_zblock<<<grid_dim, block_dim>>>
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

void Diffusion3DCUDAOpt1::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_opt1,
                                        cudaFuncCachePreferL1));
}
void Diffusion3DCUDAOpt1::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_, block_z_);
  dim3 grid_dim(nx_ / block_x_, ny_ / block_y_, 1);

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
        diffusion_kernel_opt1<<<grid_dim, block_dim>>>
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

void Diffusion3DCUDAOpt2::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_opt2,
                                        cudaFuncCachePreferL1));
}
void Diffusion3DCUDAOpt2::RunKernel(int count) {
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
        diffusion_kernel_opt2<<<grid_dim, block_dim>>>
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

void Diffusion3DCUDAXY::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_xy,
                                        cudaFuncCachePreferL1));
}

void Diffusion3DCUDAXY::RunKernel(int count) {
  int flag = 0;
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x_, block_y_);
  dim3 grid_dim(nx_ / (block_x_ * 2), ny_ / block_y_, 1);

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
        diffusion_kernel_xy<<<grid_dim, block_dim>>>
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

