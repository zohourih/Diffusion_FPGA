#define USE_LDG
#include "diffusion_cuda_shfl.h"
#include "common/cuda_util.h"

#define WARP_SIZE (32)
#define WARP_MASK (WARP_SIZE-1)
#define NUM_WB_X (BLOCK_X / WARP_SIZE)

#define USE_LDG

namespace diffusion {
namespace cuda_shfl1 {

__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  const int tid = threadIdx.x;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  const int j_end = j + BLOCK_Y;

  int p = OFFSET2D(i, j, nx);

  REAL fn[NUM_WB_X], fc[NUM_WB_X], fs[NUM_WB_X];

  for (int x = 0; x < NUM_WB_X; ++x) {
    fn[x] = f1[p+x*WARP_SIZE];
    fc[x] = (blockIdx.y > 0) ? f1[p+x*WARP_SIZE-nx] : fn[x];
  }
  
  for (; j < j_end; ++j) {
    int x_offset = 0;
    // loads in batch
    PRAGMA_UNROLL
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(fs[x], fc[x], fn[x]);
      fn[x] = (j < ny - 1) ? f1[p+x_offset+nx] : fn[x];
      x_offset += WARP_SIZE;      
    }

    // compute
    x_offset = 0;
    PRAGMA_UNROLL
    for (int x = 0; x < NUM_WB_X; ++x) {
      REAL fw = __shfl_up(fc[x], 1);
      REAL fw_prev_warp = 0;
      if (x > 0) fw_prev_warp = __shfl(fc[x-1], WARP_SIZE - 1);
      if (tid == 0) {
        if (x == 0) {
          if (i != 0) {
            fw = f1[p-1];
          }
        } else {
          fw = fw_prev_warp;
        }
      }
      REAL fe = __shfl_down(fc[x], 1);
      REAL fe_next_warp = 0;
      if (x < NUM_WB_X-1) fe_next_warp = __shfl(fc[x+1], 0);
      if (tid == WARP_SIZE -1) {
        if (x == NUM_WB_X - 1) {
          if (i + x_offset != nx - 1) {
            fe = f1[p+x_offset+1];
          }
        } else {
          fe = fe_next_warp;
        }
      }
      
      f2[p+x_offset] = cc * fc[x] + cw * fw + ce * fe
          + cs * fs[x] + cn * fn[x];
      x_offset += WARP_SIZE; 
    }

    p += nx;
  }
  return;
}

__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  const int tid = threadIdx.x;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  const int xy = nx * ny;
  
  int p = OFFSET3D(i, j, k, nx, ny);

  REAL t1[NUM_WB_X][BLOCK_Y+2], t2[NUM_WB_X][BLOCK_Y+2],
      t3[NUM_WB_X][BLOCK_Y+2];

  for (int y = 0; y < BLOCK_Y; ++y) {
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = f1[p+x*WARP_SIZE+y*nx];
      t2[x][y+1] = (k > 0) ? f1[p+x*WARP_SIZE+y*nx-xy] : t3[x][y+1];
    }
  }
  {
    int y = -1;
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = blockIdx.y == 0 ? t3[x][y+1+1] :
          f1[p+x*WARP_SIZE+y*nx];
    }
  }
  {
    int y = BLOCK_Y;
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = blockIdx.y == gridDim.y - 1 ? 
          t3[x][y+1-1] : f1[p+x*WARP_SIZE+y*nx];
    }
  }

  for (; k < k_end; ++k) {
    // load
    PRAGMA_UNROLL    
    for (int y = 0; y < BLOCK_Y; ++y) {
      PRAGMA_UNROLL
      for (int x = 0; x < NUM_WB_X; ++x) {
        SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    int y = -1;
    PRAGMA_UNROLL    
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);      
      if (blockIdx.y == 0) {
        t3[x][y+1] = t3[x][y+1+1];
      } else {
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    y = BLOCK_Y;
    PRAGMA_UNROLL        
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);      
      if (blockIdx.y == gridDim.y - 1) {
        t3[x][y+1] = t3[x][y+1-1];
      } else {
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    PRAGMA_UNROLL              
    for (int y = 1; y < BLOCK_Y+1; ++y) {
      PRAGMA_UNROLL          
      for (int x = 0; x < NUM_WB_X; ++x) {
        REAL tw = __shfl_up(t2[x][y], 1);
        REAL tw_prev_warp = 0;
        if (x > 0) tw_prev_warp = __shfl(t2[x-1][y], WARP_SIZE - 1);
        if (tid == 0) {
          if (x == 0) {
            if (blockIdx.x > 0) {
              tw = LDG(f1+p-1+(y-1)*nx);
            }
          } else {
            tw = tw_prev_warp;
          }
        }
        REAL te = __shfl_down(t2[x][y], 1);
        REAL te_next_warp = 0;
        if (x < NUM_WB_X-1) te_next_warp = __shfl(t2[x+1][y], 0);
        if (tid == WARP_SIZE -1) {
          if (x == NUM_WB_X - 1) {
            if (blockIdx.x < gridDim.x - 1) {
              te = LDG(f1+p+x*WARP_SIZE+1+(y-1)*nx);
            }
          } else {
            te = te_next_warp;
          }
        }
#if 0        
        f2[p+x*WARP_SIZE+(y-1)*nx] = cc * t2[x][y] + cw * tw
            + ce * te + cs * t2[x][y-1] + cn * t2[x][y+1]
            + cb * t1[x][y] + ct * t3[x][y];
#else
        if (x == 0 || x == NUM_WB_X - 1 || x == 2) {
          f2[p+x*WARP_SIZE+(y-1)*nx] = t2[x][y];
        } else {
          f2[p+x*WARP_SIZE+(y-1)*nx] = cc * t2[x][y] + cw * tw
            + ce * te + cs * t2[x][y-1] + cn * t2[x][y+1]
            + cb * t1[x][y] + ct * t3[x][y];
        }
#endif
      }
    }
    p += xy;
  }
}

} // namespace cuda_shfl1

void DiffusionCUDASHFL1::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(WARP_SIZE, 1);
  dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_shfl1::kernel2d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_shfl1::kernel3d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_,
           ct_, cb_, cc_);
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDASHFL1::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl1::kernel2d,
                                          cudaFuncCachePreferL1));
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl1::kernel3d,
                                          cudaFuncCachePreferL1));
}



} // namespace diffusion
