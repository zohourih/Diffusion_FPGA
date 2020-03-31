#define USE_LDG
#include "diffusion_cuda_shfl.h"
#include "common/cuda_util.h"

#define WARP_SIZE (32)
#define WARP_MASK (WARP_SIZE-1)
#define NUM_WB_X (BLOCK_X / WARP_SIZE)
#if NUM_WB_X == 4
//#define REALV real4
#define REALV float4
#endif

#define USE_LDG

namespace diffusion {

union real4 {
  REAL v[4];
  struct {
    REAL first;
    REAL middle[2];
    REAL last;
  } n;
};

namespace cuda_shfl3 {


__global__ void kernel3d(const REALV * __restrict__ f1,
                         REALV * __restrict__ f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  const int tid = threadIdx.x;
  int i = WARP_SIZE * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  const int xy = nx * ny;
  
  int p = OFFSET3D(i, j, k, nx, ny);

  REALV t1[BLOCK_Y+2], t2[BLOCK_Y+2],
      t3[BLOCK_Y+2];

  for (int y = 0; y < BLOCK_Y; ++y) {
    t3[y+1] = f1[p+y*nx];
    t2[y+1] = (k > 0) ? f1[p+y*nx-xy] : t3[y+1];
  }
  {
    int y = -1;
    t3[y+1] = blockIdx.y == 0 ? t3[y+1+1] : f1[p+y*nx];
  }
  {
    int y = BLOCK_Y;
    t3[y+1] = blockIdx.y == gridDim.y - 1 ? t3[y+1-1] : f1[p+y*nx];
  }

  for (; k < k_end; ++k) {
    // load
    PRAGMA_UNROLL
    for (int y = 0; y < BLOCK_Y; ++y) {
      SHIFT3(t1[y+1], t2[y+1], t3[y+1]);
      t3[y+1] = (k < nz - 1) ? f1[p+y*nx+xy] : t3[y+1];
    }

    int y = -1;
    SHIFT3(t1[y+1], t2[y+1], t3[y+1]);      
    if (blockIdx.y == 0) {
      t3[y+1] = t3[y+1+1];
    } else {
      t3[y+1] = (k < nz - 1) ? f1[p+y*nx+xy] : t3[y+1];
    }

    y = BLOCK_Y;
    SHIFT3(t1[y+1], t2[y+1], t3[y+1]);      
    if (blockIdx.y == gridDim.y - 1) {
      t3[y+1] = t3[y+1-1];
    } else {
      t3[y+1] = (k < nz - 1) ? f1[p+y*nx+xy] : t3[y+1];
    }

    PRAGMA_UNROLL              
    for (int y = 1; y < BLOCK_Y+1; ++y) {
      REALV f2v = {0, 0, 0, 0};
      {

        REAL tw = __shfl_up(t2[y].w, 1);

        if (tid == 0) {
          if (blockIdx.x > 0) {
            tw = LDG(reinterpret_cast<const REAL*>(&(f1[p-1+(y-1)*nx].w)));
          } else {
            tw = t2[y].x;
          }
        }

        REAL te = t2[y].y;
        f2v.x = cc * t2[y].x + cw * tw
            + ce * te + cs * t2[y-1].x + cn * t2[y+1].x
            + cb * t1[y].x + ct * t3[y].x;
      }
#if 0
      for (int x = 1; x < NUM_WB_X - 1; ++x) {
        REAL tw = t2[y].v[x-1];        
        REAL te = t2[y].v[x+1];
        f2v.v[x] = cc * t2[y].v[x] + cw * tw
            + ce * te + cs * t2[y-1].v[x] + cn * t2[y+1].v[x]
            + cb * t1[y].v[x] + ct * t3[y].v[x];
      }
#else
      {
        REAL tw = t2[y].x;
        REAL te = t2[y].z;
        f2v.y = cc * t2[y].y + cw * tw
            + ce * te + cs * t2[y-1].y + cn * t2[y+1].y
            + cb * t1[y].y + ct * t3[y].y;
      }
      {
        REAL tw = t2[y].y;
        REAL te = t2[y].w;
        f2v.z = cc * t2[y].z + cw * tw
            + ce * te + cs * t2[y-1].z + cn * t2[y+1].z
            + cb * t1[y].z + ct * t3[y].z;
      }
#endif
      {
        int x = NUM_WB_X - 1;
        REAL tw = t2[y].z;
        REAL te = __shfl_down(t2[y].x, 1);

        if (tid == WARP_SIZE -1) {
          if (blockIdx.x < gridDim.x - 1) {
            te = LDG(reinterpret_cast<const REAL*>(&(f1[p+1+(y-1)*nx].x)));
          } else {
            te = t2[y].w;
          }
        }

        f2v.w = cc * t2[y].w + cw * tw
            + ce * te + cs * t2[y-1].w + cn * t2[y+1].w
            + cb * t1[y].w + ct * t3[y].w;
      }
      f2[p+(y-1)*nx] = f2v;
    }
    p += xy;
  }
}

} // namespace cuda_shfl3

void DiffusionCUDASHFL3::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(WARP_SIZE, 1);
  dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      assert(0);
      //cuda_shfl3::kernel2d<<<grid_dim, block_dim>>>
      //(f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      cuda_shfl3::kernel3d<<<grid_dim, block_dim>>>
          (reinterpret_cast<REALV*>(f1_d_),
           reinterpret_cast<REALV*>(f2_d_),
           nx_/NUM_WB_X, ny_, nz_, ce_, cw_, cn_, cs_,
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

void DiffusionCUDASHFL3::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl3::kernel3d,
                                          cudaFuncCachePreferL1));
}



} // namespace diffusion
