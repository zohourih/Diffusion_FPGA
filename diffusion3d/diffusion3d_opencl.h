#ifndef DIFFUSION3D_DIFFUSION3D_OpenCL_H_
#define DIFFUSION3D_DIFFUSION3D_OpenCL_H_

#include <CL/cl.h>
#include "diffusion3d.h"
#include "baseline.h"

#if defined(ENABLE_ROC)
#define F1_DECL const REAL * __restrict__ f1
#else
#define F1_DECL REAL *f1
#endif

namespace diffusion3d {

class Diffusion3DOpenCL: public Baseline {
 public:
  Diffusion3DOpenCL(int nx, int ny, int nz):
      Baseline(nx, ny, nz), f1_d_(NULL), f2_d_(NULL)
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
      , power(0), energy(0)
#endif
  {
  }
  virtual std::string GetName() const {
    return std::string("opencl");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
  virtual void FinalizeBenchmark();
  virtual void DisplayResult(int count, float time);

 protected:
  cl_mem f1_d_, f2_d_;
  REAL* f1_padded;
  float kernel_time;

  // opencl stuff
  cl_context            context;
  cl_command_queue      commandQueue1;
  cl_command_queue      commandQueue2;
  cl_command_queue      commandQueue3;
  cl_program            program;
  cl_platform_id        *platforms;
  cl_context_properties ctxprop[3];
  cl_device_type        deviceType;
  cl_device_id          *deviceList;
  cl_kernel             readKernel;
  cl_kernel             writeKernel;
  cl_kernel             constKernel;
#ifdef EMULATOR
  cl_command_queue      commandQueue4;
  cl_kernel             compKernel;
#endif
  cl_uint               numPlatforms;
  cl_int                numDevices;
  cl_int                err;

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  // Power measurement parameters, only for Bittware and Nallatech's Arria 10 boards
  double                power, energy;
#endif

};

}

#endif /* DIFFUSION3D_DIFFUSION3D_OPENCL_H_ */
