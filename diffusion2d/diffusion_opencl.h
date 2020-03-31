#ifndef DIFFUSION_DIFFUSION_OpenCL_H_
#define DIFFUSION_DIFFUSION_OpenCL_H_

#include <CL/cl.h>
#include "diffusion.h"
#include "baseline.h"

#if defined(ENABLE_ROC)
#define F1_DECL const REAL * __restrict__ f1
#else
#define F1_DECL REAL *f1
#endif

namespace diffusion {

class DiffusionOpenCL: public Baseline {
 public:
  DiffusionOpenCL(int nd, const int *dims):
      Baseline(nd, dims), f1_d_(NULL), f2_d_(NULL)
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
      , power(0), energy(0)
#endif
  {
    assert(nd == 2); // 3D version is in the other folder
  }
  virtual std::string GetName() const {
    return std::string("opencl");
  }
  virtual std::string GetDescription() const {
    return std::string("optimized OpenCL implementation for Altera FPGAs");
  }
  virtual void Setup();
  virtual void InitializeInput();
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
  cl_kernel             constKernel;
  cl_kernel             readKernel;
  cl_kernel             writeKernel;
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

#endif /* DIFFUSION_DIFFUSION_OPENCL_H_ */
