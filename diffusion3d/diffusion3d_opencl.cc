//============================================================================================================
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//
// Using, modifying, and distributing this kernel file is permitted for educational, research, and non-profit
// use cases, as long as this copyright block is kept intact. Using this kernel file in any shape or form,
// including using it a template/skeleton to develop similar code, is forbidden for commercial/for-profit
// purposes, except with explicit permission from the author (Hamid Reza Zohouri).
//
// Contact point: https://www.linkedin.com/in/hamid-reza-zohouri-9aa00230/
//=============================================================================================================

#include "diffusion3d_opencl.h"
#include "common/opencl_util.h"
#include "diffusion3d_opencl_common.h"
#ifdef NO_INTERLEAVE
  #include "CL/cl_ext.h"
  #ifdef LEGACY
    #define MEM_BANK_1 CL_MEM_BANK_1_ALTERA
    #define MEM_BANK_2 CL_MEM_BANK_2_ALTERA
  #else
    #define MEM_BANK_1 CL_CHANNEL_1_INTELFPGA
    #define MEM_BANK_2 CL_CHANNEL_2_INTELFPGA
  #endif
#endif

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  #include "common/power_fpga.h"
#endif

namespace diffusion3d {

void Diffusion3DOpenCL::InitializeBenchmark() {
  size_t bufferSize = sizeof(REAL) * nx_ * ny_ * nz_ + PAD * sizeof(REAL); // PAD extra REAL-sized pad values are also taken into account
  f1_padded = (REAL*)alignedMalloc(bufferSize);

  Initialize(f1_padded + PAD, nx_, ny_, nz_,
             kx_, ky_, kz_, dx_, dy_, dz_,
             kappa_, 0.0);

  // Read source kernel binary
  size_t sourceSize;
  char *kernelSource = read_kernel("./diffusion3d_opencl.aocx", &sourceSize);

  // Display and choose device
  display_device_info(&platforms, &numPlatforms);
  select_device_type(platforms, &numPlatforms, &deviceType);
  validate_selection(platforms, &numPlatforms, ctxprop, &deviceType);

  // Create context
  context = clCreateContextFromType(ctxprop, deviceType, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("clCreateContextFromType() failed with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }

  // Get the list of OpenCL devices
  size_t deviceSize;
  CL_SAFE_CALL( clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceSize) );
  numDevices = (int) (deviceSize / sizeof(cl_device_id));
  if(numDevices < 1) {
    printf("Failed to find any device!\n");
    exit(EXIT_FAILURE);
  }

  deviceList = new cl_device_id[numDevices];
  if(!deviceList ) {
    printf("Failed to allocate memory for device list!\n");
    exit(EXIT_FAILURE);
  }
  CL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceSize, deviceList, NULL) );

  // Create command queue for the first device
  commandQueue1 = clCreateCommandQueue(context, deviceList[0], 0, &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create command queue 1 with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
  commandQueue2 = clCreateCommandQueue(context, deviceList[0], 0, &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create command queue 2 with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
  commandQueue3 = clCreateCommandQueue(context, deviceList[0], 0, &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create command queue 3 with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
#if defined(EMULATOR) && defined(LEGACY)
  commandQueue4 = clCreateCommandQueue(context, deviceList[0], 0, &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create command queue 4 with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
#endif

  // Create program from binary
  cl_program program = clCreateProgramWithBinary(context, 1, deviceList, &sourceSize, (const unsigned char**)&kernelSource, NULL, &err);
  if(err != CL_SUCCESS) {
    printf("clCreateProgramWithBinary() failed with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
  clBuildProgram_SAFE(program, 1, deviceList, NULL, NULL, NULL);

  // Create kernels
  constKernel = clCreateKernel(program, "constants", &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create constKernel with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
  readKernel = clCreateKernel(program, "read", &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create readKernel with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
  writeKernel = clCreateKernel(program, "write", &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create writeKernel with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
#if defined(EMULATOR) && defined(LEGACY)
  compKernel = clCreateKernel(program, "compute", &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create compKernel with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
#endif

  // Create device buffers
#ifdef NO_INTERLEAVE
  f1_d_ = clCreateBuffer(context, CL_MEM_READ_WRITE | MEM_BANK_1, bufferSize, NULL, &err);
#else
  f1_d_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &err);
#endif
  if (err != CL_SUCCESS) {
    printf("Failed to create f1_d_ buffer with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
#ifdef NO_INTERLEAVE
  f2_d_ = clCreateBuffer(context, CL_MEM_READ_WRITE | MEM_BANK_2, bufferSize, NULL, &err);
#else
  f2_d_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &err);
#endif
  if (err != CL_SUCCESS) {
    printf("Failed to create f2_d_ buffer with error: ");
    display_error_message(err, stderr);
    exit(EXIT_FAILURE);
  }
}

void Diffusion3DOpenCL::FinalizeBenchmark() {
  free(f1_padded);
  CL_SAFE_CALL( clReleaseMemObject(f1_d_) );
  CL_SAFE_CALL( clReleaseMemObject(f2_d_) );
  CL_SAFE_CALL( clReleaseCommandQueue(commandQueue1) );
  CL_SAFE_CALL( clReleaseCommandQueue(commandQueue2) );
  CL_SAFE_CALL( clReleaseCommandQueue(commandQueue3) );
  CL_SAFE_CALL( clReleaseContext(context) );
  CL_SAFE_CALL( clReleaseKernel(constKernel) );
  CL_SAFE_CALL( clReleaseKernel(readKernel) );
  CL_SAFE_CALL( clReleaseKernel(writeKernel) );
#if defined(EMULATOR) && defined(LEGACY)
  CL_SAFE_CALL( clReleaseCommandQueue(commandQueue4) );
  CL_SAFE_CALL( clReleaseKernel(compKernel) );
#endif
}

void Diffusion3DOpenCL::RunKernel(int count) {
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  // Power measurement parameters, only for Bittware and Nallatech's Arria 10 boards
  int flag = 0;
#endif

  // Timing parameters
  Stopwatch st;

  // Write input to device
  int pad = PAD;
  int pad_byte = PAD * sizeof(REAL);
  size_t bufferSize = sizeof(REAL) * nx_ * ny_ * nz_ + pad_byte;
  CL_SAFE_CALL( clEnqueueWriteBuffer(commandQueue2, f1_d_, CL_TRUE, 0, bufferSize, f1_padded, 0, NULL, NULL) );

  // Exit condition should be a multiple of comp_bsize_{x|y}
  int comp_bsize_x = BLOCK_X - BACK_OFF;
  int comp_bsize_y = BLOCK_Y - BACK_OFF;
  int last_col     = ((nx_ % comp_bsize_x == 0) ? nx_ + 0 : nx_ + comp_bsize_x - nx_ % comp_bsize_x) - comp_bsize_x; // exit variable is first compared, then incremented
  int last_row     = ((ny_ % comp_bsize_y == 0) ? ny_ + 0 : ny_ + comp_bsize_y - ny_ % comp_bsize_y) - comp_bsize_y; // exit variable is first compared, then incremented
  int col_blocks   = (last_col / comp_bsize_x) + 1;
  int row_blocks   = (last_row / comp_bsize_y) + 1;
  int comp_exit    = ((BLOCK_X * col_blocks * BLOCK_Y * row_blocks * (nz_ + RAD)) / ASIZE); // exit variable is first incremented, then compared
  int total_cols   = (BLOCK_X / ASIZE) * col_blocks;
  int total_rows   = BLOCK_Y * row_blocks;

  // set local and global work size
  size_t localSize[3] = {(size_t)(BLOCK_X / ASIZE), (size_t)(BLOCK_Y), (size_t)nz_};
  size_t globalSize[3] = {(size_t)total_cols, (size_t)total_rows, (size_t)nz_};

  // Set static kernel arguments
  CL_SAFE_CALL( clSetKernelArg(constKernel,  0, sizeof(int)  , (void *) &nx_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  1, sizeof(int)  , (void *) &ny_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  2, sizeof(int)  , (void *) &nz_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  3, sizeof(float), (void *) &cc_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  4, sizeof(float), (void *) &cw_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  5, sizeof(float), (void *) &ce_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  6, sizeof(float), (void *) &cs_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  7, sizeof(float), (void *) &cn_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  8, sizeof(float), (void *) &cb_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel,  9, sizeof(float), (void *) &ct_      ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel, 10, sizeof(int)  , (void *) &last_col ) );
  CL_SAFE_CALL( clSetKernelArg(constKernel, 11, sizeof(int)  , (void *) &comp_exit) );

  CL_SAFE_CALL( clSetKernelArg(readKernel ,  1, sizeof(int)  , (void *) &nx_      ) );
  CL_SAFE_CALL( clSetKernelArg(readKernel ,  2, sizeof(int)  , (void *) &ny_      ) );
  CL_SAFE_CALL( clSetKernelArg(readKernel ,  3, sizeof(int)  , (void *) &pad      ) );

  CL_SAFE_CALL( clSetKernelArg(writeKernel,  1, sizeof(int)  , (void *) &nx_      ) );
  CL_SAFE_CALL( clSetKernelArg(writeKernel,  2, sizeof(int)  , (void *) &ny_      ) );
  CL_SAFE_CALL( clSetKernelArg(writeKernel,  3, sizeof(int)  , (void *) &pad      ) );

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  #pragma omp parallel num_threads(2) shared(flag)
  {
    if (omp_get_thread_num() == 0)
    {
      #ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
        power = GetPowerFPGA(&flag);
      #else
        power = GetPowerFPGA(&flag, deviceList);
      #endif
    }
    else
    {
      #pragma omp barrier
#endif
      // Start timer
      StopwatchStart(&st);

      // Run kernel
      for (int i = 0; i < count; i += TIME) {
        int rem_iter = (count - i > TIME) ? TIME : count - i;
        CL_SAFE_CALL( clSetKernelArg(constKernel, 12, sizeof(int), &rem_iter) );

        CL_SAFE_CALL( clSetKernelArg(readKernel , 0, sizeof(cl_mem), (void *) &f1_d_) );
        CL_SAFE_CALL( clSetKernelArg(writeKernel, 0, sizeof(cl_mem), (void *) &f2_d_) );

        CL_SAFE_CALL( clEnqueueTask(commandQueue1, constKernel, 0, NULL, NULL) );
        CL_SAFE_CALL( clEnqueueNDRangeKernel(commandQueue2, readKernel , 3, NULL, globalSize, localSize, 0, 0, NULL) );
        CL_SAFE_CALL( clEnqueueNDRangeKernel(commandQueue3, writeKernel, 3, NULL, globalSize, localSize, 0, 0, NULL) );
      #if defined(EMULATOR) && defined(LEGACY)
        CL_SAFE_CALL( clEnqueueTask(commandQueue4, compKernel , 0, NULL, NULL) );
      #endif

        clFinish(commandQueue3);

        cl_mem t = f1_d_;
        f1_d_ = f2_d_;
        f2_d_ = t;
      }

      // Stop timer
      kernel_time = StopwatchStop(&st);

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
      flag = 1;
    }
  }
#endif

  // Read output from device
  CL_SAFE_CALL( clEnqueueReadBuffer(commandQueue2, f1_d_, CL_TRUE, 0, bufferSize, f1_padded, 0, NULL, NULL) );
  f1_ = f1_padded + PAD;
  return;
}

void Diffusion3DOpenCL::DisplayResult(int count, float time) {
  Baseline::DisplayResult(count, time);

  printf("Kernel-only performance:\n");
  printf("Elapsed time : %.3f (s)\n", kernel_time);
  printf("FLOPS        : %.3f (GFLOPS)\n",
         GetGFLOPS(count, kernel_time));
  printf("Throughput   : %.3f (GB/s)\n",
         GetThroughput(count, kernel_time));
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  energy = GetEnergyFPGA(power, kernel_time);
  if (power != -1) // -1 --> sensor read failure
  {
    printf("Energy usage : %.3lf (J)\n", energy);
    printf("Average power: %.3lf (Watt)\n", power);
  }
  else
  {
    printf("Failed to read power values from the sensor!\n");
  }
#endif
}

}

