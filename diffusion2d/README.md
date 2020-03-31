# README for OpenCL version
Refer to [original README](README) for other versions.

# Files
**diffusion_opencl.cc:** OpenCL host code  
**diffusion_opencl.h:** Header for host code  
**diffusion_opencl_common.h:** Common header for host and kernel code  
**diffusion_opencl_base.cl:** Base OpenCL kernel without boundary conditions  
**diffusion_opencl_neighbor_generator.sh:** Bash script to generate boundary conditions based on stencil radius  
**diffusion_opencl_replace.sh:** Script to insert boundary conditions into base OpenCL kernel and create final diffusion_opencl.cl  
**fpga_benchmark.sh:** Template benchmark script in bash

# Make
Command:

`make altera *make_options*`

&nbsp;

| Make options                  | Description | Default |
| ---                           | ---         | ---     |
| **EMULATOR=1** | Compile for emulation. | Disabled |
| **BOARD=VALUE** | Override board name. If BSP supports only one board/hardware, that board will be automatically chosen without needing to supply this option. | Disabled |
| **BSIZE=VALUE** or **BLOCK_X=VALUE** | Override block size in *x* dimension. | 4096 |
| **TIME=VALUE** | Override degree of temporal parallelism. | 4 or 12 |
| **ASIZE=VALUE** | Override vector size for global memory accesses. | 8 or 16 |
| **RAD=VALUE** | Override stencil radius. | 1 |
| **PAD=VALUE** | Override padding size. | (TIME * RAD) % 16 |
| **CSIZE=VALUE** | Override channel depth. | 16 |
| **NO_INTERLEAVE=1** | Disable interleaving of global memory arrays between external memory banks. | Disabled |


# Run

Command:

`./diffusion_altera.exe *run_options*`

&nbsp;

| Run options | Description | Default |
| ---         | ---         | ---     |
| **--nd VALUE** | Number of stencil dimensions. MUST be set to 2. | N/A |
| **--size VALUE** | Number of cells in all dimensions. | 256 |
| **--count VALUE** | Number of time-steps. | 100 |
| **--nx VALUE** | Number of cells in *x* dimension. Do not use with **--size**. | N/A |
| **--ny VALUE** | Number of cells in *y* dimension. Do not use with **--size**. | N/A |
| **--alt** | Alternative CPU verification function which is much slower than the original verification but should return 0.000000e+00 rather than a very small number as "Accuracy" if correct. | Disabled |
| **--help** | Print benchmark help and exit. | Disabled |


# Benchmark scripts

Bash-based FPGA benchmark script is provided in the repository for ease of benchmarking. However, they might or might not work on your environment out of the box and modifications will very likely be required to get them to work correctly. Specifically, the variables that are set at the top of the benchmark scripts pretty much always need to be changed.

# Important note regarding emulation

Due to the way the host code is written where the memory read and write kernels are NDRange kernels with a very large local group size, the input size for emulation must be very small or else emulation will either fail or crash with an error regarding number of threads.
