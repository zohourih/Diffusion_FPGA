# Diffusion_FPGA
This repository is based on original work from [Naoya Maruyma](https://github.com/naoyam) on implementing and optimizing first-order
Diffusion 2D and 3D stencils for CPUs, GPUs and Xeon Phi. Original repository can be found [here](https://github.com/naoyam/benchmarks).
The original implementation has been extended with an OpenCL implementation targetting Intel FPGAs using Intel FPGA SDK for OpenCL.
This version implements a highly-optimized design for FPGA with combined spatial and temporal blocking and numerous
FPGA-specific optimizations. Moreover, the OpenCL version has been extended to support up to fourth-order stencils.

# Content
## common
Common make configuration and timer, helper and power fucntions for Intel FPGAs, Nvidia GPUs and Intel CPUs

## diffusion2d/diffusion3d
Diffusion 2D/3D implementation for different hardware.
Only OpenCL version supports high-order stencils and can be run on FPGAs.
Refer to main README in each fodler for details of compiling and running the FPGA version, and original README for other versions.

# Detailed info
Refer to the following publications:

- Hamid Reza Zohouri, Artur Podobas, Satoshi Matsuoka, "Combined Spatial and Temporal Blocking for High-Performance Stencil Computation on FPGAs Using OpenCL," Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'18), Feb. 2018. [[arXiv]](https://arxiv.org/abs/1802.00438)[[ACM]](https://dl.acm.org/citation.cfm?id=3174248)[[slides]](http://isfpga.org/fpga2018/slides/5-1.pdf)
- Hamid Reza Zohouri, Artur Podobas, Satoshi Matsuoka, "High-Performance High-Order Stencil Computation on FPGAs Using OpenCL," 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW'18), May 2018. [[arXiv]](https://arxiv.org/abs/2002.05983)[[IEEE]](https://ieeexplore.ieee.org/abstract/document/8425394)
- Hamid Reza Zohouri, "High Performance Computing with FPGAs and OpenCL," PhD thesis, Tokyo Institute of Technology, Tokyo, Japan, Aug. 2018. [[PDF]](https://arxiv.org/abs/1810.09773)

The thesis has the most up-to-date results.
