#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <string>
#include <vector>
#include <map>


using std::vector;
using std::string;
using std::map;
using std::make_pair;

#include "diffusion3d.h"
#include "baseline.h"

#if defined(OPENMP)
#include "diffusion3d_openmp.h"
#endif

#if defined(OPENMP_TEMPORAL_BLOCKING)
#include "diffusion3d_openmp_temporal_blocking.h"
#endif

#if defined(CUDA)                                                       \
  || defined(CUDA_ZBLOCK)                                               \
  || defined(CUDA_OPT0)                                                 \
  || defined(CUDA_OPT1)                                                 \
  || defined(CUDA_OPT2)                                                 \
  || defined(CUDA_OPT3)                                                 \
  || defined(CUDA_SHARED)                                               \
  || defined(CUDA_SHARED1)                                              \
  || defined(CUDA_SHARED2)                                              \
  || defined(CUDA_SHARED3)                                              \
  || defined(CUDA_SHARED4)                                              \
  || defined(CUDA_SHARED5)                                              \
  || defined(CUDA_SHARED6)                                              \
  || defined(CUDA_XY)
#include "diffusion3d_cuda.h"
#endif

#if defined(CUDA_TEMPORAL_BLOCKING)
#include "diffusion3d_cuda_temporal_blocking.h"
#endif

#if defined(MIC)
#include "diffusion3d_mic.h"
#endif

#if defined(PHYSIS)
#include "diffusion3d_physis.h"
#endif

#if defined(FORTRAN) || defined(FORTRAN_ACC)
#include "diffusion3d_fortran.h"
#endif

#if defined(ALTERA)
#include "diffusion3d_opencl.h"
#endif

using namespace diffusion3d;

void Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}

void PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--count N   " << "Number of iterations\n"
     << "\t--size N    " << "Size of each dimension\n"
     << "\t--dump      " << "Dump the final data to file\n"
	<< "\t--alt       " << "Alternative test function based on baseline code\n"
	<< "\t--nx        " << "Size of input in X dimension\n"
	<< "\t--ny        " << "Size of input in Y dimension\n"
	<< "\t--nz        " << "Size of input in Z dimension\n"
     << "\t--help      " << "Display this help message\n";
}


void ProcessProgramOptions(int argc, char *argv[],
                           int &count, int &size,
                           bool &dump, bool &alt_flag,
					  int &nx, int &ny, int &nz, bool &size_flag) {
  int c;
  int input_test = 0;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"count", 1, 0, 0},
      {"size", 1, 0, 0},
      {"dump", 0, 0, 0},
      {"help", 0, 0, 0},
      {"alt", 0, 0, 0},
	 {"nx", 1, 0, 0},
	 {"ny", 1, 0, 0},
	 {"nz", 1, 0, 0},
      {0, 0, 0, 0}
    };

    c = getopt_long(argc, argv, "",
                    long_options, &option_index);
    if (c == -1) break;
    if (c != 0) {
      //std::cerr << "Invalid usage\n";
      //PrintUsage(std::cerr, argv[0]);
      //Die();
      continue;
    }

    switch(option_index) {
      case 0:
        count = atoi(optarg);
        break;
      case 1:
        size = atoi(optarg);
	   input_test = 3;
        break;
      case 2:
        dump = true;
        break;
      case 3:
        PrintUsage(std::cerr, argv[0]);
        exit(EXIT_SUCCESS);
        break;
	 case 4:
	   alt_flag = true;
        break;
	 case 5:
	   nx = atoi(optarg);
	   size_flag = true;
	   input_test++;
        break;
	 case 6:
	   ny = atoi(optarg);
	   input_test++;
        break;
	 case 7:
	   nz = atoi(optarg);
	   input_test++;
        break;
      default:
        break;
    }
  }
  
  if (input_test != 0 && input_test != 3)
  {
    PrintUsage(std::cerr, argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {

  //default values
  int nx = 256;
  int ny = 256;
  int nz = 256;
  int size = 256;
  int count = 100;
  bool dump = false;
  bool alt_flag = false;
  bool size_flag = false;

  ProcessProgramOptions(argc, argv, count, size, dump, alt_flag, nx, ny, nz, size_flag);
  if (!size_flag) {
    nx = ny = nz = size;
  }
  Diffusion3D *bmk = NULL;

#if defined(OPENMP)
  bmk = new Diffusion3DOpenMP(nx, ny, nz);
#elif defined(OPENMP_TEMPORAL_BLOCKING)
  bmk = new Diffusion3DOpenMPTemporalBlocking(nx, ny, nz);
#elif defined(CUDA)
  bmk = new Diffusion3DCUDA(nx, ny, nz);
#elif defined(CUDA_ZBLOCK)
  bmk = new Diffusion3DCUDAZBlock(nx, ny, nz);
#elif defined(CUDA_OPT0)
  bmk = new Diffusion3DCUDAOpt0(nx, ny, nz);
#elif defined(CUDA_OPT1)
  bmk = new Diffusion3DCUDAOpt1(nx, ny, nz);
#elif defined(CUDA_OPT2)
  bmk = new Diffusion3DCUDAOpt2(nx, ny, nz);
#elif defined(CUDA_OPT3)
  bmk = new Diffusion3DCUDAOpt3(nx, ny, nz);
#elif defined(CUDA_SHARED)
  bmk = new Diffusion3DCUDAShared(nx, ny, nz);
#elif defined(CUDA_SHARED1)
  bmk = new Diffusion3DCUDAShared1(nx, ny, nz);
#elif defined(CUDA_SHARED2)
  bmk = new Diffusion3DCUDAShared2(nx, ny, nz);
#elif defined(CUDA_SHARED3)
  bmk = new Diffusion3DCUDAShared3(nx, ny, nz);
#elif defined(CUDA_SHARED4)
  bmk = new Diffusion3DCUDAShared4(nx, ny, nz);
#elif defined(CUDA_SHARED5)
  bmk = new Diffusion3DCUDAShared5(nx, ny, nz);
#elif defined(CUDA_SHARED6)
  bmk = new Diffusion3DCUDAShared6(nx, ny, nz);
#elif defined(CUDA_XY)
  bmk = new Diffusion3DCUDAXY(nx, ny, nz);
#elif defined(CUDA_TEMPORAL_BLOCKING)
  bmk = new Diffusion3DCUDATemporalBlocking(nx, ny, nz);
#elif defined(MIC)
  bmk = new Diffusion3DMIC(nx, ny, nz);
#elif defined(PHYSIS)
  bmk = new Diffusion3DPhysis(nx, ny, nz, argc, argv);
#elif defined(FORTRAN) || defined(FORTRAN_ACC)
  bmk = new Diffusion3DFortran(nx, ny, nz);
#elif defined(ALTERA)
  bmk = new Diffusion3DOpenCL(nx, ny, nz);
#else
  bmk = new Baseline(nx, ny, nz);
#endif

  bmk->alt_flag = alt_flag;
  bmk->RunBenchmark(count, dump);

  return 0;
}
