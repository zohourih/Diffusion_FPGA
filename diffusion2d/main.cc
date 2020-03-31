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

#include "diffusion.h"
#include "baseline.h"
#if defined(OPENMP)
#include "diffusion_openmp.h"
#endif

#if defined(CUDA) || defined(CUDA_ZBLOCK) || defined(CUDA_L1) || defined(CUDA_RESTRICT)
#include "diffusion_cuda.h"
#endif
#ifdef CUDA_ROC
#include "diffusion_cuda_roc.h"
#endif
#ifdef CUDA_COPY
#include "diffusion_cuda_copy.h"
#endif
#if defined(CUDA_OPT1) || defined(CUDA_OPT2)
#include "diffusion_cuda_opt.h"
#endif
#if defined(CUDA_SHARED1) || defined(CUDA_SHARED2)  \
  || defined(CUDA_SHARED3) || defined(CUDA_SHARED4) \
  || defined(CUDA_SHARED5) || defined(CUDA_SHARED6) \
  || defined(CUDA_SHARED3_PREFETCH)
#include "diffusion_cuda_shared.h"
#endif
#if defined(CUDA_SHFL1) || defined(CUDA_SHFL2) \
  || defined(CUDA_SHFL3)
#include "diffusion_cuda_shfl.h"
#endif

#if 0
#if defined(OPENMP_TEMPORAL_BLOCKING)
#include "diffusion3d/diffusion3d_openmp_temporal_blocking.h"
#endif


#if defined(CUDA_TEMPORAL_BLOCKING)
#include "diffusion_cuda_temporal_blocking.h"
#endif

#if defined(MIC)
#include "diffusion_mic.h"
#endif

#if defined(PHYSIS)
#include "diffusion_physis.h"
#endif

#if defined(FORTRAN) || defined(FORTRAN_ACC)
#include "diffusion_fortran.h"
#endif
#endif

#if defined(ALTERA)
#include "diffusion_opencl.h"
#endif

using namespace diffusion;
using std::string;


static const int COUNT = 100;
static const int ND = 3;
static const int SIZE = 256;

void Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}

void PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--nd N      " << "Number of dimensions (default: " << ND << ")\n"
     << "\t--count N   " << "Number of iterations (default: " << COUNT << ")\n"
     << "\t--size N    "  << "Size of each dimension (default: " << SIZE << ")\n"
     << "\t--dump      "  << "Dump the final data to file\n"
     << "\t--warmup    "  << "Enable warming-up runs\n"
	<< "\t--alt       " << "Alternative test function based on baseline code\n"
	<< "\t--nx        " << "Size of input in X dimension (default: " << ND << ")\n"
	<< "\t--ny        " << "Size of input in Y dimension (default: " << ND << ")\n"
	<< "\t--nz        " << "Size of input in Z dimension (default: " << ND << ")\n"
     << "\t--help      "  << "Display this help message\n";
}


void ProcessProgramOptions(int argc, char *argv[],
                           int &nd, int &count, int &size,
                           bool &dump, bool &warmup, bool &alt_flag,
					  int &nx, int &ny, int &nz, bool &size_flag) {
  int c;
  int input_test = 0;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"nd", 1, 0, 0},
      {"count", 1, 0, 0},
      {"size", 1, 0, 0},
      {"dump", 0, 0, 0},
      {"warmup", 0, 0, 0},
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
        nd = atoi(optarg);
        break;
      case 1:
        count = atoi(optarg);
        break;
      case 2:
        size = atoi(optarg);
        break;
      case 3:
        dump = true;
        break;
      case 4:
        warmup = true;
        break;
      case 5:
        PrintUsage(std::cerr, argv[0]);
        exit(EXIT_SUCCESS);
        break;
	 case 6:
	   alt_flag = true;
        break;
	 case 7:
	   nx = atoi(optarg);
	   size_flag = true;
	   input_test++;
        break;
	 case 8:
	   ny = atoi(optarg);
	   input_test++;
        break;
	 case 9:
	   nz = atoi(optarg);
	   input_test++;
        break;
      default:
        break;
    }
  }

  if (input_test != 0 && input_test != 2 && input_test != 3)
  {
    PrintUsage(std::cerr, argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  int nd = ND;
  int size = SIZE; // default size
  int nx = SIZE;
  int ny = SIZE;
  int nz = SIZE;
  int count = COUNT; // default iteration count
  bool dump = false;
  bool warmup = false;
  bool alt_flag = false;
  bool size_flag = false;
  
  ProcessProgramOptions(argc, argv, nd, count, size, dump, warmup, alt_flag, nx, ny, nz, size_flag);
  if (!size_flag) {
    nx = ny = nz = size;
  }

  assert(nd >= 2 && nd <= 3);
  
  Diffusion *bmk = NULL;

  //std::vector<int> dims(nd, size);
  std::vector<int> dims(nd);
  dims[0] = nx;
  dims[1] = ny;
  if (nd > 2) {
    dims[2] = nz;
  }

#if defined(OPENMP)
  bmk = new DiffusionOpenMP(nd, dims.data());
#elif defined(OPENMP_TEMPORAL_BLOCKING)
  bmk = new DiffusionOpenMPTemporalBlocking(nx, nx, nx);
#elif defined(CUDA) || defined(CUDA_L1)
  bmk = new DiffusionCUDA(nd, dims.data());
#elif defined(CUDA_ROC)
  bmk = new DiffusionCUDAROC(nd, dims.data());
#elif defined(CUDA_RESTRICT)
  bmk = new DiffusionCUDARestrict(nd, dims.data());
#elif defined(CUDA_ZBLOCK)
  bmk = new DiffusionCUDAZBlock(nd, dims.data());
#elif defined(CUDA_COPY)
  bmk = new DiffusionCUDACopy(nd, dims.data());
#elif defined(CUDA_OPT1)
  bmk = new DiffusionCUDAOpt1(nd, dims.data());
#elif defined(CUDA_OPT2)
  bmk = new DiffusionCUDAOpt2(nd, dims.data());
#elif defined(CUDA_SHARED1)
  bmk = new DiffusionCUDAShared1(nd, dims.data());
#elif defined(CUDA_SHARED2)
  bmk = new DiffusionCUDAShared2(nd, dims.data());
#elif defined(CUDA_SHARED3)
  bmk = new DiffusionCUDAShared3(nd, dims.data());
#elif defined(CUDA_SHARED3_PREFETCH)
  bmk = new DiffusionCUDAShared3Prefetch(nd, dims.data());
#elif defined(CUDA_SHARED4)
  bmk = new DiffusionCUDAShared4(nd, dims.data());
#elif defined(CUDA_SHARED5)
  bmk = new DiffusionCUDAShared5(nd, dims.data());
#elif defined(CUDA_SHARED6)
  bmk = new DiffusionCUDAShared6(nd, dims.data());
#elif defined(CUDA_SHFL1)
  bmk = new DiffusionCUDASHFL1(nd, dims.data());
#elif defined(CUDA_SHFL2)
  bmk = new DiffusionCUDASHFL2(nd, dims.data());
#elif defined(CUDA_SHFL3)
  bmk = new DiffusionCUDASHFL3(nd, dims.data());
#elif defined(CUDA_XY)
  bmk = new DiffusionCUDAXY(nx, nx, nx);
#elif defined(CUDA_TEMPORAL_BLOCKING)
  bmk = new DiffusionCUDATemporalBlocking(nx, nx, nx);
#elif defined(MIC)
  bmk = new DiffusionMIC(nx, nx, nx);
#elif defined(PHYSIS)
  bmk = new DiffusionPhysis(nx, nx, nx, argc, argv);
#elif defined(FORTRAN) || defined(FORTRAN_ACC)
  bmk = new DiffusionFortran(nx, nx, nx);
#elif defined(ALTERA)
  bmk = new DiffusionOpenCL(nd, dims.data());
#else
  bmk = new Baseline(nd, dims.data());
#endif

  bmk->alt_flag = alt_flag;
  bmk->RunBenchmark(count, dump, warmup);

  return 0;
}
