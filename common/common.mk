ifneq ($(USE_PGI),)
include $(INCLUDEPATH)pgi.mk
else ifneq ($(USE_INTEL),)
include $(INCLUDEPATH)intel.mk
else
# use GCC by default
include $(INCLUDEPATH)gcc.mk 
endif

UNAME := $(shell uname -s)

CFLAGS = -fopenmp -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
CXXFLAGS = --std=c++11 -fopenmp -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
FFLAGS = -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
LDFLAGS = -lm -m64 -fopenmp

.SUFFIXES: .f90
.SUFFIXES: .F90
.F90.o:
	$(FC) -c $(FFLAGS) $<
.f90.o:
	$(FC) -c $(FFLAGS) $<

### MPI ###
MPICC = mpicc
MPICXX = mpicxx
MPI_INCLUDE = $(shell mpicc -show | sed 's/.*-I\([\/a-zA-Z0-9_\-]*\).*/\1/g')

### CUDA ###
NVCC = nvcc
NVCC_CFLAGS = --std=c++11 -m64 -I.. -Xcompiler -Wall -Xptxas -v -Wno-deprecated-gpu-targets # -keep
ifdef ARCH
	NVCC_ARCH = -arch $(ARCH)
else
	NVCC_ARCH = -arch sm_35
endif

ifneq ($(DEBUG),)
NVCC_CFLAGS += -g -G
else
NVCC_CFLAGS += -O3
endif
CUDA_INC = $(patsubst %bin/nvcc,%include, $(shell which $(NVCC)))
ifeq (,$(findstring Darwin,$(shell uname)))
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib64, \
		$(shell which $(NVCC)))
else
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib, \
		$(shell which $(NVCC)))
endif

ifneq ($(NVCC_USE_FAST_MATH),)
	NVCC_CFLAGS += -use_fast_math
endif

ifeq ($(UNAME),Darwin)
	NVCC_CFLAGS += -ccbin=llvm-g++
endif

.SUFFIXES: .cu

.cu.o:
	$(NVCC) -c $< $(NVCC_CFLAGS) $(NVCC_ARCH) -o $@

### Physis ###
PHYSISC_CONFIG ?= /dev/null
PHYSISC_CONFIG_KEY = $(shell basename $(PHYSISC_CONFIG))
PHYSISC_REF = $(PHYSIS_DIR)/bin/physisc-ref --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_CUDA = $(PHYSIS_DIR)/bin/physisc-cuda --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_MPI = $(PHYSIS_DIR)/bin/physisc-mpi --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_MPI_CUDA = $(PHYSIS_DIR)/bin/physisc-mpi-cuda --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSIS_BUILD_DIR_TOP = physis_build
PHYSIS_BUILD_DIR = physis_build/$(PHYSISC_CONFIG_KEY)

### ALTERA OpenCL SDK ###
AOC_VERSION = $(shell aoc --version | grep Build | cut -c 9-10)
LEGACY = $(shell echo $(AOC_VERSION)\<17 | bc)
ifeq ($(LEGACY),1)
	DASH = --
	SPACE = $(shell echo " ")
	ALTERA_HOST_CFLAGS += -DLEGACY
	ALTERA_KERNEL_FLAGS += -DLEGACY
else
	DASH = -
	SPACE = =
endif
ALTERA_KERNEL_FLAGS += -g -v $(DASH)report $(DASH)opt-arg$(SPACE)-nocaching
ALTERA_HOST_CFLAGS += -DALTERA

OPENCL_LDFLAGS = $(shell aocl link-config) -lOpenCL
OPENCL_INC = $(shell aocl compile-config)

ifdef BOARD
	ALTERA_HOST_CFLAGS += -DAOCL_BOARD_$(BOARD)
	ALTERA_KERNEL_FLAGS += $(DASH)board$(SPACE)$(BOARD)
endif

ifeq ($(EMULATOR),1)
	ALTERA_KERNEL_FLAGS += -march=emulator
	ALTERA_HOST_CFLAGS += -DEMULATOR
endif

ifeq ($(PROFILE),1)
	ALTERA_KERNEL_FLAGS += $(DASH)profile
endif

### Power ###

# Nvidia NVML from CUDA Toolkit for Nvidia GPU power measurement, CUDA_DIR must be defined in bashrc
NVML_INC = -I$(CUDA_DIR)/include
NVML_LIB = -L$(CUDA_DIR)/lib64/stubs -lnvidia-ml -fopenmp
NVCC_CFLAGS += $(NVML_INC) -Xcompiler -fopenmp
CUDA_LDFLAGS += $(NVML_LIB)
CXXFLAGS += -fopenmp

# Bittware BmcLib for power measurement on Bittware FPGA boards
# BITTWARE_TOOLKIT must be defined in bashrc and point to Bittware II Toolkit
BITTWARE_INC = -I$(BITTWARE_TOOLKIT)/include -I$(BITTWARE_SDK)/include/resources -fopenmp
BITTWARE_LIB = -L$(BITTWARE_TOOLKIT) -lbwhil -lbmclib -fopenmp
BITTWARE_FLAGS = -DAOCL_BOARD_a10pl4_dd4gb_gx115
ifeq ($(BOARD),a10pl4_dd4gb_gx115)
	ALTERA_HOST_CFLAGS += $(BITTWARE_INC) $(BITTWARE_FLAGS)
	ALTERA_HOST_LDFLAGS += $(BITTWARE_LIB)
endif

# Power measurement on Nallatech FPGA boards
# AOCL_BOARD_PACKAGE_ROOT should point to a Nallatech BSP that includes the aocl_mmd.h header
NALLATECH_INC = -I$(AOCL_BOARD_PACKAGE_ROOT)/software/include -fopenmp
NALLATECH_LIB = -fopenmp
NALLATECH_FLAGS = -DAOCL_BOARD_p385a_sch_ax115
ifeq ($(BOARD),p385a_sch_ax115)
	ALTERA_HOST_CFLAGS += $(NALLATECH_INC) $(NALLATECH_FLAGS)
	ALTERA_HOST_LDFLAGS += $(NALLATECH_INC) $(NALLATECH_FLAGS)
endif
