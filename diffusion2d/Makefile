INCLUDEPATH=../common/
include $(INCLUDEPATH)common.mk

# Minimal configuration
SRC = 
OBJ = $(filter %.o,$(SRC:%.cc=%.o)) $(filter %.o,$(SRC:%.c=%.o))

EXE = diffusion_baseline \
	diffusion_cuda.exe \
	diffusion_cuda_restrict.exe \
	diffusion_cuda_zblock.exe \
	diffusion_cuda_roc.exe \
	diffusion_cuda_opt1.exe \
	diffusion_cuda_opt2.exe \
	diffusion_cuda_shared1.exe \
	diffusion_cuda_shared2.exe \
	diffusion_cuda_shared3.exe \
	diffusion_cuda_shared4.exe \
	diffusion_cuda_shared5.exe \
	diffusion_cuda_shared6.exe \
	diffusion_cuda_shfl1.exe \
	diffusion_cuda_shfl2.exe \
	diffusion_cuda_shfl3.exe \
	diffusion_altera.exe

# Options
ifneq ($(BLOCK_X),)
	CXXFLAGS += -DBLOCK_X=$(BLOCK_X)
	NVCC_CFLAGS += -DBLOCK_X=$(BLOCK_X)
	ALTERA_HOST_CFLAGS += -DBLOCK_X=$(BLOCK_X)
	ALTERA_KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X)
endif	
ifneq ($(BLOCK_Y),)
	CXXFLAGS += -DBLOCK_Y=$(BLOCK_Y)
	NVCC_CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
	ALTERA_HOST_CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
	ALTERA_KERNEL_FLAGS += -DBLOCK_Y=$(BLOCK_Y)
endif	
ifneq ($(GRID_Z),)
	CXXFLAGS += -DGRID_Z=$(GRID_Z)
	NVCC_CFLAGS += -DGRID_Z=$(GRID_Z)
endif	
ifneq ($(UNROLL),)
	CXXFLAGS += -DUNROLL=$(UNROLL)
	NVCC_CFLAGS += -DUNROLL=$(UNROLL)
endif	
ifneq ($(DLCM),)
	NVCC_CFLAGS += -Xptxas -dlcm=$(DLCM)
endif	
ifneq ($(DSCM),)
	NVCC_CFLAGS += -Xptxas -dscm=$(DSCM)
endif	
ifneq ($(ARCH),)
	NVCC_ARCH = -arch $(ARCH)
endif	
ifneq ($(REG),)
	NVCC_ARCH += --maxrregcount $(REG)
endif

RAD := 1
ALTERA_HOST_CFLAGS += -DRAD=$(RAD)
ALTERA_KERNEL_FLAGS += -DRAD=$(RAD)
CXXFLAGS += -DRAD=$(RAD)

ifdef TIME
	ALTERA_HOST_CFLAGS += -DTIME=$(TIME)
	ALTERA_KERNEL_FLAGS += -DTIME=$(TIME)
endif

ifdef ASIZE
	ALTERA_HOST_CFLAGS += -DASIZE=$(ASIZE)
	ALTERA_KERNEL_FLAGS += -DASIZE=$(ASIZE)
endif

# set both dimensions of the block to the same value
ifdef BSIZE
	ALTERA_HOST_CFLAGS += -DBLOCK_X=$(BSIZE) -DBLOCK_Y=$(BSIZE)
	ALTERA_KERNEL_FLAGS += -DBLOCK_X=$(BSIZE) -DBLOCK_Y=$(BSIZE)
endif

ifdef CSIZE
	ALTERA_KERNEL_FLAGS += -DCSIZE=$(CSIZE)
endif

ifdef PAD
	ALTERA_HOST_CFLAGS += -DPAD=$(PAD)
	ALTERA_KERNEL_FLAGS += -DPAD=$(PAD)
endif

ifdef NO_INTERLEAVE
	ALTERA_HOST_CFLAGS += -DNO_INTERLEAVE
	ALTERA_KERNEL_FLAGS += $(DASH)no-interleaving$(SPACE)default
endif

##################################################

all: $(EXE)

.PHONY: baseline
baseline: diffusion_baseline

diffusion_baseline: $(OBJ) main_baseline.o
	$(CXX) -o $@ $^ $(LDFLAGS)

main_baseline.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)

##################################################
# .PHONY: fortran
# fortran: diffusion_fortran

# main_fortran.o: main.cc diffusion_fortran.h
# 	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN
# diffusion_fortran: $(OBJ) main_fortran.o diffusion_fortran.o
# 	$(CXX) -o $@ $^ $(LDFLAGS)
##################################################
.PHONY: openmp
openmp: diffusion_openmp

diffusion_openmp.o: CXXFLAGS += $(OPENMP_CFLAGS)
diffusion_openmp: $(OBJ) main_openmp.o diffusion_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP

# openmp_temporal_blocking: diffusion_openmp_temporal_blocking
# diffusion_openmp_temporal_blocking: $(OBJ) main_openmp_temporal_blocking.o \
# 	diffusion_openmp_temporal_blocking.o diffusion_openmp.o
# 	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

# main_openmp_temporal_blocking.o: main.cc
# 	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP_TEMPORAL_BLOCKING

# diffusion_openmp_temporal_blocking.o: CXXFLAGS += $(OPENMP_CFLAGS)
##################################################
CUDA_EXE = diffusion_cuda.exe \
	diffusion_cuda_restrict.exe \
	diffusion_cuda_zblock.exe \
	diffusion_cuda_roc.exe \
	diffusion_cuda_opt1.exe \
	diffusion_cuda_opt2.exe \
	diffusion_cuda_shared1.exe \
	diffusion_cuda_shared2.exe \
	diffusion_cuda_shared3.exe \
	diffusion_cuda_shared4.exe \
	diffusion_cuda_shared5.exe \
	diffusion_cuda_shared6.exe \
	diffusion_cuda_shfl1.exe \
	diffusion_cuda_shfl2.exe \
	diffusion_cuda_shfl3.exe

.PHONY: cuda
cuda: $(CUDA_EXE)

########################################
# diffusion_cuda (baseline)
diffusion_cuda.exe: $(OBJ) main_cuda.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion_cuda.o: diffusion_cuda.h
main_cuda.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA -I$(CUDA_INC)
########################################
# diffusion_cuda_roc.exe
diffusion_cuda_roc.exe: $(OBJ) main_cuda_roc.o diffusion_cuda_roc.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion_cuda_roc.o: diffusion_cuda_roc.h
main_cuda_roc.o: main.cc diffusion_cuda.h diffusion_cuda_roc.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_ROC -I$(CUDA_INC)
########################################
# diffusion_cuda_restrict.exe
diffusion_cuda_restrict.exe: $(OBJ) main_cuda_restrict.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_restrict.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_RESTRICT -I$(CUDA_INC)
########################################
# diffusion_cuda_zblock.exe
diffusion_cuda_zblock.exe: $(OBJ) main_cuda_zblock.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_zblock.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_ZBLOCK -I$(CUDA_INC)
########################################
# diffusion_cuda_copy.exe
diffusion_cuda_copy.exe: $(OBJ) main_cuda_copy.o diffusion_cuda.o \
	diffusion_cuda_copy.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_copy.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_COPY -I$(CUDA_INC)
########################################
# diffusion_cuda_opt1.exe
diffusion_cuda_opt1.exe: $(OBJ) main_cuda_opt1.o diffusion_cuda.o \
	diffusion_cuda_opt.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt1.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT1 -I$(CUDA_INC)
########################################
# diffusion_cuda_opt2.exe
diffusion_cuda_opt2.exe: $(OBJ) main_cuda_opt2.o diffusion_cuda.o \
	diffusion_cuda_opt.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt2.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT2 -I$(CUDA_INC)
########################################
# diffusion_cuda_shared.exe
diffusion_cuda_shared1.exe: $(OBJ) main_cuda_shared1.o diffusion_cuda.o \
	diffusion_cuda_shared1.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared1.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED1 -I$(CUDA_INC)
diffusion_cuda_shared1.o: diffusion_cuda.h diffusion_cuda_shared.h
# DISABLE L1 load caching
#diffusion_cuda_shared.o: NVCC_CFLAGS += -Xptxas -dlcm=cg
########################################
# diffusion_cuda_shared1.exe
diffusion_cuda_shared2.exe: $(OBJ) main_cuda_shared2.o diffusion_cuda.o \
	diffusion_cuda_shared2.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared2.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED2 -I$(CUDA_INC)
diffusion_cuda_shared2.o: diffusion_cuda.h diffusion_cuda_shared.h
########################################
# diffusion_cuda_shared3.exe
diffusion_cuda_shared3.exe: $(OBJ) main_cuda_shared3.o diffusion_cuda.o \
	diffusion_cuda_shared3.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared3.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED3 -I$(CUDA_INC)
diffusion_cuda_shared3.o: diffusion_cuda.h diffusion_cuda_shared.h
########################################
# diffusion_cuda_shared3.exe_prefetch
diffusion_cuda_shared3.exe_prefetch: $(OBJ) main_cuda_shared3_prefetch.o diffusion_cuda.o \
	diffusion_cuda_shared3_prefetch.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared3_prefetch.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED3_PREFETCH -I$(CUDA_INC)
diffusion_cuda_shared3_prefetch.o: diffusion_cuda.h diffusion_cuda_shared.h
########################################
# diffusion_cuda_shared4.exe
diffusion_cuda_shared4.exe: $(OBJ) main_cuda_shared4.o diffusion_cuda.o \
	diffusion_cuda_shared4.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared4.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED4 -I$(CUDA_INC)
diffusion_cuda_shared4.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared5.exe
diffusion_cuda_shared5.exe: $(OBJ) main_cuda_shared5.o diffusion_cuda.o \
	diffusion_cuda_shared5.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared5.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED5 -I$(CUDA_INC)
diffusion_cuda_shared5.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared6.exe
diffusion_cuda_shared6.exe: $(OBJ) main_cuda_shared6.o diffusion_cuda.o \
	diffusion_cuda_shared6.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared6.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED6 -I$(CUDA_INC)
diffusion_cuda_shared6.o: diffusion_cuda.h
########################################
# diffusion_cuda_shfl1.exe
diffusion_cuda_shfl1.exe: $(OBJ) main_cuda_shfl1.o diffusion_cuda.o \
	diffusion_cuda_shfl1.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shfl1.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHFL1 -I$(CUDA_INC)
diffusion_cuda_shfl1.o: diffusion_cuda.h
########################################
# diffusion_cuda_shfl2.exe
diffusion_cuda_shfl2.exe: $(OBJ) main_cuda_shfl2.o diffusion_cuda.o \
	diffusion_cuda_shfl2.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shfl2.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHFL2 -I$(CUDA_INC)
diffusion_cuda_shfl2.o: diffusion_cuda.h
########################################
# diffusion_cuda_shfl3.exe
diffusion_cuda_shfl3.exe: $(OBJ) main_cuda_shfl3.o diffusion_cuda.o \
	diffusion_cuda_shfl3.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shfl3.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHFL3 -I$(CUDA_INC)
diffusion_cuda_shfl3.o: diffusion_cuda.h
########################################
# diffusion_cuda_xy
# XY multi processing
diffusion_cuda_xy: $(OBJ) main_cuda_xy.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_xy.o: main.cc
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_XY -I$(CUDA_INC)
# Temporal blocking
diffusion_cuda_temporal_blocking.o: diffusion_cuda.h diffusion_cuda_temporal_blocking.h
diffusion_cuda_temporal_blocking: $(OBJ) main_cuda_temporal_blocking.o diffusion_cuda.o diffusion_cuda_temporal_blocking.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_temporal_blocking.o: main.cc diffusion_cuda_temporal_blocking.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_TEMPORAL_BLOCKING -I$(CUDA_INC)
##################################################
##################################################
# Fortran OpenACC
# .PHONY: fortran-acc
# fortran-acc: diffusion_fortran_acc
# main_fortran_acc.o: main.cc diffusion_fortran.h
# 	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN_ACC
# diffusion_fortran_acc.o: diffusion_fortran.F90
# 	$(FC) -o $@ -c $< $(FFLAS) $(ACC_FFLAGS) -DFORTRAN_ACC
# diffusion_fortran_acc: $(OBJ) main_fortran_acc.o diffusion_fortran_acc.o
# 	$(CXX) -o $@ $^ $(LDFLAGS) $(ACC_LDFLAGS) $(F90_LIBS)
##################################################


# include physis.mk
##################################################
.PHONY: altera
altera: altera-host \
	   altera-kernel
altera-host: diffusion_altera.exe
altera-kernel: diffusion_altera.aocx
##################################################
# diffusion_opencl
diffusion_altera.exe: $(OBJ) main_opencl.o diffusion_opencl.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENCL_LDFLAGS) $(ALTERA_HOST_LDFLAGS)
diffusion_opencl.o: diffusion_opencl.cc diffusion_opencl.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(OPENCL_INC) $(ALTERA_HOST_CFLAGS)
main_opencl.o: main.cc diffusion_opencl.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(OPENCL_INC) $(ALTERA_HOST_CFLAGS)
diffusion_altera.aocx:
	sh diffusion_opencl_replace.sh $(RAD)
	aoc diffusion_opencl.cl $(ALTERA_KERNEL_FLAGS)
##################################################

clean:
	-$(RM) *.o $(EXE)
	-$(RM) diffusion_result.*.out
	-$(RM) *.cudafe* *.gpu *.stub.c *.pptx *.cubin *.i *.ii *.fatbin *.fatbin.c
	-$(RM) *_physis.ref.* *_physis.cuda.* *_physis.mpi.* \
		*_physis.mpi-cuda.*
	-$(RM) -rf $(PHYSIS_BUILD_DIR_TOP)
