INCLUDEPATH=../common/
include $(INCLUDEPATH)common.mk

# Minimal configuration
SRC = diffusion3d.cc baseline.cc
OBJ = $(filter %.o,$(SRC:%.cc=%.o)) $(filter %.o,$(SRC:%.c=%.o))

EXE = diffusion3d_baseline.exe \
	diffusion3d_openmp.exe diffusion3d_openmp_temporal_blocking.exe \
	diffusion3d_cuda.exe \
	diffusion3d_cuda_zblock.exe \
	diffusion3d_cuda_opt0.exe \
	diffusion3d_cuda_opt1.exe \
	diffusion3d_cuda_opt2.exe \
	diffusion3d_cuda_opt3.exe \
	diffusion3d_cuda_shared.exe \
	diffusion3d_cuda_shared1.exe \
	diffusion3d_cuda_shared2.exe \
	diffusion3d_cuda_shared3.exe \
	diffusion3d_cuda_temporal_blocking.exe \
	diffusion3d_mic.exe \
	diffusion3d_fortran.exe \
	diffusion3d_fortran_acc.exe \
	diffusion3d_altera.exe

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
ifneq ($(ENABLE_ROC),)
	NVCC_CFLAGS += -DENABLE_ROC=$(ENABLE_ROC)
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

# if used, set both dimensions of the block to the same value
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

all: diffusion3d_baseline.exe diffusion3d_openmp.exe diffusion3d_openmp_temporal_blocking.exe

.PHONY: baseline
baseline: diffusion3d_baseline.exe

diffusion3d_baseline.exe: $(OBJ) main_baseline.o
	$(CXX) -o $@ $^ $(LDFLAGS)

main_baseline.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)

##################################################
.PHONY: fortran
fortran: diffusion3d_fortran.exe

main_fortran.o: main.cc diffusion3d_fortran.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN
diffusion3d_fortran.exe: $(OBJ) main_fortran.o diffusion3d_fortran.o
	$(CXX) -o $@ $^ $(LDFLAGS)
##################################################
.PHONY: openmp
openmp: diffusion3d_openmp.exe

diffusion3d_openmp.o: CXXFLAGS += $(OPENMP_CFLAGS)
diffusion3d_openmp.exe: $(OBJ) main_openmp.o diffusion3d_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP

openmp_temporal_blocking: diffusion3d_openmp_temporal_blocking.exe
diffusion3d_openmp_temporal_blocking.exe: $(OBJ) main_openmp_temporal_blocking.o \
	diffusion3d_openmp_temporal_blocking.o diffusion3d_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp_temporal_blocking.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP_TEMPORAL_BLOCKING

diffusion3d_openmp_temporal_blocking.o: CXXFLAGS += $(OPENMP_CFLAGS)
##################################################
.PHONY: cuda
cuda: diffusion3d_cuda.exe \
	diffusion3d_cuda_zblock.exe \
	diffusion3d_cuda_opt0.exe \
	diffusion3d_cuda_opt1.exe \
	diffusion3d_cuda_opt2.exe \
	diffusion3d_cuda_opt3.exe \
	diffusion3d_cuda_shared.exe \
	diffusion3d_cuda_shared1.exe \
	diffusion3d_cuda_shared2.exe \
	diffusion3d_cuda_shared3.exe \
	diffusion3d_cuda_shared4.exe \
	diffusion3d_cuda_shared5.exe \
	diffusion3d_cuda_shared6.exe \
	diffusion3d_cuda_xy.exe \
	diffusion3d_cuda_temporal_blocking.exe
##################################################
# diffusion3d_cuda (baseline)
diffusion3d_cuda.exe: $(OBJ) main_cuda.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion3d_cuda.o: diffusion3d_cuda.h
main_cuda.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA -I$(CUDA_INC)
##################################################
# diffusion3d_cuda_zblock
diffusion3d_cuda_zblock.exe: $(OBJ) main_cuda_zblock.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_zblock.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_ZBLOCK -I$(CUDA_INC)
##################################################
# diffusion3d_cuda_opt0
diffusion3d_cuda_opt0.exe: $(OBJ) main_cuda_opt0.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt0.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT0 -I$(CUDA_INC)
##################################################
# diffusion3d_cuda_opt1
diffusion3d_cuda_opt1.exe: $(OBJ) main_cuda_opt1.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt1.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT1 -I$(CUDA_INC)
##################################################
# diffusion3d_cuda_opt2
diffusion3d_cuda_opt2.exe: $(OBJ) main_cuda_opt2.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt2.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT2 -I$(CUDA_INC)
##################################################
# diffusion3d_cuda_opt3
diffusion3d_cuda_opt3.exe: $(OBJ) main_cuda_opt3.o diffusion3d_cuda.o \
	diffusion3d_cuda_opt3.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt3.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT3 -I$(CUDA_INC)
diffusion3d_cuda_opt3.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared
diffusion3d_cuda_shared.exe: $(OBJ) main_cuda_shared.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED -I$(CUDA_INC)
diffusion3d_cuda_shared.o: diffusion3d_cuda.h
# DISABLE L1 load caching
diffusion3d_cuda_shared.o: NVCC_CFLAGS += -Xptxas -dlcm=cg
##################################################
# diffusion3d_cuda_shared1
diffusion3d_cuda_shared1.exe: $(OBJ) main_cuda_shared1.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared1.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared1.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED1 -I$(CUDA_INC)
diffusion3d_cuda_shared1.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared2
diffusion3d_cuda_shared2.exe: $(OBJ) main_cuda_shared2.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared2.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared2.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED2 -I$(CUDA_INC)
diffusion3d_cuda_shared2.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared3
diffusion3d_cuda_shared3.exe: $(OBJ) main_cuda_shared3.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared3.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared3.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED3 -I$(CUDA_INC)
diffusion3d_cuda_shared3.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared4
diffusion3d_cuda_shared4.exe: $(OBJ) main_cuda_shared4.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared4.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared4.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED4 -I$(CUDA_INC)
diffusion3d_cuda_shared4.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared5
diffusion3d_cuda_shared5.exe: $(OBJ) main_cuda_shared5.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared5.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared5.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED5 -I$(CUDA_INC)
diffusion3d_cuda_shared5.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_shared6
diffusion3d_cuda_shared6.exe: $(OBJ) main_cuda_shared6.o diffusion3d_cuda.o \
	diffusion3d_cuda_shared.o diffusion3d_cuda_shared6.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared6.o: main.cc diffusion3d_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED6 -I$(CUDA_INC)
diffusion3d_cuda_shared6.o: diffusion3d_cuda.h
##################################################
# diffusion3d_cuda_xy
# XY multi processing
diffusion3d_cuda_xy.exe: $(OBJ) main_cuda_xy.o diffusion3d_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_xy.o: main.cc
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_XY -I$(CUDA_INC)
# Temporal blokcing
diffusion3d_cuda_temporal_blocking.o: diffusion3d_cuda.h diffusion3d_cuda_temporal_blocking.h
diffusion3d_cuda_temporal_blocking.exe: $(OBJ) main_cuda_temporal_blocking.o diffusion3d_cuda.o diffusion3d_cuda_temporal_blocking.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_temporal_blocking.o: main.cc diffusion3d_cuda_temporal_blocking.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_TEMPORAL_BLOCKING -I$(CUDA_INC)
##################################################
# Fortran OpenACC
.PHONY: fortran-acc
fortran-acc: diffusion3d_fortran_acc.exe
main_fortran_acc.o: main.cc diffusion3d_fortran.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN_ACC
diffusion3d_fortran_acc.o: diffusion3d_fortran.F90
	$(FC) -o $@ -c $< $(FFLAS) $(ACC_FFLAGS) -DFORTRAN_ACC
diffusion3d_fortran_acc.exe: $(OBJ) main_fortran_acc.o diffusion3d_fortran_acc.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(ACC_LDFLAGS) $(F90_LIBS)
##################################################
.PHONY: physis
physis: physis-ref physis-cuda physis-mpi physis-mpi-cuda

.PHONY: physis-ref
physis-ref: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.exe

.PHONY: physis-cuda
physis-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.exe

.PHONY: physis-mpi
physis-mpi: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.exe

.PHONY: physis-mpi-cuda
physis-mpi-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.exe

$(PHYSIS_BUILD_DIR):
	mkdir -p $(PHYSIS_BUILD_DIR)

main_physis.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DPHYSIS
# reference
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.c: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_REF) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.o: CFLAGS += -I$(PHYSIS_DIR)/include
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.ref.o \
	main_physis.o baseline.o diffusion3d.o $(PHYSIS_DIR)/lib/libphysis_rt_ref.a
	$(CXX) -o $@ $^ $(LDFLAGS)
# cuda
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.cu: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.o: NVCC_CFLAGS += -I$(PHYSIS_DIR)/include
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.cuda.o \
	main_physis.o baseline.o diffusion3d.o $(PHYSIS_DIR)/lib/libphysis_rt_cuda.a
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
# mpi
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.c: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.o: CFLAGS += -I$(PHYSIS_DIR)/include -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi.o \
	main_physis.o baseline.o diffusion3d.o $(PHYSIS_DIR)/lib/libphysis_rt_mpi.a
	$(MPICXX) -o $@ $^ $(LDFLAGS)

# mpi-cuda
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.cu: diffusion3d_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.o: NVCC_CFLAGS += -I$(PHYSIS_DIR)/include \
	-arch sm_20 -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.exe: $(PHYSIS_BUILD_DIR)/diffusion3d_physis.mpi-cuda.o \
	main_physis.o baseline.o diffusion3d.o $(PHYSIS_DIR)/lib/libphysis_rt_mpi_cuda.a
	$(MPICXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
##################################################
.PHONY: mic
mic: diffusion3d_mic.exe

diffusion3d_mic.o: CXXFLAGS += $(OPENMP_CFLAGS)
diffusion3d_mic.exe: $(OBJ) main_mic.o diffusion3d_mic.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)
main_mic.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP
##################################################
.PHONY: altera
altera: altera-host \
	   altera-kernel
altera-host: diffusion3d_altera.exe
altera-kernel: diffusion3d_altera.aocx
##################################################
# diffusion3d_opencl
diffusion3d_altera.exe: $(OBJ) main_opencl.o diffusion3d_opencl.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENCL_LDFLAGS) $(ALTERA_HOST_LDFLAGS)
diffusion3d_opencl.o: diffusion3d_opencl.cc diffusion3d_opencl.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(OPENCL_INC) $(ALTERA_HOST_CFLAGS)
main_opencl.o: main.cc diffusion3d_opencl.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(OPENCL_INC) $(ALTERA_HOST_CFLAGS)
diffusion3d_altera.aocx:
	sh diffusion3d_opencl_replace.sh $(RAD)
	aoc diffusion3d_opencl.cl $(ALTERA_KERNEL_FLAGS)
##################################################

clean:
	-$(RM) *.o $(EXE)
	-$(RM) diffusion3d_result.*.out
	-$(RM) *.cudafe* *.gpu *.stub.c *.pptx *.cubin *.i *.ii *.fatbin *.fatbin.c
	-$(RM) *_physis.ref.* *_physis.cuda.* *_physis.mpi.* \
		*_physis.mpi-cuda.*
	-$(RM) -rf $(PHYSIS_BUILD_DIR_TOP)
