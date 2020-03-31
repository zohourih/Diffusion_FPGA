.PHONY: physis
physis: physis-ref physis-cuda physis-mpi physis-mpi-cuda

.PHONY: physis-ref
physis-ref: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion_physis.ref

.PHONY: physis-cuda
physis-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion_physis.cuda

.PHONY: physis-mpi
physis-mpi: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion_physis.mpi

.PHONY: physis-mpi-cuda
physis-mpi-cuda: $(PHYSIS_BUILD_DIR) $(PHYSIS_BUILD_DIR)/diffusion_physis.mpi-cuda

$(PHYSIS_BUILD_DIR):
	mkdir -p $(PHYSIS_BUILD_DIR)

main_physis.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DPHYSIS
# reference
$(PHYSIS_BUILD_DIR)/diffusion_physis.ref.c: diffusion_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_REF) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion_physis.ref.o: CFLAGS += -I$(PHYSIS_DIR)/include
$(PHYSIS_BUILD_DIR)/diffusion_physis.ref: $(PHYSIS_BUILD_DIR)/diffusion_physis.ref.o \
	main_physis.o baseline.o diffusion.o $(PHYSIS_DIR)/lib/libphysis_rt_ref.a
	$(CXX) -o $@ $^ $(LDFLAGS)
# cuda
$(PHYSIS_BUILD_DIR)/diffusion_physis.cuda.cu: diffusion_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion_physis.cuda.o: NVCC_CFLAGS += -I$(PHYSIS_DIR)/include
$(PHYSIS_BUILD_DIR)/diffusion_physis.cuda: $(PHYSIS_BUILD_DIR)/diffusion_physis.cuda.o \
	main_physis.o baseline.o diffusion.o $(PHYSIS_DIR)/lib/libphysis_rt_cuda.a
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
# mpi
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi.c: diffusion_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi.o: CFLAGS += -I$(PHYSIS_DIR)/include -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi: $(PHYSIS_BUILD_DIR)/diffusion_physis.mpi.o \
	main_physis.o baseline.o diffusion.o $(PHYSIS_DIR)/lib/libphysis_rt_mpi.a
	$(MPICXX) -o $@ $^ $(LDFLAGS)

# mpi-cuda
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi-cuda.cu: diffusion_physis.c $(PHYSISC_CONFIG)
	cd $(PHYSIS_BUILD_DIR) && $(PHYSISC_MPI_CUDA) ../../$<
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi-cuda.o: NVCC_CFLAGS += -I$(PHYSIS_DIR)/include \
	-arch sm_20 -I$(MPI_INCLUDE)
$(PHYSIS_BUILD_DIR)/diffusion_physis.mpi-cuda: $(PHYSIS_BUILD_DIR)/diffusion_physis.mpi-cuda.o \
	main_physis.o baseline.o diffusion.o $(PHYSIS_DIR)/lib/libphysis_rt_mpi_cuda.a
	$(MPICXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
