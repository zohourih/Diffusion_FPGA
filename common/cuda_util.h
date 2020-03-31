#ifndef COMMON_CUDA_UTIL_H_
#define COMMON_CUDA_UTIL_H_

#define FORCE_CHECK_CUDA(call)  do {                                    \
    const cudaError_t status = call;                                    \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

#ifdef DEBUG
#define CHECK_CUDA(call)  FORCE_CHECK_CUDA(call)
#else
#define CHECK_CUDA(call)  (call)
#endif


#endif // COMMON_CUDA_UTIL_H_
