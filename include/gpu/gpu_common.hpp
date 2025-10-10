#ifndef EMS_DEM_GPU_COMMON_HPP
#define EMS_DEM_GPU_COMMON_HPP

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

namespace emsdem {

// HIP error checking macro
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Default block size for kernels
constexpr int BLOCK_SIZE = 256;

// Get optimal grid size for N elements
inline int getGridSize(int n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// Device function attribute
#define DEVICE __device__
#define HOST __host__
#define GLOBAL __global__
#define HOST_DEVICE __host__ __device__

} // namespace emsdem

#endif // EMS_DEM_GPU_COMMON_HPP
