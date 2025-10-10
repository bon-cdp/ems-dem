#ifndef EMS_DEM_GPU_NEIGHBOR_HPP
#define EMS_DEM_GPU_NEIGHBOR_HPP

#include "gpu_common.hpp"
#include "gpu_particle.hpp"
#include "core/vec3.hpp"

namespace emsdem {

/**
 * GPU neighbor list using spatial hashing
 * Simplified approach: each particle checks all other particles (brute force for first version)
 * TODO: Implement spatial grid in future for better performance
 */
struct NeighborListGPU {
    int* neighbor_counts;      // Number of neighbors for each particle
    int* neighbor_list;        // Flattened neighbor list
    int max_neighbors_per_particle;
    int num_particles;

    NeighborListGPU() : neighbor_counts(nullptr), neighbor_list(nullptr),
                       max_neighbors_per_particle(50), num_particles(0) {}

    void allocate(int n, int max_neighbors = 50) {
        num_particles = n;
        max_neighbors_per_particle = max_neighbors;

        HIP_CHECK(hipMalloc(&neighbor_counts, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&neighbor_list, n * max_neighbors * sizeof(int)));
    }

    void free() {
        if (neighbor_counts) HIP_CHECK(hipFree(neighbor_counts));
        if (neighbor_list) HIP_CHECK(hipFree(neighbor_list));
        neighbor_counts = nullptr;
        neighbor_list = nullptr;
    }

    // Build neighbor list (brute force)
    void build(const ParticleDataGPU& particles, real cutoff_radius);
};

} // namespace emsdem

#endif // EMS_DEM_GPU_NEIGHBOR_HPP
