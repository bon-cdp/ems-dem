#include "gpu/gpu_neighbor.hpp"

namespace emsdem {

// Simple brute-force neighbor search (O(N^2) but no external dependencies)
// TODO: Implement optimized spatial grid later with custom sorting
__global__ void buildNeighborListSimpleKernel(const Vec3* position, const real* radius,
                                               int* neighbor_counts, int* neighbor_list,
                                               int max_neighbors, real skin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int count = 0;
    Vec3 pos_i = position[i];
    real r_i = radius[i];

    // Check all other particles
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        Vec3 pos_j = position[j];
        real r_j = radius[j];

        Vec3 rij = pos_j - pos_i;
        real dist_sq = rij.lengthSquared();
        real cutoff = r_i + r_j + skin;
        real cutoff_sq = cutoff * cutoff;

        if (dist_sq < cutoff_sq) {
            if (count < max_neighbors) {
                neighbor_list[i * max_neighbors + count] = j;
                count++;
            }
        }
    }

    neighbor_counts[i] = count;
}

void NeighborListGPU::build(const ParticleDataGPU& particles, real cutoff_radius) {
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    // Clear neighbor counts
    HIP_CHECK(hipMemset(neighbor_counts, 0, n * sizeof(int)));

    // Build neighbor list (simple brute force for now)
    // TODO: Replace with optimized spatial grid once we have sorting working
    hipLaunchKernelGGL(buildNeighborListSimpleKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.radius,
                       neighbor_counts, neighbor_list,
                       max_neighbors_per_particle, cutoff_radius, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

} // namespace emsdem
