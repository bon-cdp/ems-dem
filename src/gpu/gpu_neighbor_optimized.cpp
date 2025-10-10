#include "gpu/gpu_neighbor.hpp"

namespace emsdem {

/**
 * Optimized spatial grid neighbor search - O(N) complexity
 *
 * Algorithm:
 * 1. Hash particles to grid cells
 * 2. Count particles per cell (atomic add)
 * 3. Compute cell start offsets (exclusive scan/prefix sum)
 * 4. Fill cell arrays
 * 5. For each particle, check 27 neighboring cells
 *
 * TODO: Replace atomic operations with faster binning strategy
 */

// Compute grid cell index for a position
__device__ inline int getCellIndex(const Vec3& pos, real cell_size, int3 grid_dims, Vec3 grid_min) {
    int ix = (int)floorf((pos.x - grid_min.x) / cell_size);
    int iy = (int)floorf((pos.y - grid_min.y) / cell_size);
    int iz = (int)floorf((pos.z - grid_min.z) / cell_size);

    // Clamp to grid bounds
    ix = max(0, min(ix, grid_dims.x - 1));
    iy = max(0, min(iy, grid_dims.y - 1));
    iz = max(0, min(iz, grid_dims.z - 1));

    return (iz * grid_dims.y + iy) * grid_dims.x + ix;
}

// Kernel 1: Count particles per cell
__global__ void countParticlesPerCellKernel(
    const Vec3* position, int* cell_counts,
    real cell_size, int3 grid_dims, Vec3 grid_min, int num_cells, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell_idx = getCellIndex(position[i], cell_size, grid_dims, grid_min);
    if (cell_idx >= 0 && cell_idx < num_cells) {
        atomicAdd(&cell_counts[cell_idx], 1);
    }
}

// Kernel 2: Compute cell start offsets (exclusive scan on CPU for simplicity)
// This could be replaced with GPU parallel scan for better performance

// Kernel 3: Fill cell particle lists
__global__ void fillCellListsKernel(
    const Vec3* position, int* cell_particle_list, int* cell_counts, const int* cell_starts,
    real cell_size, int3 grid_dims, Vec3 grid_min, int num_cells, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell_idx = getCellIndex(position[i], cell_size, grid_dims, grid_min);
    if (cell_idx >= 0 && cell_idx < num_cells) {
        int offset = atomicAdd(&cell_counts[cell_idx], 1);
        int write_idx = cell_starts[cell_idx] + offset;
        cell_particle_list[write_idx] = i;
    }
}

// Kernel 4: Build neighbor lists using spatial grid
__global__ void buildNeighborListGridKernel(
    const Vec3* position, const real* radius,
    int* neighbor_counts, int* neighbor_list,
    const int* cell_particle_list, const int* cell_starts, const int* cell_ends,
    int max_neighbors, real cell_size, real skin,
    int3 grid_dims, Vec3 grid_min, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 pos_i = position[i];
    real r_i = radius[i];
    int count = 0;

    // Get particle's cell
    int ix = (int)floorf((pos_i.x - grid_min.x) / cell_size);
    int iy = (int)floorf((pos_i.y - grid_min.y) / cell_size);
    int iz = (int)floorf((pos_i.z - grid_min.z) / cell_size);

    // Check 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;

                // Check bounds
                if (nx < 0 || nx >= grid_dims.x ||
                    ny < 0 || ny >= grid_dims.y ||
                    nz < 0 || nz >= grid_dims.z) {
                    continue;
                }

                int cell_idx = (nz * grid_dims.y + ny) * grid_dims.x + nx;
                int start = cell_starts[cell_idx];
                int end = cell_ends[cell_idx];

                // Check all particles in this cell
                for (int k = start; k <= end && k >= 0; k++) {
                    int j = cell_particle_list[k];
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
            }
        }
    }

    neighbor_counts[i] = count;
}

void NeighborListGPU::build(const ParticleDataGPU& particles, real cutoff_radius) {
    int n = particles.num_particles;

    // For small particle counts, use brute force (current simple version)
    // For N > 1000, use spatial grid
    const int SPATIAL_GRID_THRESHOLD = 1000;

    if (n < SPATIAL_GRID_THRESHOLD) {
        // Use existing brute-force kernel (already in gpu_neighbor.cpp)
        int grid_size = getGridSize(n);

        // This is the simple brute-force kernel we already have
        extern __global__ void buildNeighborListSimpleKernel(
            const Vec3*, const real*, int*, int*, int, real, int);

        HIP_CHECK(hipMemset(neighbor_counts, 0, n * sizeof(int)));

        hipLaunchKernelGGL(buildNeighborListSimpleKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                           particles.position, particles.radius,
                           neighbor_counts, neighbor_list,
                           max_neighbors_per_particle, cutoff_radius, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        return;
    }

    // TODO: Implement optimized spatial grid for N > 1000
    // For now, fall back to brute force
    // This will be optimized in a future update

    int grid_size = getGridSize(n);
    extern __global__ void buildNeighborListSimpleKernel(
        const Vec3*, const real*, int*, int*, int, real, int);

    HIP_CHECK(hipMemset(neighbor_counts, 0, n * sizeof(int)));

    hipLaunchKernelGGL(buildNeighborListSimpleKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.radius,
                       neighbor_counts, neighbor_list,
                       max_neighbors_per_particle, cutoff_radius, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

} // namespace emsdem
