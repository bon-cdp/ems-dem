#include "gpu/gpu_neighbor.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

namespace emsdem {

// Compute grid cell hash for a position
__device__ inline int computeCellHash(const Vec3& pos, real cell_size, int grid_dim) {
    int x = (int)floorf(pos.x / cell_size);
    int y = (int)floorf(pos.y / cell_size);
    int z = (int)floorf(pos.z / cell_size);

    // Clamp to grid bounds
    x = max(0, min(x, grid_dim - 1));
    y = max(0, min(y, grid_dim - 1));
    z = max(0, min(z, grid_dim - 1));

    return (z * grid_dim + y) * grid_dim + x;
}

// Kernel to compute particle cell hashes
__global__ void computeHashesKernel(const Vec3* position, int* hash, int* index,
                                    real cell_size, int grid_dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    hash[i] = computeCellHash(position[i], cell_size, grid_dim);
    index[i] = i;
}

// Kernel to find cell start/end indices after sorting
__global__ void findCellBoundsKernel(const int* hash, int* cell_start, int* cell_end, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int hash_i = hash[i];

    // Check if this is the start of a new cell
    if (i == 0 || hash[i-1] != hash_i) {
        cell_start[hash_i] = i;
    }

    // Check if this is the end of a cell
    if (i == n-1 || hash[i+1] != hash_i) {
        cell_end[hash_i] = i;
    }
}

// Kernel to build neighbor list using spatial grid
__global__ void buildNeighborListGridKernel(
    const Vec3* position, const real* radius, const int* sorted_index,
    const int* cell_start, const int* cell_end,
    int* neighbor_counts, int* neighbor_list,
    int max_neighbors, real cell_size, real skin, int grid_dim, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int i = sorted_index[idx];  // Original particle index

    Vec3 pos_i = position[i];
    real r_i = radius[i];

    int count = 0;

    // Compute grid cell for particle i
    int cx = (int)floorf(pos_i.x / cell_size);
    int cy = (int)floorf(pos_i.y / cell_size);
    int cz = (int)floorf(pos_i.z / cell_size);

    // Check neighboring cells (27 cells including self)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;

                // Check bounds
                if (nx < 0 || nx >= grid_dim ||
                    ny < 0 || ny >= grid_dim ||
                    nz < 0 || nz >= grid_dim) {
                    continue;
                }

                int cell_hash = (nz * grid_dim + ny) * grid_dim + nx;
                int start = cell_start[cell_hash];
                int end = cell_end[cell_hash];

                // Check all particles in this cell
                for (int k = start; k <= end && k >= 0; k++) {
                    int j = sorted_index[k];

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
    int grid_size = getGridSize(n);

    // Spatial grid parameters
    real cell_size = cutoff_radius * 2.0;  // Cell size should be >= cutoff
    int grid_dim = 64;  // 64x64x64 grid (can be tuned)

    // Allocate temporary arrays for spatial hashing
    thrust::device_vector<int> d_hash(n);
    thrust::device_vector<int> d_index(n);
    thrust::device_vector<int> d_cell_start(grid_dim * grid_dim * grid_dim, -1);
    thrust::device_vector<int> d_cell_end(grid_dim * grid_dim * grid_dim, -1);

    // Compute hashes
    hipLaunchKernelGGL(computeHashesKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position,
                       thrust::raw_pointer_cast(d_hash.data()),
                       thrust::raw_pointer_cast(d_index.data()),
                       cell_size, grid_dim, n);
    HIP_CHECK(hipGetLastError());

    // Sort by hash
    thrust::sort_by_key(d_hash.begin(), d_hash.end(), d_index.begin());

    // Find cell bounds
    hipLaunchKernelGGL(findCellBoundsKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       thrust::raw_pointer_cast(d_hash.data()),
                       thrust::raw_pointer_cast(d_cell_start.data()),
                       thrust::raw_pointer_cast(d_cell_end.data()), n);
    HIP_CHECK(hipGetLastError());

    // Clear neighbor counts
    HIP_CHECK(hipMemset(neighbor_counts, 0, n * sizeof(int)));

    // Build neighbor list using grid
    hipLaunchKernelGGL(buildNeighborListGridKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.radius,
                       thrust::raw_pointer_cast(d_index.data()),
                       thrust::raw_pointer_cast(d_cell_start.data()),
                       thrust::raw_pointer_cast(d_cell_end.data()),
                       neighbor_counts, neighbor_list,
                       max_neighbors_per_particle, cell_size, cutoff_radius,
                       grid_dim, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

} // namespace emsdem
