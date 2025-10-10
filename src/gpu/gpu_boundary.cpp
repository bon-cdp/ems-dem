#include "gpu/gpu_boundary.hpp"

namespace emsdem {

// Kernel to update triangle positions for translating boundaries
__global__ void updateTrianglePositionsKernel(Triangle* triangles, Vec3 displacement, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    triangles[idx].v0 += displacement;
    triangles[idx].v1 += displacement;
    triangles[idx].v2 += displacement;
    // Normal doesn't change for translation
}

// TODO: Add kernel for rotating boundaries (update vertices + normals)

void BoundaryDataGPU::updatePositions(real dt) {
    if (motion_type != BoundaryMotionType::TRANSLATING) return;
    if (velocity.lengthSquared() < 1e-12) return;

    Vec3 displacement = velocity * dt;
    int grid_size = getGridSize(num_triangles);

    hipLaunchKernelGGL(updateTrianglePositionsKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       triangles, displacement, num_triangles);
    HIP_CHECK(hipGetLastError());
}

} // namespace emsdem
