#ifndef EMS_DEM_GPU_BOUNDARY_HPP
#define EMS_DEM_GPU_BOUNDARY_HPP

#include "gpu_common.hpp"
#include "core/vec3.hpp"
#include "geometry/triangle.hpp"
#include <vector>

namespace emsdem {

/**
 * GPU boundary motion types
 *
 * 1. STATIC: No motion
 * 2. CONVEYOR: Fixed geometry, applies tangential velocity to particles
 * 3. TRANSLATING: Linear motion (velocity vector)
 * 4. ROTATING: Rotation about axis (angular velocity vector + center)
 *
 * TODO: Validate all motion types against physical experiments
 */
enum class BoundaryMotionType {
    STATIC = 0,
    CONVEYOR = 1,
    TRANSLATING = 2,
    ROTATING = 3
};

/**
 * Boundary data for GPU (per-triangle or per-mesh)
 */
struct BoundaryDataGPU {
    Triangle* triangles;           // Device pointer to triangle array
    int num_triangles;
    int material_id;

    // Motion parameters
    BoundaryMotionType motion_type;
    Vec3 velocity;                 // For CONVEYOR and TRANSLATING
    Vec3 angular_velocity;         // For ROTATING (axis * angular_speed)
    Vec3 rotation_center;          // For ROTATING

    BoundaryDataGPU()
        : triangles(nullptr), num_triangles(0), material_id(0),
          motion_type(BoundaryMotionType::STATIC),
          velocity(0), angular_velocity(0), rotation_center(0)
    {}

    void allocate(int n) {
        num_triangles = n;
        HIP_CHECK(hipMalloc(&triangles, n * sizeof(Triangle)));
    }

    void free() {
        if (triangles) HIP_CHECK(hipFree(triangles));
        triangles = nullptr;
    }

    // Copy triangles from host
    void copyTrianglesFromHost(const std::vector<Triangle>& host_triangles) {
        HIP_CHECK(hipMemcpy(triangles, host_triangles.data(),
                           host_triangles.size() * sizeof(Triangle),
                           hipMemcpyHostToDevice));
    }

    // Update triangle positions for translating boundaries
    void updatePositions(real dt);
};

/**
 * Compute effective velocity at a point on the boundary
 * Used for contact force calculations
 */
__device__ inline Vec3 computeBoundaryVelocity(
    const BoundaryDataGPU& boundary,
    const Vec3& contact_point)
{
    switch (boundary.motion_type) {
        case BoundaryMotionType::STATIC:
            return Vec3(0, 0, 0);

        case BoundaryMotionType::CONVEYOR:
            // Fixed geometry, but applies tangential velocity to particles
            return boundary.velocity;

        case BoundaryMotionType::TRANSLATING:
            // Entire boundary moves with constant velocity
            return boundary.velocity;

        case BoundaryMotionType::ROTATING:
            // v = omega x r (where r is from rotation center to contact point)
            {
                Vec3 r = contact_point - boundary.rotation_center;
                return boundary.angular_velocity.cross(r);
            }

        default:
            return Vec3(0, 0, 0);
    }
}

} // namespace emsdem

#endif // EMS_DEM_GPU_BOUNDARY_HPP
