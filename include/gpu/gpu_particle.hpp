#ifndef EMS_DEM_GPU_PARTICLE_HPP
#define EMS_DEM_GPU_PARTICLE_HPP

#include "core/vec3.hpp"
#include "gpu_common.hpp"

namespace emsdem {

// Forward declaration
class Domain;

/**
 * GPU-friendly particle structure (Structure of Arrays)
 * Better memory coalescing than Array of Structures
 */
struct ParticleDataGPU {
    // Arrays stored on GPU
    Vec3* position;           // Device pointer
    Vec3* velocity;
    Vec3* angular_velocity;
    Vec3* force;
    Vec3* torque;

    real* radius;
    real* mass;
    real* inertia;

    int* material_id;
    int* id;
    bool* is_fixed;

    int num_particles;

    // Constructor
    ParticleDataGPU() : position(nullptr), velocity(nullptr), angular_velocity(nullptr),
                        force(nullptr), torque(nullptr), radius(nullptr), mass(nullptr),
                        inertia(nullptr), material_id(nullptr), id(nullptr),
                        is_fixed(nullptr), num_particles(0) {}

    // Allocate GPU memory
    void allocate(int n) {
        num_particles = n;

        HIP_CHECK(hipMalloc(&position, n * sizeof(Vec3)));
        HIP_CHECK(hipMalloc(&velocity, n * sizeof(Vec3)));
        HIP_CHECK(hipMalloc(&angular_velocity, n * sizeof(Vec3)));
        HIP_CHECK(hipMalloc(&force, n * sizeof(Vec3)));
        HIP_CHECK(hipMalloc(&torque, n * sizeof(Vec3)));

        HIP_CHECK(hipMalloc(&radius, n * sizeof(real)));
        HIP_CHECK(hipMalloc(&mass, n * sizeof(real)));
        HIP_CHECK(hipMalloc(&inertia, n * sizeof(real)));

        HIP_CHECK(hipMalloc(&material_id, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&id, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&is_fixed, n * sizeof(bool)));
    }

    // Free GPU memory
    void free() {
        if (position) HIP_CHECK(hipFree(position));
        if (velocity) HIP_CHECK(hipFree(velocity));
        if (angular_velocity) HIP_CHECK(hipFree(angular_velocity));
        if (force) HIP_CHECK(hipFree(force));
        if (torque) HIP_CHECK(hipFree(torque));
        if (radius) HIP_CHECK(hipFree(radius));
        if (mass) HIP_CHECK(hipFree(mass));
        if (inertia) HIP_CHECK(hipFree(inertia));
        if (material_id) HIP_CHECK(hipFree(material_id));
        if (id) HIP_CHECK(hipFree(id));
        if (is_fixed) HIP_CHECK(hipFree(is_fixed));

        position = nullptr;
        num_particles = 0;
    }

    // Copy from host domain
    void copyFromHost(const Domain& domain);

    // Copy to host domain
    void copyToHost(Domain& domain) const;
};

} // namespace emsdem

#endif // EMS_DEM_GPU_PARTICLE_HPP
