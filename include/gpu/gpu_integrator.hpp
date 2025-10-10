#ifndef EMS_DEM_GPU_INTEGRATOR_HPP
#define EMS_DEM_GPU_INTEGRATOR_HPP

#include "gpu_common.hpp"
#include "gpu_particle.hpp"
#include "core/vec3.hpp"

namespace emsdem {

/**
 * GPU kernels for time integration
 */
namespace gpu_integrator {

    // Clear forces on GPU
    void clearForces(ParticleDataGPU& particles);

    // Apply gravity on GPU
    void applyGravity(ParticleDataGPU& particles, const Vec3& gravity);

    // Velocity-Verlet: half-step velocity update
    void velocityHalfStep(ParticleDataGPU& particles, real dt);

    // Velocity-Verlet: position update
    void positionUpdate(ParticleDataGPU& particles, real dt);

    // Velocity-Verlet: complete velocity update
    void velocityFullStep(ParticleDataGPU& particles, real dt);

    // Combined integration step
    void integrate(ParticleDataGPU& particles, const Vec3& gravity, real dt);

} // namespace gpu_integrator

} // namespace emsdem

#endif // EMS_DEM_GPU_INTEGRATOR_HPP
