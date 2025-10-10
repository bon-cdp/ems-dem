#ifndef EMS_DEM_GPU_FORCES_HPP
#define EMS_DEM_GPU_FORCES_HPP

#include "gpu_common.hpp"
#include "gpu_particle.hpp"
#include "gpu_neighbor.hpp"
#include "gpu_boundary.hpp"
#include "core/material.hpp"

namespace emsdem {

/**
 * GPU material properties (simplified, stored as device constant)
 */
struct MaterialPropsGPU {
    real density;
    real youngs_modulus;
    real poisson_ratio;
    real friction_static;
    real friction_rolling;
    real restitution_coeff;

    HOST_DEVICE MaterialPropsGPU(const MaterialProperties& mat)
        : density(mat.density),
          youngs_modulus(mat.youngs_modulus),
          poisson_ratio(mat.poisson_ratio),
          friction_static(mat.friction_static),
          friction_rolling(mat.friction_rolling),
          restitution_coeff(mat.restitution_coeff)
    {}

    // Effective Young's modulus
    HOST_DEVICE static real effectiveYoungs(const MaterialPropsGPU& m1, const MaterialPropsGPU& m2) {
        real inv_E1 = (1.0 - m1.poisson_ratio * m1.poisson_ratio) / m1.youngs_modulus;
        real inv_E2 = (1.0 - m2.poisson_ratio * m2.poisson_ratio) / m2.youngs_modulus;
        return 1.0 / (inv_E1 + inv_E2);
    }

    HOST_DEVICE static real effectiveRadius(real r1, real r2) {
        return (r1 * r2) / (r1 + r2);
    }

    HOST_DEVICE static real effectiveMass(real m1, real m2) {
        return (m1 * m2) / (m1 + m2);
    }
};

/**
 * GPU contact force computation
 */
namespace gpu_forces {

    // Compute particle-particle contact forces using neighbor list
    void computeParticleParticleForces(ParticleDataGPU& particles,
                                       const NeighborListGPU& neighbors,
                                       const MaterialPropsGPU* materials_gpu);

    // Compute particle-wall contact forces with boundary motion support
    void computeParticleWallForces(ParticleDataGPU& particles,
                                   const BoundaryDataGPU* boundaries_gpu,
                                   int num_boundaries,
                                   const MaterialPropsGPU* materials_gpu);

} // namespace gpu_forces

} // namespace emsdem

#endif // EMS_DEM_GPU_FORCES_HPP
