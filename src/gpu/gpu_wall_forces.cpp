#include "gpu/gpu_forces.hpp"

namespace emsdem {

// Particle-wall contact force kernel with boundary motion support
__global__ void computeParticleWallForcesKernel(
    const Vec3* position, const Vec3* velocity, const Vec3* angular_velocity,
    Vec3* force, Vec3* torque,
    const real* radius, const real* mass,
    const int* material_id, const bool* is_fixed,
    const BoundaryDataGPU* boundaries,
    const MaterialPropsGPU* materials,
    int num_boundaries, int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    if (is_fixed[i]) return;

    Vec3 pos_i = position[i];
    Vec3 vel_i = velocity[i];
    Vec3 omega_i = angular_velocity[i];
    real r_i = radius[i];
    real m_i = mass[i];
    int mat_i = material_id[i];

    Vec3 total_force(0, 0, 0);
    Vec3 total_torque(0, 0, 0);

    // Loop over all boundaries
    for (int b = 0; b < num_boundaries; b++) {
        const BoundaryDataGPU& boundary = boundaries[b];

        // Loop over all triangles in this boundary
        for (int t = 0; t < boundary.num_triangles; t++) {
            const Triangle& tri = boundary.triangles[t];

            // Find closest point on triangle to particle center
            Vec3 closest = tri.closestPoint(pos_i);
            Vec3 to_particle = pos_i - closest;
            real dist = to_particle.length();

            // Contact detection
            real overlap = r_i - dist;
            if (overlap <= 0) continue;  // No contact

            // Normal direction (from wall to particle)
            // This is correct: normal should point from closest point toward particle center
            // Force will push particle away from wall in this direction
            Vec3 normal = to_particle / dist;

            // Compute boundary velocity at contact point
            // This handles CONVEYOR, TRANSLATING, and ROTATING boundaries
            Vec3 wall_vel = computeBoundaryVelocity(boundary, closest);

            // Relative velocity at contact point
            Vec3 v_contact = vel_i + omega_i.cross(-normal * r_i);
            Vec3 v_rel = v_contact - wall_vel;

            // Normal and tangential components
            real v_n = v_rel.dot(normal);
            Vec3 v_t = v_rel - normal * v_n;

            // Material properties
            MaterialPropsGPU mat_p = materials[mat_i];
            MaterialPropsGPU mat_w = materials[tri.material_id];

            real E_eff = MaterialPropsGPU::effectiveYoungs(mat_p, mat_w);
            real R_eff = r_i;  // For particle-plane, R_eff = particle radius

            real friction = 0.5f * (mat_p.friction_static + mat_w.friction_static);
            real e_rest = 0.5f * (mat_p.restitution_coeff + mat_w.restitution_coeff);

            // Normal force (Hertz contact + damping)
            real sqrt_overlap_R = sqrtf(overlap * R_eff);
            real k_n = (4.0f / 3.0f) * E_eff * sqrt_overlap_R;

            real ln_e = logf(e_rest);
            real eta_n = -2.0f * sqrtf(m_i * k_n) * ln_e / sqrtf(M_PI * M_PI + ln_e * ln_e);

            // Standard DEM damping (eta_n is negative, adds damping when approaching)
            real F_n_mag = k_n * overlap + eta_n * v_n;
            Vec3 F_normal = normal * F_n_mag;

            // Tangential force
            real v_t_mag = v_t.length();
            Vec3 F_tangential(0, 0, 0);

            if (v_t_mag > 1e-10f) {
                Vec3 t_dir = v_t / v_t_mag;
                real eta_t = eta_n;
                real F_t_mag = eta_t * v_t_mag;

                // Coulomb friction limit
                real F_t_max = friction * fabsf(F_n_mag);
                if (F_t_mag > F_t_max) {
                    F_t_mag = F_t_max;
                }

                F_tangential = -t_dir * F_t_mag;
            }

            // Accumulate forces
            Vec3 F_total = F_normal + F_tangential;
            total_force += F_total;

            // Torque from tangential force
            Vec3 r_contact = -normal * r_i;
            total_torque += r_contact.cross(F_tangential);
        }
    }

    // Atomic add to particle forces
    atomicAdd(&force[i].x, total_force.x);
    atomicAdd(&force[i].y, total_force.y);
    atomicAdd(&force[i].z, total_force.z);

    atomicAdd(&torque[i].x, total_torque.x);
    atomicAdd(&torque[i].y, total_torque.y);
    atomicAdd(&torque[i].z, total_torque.z);
}

namespace gpu_forces {

void computeParticleWallForces(ParticleDataGPU& particles,
                               const BoundaryDataGPU* boundaries_gpu,
                               int num_boundaries,
                               const MaterialPropsGPU* materials_gpu)
{
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    hipLaunchKernelGGL(computeParticleWallForcesKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.velocity, particles.angular_velocity,
                       particles.force, particles.torque,
                       particles.radius, particles.mass,
                       particles.material_id, particles.is_fixed,
                       boundaries_gpu, materials_gpu,
                       num_boundaries, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

} // namespace gpu_forces

} // namespace emsdem
