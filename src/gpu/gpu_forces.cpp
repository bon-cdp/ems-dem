#include "gpu/gpu_forces.hpp"
#include <hip/hip_runtime.h>

namespace emsdem {

// Hertz-Mindlin contact force kernel
__global__ void computeContactForcesKernel(
    const Vec3* position, const Vec3* velocity, const Vec3* angular_velocity,
    Vec3* force, Vec3* torque,
    const real* radius, const real* mass, const real* inertia,
    const int* material_id, const bool* is_fixed,
    const int* neighbor_counts, const int* neighbor_list,
    const MaterialPropsGPU* materials,
    int max_neighbors, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (is_fixed[i]) return;

    Vec3 pos_i = position[i];
    Vec3 vel_i = velocity[i];
    Vec3 omega_i = angular_velocity[i];
    real r_i = radius[i];
    real m_i = mass[i];
    int mat_i = material_id[i];

    Vec3 total_force(0);
    Vec3 total_torque(0);

    int num_neighbors = neighbor_counts[i];

    // Loop over neighbors
    for (int k = 0; k < num_neighbors; k++) {
        int j = neighbor_list[i * max_neighbors + k];

        Vec3 pos_j = position[j];
        Vec3 vel_j = velocity[j];
        Vec3 omega_j = angular_velocity[j];
        real r_j = radius[j];
        real m_j = mass[j];
        int mat_j = material_id[j];

        // Relative position and distance
        Vec3 rij = pos_j - pos_i;
        real dist = rij.length();

        // Contact detection
        real overlap = r_i + r_j - dist;
        if (overlap <= 0) continue;

        // Normal direction
        Vec3 normal = rij / dist;

        // Relative velocity at contact point
        Vec3 vi_contact = vel_i + omega_i.cross(-normal * r_i);
        Vec3 vj_contact = vel_j + omega_j.cross(normal * r_j);
        Vec3 v_rel = vj_contact - vi_contact;

        // Normal and tangential components
        real v_n = v_rel.dot(normal);
        Vec3 v_t = v_rel - normal * v_n;

        // Material properties
        MaterialPropsGPU mat_pi = materials[mat_i];
        MaterialPropsGPU mat_pj = materials[mat_j];

        real E_eff = MaterialPropsGPU::effectiveYoungs(mat_pi, mat_pj);
        real R_eff = MaterialPropsGPU::effectiveRadius(r_i, r_j);
        real m_eff = MaterialPropsGPU::effectiveMass(m_i, m_j);

        real friction = 0.5 * (mat_pi.friction_static + mat_pj.friction_static);
        real e_rest = 0.5 * (mat_pi.restitution_coeff + mat_pj.restitution_coeff);

        // Normal force (Hertz contact + damping)
        real sqrt_overlap_R = sqrtf(overlap * R_eff);
        real k_n = (4.0 / 3.0) * E_eff * sqrt_overlap_R;

        real ln_e = logf(e_rest);
        real eta_n = -2.0 * sqrtf(m_eff * k_n) * ln_e / sqrtf(M_PI * M_PI + ln_e * ln_e);

        // Standard DEM damping (eta_n is negative, adds damping when approaching)
        real F_n_mag = k_n * overlap + eta_n * v_n;
        Vec3 F_normal = normal * F_n_mag;

        // Tangential force
        real v_t_mag = v_t.length();
        Vec3 F_tangential(0);

        if (v_t_mag > 1e-10) {
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

        // Accumulate forces (Newton's 3rd law)
        // Note: Each pair (i,j) is in the neighbor list, so we apply force to i
        // The neighbor list construction ensures j also gets the opposite force
        Vec3 F_total = F_normal + F_tangential;
        total_force += F_total;  // Force on particle i FROM particle j

        // Torque from tangential force
        Vec3 r_contact = -normal * r_i;
        total_torque += r_contact.cross(F_tangential);
    }

    // Atomic add to force and torque
    atomicAdd(&force[i].x, total_force.x);
    atomicAdd(&force[i].y, total_force.y);
    atomicAdd(&force[i].z, total_force.z);

    atomicAdd(&torque[i].x, total_torque.x);
    atomicAdd(&torque[i].y, total_torque.y);
    atomicAdd(&torque[i].z, total_torque.z);
}

namespace gpu_forces {

void computeParticleParticleForces(ParticleDataGPU& particles,
                                   const NeighborListGPU& neighbors,
                                   const MaterialPropsGPU* materials_gpu)
{
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    hipLaunchKernelGGL(computeContactForcesKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.velocity, particles.angular_velocity,
                       particles.force, particles.torque,
                       particles.radius, particles.mass, particles.inertia,
                       particles.material_id, particles.is_fixed,
                       neighbors.neighbor_counts, neighbors.neighbor_list,
                       materials_gpu,
                       neighbors.max_neighbors_per_particle, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

} // namespace gpu_forces

} // namespace emsdem
