#include "gpu/gpu_particle.hpp"
#include "core/domain.hpp"
#include <vector>

namespace emsdem {

void ParticleDataGPU::copyFromHost(const Domain& domain) {
    int n = static_cast<int>(domain.particles.size());

    // Prepare host arrays
    std::vector<Vec3> h_pos(n), h_vel(n), h_omega(n), h_force(n), h_torque(n);
    std::vector<real> h_radius(n), h_mass(n), h_inertia(n);
    std::vector<int> h_mat_id(n), h_id(n);
    std::vector<bool> h_fixed(n);

    for (int i = 0; i < n; i++) {
        const auto& p = domain.particles[i];
        h_pos[i] = p.position;
        h_vel[i] = p.velocity;
        h_omega[i] = p.angular_velocity;
        h_force[i] = p.force;
        h_torque[i] = p.torque;
        h_radius[i] = p.radius;
        h_mass[i] = p.mass;
        h_inertia[i] = p.inertia;
        h_mat_id[i] = p.material_id;
        h_id[i] = p.id;
        h_fixed[i] = p.is_fixed;
    }

    // Copy to GPU
    HIP_CHECK(hipMemcpy(position, h_pos.data(), n * sizeof(Vec3), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(velocity, h_vel.data(), n * sizeof(Vec3), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(angular_velocity, h_omega.data(), n * sizeof(Vec3), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(force, h_force.data(), n * sizeof(Vec3), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(torque, h_torque.data(), n * sizeof(Vec3), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(radius, h_radius.data(), n * sizeof(real), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(mass, h_mass.data(), n * sizeof(real), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(inertia, h_inertia.data(), n * sizeof(real), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(material_id, h_mat_id.data(), n * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(id, h_id.data(), n * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(is_fixed, h_fixed.data(), n * sizeof(bool), hipMemcpyHostToDevice));
}

void ParticleDataGPU::copyToHost(Domain& domain) const {
    int n = num_particles;

    // Prepare host arrays
    std::vector<Vec3> h_pos(n), h_vel(n), h_omega(n), h_force(n), h_torque(n);

    // Copy from GPU
    HIP_CHECK(hipMemcpy(h_pos.data(), position, n * sizeof(Vec3), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_vel.data(), velocity, n * sizeof(Vec3), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_omega.data(), angular_velocity, n * sizeof(Vec3), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_force.data(), force, n * sizeof(Vec3), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_torque.data(), torque, n * sizeof(Vec3), hipMemcpyDeviceToHost));

    // Update domain particles
    for (int i = 0; i < n; i++) {
        domain.particles[i].position = h_pos[i];
        domain.particles[i].velocity = h_vel[i];
        domain.particles[i].angular_velocity = h_omega[i];
        domain.particles[i].force = h_force[i];
        domain.particles[i].torque = h_torque[i];
    }
}

} // namespace emsdem
