#include "core/domain.hpp"
#include "core/particle.hpp"
#include "core/material.hpp"
#include "geometry/mesh.hpp"
#include "geometry/primitives.hpp"
#include "gpu/gpu_particle.hpp"
#include "gpu/gpu_neighbor.hpp"
#include "gpu/gpu_forces.hpp"
#include "gpu/gpu_integrator.hpp"
#include "io/vtk_writer.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace emsdem;

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "   EMS-DEM: GPU Angle of Repose Test\n";
    std::cout << "===========================================\n\n";

    // ========== Simulation Parameters ==========
    const real particle_radius = 0.005;  // 5 mm particles
    const real particle_density = 2500;  // Glass beads: 2500 kg/m^3

    const real box_size = 0.2;           // 20 cm box
    const real fill_height = 0.1;        // Fill to 10 cm

    const real dt_safety = 0.2;
    const int  output_interval = 50;
    const real sim_time = 2.0;           // 2 seconds total

    // ========== Setup Domain ==========
    Domain domain;
    domain.gravity = Vec3(0, 0, -9.81);
    domain.neighbor_skin = 0.5 * particle_radius;

    // Glass beads material (default material 0)
    std::cout << "Using glass beads material\n";
    std::cout << "  Density: " << domain.materials[0].density << " kg/m^3\n";
    std::cout << "  Young's modulus: " << domain.materials[0].youngs_modulus / 1e9 << " GPa\n";
    std::cout << "  Friction: " << domain.materials[0].friction_static << "\n\n";

    // ========== Create Particles ==========
    std::cout << "Creating particles...\n";

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<real> dist_x(-box_size/2 + 2*particle_radius,
                                                  box_size/2 - 2*particle_radius);
    std::uniform_real_distribution<real> dist_y(-box_size/2 + 2*particle_radius,
                                                  box_size/2 - 2*particle_radius);

    int num_particles = 0;
    real z_current = particle_radius * 2;

    while (z_current < fill_height) {
        for (int attempt = 0; attempt < 50; attempt++) {
            real x = dist_x(gen);
            real y = dist_y(gen);
            Vec3 pos(x, y, z_current);

            // Check overlap with existing particles
            bool overlap = false;
            for (const auto& p : domain.particles) {
                real dist = (p.position - pos).length();
                if (dist < 2.0 * particle_radius * 0.95) {
                    overlap = true;
                    break;
                }
            }

            if (!overlap) {
                Particle p(pos, particle_radius, particle_density, 0);
                domain.addParticle(p);
                num_particles++;
                break;
            }
        }
        z_current += 1.5 * particle_radius;
    }

    std::cout << "  Created " << num_particles << " particles\n\n";

    // ========== Calculate Timestep ==========
    real dt_critical = domain.calculateCriticalTimestep();
    real dt = dt_safety * dt_critical;

    std::cout << "Timestep:\n";
    std::cout << "  Critical dt: " << dt_critical << " s\n";
    std::cout << "  Actual dt:   " << dt << " s\n";
    std::cout << "  Steps needed: " << (int)(sim_time / dt) << "\n\n";

    // ========== Setup GPU ==========
    std::cout << "Allocating GPU memory...\n";

    ParticleDataGPU particles_gpu;
    particles_gpu.allocate(num_particles);
    particles_gpu.copyFromHost(domain);

    NeighborListGPU neighbors_gpu;
    neighbors_gpu.allocate(num_particles, 50);  // Max 50 neighbors per particle

    // Materials on GPU
    MaterialPropsGPU mat_gpu(domain.materials[0]);
    MaterialPropsGPU* d_materials;
    HIP_CHECK(hipMalloc(&d_materials, sizeof(MaterialPropsGPU)));
    HIP_CHECK(hipMemcpy(d_materials, &mat_gpu, sizeof(MaterialPropsGPU), hipMemcpyHostToDevice));

    std::cout << "  GPU memory allocated for " << num_particles << " particles\n\n";

    // ========== Simulation Loop ==========
    real current_time = 0;
    int step = 0;
    int output_count = 0;
    std::vector<std::pair<int, real>> vtk_timesteps;

    std::cout << "Starting GPU simulation...\n";
    std::cout << "===========================================\n\n";

    // Initial output
    particles_gpu.copyToHost(domain);
    VTKWriter::writeParticles(domain, "results/particles", output_count);
    vtk_timesteps.push_back({output_count, current_time});
    output_count++;

    while (current_time < sim_time) {
        // Clear forces
        gpu_integrator::clearForces(particles_gpu);

        // Apply gravity
        gpu_integrator::applyGravity(particles_gpu, domain.gravity);

        // Build neighbor list
        real cutoff = 2.0 * particle_radius + domain.neighbor_skin;
        neighbors_gpu.build(particles_gpu, cutoff);

        // Compute contact forces
        gpu_forces::computeParticleParticleForces(particles_gpu, neighbors_gpu, d_materials);

        // Integrate (half-step)
        gpu_integrator::velocityHalfStep(particles_gpu, dt);
        gpu_integrator::positionUpdate(particles_gpu, dt);

        // Recompute forces at new positions
        gpu_integrator::clearForces(particles_gpu);
        gpu_integrator::applyGravity(particles_gpu, domain.gravity);
        neighbors_gpu.build(particles_gpu, cutoff);
        gpu_forces::computeParticleParticleForces(particles_gpu, neighbors_gpu, d_materials);

        // Complete velocity update
        gpu_integrator::velocityFullStep(particles_gpu, dt);

        // Output
        if (step % output_interval == 0) {
            particles_gpu.copyToHost(domain);
            real ke = domain.totalKineticEnergy();

            std::cout << "  Step " << step << ", t = " << current_time
                     << " s, KE = " << ke << " J\n";

            VTKWriter::writeParticles(domain, "results/particles", output_count);
            vtk_timesteps.push_back({output_count, current_time});
            output_count++;
        }

        current_time += dt;
        step++;
    }

    // Final output
    particles_gpu.copyToHost(domain);
    VTKWriter::writeParticles(domain, "results/particles", output_count);
    vtk_timesteps.push_back({output_count, current_time});
    VTKWriter::writeCollection("results/particles", vtk_timesteps);

    std::cout << "\n===========================================\n";
    std::cout << "Simulation complete!\n";
    std::cout << "  Total steps: " << step << "\n";
    std::cout << "  Final time:  " << current_time << " s\n";
    std::cout << "  VTK files:   results/particles_*.vtk\n";
    std::cout << "  Collection:  results/particles.pvd\n";
    std::cout << "===========================================\n";

    // Cleanup GPU
    particles_gpu.free();
    neighbors_gpu.free();
    HIP_CHECK(hipFree(d_materials));

    return 0;
}
