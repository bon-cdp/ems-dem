#include "core/domain.hpp"
#include "core/particle.hpp"
#include "core/material.hpp"
#include "geometry/mesh.hpp"
#include "geometry/primitives.hpp"
#include "geometry/triangle.hpp"
#include "gpu/gpu_particle.hpp"
#include "gpu/gpu_neighbor.hpp"
#include "gpu/gpu_forces.hpp"
#include "gpu/gpu_integrator.hpp"
#include "gpu/gpu_boundary.hpp"
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

    const real box_size = 0.3;           // 30 cm box
    const real fill_height = 0.25;       // Fill to 25 cm (target: 1000+ particles)

    const real dt_target = 1e-4;         // Target timestep: 1e-4 s (user requirement)
    const int  output_interval = 100;    // Output every 100 steps (0.01s intervals)
    const real sim_time = 10.0;          // Full simulation: 10 seconds

    // ========== Setup Domain ==========
    Domain domain;
    domain.gravity = Vec3(0, -9.81, 0);  // Y-up coordinate system
    domain.neighbor_skin = 0.5 * particle_radius;

    // Glass beads material (default material 0)
    std::cout << "Using DEM-calibrated glass beads material\n";
    std::cout << "  Density: " << domain.materials[0].density << " kg/m^3\n";
    std::cout << "  Young's modulus: " << domain.materials[0].youngs_modulus / 1e6 << " MPa (effective)\n";
    std::cout << "  Friction: " << domain.materials[0].friction_static << "\n";
    std::cout << "  Restitution: " << domain.materials[0].restitution_coeff << "\n\n";

    // ========== Load Geometry ==========
    std::cout << "Loading geometry from STL files...\n";

    // Load box mesh
    Mesh box_mesh(0);
    if (!box_mesh.loadSTL("pipe - Box^Assem1-1.STL")) {
        std::cerr << "ERROR: Failed to load box STL file\n";
        return 1;
    }

    // Load pipe mesh
    Mesh pipe_mesh(0);
    if (!pipe_mesh.loadSTL("pipe - Pipe^Assem1-1.STL")) {
        std::cerr << "ERROR: Failed to load pipe STL file\n";
        return 1;
    }

    std::cout << "  Box: " << box_mesh.numTriangles() << " triangles\n";
    std::cout << "  Pipe: " << pipe_mesh.numTriangles() << " triangles\n\n";

    // ========== Create Particles ==========
    std::cout << "Creating particles in square injection region...\n";

    // Square injection region (centered above origin)
    const real inject_width = 0.08;  // 8cm x 8cm square region
    const real inject_x_min = -inject_width / 2;
    const real inject_x_max = inject_width / 2;
    const real inject_z_min = -inject_width / 2;
    const real inject_z_max = inject_width / 2;

    // Grid-based packing with slight randomization
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<real> jitter(-0.3 * particle_radius, 0.3 * particle_radius);

    int num_particles = 0;
    real spacing = 2.05 * particle_radius;  // Slight gap for settling

    // Generate particles in regular grid within square region
    for (real y = particle_radius * 2; y < fill_height; y += spacing) {
        for (real x = inject_x_min + particle_radius; x < inject_x_max; x += spacing) {
            for (real z = inject_z_min + particle_radius; z < inject_z_max; z += spacing) {
                // Add small random jitter to avoid perfect alignment
                Vec3 pos(x + jitter(gen), y + jitter(gen), z + jitter(gen));

                // Bounds check (ensure still in injection region)
                if (pos.x < inject_x_min + particle_radius || pos.x > inject_x_max - particle_radius) continue;
                if (pos.z < inject_z_min + particle_radius || pos.z > inject_z_max - particle_radius) continue;
                if (pos.y < particle_radius) continue;

                Particle p(pos, particle_radius, particle_density, 0);
                domain.addParticle(p);
                num_particles++;
            }
        }
    }

    std::cout << "  Created " << num_particles << " particles\n";
    std::cout << "  Injection region: " << inject_width*100 << "cm x " << inject_width*100 << "cm square\n";
    std::cout << "  Fill height: " << fill_height*100 << " cm\n\n";

    // ========== Calculate Timestep ==========
    real dt_rayleigh = domain.calculateRayleighTimestep();
    real dt_hertz = domain.calculateHertzTimestep(1.0);  // Assume max v = 1 m/s
    real dt = dt_target;

    std::cout << "Timestep Analysis:\n";
    std::cout << "  Rayleigh timestep: " << dt_rayleigh << " s (20% of theoretical)\n";
    std::cout << "  Hertz timestep:    " << dt_hertz << " s (10% of theoretical)\n";
    std::cout << "  Target timestep:   " << dt_target << " s (user-specified)\n";

    // Safety check
    if (dt_target > 0.5 * dt_rayleigh) {
        std::cout << "  WARNING: Target dt exceeds 50% of Rayleigh criterion!\n";
        std::cout << "           Simulation may be unstable. Recommended: dt < " << (0.5 * dt_rayleigh) << " s\n";
    } else {
        std::cout << "  OK: Target dt = " << (dt_target / dt_rayleigh * 100) << "% of Rayleigh timestep\n";
    }

    std::cout << "  Using dt = " << dt << " s\n";
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

    // Boundaries on GPU
    BoundaryDataGPU box_gpu, pipe_gpu;

    // Allocate and copy box
    box_gpu.allocate(box_mesh.numTriangles());
    box_gpu.copyTrianglesFromHost(box_mesh.triangles);
    box_gpu.material_id = 0;
    box_gpu.motion_type = BoundaryMotionType::STATIC;

    // Allocate and copy pipe
    pipe_gpu.allocate(pipe_mesh.numTriangles());
    pipe_gpu.copyTrianglesFromHost(pipe_mesh.triangles);
    pipe_gpu.material_id = 0;
    pipe_gpu.motion_type = BoundaryMotionType::STATIC;

    // Create array of boundaries on device
    BoundaryDataGPU boundaries_host[2] = {box_gpu, pipe_gpu};
    BoundaryDataGPU* d_boundaries;
    HIP_CHECK(hipMalloc(&d_boundaries, 2 * sizeof(BoundaryDataGPU)));
    HIP_CHECK(hipMemcpy(d_boundaries, boundaries_host, 2 * sizeof(BoundaryDataGPU), hipMemcpyHostToDevice));

    std::cout << "  GPU memory allocated for " << num_particles << " particles\n";
    std::cout << "  GPU boundaries: Box (" << box_mesh.numTriangles() << " tris), Pipe ("
              << pipe_mesh.numTriangles() << " tris)\n\n";

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

        // Compute contact forces (particle-particle)
        gpu_forces::computeParticleParticleForces(particles_gpu, neighbors_gpu, d_materials);

        // Compute wall forces (particle-boundary)
        gpu_forces::computeParticleWallForces(particles_gpu, d_boundaries, 2, d_materials);

        // Integrate (half-step)
        gpu_integrator::velocityHalfStep(particles_gpu, dt);
        gpu_integrator::positionUpdate(particles_gpu, dt);

        // Recompute forces at new positions
        gpu_integrator::clearForces(particles_gpu);
        gpu_integrator::applyGravity(particles_gpu, domain.gravity);
        neighbors_gpu.build(particles_gpu, cutoff);
        gpu_forces::computeParticleParticleForces(particles_gpu, neighbors_gpu, d_materials);
        gpu_forces::computeParticleWallForces(particles_gpu, d_boundaries, 2, d_materials);

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
    box_gpu.free();
    pipe_gpu.free();
    HIP_CHECK(hipFree(d_materials));
    HIP_CHECK(hipFree(d_boundaries));

    return 0;
}
