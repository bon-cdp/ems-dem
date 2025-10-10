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

/**
 * Angle of Repose Test
 *
 * Setup:
 * 1. Box container at bottom
 * 2. Cylindrical pipe inside the box
 * 3. Fill pipe with particles
 * 4. Let particles settle
 * 5. Pull pipe upward slowly
 * 6. Measure final angle of repose
 */

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "   EMS-DEM: Angle of Repose Test\n";
    std::cout << "===========================================\n\n";

    // ========== Simulation Parameters ==========
    const real box_width = 0.3;       // 30 cm
    const real box_length = 0.3;      // 30 cm
    const real box_height = 0.2;      // 20 cm

    const real pipe_radius = 0.05;    // 5 cm
    const real pipe_height = 0.15;    // 15 cm

    const real particle_radius = 0.003;  // 3 mm particles
    const real particle_density = 2500;  // Glass beads: 2500 kg/m^3

    const real fill_fraction = 0.6;   // Fill 60% of pipe volume

    const real dt_safety = 0.2;       // Safety factor for timestep
    const int  output_interval = 100; // VTK output every N steps

    const real settle_time = 0.5;     // Settle for 0.5 seconds
    const real withdraw_speed = 0.05; // Pull pipe up at 5 cm/s
    const real withdraw_distance = pipe_height;  // Pull pipe completely out

    // ========== Setup Domain ==========
    Domain domain;
    domain.gravity = Vec3(0, 0, -9.81);
    domain.neighbor_skin = 0.5 * particle_radius;  // Neighbor list skin distance

    // Add glass beads material
    int mat_glass = 0;  // Default material is already glass beads

    // Add steel material for walls
    int mat_steel = domain.addMaterial(MaterialProperties::steel());

    std::cout << "Domain setup complete\n";

    // ========== Create Geometry ==========
    std::vector<Mesh> walls;

    // Create box container
    Vec3 box_min(-box_width/2, -box_length/2, 0);
    Vec3 box_max(box_width/2, box_length/2, box_height);
    Mesh box = Primitives::createBox(box_min, box_max, mat_steel);
    walls.push_back(box);

    // Create cylindrical pipe (centered in box)
    Vec3 pipe_center(0, 0, 0.01);  // Slightly above box bottom
    Vec3 pipe_axis(0, 0, 1);
    Mesh pipe = Primitives::createCylinder(pipe_center, pipe_radius, pipe_height, pipe_axis, 20, mat_steel);
    walls.push_back(pipe);

    std::cout << "Geometry created:\n";
    std::cout << "  Box: " << box.numTriangles() << " triangles\n";
    std::cout << "  Pipe: " << pipe.numTriangles() << " triangles\n\n";

    // ========== Fill Pipe with Particles ==========
    std::cout << "Filling pipe with particles...\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real> dist_x(-pipe_radius * 0.9, pipe_radius * 0.9);
    std::uniform_real_distribution<real> dist_y(-pipe_radius * 0.9, pipe_radius * 0.9);

    int num_particles = 0;
    real z_current = pipe_center.z + 2 * particle_radius;
    real z_max = pipe_center.z + pipe_height * fill_fraction;

    while (z_current < z_max) {
        for (int attempt = 0; attempt < 100; attempt++) {
            real x = dist_x(gen);
            real y = dist_y(gen);

            // Check if inside pipe radius
            if (std::sqrt(x*x + y*y) + particle_radius > pipe_radius) {
                continue;
            }

            Vec3 pos(x, y, z_current);

            // Check overlap with existing particles
            bool overlap = false;
            for (const auto& p : domain.particles) {
                real dist = (p.position - pos).length();
                if (dist < 2.0 * particle_radius) {
                    overlap = true;
                    break;
                }
            }

            if (!overlap) {
                Particle p(pos, particle_radius, particle_density, mat_glass);
                domain.addParticle(p);
                num_particles++;
                break;
            }
        }

        z_current += 1.8 * particle_radius;  // Move up for next layer
    }

    std::cout << "  Created " << num_particles << " particles\n\n";

    // ========== Calculate Timestep ==========
    real dt_critical = domain.calculateCriticalTimestep();
    real dt = dt_safety * dt_critical;

    std::cout << "Timestep:\n";
    std::cout << "  Critical dt: " << dt_critical << " s\n";
    std::cout << "  Actual dt:   " << dt << " s (safety factor " << dt_safety << ")\n\n";

    // ========== Update domain bounds for neighbor search ==========
    domain.updateBounds();

    // ========== Simulation Loop ==========
    NeighborSearch neighbor_search;

    real current_time = 0;
    int step = 0;
    int output_count = 0;

    std::vector<std::pair<int, real>> vtk_timesteps;  // For ParaView collection

    std::cout << "Starting simulation...\n";
    std::cout << "===========================================\n\n";

    // Phase 1: Settle particles
    std::cout << "Phase 1: Settling particles (" << settle_time << " s)\n";
    while (current_time < settle_time) {
        // Clear forces
        domain.clearForces();

        // Apply gravity
        domain.applyGravity();

        // Build neighbor list
        neighbor_search.buildNeighborList(domain);

        // Compute contact forces
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);

        // Integrate
        VelocityVerletIntegrator::integrate(domain, dt);
        domain.clearForces();
        domain.applyGravity();
        neighbor_search.buildNeighborList(domain);
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);
        VelocityVerletIntegrator::completeVelocityUpdate(domain, dt);

        // Output
        if (step % output_interval == 0) {
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

    // Phase 2: Withdraw pipe
    std::cout << "\nPhase 2: Withdrawing pipe\n";
    real withdraw_time = withdraw_distance / withdraw_speed;
    real phase2_end = current_time + withdraw_time;

    walls[1].velocity = Vec3(0, 0, withdraw_speed);  // Pipe moves upward

    while (current_time < phase2_end) {
        // Clear forces
        domain.clearForces();

        // Apply gravity
        domain.applyGravity();

        // Update pipe position
        walls[1].updatePosition(dt);

        // Build neighbor list
        neighbor_search.buildNeighborList(domain);

        // Compute contact forces
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);

        // Integrate
        VelocityVerletIntegrator::integrate(domain, dt);
        domain.clearForces();
        domain.applyGravity();
        neighbor_search.buildNeighborList(domain);
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);
        VelocityVerletIntegrator::completeVelocityUpdate(domain, dt);

        // Output
        if (step % output_interval == 0) {
            real ke = domain.totalKineticEnergy();
            std::cout << "  Step " << step << ", t = " << current_time
                     << " s, KE = " << ke << " J, pipe z = "
                     << walls[1].triangles[0].v0.z << " m\n";

            VTKWriter::writeParticles(domain, "results/particles", output_count);
            vtk_timesteps.push_back({output_count, current_time});
            output_count++;
        }

        current_time += dt;
        step++;
    }

    // Phase 3: Final settling
    std::cout << "\nPhase 3: Final settling\n";
    real phase3_duration = 1.0;  // 1 second
    real phase3_end = current_time + phase3_duration;

    while (current_time < phase3_end) {
        // Clear forces
        domain.clearForces();

        // Apply gravity
        domain.applyGravity();

        // Build neighbor list
        neighbor_search.buildNeighborList(domain);

        // Compute contact forces
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);

        // Integrate
        VelocityVerletIntegrator::integrate(domain, dt);
        domain.clearForces();
        domain.applyGravity();
        neighbor_search.buildNeighborList(domain);
        ContactForces::computeParticleParticleForces(domain, neighbor_search);
        ContactForces::computeParticleWallForces(domain, walls);
        VelocityVerletIntegrator::completeVelocityUpdate(domain, dt);

        // Output
        if (step % output_interval == 0) {
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

    // TODO: Calculate angle of repose from final particle positions

    return 0;
}
