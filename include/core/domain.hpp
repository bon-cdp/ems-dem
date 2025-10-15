#ifndef EMS_DEM_DOMAIN_HPP
#define EMS_DEM_DOMAIN_HPP

#include "particle.hpp"
#include "material.hpp"
#include "vec3.hpp"
#include <vector>
#include <memory>

namespace emsdem {

/**
 * Simulation domain
 * Manages particles, materials, and simulation parameters
 */
class Domain {
public:
    // Particles
    std::vector<Particle> particles;

    // Materials
    std::vector<MaterialProperties> materials;

    // Simulation parameters
    real timestep;                  // Timestep size (dt)
    real current_time;              // Current simulation time
    int step_count;                 // Number of steps completed

    // Physical parameters
    Vec3 gravity;                   // Gravity vector (y-up system: (0, -9.81, 0))

    // Domain bounds (for spatial binning)
    Vec3 min_bound;
    Vec3 max_bound;

    // Neighbor search parameters
    real neighbor_skin;             // Skin distance for neighbor lists
    real max_particle_radius;       // Maximum particle radius (for neighbor search)

    // Constructor
    Domain()
        : timestep(0),
          current_time(0),
          step_count(0),
          gravity(0, -9.81, 0),  // Y-down gravity
          min_bound(-10, -10, -10),
          max_bound(10, 10, 10),
          neighbor_skin(0),
          max_particle_radius(0)
    {
        // Add default DEM-calibrated material (glass beads with effective E)
        materials.push_back(MaterialProperties::demGlassBeads());
    }

    // Add a particle
    int addParticle(const Particle& p) {
        int id = static_cast<int>(particles.size());
        Particle particle = p;
        particle.id = id;
        particles.push_back(particle);

        // Update max radius
        if (particle.radius > max_particle_radius) {
            max_particle_radius = particle.radius;
        }

        return id;
    }

    // Add a material
    int addMaterial(const MaterialProperties& mat) {
        materials.push_back(mat);
        return static_cast<int>(materials.size()) - 1;
    }

    // Get number of particles
    size_t numParticles() const {
        return particles.size();
    }

    // Calculate total kinetic energy
    real totalKineticEnergy() const {
        real total_ke = 0;
        for (const auto& p : particles) {
            total_ke += p.kineticEnergy();
        }
        return total_ke;
    }

    // Update domain bounds based on particle positions
    void updateBounds(real padding = 0.1) {
        if (particles.empty()) return;

        min_bound = particles[0].position;
        max_bound = particles[0].position;

        for (const auto& p : particles) {
            for (int i = 0; i < 3; i++) {
                if (p.position[i] - p.radius < min_bound[i]) {
                    min_bound[i] = p.position[i] - p.radius;
                }
                if (p.position[i] + p.radius > max_bound[i]) {
                    max_bound[i] = p.position[i] + p.radius;
                }
            }
        }

        // Add padding
        Vec3 pad_vec = (max_bound - min_bound) * padding;
        min_bound -= pad_vec;
        max_bound += pad_vec;
    }

    // Calculate critical timestep based on Rayleigh criterion
    // Rayleigh wave propagation: dt_R = π*R / c_R
    // where c_R ≈ 0.87 * sqrt(G/ρ) is Rayleigh wave speed
    // Safety factor: use 10-20% of Rayleigh timestep
    // Reference: LIGGGHTS documentation, O'Sullivan 2011
    real calculateRayleighTimestep() const {
        if (particles.empty()) return 1e-5;

        real min_dt = 1e10;

        for (const auto& p : particles) {
            if (p.is_fixed) continue;

            const auto& mat = materials[p.material_id];

            // Shear modulus: G = E / (2*(1+ν))
            real G = mat.youngs_modulus / (2.0 * (1.0 + mat.poisson_ratio));

            // Rayleigh wave speed: c_R ≈ 0.87 * sqrt(G/ρ)
            real c_rayleigh = 0.87 * std::sqrt(G / mat.density);

            // Rayleigh timestep: dt_R = π*R / c_R
            real dt_r = M_PI * p.radius / c_rayleigh;

            if (dt_r < min_dt) {
                min_dt = dt_r;
            }
        }

        // Safety factor: 20% of Rayleigh timestep (conservative)
        return 0.2 * min_dt;
    }

    // Legacy function - now calls proper Rayleigh calculation
    real calculateCriticalTimestep() const {
        return calculateRayleighTimestep();
    }

    // Calculate Hertz timestep for dynamic collisions
    // dt_H = 2.87 * (m_eff^2 / (R_eff * E_eff^2 * v_max))^0.2
    // Reference: LIGGGHTS fix check/timestep/gran
    real calculateHertzTimestep(real v_max = 1.0) const {
        if (particles.empty()) return 1e-5;

        real min_dt = 1e10;

        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].is_fixed) continue;

            for (size_t j = i+1; j < particles.size(); j++) {
                if (particles[j].is_fixed) continue;

                const auto& p1 = particles[i];
                const auto& p2 = particles[j];
                const auto& mat1 = materials[p1.material_id];
                const auto& mat2 = materials[p2.material_id];

                real m_eff = MaterialProperties::effectiveMass(p1.mass, p2.mass);
                real r_eff = MaterialProperties::effectiveRadius(p1.radius, p2.radius);
                real E_eff = MaterialProperties::effectiveYoungs(mat1, mat2);

                // Hertz timestep formula
                real dt_h = 2.87 * std::pow(m_eff * m_eff / (r_eff * E_eff * E_eff * v_max), 0.2);

                if (dt_h < min_dt) {
                    min_dt = dt_h;
                }
            }
        }

        // Safety factor: 10% of Hertz timestep
        return 0.1 * min_dt;
    }

    // Get domain size
    Vec3 domainSize() const {
        return max_bound - min_bound;
    }

    // Clear all forces on particles
    void clearForces() {
        for (auto& p : particles) {
            p.clearForces();
        }
    }

    // Apply gravity to all particles
    void applyGravity() {
        for (auto& p : particles) {
            p.applyGravity(gravity);
        }
    }
};

} // namespace emsdem

#endif // EMS_DEM_DOMAIN_HPP
