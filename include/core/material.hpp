#ifndef EMS_DEM_MATERIAL_HPP
#define EMS_DEM_MATERIAL_HPP

#include "vec3.hpp"

namespace emsdem {

/**
 * Material properties for DEM particles and boundaries
 * Used for Hertz-Mindlin contact model
 */
struct MaterialProperties {
    // Particle properties
    real density;              // kg/m^3
    real youngs_modulus;       // Pa (Young's modulus)
    real poisson_ratio;        // [-] (Poisson's ratio, typically 0.2-0.3)

    // Friction coefficients
    real friction_static;      // [-] (coefficient of static friction)
    real friction_rolling;     // [-] (coefficient of rolling resistance)

    // Damping
    real restitution_coeff;    // [-] (coefficient of restitution, 0=perfectly inelastic, 1=perfectly elastic)

    // Cohesion (optional, for future)
    real cohesion_energy;      // J/m^2 (surface energy density)

    // Default constructor: glass beads
    MaterialProperties()
        : density(2500.0),              // Glass density
          youngs_modulus(70e9),         // 70 GPa for glass
          poisson_ratio(0.23),
          friction_static(0.3),
          friction_rolling(0.01),
          restitution_coeff(0.9),
          cohesion_energy(0.0)          // Non-cohesive by default
    {}

    // Helper: Effective Young's modulus for Hertz contact
    static real effectiveYoungs(const MaterialProperties& m1, const MaterialProperties& m2) {
        real inv_E1 = (1.0 - m1.poisson_ratio * m1.poisson_ratio) / m1.youngs_modulus;
        real inv_E2 = (1.0 - m2.poisson_ratio * m2.poisson_ratio) / m2.youngs_modulus;
        return 1.0 / (inv_E1 + inv_E2);
    }

    // Helper: Effective radius for contact
    static real effectiveRadius(real r1, real r2) {
        return (r1 * r2) / (r1 + r2);
    }

    // Helper: Effective mass for contact
    static real effectiveMass(real m1, real m2) {
        return (m1 * m2) / (m1 + m2);
    }

    // Predefined materials
    static MaterialProperties glassBeads() {
        return MaterialProperties(); // Default is glass beads
    }

    static MaterialProperties sand() {
        MaterialProperties m;
        m.density = 1600.0;
        m.youngs_modulus = 1e7;     // 10 MPa (sand is much softer than glass)
        m.poisson_ratio = 0.3;
        m.friction_static = 0.5;
        m.friction_rolling = 0.02;
        m.restitution_coeff = 0.6;
        return m;
    }

    static MaterialProperties steel() {
        MaterialProperties m;
        m.density = 7850.0;
        m.youngs_modulus = 200e9;   // 200 GPa
        m.poisson_ratio = 0.3;
        m.friction_static = 0.4;
        m.friction_rolling = 0.001;
        m.restitution_coeff = 0.7;
        return m;
    }
};

} // namespace emsdem

#endif // EMS_DEM_MATERIAL_HPP
