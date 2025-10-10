#ifndef EMS_DEM_PARTICLE_HPP
#define EMS_DEM_PARTICLE_HPP

#include "vec3.hpp"
#include "material.hpp"

namespace emsdem {

/**
 * Particle structure
 * Stores all state information for a single spherical particle
 */
struct Particle {
    // Position and orientation
    Vec3 position;
    Vec3 velocity;
    Vec3 angular_velocity;   // Angular velocity vector (omega)

    // Forces and torques (accumulated during timestep)
    Vec3 force;
    Vec3 torque;

    // Physical properties
    real radius;
    real mass;
    real inertia;            // Moment of inertia for sphere: (2/5) * m * r^2

    // Material index (references material properties)
    int material_id;

    // Particle ID (unique identifier)
    int id;

    // Flags
    bool is_fixed;           // Fixed particles don't move (for boundary particles)

    // Constructor
    Particle()
        : position(0),
          velocity(0),
          angular_velocity(0),
          force(0),
          torque(0),
          radius(0.001),     // 1mm default
          mass(0),
          inertia(0),
          material_id(0),
          id(-1),
          is_fixed(false)
    {}

    Particle(Vec3 pos, real r, real density, int mat_id = 0, int pid = -1)
        : position(pos),
          velocity(0),
          angular_velocity(0),
          force(0),
          torque(0),
          radius(r),
          material_id(mat_id),
          id(pid),
          is_fixed(false)
    {
        // Calculate mass and inertia from radius and density
        real volume = (4.0 / 3.0) * M_PI * r * r * r;
        mass = density * volume;
        inertia = (2.0 / 5.0) * mass * r * r;
    }

    // Update mass and inertia when radius changes
    void updateMassInertia(real density) {
        real volume = (4.0 / 3.0) * M_PI * radius * radius * radius;
        mass = density * volume;
        inertia = (2.0 / 5.0) * mass * radius * radius;
    }

    // Reset forces and torques (called at start of each timestep)
    void clearForces() {
        force = Vec3(0);
        torque = Vec3(0);
    }

    // Apply gravity
    void applyGravity(const Vec3& gravity) {
        if (!is_fixed) {
            force += mass * gravity;
        }
    }

    // Get kinetic energy
    real kineticEnergy() const {
        real trans_ke = 0.5 * mass * velocity.lengthSquared();
        real rot_ke = 0.5 * inertia * angular_velocity.lengthSquared();
        return trans_ke + rot_ke;
    }
};

} // namespace emsdem

#endif // EMS_DEM_PARTICLE_HPP
