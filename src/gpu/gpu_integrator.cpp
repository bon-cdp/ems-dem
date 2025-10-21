#include "gpu/gpu_integrator.hpp"

namespace emsdem {

// Kernel to clear forces
__global__ void clearForcesKernel(Vec3* force, Vec3* torque, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        force[idx] = Vec3(0);
        torque[idx] = Vec3(0);
    }
}

// Kernel to apply gravity
__global__ void applyGravityKernel(Vec3* force, const real* mass, const bool* is_fixed,
                                    Vec3 gravity, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && !is_fixed[idx]) {
        force[idx] += mass[idx] * gravity;
    }
}

// Kernel for half-step velocity update
__global__ void velocityHalfStepKernel(Vec3* velocity, Vec3* angular_velocity,
                                        const Vec3* force, const Vec3* torque,
                                        const real* mass, const real* inertia,
                                        const bool* is_fixed, real half_dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && !is_fixed[idx]) {
        // Linear velocity
        Vec3 accel = force[idx] / mass[idx];
        velocity[idx] += accel * half_dt;

        // Angular velocity
        Vec3 angular_accel = torque[idx] / inertia[idx];
        angular_velocity[idx] += angular_accel * half_dt;
    }
}

// Kernel for position update
__global__ void positionUpdateKernel(Vec3* position, const Vec3* velocity,
                                     const bool* is_fixed, real dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && !is_fixed[idx]) {
        position[idx] += velocity[idx] * dt;
    }
}

namespace gpu_integrator {

void clearForces(ParticleDataGPU& particles) {
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    hipLaunchKernelGGL(clearForcesKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.force, particles.torque, n);
    HIP_CHECK(hipGetLastError());
}

void applyGravity(ParticleDataGPU& particles, const Vec3& gravity) {
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    hipLaunchKernelGGL(applyGravityKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.force, particles.mass, particles.is_fixed, gravity, n);
    HIP_CHECK(hipGetLastError());
}

void velocityHalfStep(ParticleDataGPU& particles, real dt) {
    int n = particles.num_particles;
    int grid_size = getGridSize(n);
    real half_dt = 0.5 * dt;

    hipLaunchKernelGGL(velocityHalfStepKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.velocity, particles.angular_velocity,
                       particles.force, particles.torque,
                       particles.mass, particles.inertia, particles.is_fixed,
                       half_dt, n);
    HIP_CHECK(hipGetLastError());
}

void positionUpdate(ParticleDataGPU& particles, real dt) {
    int n = particles.num_particles;
    int grid_size = getGridSize(n);

    hipLaunchKernelGGL(positionUpdateKernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       particles.position, particles.velocity, particles.is_fixed, dt, n);
    HIP_CHECK(hipGetLastError());
}

void velocityFullStep(ParticleDataGPU& particles, real dt) {
    // Complete the velocity update (second half of Velocity Verlet)
    // This is the same kernel as half-step, just called with the correct half-timestep
    velocityHalfStep(particles, dt);  // velocityHalfStep internally uses 0.5*dt
}

void integrate(ParticleDataGPU& particles, const Vec3& gravity, real dt) {
    clearForces(particles);
    applyGravity(particles, gravity);
    velocityHalfStep(particles, dt);
    positionUpdate(particles, dt);
}

} // namespace gpu_integrator

} // namespace emsdem
