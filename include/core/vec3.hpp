#ifndef EMS_DEM_VEC3_HPP
#define EMS_DEM_VEC3_HPP

#include <cmath>
#include <iostream>

namespace emsdem {

// Precision type: float or double
#ifdef USE_DOUBLE_PRECISION
using real = double;
#else
using real = float;
#endif

/**
 * 3D vector class for positions, velocities, forces, etc.
 */
struct Vec3 {
    real x, y, z;

    // Constructors (GPU-compatible)
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(real x_, real y_, real z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ Vec3(real val) : x(val), y(val), z(val) {}

    // Vector operations (GPU-compatible)
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(real scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    __host__ __device__ Vec3 operator/(real scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    __host__ __device__ Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    __host__ __device__ Vec3& operator*=(real scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    __host__ __device__ Vec3& operator/=(real scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

    // Dot product
    __host__ __device__ real dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    // Cross product
    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y,
                   z * v.x - x * v.z,
                   x * v.y - y * v.x);
    }

    // Length operations
    __host__ __device__ real lengthSquared() const { return x * x + y * y + z * z; }
    __host__ __device__ real length() const {
#ifdef __HIP_DEVICE_COMPILE__
        return sqrtf(lengthSquared());
#else
        return std::sqrt(lengthSquared());
#endif
    }

    __host__ __device__ Vec3 normalized() const {
        real len = length();
        return len > 1e-10 ? (*this / len) : Vec3(0, 0, 0);
    }

    __host__ __device__ void normalize() {
        real len = length();
        if (len > 1e-10) {
            *this /= len;
        }
    }

    // Component access
    __host__ __device__ real& operator[](int i) { return (&x)[i]; }
    __host__ __device__ const real& operator[](int i) const { return (&x)[i]; }

    // I/O
    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

// Scalar * vector (GPU-compatible)
__host__ __device__ inline Vec3 operator*(real scalar, const Vec3& v) {
    return v * scalar;
}

} // namespace emsdem

#endif // EMS_DEM_VEC3_HPP
