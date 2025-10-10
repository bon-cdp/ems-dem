#ifndef EMS_DEM_TRIANGLE_HPP
#define EMS_DEM_TRIANGLE_HPP

#include "core/vec3.hpp"

namespace emsdem {

/**
 * Triangle for boundary geometry (walls, chutes, etc.)
 */
struct Triangle {
    Vec3 v0, v1, v2;    // Vertices
    Vec3 normal;        // Outward-facing normal
    int material_id;    // Material ID for contact

    Triangle() : v0(0), v1(0), v2(0), normal(0, 0, 1), material_id(0) {}

    Triangle(const Vec3& vertex0, const Vec3& vertex1, const Vec3& vertex2, int mat_id = 0)
        : v0(vertex0), v1(vertex1), v2(vertex2), material_id(mat_id)
    {
        computeNormal();
    }

    // Compute normal from vertices (right-hand rule)
    void computeNormal() {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        normal = edge1.cross(edge2);
        normal.normalize();
    }

    // Get centroid
    Vec3 centroid() const {
        return (v0 + v1 + v2) / 3.0;
    }

    // Get area
    real area() const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        return 0.5 * edge1.cross(edge2).length();
    }

    // Check if point is inside triangle (barycentric coordinates)
    bool containsPoint2D(const Vec3& p, Vec3& barycentric) const {
        Vec3 v0v1 = v1 - v0;
        Vec3 v0v2 = v2 - v0;
        Vec3 v0p = p - v0;

        real dot00 = v0v1.dot(v0v1);
        real dot01 = v0v1.dot(v0v2);
        real dot02 = v0v1.dot(v0p);
        real dot11 = v0v2.dot(v0v2);
        real dot12 = v0v2.dot(v0p);

        real inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        real u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        real v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        barycentric = Vec3(1.0 - u - v, u, v);

        return (u >= 0) && (v >= 0) && (u + v <= 1.0);
    }

    // Ray-triangle intersection (Möller–Trumbore algorithm)
    bool intersectRay(const Vec3& origin, const Vec3& direction, real& t, Vec3& hit_point) const {
        const real EPSILON = 1e-8;

        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = direction.cross(edge2);
        real a = edge1.dot(h);

        if (std::abs(a) < EPSILON) {
            return false; // Ray is parallel to triangle
        }

        real f = 1.0 / a;
        Vec3 s = origin - v0;
        real u = f * s.dot(h);

        if (u < 0.0 || u > 1.0) {
            return false;
        }

        Vec3 q = s.cross(edge1);
        real v = f * direction.dot(q);

        if (v < 0.0 || u + v > 1.0) {
            return false;
        }

        t = f * edge2.dot(q);

        if (t > EPSILON) {
            hit_point = origin + direction * t;
            return true;
        }

        return false;
    }

    // Closest point on triangle to a given point
    Vec3 closestPoint(const Vec3& p) const {
        Vec3 ab = v1 - v0;
        Vec3 ac = v2 - v0;
        Vec3 ap = p - v0;

        real d1 = ab.dot(ap);
        real d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) return v0;

        Vec3 bp = p - v1;
        real d3 = ab.dot(bp);
        real d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) return v1;

        Vec3 cp = p - v2;
        real d5 = ab.dot(cp);
        real d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) return v2;

        real vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            real v = d1 / (d1 - d3);
            return v0 + ab * v;
        }

        real vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            real v = d2 / (d2 - d6);
            return v0 + ac * v;
        }

        real va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            real v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + (v2 - v1) * v;
        }

        real denom = 1.0 / (va + vb + vc);
        real v = vb * denom;
        real w = vc * denom;
        return v0 + ab * v + ac * w;
    }
};

} // namespace emsdem

#endif // EMS_DEM_TRIANGLE_HPP
