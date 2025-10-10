#ifndef EMS_DEM_PRIMITIVES_HPP
#define EMS_DEM_PRIMITIVES_HPP

#include "mesh.hpp"
#include <cmath>

namespace emsdem {

/**
 * Geometric primitives for creating simple boundaries
 */
class Primitives {
public:
    /**
     * Create a box mesh (6 faces, 12 triangles)
     * @param min_corner Minimum corner of box
     * @param max_corner Maximum corner of box
     * @param material_id Material ID for contact
     */
    static Mesh createBox(const Vec3& min_corner, const Vec3& max_corner, int material_id = 0) {
        Mesh mesh(material_id);

        real xmin = min_corner.x, ymin = min_corner.y, zmin = min_corner.z;
        real xmax = max_corner.x, ymax = max_corner.y, zmax = max_corner.z;

        // Define 8 corners
        Vec3 corners[8] = {
            Vec3(xmin, ymin, zmin), Vec3(xmax, ymin, zmin),
            Vec3(xmax, ymax, zmin), Vec3(xmin, ymax, zmin),
            Vec3(xmin, ymin, zmax), Vec3(xmax, ymin, zmax),
            Vec3(xmax, ymax, zmax), Vec3(xmin, ymax, zmax)
        };

        // Bottom face (z = zmin) - normal pointing down
        mesh.addTriangle(Triangle(corners[0], corners[2], corners[1], material_id));
        mesh.addTriangle(Triangle(corners[0], corners[3], corners[2], material_id));

        // Top face (z = zmax) - normal pointing up
        mesh.addTriangle(Triangle(corners[4], corners[5], corners[6], material_id));
        mesh.addTriangle(Triangle(corners[4], corners[6], corners[7], material_id));

        // Front face (y = ymin)
        mesh.addTriangle(Triangle(corners[0], corners[1], corners[5], material_id));
        mesh.addTriangle(Triangle(corners[0], corners[5], corners[4], material_id));

        // Back face (y = ymax)
        mesh.addTriangle(Triangle(corners[3], corners[7], corners[6], material_id));
        mesh.addTriangle(Triangle(corners[3], corners[6], corners[2], material_id));

        // Left face (x = xmin)
        mesh.addTriangle(Triangle(corners[0], corners[4], corners[7], material_id));
        mesh.addTriangle(Triangle(corners[0], corners[7], corners[3], material_id));

        // Right face (x = xmax)
        mesh.addTriangle(Triangle(corners[1], corners[2], corners[6], material_id));
        mesh.addTriangle(Triangle(corners[1], corners[6], corners[5], material_id));

        return mesh;
    }

    /**
     * Create a plane mesh (2 triangles)
     * @param corner Bottom-left corner
     * @param width Width along x-axis
     * @param height Height along y-axis
     * @param normal Normal direction (will be normalized)
     * @param material_id Material ID for contact
     */
    static Mesh createPlane(const Vec3& corner, real width, real height,
                           const Vec3& normal_dir, int material_id = 0) {
        Mesh mesh(material_id);

        // Create local coordinate system
        Vec3 normal = normal_dir.normalized();
        Vec3 tangent1(1, 0, 0);
        if (std::abs(normal.x) > 0.9) {
            tangent1 = Vec3(0, 1, 0);
        }
        tangent1 = (tangent1 - normal * tangent1.dot(normal)).normalized();
        Vec3 tangent2 = normal.cross(tangent1);

        // Define 4 corners of plane
        Vec3 v0 = corner;
        Vec3 v1 = corner + tangent1 * width;
        Vec3 v2 = corner + tangent1 * width + tangent2 * height;
        Vec3 v3 = corner + tangent2 * height;

        // Create two triangles
        mesh.addTriangle(Triangle(v0, v1, v2, material_id));
        mesh.addTriangle(Triangle(v0, v2, v3, material_id));

        return mesh;
    }

    /**
     * Create a cylinder mesh
     * @param center Center of cylinder base
     * @param radius Radius of cylinder
     * @param height Height of cylinder
     * @param axis Axis direction (will be normalized)
     * @param segments Number of segments around circumference
     * @param material_id Material ID for contact
     */
    static Mesh createCylinder(const Vec3& center, real radius, real height,
                              const Vec3& axis_dir, int segments, int material_id = 0) {
        Mesh mesh(material_id);

        Vec3 axis = axis_dir.normalized();

        // Create local coordinate system
        Vec3 radial1(1, 0, 0);
        if (std::abs(axis.x) > 0.9) {
            radial1 = Vec3(0, 1, 0);
        }
        radial1 = (radial1 - axis * radial1.dot(axis)).normalized();
        Vec3 radial2 = axis.cross(radial1);

        // Create vertices around circumference at bottom and top
        std::vector<Vec3> bottom_verts, top_verts;
        for (int i = 0; i < segments; i++) {
            real angle = 2.0 * M_PI * i / segments;
            Vec3 offset = radial1 * (radius * std::cos(angle)) + radial2 * (radius * std::sin(angle));
            bottom_verts.push_back(center + offset);
            top_verts.push_back(center + axis * height + offset);
        }

        // Create side triangles
        for (int i = 0; i < segments; i++) {
            int next = (i + 1) % segments;
            mesh.addTriangle(Triangle(bottom_verts[i], top_verts[i], top_verts[next], material_id));
            mesh.addTriangle(Triangle(bottom_verts[i], top_verts[next], bottom_verts[next], material_id));
        }

        // Create bottom and top caps (triangle fans)
        Vec3 bottom_center = center;
        Vec3 top_center = center + axis * height;

        for (int i = 0; i < segments; i++) {
            int next = (i + 1) % segments;
            // Bottom cap (normal pointing down)
            mesh.addTriangle(Triangle(bottom_center, bottom_verts[next], bottom_verts[i], material_id));
            // Top cap (normal pointing up)
            mesh.addTriangle(Triangle(top_center, top_verts[i], top_verts[next], material_id));
        }

        return mesh;
    }
};

} // namespace emsdem

#endif // EMS_DEM_PRIMITIVES_HPP
