#ifndef EMS_DEM_MESH_HPP
#define EMS_DEM_MESH_HPP

#include "triangle.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace emsdem {

/**
 * Triangle mesh for boundaries
 * Can load from STL files (ASCII format)
 */
class Mesh {
public:
    std::vector<Triangle> triangles;
    int material_id;
    Vec3 velocity;          // For translating boundaries
    Vec3 angular_velocity;  // For rotating boundaries
    Vec3 rotation_center;   // Center of rotation
    Vec3 conveyor_velocity; // Tangential velocity applied to particles (conveyor-type)

    Mesh(int mat_id = 0)
        : material_id(mat_id),
          velocity(0),
          angular_velocity(0),
          rotation_center(0),
          conveyor_velocity(0)
    {}

    // Add a triangle
    void addTriangle(const Triangle& tri) {
        Triangle t = tri;
        t.material_id = material_id;
        triangles.push_back(t);
    }

    // Load from ASCII STL file
    bool loadSTL(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open STL file: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::vector<Vec3> vertices;
        Vec3 normal;

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));

            // Check for keywords
            if (line.substr(0, 5) == "facet") {
                // Parse normal
                std::istringstream iss(line);
                std::string facet, norm;
                iss >> facet >> norm >> normal.x >> normal.y >> normal.z;
                vertices.clear();
            }
            else if (line.substr(0, 6) == "vertex") {
                // Parse vertex
                std::istringstream iss(line);
                std::string vert;
                Vec3 v;
                iss >> vert >> v.x >> v.y >> v.z;
                vertices.push_back(v);
            }
            else if (line.substr(0, 8) == "endfacet") {
                // Create triangle
                if (vertices.size() == 3) {
                    Triangle tri(vertices[0], vertices[1], vertices[2], material_id);
                    triangles.push_back(tri);
                }
            }
        }

        file.close();
        std::cout << "Loaded STL mesh: " << filename << " (" << triangles.size() << " triangles)" << std::endl;
        return true;
    }

    // Save to ASCII STL file
    bool saveSTL(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        file << "solid mesh\n";
        for (const auto& tri : triangles) {
            file << "  facet normal " << tri.normal.x << " " << tri.normal.y << " " << tri.normal.z << "\n";
            file << "    outer loop\n";
            file << "      vertex " << tri.v0.x << " " << tri.v0.y << " " << tri.v0.z << "\n";
            file << "      vertex " << tri.v1.x << " " << tri.v1.y << " " << tri.v1.z << "\n";
            file << "      vertex " << tri.v2.x << " " << tri.v2.y << " " << tri.v2.z << "\n";
            file << "    endloop\n";
            file << "  endfacet\n";
        }
        file << "endsolid mesh\n";

        file.close();
        return true;
    }

    // Get bounding box
    void getBounds(Vec3& min_bound, Vec3& max_bound) const {
        if (triangles.empty()) return;

        min_bound = triangles[0].v0;
        max_bound = triangles[0].v0;

        for (const auto& tri : triangles) {
            for (int i = 0; i < 3; i++) {
                min_bound[i] = std::min({min_bound[i], tri.v0[i], tri.v1[i], tri.v2[i]});
                max_bound[i] = std::max({max_bound[i], tri.v0[i], tri.v1[i], tri.v2[i]});
            }
        }
    }

    // Translate mesh
    void translate(const Vec3& offset) {
        for (auto& tri : triangles) {
            tri.v0 += offset;
            tri.v1 += offset;
            tri.v2 += offset;
        }
    }

    // Update mesh position (for moving boundaries)
    void updatePosition(real dt) {
        if (velocity.lengthSquared() > 1e-10) {
            translate(velocity * dt);
        }

        // TODO: Add rotation support in future
    }

    // Get number of triangles
    size_t numTriangles() const {
        return triangles.size();
    }
};

} // namespace emsdem

#endif // EMS_DEM_MESH_HPP
