#ifndef EMS_DEM_VTK_WRITER_HPP
#define EMS_DEM_VTK_WRITER_HPP

#include "core/domain.hpp"
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace emsdem {

/**
 * VTK Legacy format writer for ParaView visualization
 * Writes particles as unstructured grid with sphere glyphs
 */
class VTKWriter {
public:
    /**
     * Write particles to VTK file
     * @param domain The simulation domain
     * @param filename Output filename (without extension)
     * @param timestep Current timestep number (appended to filename)
     */
    static void writeParticles(const Domain& domain, const std::string& base_filename, int timestep) {
        // Generate filename with timestep
        std::ostringstream oss;
        oss << base_filename << "_" << std::setfill('0') << std::setw(6) << timestep << ".vtk";
        std::string filename = oss.str();

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        const auto& particles = domain.particles;
        size_t nparticles = particles.size();

        // Write VTK header
        file << "# vtk DataFile Version 3.0\n";
        file << "EMS-DEM Particle Data - Step " << timestep << "\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n\n";

        // Write points (particle centers)
        file << "POINTS " << nparticles << " float\n";
        for (const auto& p : particles) {
            file << p.position.x << " " << p.position.y << " " << p.position.z << "\n";
        }
        file << "\n";

        // Write cells (each particle is a vertex cell)
        file << "CELLS " << nparticles << " " << (nparticles * 2) << "\n";
        for (size_t i = 0; i < nparticles; i++) {
            file << "1 " << i << "\n";
        }
        file << "\n";

        // Write cell types (VTK_VERTEX = 1)
        file << "CELL_TYPES " << nparticles << "\n";
        for (size_t i = 0; i < nparticles; i++) {
            file << "1\n";
        }
        file << "\n";

        // Write point data
        file << "POINT_DATA " << nparticles << "\n";

        // Radius (for sphere glyphs in ParaView)
        file << "SCALARS radius float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) {
            file << p.radius << "\n";
        }
        file << "\n";

        // Velocity magnitude
        file << "SCALARS velocity_magnitude float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) {
            file << p.velocity.length() << "\n";
        }
        file << "\n";

        // Velocity vector
        file << "VECTORS velocity float\n";
        for (const auto& p : particles) {
            file << p.velocity.x << " " << p.velocity.y << " " << p.velocity.z << "\n";
        }
        file << "\n";

        // Angular velocity
        file << "VECTORS angular_velocity float\n";
        for (const auto& p : particles) {
            file << p.angular_velocity.x << " " << p.angular_velocity.y << " " << p.angular_velocity.z << "\n";
        }
        file << "\n";

        // Force
        file << "VECTORS force float\n";
        for (const auto& p : particles) {
            file << p.force.x << " " << p.force.y << " " << p.force.z << "\n";
        }
        file << "\n";

        // Material ID
        file << "SCALARS material_id int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) {
            file << p.material_id << "\n";
        }
        file << "\n";

        // Particle ID
        file << "SCALARS particle_id int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) {
            file << p.id << "\n";
        }
        file << "\n";

        // Kinetic energy
        file << "SCALARS kinetic_energy float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (const auto& p : particles) {
            file << p.kineticEnergy() << "\n";
        }
        file << "\n";

        file.close();
        std::cout << "Wrote VTK file: " << filename << " (" << nparticles << " particles)" << std::endl;
    }

    /**
     * Write ParaView collection file (.pvd) for time series
     * Call this at the end of simulation to create animation
     */
    static void writeCollection(const std::string& base_filename,
                               const std::vector<std::pair<int, real>>& timesteps) {
        std::string pvd_filename = base_filename + ".pvd";
        std::ofstream file(pvd_filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << pvd_filename << std::endl;
            return;
        }

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";

        for (const auto& ts : timesteps) {
            int step = ts.first;
            real time = ts.second;

            std::ostringstream oss;
            oss << base_filename << "_" << std::setfill('0') << std::setw(6) << step << ".vtk";

            file << "    <DataSet timestep=\"" << time << "\" file=\"" << oss.str() << "\"/>\n";
        }

        file << "  </Collection>\n";
        file << "</VTKFile>\n";

        file.close();
        std::cout << "Wrote ParaView collection: " << pvd_filename << std::endl;
    }
};

} // namespace emsdem

#endif // EMS_DEM_VTK_WRITER_HPP
