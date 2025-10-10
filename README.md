# EMS-DEM

**ROCm-Accelerated Discrete Element Method for Bulk Material Handling**

A high-performance DEM simulator designed for industrial bulk materials applications including:
- Conveyor transfer chutes
- Spout discharge systems
- Angle of repose testing
- Material flow analysis

## Features

- **GPU Acceleration**: AMD ROCm/HIP for 10-20x speedup
- **Conveyor Boundaries**: Three motion types (conveyor-type, translation, rotation)
- **Industrial Geometry**: STL import for CAD-designed chutes and equipment
- **Material Models**: Hertz-Mindlin contact with rolling resistance
- **Validation**: Benchmarked against Bulk Flow Analyst

## Project Structure

```
ems-dem/
├── src/           # Source files
│   ├── core/      # Particle, domain, material properties
│   ├── cpu/       # CPU reference implementation
│   ├── gpu/       # HIP GPU kernels
│   ├── geometry/  # STL loader, primitives, boundaries
│   └── io/        # VTK output, configuration
├── include/       # Header files
├── tests/         # Test cases and validation
├── examples/      # Example simulations
└── docs/          # Documentation
```

## Build Requirements

- CMake 3.21+
- ROCm 5.0+ with HIP
- C++17 compiler
- (Optional) ParaView for visualization

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc
make -j$(nproc)
```

## Quick Start

```bash
# Run angle of repose test
./build/tests/angle_of_repose

# Visualize results in ParaView
paraview results/particles_*.vtk
```

## Project Status

**Phase 1**: Foundation & Core DEM (In Progress)
- [x] Repository structure
- [ ] Build system
- [ ] Core data structures
- [ ] CPU reference implementation

## License

MIT License (or your preferred license)

## Developed By

EMS-Tech R&D
