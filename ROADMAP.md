# EMS-DEM Development Roadmap

## ✅ Phase 1: Foundation & Core DEM (COMPLETE)
- [x] Repository structure and build system
- [x] CMake configured for hipcc (AMD Radeon 780M - gfx1103)
- [x] Core data structures (Particle, Domain, Material)
- [x] VTK output for ParaView visualization
- [x] Geometry system (STL loader, primitives)
- [x] GPU memory management (Structure of Arrays)
- [x] GPU integration kernel (Velocity-Verlet)
- [x] GPU particle-particle forces (Hertz-Mindlin)
- [x] GPU neighbor search (brute force O(N²) working)
- [x] First successful GPU simulation (12 particles)
- [x] Coordinate system: Y-up (gravity = (0, -9.81, 0))

## ✅ Phase 2: Boundary Motion & Particle-Wall Forces (COMPLETE)
- [x] Boundary motion types:
  - [x] CONVEYOR: Fixed geometry, applies tangential velocity
  - [x] TRANSLATING: Linear motion of geometry
  - [x] ROTATING: Rotation about arbitrary axis
- [x] Particle-wall contact force kernel
- [x] GPU-compatible triangle geometry functions
- [x] Integration with Hertz-Mindlin model
- [x] Boundary velocity computation on GPU

**TODO: Validate boundary motion against physical experiments**

## 🚧 Phase 3: Optimization & Scale (IN PROGRESS)
- [x] Neighbor search optimization framework (adaptive N²/spatial grid)
- [ ] GPU parallel scan for cell offsets
- [ ] Spatial grid for N > 1000 (replace O(N²))
- [ ] Test with 1k, 10k, 100k particles
- [ ] Memory optimization (pinned memory, streams)
- [ ] Profiling and performance tuning

## 📋 Phase 4: Production Features (PLANNED)
- [ ] Complete angle of repose test with pipe withdrawal
- [ ] STL geometry loading for real chute designs
- [ ] Multiple conveyor belt configuration
- [ ] Spout discharge optimization
- [ ] Material property calibration tool
- [ ] Polydisperse particle distributions
- [ ] Rolling resistance models

## 🎯 Phase 5: Validation (PLANNED)
- [ ] Angle of repose validation (known materials)
- [ ] Compare with Bulk Flow Analyst results
- [ ] Conveyor belt velocity transfer validation
- [ ] Transfer chute flow pattern validation
- [ ] Material flow rate benchmarks
- [ ] Documentation and case studies

## 🔬 Future Enhancements
- [ ] Multi-sphere clumps (rigid bodies)
- [ ] Particle breakage mechanics
- [ ] Wear prediction from contact forces
- [ ] Heat transfer for hot materials
- [ ] Moisture/cohesion models
- [ ] Multi-GPU support for massive simulations

## 📊 Current Status
**Working:** GPU DEM with particle-particle and particle-wall forces
**Performance:** O(N²) neighbor search (good for N < 1000)
**Target Hardware:** AMD Radeon 780M (gfx1103)
**Repository:** https://github.com/bon-cdp/ems-dem

## 🎓 Learning & Development Notes
- Neighbor search: Need rocThrust or custom radix sort for optimal O(N) grid
- Boundary motion: Three types implemented, validation pending
- Coordinate system: Y-up convention (industry standard)
- Build system: hipcc with gfx1103 target
- Memory layout: SoA for GPU memory coalescing
