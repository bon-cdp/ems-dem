# Data Directory

## Geometry Files

Place STL geometry files in `geometry/` subdirectory:

```
data/
└── geometry/
    ├── box.stl       # Rectangular container
    ├── pipe.stl      # Cylindrical pipe for angle of repose test
    └── ...           # Other geometry files
```

**Important**: STL files must be in **ASCII format**, not binary.

## Material Configuration

Material properties are currently hardcoded in `MaterialProperties` class.
Future: Add material configuration files here.

## Output

Simulation results will be saved to `../build/results/` directory.
