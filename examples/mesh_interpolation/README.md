# Mesh Interpolation Example

This example demonstrates how to interpolate a field from one mesh to another using ArborX.

## Overview

The driver reads two mesh files:
1. **Source mesh**: Contains elements and a nodal field to be interpolated
2. **Target mesh**: Contains nodes where the field values need to be computed

The algorithm:
1. Builds an axis-aligned bounding volume hierarchy (BVH) from the source mesh elements
2. For each node in the target mesh, queries the BVH to find the containing element
3. Computes barycentric coordinates of the node within the element
4. Interpolates the field value using linear interpolation with barycentric coordinates
5. Writes the interpolated field to the target mesh

## Supported Geometries

- **2D**: Triangular elements
- **3D**: Tetrahedral elements

## Usage

```bash
mpirun -np <num_ranks> ./ArborX_Example_Mesh_Interpolation.exe \
  --source-filename <source.exo> \
  --target-filename <target.exo> \
  --output-filename <output.exo> \
  --source-field-name <field_name> \
  --target-field-name <interpolated_field_name> \
  [--verbose]
```

### Command-line Options

- `--help, -h`: Display help message
- `--source-filename, -s`: Path to source mesh file (Exodus format)
- `--target-filename, -t`: Path to target mesh file (Exodus format)
- `--output-filename, -o`: Path to output mesh file (default: output.exo)
- `--source-field-name, -f`: Name of field to interpolate (default: field)
- `--target-field-name, -F`: Name of interpolated field in output (default: interpolated_field)
- `--source-block-names, -b`: Source element block names (optional)
- `--target-block-names, -B`: Target element block names (optional)
- `--restart-index, -r`: Restart index for reading mesh data (default: -1 for last)
- `--verbose, -v`: Enable verbose output and timing information

## Implementation Details

### Key Components

1. **Mesh Reading**: Uses Panzer STK interface to read Exodus mesh files
2. **BVH Construction**: ArborX's BoundingVolumeHierarchy for spatial indexing
3. **Intersection Queries**: Uses `ArborX::intersects()` to find containing elements
4. **Barycentric Coordinates**: Computed using `ArborX::Experimental::barycentricCoordinates()`
5. **Linear Interpolation**: Field values interpolated using barycentric coordinate weights

### Performance Considerations

- Queries are parallelized across target nodes
- Mesh data is kept on the device (GPU) when using GPU execution spaces
- MPI support for distributed mesh processing

## Dependencies

- ArborX >= 2.1.99
- Trilinos (STK, SEACAS, Exodus)
- Boost (program_options)
- Kokkos
- MPI

## Building

This example is only built when the following conditions are met:
- `ARBORX_ENABLE_MPI` is enabled
- `ARBORX_ENABLE_TRILINOS` is enabled
- Boost program_options is available

## Notes

- Currently supports only nodal field interpolation
- The first containing element is used if a node belongs to multiple elements
- Nodes that don't belong to any element will have zero interpolated value
