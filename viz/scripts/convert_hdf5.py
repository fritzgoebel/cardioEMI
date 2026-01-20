#!/usr/bin/env python3
"""
Convert HDF5 mesh data to web-friendly binary format.
Can be run standalone or imported as a module.

Standalone usage: python viz/scripts/convert_hdf5.py [mesh_name]
Module usage: from convert_hdf5 import convert_mesh
"""
import h5py
import numpy as np
import json
import os
from pathlib import Path


def convert_mesh(input_h5_path: Path, output_dir: Path, progress_callback=None) -> dict:
    """
    Convert HDF5 mesh to visualization binary format.

    Args:
        input_h5_path: Path to .h5 file
        output_dir: Where to write output files
        progress_callback: Optional callback(percent, message) for progress updates

    Returns:
        Metadata dict with vertex_count, facet_count, bounds, etc.
    """
    def report(percent, message):
        if progress_callback:
            progress_callback(percent, message)
        print(f"  [{percent:3d}%] {message}")

    os.makedirs(output_dir, exist_ok=True)

    report(0, f"Reading HDF5 file: {input_h5_path.name}")

    with h5py.File(input_h5_path, 'r') as f:
        # Load geometry (vertices)
        report(10, "Loading vertices...")
        vertices = f['/Mesh/mesh/geometry'][:].astype(np.float32)
        report(20, f"Loaded {len(vertices)} vertices")

        # Load facet data
        report(30, "Loading facet data...")
        facet_tags = f['/Mesh/facet_tags/Values'][:]
        facet_topo = f['/Mesh/facet_tags/topology'][:]
        report(40, f"Loaded {len(facet_tags)} facets")

        # Filter to membrane facets only (positive tags = internal membranes)
        report(50, "Filtering membrane facets...")
        membrane_mask = facet_tags > 0
        membrane_facets = facet_topo[membrane_mask].astype(np.uint32)
        membrane_tag_values = facet_tags[membrane_mask].astype(np.int32)
        report(55, f"Found {len(membrane_facets)} membrane triangles")

        # Calculate bounds
        bounds = {
            "x": [float(vertices[:,0].min()), float(vertices[:,0].max())],
            "y": [float(vertices[:,1].min()), float(vertices[:,1].max())],
            "z": [float(vertices[:,2].min()), float(vertices[:,2].max())]
        }

        # Try to determine mesh_conversion_factor from the scale
        # If bounds are very small (< 1), likely already in cm, factor is 1.0
        # If bounds are large (> 10), likely in micrometers, factor is 0.0001
        max_extent = max(
            bounds['x'][1] - bounds['x'][0],
            bounds['y'][1] - bounds['y'][0],
            bounds['z'][1] - bounds['z'][0]
        )
        mesh_conversion_factor = 0.0001 if max_extent > 10 else 1.0

        # Save binary data
        report(60, "Writing vertex data...")
        vertices_path = output_dir / "mesh_vertices.bin"
        vertices.tofile(vertices_path)

        report(70, "Writing facet data...")
        facets_path = output_dir / "membrane_facets.bin"
        membrane_facets.tofile(facets_path)

        report(80, "Writing tag data...")
        tags_path = output_dir / "membrane_tags.bin"
        membrane_tag_values.tofile(tags_path)

        # Build metadata
        metadata = {
            "vertex_count": len(vertices),
            "facet_count": len(membrane_facets),
            "bounds": bounds,
            "mesh_conversion_factor": mesh_conversion_factor,
            "unique_tags": sorted(set(membrane_tag_values.tolist())),
            "source_file": input_h5_path.name
        }

        report(90, "Writing metadata...")
        metadata_path = output_dir / "mesh_metadata.json"
        with open(metadata_path, 'w') as mf:
            json.dump(metadata, mf, indent=2)

        report(100, "Conversion complete!")

    return metadata


def main():
    """Command-line entry point."""
    import sys

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    # Default mesh or from command line
    if len(sys.argv) > 1:
        mesh_name = sys.argv[1]
    else:
        mesh_name = "pepe36_colored"

    # Find HDF5 file
    h5_path = PROJECT_ROOT / "data" / f"{mesh_name}.h5"
    if not h5_path.exists():
        print(f"Error: HDF5 file not found: {h5_path}")
        sys.exit(1)

    # Output to mesh-specific subdirectory
    output_dir = SCRIPT_DIR.parent / "data" / mesh_name

    print(f"Converting mesh: {mesh_name}")
    print(f"Input: {h5_path}")
    print(f"Output: {output_dir}")
    print()

    metadata = convert_mesh(h5_path, output_dir)

    print()
    print(f"Conversion complete!")
    print(f"  Vertices: {metadata['vertex_count']}")
    print(f"  Facets: {metadata['facet_count']}")
    print(f"  Conversion factor: {metadata['mesh_conversion_factor']}")


if __name__ == "__main__":
    main()
