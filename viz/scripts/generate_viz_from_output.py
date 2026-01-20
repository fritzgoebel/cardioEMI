#!/usr/bin/env python3
"""
Generate visualization data from simulation output.
Extracts membrane facets from the tags.h5 file and creates binary visualization files.

This version creates per-facet visualization data to correctly handle
cell-cell junction disks where multiple membranes meet.
"""
import h5py
import numpy as np
import json
import pickle
from pathlib import Path


def generate_viz_data(sim_output_dir: Path, viz_output_dir: Path, progress_callback=None) -> dict:
    """
    Generate visualization binary files from simulation output.

    Creates "expanded" mesh where each facet has its own vertices (no sharing)
    so that per-facet voltages can be correctly displayed.

    Args:
        sim_output_dir: Directory containing simulation output (v.h5, tags.h5, etc.)
        viz_output_dir: Directory to write visualization files
        progress_callback: Optional callback(percent, message)

    Returns:
        Metadata dict
    """
    def report(percent, message):
        if progress_callback:
            progress_callback(percent, message)
        print(f"  [{percent:3d}%] {message}")

    sim_output_dir = Path(sim_output_dir)
    viz_output_dir = Path(viz_output_dir)

    # Check for required files
    tags_h5 = sim_output_dir / 'tags.h5'
    v_h5 = sim_output_dir / 'v.h5'
    facet_map_pickle = sim_output_dir / 'facet_tag_to_pair.pickle'

    if not tags_h5.exists():
        raise FileNotFoundError(f"tags.h5 not found in {sim_output_dir}")
    if not v_h5.exists():
        raise FileNotFoundError(f"v.h5 not found in {sim_output_dir}")

    viz_output_dir.mkdir(parents=True, exist_ok=True)

    report(0, "Loading mesh geometry from simulation output...")

    with h5py.File(v_h5, 'r') as f:
        # Load vertices from the voltage output (has the scaled mesh)
        vertices = f['Mesh']['mesh']['geometry'][:].astype(np.float32)
        report(10, f"Loaded {len(vertices)} vertices")

    report(15, "Loading facet tags...")

    with h5py.File(tags_h5, 'r') as f:
        facet_tags_group = None

        if 'MeshTags' in f and 'facet_tags' in f['MeshTags']:
            facet_tags_group = f['MeshTags']['facet_tags']
        elif 'Mesh' in f and 'mesh' in f['Mesh']:
            mesh_group = f['Mesh']['mesh']
            for name in ['facet_tags', 'boundaries']:
                if name in mesh_group:
                    facet_tags_group = mesh_group[name]
                    break

        if facet_tags_group is None:
            raise ValueError("Could not find facet tags in tags.h5")

        facet_tags = facet_tags_group['Values'][:].flatten()
        facet_topo = facet_tags_group['topology'][:]

        report(20, f"Loaded {len(facet_tags)} facets")

    # Filter to membrane facets only (positive tags = internal membranes)
    membrane_mask = facet_tags > 0
    membrane_facets = facet_topo[membrane_mask]
    membrane_tag_values = facet_tags[membrane_mask].astype(np.int32)

    report(25, f"Found {len(membrane_facets)} membrane triangles")

    # Load facet tag to pair mapping
    facet_tag_to_pair = {}
    if facet_map_pickle.exists():
        with open(facet_map_pickle, 'rb') as f:
            facet_tag_to_pair = pickle.load(f)
        report(28, f"Loaded facet-to-pair mapping ({len(facet_tag_to_pair)} tags)")
    else:
        report(28, "No facet_tag_to_pair.pickle found, using fallback v.h5")

    # Discover available vij files
    vij_files = {}
    for vij_path in sim_output_dir.glob('v_*_*.h5'):
        # Parse filename like v_0_1.h5
        parts = vij_path.stem.split('_')
        if len(parts) == 3:
            try:
                i, j = int(parts[1]), int(parts[2])
                vij_files[(i, j)] = vij_path
            except ValueError:
                pass

    report(30, f"Found {len(vij_files)} per-membrane voltage files")

    # Determine coordinate scale
    max_extent = max(
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    )

    if max_extent < 1:
        vertices = vertices * 10000  # cm to micrometers
        mesh_conversion_factor = 0.0001
        report(32, "Converted coordinates from cm to micrometers")
    else:
        mesh_conversion_factor = 0.0001 if max_extent > 10 else 1.0

    # Calculate bounds
    bounds = {
        "x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
        "y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
        "z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())]
    }

    report(35, "Creating expanded mesh (per-facet vertices)...")

    # Create expanded mesh: each facet gets its own 3 vertices
    num_facets = len(membrane_facets)
    expanded_vertices = np.zeros((num_facets * 3, 3), dtype=np.float32)
    expanded_facets = np.zeros((num_facets, 3), dtype=np.uint32)

    # Map each facet to its (i,j) pair index
    unique_pairs = sorted(set(facet_tag_to_pair.values())) if facet_tag_to_pair else []
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}

    # For each facet, store which vij it uses (index into unique_pairs)
    facet_pair_indices = np.zeros(num_facets, dtype=np.int32)

    # Also store the original vertex indices for voltage lookup
    facet_orig_vertices = np.zeros((num_facets, 3), dtype=np.uint32)

    for facet_idx, (facet, tag) in enumerate(zip(membrane_facets, membrane_tag_values)):
        # Expand vertices
        base_vertex = facet_idx * 3
        expanded_vertices[base_vertex:base_vertex+3] = vertices[facet]
        expanded_facets[facet_idx] = [base_vertex, base_vertex + 1, base_vertex + 2]

        # Store original vertex indices
        facet_orig_vertices[facet_idx] = facet

        # Map to pair
        if tag in facet_tag_to_pair:
            pair = facet_tag_to_pair[tag]
            if pair in pair_to_idx:
                facet_pair_indices[facet_idx] = pair_to_idx[pair]

    report(50, f"Expanded to {len(expanded_vertices)} vertices")

    # Save binary data
    report(55, "Writing expanded vertex data...")
    expanded_vertices.tofile(viz_output_dir / "mesh_vertices.bin")

    report(60, "Writing facet data...")
    expanded_facets.tofile(viz_output_dir / "membrane_facets.bin")

    report(65, "Writing facet metadata...")
    membrane_tag_values.tofile(viz_output_dir / "membrane_tags.bin")
    facet_pair_indices.tofile(viz_output_dir / "facet_pair_indices.bin")
    facet_orig_vertices.tofile(viz_output_dir / "facet_orig_vertices.bin")

    # Build metadata (convert numpy types to native Python for JSON serialization)
    metadata = {
        "vertex_count": len(expanded_vertices),
        "facet_count": num_facets,
        "bounds": bounds,
        "mesh_conversion_factor": mesh_conversion_factor,
        "unique_tags": sorted(set(int(t) for t in membrane_tag_values.tolist())),
        "unique_pairs": [[int(x) for x in p] for p in unique_pairs],
        "vij_files": {f"{i}_{j}": str(path.name) for (i, j), path in vij_files.items()},
        "source": str(sim_output_dir),
        "expanded_mesh": True
    }

    report(70, "Writing metadata...")
    with open(viz_output_dir / "mesh_metadata.json", 'w') as mf:
        json.dump(metadata, mf, indent=2)

    # Save pair mapping for voltage lookup (convert numpy types to native Python)
    pair_mapping = {
        "pairs": [[int(x) for x in p] for p in unique_pairs],
        "facet_tag_to_pair": {str(k): [int(x) for x in v] for k, v in facet_tag_to_pair.items()}
    }
    with open(viz_output_dir / "pair_mapping.json", 'w') as f:
        json.dump(pair_mapping, f, indent=2)

    report(100, "Visualization data generated!")

    return metadata


def main():
    """Command-line entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_viz_from_output.py <sim_output_dir> [viz_output_dir]")
        print("Example: python generate_viz_from_output.py pepe36_colored_sim")
        sys.exit(1)

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    sim_output_dir = PROJECT_ROOT / sys.argv[1]

    # Default viz output to viz/data/{sim_name}
    if len(sys.argv) > 2:
        viz_output_dir = Path(sys.argv[2])
    else:
        sim_name = Path(sys.argv[1]).name
        viz_output_dir = SCRIPT_DIR.parent / 'data' / sim_name

    print(f"Generating visualization data...")
    print(f"  Simulation output: {sim_output_dir}")
    print(f"  Visualization output: {viz_output_dir}")
    print()

    metadata = generate_viz_data(sim_output_dir, viz_output_dir)

    print()
    print(f"Done!")
    print(f"  Vertices: {metadata['vertex_count']}")
    print(f"  Facets: {metadata['facet_count']}")


if __name__ == "__main__":
    main()
