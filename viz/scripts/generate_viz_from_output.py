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

    # Extract exterior boundary facets (ECS surface) - tag 0 or negative
    exterior_mask = facet_tags <= 0
    exterior_facets = facet_topo[exterior_mask]
    report(26, f"Found {len(exterior_facets)} exterior boundary triangles")

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

    # Load DOF rank ownership if available
    dof_ranks_pickle = sim_output_dir / 'dof_ranks.pickle'
    dof_rank_info = None
    if dof_ranks_pickle.exists():
        with open(dof_ranks_pickle, 'rb') as f:
            dof_rank_info = pickle.load(f)
        report(32, f"Loaded DOF rank ownership ({dof_rank_info['num_ranks']} ranks)")

    # Extract partition cut facets from cell topology if we have rank data
    # These are internal facets where adjacent intracellular cells belong to different MPI ranks
    # We exclude ECS (cell tag 0) - only show cuts through intracellular space
    partition_cut_facets = None
    if dof_rank_info is not None:
        report(33, "Extracting partition cut facets from cell topology...")
        dof_ranks = dof_rank_info['ranks']
        from collections import defaultdict, Counter

        # Load cell topology and tags
        with h5py.File(tags_h5, 'r') as f:
            cell_topo = f['MeshTags']['cell_tags']['topology'][:]
            cell_tags = f['MeshTags']['cell_tags']['Values'][:].flatten()

        # Each tetrahedron has 4 triangular faces
        tet_face_indices = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]

        # Build facet -> cells adjacency map
        report(33, "Building facet-to-cell adjacency...")
        facet_to_cells = defaultdict(list)
        for cell_idx, cell in enumerate(cell_topo):
            for face_idx in tet_face_indices:
                face = tuple(sorted([cell[i] for i in face_idx]))
                facet_to_cells[face].append(cell_idx)

        # Determine cell rank by majority vote of vertex ranks
        def get_cell_rank(cell_idx):
            cell = cell_topo[cell_idx]
            ranks = [dof_ranks[v] for v in cell if v < len(dof_ranks)]
            if ranks:
                return Counter(ranks).most_common(1)[0][0]
            return 0

        # Find facets where adjacent intracellular cells have different ranks
        # Only include facets where BOTH cells are intracellular (cell tag > 0)
        report(34, "Finding partition boundary facets (intracellular only)...")
        partition_boundary_facets = []
        for face, cells in facet_to_cells.items():
            if len(cells) == 2:  # Internal facet (shared by 2 cells)
                # Check if both cells are intracellular (not ECS)
                tag0 = cell_tags[cells[0]]
                tag1 = cell_tags[cells[1]]
                if tag0 > 0 and tag1 > 0:  # Both intracellular
                    rank0 = get_cell_rank(cells[0])
                    rank1 = get_cell_rank(cells[1])
                    if rank0 != rank1:
                        partition_boundary_facets.append(face)

        if partition_boundary_facets:
            partition_cut_facets = np.array(partition_boundary_facets, dtype=np.int64)
            report(35, f"Found {len(partition_cut_facets)} partition cut facets (intracellular)")
        else:
            report(35, "No partition cut facets found")

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

    # Map each facet to its (i,j) pair index
    unique_pairs = sorted(set(facet_tag_to_pair.values())) if facet_tag_to_pair else []
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}

    # If we have rank data, we need to duplicate boundary facets for each rank
    # so that each partition has its own independent mesh
    if dof_rank_info is not None:
        dof_ranks = dof_rank_info['ranks']
        num_ranks = dof_rank_info['num_ranks']

        # First pass: count total facets including duplicates for boundary facets
        expanded_vertex_list = []
        expanded_facet_list = []
        facet_pair_list = []
        facet_orig_verts_list = []
        expanded_rank_list = []
        expanded_tag_list = []

        boundary_facet_count = 0
        vertex_idx = 0

        for facet, tag in zip(membrane_facets, membrane_tag_values):
            # Get ranks for this facet's vertices
            facet_ranks = [dof_ranks[v] if v < len(dof_ranks) else 0 for v in facet]
            unique_facet_ranks = set(facet_ranks)

            if len(unique_facet_ranks) == 1:
                # Interior facet: all vertices same rank, create once
                rank = facet_ranks[0]
                expanded_vertex_list.append(vertices[facet])
                expanded_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                expanded_rank_list.extend([rank, rank, rank])
                facet_orig_verts_list.append(facet)
                expanded_tag_list.append(tag)

                # Map to pair
                if tag in facet_tag_to_pair:
                    pair = facet_tag_to_pair[tag]
                    facet_pair_list.append(pair_to_idx.get(pair, 0))
                else:
                    facet_pair_list.append(0)

                vertex_idx += 3
            else:
                # Boundary facet: duplicate for each rank that owns at least one vertex
                boundary_facet_count += 1
                for rank in unique_facet_ranks:
                    expanded_vertex_list.append(vertices[facet])
                    expanded_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                    # Assign ALL vertices to this rank (so facet moves uniformly)
                    expanded_rank_list.extend([rank, rank, rank])
                    facet_orig_verts_list.append(facet)
                    expanded_tag_list.append(tag)

                    if tag in facet_tag_to_pair:
                        pair = facet_tag_to_pair[tag]
                        facet_pair_list.append(pair_to_idx.get(pair, 0))
                    else:
                        facet_pair_list.append(0)

                    vertex_idx += 3

        # Convert to numpy arrays
        num_output_facets = len(expanded_facet_list)
        expanded_vertices = np.vstack(expanded_vertex_list).astype(np.float32)
        expanded_facets = np.array(expanded_facet_list, dtype=np.uint32)
        facet_pair_indices = np.array(facet_pair_list, dtype=np.int32)
        facet_orig_vertices = np.array(facet_orig_verts_list, dtype=np.uint32)
        expanded_ranks = np.array(expanded_rank_list, dtype=np.int32)
        expanded_tags = np.array(expanded_tag_list, dtype=np.int32)

        report(50, f"Expanded to {len(expanded_vertices)} vertices ({boundary_facet_count} boundary facets duplicated)")

    else:
        # No rank data: simple expansion without duplication
        num_facets = len(membrane_facets)
        expanded_vertices = np.zeros((num_facets * 3, 3), dtype=np.float32)
        expanded_facets = np.zeros((num_facets, 3), dtype=np.uint32)
        facet_pair_indices = np.zeros(num_facets, dtype=np.int32)
        facet_orig_vertices = np.zeros((num_facets, 3), dtype=np.uint32)
        expanded_ranks = None
        expanded_tags = membrane_tag_values.copy()

        for facet_idx, (facet, tag) in enumerate(zip(membrane_facets, membrane_tag_values)):
            base_vertex = facet_idx * 3
            expanded_vertices[base_vertex:base_vertex+3] = vertices[facet]
            expanded_facets[facet_idx] = [base_vertex, base_vertex + 1, base_vertex + 2]
            facet_orig_vertices[facet_idx] = facet

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
    expanded_tags.tofile(viz_output_dir / "membrane_tags.bin")
    facet_pair_indices.tofile(viz_output_dir / "facet_pair_indices.bin")
    facet_orig_vertices.tofile(viz_output_dir / "facet_orig_vertices.bin")

    # Save rank data if available (already computed during expansion)
    if dof_rank_info is not None and expanded_ranks is not None:
        report(68, "Writing DOF rank data...")
        num_ranks = dof_rank_info['num_ranks']

        expanded_ranks.tofile(viz_output_dir / "dof_ranks.bin")

        # Compute rank centroids for explosion effect
        rank_centroids = []
        for rank in range(num_ranks):
            rank_mask = expanded_ranks == rank
            if np.any(rank_mask):
                rank_verts = expanded_vertices[rank_mask]
                centroid = rank_verts.mean(axis=0)
                rank_centroids.append(centroid.tolist())
            else:
                rank_centroids.append([0.0, 0.0, 0.0])

        # Global centroid
        global_centroid = expanded_vertices.mean(axis=0).tolist()

        # Also save rank metadata
        with open(viz_output_dir / "rank_metadata.json", 'w') as f:
            json.dump({
                'num_ranks': int(num_ranks),
                'global_size': int(dof_rank_info['global_size']),
                'rank_centroids': rank_centroids,
                'global_centroid': global_centroid
            }, f, indent=2)

    # Save exterior facets for ECS visualization
    if len(exterior_facets) > 0:
        report(72, "Writing exterior boundary (ECS) data...")

        # If we have rank data, duplicate boundary facets for each partition
        if dof_rank_info is not None:
            dof_ranks = dof_rank_info['ranks']
            ext_vertex_list = []
            ext_facet_list = []
            ext_rank_list = []
            ext_boundary_count = 0
            vertex_idx = 0

            for facet in exterior_facets:
                facet_ranks = [dof_ranks[v] if v < len(dof_ranks) else 0 for v in facet]
                unique_facet_ranks = set(facet_ranks)

                if len(unique_facet_ranks) == 1:
                    # Interior facet
                    rank = facet_ranks[0]
                    ext_vertex_list.append(vertices[facet])
                    ext_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                    ext_rank_list.extend([rank, rank, rank])
                    vertex_idx += 3
                else:
                    # Boundary facet: duplicate for each rank
                    ext_boundary_count += 1
                    for rank in unique_facet_ranks:
                        ext_vertex_list.append(vertices[facet])
                        ext_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                        ext_rank_list.extend([rank, rank, rank])
                        vertex_idx += 3

            ext_expanded_vertices = np.vstack(ext_vertex_list).astype(np.float32)
            ext_expanded_facets = np.array(ext_facet_list, dtype=np.uint32)
            ext_expanded_ranks = np.array(ext_rank_list, dtype=np.int32)
            num_ext_output_facets = len(ext_facet_list)

            ext_expanded_vertices.tofile(viz_output_dir / "ecs_vertices.bin")
            ext_expanded_facets.tofile(viz_output_dir / "ecs_facets.bin")
            ext_expanded_ranks.tofile(viz_output_dir / "ecs_ranks.bin")

            report(75, f"Saved {num_ext_output_facets} ECS facets ({ext_boundary_count} boundary duplicated)")
        else:
            # No rank data: simple expansion
            num_ext_facets = len(exterior_facets)
            ext_expanded_vertices = np.zeros((num_ext_facets * 3, 3), dtype=np.float32)
            ext_expanded_facets = np.zeros((num_ext_facets, 3), dtype=np.uint32)

            for facet_idx, facet in enumerate(exterior_facets):
                base_vertex = facet_idx * 3
                ext_expanded_vertices[base_vertex:base_vertex+3] = vertices[facet]
                ext_expanded_facets[facet_idx] = [base_vertex, base_vertex + 1, base_vertex + 2]

            ext_expanded_vertices.tofile(viz_output_dir / "ecs_vertices.bin")
            ext_expanded_facets.tofile(viz_output_dir / "ecs_facets.bin")
            num_ext_output_facets = num_ext_facets

            report(75, f"Saved {num_ext_facets} ECS facets")
    else:
        num_ext_output_facets = 0
        ext_expanded_vertices = np.array([])

    # Save partition cut facets (internal facets at partition boundaries)
    num_cut_output_facets = 0
    if partition_cut_facets is not None and len(partition_cut_facets) > 0 and dof_rank_info is not None:
        report(78, "Writing partition cut facets...")
        dof_ranks = dof_rank_info['ranks']

        cut_vertex_list = []
        cut_facet_list = []
        cut_rank_list = []
        vertex_idx = 0

        for facet in partition_cut_facets:
            facet_ranks = [dof_ranks[v] if v < len(dof_ranks) else 0 for v in facet]
            unique_facet_ranks = set(facet_ranks)

            # Duplicate for each rank that has vertices on this facet
            for rank in unique_facet_ranks:
                cut_vertex_list.append(vertices[facet])
                cut_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                cut_rank_list.extend([rank, rank, rank])
                vertex_idx += 3

        if cut_vertex_list:
            cut_expanded_vertices = np.vstack(cut_vertex_list).astype(np.float32)
            cut_expanded_facets = np.array(cut_facet_list, dtype=np.uint32)
            cut_expanded_ranks = np.array(cut_rank_list, dtype=np.int32)
            num_cut_output_facets = len(cut_facet_list)

            cut_expanded_vertices.tofile(viz_output_dir / "cut_vertices.bin")
            cut_expanded_facets.tofile(viz_output_dir / "cut_facets.bin")
            cut_expanded_ranks.tofile(viz_output_dir / "cut_ranks.bin")

            report(80, f"Saved {num_cut_output_facets} partition cut facets")

    # Build metadata (convert numpy types to native Python for JSON serialization)
    # Use actual output counts (which may include duplicated boundary facets)
    num_output_facets = len(expanded_facets)
    metadata = {
        "vertex_count": len(expanded_vertices),
        "ecs_vertex_count": len(ext_expanded_vertices) if len(exterior_facets) > 0 else 0,
        "ecs_facet_count": num_ext_output_facets,
        "facet_count": num_output_facets,
        "bounds": bounds,
        "mesh_conversion_factor": mesh_conversion_factor,
        "unique_tags": sorted(set(int(t) for t in membrane_tag_values.tolist())),
        "unique_pairs": [[int(x) for x in p] for p in unique_pairs],
        "vij_files": {f"{i}_{j}": str(path.name) for (i, j), path in vij_files.items()},
        "source": str(sim_output_dir),
        "expanded_mesh": True,
        "has_rank_data": dof_rank_info is not None,
        "num_ranks": int(dof_rank_info['num_ranks']) if dof_rank_info else None,
        "cut_facet_count": num_cut_output_facets,
        "cut_vertex_count": num_cut_output_facets * 3
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
