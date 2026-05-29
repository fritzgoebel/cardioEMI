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
import os
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


_TET_FACE_INDICES = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)


def _build_cell_surface(cell_topo, cell_tags, vertices, tag):
    """Closed boundary surface of all tets with cell_tag == tag.

    Returns expanded-mesh arrays (each triangle has its own 3 vertices) so
    that per-vertex φ_i can be looked up via an orig-vertex map.
    """
    tag_tets = cell_topo[cell_tags == tag]
    if len(tag_tets) == 0:
        return None

    face_counts = Counter()
    for tet in tag_tets:
        for a, b, c in _TET_FACE_INDICES:
            face = (int(tet[a]), int(tet[b]), int(tet[c]))
            face = tuple(sorted(face))
            face_counts[face] += 1

    boundary_faces = [face for face, count in face_counts.items() if count == 1]
    if not boundary_faces:
        return None

    n_faces = len(boundary_faces)
    expanded_vertices = np.empty((n_faces * 3, 3), dtype=np.float32)
    expanded_facets = np.empty((n_faces, 3), dtype=np.uint32)
    orig_verts = np.empty(n_faces * 3, dtype=np.uint32)

    for fi, face in enumerate(boundary_faces):
        base = fi * 3
        for li, vidx in enumerate(face):
            expanded_vertices[base + li] = vertices[vidx]
            orig_verts[base + li] = vidx
        expanded_facets[fi] = (base, base + 1, base + 2)

    return expanded_vertices, expanded_facets, orig_verts


def _generate_cross_section_data(sim_output_dir, viz_output_dir, vertices,
                                 ecs_tag=None, timestep_keys=None, report=None,
                                 progress_start=0, progress_end=100):
    """Extract per-cell surfaces, ECS tetrahedral volume, and per-timestep
    φ_i / φ_e values for the cross-section visualization mode.

    Returns a dict to merge into mesh_metadata.json under `cross_section`,
    or None if the simulation output lacks the volume potentials.
    """
    solution_h5 = sim_output_dir / 'solution.h5'
    tags_h5 = sim_output_dir / 'tags.h5'
    if not solution_h5.exists():
        report(progress_start, "solution.h5 not found — skipping cross-section export")
        return None

    with h5py.File(tags_h5, 'r') as f:
        cell_topo = f['MeshTags']['cell_tags']['topology'][:]
        cell_tags = f['MeshTags']['cell_tags']['Values'][:].flatten()

    unique_cell_tags = sorted(set(int(t) for t in cell_tags.tolist()))
    if ecs_tag is None:
        ecs_tag = unique_cell_tags[0] if unique_cell_tags else 0
    intra_tags = [t for t in unique_cell_tags if t != ecs_tag]
    timestep_keys = list(timestep_keys or [])

    cells_dir = viz_output_dir / 'cells'
    cells_dir.mkdir(parents=True, exist_ok=True)
    phi_i_dir = viz_output_dir / 'phi_i'
    phi_i_dir.mkdir(parents=True, exist_ok=True)
    ecs_vol_dir = viz_output_dir / 'ecs_volume'
    ecs_vol_dir.mkdir(parents=True, exist_ok=True)
    phi_e_dir = viz_output_dir / 'phi_e'
    phi_e_dir.mkdir(parents=True, exist_ok=True)

    # ECS shell expanded mesh exists from the existing voltage pass — load the
    # orig-vertex map so we can output φ_e aligned with shell vertex order too.
    ecs_shell_orig_path = viz_output_dir / 'ecs_orig_vertices.bin'
    ecs_shell_orig = None
    if ecs_shell_orig_path.exists():
        ecs_shell_orig = np.fromfile(ecs_shell_orig_path, dtype=np.uint32)
        phi_e_shell_dir = viz_output_dir / 'phi_e_shell'
        phi_e_shell_dir.mkdir(parents=True, exist_ok=True)
    else:
        phi_e_shell_dir = None

    progress_span = max(progress_end - progress_start, 1)
    n_tags = max(len(intra_tags), 1)

    cell_orig_verts_by_tag = {}
    cells_meta = {}

    for ti_tag, tag in enumerate(intra_tags):
        report(
            progress_start + int(progress_span * 0.3 * ti_tag / n_tags),
            f"Extracting cell surface for tag {tag}...",
        )
        result = _build_cell_surface(cell_topo, cell_tags, vertices, tag)
        if result is None:
            continue
        exp_verts, exp_facets, orig_verts = result
        exp_verts.tofile(cells_dir / f'{tag}_vertices.bin')
        exp_facets.tofile(cells_dir / f'{tag}_facets.bin')
        orig_verts.tofile(cells_dir / f'{tag}_orig_verts.bin')
        cell_orig_verts_by_tag[tag] = orig_verts
        cells_meta[str(tag)] = {
            "vertex_count": int(len(exp_verts)),
            "facet_count": int(len(exp_facets)),
        }

    report(progress_start + int(progress_span * 0.35), "Extracting ECS volume mesh...")
    ecs_mask = cell_tags == ecs_tag
    ecs_tets = cell_topo[ecs_mask]
    if len(ecs_tets) == 0:
        report(progress_start + int(progress_span * 0.35),
               "No ECS tetrahedra found — cross-section export aborted")
        return None

    ecs_vert_indices = np.unique(ecs_tets.flatten())
    orig_to_local = -np.ones(len(vertices), dtype=np.int64)
    orig_to_local[ecs_vert_indices] = np.arange(len(ecs_vert_indices))
    ecs_tets_local = orig_to_local[ecs_tets].astype(np.uint32)
    ecs_vol_vertices = vertices[ecs_vert_indices].astype(np.float32)
    ecs_vol_vertices.tofile(ecs_vol_dir / 'vertices.bin')
    ecs_tets_local.tofile(ecs_vol_dir / 'tets.bin')
    ecs_vert_indices.astype(np.uint32).tofile(ecs_vol_dir / 'orig_verts.bin')

    report(progress_start + int(progress_span * 0.4),
           f"ECS volume: {len(ecs_vol_vertices)} vertices, {len(ecs_tets_local)} tets")

    phi_i_min = float('inf')
    phi_i_max = float('-inf')
    phi_e_min = float('inf')
    phi_e_max = float('-inf')

    with h5py.File(solution_h5, 'r') as f:
        functions = f['Function']
        # Sanity check: which u_<tag> groups exist
        available = set(functions.keys())

        for ti, ts_key in enumerate(timestep_keys):
            for tag in intra_tags:
                func_name = f'u_{tag}'
                if func_name not in available or tag not in cell_orig_verts_by_tag:
                    continue
                orig_verts = cell_orig_verts_by_tag[tag]
                cell_dir = phi_i_dir / str(tag)
                cell_dir.mkdir(parents=True, exist_ok=True)
                u_group = functions[func_name]
                # solution.h5 may be missing the initial t=0 frame even when
                # v_i_j.h5 has it (main.py writes the initial state to v_i_j
                # but not solution). Emit a zero file so the cross-section
                # index lines up with the voltage timestep index.
                if ts_key not in u_group:
                    phi_i = np.zeros(len(orig_verts), dtype=np.float32)
                    phi_i.tofile(cell_dir / f'{ti}.bin')
                    continue
                u_data = np.nan_to_num(
                    u_group[ts_key][:].flatten(), nan=0.0
                ).astype(np.float32)
                if len(u_data) == 0:
                    np.zeros(len(orig_verts), dtype=np.float32).tofile(
                        cell_dir / f'{ti}.bin')
                    continue
                safe = np.clip(orig_verts, 0, len(u_data) - 1)
                phi_i = u_data[safe].copy()
                phi_i[orig_verts >= len(u_data)] = 0.0
                phi_i.tofile(cell_dir / f'{ti}.bin')
                if phi_i.size:
                    phi_i_min = min(phi_i_min, float(phi_i.min()))
                    phi_i_max = max(phi_i_max, float(phi_i.max()))

            ecs_func = f'u_{ecs_tag}'
            ecs_present = ecs_func in available and ts_key in functions[ecs_func]
            if ecs_present:
                u_e = np.nan_to_num(
                    functions[ecs_func][ts_key][:].flatten(), nan=0.0
                ).astype(np.float32)
            else:
                u_e = np.zeros(0, dtype=np.float32)

            if len(u_e):
                safe = np.clip(ecs_vert_indices, 0, len(u_e) - 1)
                phi_e = u_e[safe].copy()
                phi_e[ecs_vert_indices >= len(u_e)] = 0.0
            else:
                phi_e = np.zeros(len(ecs_vert_indices), dtype=np.float32)
            phi_e.tofile(phi_e_dir / f'{ti}.bin')
            if phi_e.size:
                phi_e_min = min(phi_e_min, float(phi_e.min()))
                phi_e_max = max(phi_e_max, float(phi_e.max()))

            if phi_e_shell_dir is not None and ecs_shell_orig is not None:
                if len(u_e):
                    safe_shell = np.clip(ecs_shell_orig, 0, len(u_e) - 1)
                    phi_shell = u_e[safe_shell].copy()
                    phi_shell[ecs_shell_orig >= len(u_e)] = 0.0
                else:
                    phi_shell = np.zeros(len(ecs_shell_orig), dtype=np.float32)
                phi_shell.tofile(phi_e_shell_dir / f'{ti}.bin')
                if phi_shell.size:
                    phi_e_min = min(phi_e_min, float(phi_shell.min()))
                    phi_e_max = max(phi_e_max, float(phi_shell.max()))

            if (ti + 1) % 10 == 0 or ti + 1 == len(timestep_keys):
                pct = progress_start + int(
                    progress_span * (0.4 + 0.6 * (ti + 1) / max(len(timestep_keys), 1))
                )
                report(pct, f"Cross-section timestep {ti + 1}/{len(timestep_keys)}")

    if phi_i_min == float('inf'):
        phi_i_min, phi_i_max = 0.0, 0.0
    if phi_e_min == float('inf'):
        phi_e_min, phi_e_max = 0.0, 0.0

    return {
        "ecs_tag": int(ecs_tag),
        "cell_tags": [int(t) for t in intra_tags if t in cell_orig_verts_by_tag],
        "cells": cells_meta,
        "ecs_volume": {
            "vertex_count": int(len(ecs_vol_vertices)),
            "tet_count": int(len(ecs_tets_local)),
        },
        "phi_i_range": [phi_i_min, phi_i_max],
        "phi_e_range": [phi_e_min, phi_e_max],
    }


def generate_viz_data(sim_output_dir: Path, viz_output_dir: Path, progress_callback=None, membrane_only=False) -> dict:
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
    if not membrane_only:
        exterior_mask = facet_tags <= 0
        exterior_facets = facet_topo[exterior_mask]
        report(26, f"Found {len(exterior_facets)} exterior boundary triangles")
    else:
        exterior_facets = np.array([])

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

    # Load DOF rank ownership if available (skip for membrane_only)
    dof_rank_info = None
    dof_contributors_info = None
    if not membrane_only:
        dof_ranks_pickle = sim_output_dir / 'dof_ranks.pickle'
        if dof_ranks_pickle.exists():
            with open(dof_ranks_pickle, 'rb') as f:
                dof_rank_info = pickle.load(f)
            report(32, f"Loaded DOF rank ownership ({dof_rank_info['num_ranks']} ranks)")

        # Load DOF contribution data if available (which ranks contribute to each DOF)
        dof_contributors_pickle = sim_output_dir / 'dof_contributors.pickle'
        if dof_contributors_pickle.exists():
            with open(dof_contributors_pickle, 'rb') as f:
                dof_contributors_info = pickle.load(f)
            report(32, f"Loaded DOF contribution data ({len(dof_contributors_info['contributors'])} DOFs)")

    # Extract partition cut facets from cell topology if we have rank data (skip for membrane_only)
    # These are internal facets where adjacent intracellular cells belong to different MPI ranks
    # We exclude ECS (cell tag 0) - only show cuts through intracellular space
    partition_cut_facets = None
    if dof_rank_info is not None and not membrane_only:
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

    # If we have rank data, assign facets to ranks based on contribution
    if dof_rank_info is not None:
        dof_ranks = dof_rank_info['ranks']
        num_ranks = dof_rank_info['num_ranks']

        # Get contributors data if available
        contributors = dof_contributors_info['contributors'] if dof_contributors_info else None

        expanded_vertex_list = []
        expanded_facet_list = []
        facet_pair_list = []
        facet_orig_verts_list = []
        expanded_rank_list = []
        expanded_tag_list = []

        shared_facet_count = 0
        vertex_idx = 0

        for facet, tag in zip(membrane_facets, membrane_tag_values):
            if contributors:
                # Find ranks that contribute to ALL three vertices of this facet
                # A rank shows this facet if it contributes to all three DOFs
                contributing_ranks_per_vertex = [
                    set(contributors.get(int(v), [dof_ranks[v] if v < len(dof_ranks) else 0]))
                    for v in facet
                ]
                # Intersection: ranks that contribute to all three vertices
                common_ranks = contributing_ranks_per_vertex[0] & contributing_ranks_per_vertex[1] & contributing_ranks_per_vertex[2]

                if len(common_ranks) == 0:
                    # Fallback to owner rank if no common contributors
                    common_ranks = {dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0}

                if len(common_ranks) > 1:
                    shared_facet_count += 1

                # Create facet for each contributing rank
                for rank in sorted(common_ranks):
                    expanded_vertex_list.append(vertices[facet])
                    expanded_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                    expanded_rank_list.extend([rank, rank, rank])
                    facet_orig_verts_list.append(facet)
                    expanded_tag_list.append(tag)

                    if tag in facet_tag_to_pair:
                        pair = facet_tag_to_pair[tag]
                        facet_pair_list.append(pair_to_idx.get(pair, 0))
                    else:
                        facet_pair_list.append(0)

                    vertex_idx += 3
            else:
                # No contribution data: use ownership (single rank per facet)
                rank = dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0

                expanded_vertex_list.append(vertices[facet])
                expanded_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
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

        report(50, f"Expanded to {len(expanded_vertices)} vertices ({shared_facet_count} facets shared by multiple ranks)")

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

        # If we have rank data, assign facets based on contribution
        if dof_rank_info is not None:
            dof_ranks = dof_rank_info['ranks']
            contributors = dof_contributors_info['contributors'] if dof_contributors_info else None

            ext_vertex_list = []
            ext_facet_list = []
            ext_rank_list = []
            ext_orig_verts_list = []  # Track original vertex indices for interface highlighting
            ext_shared_count = 0
            vertex_idx = 0

            for facet in exterior_facets:
                if contributors:
                    # Find ranks that contribute to ALL three vertices
                    contributing_ranks_per_vertex = [
                        set(contributors.get(int(v), [dof_ranks[v] if v < len(dof_ranks) else 0]))
                        for v in facet
                    ]
                    common_ranks = contributing_ranks_per_vertex[0] & contributing_ranks_per_vertex[1] & contributing_ranks_per_vertex[2]

                    if len(common_ranks) == 0:
                        common_ranks = {dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0}

                    if len(common_ranks) > 1:
                        ext_shared_count += 1

                    for rank in sorted(common_ranks):
                        ext_vertex_list.append(vertices[facet])
                        ext_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                        ext_rank_list.extend([rank, rank, rank])
                        ext_orig_verts_list.append(facet)  # Track original vertices
                        vertex_idx += 3
                else:
                    # No contribution data: use ownership
                    rank = dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0
                    ext_vertex_list.append(vertices[facet])
                    ext_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                    ext_rank_list.extend([rank, rank, rank])
                    ext_orig_verts_list.append(facet)  # Track original vertices
                    vertex_idx += 3

            ext_expanded_vertices = np.vstack(ext_vertex_list).astype(np.float32)
            ext_expanded_facets = np.array(ext_facet_list, dtype=np.uint32)
            ext_expanded_ranks = np.array(ext_rank_list, dtype=np.int32)
            ext_orig_vertices = np.array(ext_orig_verts_list, dtype=np.uint32)
            num_ext_output_facets = len(ext_facet_list)

            ext_expanded_vertices.tofile(viz_output_dir / "ecs_vertices.bin")
            ext_expanded_facets.tofile(viz_output_dir / "ecs_facets.bin")
            ext_expanded_ranks.tofile(viz_output_dir / "ecs_ranks.bin")
            ext_orig_vertices.tofile(viz_output_dir / "ecs_orig_vertices.bin")

            report(75, f"Saved {num_ext_output_facets} ECS facets ({ext_shared_count} shared by multiple ranks)")
        else:
            # No rank data: simple expansion
            num_ext_facets = len(exterior_facets)
            ext_expanded_vertices = np.zeros((num_ext_facets * 3, 3), dtype=np.float32)
            ext_expanded_facets = np.zeros((num_ext_facets, 3), dtype=np.uint32)
            ext_orig_vertices = np.zeros((num_ext_facets, 3), dtype=np.uint32)

            for facet_idx, facet in enumerate(exterior_facets):
                base_vertex = facet_idx * 3
                ext_expanded_vertices[base_vertex:base_vertex+3] = vertices[facet]
                ext_expanded_facets[facet_idx] = [base_vertex, base_vertex + 1, base_vertex + 2]
                ext_orig_vertices[facet_idx] = facet

            ext_expanded_vertices.tofile(viz_output_dir / "ecs_vertices.bin")
            ext_expanded_facets.tofile(viz_output_dir / "ecs_facets.bin")
            ext_orig_vertices.tofile(viz_output_dir / "ecs_orig_vertices.bin")
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
        contributors = dof_contributors_info['contributors'] if dof_contributors_info else None

        cut_vertex_list = []
        cut_facet_list = []
        cut_rank_list = []
        cut_shared_count = 0
        vertex_idx = 0

        for facet in partition_cut_facets:
            if contributors:
                # Find ranks that contribute to ALL three vertices
                contributing_ranks_per_vertex = [
                    set(contributors.get(int(v), [dof_ranks[v] if v < len(dof_ranks) else 0]))
                    for v in facet
                ]
                common_ranks = contributing_ranks_per_vertex[0] & contributing_ranks_per_vertex[1] & contributing_ranks_per_vertex[2]

                if len(common_ranks) == 0:
                    common_ranks = {dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0}

                if len(common_ranks) > 1:
                    cut_shared_count += 1

                for rank in sorted(common_ranks):
                    cut_vertex_list.append(vertices[facet])
                    cut_facet_list.append([vertex_idx, vertex_idx + 1, vertex_idx + 2])
                    cut_rank_list.extend([rank, rank, rank])
                    vertex_idx += 3
            else:
                # No contribution data: use ownership
                rank = dof_ranks[facet[0]] if facet[0] < len(dof_ranks) else 0
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

            shared_msg = f" ({cut_shared_count} shared by multiple ranks)" if contributors else ""
            report(80, f"Saved {num_cut_output_facets} partition cut facets{shared_msg}")

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

    # Generate per-timestep voltage binary files
    report(75, "Generating voltage binaries...")
    voltages_dir = viz_output_dir / 'voltages'
    voltages_dir.mkdir(parents=True, exist_ok=True)

    def parse_time(key):
        return float(key.replace('_', '.'))

    times = []
    timestep_keys = []
    v_min_all = float('inf')
    v_max_all = float('-inf')
    num_facets = len(facet_orig_vertices)

    if len(vij_files) > 0:
        # Per-facet voltage from v_i_j.h5 files
        first_vij_path = list(vij_files.values())[0]
        with h5py.File(first_vij_path, 'r') as f:
            func_name = list(f['Function'].keys())[0]
            timestep_keys = sorted(f['Function'][func_name].keys(), key=parse_time)

        max_timesteps = 100
        if len(timestep_keys) > max_timesteps:
            step_size = len(timestep_keys) // max_timesteps
            timestep_keys = timestep_keys[::step_size][:max_timesteps]

        vij_data = {}
        for pair, vij_path in vij_files.items():
            with h5py.File(vij_path, 'r') as f:
                func_name = list(f['Function'].keys())[0]
                v_group = f['Function'][func_name]
                vij_data[pair] = {}
                for key in timestep_keys:
                    if key in v_group:
                        vij_data[pair][key] = np.nan_to_num(
                            v_group[key][:].flatten(), nan=0.0).astype(np.float32)

        # Pre-compute per-pair facet groupings for vectorized voltage lookup
        pair_facet_groups = {}
        for pair_idx_val, pair in enumerate(unique_pairs):
            mask = facet_pair_indices == pair_idx_val
            if np.any(mask):
                facet_idxs = np.where(mask)[0]
                pair_facet_groups[pair] = (facet_idxs, facet_orig_vertices[facet_idxs])

        num_workers = min(os.cpu_count() or 1, len(timestep_keys))
        report(76, f"Processing {len(timestep_keys)} timesteps with {num_workers} workers...")

        def process_timestep(args):
            ti, key = args
            expanded_voltages = np.zeros(num_facets * 3, dtype=np.float32)
            for pair, (fidxs, orig_verts) in pair_facet_groups.items():
                if pair in vij_data and key in vij_data[pair]:
                    v_data = vij_data[pair][key]
                    safe_verts = np.clip(orig_verts, 0, len(v_data) - 1)
                    voltages = v_data[safe_verts]  # (N, 3)
                    # Zero out any that were clipped (original >= len(v_data))
                    voltages[orig_verts >= len(v_data)] = 0.0
                    base = fidxs * 3
                    expanded_voltages[base] = voltages[:, 0]
                    expanded_voltages[base + 1] = voltages[:, 1]
                    expanded_voltages[base + 2] = voltages[:, 2]
            out_path = str(voltages_dir / f'{ti}.bin')
            expanded_voltages.tofile(out_path)
            return ti, float(np.min(expanded_voltages)), float(np.max(expanded_voltages)), parse_time(key)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_timestep, (ti, key)): ti
                       for ti, key in enumerate(timestep_keys)}
            results = {}
            done_count = 0
            for future in as_completed(futures):
                ti, vmin, vmax, t = future.result()
                results[ti] = (vmin, vmax, t)
                done_count += 1
                if done_count % 10 == 0:
                    report(75 + int(25 * done_count / len(timestep_keys)),
                           f"Voltage timestep {done_count}/{len(timestep_keys)}")

        for ti in range(len(timestep_keys)):
            vmin, vmax, t = results[ti]
            v_min_all = min(v_min_all, vmin)
            v_max_all = max(v_max_all, vmax)
            times.append(t)
    else:
        # Fallback: single v.h5
        v_h5 = sim_output_dir / 'v.h5'
        if v_h5.exists():
            with h5py.File(v_h5, 'r') as f:
                v_group = f['Function']['v']
                timestep_keys = sorted(v_group.keys(), key=parse_time)
                max_timesteps = 100
                if len(timestep_keys) > max_timesteps:
                    step_size = len(timestep_keys) // max_timesteps
                    timestep_keys = timestep_keys[::step_size][:max_timesteps]
                for ti, key in enumerate(timestep_keys):
                    v_data = np.nan_to_num(
                        v_group[key][:].flatten(), nan=0.0).astype(np.float32)
                    v_data.tofile(voltages_dir / f'{ti}.bin')
                    v_min_all = min(v_min_all, float(np.min(v_data)))
                    v_max_all = max(v_max_all, float(np.max(v_data)))
                    times.append(parse_time(key))

    if v_min_all == float('inf'):
        v_min_all, v_max_all = 0.0, 0.0

    # Save voltage metadata
    metadata['times'] = times
    metadata['vMin'] = v_min_all
    metadata['vMax'] = v_max_all

    # Cross-section data (per-cell surfaces, ECS volume, φ_i / φ_e per timestep).
    # Skipped in membrane_only mode and when solution.h5 is unavailable.
    if not membrane_only and timestep_keys:
        report(96, "Extracting cross-section volume data...")
        cs_meta = _generate_cross_section_data(
            sim_output_dir, viz_output_dir, vertices,
            ecs_tag=None,
            timestep_keys=timestep_keys,
            report=report,
            progress_start=96,
            progress_end=99,
        )
        if cs_meta is not None:
            metadata['cross_section'] = cs_meta

    with open(viz_output_dir / "mesh_metadata.json", 'w') as mf:
        json.dump(metadata, mf, indent=2)

    report(100, "Visualization data generated!")

    return metadata


def main():
    """Command-line entry point."""
    import sys

    # Parse --membrane-only flag
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    membrane_only = '--membrane-only' in sys.argv

    if len(args) < 1:
        print("Usage: python generate_viz_from_output.py [--membrane-only] <sim_output_dir> [viz_output_dir]")
        print("Example: python generate_viz_from_output.py pepe36_colored_sim")
        sys.exit(1)

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    sim_output_dir = PROJECT_ROOT / args[0]

    # Default viz output to viz/data/{sim_name}
    if len(args) > 1:
        viz_output_dir = Path(args[1])
    else:
        sim_name = Path(args[0]).name
        viz_output_dir = SCRIPT_DIR.parent / 'data' / sim_name

    print(f"Generating visualization data...")
    print(f"  Simulation output: {sim_output_dir}")
    print(f"  Visualization output: {viz_output_dir}")
    if membrane_only:
        print(f"  Mode: membrane only (no ECS/rank/partition data)")
    print()

    def progress(percent, message):
        print(f"PROGRESS:{percent}:{message}", flush=True)

    metadata = generate_viz_data(sim_output_dir, viz_output_dir, progress_callback=progress, membrane_only=membrane_only)

    print()
    print(f"Done!")
    print(f"  Vertices: {metadata['vertex_count']}")
    print(f"  Facets: {metadata['facet_count']}")


if __name__ == "__main__":
    main()
