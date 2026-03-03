"""
Custom mesh partitioning utilities for component-based MPI distribution.

This module provides functions to load meshes with custom partitioning that
keeps ECS+cell component pairs together on the same MPI rank.

For component partitioning:
- Even tags (0, 2, 4, ...) are ECS chunks
- Odd tags (1, 3, 5, ...) are cells inside those ECS chunks
- Each component = (2*i, 2*i+1) = one ECS chunk + one cell
- METIS partitions components to minimize interface cuts between partitions
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI


def compute_component_partitions(tag_interfaces, num_partitions):
    """Compute METIS-based partitioning of component graph.

    Components are pairs of consecutive tags: (2*i, 2*i+1) = (ECS chunk, cell).

    Args:
        tag_interfaces: dict mapping tag -> set of interface IDs
        num_partitions: number of partitions (MPI ranks)

    Returns:
        tag_to_partition: dict mapping original tag -> partition ID
        stats: dict with partitioning statistics
    """
    try:
        import pymetis
        has_pymetis = True
    except ImportError:
        has_pymetis = False

    num_tags = len(tag_interfaces)
    num_components = num_tags // 2

    # Build component connectivity graph
    component_interfaces = {}
    for c in range(num_components):
        ecs_tag = 2 * c
        cell_tag = 2 * c + 1
        ecs_ifaces = set(int(x) for x in tag_interfaces.get(ecs_tag, set()) if int(x) != ecs_tag)
        cell_ifaces = set(int(x) for x in tag_interfaces.get(cell_tag, set()) if int(x) != cell_tag)
        component_interfaces[c] = ecs_ifaces | cell_ifaces

    # Find shared interfaces between components
    interface_to_components = defaultdict(set)
    for c, ifaces in component_interfaces.items():
        for iface in ifaces:
            interface_to_components[iface].add(c)

    # Build adjacency list for METIS
    adjacency = [[] for _ in range(num_components)]
    for iface, components in interface_to_components.items():
        if len(components) == 2:
            c1, c2 = list(components)
            if c2 not in adjacency[c1]:
                adjacency[c1].append(c2)
            if c1 not in adjacency[c2]:
                adjacency[c2].append(c1)

    num_edges = sum(len(adj) for adj in adjacency) // 2

    # Partition using METIS or fallback to round-robin
    if has_pymetis and num_partitions > 1:
        n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency)
        method = 'metis'
    else:
        membership = [i % num_partitions for i in range(num_components)]
        n_cuts = 0
        method = 'round_robin'

    # Create tag -> partition mapping
    tag_to_partition = {}
    for c in range(num_components):
        tag_to_partition[2 * c] = membership[c]      # ECS tag
        tag_to_partition[2 * c + 1] = membership[c]  # Cell tag

    # Compute balance statistics
    partition_sizes = defaultdict(int)
    for part in membership:
        partition_sizes[part] += 1

    stats = {
        'num_components': num_components,
        'num_partitions': num_partitions,
        'num_edges': num_edges,
        'edge_cuts': n_cuts,
        'method': method,
        'components_per_partition': dict(partition_sizes),
    }

    return tag_to_partition, stats


def load_mesh_with_component_partitioning(
    comm,
    colored_mesh_file,
    original_mesh_file=None,
    ghost_mode=None,
):
    """Load colored mesh with component-based custom partitioning.

    This function loads the graph-colored mesh but partitions it based on
    component structure from the original mesh. This keeps ECS+cell pairs
    together on the same MPI rank while using the efficient 4-tag colored mesh
    for simulation.

    Args:
        comm: MPI communicator
        colored_mesh_file: Path to the colored mesh XDMF file (4 tags)
        original_mesh_file: Path to the original mesh XDMF file (44 tags).
                           If None, derived from colored_mesh_file.
        ghost_mode: GhostMode for mesh (default: shared_facet)

    Returns:
        mesh: The partitioned DOLFINx mesh
        subdomains: MeshTags for cell subdomains (colored tags: 0-3)
        boundaries: MeshTags for facet boundaries
    """
    from dolfinx import mesh as dfx_mesh, io
    from dolfinx.cpp.graph import AdjacencyList_int32
    import basix
    from scipy.spatial import cKDTree

    if ghost_mode is None:
        ghost_mode = dfx_mesh.GhostMode.shared_facet

    # Derive original mesh file if not specified
    if original_mesh_file is None:
        original_mesh_file = colored_mesh_file.replace("_colored", "")
        if original_mesh_file == colored_mesh_file:
            raise ValueError(
                "Could not derive original_mesh_file from colored_mesh_file. "
                "Please specify original_mesh_file explicitly."
            )

    rank = comm.rank
    num_partitions = comm.size

    # Load all data on rank 0 and compute partitioning
    if rank == 0:
        # Load original mesh interface connectivity for METIS partitioning
        orig_pickle = Path(original_mesh_file).with_suffix('.pickle')
        with open(orig_pickle, 'rb') as f:
            orig_tag_interfaces = pickle.load(f)

        # Compute METIS partitioning based on components
        tag_to_partition, stats = compute_component_partitions(orig_tag_interfaces, num_partitions)
        print(f"Component partitioning: {stats['method']}, "
              f"{stats['num_components']} components, "
              f"{stats['edge_cuts']} edge cuts")

        # Read original mesh to get cell -> original_tag mapping
        with io.XDMFFile(MPI.COMM_SELF, original_mesh_file, 'r') as xdmf:
            orig_mesh = xdmf.read_mesh(ghost_mode=dfx_mesh.GhostMode.none)
            orig_subdomains = xdmf.read_meshtags(orig_mesh, name="cell_tags")

        orig_cell_tags = orig_subdomains.values

        # Compute cell centroids for the original mesh (vectorized)
        orig_coords = orig_mesh.geometry.x
        orig_cells = orig_mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
        orig_centroids = orig_coords[orig_cells].mean(axis=1)

        # Build KD-tree for original mesh centroids
        orig_tree = cKDTree(orig_centroids)

        # Read colored mesh
        with io.XDMFFile(MPI.COMM_SELF, colored_mesh_file, 'r') as xdmf:
            colored_mesh = xdmf.read_mesh(ghost_mode=dfx_mesh.GhostMode.none)
            colored_subdomains = xdmf.read_meshtags(colored_mesh, name="cell_tags")
            colored_mesh.topology.create_connectivity(2, 0)
            colored_boundaries = xdmf.read_meshtags(colored_mesh, name="facet_tags")

        colored_cell_tags = colored_subdomains.values
        colored_coords = colored_mesh.geometry.x
        colored_cells = colored_mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
        geometry = colored_coords.copy()
        cell_topology = colored_cells.astype(np.int64)

        # Compute cell centroids for the colored mesh (vectorized)
        colored_centroids = colored_coords[colored_cells].mean(axis=1)

        # Match colored mesh cells to original mesh cells by centroid
        _, orig_indices = orig_tree.query(colored_centroids)

        # Map each colored cell to partition based on its matched original tag (vectorized)
        matched_orig_tags = orig_cell_tags[orig_indices]
        cell_partitions = np.array([
            tag_to_partition.get(int(t), i % num_partitions)
            for i, t in enumerate(matched_orig_tags)
        ], dtype=np.int32)

        # Build cell-pair -> tag map for membrane facets
        # This is more robust than centroid matching because it uses exact cell indices
        # Each interior membrane facet connects exactly 2 cells
        colored_mesh.topology.create_connectivity(2, 3)
        f_to_c_orig = colored_mesh.topology.connectivity(2, 3)

        cellpair_to_tag = {}
        for local_idx, tag in zip(colored_boundaries.indices, colored_boundaries.values):
            cells = f_to_c_orig.links(local_idx)
            if len(cells) == 2:
                # Use sorted cell pair as key (order-independent)
                cellpair = (min(cells[0], cells[1]), max(cells[0], cells[1]))
                cellpair_to_tag[cellpair] = int(tag)

        print(f"Built cell-pair map: {len(cellpair_to_tag)} membrane facets")

        # Build cell-to-facet and facet-to-cell connectivity for ghost computation
        colored_mesh.topology.create_connectivity(3, 2)
        colored_mesh.topology.create_connectivity(2, 3)
        c_to_f = colored_mesh.topology.connectivity(3, 2)
        f_to_c = colored_mesh.topology.connectivity(2, 3)
        num_cells = len(colored_cells)

        # Vectorized ghost computation using adjacency arrays
        # Get raw connectivity arrays for faster access
        c_to_f_array = c_to_f.array
        c_to_f_offsets = c_to_f.offsets
        f_to_c_array = f_to_c.array
        f_to_c_offsets = f_to_c.offsets

        # For each cell, collect ghost ranks (use list of sets, but minimize Python overhead)
        cell_ghost_ranks = [set() for _ in range(num_cells)]

        for cell in range(num_cells):
            cell_owner = cell_partitions[cell]
            # Get facets for this cell
            f_start, f_end = c_to_f_offsets[cell], c_to_f_offsets[cell + 1]
            for fi in range(f_start, f_end):
                facet = c_to_f_array[fi]
                # Get cells sharing this facet
                c_start, c_end = f_to_c_offsets[facet], f_to_c_offsets[facet + 1]
                for ci in range(c_start, c_end):
                    neighbor = f_to_c_array[ci]
                    if neighbor != cell:
                        neighbor_owner = cell_partitions[neighbor]
                        if neighbor_owner != cell_owner:
                            cell_ghost_ranks[cell].add(neighbor_owner)

        # Build adjacency list: owner rank first, then ghost ranks
        # Pre-calculate total size for efficiency
        total_entries = num_cells + sum(len(g) for g in cell_ghost_ranks)
        adj_data = np.empty(total_entries, dtype=np.int32)
        adj_offsets = np.empty(num_cells + 1, dtype=np.int32)

        offset = 0
        for cell in range(num_cells):
            adj_offsets[cell] = offset
            adj_data[offset] = cell_partitions[cell]
            offset += 1
            for ghost in sorted(cell_ghost_ranks[cell]):
                adj_data[offset] = ghost
                offset += 1
        adj_offsets[num_cells] = offset

        num_ghosted = sum(1 for g in cell_ghost_ranks if len(g) > 0)
        print(f"Loaded {len(colored_cell_tags)} colored cells, {num_ghosted} need ghosting")
    else:
        geometry = np.empty((0, 3), dtype=np.float64)
        cell_topology = np.empty((0, 4), dtype=np.int64)
        colored_cell_tags = np.empty(0, dtype=np.int32)
        cellpair_to_tag = {}
        adj_data = np.array([], dtype=np.int32)
        adj_offsets = np.array([0], dtype=np.int32)

    # Broadcast data to all ranks
    colored_cell_tags = comm.bcast(colored_cell_tags, root=0)
    cellpair_to_tag = comm.bcast(cellpair_to_tag, root=0)

    # Create custom partitioner with ghost destinations
    # Note: partitioner is called with LOCAL cells - only rank 0 has cells initially
    def custom_partitioner(mpi_comm, nparts, graph, ghosted):
        num_local_cells = graph.num_nodes
        if num_local_cells == 0:
            # No local cells (rank != 0 initially)
            return AdjacencyList_int32(np.array([], dtype=np.int32), np.array([0], dtype=np.int32))
        else:
            # Return the precomputed adjacency list for local cells
            return AdjacencyList_int32(adj_data, adj_offsets)

    partitioner = dfx_mesh.create_cell_partitioner(custom_partitioner, ghost_mode)

    # Create mesh with custom partitioning
    coord_elem = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    msh = dfx_mesh.create_mesh(comm, cell_topology, geometry, coord_elem, partitioner)

    # Create cell tags on the partitioned mesh (using colored tags 0-3)
    orig_cell_idx = msh.topology.original_cell_index
    local_cell_tags = colored_cell_tags[orig_cell_idx]
    subdomains = dfx_mesh.meshtags(
        msh, msh.topology.dim,
        np.arange(len(local_cell_tags), dtype=np.int32),
        local_cell_tags
    )
    subdomains.name = "cell_tags"

    # Reconstruct facet tags using cell-pair matching
    # Each membrane facet connects exactly 2 cells; we use original_cell_index
    # (reliably maintained by DOLFINx for all cells including ghosts) to look up tags
    msh.topology.create_connectivity(2, 3)

    facet_imap = msh.topology.index_map(2)
    num_all_facets = facet_imap.size_local + facet_imap.num_ghosts
    f_to_c_new = msh.topology.connectivity(2, 3)
    orig_cell_idx = msh.topology.original_cell_index

    local_facet_indices = []
    local_facet_tags = []

    if cellpair_to_tag and num_all_facets > 0:
        for f in range(num_all_facets):
            cells = f_to_c_new.links(f)
            if len(cells) == 2:
                orig_c0 = int(orig_cell_idx[cells[0]])
                orig_c1 = int(orig_cell_idx[cells[1]])
                cellpair = (min(orig_c0, orig_c1), max(orig_c0, orig_c1))
                tag = cellpair_to_tag.get(cellpair)
                if tag is not None:
                    local_facet_indices.append(f)
                    local_facet_tags.append(tag)

    boundaries = dfx_mesh.meshtags(
        msh, 2,
        np.array(local_facet_indices, dtype=np.int32),
        np.array(local_facet_tags, dtype=np.int32)
    )
    boundaries.name = "facet_tags"

    total_matched = comm.reduce(len(local_facet_indices), op=MPI.SUM, root=0)
    if rank == 0:
        print(f"Created mesh with {msh.topology.index_map(3).size_global} cells")
        print(f"Facet tag reconstruction: {total_matched} facets matched "
              f"(expected ~2x {len(cellpair_to_tag)} with ghosting)")

    return msh, subdomains, boundaries
