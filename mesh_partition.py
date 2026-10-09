"""
Custom mesh partitioning utilities for component-based MPI distribution.

This module provides functions to load meshes with custom partitioning that
keeps ECS+cell component pairs together on the same MPI rank (the default),
or - selectable via `granularity='tag'` - one rank per individual volume tag.

Tag convention:
- Even tags (0, 2, 4, ...) are ECS chunks
- Odd tags (1, 3, 5, ...) are cells inside those ECS chunks
- 'component' granularity: each component = (2*i, 2*i+1) = one ECS chunk +
  one cell, kept on the same rank; METIS balances units across ranks
  (weighted by size, so a rank can end up with several units or none)
- 'tag' granularity: every tag (ECS chunk or cell) is its own unit, assigned
  directly to its own rank (1:1, no METIS) - `num_partitions` must equal the
  number of tags exactly
"""

import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI


def compute_component_partitions(tag_interfaces, num_partitions, granularity='component'):
    """Compute a partitioning of the tag graph, METIS-balanced or a strict bijection.

    Args:
        tag_interfaces: dict mapping tag -> set of interface IDs
        num_partitions: number of partitions (MPI ranks)
        granularity: 'component' (default) merges each (ECS chunk, cell) tag
            pair - (2*i, 2*i+1) - into one partitioning unit, then lets METIS
            balance units across `num_partitions` ranks (may put several
            units on one rank and none on another, weighted by unit size).
            'tag' instead requires `num_partitions == number of tags` and
            assigns each tag its own rank directly (no METIS, no merging) -
            matching a domain-decomposition convergence theory (e.g. BDDC)
            written in terms of one subdomain per volume tag. METIS is
            deliberately skipped for 'tag': balancing by weighted size would
            happily group several small-weight tags onto one rank while
            leaving another rank empty, which is exactly what this mode is
            for avoiding.

    Returns:
        tag_to_partition: dict mapping original tag -> partition ID
        stats: dict with partitioning statistics
    """
    try:
        import pymetis
        has_pymetis = True
    except ImportError:
        has_pymetis = False

    all_tags = sorted(tag_interfaces.keys())
    num_tags = len(all_tags)

    if granularity == 'tag':
        units = [[t] for t in all_tags]
    elif granularity == 'component':
        if num_tags % 2 != 0:
            raise ValueError(
                f"Component partitioning needs an even number of tags "
                f"((ECS chunk, cell) pairs), got {num_tags}."
            )
        units = [[all_tags[2 * c], all_tags[2 * c + 1]] for c in range(num_tags // 2)]
    else:
        raise ValueError(f"Unknown granularity {granularity!r}, expected 'component' or 'tag'.")

    num_units = len(units)

    # Build unit connectivity graph
    unit_interfaces = {}
    for u, tags in enumerate(units):
        ifaces = set()
        for t in tags:
            ifaces |= set(int(x) for x in tag_interfaces.get(t, set()) if int(x) != t)
        unit_interfaces[u] = ifaces

    # Find shared interfaces between units
    interface_to_units = defaultdict(set)
    for u, ifaces in unit_interfaces.items():
        for iface in ifaces:
            interface_to_units[iface].add(u)

    # Build adjacency list for METIS
    adjacency = [[] for _ in range(num_units)]
    for iface, us in interface_to_units.items():
        if len(us) == 2:
            u1, u2 = list(us)
            if u2 not in adjacency[u1]:
                adjacency[u1].append(u2)
            if u1 not in adjacency[u2]:
                adjacency[u2].append(u1)

    num_edges = sum(len(adj) for adj in adjacency) // 2

    if granularity == 'tag':
        # One tag per rank, no exceptions: METIS balances by weighted size,
        # which happily puts several small-weight tags on one rank and leaves
        # another empty - exactly the opposite of what this mode is for (a
        # domain-decomposition theory stated per volume tag). So skip it
        # entirely and require a strict bijection instead.
        if num_partitions != num_units:
            raise ValueError(
                f"component_granularity 'tag' needs exactly one MPI rank per "
                f"tag: got {num_partitions} ranks for {num_units} tags. Run "
                f"with {num_units} ranks, or use component_granularity: "
                f"component to let METIS balance multiple tags per rank."
            )
        membership = list(range(num_units))
        n_cuts = num_edges  # every edge joins two different tags/ranks by construction
        method = 'identity'
    # Partition using METIS or fallback to round-robin
    elif has_pymetis and num_partitions > 1:
        n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency)
        method = 'metis'
    else:
        membership = [i % num_partitions for i in range(num_units)]
        n_cuts = 0
        method = 'round_robin'

    # Create tag -> partition mapping
    tag_to_partition = {}
    for u, tags in enumerate(units):
        for t in tags:
            tag_to_partition[t] = membership[u]

    # Compute balance statistics
    partition_sizes = defaultdict(int)
    for part in membership:
        partition_sizes[part] += 1

    stats = {
        'granularity': granularity,
        'num_components': num_units,
        'num_partitions': num_partitions,
        'num_edges': num_edges,
        'edge_cuts': n_cuts,
        'method': method,
        'components_per_partition': dict(partition_sizes),
    }

    return tag_to_partition, stats


# Weak-scaling mesh names encode the geometry:
#   <plus|cell>_<nx>x<ny>x<nz>_n<n>_L<L>[_p<pad>][_a<ax>]
# (same convention as geometry/generate_weak_scaling_mesh.py and viz/server.py).
# 'cell' meshes stretch each box to ax*L in x; 'plus' meshes are always cubes.
WS_NAME_RE = re.compile(
    r'^(plus|cell)_(\d+)x(\d+)x(\d+)_n(\d+)_L([0-9p]+?)(?:_p(\d+))?'
    r'(?:_a(\d+))?(?:_s(\d+)t(\d+)(?:u(\d+))?)?'
    r'(?:_d(\d+)(?:z(\d+))?)?(?:_r(\d+))?(?:_g(\d+))?$')


def parse_weak_scaling_geometry(mesh_file):
    """Recover the box-lattice geometry from a weak-scaling mesh filename.

    Returns dict(shape, nx, ny, nz, n, L, pad, ax) or None if the name does not
    match. `pad` counts whole boxes for the 'cell' shape and voxels for 'plus';
    `cube_subdomain_ids` reads `shape` to know which.
    """
    m = WS_NAME_RE.match(Path(mesh_file).stem)
    if not m:
        return None
    return {
        'shape': m.group(1),
        'nx': int(m.group(2)), 'ny': int(m.group(3)), 'nz': int(m.group(4)),
        'n': int(m.group(5)), 'L': float(m.group(6).replace('p', '.')),
        'pad': int(m.group(7)) if m.group(7) else 0,
        'ax': int(m.group(8)) if m.group(8) else 1,
        'slabs': int(m.group(9)) if m.group(9) else 0,
        'step_y': int(m.group(10)) if m.group(10) else 1,
        'step_z': int(m.group(11)) if m.group(11) else (
            int(m.group(10)) if m.group(10) else 1),
    }


def cube_subdomain_ids(centroids, cell_tags, geom, ecs_tag):
    """Subdomain index of every cell: 2*c for the cell of cube c, 2*c+1 for its ECS.

    Boxes are ax*L long in x (ax = 1 for the original cubic 'plus' meshes) and
    are binned from the tet centroids. The `cell` lattice is shifted and wrapped:
    box (I,J,K) sits at x = I*Lx + ((J*step_y + K*step_z) mod slabs)*Lx/slabs, so y and z are binned
    first and their offset subtracted before binning x. Box walls fall on element
    faces either way, so a centroid never sits on a boundary and the binning is
    exact. Padding cells lie outside every box and are clamped in.
    """
    nx, ny, nz = int(geom['nx']), int(geom['ny']), int(geom['nz'])
    L, n, pad = float(geom['L']), int(geom['n']), int(geom.get('pad', 0))
    ax = int(geom.get('ax', 1) or 1)
    shape = geom.get('shape', 'plus')
    box = np.array([ax * L, L, L])

    # Block origin: the padding wraps the block on every side, `pad` element
    # layers of size L/n thick -- a boundary layer for the unstructured 'cell'
    # meshes, voxels for the 'plus' ones, but the same thickness either way.
    origin = pad * (L / n)

    p = np.asarray(centroids) - origin
    idx = np.empty((len(p), 3), np.int64)
    # Clamp y and z into the block *before* the wrap. Padding lies outside every
    # box, so its raw J is -1 or ny, and feeding that to the wrap subtracts a
    # spurious slab from x -- which silently binned padding cells into the wrong
    # box, a slab-wide band of them along each box boundary. They must be binned
    # as the box they are clamped into.
    idx[:, 1] = np.clip(np.floor(p[:, 1] / L), 0, ny - 1)
    idx[:, 2] = np.clip(np.floor(p[:, 2] / L), 0, nz - 1)
    if shape == 'cell':
        slabs = int(geom.get('slabs') or 2)
        py = int(geom.get('step_y') or 1)
        pz = int(geom.get('step_z') or py)
        wrap = np.mod(idx[:, 1] * py + idx[:, 2] * pz, slabs)
        idx[:, 0] = np.floor((p[:, 0] - wrap * (ax * L / slabs)) / box[0])
    else:
        idx[:, 0] = np.floor(p[:, 0] / box[0])
    idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
    cube = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]

    is_ecs = (np.asarray(cell_tags) == ecs_tag).astype(np.int64)
    return 2 * cube + is_ecs


def compute_cube_partitions(centroids, cell_tags, geom, ecs_tag, num_partitions):
    """One subdomain per plus cell, one per the ECS remainder of the same cube.

    The weak-scaling mesh is an nx x ny x nz lattice of cubes of side L, each
    holding exactly one intracellular "3D plus" cell; `pad` voxels of pure ECS
    wrap the whole block. Every cell is binned into its cube from its centroid
    (cube walls fall on voxel walls, so this is exact), giving subdomains

        2*c     -> the cell of cube c
        2*c + 1 -> the ECS filling the rest of cube c

    Padding cells lie outside every cube and are clamped into the nearest one.

    With num_partitions == 2 * n_cubes each subdomain is its own MPI rank. Fewer
    ranks are allowed when they divide the subdomain count evenly: consecutive
    subdomains are then blocked together, which keeps each cell with the ECS of
    its own cube (2 subdomains/rank = one full cube per rank, and so on).

    Args:
        centroids: (num_cells, 3) cell centroids, in mesh (unscaled) units
        cell_tags: (num_cells,) volume tags
        geom: dict(nx, ny, nz, n, L, pad)
        ecs_tag: volume tag of the extracellular space
        num_partitions: number of MPI ranks

    Returns:
        cell_partitions: (num_cells,) int32 rank per cell
        stats: dict with partitioning statistics
    """
    nx, ny, nz = geom['nx'], geom['ny'], geom['nz']

    n_cubes = nx * ny * nz
    n_sub = 2 * n_cubes

    if n_sub % num_partitions != 0:
        raise ValueError(
            f"Cube partitioning needs the number of MPI ranks to divide "
            f"{n_sub} = 2 x {nx}x{ny}x{nz} subdomains (one per cell, one per "
            f"cube remainder), got {num_partitions} ranks. Use "
            f"{n_sub} ranks for one subdomain per rank."
        )

    subdomain = cube_subdomain_ids(centroids, cell_tags, geom, ecs_tag)

    subs_per_rank = n_sub // num_partitions
    cell_partitions = (subdomain // subs_per_rank).astype(np.int32)

    cells_per_rank = np.bincount(cell_partitions, minlength=num_partitions)
    stats = {
        'num_cubes': n_cubes,
        'num_subdomains': n_sub,
        'subdomains_per_rank': subs_per_rank,
        'min_cells_per_rank': int(cells_per_rank.min()),
        'max_cells_per_rank': int(cells_per_rank.max()),
    }
    return cell_partitions, stats


def _build_ghost_adjacency(msh, cell_partitions):
    """Adjacency list for a custom partitioner: owner rank first, then ghosts.

    A cell must be ghosted on every rank owning a facet-neighbour, otherwise the
    interior-facet (membrane) integrals lose contributions. Vectorized over the
    facet-to-cell connectivity: interior facets link exactly two cells, and a
    facet whose two cells have different owners ghosts each onto the other.
    """
    msh.topology.create_connectivity(3, 2)
    msh.topology.create_connectivity(2, 3)
    f_to_c = msh.topology.connectivity(2, 3)

    num_cells = len(cell_partitions)
    offsets = f_to_c.offsets
    links = f_to_c.array

    # Interior facets: exactly two adjacent cells.
    interior = np.nonzero(np.diff(offsets) == 2)[0]
    starts = offsets[interior]
    c0 = links[starts].astype(np.int64)
    c1 = links[starts + 1].astype(np.int64)
    p0 = cell_partitions[c0]
    p1 = cell_partitions[c1]

    cut = p0 != p1
    cells = np.concatenate([c0[cut], c1[cut]])
    ghosts = np.concatenate([p1[cut], p0[cut]])

    # Unique (cell, ghost rank) pairs, sorted by cell.
    order = np.lexsort((ghosts, cells))
    cells = cells[order]
    ghosts = ghosts[order]
    if len(cells):
        keep = np.ones(len(cells), dtype=bool)
        keep[1:] = (cells[1:] != cells[:-1]) | (ghosts[1:] != ghosts[:-1])
        cells = cells[keep]
        ghosts = ghosts[keep]

    counts = np.bincount(cells, minlength=num_cells)

    adj_offsets = np.zeros(num_cells + 1, dtype=np.int32)
    np.cumsum(1 + counts, out=adj_offsets[1:])
    adj_data = np.empty(adj_offsets[-1], dtype=np.int32)
    adj_data[adj_offsets[:-1]] = cell_partitions                  # owner first
    if len(cells):
        first = np.cumsum(counts) - counts                        # per-cell block start
        pos_in_cell = np.arange(len(cells)) - first[cells]
        adj_data[adj_offsets[cells] + 1 + pos_in_cell] = ghosts   # then ghost ranks

    num_ghosted = int(np.count_nonzero(counts))
    return adj_data, adj_offsets, num_ghosted


def _build_cellpair_map(msh, boundaries, num_cells):
    """Membrane facet tags keyed by the (sorted) original cell pair they join.

    Returned as a sorted int64 key array (lo * num_cells + hi) plus the matching
    int32 tags rather than as a dict: the weak-scaling meshes have millions of
    membrane facets, and a dict of tuple keys costs ~160 bytes per entry against
    12 bytes here - gigabytes per rank once the map is replicated.
    """
    msh.topology.create_connectivity(2, 3)
    f_to_c = msh.topology.connectivity(2, 3)
    offsets, links = f_to_c.offsets, f_to_c.array

    # Membrane facets are interior: exactly two adjacent cells.
    idx = np.asarray(boundaries.indices, dtype=np.int64)
    interior = (offsets[idx + 1] - offsets[idx]) == 2
    starts = offsets[idx[interior]]
    c0 = links[starts].astype(np.int64)
    c1 = links[starts + 1].astype(np.int64)

    keys = np.minimum(c0, c1) * np.int64(num_cells) + np.maximum(c0, c1)
    tags = np.asarray(boundaries.values, dtype=np.int32)[interior]
    del idx, interior, starts, c0, c1

    order = np.argsort(keys, kind='stable')
    keys, tags = keys[order], tags[order]
    if len(keys):
        # Duplicate cell pairs: keep the last tag, as the dict assignment did.
        keep = np.ones(len(keys), dtype=bool)
        keep[:-1] = keys[1:] != keys[:-1]
        keys, tags = keys[keep], tags[keep]
    return keys, tags


def _bcast_shared(comm, arr, dtype):
    """Broadcast a rank-0 array as one MPI shared-memory copy per compute node.

    `comm.bcast` hands every rank a private copy; with 128 ranks per node and a
    cell-pair map of millions of facets that alone exhausts node memory. The
    array is broadcast between node roots only and read from a shared window by
    everyone else. Returns (view, handle) - pass the handle to _free_shared once
    done reading, after which the view must not be touched.
    """
    dtype = np.dtype(dtype)

    n = np.array(arr.size if comm.rank == 0 else 0, dtype=np.int64)
    comm.Bcast(n, root=0)
    n = int(n)
    if n == 0:
        return np.empty(0, dtype=dtype), None

    try:
        node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
    except Exception:
        node = None

    # Both branches below are collective over comm, so all ranks must agree.
    if not comm.allreduce(node is not None, op=MPI.LAND):
        # No shared memory available: plain buffer Bcast, which at least avoids
        # the pickle round-trip (and its transient copies) of comm.bcast.
        if node is not None:
            node.Free()
        out = np.ascontiguousarray(arr, dtype=dtype) if comm.rank == 0 \
            else np.empty(n, dtype=dtype)
        comm.Bcast(out, root=0)
        return out, None

    win = MPI.Win.Allocate_shared(
        n * dtype.itemsize if node.rank == 0 else 0, dtype.itemsize, comm=node)
    buf, _ = win.Shared_query(0)
    out = np.ndarray(buffer=buf, dtype=dtype, shape=(n,))

    roots = comm.Split(0 if node.rank == 0 else MPI.UNDEFINED, comm.rank)
    if roots != MPI.COMM_NULL:
        if comm.rank == 0:
            out[:] = arr
        roots.Bcast(out, root=0)
        roots.Free()

    node.Barrier()   # the window is filled only after the node root's Bcast
    return out, (win, node)


def _free_shared(*handles):
    """Release the shared windows handed out by _bcast_shared (collective)."""
    for handle in handles:
        if handle is None:
            continue
        win, node = handle
        node.Barrier()   # every reader is done before the memory goes away
        win.Free()
        node.Free()


def _finalize_partitioned_mesh(comm, cell_topology, geometry, cell_tags,
                               cellpair_keys, cellpair_tags, cellpair_stride,
                               adj_data, adj_offsets, ghost_mode):
    """Distribute a rank-0 mesh with a precomputed partition and rebuild its tags.

    `adj_data`/`adj_offsets` carry the per-cell (owner, ghost ranks...) adjacency
    for rank 0's cells - the custom partitioner hands them straight back to
    DOLFINx; every other rank starts without cells. Cell tags follow
    original_cell_index, and facet tags are recovered through the
    (original cell pair) -> tag map from _build_cellpair_map, which survives
    redistribution while vertex global indices do not.
    """
    from dolfinx import mesh as dfx_mesh
    from dolfinx.cpp.graph import AdjacencyList_int32
    import basix

    rank = comm.rank

    # Share rank 0's data: one copy per node, not one per rank.
    stride = np.int64(comm.bcast(cellpair_stride, root=0))
    cell_tags, tags_win = _bcast_shared(comm, cell_tags, np.int32)
    keys, keys_win = _bcast_shared(comm, cellpair_keys, np.int64)
    pair_tags, pair_win = _bcast_shared(comm, cellpair_tags, np.int32)

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

    # Create cell tags on the partitioned mesh
    orig_cell_idx = msh.topology.original_cell_index
    local_cell_tags = cell_tags[orig_cell_idx]
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
    offsets, links = f_to_c_new.offsets, f_to_c_new.array

    # Vectorized lookup over every facet with two adjacent cells (local + ghost).
    nf = min(num_all_facets, len(offsets) - 1)
    interior = np.nonzero(offsets[1:nf + 1] - offsets[:nf] == 2)[0]
    starts = offsets[interior]
    oc0 = orig_cell_idx[links[starts]].astype(np.int64)
    oc1 = orig_cell_idx[links[starts + 1]].astype(np.int64)
    query = np.minimum(oc0, oc1) * stride + np.maximum(oc0, oc1)
    del starts, oc0, oc1

    if len(keys) and len(query):
        pos = np.searchsorted(keys, query)
        np.clip(pos, 0, len(keys) - 1, out=pos)
        hit = keys[pos] == query
        local_facet_indices = interior[hit].astype(np.int32)
        local_facet_tags = pair_tags[pos[hit]].astype(np.int32)
    else:
        local_facet_indices = np.empty(0, dtype=np.int32)
        local_facet_tags = np.empty(0, dtype=np.int32)

    boundaries = dfx_mesh.meshtags(
        msh, 2, local_facet_indices, local_facet_tags)
    boundaries.name = "facet_tags"

    num_pairs = len(keys)
    total_matched = comm.reduce(len(local_facet_indices), op=MPI.SUM, root=0)

    # Views into the shared windows die with them; everything kept above is a copy.
    _free_shared(tags_win, keys_win, pair_win)

    if rank == 0:
        print(f"Created mesh with {msh.topology.index_map(3).size_global} cells")
        print(f"Facet tag reconstruction: {total_matched} facets matched "
              f"(expected ~2x {num_pairs} with ghosting)")

    return msh, subdomains, boundaries


def load_mesh_with_component_partitioning(
    comm,
    colored_mesh_file,
    original_mesh_file=None,
    ghost_mode=None,
    granularity='component',
):
    """Load colored mesh with component-based custom partitioning.

    This function loads the graph-colored mesh but partitions it based on the
    tag structure of the original mesh, while using the efficient 4-tag
    colored mesh for simulation.

    Args:
        comm: MPI communicator
        colored_mesh_file: Path to the colored mesh XDMF file (4 tags)
        original_mesh_file: Path to the original mesh XDMF file (44 tags).
                           If None, derived from colored_mesh_file.
        ghost_mode: GhostMode for mesh (default: shared_facet)
        granularity: 'component' (default) keeps each ECS+cell tag pair on the
                     same rank; 'tag' gives every individual tag its own rank
                     (see compute_component_partitions).

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

        # Compute METIS partitioning based on components (or individual tags)
        tag_to_partition, stats = compute_component_partitions(
            orig_tag_interfaces, num_partitions, granularity=granularity)
        unit_label = 'tags' if granularity == 'tag' else 'components'
        print(f"Component partitioning ({granularity}): {stats['method']}, "
              f"{stats['num_components']} {unit_label}, "
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

        colored_cell_tags = np.array(colored_subdomains.values, dtype=np.int32)
        colored_coords = colored_mesh.geometry.x
        colored_cells = colored_mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
        geometry = colored_coords.copy()
        cell_topology = colored_cells.astype(np.int64)

        # Compute cell centroids for the colored mesh (vectorized)
        colored_centroids = colored_coords[colored_cells].mean(axis=1)

        # Match colored mesh cells to original mesh cells by centroid
        _, orig_indices = orig_tree.query(colored_centroids)

        # Map each colored cell to partition based on its matched original tag (vectorized)
        matched_orig_tags = np.asarray(orig_cell_tags)[orig_indices]
        cell_partitions = np.array([
            tag_to_partition.get(int(t), i % num_partitions)
            for i, t in enumerate(matched_orig_tags)
        ], dtype=np.int32)

        # The original mesh was only needed for the centroid match - drop it
        # before the (much larger) redistribution buffers are allocated.
        del orig_tree, orig_centroids, orig_indices, orig_coords, orig_cells
        del orig_cell_tags, orig_subdomains, orig_mesh

        # Cell-pair -> tag map for membrane facets: more robust than centroid
        # matching because it uses exact cell indices.
        cellpair_stride = len(colored_cell_tags)
        cellpair_keys, cellpair_tags = _build_cellpair_map(
            colored_mesh, colored_boundaries, cellpair_stride)
        print(f"Built cell-pair map: {len(cellpair_keys)} membrane facets")

        # Ghost destinations from cell-facet connectivity
        adj_data, adj_offsets, num_ghosted = _build_ghost_adjacency(
            colored_mesh, cell_partitions)
        print(f"Loaded {len(colored_cell_tags)} colored cells, {num_ghosted} need ghosting")

        # Everything needed downstream is a copy: free the serial colored mesh.
        del colored_coords, colored_cells, colored_centroids
        del colored_subdomains, colored_boundaries, colored_mesh
    else:
        geometry = np.empty((0, 3), dtype=np.float64)
        cell_topology = np.empty((0, 4), dtype=np.int64)
        colored_cell_tags = np.empty(0, dtype=np.int32)
        cellpair_keys = np.empty(0, dtype=np.int64)
        cellpair_tags = np.empty(0, dtype=np.int32)
        cellpair_stride = 0
        adj_data = np.array([], dtype=np.int32)
        adj_offsets = np.array([0], dtype=np.int32)

    return _finalize_partitioned_mesh(
        comm, cell_topology, geometry, colored_cell_tags,
        cellpair_keys, cellpair_tags, cellpair_stride,
        adj_data, adj_offsets, ghost_mode)


def load_mesh_with_cube_partitioning(
    comm,
    mesh_file,
    geometry_params=None,
    ecs_tag=None,
    ghost_mode=None,
):
    """Load a weak-scaling mesh with one MPI subdomain per cell and per cube ECS.

    Intended for the `plus_*` weak-scaling meshes (see
    geometry/generate_weak_scaling_mesh.py): an nx x ny x nz lattice of cubes,
    each holding exactly one intracellular "3D plus" cell. Cube c yields two
    subdomains - its cell (2*c) and the ECS filling the rest of that cube
    (2*c + 1) - so the run needs 2 * nx*ny*nz ranks for a subdomain each (fewer
    ranks are allowed when they divide that count; see compute_cube_partitions).

    Args:
        comm: MPI communicator
        mesh_file: Path to the weak-scaling mesh XDMF file
        geometry_params: dict(nx, ny, nz, n, L, pad) overriding what is parsed
                         from the mesh filename
        ecs_tag: volume tag of the extracellular space (default: smallest tag)
        ghost_mode: GhostMode for mesh (default: shared_facet)

    Returns:
        mesh: The partitioned DOLFINx mesh
        subdomains: MeshTags for cell subdomains
        boundaries: MeshTags for facet boundaries
    """
    from dolfinx import mesh as dfx_mesh, io

    if ghost_mode is None:
        ghost_mode = dfx_mesh.GhostMode.shared_facet

    geom = dict(geometry_params) if geometry_params else parse_weak_scaling_geometry(mesh_file)
    if geom is None:
        raise ValueError(
            f"Cube partitioning could not read the cube lattice from "
            f"'{mesh_file}'. Expected a weak-scaling mesh named "
            f"<plus|cell>_<nx>x<ny>x<nz>_n<n>_L<L>[_p<pad>][_a<ax>].xdmf, or give "
            f"the geometry explicitly via cube_partition: "
            f"{{nx, ny, nz, n, L, pad, ax}} in the input .yml."
        )
    missing = {'nx', 'ny', 'nz', 'n', 'L'} - set(geom)
    if missing:
        raise ValueError(f"cube_partition is missing key(s): {sorted(missing)}")
    geom.setdefault('pad', 0)
    geom.setdefault('ax', 1)
    geom.setdefault('shape', 'cell' if geom['ax'] != 1 else 'plus')
    geom.setdefault('slabs', 0)
    geom.setdefault('step_y', 1)
    geom.setdefault('step_z', 1)

    rank = comm.rank
    num_partitions = comm.size

    if rank == 0:
        with io.XDMFFile(MPI.COMM_SELF, mesh_file, 'r') as xdmf:
            local_mesh = xdmf.read_mesh(ghost_mode=dfx_mesh.GhostMode.none)
            local_subdomains = xdmf.read_meshtags(local_mesh, name="cell_tags")
            local_mesh.topology.create_connectivity(2, 0)
            local_boundaries = xdmf.read_meshtags(local_mesh, name="facet_tags")

        cell_tags = np.array(local_subdomains.values, dtype=np.int32)
        coords = local_mesh.geometry.x
        cells = local_mesh.topology.connectivity(3, 0).array.reshape(-1, 4)
        geometry = coords.copy()
        cell_topology = cells.astype(np.int64)

        centroids = coords[cells].mean(axis=1)

        tag = int(cell_tags.min()) if ecs_tag is None else int(ecs_tag)
        cell_partitions, stats = compute_cube_partitions(
            centroids, cell_tags, geom, tag, num_partitions)

        print(f"Cube partitioning: {stats['num_cubes']} cubes -> "
              f"{stats['num_subdomains']} subdomains "
              f"({stats['subdomains_per_rank']} per rank over {num_partitions} ranks)")
        print(f"  Cells per rank: {stats['min_cells_per_rank']} - "
              f"{stats['max_cells_per_rank']}")

        # Cell-pair -> tag map for membrane facets (survives redistribution)
        cellpair_stride = len(cell_tags)
        cellpair_keys, cellpair_tags = _build_cellpair_map(
            local_mesh, local_boundaries, cellpair_stride)
        print(f"Built cell-pair map: {len(cellpair_keys)} membrane facets")

        adj_data, adj_offsets, num_ghosted = _build_ghost_adjacency(
            local_mesh, cell_partitions)
        print(f"Loaded {len(cell_tags)} cells, {num_ghosted} need ghosting")

        # Everything needed downstream is a copy: free the serial mesh before
        # DOLFINx allocates its redistribution buffers on this rank.
        del coords, cells, centroids, local_subdomains, local_boundaries, local_mesh
    else:
        geometry = np.empty((0, 3), dtype=np.float64)
        cell_topology = np.empty((0, 4), dtype=np.int64)
        cell_tags = np.empty(0, dtype=np.int32)
        cellpair_keys = np.empty(0, dtype=np.int64)
        cellpair_tags = np.empty(0, dtype=np.int32)
        cellpair_stride = 0
        adj_data = np.array([], dtype=np.int32)
        adj_offsets = np.array([0], dtype=np.int32)

    return _finalize_partitioned_mesh(
        comm, cell_topology, geometry, cell_tags,
        cellpair_keys, cellpair_tags, cellpair_stride,
        adj_data, adj_offsets, ghost_mode)
