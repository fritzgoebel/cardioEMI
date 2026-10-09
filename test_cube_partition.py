"""Verify cube partitioning of a weak-scaling mesh (partition_mode: cube).

Each cube of the plus-cell lattice must yield exactly two subdomains: its cell
and the ECS filling the rest of that cube. Run with as many ranks as there are
subdomains (2 * nx*ny*nz), or any divisor of that count:

    mpirun -n 2 python3 test_cube_partition.py data/plus_1x1x1_n8_L25_p2.xdmf
    mpirun -n 4 python3 test_cube_partition.py data/plus_2x1x1_n8_L25_p2.xdmf

Checks, against the same mesh read serially:
  1. every rank owns whole subdomains only (no cell/ECS mixing within a cube),
  2. cell tags survive redistribution (global per-tag counts match),
  3. membrane facet tags survive redistribution (owned tagged facets match).
"""

import sys
import numpy as np
import dolfinx as dfx
from mpi4py import MPI

from mesh_partition import (
    cube_subdomain_ids,
    load_mesh_with_cube_partitioning,
    parse_weak_scaling_geometry,
)

comm = MPI.COMM_WORLD
mesh_file = sys.argv[1] if len(sys.argv) > 1 else "data/plus_4x4x4_n8_L25_p4.xdmf"

geom = parse_weak_scaling_geometry(mesh_file)
if geom is None:
    raise SystemExit(f"Not a weak-scaling mesh name: {mesh_file}")

n_sub = 2 * geom['nx'] * geom['ny'] * geom['nz']
if n_sub % comm.size != 0:
    raise SystemExit(f"Run with a rank count dividing {n_sub} (got {comm.size})")

# ---- reference: serial read of the same mesh -------------------------------
if comm.rank == 0:
    with dfx.io.XDMFFile(MPI.COMM_SELF, mesh_file, 'r') as xdmf:
        ref_mesh = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.none)
        ref_tags = xdmf.read_meshtags(ref_mesh, name="cell_tags")
        ref_mesh.topology.create_connectivity(2, 0)
        ref_facets = xdmf.read_meshtags(ref_mesh, name="facet_tags")

    ref_tag_counts = dict(zip(*np.unique(ref_tags.values, return_counts=True)))
    ref_membrane = int(np.count_nonzero(ref_facets.values != -5))
    ecs_tag = int(ref_tags.values.min())
else:
    ref_tag_counts, ref_membrane, ecs_tag = None, None, None

ref_tag_counts = comm.bcast(ref_tag_counts, root=0)
ref_membrane = comm.bcast(ref_membrane, root=0)
ecs_tag = comm.bcast(ecs_tag, root=0)

# ---- partitioned read ------------------------------------------------------
mesh, subdomains, boundaries = load_mesh_with_cube_partitioning(
    comm, mesh_file=mesh_file, ecs_tag=ecs_tag,
    ghost_mode=dfx.mesh.GhostMode.shared_facet)

num_owned = mesh.topology.index_map(3).size_local
coords = mesh.geometry.x
cells = mesh.geometry.dofmap.reshape(-1, 4)
centroids = coords[cells].mean(axis=1)[:num_owned]
owned_tags = subdomains.values[:num_owned]

subs = cube_subdomain_ids(centroids, owned_tags, geom, ecs_tag)
my_subs = sorted(set(int(s) for s in subs))

# ---- 1. whole subdomains per rank ------------------------------------------
expected_per_rank = n_sub // comm.size
ok_subs = (len(my_subs) == expected_per_rank and
           my_subs == list(range(my_subs[0], my_subs[0] + expected_per_rank)))

all_subs = comm.gather((comm.rank, my_subs, int(num_owned)), root=0)

# ---- 2. cell tags ----------------------------------------------------------
local_counts = {int(t): int(c) for t, c in
                zip(*np.unique(owned_tags, return_counts=True))}
gathered_counts = comm.gather(local_counts, root=0)

# ---- 3. membrane facet tags ------------------------------------------------
facet_imap = mesh.topology.index_map(2)
owned_membrane = int(np.count_nonzero(
    (boundaries.indices < facet_imap.size_local) & (boundaries.values != -5)))
total_membrane = comm.reduce(owned_membrane, op=MPI.SUM, root=0)

ok_subs_all = comm.reduce(ok_subs, op=MPI.LAND, root=0)

if comm.rank == 0:
    print(f"\nMesh: {mesh_file}")
    print(f"  lattice {geom['nx']}x{geom['ny']}x{geom['nz']}, n={geom['n']}, "
          f"L={geom['L']}, pad={geom['pad']} -> {n_sub} subdomains on {comm.size} ranks")

    print("\nOwned subdomains per rank (2c = cell of cube c, 2c+1 = its ECS):")
    for rank, subs_r, n_owned in all_subs:
        label = ", ".join(f"{'cell' if s % 2 == 0 else 'ECS '} of cube {s // 2}"
                          for s in subs_r)
        print(f"  rank {rank}: {n_owned:8d} cells  [{label}]")

    total_counts = {}
    for c in gathered_counts:
        for t, v in c.items():
            total_counts[t] = total_counts.get(t, 0) + v
    ok_tags = total_counts == {int(k): int(v) for k, v in ref_tag_counts.items()}
    ok_membrane = total_membrane == ref_membrane

    print(f"\n1. whole subdomains per rank: {'PASS' if ok_subs_all else 'FAIL'}")
    print(f"2. cell tags preserved:       {'PASS' if ok_tags else 'FAIL'} "
          f"({total_counts} vs {dict(ref_tag_counts)})")
    print(f"3. membrane facets preserved: {'PASS' if ok_membrane else 'FAIL'} "
          f"({total_membrane} vs {ref_membrane})")

    if not (ok_subs_all and ok_tags and ok_membrane):
        sys.exit(1)
    print("\nAll checks passed.")
