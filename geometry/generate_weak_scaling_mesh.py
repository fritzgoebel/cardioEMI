"""
Generate a repeatable "3D-plus cell" mesh for EMI weak-scaling studies.

Geometry
--------
The domain is an Nx x Ny x Nz tiling of identical cubes of side L. Each cube
contains exactly ONE intracellular cell shaped like a three-dimensional plus
(the union of three orthogonal square bars of cross-section w x w through the
cube centre). Each bar reaches the centre of a pair of opposite cube faces with
a w x w patch, so a cell connects to its six axis-neighbours arm-to-arm
(intracellular-intracellular / gap-junction membranes) while the extracellular
space (ECS) fills the eight corners and twelve edges of every cube.

Exact equal volumes
-------------------
Volume of the plus is  3 w^2 L - 2 w^3.  Requiring it to equal half the cube,
3 w^2 L - 2 w^3 = L^3 / 2, gives 4 r^3 - 6 r^2 + 1 = 0 with r = w/L, whose clean
root is r = 1/2. So with bars exactly half the cube width (w = L/2) the ECS and
ICS volumes are EXACTLY equal - no approximation.

Meshing
-------
Each cube is a structured n x n x n voxel grid (n a multiple of 4 so the plus
interfaces at L/4 and 3L/4 fall on grid lines and are represented exactly). Each
voxel is split into 6 tetrahedra via the Kuhn/Freudenthal triangulation (shared
000-111 diagonal), which tiles space conformingly. Everything is vectorized in
numpy, so arbitrary tilings are generated cheaply.

Tagging (colored, checkerboard)
-------------------------------
Cells are 2-colored on the 3D checkerboard (axis-neighbours always differ in
parity, so this is a valid coloring): ECS -> tag 0, intracellular -> tag 1 or 2.
Membrane facet tags are encoded as  min_tag * (N_TAGS + 1) + max_tag  and a
pickle dict maps each volume tag -> set of membrane tags it touches, exactly as
the rest of the cardioEMI pipeline expects.

Usage
-----
    python generate_weak_scaling_mesh.py --nx 4 --ny 4 --nz 4 --n 16 --L 25.0 \
        --prefix ../data/plus_4x4x4_n16

    # then in the input .yml:
    #   mesh_file:            "data/plus_4x4x4_n16.xdmf"
    #   tags_dictionary_file: "data/plus_4x4x4_n16.pickle"
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

# Reuse the exact XDMF/HDF5 writer used by the rest of the pipeline.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_pts_elem import write_xdmf_h5


def build_vertices(Gx, Gy, Gz, h):
    """Structured vertex grid of size (Gx+1, Gy+1, Gz+1) with spacing h.
    vid(i,j,k) = (i*(Gy+1)+j)*(Gz+1)+k (C-order)."""
    xs = np.arange(Gx + 1) * h
    ys = np.arange(Gy + 1) * h
    zs = np.arange(Gz + 1) * h
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return points


def build_tets_and_cell_tags(gx, gy, gz, n, pad):
    """Kuhn-split every voxel into 6 tets; tag each by ECS(0)/checkerboard ICS(1,2).

    The cell block is (gx, gy, gz) voxels; `pad` extra ECS voxels wrap it on every
    side, so the full grid is (gx+2*pad, gy+2*pad, gz+2*pad). With pad > 0 every
    cell membrane faces ECS instead of the raw domain boundary (ICS/ECS volumes are
    then no longer equal, since the padding is pure ECS)."""
    Gx, Gy, Gz = gx + 2 * pad, gy + 2 * pad, gz + 2 * pad

    # Per-voxel integer coordinates (C-order to match build_vertices).
    ii, jj, kk = np.meshgrid(np.arange(Gx), np.arange(Gy), np.arange(Gz), indexing="ij")
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel()

    def vid(a, b, c):
        return (a * (Gy + 1) + b) * (Gz + 1) + c

    # 8 voxel corners.
    c000 = vid(ii,     jj,     kk)
    c001 = vid(ii,     jj,     kk + 1)
    c010 = vid(ii,     jj + 1, kk)
    c011 = vid(ii,     jj + 1, kk + 1)
    c100 = vid(ii + 1, jj,     kk)
    c101 = vid(ii + 1, jj,     kk + 1)
    c110 = vid(ii + 1, jj + 1, kk)
    c111 = vid(ii + 1, jj + 1, kk + 1)

    # Freudenthal/Kuhn triangulation: 6 tets, each a shortest edge-path 000->111.
    tets = [
        np.stack([c000, c100, c110, c111], axis=1),
        np.stack([c000, c100, c101, c111], axis=1),
        np.stack([c000, c010, c110, c111], axis=1),
        np.stack([c000, c010, c011, c111], axis=1),
        np.stack([c000, c001, c101, c111], axis=1),
        np.stack([c000, c001, c011, c111], axis=1),
    ]
    topology = np.concatenate(tets, axis=0).astype(np.int64)

    # ---- classify each voxel as ICS (in the plus) or ECS ----
    # Shift to block-local coordinates; voxels outside the block are padding (ECS).
    bi, bj, bk = ii - pad, jj - pad, kk - pad
    in_block = ((bi >= 0) & (bi < gx) &
                (bj >= 0) & (bj < gy) &
                (bk >= 0) & (bk < gz))

    I, J, K = bi // n, bj // n, bk // n           # which cube (valid in block)
    li, lj, lk = bi % n, bj % n, bk % n           # local voxel index in the cube
    lo, hi = n // 4, 3 * n // 4                    # central band [L/4, 3L/4)
    cx = (li >= lo) & (li < hi)
    cy = (lj >= lo) & (lj < hi)
    cz = (lk >= lo) & (lk < hi)
    # plus = union of 3 bars = "at least two of the three coords are central"
    ics = in_block & ((cx & cy) | (cx & cz) | (cy & cz))

    color = (I + J + K) % 2                        # 3D checkerboard 2-coloring
    voxtag = np.where(ics, 1 + color, 0).astype(np.int32)

    n_ics = int(ics.sum())
    n_ecs = int(ics.size - n_ics)
    if pad == 0:
        assert n_ics == n_ecs, f"ICS/ECS voxel counts differ: {n_ics} vs {n_ecs}"

    cell_tags = np.tile(voxtag, 6)                 # 6 tets per voxel, block order
    return topology, cell_tags, (n_ics, n_ecs)


def build_facets(topology, cell_tags, points):
    """Extract every unique face, tag membrane faces, build the membrane dict.

    Facets are written with a globally consistent winding: each face is oriented
    so its normal points away from the lower-tag adjacent cell. Consistent winding
    is essential for the viewer, whose vertex-normal averaging otherwise cancels
    to zero on this structured, coplanar mesh and renders everything unlit/black.
    """
    num_tags = len(np.unique(cell_tags))
    DEFAULT = -5

    # 4 faces per tet; opp_local[k] is the tet vertex opposite face k.
    face_local = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    opp_local = np.array([0, 1, 2, 3])
    faces = topology[:, face_local].reshape(-1, 3)          # (4M, 3) tet-local order
    opp = topology[:, opp_local].reshape(-1)                # (4M,) opposite vertex
    face_cell_tag = np.repeat(cell_tags, 4)                 # (4M,)

    fkey = np.sort(faces, axis=1)                           # canonical grouping key
    # Sort by face key (primary) then cell tag ascending, so each group's first
    # row is the contribution from the lower-tag cell -> deterministic orientation.
    order = np.lexsort((face_cell_tag, fkey[:, 2], fkey[:, 1], fkey[:, 0]))
    fkey = fkey[order]
    faces = faces[order]
    opp = opp[order]
    face_cell_tag = face_cell_tag[order]

    # Group identical consecutive faces.
    same_as_prev = np.zeros(len(fkey), dtype=bool)
    same_as_prev[1:] = np.all(fkey[1:] == fkey[:-1], axis=1)
    group_start = np.nonzero(~same_as_prev)[0]
    counts = np.diff(np.append(group_start, len(fkey)))

    # Representative triangle = first (lower-tag) row of each group, oriented so
    # the normal points away from that tet's opposite vertex (outward from cell).
    rep = faces[group_start].astype(np.int64)
    vp = points[opp[group_start]]
    v0, v1, v2 = points[rep[:, 0]], points[rep[:, 1]], points[rep[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum('ij,ij->i', normal, v0 - vp) < 0.0
    rep[flip] = rep[flip][:, [0, 2, 1]]
    unique_faces = rep

    uni_tag = np.full(len(group_start), DEFAULT, dtype=np.int32)

    # Interior faces are shared by exactly two tets (count == 2). Within a group
    # rows are tag-sorted, so the two rows carry the (min, max) cell tags.
    pair = counts == 2
    gp = group_start[pair]
    tmin = face_cell_tag[gp]
    tmax = face_cell_tag[gp + 1]
    membrane = tmin != tmax
    mn = tmin[membrane]
    mx = tmax[membrane]
    enc = (mn * (num_tags + 1) + mx).astype(np.int32)
    uni_tag[np.nonzero(pair)[0][membrane]] = enc

    # membrane dict: tag -> set of membrane tags touching it
    membrane_dict = {int(t): set() for t in np.unique(cell_tags)}
    for a, b, e in zip(mn.tolist(), mx.tolist(), enc.tolist()):
        membrane_dict[a].add(int(e))
        membrane_dict[b].add(int(e))

    return unique_faces, uni_tag, membrane_dict


def generate(nx, ny, nz, n, L, prefix, pad=0):
    assert n % 4 == 0, "n (voxels per cube edge) must be a multiple of 4"
    assert pad >= 0, "pad (ECS padding in voxels) must be >= 0"
    t0 = time.perf_counter()
    gx, gy, gz = nx * n, ny * n, nz * n
    Gx, Gy, Gz = gx + 2 * pad, gy + 2 * pad, gz + 2 * pad
    h = L / n

    print(f"Domain: {nx} x {ny} x {nz} cubes, {n}^3 voxels/cube, L = {L}")
    print(f"ECS padding: {pad} voxels ({pad * h:g} per side)")
    print(f"Global voxel grid: {Gx} x {Gy} x {Gz} (cell block {gx} x {gy} x {gz})")

    points = build_vertices(Gx, Gy, Gz, h)
    topology, cell_tags, (n_ics, n_ecs) = build_tets_and_cell_tags(gx, gy, gz, n, pad)
    print(f"Vertices: {len(points)}   Tets: {len(topology)}")
    eq = " (exactly equal volumes)" if pad == 0 else f" (ECS/ICS ratio {n_ecs / n_ics:.3f})"
    print(f"ICS voxels: {n_ics}   ECS voxels: {n_ecs}{eq}")

    facet_topology, facet_tags, membrane_dict = build_facets(topology, cell_tags, points)
    n_membrane = int(np.sum(facet_tags != -5))
    print(f"Unique faces: {len(facet_topology)}   Membrane facets: {n_membrane}")
    print("Membrane dict (tag -> membrane tags):")
    for k in sorted(membrane_dict):
        label = "ECS" if k == 0 else f"ICS-{k}"
        print(f"    {k} ({label}): {sorted(membrane_dict[k])}")

    xdmf_file = prefix + ".xdmf"
    h5_file = prefix + ".h5"
    pickle_file = prefix + ".pickle"

    write_xdmf_h5(points, topology, cell_tags, facet_topology, facet_tags,
                  xdmf_file, h5_file)
    with open(pickle_file, "wb") as f:
        pickle.dump(membrane_dict, f)

    print(f"\nDone in {time.perf_counter() - t0:.2f}s")
    print(f"  Mesh:         {xdmf_file}")
    print(f"  Data:         {h5_file}")
    print(f"  Connectivity: {pickle_file}")
    print("\nInput .yml:")
    print(f'  mesh_file:            "{xdmf_file}"')
    print(f'  tags_dictionary_file: "{pickle_file}"')


def main():
    p = argparse.ArgumentParser(description="Generate a 3D-plus-cell weak-scaling mesh")
    p.add_argument("--nx", type=int, default=4, help="cubes in x")
    p.add_argument("--ny", type=int, default=4, help="cubes in y")
    p.add_argument("--nz", type=int, default=4, help="cubes in z")
    p.add_argument("--n", type=int, default=16,
                   help="voxels per cube edge (multiple of 4)")
    p.add_argument("--L", type=float, default=25.0, help="cube edge length")
    p.add_argument("--pad", type=int, default=0,
                   help="ECS padding layer thickness in voxels wrapping the cell block")
    p.add_argument("--prefix", type=str, default=None,
                   help="output path prefix (default ../data/plus_<nx>x<ny>x<nz>_n<n>)")
    args = p.parse_args()

    prefix = args.prefix or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        f"plus_{args.nx}x{args.ny}x{args.nz}_n{args.n}")

    generate(args.nx, args.ny, args.nz, args.n, args.L, prefix, pad=args.pad)


if __name__ == "__main__":
    main()
