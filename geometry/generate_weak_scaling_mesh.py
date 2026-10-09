"""
Generate a repeatable cell-lattice mesh for EMI weak-scaling studies.

Domain
------
An Nx x Ny x Nz tiling of identical boxes of size (ax*L) x L x L, one cell per
box, so the whole domain is a perfect weak-scaling unit: add boxes, add ranks.

Cell shapes (--shape)
---------------------
`cell` (default) -- an unstructured tetrahedral mesh built with gmsh + OpenCASCADE
(see cell_box_gmsh.py):

  * a **body** cylinder along the x axis spanning the whole box, so both x-face
    cross-sections are full-radius discs and cells butt together end to end the
    way myocytes meet at an intercalated disc;
  * four **connectors**, each an exact circular disc on a lateral face extruded
    along an *oblique* vector, so the interface stays round while the connector
    leans in x. They enter at x = lat_x on the y = 0 / z = 0 faces and leave at
    x = ax*L - lat_x on the y = L / z = L faces.

`plus` -- the original geometry, on a structured voxel grid: three orthogonal
square bars of cross-section L/2 x L/2 through the box centre, each voxel
Kuhn/Freudenthal-split into 6 tets. Requires --ax 1, and gives exactly 50/50
ICS/ECS volumes (3w^2 L - 2w^3 = L^3/2 has the clean root w = L/2).

Why it tiles
------------
Boxes are 2-colored on the 3D checkerboard, (I+J+K) % 2, which keeps touching
cells on different tags. For `cell` the same parity also flips orientation: odd
boxes hold the box mesh mirrored in x. That is what closes the lat_x offset -- a
box's +y connector, at ax*L - lat_x, arrives exactly where its +y neighbour's
mirrored -y connector does. The x faces mate under a pure translation instead,
which is why both end discs must be the same disc.

Mating faces must carry *identical triangulations*, so the box is meshed with
gmsh periodic constraints, which accept the reflective transform the mirroring
needs:

    +x <- -x  translate (Lx,0,0)      +y <- -y  (x,y,z) -> (Lx-x, y+L, z)
                                      +z <- -z  (x,y,z) -> (Lx-x, y, z+L)

Tiling then only has to weld coincident nodes, and `build_lattice` verifies the
result by checking that the welded boundary area equals the block's surface area
-- any face left unwelded would show up as extra boundary.

`plus` is x-symmetric and its structured grid is periodic under translation, so
it is tiled without mirroring (mirroring would flip the Kuhn diagonals and break
conformity).

Meshing resolution
------------------
`--n` is elements per L for both shapes: the `cell` target element size is L/n,
`plus` uses n voxels per cube edge (a multiple of 4).

Padding
-------
`--pad` wraps the block in pure-ECS material so no cell membrane sits on the raw
domain boundary. For `cell` it counts whole extra boxes (meshed identically, just
tagged ECS, so the faces still match); for `plus` it counts voxels, as before.

Tagging
-------
ECS -> tag 0, intracellular -> tag 1 or 2. Membrane facet tags are encoded as
min_tag * (N_TAGS + 1) + max_tag and a pickle dict maps each volume tag -> set of
membrane tags it touches, as the rest of the cardioEMI pipeline expects.

Usage
-----
    python generate_weak_scaling_mesh.py --nx 4 --ny 4 --nz 4 --n 8 --L 25.0 \
        --ax 4

    # then in the input .yml (see input_cell_weak_scaling.yml):
    #   mesh_file:            "data/cell_4x4x4_n8_L25_a4.xdmf"
    #   tags_dictionary_file: "data/cell_4x4x4_n8_L25_a4.pickle"

Tune the shape with --dry-run, which prints ASCII slices of the analytic cell
solid and the resulting volumes without meshing or writing anything.
"""

import os
import sys
import time
import pickle
import argparse
from dataclasses import dataclass, replace

import numpy as np

# Reuse the exact XDMF/HDF5 writer used by the rest of the pipeline.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_pts_elem import write_xdmf_h5
from cell_box_gmsh import mesh_box


# --------------------------------------------------------------------------
# Cell shape
# --------------------------------------------------------------------------

@dataclass
class CellShape:
    """Proportions of one cell inside its (ax*L) x L x L box, in units of L."""
    ax:     int   = 4        # x aspect: box is (ax*L) x L x L
    body_r: float = 0.38     # cell radius; both x-face discs are this too
    lat_r:  float = 0.26     # radius of the lateral connectors

    # Each lateral direction has TWO connectors, not one tube through the cell:
    # one stub near the small-x end and one near the large-x end, both `d` from
    # their extreme. That distance alone fixes the lattice shift -- cell A's
    # +y interface at Lx-d must land on its neighbour's -y interface at d, so
    # the neighbour sits s = Lx - 2d further along x. The stubs' *angle* is then
    # free, because a stub only has to reach the body.
    d_y:    float = 0.5      # y interfaces, from the near/far x face
    d_z:    float = 0.5      # z interfaces (independent of d_y)
    lean:   float = 55.0     # stub angle off the cell axis, degrees (90 = straight
                             # out of the face, smaller leans harder towards +x)

    # The shift has to be rational for the shifted periodicity to be exact, so
    # both directions are expressed over a common denominator.
    slabs:  int   = 0        # q: x slabs per box (0 = derive from d_y, d_z)
    step_y: int   = 0        # p_y: slabs the lattice shifts per y step
    step_z: int   = 0        # p_z: per z step

    def __post_init__(self):
        if not self.slabs or not self.step_y or not self.step_z:
            q, py, pz = self.resolve(self.ax, self.d_y, self.d_z)
            self.slabs, self.step_y, self.step_z = q, py, pz
        # Snap the distances onto the realised shifts, so what is reported and
        # drawn is what is built.
        self.d_y = 0.5 * (self.ax - self.shift_y())
        self.d_z = 0.5 * (self.ax - self.shift_z())

    @staticmethod
    def resolve(ax, d_y, d_z, max_slabs=24):
        """Common denominator q and numerators for the two lateral shifts.

        s/Lx = 1 - 2d/Lx for each direction; both must be k/q over the *same* q,
        since a box's offset is (J*p_y + K*p_z)/q of the box length.
        """
        want = [(ax - 2 * d) / ax for d in (d_y, d_z)]
        best, best_err = (2, 1, 1), float('inf')
        for q in range(2, max_slabs + 1):
            ps = [min(q - 1, max(1, int(round(w * q)))) for w in want]
            err = max(abs(w - p / q) for w, p in zip(want, ps))
            if err < best_err - 1e-12:
                best, best_err = (q, ps[0], ps[1]), err
        return best

    def radii(self):
        return dict(body=self.body_r, lat=self.lat_r)

    def shift_y(self):
        """Lattice shift in x per step in y, in units of L."""
        return self.ax * self.step_y / self.slabs

    def shift_z(self):
        return self.ax * self.step_z / self.slabs

    def lean_x(self):
        """How far a stub travels in x on its way from its face to the body."""
        t = np.tan(np.radians(min(89.9, max(5.0, self.lean))))
        return 0.5 / t

    def lean_angle(self):
        """Stub angle off the cell axis, in degrees."""
        return float(np.degrees(np.arctan2(0.5, self.lean_x())))


def cell_contains(p, shape, L):
    """Analytic inside-test for the cell solid, for previews and volume checks.

    `p` is (..., 3) in box-local coordinates. Each of the four lateral stubs is
    an oblique tube: at every station along its face normal its cross-section is
    the same circle, translated in x -- which is what extruding a disc produces.
    A stub runs from its face to the body axis, half the box across.
    """
    Lx, c = shape.ax * L, 0.5 * L
    R, rl = shape.body_r * L, shape.lat_r * L
    lean = shape.lean_x() * L
    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    inside = (y - c) ** 2 + (z - c) ** 2 < R * R
    for v, w, x0 in ((y, z, shape.d_y * L), (L - y, z, Lx - shape.d_y * L),
                     (z, y, shape.d_z * L), (L - z, y, Lx - shape.d_z * L)):
        # t runs 0..1 from the face to the body axis; the far stub is the near
        # one reflected in its face, so both lean the same way in x.
        t = v / c
        sgn = 1.0 if x0 < 0.5 * Lx else -1.0
        inside |= ((t >= 0) & (t <= 1)
                   & ((x - (x0 + sgn * lean * t)) ** 2 + (w - c) ** 2 < rl * rl))
    return inside


def check_shape(shape, L, n):
    """Reject shapes that cannot tile, with an actionable message."""
    A = shape.ax
    if A < 2:
        raise ValueError("the cell shape needs --ax >= 2")
    lean, r = shape.lean_x(), shape.lat_r
    for nm, d in (('y', shape.d_y), ('z', shape.d_z)):
        if d - r <= 0:
            raise ValueError(
                f"the {nm} connectors are only {d:.2f} L from the x faces but "
                f"are {r:.2f} L in radius. Raise the {nm} distance or shrink "
                f"--lat-r.")
        if 2 * d >= A:
            raise ValueError(
                f"the {nm} connectors at {d:.2f} L from each end leave no shift "
                f"(the box is {A} L): the two would coincide. Lower the distance.")
        if d + lean + r >= A - d:
            raise ValueError(
                f"the {nm} stub leans {lean:.2f} L in x, which from {d:.2f} L "
                f"runs into its opposite number at {A - d:.2f} L. Raise --lean "
                f"towards 90 deg, or lower the distance.")
    if max(shape.body_r, shape.lat_r) >= 0.5:
        raise ValueError("a radius reaches the box wall, leaving no ECS: "
                         "shrink --body-r or --lat-r")
    if shape.lat_r > 0.95 * shape.body_r:
        raise ValueError(
            f"--lat-r ({shape.lat_r}) must stay below 0.95 * --body-r "
            f"({shape.body_r}): at the body radius a stub's far cap is tangent "
            f"to the body surface and the OCC boolean degenerates")
    h = L / n
    return {"the ECS channel at the box face centres": (0.5 - shape.body_r) * L / h,
            "the connectors": 2 * shape.lat_r * L / h}


def ascii_slices(shape, L, width=110, rows=26):
    """Print mid-plane slices of the analytic cell solid, to judge the shape."""
    A = shape.ax

    def render(title, hlab, vlab, to_point):
        cols = min(width, int(rows * A))
        hs = (np.arange(cols) + 0.5) / cols
        vs = (np.arange(rows) + 0.5) / rows
        H, V = np.meshgrid(hs, vs, indexing='ij')
        inside = cell_contains(to_point(H, V), shape, L)
        print(f"    {title}  ({hlab} across, {vlab} down)")
        for row in inside.T:
            print("      " + "".join('#' if v else '.' for v in row))

    render("y = L/2", "x", "z", lambda H, V: np.stack(
        [H * A * L, np.full_like(H, 0.5 * L), V * L], axis=-1))
    render("z = L/2", "x", "y", lambda H, V: np.stack(
        [H * A * L, V * L, np.full_like(H, 0.5 * L)], axis=-1))


def classify_box_plus(n):
    """The original 3D-plus cell: three orthogonal L/2 bars, exactly 50% ICS."""
    i = np.arange(n)
    c = (i >= n // 4) & (i < 3 * n // 4)
    cx, cy, cz = c[:, None, None], c[None, :, None], c[None, None, :]
    return (cx & cy) | (cx & cz) | (cy & cz)


def build_vertices(Gx, Gy, Gz, h):
    """Structured vertex grid of size (Gx+1, Gy+1, Gz+1) with spacing h.
    vid(i,j,k) = (i*(Gy+1)+j)*(Gz+1)+k (C-order)."""
    xs = np.arange(Gx + 1) * h
    ys = np.arange(Gy + 1) * h
    zs = np.arange(Gz + 1) * h
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def build_tets(Gx, Gy, Gz):
    """Kuhn-split every voxel of the (Gx, Gy, Gz) grid into 6 tets.

    Tets are emitted in 6 blocks of Gx*Gy*Gz, all in C-order voxel numbering,
    so a per-voxel array `a` becomes a per-tet array with np.tile(a, 6).
    """
    ii, jj, kk = np.meshgrid(np.arange(Gx), np.arange(Gy), np.arange(Gz), indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    def vid(a, b, c):
        return (a * (Gy + 1) + b) * (Gz + 1) + c

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
    return np.concatenate(tets, axis=0).astype(np.int64)


def build_voxel_tags(nx, ny, nz, ics_box, pad):
    """Tile the plus cell's voxel mask over the lattice and 2-color it.

    Boxes are 2-colored on the 3D checkerboard by (I+J+K) % 2. Axis-neighbours
    always differ in parity, which is what keeps touching cells on different
    tags (ECS -> 0, intracellular -> 1 or 2). Unlike the `cell` shape this
    lattice needs no mirroring: the plus is x-symmetric, and mirroring the
    structured grid would flip its Kuhn diagonals and break conformity.

    `pad` voxels of pure ECS wrap the whole block.
    """
    bx, by, bz = ics_box.shape
    ics = np.tile(ics_box, (nx, ny, nz))

    I = np.arange(nx * bx) // bx
    J = np.arange(ny * by) // by
    K = np.arange(nz * bz) // bz
    color = (I[:, None, None] + J[None, :, None] + K[None, None, :]) % 2

    tags = np.where(ics, 1 + color, 0).astype(np.int32)
    if pad:
        full = np.zeros((nx * bx + 2 * pad, ny * by + 2 * pad, nz * bz + 2 * pad),
                        dtype=np.int32)
        full[pad:pad + nx * bx, pad:pad + ny * by, pad:pad + nz * bz] = tags
        tags = full
    return tags


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




# --------------------------------------------------------------------------
# Lattice assembly
# --------------------------------------------------------------------------

def _split_into_slabs(box, slabs, Lx):
    """Cut the box mesh into its `slabs` x slabs, each with compacted points.

    The box is meshed as `slabs` separate solids, so a slab's faces already mate
    with its neighbours' -- and with the next box's across the periodic x face.
    A row of boxes is therefore a *cyclic chain of slabs*, which is what lets a
    partial-box gap be filled exactly.
    """
    pts, tets, ics = box
    slab_w = Lx / slabs
    which = np.clip((pts[tets].mean(axis=1)[:, 0] / slab_w).astype(int), 0, slabs - 1)
    out = []
    for s in range(slabs):
        sel = which == s
        t = tets[sel]
        used, inv = np.unique(t, return_inverse=True)
        out.append((pts[used], inv.reshape(-1, 4), ics[sel]))
    return out


def build_lattice(nx, ny, nz, pad, box, box_size, slabs, step_y, step_z, weld_tol):
    """Stamp one box mesh over the shifted lattice and weld coincident nodes.

    Every box is a pure translate -- no mirroring -- so every connector leans the
    same way and the tissue has one fibre direction. That forces the shift: a
    connector entering the y = 0 face leaves the y = L face `shift` further along
    x, so the box above has to sit there. The offset is taken modulo the box, so
    rows zig-zag between a few x-positions rather than marching off diagonally:

        box (I,J,K) at x = I*Lx + ((J*step_y + K*step_z) mod slabs) * Lx/slabs

    **The zig-zag is then filled with ECS.** Left as-is the block is a staircase,
    with each row starting and ending at a different x; the extracellular space
    should instead fill out to a cuboid. Each row is extended at both ends with
    pure-ECS *slabs* -- the same slabs the box is already cut into, continuing
    the row's cyclic chain, so they mate exactly -- until every row spans the
    same x range. Every row gains the same number of slabs, so the result is a
    true cuboid with no ragged ends.

    That also removes a defect the staircase had: at the ragged ends only the
    diagonal (J+K) quadrants existed, so two of them met along a line and the
    block was non-manifold there, which is why a boundary layer could not be
    built on it.

    Wrapping changes who a connector reaches. When (J+K)*step rolls over a
    multiple of slabs, the y connector of box (I,J,K) lands a box further along
    x than (I,J+1,K) -- and those two have the *same* (I+J+K) parity, so the
    plain checkerboard would give two touching cells the same tag. Colouring by
    the *unwrapped* index I - walk//slabs fixes it: that is the index the cell
    would have had on the diagonal lattice, where the checkerboard is right.

    Args:
        nx, ny, nz: boxes holding a cell
        pad: extra shells of boxes, meshed the same but tagged pure ECS
        box: (points, tets, is_ics) of the single box, at the origin
        box_size: (Lx, L, L)
        slabs: x slabs per box (the shifts' common denominator; 1 for `plus`)
        step_y, step_z: slabs the lattice shifts per y resp. z step
        weld_tol: coincident-node tolerance

    Returns:
        points (N,3), tets (M,4), cell_tags (M,) with ECS 0 and ICS 1 / 2
    """
    box_size = np.asarray(box_size, float)
    Lx = box_size[0]
    slabs = max(1, int(slabs))
    step_y, step_z = int(step_y), int(step_z)
    slab_w = Lx / slabs

    counts = (nx + 2 * pad, ny + 2 * pad, nz + 2 * pad)
    pieces = _split_into_slabs(box, slabs, Lx)

    # How far the rows are offset from one another, in slabs. Every row is grown
    # by exactly this many slabs in total, so they all end up the same length.
    walks = [(J * step_y + K * step_z) % slabs
             for J in range(counts[1]) for K in range(counts[2])]
    w_max = max(walks)
    n_j = w_max + counts[0] * slabs          # slabs spanning the filled cuboid

    pts, tets, tags, offset = [], [], [], 0
    for J in range(counts[1]):
        for K in range(counts[2]):
            walk = J * step_y + K * step_z
            w = walk % slabs
            for j in range(n_j):
                s = (j - w) % slabs          # continue this row's cyclic chain
                sp, st, sics = pieces[s]
                shift_x = (j - s) * slab_w
                p = sp + np.array([shift_x, J * box_size[1], K * box_size[2]])
                # Inside the row's own boxes? Only then does it carry a cell.
                inside = w <= j < w + counts[0] * slabs
                I = (j - w) // slabs if inside else -1
                core = (inside and pad <= I < pad + nx and pad <= J < pad + ny
                        and pad <= K < pad + nz)
                if core:
                    color = (I - walk // slabs + J + K) % 2
                    tag = np.where(sics, 1 + color, 0).astype(np.int32)
                else:
                    tag = np.zeros(len(st), np.int32)
                pts.append(p)
                tets.append(st + offset)
                tags.append(tag)
                offset += len(sp)

    pts = np.concatenate(pts)
    tets = np.concatenate(tets)
    tags = np.concatenate(tags)

    # Weld: nodes on a shared face are bit-close because the box mesh is
    # periodic, so quantising to weld_tol collapses exactly the duplicates.
    q = np.round(pts / weld_tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    points = pts[first]
    tets = inv.reshape(-1)[tets]

    # The filled block is a cuboid, so its free surface is the plain formula.
    W = n_j * slab_w
    H, D = counts[1] * box_size[1], counts[2] * box_size[2]
    _verify_conforming(points, tets, 2 * (W * H + W * D + H * D))
    return points, tets, tags


def _boundary_sheets(bf, bt, tets, n_points):
    """Label each (face, corner) of a boundary triangulation by its *sheet*.

    Two corners at the same node are one sheet when their faces are joined
    through the fan of faces around that node. On a manifold surface that is one
    sheet per node. The zig-zag block is not manifold: at its ragged x ends only
    the diagonal (J+K) quadrants exist, so two of them meet along a line, and the
    four boundary faces along that line share a single edge. Unioning blindly
    across such an edge would fuse the two quadrants into one sheet whose
    averaged normal cancels to noise -- so at an edge carrying more than two
    boundary faces the faces are grouped by which side of the pinch they are on,
    found by walking the fan of tets around the edge through interior faces.

    Returns (labels shaped like bf, number of sheets).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    F = len(bf)
    item = np.arange(3 * F).reshape(F, 3)
    other1, other2 = bf[:, [1, 2, 0]], bf[:, [2, 0, 1]]
    keys = np.concatenate([bf.ravel() * n_points + other1.ravel(),
                           bf.ravel() * n_points + other2.ravel()])
    ids = np.concatenate([item.ravel(), item.ravel()])
    fidx = np.concatenate([np.repeat(np.arange(F), 3)] * 2)
    # Undirected edge id, so the two corners of an edge share a multiplicity.
    lo = np.minimum(np.concatenate([bf.ravel()] * 2),
                    np.concatenate([other1.ravel(), other2.ravel()]))
    hi = np.maximum(np.concatenate([bf.ravel()] * 2),
                    np.concatenate([other1.ravel(), other2.ravel()]))
    eid = lo * n_points + hi

    # A boundary edge is manifold when exactly two boundary faces carry it; each
    # face contributes it twice (once per endpoint), hence the 4.
    uniq_e, inv_e, cnt_e = np.unique(eid, return_inverse=True, return_counts=True)
    nonmanifold = cnt_e[inv_e] > 4

    order = np.argsort(keys[~nonmanifold], kind='stable')
    k = keys[~nonmanifold][order]
    i = ids[~nonmanifold][order]
    same = k[1:] == k[:-1]
    rows, cols = list(i[:-1][same]), list(i[1:][same])

    bad = np.unique(eid[nonmanifold])
    if len(bad):
        side = _pinch_sides(bad, bf, bt, tets, n_points)
        sel = np.nonzero(nonmanifold)[0]
        grp = {}
        for s in sel:
            grp.setdefault((keys[s], side[(eid[s], fidx[s])]), []).append(ids[s])
        for members in grp.values():
            for a, b in zip(members, members[1:]):
                rows.append(a)
                cols.append(b)

    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(3 * F, 3 * F))
    n_sheets, labels = connected_components(g, directed=False)
    return labels.reshape(F, 3), n_sheets


def _pinch_sides(bad_edges, bf, bt, tets, n_points):
    """For each non-manifold edge, which side of the pinch each boundary face is.

    Walks the fan of tets around the edge, joining two tets when they share a
    triangle containing the edge that is *not* a boundary face; a boundary face
    is where one side of the pinch ends. `bt` is the *owning tet* of each
    boundary face -- not its opposite vertex, which is a tempting mix-up that
    silently makes every face land on the same side. Only runs on pinch edges.
    """
    # node -> tets, built once as a CSR-style index.
    flat = tets.ravel()
    order = np.argsort(flat, kind='stable')
    owner = order // 4
    starts = np.searchsorted(flat[order], np.arange(n_points + 1))

    bset = set(map(tuple, np.sort(bf, axis=1).tolist()))
    face_at = {}
    for f in range(len(bf)):
        tri = np.sort(bf[f])
        for a, b in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            face_at.setdefault(int(a) * n_points + int(b), []).append(f)

    side = {}
    for e in bad_edges:
        a, b = int(e) // n_points, int(e) % n_points
        cand = set(owner[starts[a]:starts[a + 1]]) & set(owner[starts[b]:starts[b + 1]])
        parent = {t: t for t in cand}

        def find(t):
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        tri_map = {}
        for t in cand:
            rest = [n for n in tets[t] if n != a and n != b]
            for c in rest:
                key = tuple(sorted((a, b, int(c))))
                if key in bset:          # a boundary face closes this side
                    continue
                tri_map.setdefault(key, []).append(t)
        for shared in tri_map.values():
            for t in shared[1:]:
                ra, rb = find(shared[0]), find(t)
                if ra != rb:
                    parent[ra] = rb
        for f in face_at.get(int(e), []):
            side[(int(e), f)] = find(int(bt[f])) if int(bt[f]) in parent else -1
    return side


def pad_boundary_layers(points, tets, tags, layers, step):
    """Wrap the block in `layers` element-thick layers of pure ECS.

    A boundary *layer*, not extra boxes: every boundary node is offset along its
    area-averaged outward normal and the shell is filled with prisms, three tets
    each. Two properties make this the right construction here:

      * it shares the block's boundary triangulation exactly, so it is conforming
        by construction -- no welding, no coincident-node bookkeeping;
      * it is built from the *surface*, not from six planar slabs, so convex
        edges get a mitre and there are no gaps or overlaps at the corners. That
        matters because the zig-zag block is not convex: extruding face patches
        along their own axis would collide with the neighbouring row.

    The offset is scaled by 1/(n_hat . n_face) so each face plane really does
    move out by `step` rather than by its cosine, which is what makes the mitres
    square. At a re-entrant corner the offset points into the notch, so the layer
    must stay thin compared with the notch -- `check_pad` enforces that.
    """
    face_local = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    faces = tets[:, face_local].reshape(-1, 3)
    opp = tets[:, [0, 1, 2, 3]].reshape(-1)
    own = np.repeat(np.arange(len(tets)), 4)
    key = np.sort(faces, axis=1)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    k, f, o, w = key[order], faces[order], opp[order], own[order]
    start = np.nonzero(np.r_[True, np.any(k[1:] != k[:-1], axis=1)])[0]
    counts = np.diff(np.r_[start, len(k)])
    sel = start[counts == 1]
    bf, bo, bt = f[sel], o[sel], w[sel]

    # Orient each boundary triangle outward (away from its tet's fourth vertex).
    v0, v1, v2 = points[bf[:, 0]], points[bf[:, 1]], points[bf[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    inward = np.einsum('ij,ij->i', nrm, points[bo] - v0) > 0
    bf[inward] = bf[inward][:, [0, 2, 1]]
    nrm[inward] *= -1.0
    unit = nrm / np.linalg.norm(nrm, axis=1)[:, None]

    # Split the boundary into *sheets*: one per node per connected fan of faces
    # around it. On a manifold surface that is one sheet per node and changes
    # nothing, but the zig-zag block is not manifold -- at its ragged x ends only
    # the diagonal (J+K) quadrants exist, so two of them meet along a pinch line.
    # A node there sees -y, +y, -z and +z at once, its averaged normal cancels to
    # noise, and the layer folds. Giving each sheet its own offset nodes lets the
    # shell split along the pinch exactly as the block itself does.
    sheet, n_sheets = _boundary_sheets(bf, bt, tets, len(points))

    # Node normals, area-weighted (a cross product's length is twice the area)
    # and of fixed length `step`. A *square* mitre -- offsetting by step along
    # each distinct face direction -- is tempting because it grows the bounding
    # box exactly, but it moves a convex-corner node `step` tangentially, which
    # is the node spacing: the corner lands exactly on its neighbour's offset and
    # the prisms between them collapse. Averaging keeps the direction field
    # smooth across an edge, so consecutive nodes cannot cross; the layer is
    # correspondingly thinner at edges (by cos) which is only cosmetic.
    acc = np.zeros((n_sheets, 3))
    np.add.at(acc, sheet.reshape(-1), np.repeat(nrm, 3, axis=0))
    mag = np.linalg.norm(acc, axis=1)
    if mag.min() <= 1e-9 * np.abs(nrm).max():
        raise RuntimeError("a boundary sheet has no well-defined outward "
                           "normal; the block's surface is degenerate")
    offset = acc / mag[:, None] * step
    sheet_node = np.zeros(n_sheets, np.int64)
    sheet_node[sheet.reshape(-1)] = bf.reshape(-1)

    # Safeguard: never let a node travel further than a fraction of its own
    # shortest incident boundary edge, or the shell folds where the surface is
    # awkward -- at the ends of a pinch line, say, where the block goes from
    # touching-along-a-line to solid. The layer just gets locally thinner.
    e0 = bf.reshape(-1)
    e1 = np.concatenate([bf[:, [1, 2, 0]].reshape(-1, 1),
                         bf[:, [2, 0, 1]].reshape(-1, 1)], axis=1)
    elen = np.linalg.norm(points[e1] - points[e0][:, None, :], axis=2).min(axis=1)
    short = np.full(n_sheets, np.inf)
    np.minimum.at(short, sheet.reshape(-1), elen)
    cap = np.minimum(1.0, 0.45 * short / (layers * step))
    offset *= cap[:, None]

    # Stack the layers; level 0 is the block's own boundary nodes, so the shell
    # is welded to the block for free. Levels above are per sheet.
    levels = [bf]
    new_pts = [points]
    base = len(points)
    for l in range(1, layers + 1):
        new_pts.append(points[sheet_node] + offset * l)
        levels.append(base + sheet)
        base += n_sheets

    # Split each prism into three tets with the base nodes in ascending global
    # order, so neighbouring prisms pick the same diagonal on the quad they
    # share and the shell stays conforming.
    perm = np.argsort(bf, axis=1)
    rows = np.arange(len(bf))[:, None]
    shell = []
    for l in range(layers):
        b = levels[l][rows, perm]
        t = levels[l + 1][rows, perm]
        shell.append(np.stack([b[:, 0], b[:, 1], b[:, 2], t[:, 2]], axis=1))
        shell.append(np.stack([b[:, 0], b[:, 1], t[:, 2], t[:, 1]], axis=1))
        shell.append(np.stack([b[:, 0], t[:, 1], t[:, 2], t[:, 0]], axis=1))
    shell = np.concatenate(shell)

    points = np.concatenate(new_pts)
    tets = np.concatenate([tets, shell])
    tags = np.concatenate([tags, np.zeros(len(shell), np.int32)])

    v = points[tets]
    neg = np.einsum('ij,ij->i', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                    v[:, 3] - v[:, 0]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    # The block sat at the origin and the shell grew outward by at most
    # layers*step, so shifting by that puts the block at layers*step -- the
    # convention mesh_partition assumes when it strips the padding -- and leaves
    # every padded node at a non-negative coordinate.
    points = points + layers * step

    v = points[tets]
    vol = np.einsum('ij,ij->i', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                    v[:, 3] - v[:, 0])
    if vol.min() <= 0:
        raise RuntimeError(
            f"{int((vol <= 0).sum())} boundary-layer tets came out degenerate: "
            f"the layer is thick enough for the offset surface to fold over "
            f"itself. Lower --pad or raise --n.")
    return points, tets, tags


def _verify_conforming(points, tets, want_area):
    """The welded mesh must have exactly the block's free surface as boundary.

    A face left unwelded shows up twice as a boundary face, so comparing the
    total area of once-used faces against the expected free surface catches any
    node that failed to weld -- far stronger than eyeballing node counts.
    """
    face_local = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    faces = np.sort(tets[:, face_local].reshape(-1, 3), axis=1)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    f = faces[order]
    start = np.nonzero(np.r_[True, np.any(f[1:] != f[:-1], axis=1)])[0]
    counts = np.diff(np.r_[start, len(f)])

    if counts.max() > 2:
        raise RuntimeError(f"welding collapsed distinct nodes: "
                           f"{int((counts > 2).sum())} faces are shared by more "
                           f"than two tets. Lower the weld tolerance.")
    b = f[start[counts == 1]]
    v0, v1, v2 = points[b[:, 0]], points[b[:, 1]], points[b[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    if abs(area - want_area) > 1e-6 * want_area:
        raise RuntimeError(
            f"the tiled mesh is not conforming: boundary area {area:.3f} but the "
            f"block's free surface is {want_area:.3f}. Some face nodes did not "
            f"weld, so neighbouring boxes are not glued.")


def mesh_name(kind, nx, ny, nz, n, L, pad=0, ax=1, slabs=0, step_y=1, step_z=1,
              d_y=0.0, d_z=0.0, lat_r=0.0, lean=0.0):
    """Canonical mesh name. Optional parts are omitted at their defaults, so
    legacy names still parse. Distances and radii are in hundredths of L, the
    lean in whole degrees."""
    Ls = ('%g' % L).replace('.', 'p')
    name = f'{kind}_{nx}x{ny}x{nz}_n{n}_L{Ls}'
    if pad:
        name += f'_p{pad}'
    if ax != 1:
        name += f'_a{ax}'
    if slabs:
        name += f'_s{slabs}t{step_y}'
        if step_z != step_y:
            name += f'u{step_z}'
    if d_y or d_z:
        name += f'_d{int(round((d_y or d_z) * 100))}'
        if d_z and d_z != d_y:
            name += f'z{int(round(d_z * 100))}'
    if lat_r:
        name += f'_r{int(round(lat_r * 100))}'
    if lean:
        name += f'_g{int(round(lean))}'
    return name


def tet_volumes(points, tets):
    v = points[tets]
    return np.abs(np.einsum('ij,ij->i', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                            v[:, 3] - v[:, 0])) / 6.0


def generate(nx, ny, nz, n, L, prefix, pad=0, kind='cell', shape=None,
             dry_run=False, preview=True, verbose=False):
    if pad < 0:
        raise ValueError("--pad must be >= 0")
    t0 = time.perf_counter()
    shape = shape or CellShape()

    if kind == 'plus':
        if n % 4 != 0 or n < 4:
            raise ValueError("the plus shape needs n a multiple of 4 (>= 4)")
        if shape.ax != 1:
            raise ValueError("the plus shape is only defined for --ax 1")
        A = 1
    else:
        if n < 4:
            raise ValueError("the cell shape needs n >= 4 elements per L")
        A = shape.ax
        thin = check_shape(shape, L, n)
        sy, sz = shape.shift_y() * L, shape.shift_z() * L
        sh = max(sy, sz)
        print(f"Domain: {nx} x {ny} x {nz} boxes of {A * L:g} x {L:g} x {L:g}, "
              f"shape 'cell', element size {L / n:g}")
        print(f"Cell: radius {shape.body_r * L:.2f} (x end discs the same), "
              f"connectors r = {shape.lat_r * L:.2f}, leaning "
              f"{shape.lean_angle():.0f} deg off the cell axis")
        print(f"      y interfaces at x = {shape.d_y * L:.2f} and "
              f"{(A - shape.d_y) * L:.2f}; z interfaces at "
              f"{shape.d_z * L:.2f} and {(A - shape.d_z) * L:.2f} "
              f"of {A * L:.0f} -- both {shape.d_y * L:.2f} resp. "
              f"{shape.d_z * L:.2f} from their nearest end")
        print(f"Lattice: shifted by {sy:.2f} per y step ({shape.step_y}/"
              f"{shape.slabs} of the box) and {sz:.2f} per z step "
              f"({shape.step_z}/{shape.slabs}) -- forced by the interface "
              f"distances, since s = Lx - 2d")
        # Rows start at different x, and the gaps that leaves are filled with
        # pure-ECS slabs so the block is a cuboid rather than a staircase.
        slab_w = A * L / shape.slabs
        used = sorted({(J * shape.step_y + K * shape.step_z) % shape.slabs
                       for J in range(ny + 2 * pad)
                       for K in range(nz + 2 * pad)})
        fill = used[-1] * slab_w
        bb = ((nx + 2 * pad) * A * L + fill, (ny + 2 * pad) * L, (nz + 2 * pad) * L)
        print(f"  rows start at {len(used)} x-positions {slab_w:.0f} apart; the "
              f"gaps are filled with ECS ({fill:.0f} per row, "
              f"{100 * fill / bb[0]:.0f}% of the length) so the block is a cuboid")
        print(f"  bounding box {bb[0]:.0f} x {bb[1]:.0f} x {bb[2]:.0f}")
        for what, v in thin.items():
            print(f"  {what}: {v:.1f} elements"
                  + ("   WARNING: raise --n" if v < 2 else ""))

    if kind == 'cell' and (dry_run or preview):
        # Sample the analytic solid: the ICS fraction is a property of the
        # geometry, so it is known before any meshing happens.
        g = np.stack(np.meshgrid(*[(np.arange(m) + 0.5) / m
                                   for m in (64 * A, 64, 64)], indexing='ij'), -1)
        frac = float(cell_contains(g * np.array([A * L, L, L]), shape, L).mean())
        print(f"ICS volume fraction (per box): {frac:.4f}")
        print(f"Cube-partition load balance (cell : ECS remainder): "
              f"1 : {(1 - frac) / frac:.2f}")
        if preview:
            print("  Slices through one box (analytic solid):")
            ascii_slices(shape, L)
    if dry_run:
        print("\n--dry-run: nothing meshed or written")
        return

    # ---- one box, then the lattice --------------------------------------
    if kind == 'plus':
        gx, gy, gz = nx * n, ny * n, nz * n
        Gx, Gy, Gz = gx + 2 * pad, gy + 2 * pad, gz + 2 * pad
        h = L / n
        print(f"Domain: {nx} x {ny} x {nz} cubes of {L:g}^3, shape 'plus'")
        print(f"Global voxel grid: {Gx} x {Gy} x {Gz} (cell block {gx} x {gy} x {gz})")
        points = build_vertices(Gx, Gy, Gz, h)
        topology = build_tets(Gx, Gy, Gz)
        cell_tags = np.tile(build_voxel_tags(nx, ny, nz, classify_box_plus(n),
                                             pad).ravel(), 6)
    else:
        print(f"Meshing one box with gmsh ...")
        box = mesh_box(shape, L, L / n, verbose=verbose)
        print(f"  box: {len(box[0])} nodes, {len(box[1])} tets "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"Tiling {nx * ny * nz} boxes ...")
        points, topology, cell_tags = build_lattice(
            nx, ny, nz, 0, box, (A * L, L, L),
            slabs=shape.slabs, step_y=shape.step_y, step_z=shape.step_z,
            weld_tol=L * 1e-7)
        if pad:
            before = len(topology)
            points, topology, cell_tags = pad_boundary_layers(
                points, topology, cell_tags, pad, L / n)
            print(f"ECS boundary layers: {pad} x {L / n:g} = {pad * L / n:g} thick, "
                  f"+{len(topology) - before} tets "
                  f"({100 * (len(topology) - before) / before:.0f}%)")

    vol = tet_volumes(points, topology)
    ics = vol[cell_tags > 0].sum()
    print(f"Vertices: {len(points)}   Tets: {len(topology)}")
    print(f"ICS volume fraction (whole domain): {ics / vol.sum():.4f}")

    facet_topology, facet_tags, membrane_dict = build_facets(topology, cell_tags, points)
    n_membrane = int(np.sum(facet_tags != -5))
    print(f"Unique faces: {len(facet_topology)}   Membrane facets: {n_membrane}")
    print("Membrane dict (tag -> membrane tags):")
    for k in sorted(membrane_dict):
        label = "ECS" if k == 0 else f"ICS-{k}"
        print(f"    {k} ({label}): {sorted(membrane_dict[k])}")

    xdmf_file, h5_file = prefix + ".xdmf", prefix + ".h5"
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
    d = CellShape()
    p = argparse.ArgumentParser(
        description="Generate a cell-lattice weak-scaling mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--nx", type=int, default=4, help="boxes in x")
    p.add_argument("--ny", type=int, default=4, help="boxes in y")
    p.add_argument("--nz", type=int, default=4, help="boxes in z")
    p.add_argument("--n", type=int, default=8,
                   help="elements per L (cell: element size L/n; plus: voxels "
                        "per cube edge, a multiple of 4)")
    p.add_argument("--L", type=float, default=25.0, help="box edge in y and z")
    p.add_argument("--ax", type=int, default=d.ax,
                   help="x aspect: the box is (ax*L) x L x L")
    p.add_argument("--pad", type=int, default=0,
                   help="pure-ECS padding around the block, in element layers "
                        "of size L/n (cell: a boundary layer on the block's "
                        "surface; plus: voxels)")
    p.add_argument("--shape", choices=("cell", "plus"), default="cell",
                   help="'cell' = gmsh tetrahedra, 'plus' = original voxel grid")

    g = p.add_argument_group("cell shape (lengths in units of L)")
    g.add_argument("--body-r", type=float, default=d.body_r,
                   help="cell radius; the x end discs are the same")
    g.add_argument("--lat-r", type=float, default=d.lat_r,
                   help="radius of the connector stubs (at most --body-r)")
    g.add_argument("--dist", type=float, default=d.d_y,
                   help="distance of the y interfaces from their nearest x face. "
                        "This alone sets the lattice shift: s = Lx - 2*dist")
    g.add_argument("--dist-z", type=float, default=0.0,
                   help="the same for z (0 = same as --dist). The two directions "
                        "are independent")
    g.add_argument("--lean", type=float, default=d.lean,
                   help="stub angle off the cell axis, in degrees; 90 points "
                        "straight out of the face, smaller leans towards +x. "
                        "Free of the interface positions")
    g.add_argument("--slabs", type=int, default=0,
                   help="override the shifts' common denominator (0 = derive "
                        "from the distances)")

    p.add_argument("--dry-run", action="store_true",
                   help="print the shape and ASCII slices, mesh nothing")
    p.add_argument("--no-preview", action="store_true", help="skip the ASCII slices")
    p.add_argument("--verbose", action="store_true", help="let gmsh log")
    p.add_argument("--prefix", type=str, default=None,
                   help="output path prefix (default ../data/<canonical name>)")
    args = p.parse_args()

    ax = 1 if args.shape == 'plus' else args.ax
    shape = CellShape(ax=ax, body_r=args.body_r, lat_r=args.lat_r,
                      d_y=args.dist, d_z=args.dist_z or args.dist,
                      lean=args.lean, slabs=args.slabs)
    prefix = args.prefix or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        mesh_name(args.shape, args.nx, args.ny, args.nz, args.n, args.L,
                  args.pad,
                  ax,
                  shape.slabs if args.shape == 'cell' else 0,
                  shape.step_y if args.shape == 'cell' else 1,
                  shape.step_z if args.shape == 'cell' else 1,
                  shape.d_y if args.shape == 'cell' else 0.0,
                  shape.d_z if args.shape == 'cell' else 0.0,
                  args.lat_r if args.shape == 'cell' else 0.0,
                  args.lean if args.shape == 'cell' else 0.0))

    generate(args.nx, args.ny, args.nz, args.n, args.L, prefix, pad=args.pad,
             kind=args.shape, shape=shape, dry_run=args.dry_run,
             preview=not args.no_preview, verbose=args.verbose)


if __name__ == "__main__":
    main()
