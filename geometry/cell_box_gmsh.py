"""Unstructured (gmsh) mesh of one box of the weak-scaling cell lattice.

The box is (ax*L) x L x L and holds one cell:

  * a **body** cylinder along the x axis spanning the whole box, so both x-face
    cross-sections are full-radius discs and cells butt together end to end;
  * four **connector stubs**, one per lateral face, each running from its face
    to the body axis. The y pair sits `d_y` from the small-x and large-x faces
    respectively (likewise z), and that distance alone sets the lattice shift:
    a cell's +y interface at Lx-d must land on its neighbour's -y interface at
    d, so the neighbour sits s = Lx - 2d further along x. The stubs' *angle* is
    therefore free -- a stub only has to reach the body. Each is a circular disc
    extruded along an oblique vector rather than a cylinder swept along a tilted
    axis, so every cross-section parallel to its face is an exact circle: the
    interface stays round while the stub leans.

Every cell is a pure **translate** of every other one, so every connector leans
the same way and the tissue has one consistent fibre direction. That is only
possible if the lattice is *shifted*: a connector entering the y = 0 face at
x = lat_x leaves the y = L face at x = lat_x + s, so the neighbour above must sit
s further along x. The lattice vectors are therefore

    a1 = (Lx, 0, 0)      a2 = (s_y, L, 0)      a3 = (s_z, 0, L)

and the block of boxes is a staircase, not a cuboid. y and z may use
different shifts, so a box's offset is (J*p_y + K*p_z)/slabs of the box length.

The rectangular box is still a fundamental domain of that lattice, but its +y
face is the -y face translated by (s, L, 0), which runs off the end in x. Split
the box into `m = Lx / s` equal slabs along x and the correspondence becomes
exact, slab by slab: slab k of the -y face maps to slab k+1 of the +y face, and
the last slab wraps to the first under the extra -a1. So gmsh gets m pure
translations per face pair instead of one impossible one:

    +x <- -x   translate (Lx, 0, 0)
    +y <- -y   translate (s, L, 0), less (Lx,0,0) for the last slab
    +z <- -z   translate (s, 0, L), likewise

Mating faces then carry identical triangulations and tiling only has to weld
coincident nodes.
"""

import numpy as np

# Tet element type in gmsh's numbering.
_GMSH_TET = 4


def _oblique_tube(occ, centre, normal, vec, radius):
    """A disc of `radius` at `centre` with the given face `normal` ('y' or 'z'),
    extruded along `vec`. Returns the volume tag."""
    d = occ.addDisk(*centre, radius, radius)          # addDisk gives normal +z
    if normal == 'y':
        occ.rotate([(2, d)], *centre, 1, 0, 0, -np.pi / 2)
    return [t for dim, t in occ.extrude([(2, d)], *vec) if dim == 3][0]


def _plane_surfaces(gmsh, Lx, Ly, Lz, tol=1e-6):
    """Group every surface by the box face plane it lies in ('-x', '+y', ...).

    A surface belongs to a plane when its whole bounding box is flat against it.
    The internal slab interfaces sit at intermediate x and so are skipped.
    """
    out = {}
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        for i, (lo, hi, nm) in enumerate(((0, Lx, 'x'), (0, Ly, 'y'), (0, Lz, 'z'))):
            if abs(bb[i] - lo) < tol and abs(bb[i + 3] - lo) < tol:
                out.setdefault('-' + nm, []).append(tag)
            elif abs(bb[i] - hi) < tol and abs(bb[i + 3] - hi) < tol:
                out.setdefault('+' + nm, []).append(tag)
    return out


def _signature(gmsh, tag):
    return (np.array(gmsh.model.occ.getCenterOfMass(2, tag)),
            gmsh.model.occ.getMass(2, tag))


def _pair_shifted(gmsh, masters, slaves, shift, Lx, tol=1e-6):
    """Pair each master face piece with its image under `shift`, wrapping in x.

    Returns [(master, slave, transform)]. Pieces are matched on centre of mass
    *and* area: on an x face the membrane disc and the ECS annulus around it
    share a centre of mass, so position alone is ambiguous.
    """
    sig = {t: _signature(gmsh, t) for t in list(masters) + list(slaves)}
    pairs = []
    for m in masters:
        com, area = sig[m]
        t = np.asarray(shift, float)
        if com[0] + t[0] > Lx + tol:            # this slab wraps round the box
            t = t - np.array([Lx, 0.0, 0.0])
        want = com + t
        hit = [s for s in slaves
               if np.abs(sig[s][0] - want).max() < tol
               and abs(sig[s][1] - area) < tol * max(1.0, area)]
        if len(hit) != 1:
            raise RuntimeError(
                f"cannot pair face surface {m} (centre {np.round(com, 3)}, area "
                f"{area:.4f}) with its image at {np.round(want, 3)}: {len(hit)} "
                f"candidates. The shifted faces are not images of each other, so "
                f"the lattice cannot tile.")
        pairs.append((m, hit[0], t))
    return pairs


def mesh_box(shape, L, h, curvature=16, verbose=False):
    """Mesh one box of the lattice.

    Args:
        shape: CellShape (radii and offsets in units of L)
        L: box edge in y and z; the box is (shape.ax*L) x L x L
        h: target element size, in the same units as L
        curvature: elements per 2*pi of curvature (0 disables curvature refinement)

    Returns:
        points:  (N, 3) node coordinates
        tets:    (M, 4) tet -> node indices
        is_ics:  (M,) bool, True for intracellular tets
    """
    import gmsh

    r = shape.radii()
    R, r_lat = r['body'] * L, r['lat'] * L
    Lx, Ly, Lz = shape.ax * L, L, L
    c = 0.5 * L
    m = shape.slabs
    s_y, s_z = shape.shift_y() * L, shape.shift_z() * L
    d_y, d_z = shape.d_y * L, shape.d_z * L
    lean = shape.lean_x() * L        # x travel from a face to the body axis

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cell_box")
        occ = gmsh.model.occ

        # Slabs, so every face piece has an exact image one slab further along.
        slab_w = Lx / m
        slabs = [(3, occ.addBox(k * slab_w, 0, 0, slab_w, Ly, Lz)) for k in range(m)]

        body = occ.addCylinder(0, c, c, Lx, 0, 0, R)
        # Four stubs, each running from its own face to the body axis. The two
        # in a direction are NOT one tube through the cell: the near one sits d
        # from the small-x face, the far one d from the large-x face, and that
        # distance alone sets the lattice shift. Both lean the same way in x.
        arms = [_oblique_tube(occ, (d_y, 0.0, c), 'y', (lean, c, 0), r_lat),
                _oblique_tube(occ, (Lx - d_y, Ly, c), 'y', (-lean, -c, 0), r_lat),
                _oblique_tube(occ, (d_z, c, 0.0), 'z', (lean, 0, c), r_lat),
                _oblique_tube(occ, (Lx - d_z, c, Lz), 'z', (-lean, 0, -c), r_lat)]
        # Fuse ONE tool at a time. A multi-tool occ.fuse whose tools overlap each
        # other silently returns just the object -- the arms vanish with no error.
        cell = [(3, body)]
        for a in arms:
            cell, _ = occ.fuse(cell, [(3, a)])

        _, fmap = occ.fragment(slabs, cell)
        occ.synchronize()

        # fragment's map is per input: the last entry is what the cell became.
        ics_vols = [t for d, t in fmap[-1]]
        all_vols = [t for d, t in gmsh.model.getEntities(3)]
        ecs_vols = [t for t in all_vols if t not in ics_vols]
        if not ics_vols or not ecs_vols:
            raise RuntimeError(f"fragment gave ICS {ics_vols}, ECS {ecs_vols}; "
                               f"the cell must be strictly inside the box")

        # ---- periodic meshing constraints ------------------------------
        planes = _plane_surfaces(gmsh, Lx, Ly, Lz)
        # Both mating discs are built independently by addDisk, and the maps are
        # pure translations, so their seam points already correspond -- no OCC
        # mirror copies needed as an earlier mirrored variant required.
        for nm, shift in (('x', (Lx, 0, 0)), ('y', (s_y, Ly, 0)), ('z', (s_z, 0, Lz))):
            # _pair_shifted wraps a piece whose image runs past Lx, so a general
            # step/slabs shift needs no special casing here.
            for master, slave, t in _pair_shifted(gmsh, planes['-' + nm],
                                                  planes['+' + nm], shift, Lx):
                M = [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1]
                gmsh.model.mesh.setPeriodic(2, [slave], [master], M)

        # ---- mesh ------------------------------------------------------
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h / 4.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:
            raise RuntimeError(
                f"gmsh could not mesh the box: {e}\n"
                f"This is almost always the connectors grazing the body at too "
                f"shallow an angle. Raise --lean (now "
                f"{shape.lean_angle():.0f} deg off the cell axis) or lower "
                f"--lat-r (now {shape.lat_r}).") from None

        # ---- extract ---------------------------------------------------
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords, float).reshape(-1, 3)
        lookup = np.zeros(int(ntags.max()) + 1, np.int64)
        lookup[np.asarray(ntags, np.int64)] = np.arange(len(ntags))

        tets, is_ics = [], []
        for vols, ics in ((ics_vols, True), (ecs_vols, False)):
            for v in vols:
                etypes, _, enodes = gmsh.model.mesh.getElements(3, v)
                for et, nodes in zip(etypes, enodes):
                    if et != _GMSH_TET:
                        continue
                    t = lookup[np.asarray(nodes, np.int64).reshape(-1, 4)]
                    tets.append(t)
                    is_ics.append(np.full(len(t), ics))
        tets = np.concatenate(tets)
        is_ics = np.concatenate(is_ics)
    finally:
        gmsh.finalize()

    v = points[tets]
    neg = np.einsum('ij,ij->i', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                    v[:, 3] - v[:, 0]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return points, tets, is_ics
