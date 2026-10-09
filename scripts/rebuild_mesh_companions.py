"""Rebuild the .xdmf and .pickle companions of a cardioEMI mesh .h5.

The .xdmf is a fixed XML wrapper around the h5 datasets (only shapes and the
filename vary). The .pickle is the tags dictionary {volume_tag: set of facet
tags touching it}, recomputed from the topology and tag arrays: every tagged
facet is matched to its adjacent cells via sorted-vertex keys, and its facet
tag is added to the adjacent cells' volume tags.

Usage: python3 scripts/rebuild_mesh_companions.py data/gonzo37_colored.h5
       [--check]   compare against existing .pickle/.xdmf instead of writing
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

XDMF_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<Xdmf xmlns:xi="https://www.w3.org/2001/XInclude" Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Tetrahedron" NumberOfElements="{nc}">
        <DataItem Dimensions="{nc} 4" NumberType="Int" Format="HDF">{h5}:/Mesh/mesh/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{nv} 3" Format="HDF">{h5}:/Mesh/mesh/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="facet_tags" GridType="Uniform">
      <xi:include xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)"/>
      <Topology TopologyType="Triangle" NumberOfElements="{nf}">
        <DataItem Dimensions="{nf} 3" NumberType="Int" Format="HDF">{h5}:/Mesh/facet_tags/topology</DataItem>
      </Topology>
      <Attribute Name="facet_tags" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{nf}" Format="HDF">{h5}:/Mesh/facet_tags/Values</DataItem>
      </Attribute>
    </Grid>
    <Grid Name="cell_tags" GridType="Uniform">
      <xi:include xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)"/>
      <Topology TopologyType="Tetrahedron" NumberOfElements="{nc}">
        <DataItem Dimensions="{nc} 4" NumberType="Int" Format="HDF">{h5}:/Mesh/cell_tags/topology</DataItem>
      </Topology>
      <Attribute Name="cell_tags" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{nc}" Format="HDF">{h5}:/Mesh/cell_tags/Values</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""

TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def face_keys(tris, n_verts):
    """Encode sorted vertex triples as single int64 keys."""
    s = np.sort(tris.astype(np.int64), axis=1)
    return (s[:, 0] * n_verts + s[:, 1]) * n_verts + s[:, 2]


def rebuild(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        cells = f["Mesh/mesh/topology"][:]
        n_verts = f["Mesh/mesh/geometry"].shape[0]
        cell_tags = f["Mesh/cell_tags/Values"][:]
        facets = f["Mesh/facet_tags/topology"][:]
        facet_tags = f["Mesh/facet_tags/Values"][:]
    n_cells = len(cells)
    n_facets_total = len(facets)
    # non-membrane facets carry the DEFAULT (-5) filler tag and are not part
    # of the tags dictionary (see geometry.py get_facet_tags_and_dictionary)
    membrane = facet_tags >= 0
    facets_mem = facets[membrane]
    facet_tags_mem = facet_tags[membrane]
    print(f"{h5_path.name}: {n_cells} cells, {n_verts} vertices, "
          f"{n_facets_total} facets ({membrane.sum()} membrane), "
          f"{len(np.unique(cell_tags))} volume tags")

    # all 4 faces of every tet, keyed; cell index alongside
    tet_faces = cells[:, TET_FACES].reshape(-1, 3)
    tf_keys = face_keys(tet_faces, n_verts)
    del tet_faces
    tf_cells = np.repeat(np.arange(n_cells, dtype=np.int64), 4)
    order = np.argsort(tf_keys, kind="stable")
    tf_keys = tf_keys[order]
    tf_cells = tf_cells[order]

    f_keys = face_keys(facets_mem, n_verts)
    # each facet matches 1 (boundary) or 2 (interior) tet faces
    lo = np.searchsorted(tf_keys, f_keys, side="left")
    hi = np.searchsorted(tf_keys, f_keys, side="right")
    if not ((hi - lo) >= 1).all():
        raise RuntimeError("facet without adjacent cell — inconsistent mesh")

    tags_dict = {int(t): set() for t in np.unique(cell_tags)}
    counts = hi - lo
    # gather (cell_tag, facet_tag) pairs for all adjacencies
    for off in range(counts.max()):
        m = counts > off
        adj_cells = tf_cells[lo[m] + off]
        ct = cell_tags[adj_cells]
        ft = facet_tags_mem[m]
        pairs = np.unique(np.stack([ct, ft], axis=1), axis=0)
        for c, t in pairs:
            tags_dict[int(c)].add(np.int32(t))

    tags_dict = {np.int32(k): v for k, v in tags_dict.items()}
    xdmf = XDMF_TEMPLATE.format(nc=n_cells, nv=n_verts, nf=n_facets_total,
                                h5=h5_path.name)
    return tags_dict, xdmf


def main():
    h5_path = Path(sys.argv[1])
    check = "--check" in sys.argv
    tags_dict, xdmf = rebuild(h5_path)
    pkl_path = h5_path.with_suffix(".pickle")
    xdmf_path = h5_path.with_suffix(".xdmf")

    if check:
        with open(pkl_path, "rb") as f:
            ref = pickle.load(f)
        ref_norm = {int(k): {int(x) for x in v} for k, v in ref.items()}
        new_norm = {int(k): {int(x) for x in v} for k, v in tags_dict.items()}
        print("pickle match:", ref_norm == new_norm)
        ref_xdmf = xdmf_path.read_text()
        print("xdmf match:", ref_xdmf.strip() == xdmf.strip())
    else:
        for p in (pkl_path, xdmf_path):
            if p.exists():
                raise SystemExit(f"{p} already exists — refusing to overwrite")
        with open(pkl_path, "wb") as f:
            pickle.dump(tags_dict, f)
        xdmf_path.write_text(xdmf)
        print(f"wrote {pkl_path} and {xdmf_path}")


if __name__ == "__main__":
    main()
