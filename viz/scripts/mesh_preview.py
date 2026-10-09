#!/usr/bin/env python3
"""Membrane-only, size-capped preview of a mesh for the webapp viewer.

Run next to the mesh (on a cluster, inside the DOLFINx container) so only the
preview travels: the membrane triangles (facet tag > 0) and just the vertices
they use, written in the viewer's format (mesh_vertices.bin, membrane_facets.bin,
membrane_tags.bin, mesh_metadata.json). `bounds` in the metadata are those of
the whole mesh, ECS included, so sliders and the stimulus box match a full
conversion.

Above --max-facets triangles the surface is coarsened by vertex clustering:
vertices are snapped to a cubic grid, each occupied cell becomes one vertex at
the mean of its members, and triangles that collapse or duplicate drop out.
Crude but fast and numpy-only; the cell size starts from the membrane area
(a triangle covers about h^2/2) and grows until the budget is met.

usage: mesh_preview.py <mesh.h5> <out_dir> [--max-facets N]
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

CHUNK = 1 << 22


def _bounds_and_rows(dataset, rows):
    """Whole-mesh bounds plus the coordinates of `rows` (sorted), streamed in
    chunks: h5py point selection with millions of indices is orders of
    magnitude slower than reading everything and indexing in numpy."""
    lo = hi = None
    picked = []
    for i in range(0, dataset.shape[0], CHUNK):
        v = dataset[i:i + CHUNK]
        lo = v.min(0) if lo is None else np.minimum(lo, v.min(0))
        hi = v.max(0) if hi is None else np.maximum(hi, v.max(0))
        a, b = np.searchsorted(rows, [i, i + len(v)])
        picked.append(v[rows[a:b] - i])
    return lo, hi, np.concatenate(picked) if picked else np.zeros((0, 3))


def _membrane(f):
    """Membrane triangles (global vertex ids) and their tags, read in chunks."""
    topo, vals = f['/Mesh/facet_tags/topology'], f['/Mesh/facet_tags/Values']
    tris, tags = [], []
    for i in range(0, vals.shape[0], CHUNK):
        t = vals[i:i + CHUNK]
        keep = t > 0
        if keep.any():
            tris.append(topo[i:i + CHUNK][keep].astype(np.int64))
            tags.append(t[keep].astype(np.int32))
    if not tris:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int32)
    return np.concatenate(tris), np.concatenate(tags)


def _cluster(v, tris, tags, h):
    """Vertex clustering on a grid of cell size h."""
    q = np.floor((v - v.min(0)) / h).astype(np.int64)
    n = q.max(0) + 1
    key = (q[:, 0] * n[1] + q[:, 1]) * n[2] + q[:, 2]
    cells, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, minlength=len(cells)).astype(np.float64)
    nv = np.stack([np.bincount(inv, weights=v[:, k], minlength=len(cells)) / counts
                   for k in range(3)], 1)
    t = inv[tris]
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t, g = t[ok], tags[ok]
    # Same three cells and tag = same triangle after snapping; keep the first.
    s = np.sort(t, 1)
    order = np.lexsort((g, s[:, 2], s[:, 1], s[:, 0]))
    s, gs = s[order], g[order]
    new = np.ones(len(order), bool)
    new[1:] = (np.diff(s, axis=0) != 0).any(1) | (np.diff(gs) != 0)
    first = np.sort(order[new])
    return nv, t[first], g[first]


def make_preview(h5_path, out_dir, max_facets=1_000_000):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        tris, tags = _membrane(f)
        used, local = np.unique(tris, return_inverse=True)  # used is sorted
        tris = local.reshape(-1, 3)
        lo, hi, v = _bounds_and_rows(f['/Mesh/mesh/geometry'], used)
    v = v.astype(np.float64)
    full_facets = len(tris)

    h = None
    if full_facets > max_facets:
        a = v[tris]
        area = 0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1).sum()
        h = float(np.sqrt(2.0 * area / max_facets))
        while True:
            nv, nt, ng = _cluster(v, tris, tags, h)
            if len(nt) <= max_facets:
                break
            h *= 1.2
        v, tris, tags = nv, nt, ng

    v.astype(np.float32).tofile(out_dir / 'mesh_vertices.bin')
    tris.astype(np.uint32).tofile(out_dir / 'membrane_facets.bin')
    tags.astype(np.int32).tofile(out_dir / 'membrane_tags.bin')
    ext = float((hi - lo).max())
    meta = {
        'vertex_count': int(len(v)),
        'facet_count': int(len(tris)),
        'bounds': {a: [float(lo[i]), float(hi[i])] for i, a in enumerate('xyz')},
        'mesh_conversion_factor': 0.0001 if ext > 10 else 1.0,
        'unique_tags': sorted(int(t) for t in np.unique(tags)),
        'source_file': Path(h5_path).name,
        'preview': {
            'max_facets': int(max_facets),
            'full_facets': int(full_facets),
            'cluster_size': h,  # None = full resolution
            'seconds': round(time.time() - t0, 1),
        },
    }
    with open(out_dir / 'mesh_metadata.json', 'w') as mf:
        json.dump(meta, mf, indent=2)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('h5')
    ap.add_argument('out_dir')
    ap.add_argument('--max-facets', type=int, default=1_000_000)
    args = ap.parse_args()
    print('@@PREVIEW ' + json.dumps(make_preview(args.h5, args.out_dir, args.max_facets)))
