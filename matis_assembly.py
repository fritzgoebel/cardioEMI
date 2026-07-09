"""
MATIS matrix assembly for PETSc PCBDDC preconditioner.

Builds the MATIS local matrix from the same global-index COO triples that the
Ginkgo DdMatrix path uses. Each rank's local matrix contains:
  - rows for every DOF this rank owns
  - rows for non-owned DOFs ONLY when this rank's cells contribute to them

This matches Ginkgo's DdMatrix layout and avoids the empty ghost rows that
multiphenicsx's restriction.index_map carries along, which previously made
PCBDDC's Neumann sub-PC factor a much larger local matrix than necessary.
"""

import numpy as np
from petsc4py import PETSc
from scipy.sparse import coo_matrix

from native_assembly import assemble_block_to_coo


def assemble_block_to_matis(forms, restrictions, bcs, comm):
    """
    Assemble block forms into PETSc MATIS format for use with PCBDDC.

    Returns
    -------
    A_is : PETSc.Mat
        MATIS matrix ready for use with PCBDDC.
    """
    rank = comm.rank

    # Same un-summed global-index COO triples that the Ginkgo DdMatrix uses.
    coo = assemble_block_to_coo(forms, restrictions, bcs, comm)
    global_rows, global_cols, vals, n_global, row_ranges = coo[0], coo[1], coo[2], coo[3], coo[4]

    owned_start = int(row_ranges[rank])
    owned_end   = int(row_ranges[rank + 1])
    n_owned     = owned_end - owned_start

    # Ghost set: globals that appear in this rank's COO but are not owned here.
    # Take the union over rows and cols so the local matrix is square.
    outside_r = (global_rows < owned_start) | (global_rows >= owned_end)
    outside_c = (global_cols < owned_start) | (global_cols >= owned_end)
    ghost_globals = np.unique(np.concatenate([global_rows[outside_r], global_cols[outside_c]]))

    # Local-to-global: owned rows first, then ghosts.
    l2g = np.concatenate([
        np.arange(owned_start, owned_end, dtype=PETSc.IntType),
        ghost_globals.astype(PETSc.IntType),
    ])
    n_local = len(l2g)

    # Vectorized global -> local lookup (owned by offset, ghost via searchsorted).
    def _to_local(g):
        out = np.empty(g.shape, dtype=np.int32)
        owned_mask = (g >= owned_start) & (g < owned_end)
        out[owned_mask] = (g[owned_mask] - owned_start).astype(np.int32)
        if (~owned_mask).any():
            ghost_pos = np.searchsorted(ghost_globals, g[~owned_mask])
            out[~owned_mask] = (n_owned + ghost_pos).astype(np.int32)
        return out

    local_rows_arr = _to_local(global_rows)
    local_cols_arr = _to_local(global_cols)
    local_vals_arr = np.ascontiguousarray(vals, dtype=np.float64)

    # Sum duplicates (multiple cells contributing to the same (i,j)).
    local_coo = coo_matrix(
        (local_vals_arr, (local_rows_arr, local_cols_arr)),
        shape=(n_local, n_local),
    )
    local_csr = local_coo.tocsr()

    A_local = PETSc.Mat().createAIJWithArrays(
        size=(n_local, n_local),
        csr=(local_csr.indptr.astype(PETSc.IntType),
             local_csr.indices.astype(PETSc.IntType),
             local_csr.data),
        comm=PETSc.COMM_SELF,
    )

    lgmap = PETSc.LGMap().create(l2g, comm=comm)

    A_is = PETSc.Mat()
    A_is.create(comm)
    A_is.setType(PETSc.Mat.Type.IS)
    A_is.setSizes(((n_owned, n_global), (n_owned, n_global)))
    A_is.setLGMap(lgmap, lgmap)
    A_is.setISLocalMat(A_local)
    A_is.assemble()

    # Per-rank size diagnostics so we can see the local-mat scope on every rank.
    n_ghosts = n_local - n_owned
    nnz = int(local_csr.nnz)
    sizes = comm.gather((rank, n_owned, n_ghosts, n_local, nnz), root=0)
    if comm.rank == 0:
        m, n = A_is.getSize()
        print(f"  MATIS matrix size: {m} x {n}")
        for r, no, ng, nl, nz in sizes:
            print(f"    rank {r}: owned={no} ghosts={ng} local={nl} nnz={nz}")

    return A_is
