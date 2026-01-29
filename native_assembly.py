"""
Native matrix assembly for direct Ginkgo integration.

This module assembles DOLFINx block forms directly into COO format with global
indices, bypassing PETSc. It uses DOLFINx's native assembly which returns
MatrixCSR with un-accumulated ghost values - exactly what Ginkgo's
assembly_mode::communicate needs.

The key insight is that dfx.fem.assemble_matrix returns data BEFORE ghost
communication, while PETSc's MatMPIAIJGetSeqAIJ requires assembly first.
"""

import numpy as np
import dolfinx as dfx
from mpi4py import MPI


def _get_bc_dofs_per_block(bcs, restrictions, n_blocks):
    """
    Extract BC DOF indices (local unrestricted) for each block.

    Returns a list of sets, one per block, containing local unrestricted
    DOF indices that have Dirichlet BCs applied.
    """
    bc_dofs_per_block = [set() for _ in range(n_blocks)]

    for bc in bcs:
        # Get the function space this BC applies to
        bc_V = bc.function_space

        # Find which block this BC belongs to by matching function spaces
        for block_idx in range(n_blocks):
            # The restriction's dofmap comes from the original function space
            # We need to check if this BC's function space matches the block's space
            # by comparing the dofmap's index_map
            restr = restrictions[block_idx]

            # Get BC DOF indices (these are local unrestricted indices in bc_V)
            # bc.dof_indices returns (dofs, owned_mask)
            bc_dof_indices = bc.dof_indices()[0]

            # Check if this BC's function space matches this block
            # by seeing if the BC dofs are in the restriction's unrestricted_to_restricted map
            # This works because each block has a cloned function space
            if len(bc_dof_indices) > 0:
                # Check if at least one BC dof is in this block's restriction
                sample_dof = bc_dof_indices[0]
                if sample_dof in restr.unrestricted_to_restricted:
                    # This BC belongs to this block
                    bc_dofs_per_block[block_idx].update(bc_dof_indices)
                    break

    return bc_dofs_per_block


def assemble_block_to_coo(forms, restrictions, bcs, comm):
    """
    Assemble block forms into COO format with global indices.

    Each rank assembles its local contributions using DOLFINx native assembly,
    which returns MatrixCSR with un-accumulated ghost values. The COO data
    can be passed to Ginkgo's read_distributed with assembly_mode::communicate,
    which handles summing contributions from different ranks.

    Dirichlet BCs are applied symmetrically: both rows AND columns corresponding
    to BC DOFs are zeroed, with 1.0 on the diagonal. This preserves matrix
    symmetry, which is important for CG solver and BDDC preconditioner.

    Note: This assumes BC values are zero (no RHS modification needed for lifting).

    Parameters
    ----------
    forms : list of list of Form
        Block forms a[i][j], where None indicates zero block
    restrictions : list of DofMapRestriction
        Restrictions defining active DOFs for each block row/column
    bcs : list of DirichletBC
        Boundary conditions (used to identify BC DOFs for diagonal)
    comm : MPI.Comm
        MPI communicator

    Returns
    -------
    row_indices : ndarray[int64]
        Global row indices of non-zeros
    col_indices : ndarray[int64]
        Global column indices of non-zeros
    values : ndarray[float64]
        Non-zero values
    global_size : int
        Total size of the global block matrix
    row_ranges : ndarray[int64]
        Partition ranges: rank i owns rows [row_ranges[i], row_ranges[i+1])
    matrix_to_vertex : dict[int, int]
        Mapping from matrix global DOF index to mesh global vertex index
        (for owned DOFs on this rank only)
    contributed_vertices : set[int]
        Set of global mesh vertex indices where this rank contributes
        nonzero values (used for partition visualization)
    """
    n_blocks = len(forms)
    rank = comm.rank
    size = comm.size

    # Extract BC DOFs for each block
    bc_dofs_per_block = _get_bc_dofs_per_block(bcs, restrictions, n_blocks)

    # Compute block sizes from restrictions
    # Each restriction has an index_map with size_local and size_global
    block_sizes_local = []
    block_sizes_global = []

    for r in restrictions:
        block_sizes_local.append(r.index_map.size_local)
        block_sizes_global.append(r.index_map.size_global)

    total_global_size = sum(block_sizes_global)
    total_local_size = sum(block_sizes_local)

    # Compute row partition ranges across all ranks (rank-major layout)
    # PETSc uses rank-major ordering: all blocks for rank 0, then all blocks for rank 1, etc.
    all_local_sizes = comm.allgather(total_local_size)
    row_ranges = [0]
    for s in all_local_sizes:
        row_ranges.append(row_ranges[-1] + s)

    # Compute the global start index for each block on each rank
    # block_starts[r][b] = global start index for block b on rank r
    all_block_sizes_local = []
    for b in range(n_blocks):
        all_block_sizes_local.append(comm.allgather(block_sizes_local[b]))

    # block_global_starts[b] = global start index for block b on THIS rank
    block_global_starts = []
    for b in range(n_blocks):
        # Sum of: all blocks on ranks < this rank + blocks 0..b-1 on this rank
        start = row_ranges[rank]  # Start of this rank's DOFs
        for prev_b in range(b):
            start += block_sizes_local[prev_b]
        block_global_starts.append(start)

    # Collect all COO entries
    all_rows = []
    all_cols = []
    all_vals = []

    # Track BC rows we've seen (to add diagonal 1.0 entries)
    bc_rows_added = set()

    # Track mapping from matrix global DOF index to mesh vertex index
    # For P1 elements, unrestricted local DOF index = mesh vertex index
    matrix_to_vertex = {}

    # Track mesh vertices where this rank contributes nonzero values
    # This is used for partition visualization to show facets only where
    # a rank has actual nonzero contributions to all three DOFs
    contributed_vertices = set()

    for i in range(n_blocks):
        bc_dofs_i = bc_dofs_per_block[i]

        for j in range(n_blocks):
            bc_dofs_j = bc_dofs_per_block[j]
            form_ij = forms[i][j]
            if form_ij is None:
                continue

            # Assemble using native DOLFINx (returns MatrixCSR with un-accumulated ghosts)
            A_ij = dfx.fem.assemble_matrix(form_ij)

            # Get CSR data
            indptr = A_ij.indptr
            indices = A_ij.indices
            data = A_ij.data

            # Get the unrestricted-to-restricted mappings (LOCAL indices)
            # Keys: local unrestricted indices, Values: local restricted indices
            row_u2r = restrictions[i].unrestricted_to_restricted
            col_u2r = restrictions[j].unrestricted_to_restricted

            # Get restriction index maps for converting restricted local → global
            row_restr_imap = restrictions[i].index_map
            col_restr_imap = restrictions[j].index_map

            # Build a function to convert global restricted index to global block matrix index
            # This accounts for the rank-major ordering PETSc uses
            def restricted_to_block_matrix(global_restr_idx, block_idx, restr_imap, all_block_local):
                """Convert global restricted index to global block matrix index."""
                # Find which rank owns this global restricted index
                cumsum = 0
                owner_rank = -1
                local_on_owner = -1
                for r in range(size):
                    if cumsum + all_block_local[r] > global_restr_idx:
                        owner_rank = r
                        local_on_owner = global_restr_idx - cumsum
                        break
                    cumsum += all_block_local[r]

                if owner_rank == -1:
                    raise ValueError(f"Could not find owner for global_restr_idx={global_restr_idx}")

                # Compute global block matrix index for this DOF
                # = sum of total DOFs on ranks < owner + sum of block sizes before block_idx on owner + local_on_owner
                global_idx = row_ranges[owner_rank]
                for b in range(block_idx):
                    global_idx += all_block_sizes_local[b][owner_rank]
                global_idx += local_on_owner
                return global_idx

            # Process each row (local unrestricted index)
            n_total_rows = len(indptr) - 1
            for local_row_unr in range(n_total_rows):
                # Map local unrestricted → local restricted (skip if not in restriction)
                if local_row_unr not in row_u2r:
                    continue
                local_row_restr = row_u2r[local_row_unr]

                # Check if this is a BC row
                is_bc_row = local_row_unr in bc_dofs_i

                # Convert local restricted → global restricted → global block matrix
                if local_row_restr < row_restr_imap.size_local:
                    # Owned row
                    global_row = block_global_starts[i] + local_row_restr
                else:
                    # Ghost row: get global restricted index from ghosts array
                    ghost_idx = local_row_restr - row_restr_imap.size_local
                    global_restr_row = row_restr_imap.ghosts[ghost_idx]
                    global_row = restricted_to_block_matrix(global_restr_row, i, row_restr_imap, all_block_sizes_local[i])

                # For BC rows on diagonal blocks, add 1.0 on diagonal
                if is_bc_row and i == j and global_row not in bc_rows_added:
                    all_rows.append(global_row)
                    all_cols.append(global_row)
                    all_vals.append(1.0)
                    bc_rows_added.add(global_row)

                    # Track matrix-to-vertex mapping for this owned row (has actual entry)
                    test_space = form_ij.function_spaces[0]
                    global_vertex = int(test_space.dofmap.index_map.local_to_global(np.array([local_row_unr], dtype=np.int32))[0])
                    if local_row_restr < row_restr_imap.size_local and global_row not in matrix_to_vertex:
                        matrix_to_vertex[global_row] = global_vertex
                    # Track this vertex as having nonzero contribution from this rank
                    contributed_vertices.add(global_vertex)

                # Skip all other entries for BC rows
                if is_bc_row:
                    continue

                # Process columns in this row
                row_has_entry = False
                test_space = form_ij.function_spaces[0]
                trial_space = form_ij.function_spaces[1]
                for k in range(indptr[local_row_unr], indptr[local_row_unr + 1]):
                    local_col_unr = indices[k]
                    val = data[k]

                    if val == 0.0:
                        continue

                    # Map local unrestricted → local restricted
                    if local_col_unr not in col_u2r:
                        continue

                    # Skip BC columns for symmetric BC application
                    # (zeroing columns in addition to rows preserves symmetry)
                    if local_col_unr in bc_dofs_j:
                        continue

                    local_col_restr = col_u2r[local_col_unr]

                    # Convert local restricted → global block matrix index
                    if local_col_restr < col_restr_imap.size_local:
                        # Owned column
                        global_col = block_global_starts[j] + local_col_restr
                    else:
                        # Ghost column
                        ghost_idx = local_col_restr - col_restr_imap.size_local
                        global_restr_col = col_restr_imap.ghosts[ghost_idx]
                        global_col = restricted_to_block_matrix(global_restr_col, j, col_restr_imap, all_block_sizes_local[j])

                    all_rows.append(global_row)
                    all_cols.append(global_col)
                    all_vals.append(val)
                    row_has_entry = True

                    # Track column vertex as having nonzero contribution
                    global_col_vertex = int(trial_space.dofmap.index_map.local_to_global(np.array([local_col_unr], dtype=np.int32))[0])
                    contributed_vertices.add(global_col_vertex)

                # Track matrix-to-vertex mapping only for owned rows that have actual entries
                if row_has_entry:
                    global_row_vertex = int(test_space.dofmap.index_map.local_to_global(np.array([local_row_unr], dtype=np.int32))[0])
                    # Track row vertex as having nonzero contribution
                    contributed_vertices.add(global_row_vertex)
                    if local_row_restr < row_restr_imap.size_local and global_row not in matrix_to_vertex:
                        matrix_to_vertex[global_row] = global_row_vertex

    # Convert to numpy arrays
    row_indices = np.array(all_rows, dtype=np.int64)
    col_indices = np.array(all_cols, dtype=np.int64)
    values = np.array(all_vals, dtype=np.float64)

    return row_indices, col_indices, values, total_global_size, np.array(row_ranges, dtype=np.int64), matrix_to_vertex, contributed_vertices
