"""
Test script for native_assembly.py

This script compares the output of native assembly (COO format) against
PETSc assembly via multiphenicsx to verify correctness.

Run in Docker:
    docker run --rm -v "$(pwd):/work" -w /work ghcr.io/fenics/dolfinx/dolfinx:v0.9.0 \
        python test_native_assembly.py
"""

import numpy as np
import dolfinx as dfx
import multiphenicsx.fem
import multiphenicsx.fem.petsc
from mpi4py import MPI
from ufl import inner, grad, dx, TrialFunction, TestFunction
from petsc4py import PETSc

from native_assembly import assemble_block_to_coo


def compare_matrices(A_petsc_dense, A_native_dense, rtol=1e-12, atol=1e-14):
    """Compare two matrices with relative and absolute tolerance.

    Returns (passed, max_rel_diff, max_abs_diff, diff_indices)
    """
    abs_diff = np.abs(A_petsc_dense - A_native_dense)

    # Compute relative difference where PETSc is non-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(A_petsc_dense) + atol)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Entry fails if both absolute AND relative difference exceed tolerances
    failed = (abs_diff > atol) & (rel_diff > rtol)
    diff_indices = np.where(failed)

    passed = len(diff_indices[0]) == 0
    return passed, max_rel_diff, max_abs_diff, diff_indices


def test_single_block():
    """Test with a single block (no block structure, just restriction)."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("=" * 60)
        print("Test 1: Single block with restriction")
        print("=" * 60)

    # Create a simple unit square mesh
    mesh = dfx.mesh.create_unit_square(comm, 4, 4)
    V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

    # Create a restriction: only interior DOFs (exclude boundary)
    boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    all_dofs = np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)
    interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)
    interior_dofs = np.sort(interior_dofs)

    if rank == 0:
        print(f"  Total DOFs (local+ghost): {len(all_dofs)}")
        print(f"  Interior DOFs: {len(interior_dofs)}")

    # Create restriction
    restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, interior_dofs)

    # Create a simple Laplacian form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Compile form
    a_compiled = dfx.fem.form(a)

    # Assemble with PETSc via multiphenicsx
    A_petsc = multiphenicsx.fem.petsc.assemble_matrix_block(
        [[a_compiled]], bcs=[], restriction=([restriction], [restriction])
    )
    A_petsc.assemble()

    # Assemble with native assembly
    coo_rows, coo_cols, coo_vals, global_size, row_ranges = assemble_block_to_coo(
        [[a_compiled]], [restriction], [], comm
    )

    if rank == 0:
        print(f"  PETSc matrix size: {A_petsc.getSize()}")
        print(f"  Native global size: {global_size}")
        print(f"  Native COO entries: {len(coo_vals)}")

    # Gather COO data to rank 0
    all_coo_rows = comm.gather(coo_rows, root=0)
    all_coo_cols = comm.gather(coo_cols, root=0)
    all_coo_vals = comm.gather(coo_vals, root=0)

    # Convert PETSc matrix to sequential on rank 0 for comparison
    # Use PETSc's convertToSeqMat for proper handling
    m_global, n_global = A_petsc.getSize()

    if comm.size > 1:
        # For parallel case, gather matrix to rank 0
        from petsc4py import PETSc as P
        # Create sequential matrix on rank 0
        if rank == 0:
            A_seq = P.Mat().createAIJ([m_global, n_global], comm=P.COMM_SELF)
            A_seq.setUp()
        else:
            A_seq = None

        # Each rank contributes its local rows
        row_start, row_end = A_petsc.getOwnershipRange()
        local_rows = []
        local_cols = []
        local_vals = []
        for i in range(row_start, row_end):
            cols, vals = A_petsc.getRow(i)
            for c, v in zip(cols, vals):
                local_rows.append(i)
                local_cols.append(c)
                local_vals.append(v)

        # Gather to rank 0
        all_petsc_rows = comm.gather(local_rows, root=0)
        all_petsc_cols = comm.gather(local_cols, root=0)
        all_petsc_vals = comm.gather(local_vals, root=0)
    else:
        all_petsc_rows = None
        all_petsc_cols = None
        all_petsc_vals = None

    if rank == 0:
        # Concatenate native COO data
        all_rows = np.concatenate(all_coo_rows)
        all_cols = np.concatenate(all_coo_cols)
        all_vals = np.concatenate(all_coo_vals)

        print(f"  Total COO entries (all ranks): {len(all_vals)}")

        # Build scipy sparse matrix and sum duplicates
        from scipy.sparse import coo_matrix
        A_native = coo_matrix((all_vals, (all_rows, all_cols)), shape=(global_size, global_size))
        A_native = A_native.tocsr()  # This sums duplicates

        print(f"  Native matrix nnz (after summing duplicates): {A_native.nnz}")

        # Build PETSc matrix as scipy sparse
        if comm.size > 1:
            petsc_rows = np.concatenate(all_petsc_rows)
            petsc_cols = np.concatenate(all_petsc_cols)
            petsc_vals = np.concatenate(all_petsc_vals)
        else:
            # Single rank - extract directly
            petsc_rows = []
            petsc_cols = []
            petsc_vals = []
            for i in range(m_global):
                cols, vals = A_petsc.getRow(i)
                for c, v in zip(cols, vals):
                    petsc_rows.append(i)
                    petsc_cols.append(c)
                    petsc_vals.append(v)
            petsc_rows = np.array(petsc_rows)
            petsc_cols = np.array(petsc_cols)
            petsc_vals = np.array(petsc_vals)

        A_petsc_scipy = coo_matrix((petsc_vals, (petsc_rows, petsc_cols)), shape=(m_global, n_global))
        A_petsc_scipy = A_petsc_scipy.tocsr()

        A_native_dense = A_native.toarray()
        A_petsc_dense = A_petsc_scipy.toarray()

        # Compare with relative tolerance
        passed, max_rel_diff, max_abs_diff, diff_indices = compare_matrices(
            A_petsc_dense, A_native_dense, rtol=1e-12, atol=1e-14
        )
        print(f"  Max absolute diff: {max_abs_diff:.2e}, max relative diff: {max_rel_diff:.2e}")

        if passed:
            print("  PASS: Matrices match!")
            return True
        else:
            print("  FAIL: Matrices differ!")
            print(f"  Differences at {len(diff_indices[0])} entries")
            for idx in range(min(5, len(diff_indices[0]))):
                i, j = diff_indices[0][idx], diff_indices[1][idx]
                print(f"    ({i}, {j}): PETSc={A_petsc_dense[i,j]:.6e}, Native={A_native_dense[i,j]:.6e}")
            return False

    return True


def test_two_blocks():
    """Test with a 2x2 block structure using the same restriction (tests block offset logic)."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n" + "=" * 60)
        print("Test 2: Two blocks (2x2 block matrix, same restriction)")
        print("=" * 60)

    # Create a simple mesh
    mesh = dfx.mesh.create_unit_square(comm, 4, 4)
    V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

    # Use interior DOFs for both blocks (same restriction)
    # This tests the block offset logic in a parallel-safe way
    boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    all_dofs = np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)
    interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)
    interior_dofs = np.sort(interior_dofs)

    if rank == 0:
        print(f"  Total DOFs: {len(all_dofs)}")
        print(f"  Interior DOFs (per block): {len(interior_dofs)}")

    # Create restrictions - same for both blocks
    restriction_0 = multiphenicsx.fem.DofMapRestriction(V.dofmap, interior_dofs)
    restriction_1 = multiphenicsx.fem.DofMapRestriction(V.dofmap, interior_dofs)
    restrictions = [restriction_0, restriction_1]

    # Create forms
    u = TrialFunction(V)
    v = TestFunction(V)

    # Block structure:
    # [ a00   0  ]   where a_ij = inner(grad(u), grad(v)) * dx
    # [  0   a11 ]
    # Off-diagonal blocks are None (realistic for decoupled systems)
    a00 = inner(grad(u), grad(v)) * dx
    a11 = inner(grad(u), grad(v)) * dx

    forms = [
        [dfx.fem.form(a00), None],
        [None, dfx.fem.form(a11)]
    ]

    # Assemble with PETSc via multiphenicsx
    A_petsc = multiphenicsx.fem.petsc.assemble_matrix_block(
        forms, bcs=[], restriction=(restrictions, restrictions)
    )
    A_petsc.assemble()

    # Assemble with native assembly
    coo_rows, coo_cols, coo_vals, global_size, row_ranges = assemble_block_to_coo(
        forms, restrictions, [], comm
    )

    if rank == 0:
        print(f"  PETSc matrix size: {A_petsc.getSize()}")
        print(f"  Native global size: {global_size}")
        print(f"  Native COO entries: {len(coo_vals)}")

    # Gather COO data to rank 0
    all_coo_rows = comm.gather(coo_rows, root=0)
    all_coo_cols = comm.gather(coo_cols, root=0)
    all_coo_vals = comm.gather(coo_vals, root=0)

    # Gather PETSc matrix data from all ranks
    m_global, n_global = A_petsc.getSize()
    row_start, row_end = A_petsc.getOwnershipRange()
    local_rows = []
    local_cols = []
    local_vals = []
    for i in range(row_start, row_end):
        cols, vals = A_petsc.getRow(i)
        for c, v in zip(cols, vals):
            local_rows.append(i)
            local_cols.append(c)
            local_vals.append(v)

    all_petsc_rows = comm.gather(local_rows, root=0)
    all_petsc_cols = comm.gather(local_cols, root=0)
    all_petsc_vals = comm.gather(local_vals, root=0)

    if rank == 0:
        all_rows = np.concatenate(all_coo_rows)
        all_cols = np.concatenate(all_coo_cols)
        all_vals = np.concatenate(all_coo_vals)

        from scipy.sparse import coo_matrix
        A_native = coo_matrix((all_vals, (all_rows, all_cols)), shape=(global_size, global_size))
        A_native = A_native.tocsr()

        print(f"  Native matrix nnz: {A_native.nnz}")

        # Build PETSc matrix as scipy sparse
        petsc_rows = np.concatenate(all_petsc_rows)
        petsc_cols = np.concatenate(all_petsc_cols)
        petsc_vals = np.concatenate(all_petsc_vals)
        A_petsc_scipy = coo_matrix((petsc_vals, (petsc_rows, petsc_cols)), shape=(m_global, n_global))
        A_petsc_scipy = A_petsc_scipy.tocsr()

        A_native_dense = A_native.toarray()
        A_petsc_dense = A_petsc_scipy.toarray()

        # Compare with relative tolerance
        passed, max_rel_diff, max_abs_diff, diff_indices = compare_matrices(
            A_petsc_dense, A_native_dense, rtol=1e-12, atol=1e-14
        )
        print(f"  Max absolute diff: {max_abs_diff:.2e}, max relative diff: {max_rel_diff:.2e}")

        if passed:
            print("  PASS: Matrices match!")
            return True
        else:
            print("  FAIL: Matrices differ!")
            print(f"  Differences at {len(diff_indices[0])} entries")
            for idx in range(min(5, len(diff_indices[0]))):
                i, j = diff_indices[0][idx], diff_indices[1][idx]
                print(f"    ({i}, {j}): PETSc={A_petsc_dense[i,j]:.6e}, Native={A_native_dense[i,j]:.6e}")

            # Print row 0 nonzeros to understand the pattern
            print(f"  PETSc row 0 nonzeros:")
            for j in range(A_petsc_dense.shape[1]):
                if abs(A_petsc_dense[0, j]) > 1e-14:
                    print(f"    col {j}: {A_petsc_dense[0,j]:.2e}")
            print(f"  Native row 0 nonzeros:")
            for j in range(A_native_dense.shape[1]):
                if abs(A_native_dense[0, j]) > 1e-14:
                    print(f"    col {j}: {A_native_dense[0,j]:.2e}")
            return False

    return True


def test_coupled_blocks():
    """Test with a coupled 2x2 block structure (Laplacian diagonal, mass off-diagonal)."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n" + "=" * 60)
        print("Test 3: Coupled blocks (Laplacian diagonal, mass off-diagonal)")
        print("=" * 60)

    # Create a simple mesh
    mesh = dfx.mesh.create_unit_square(comm, 4, 4)
    V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

    # Use interior DOFs for both blocks (same restriction)
    boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    all_dofs = np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)
    interior_dofs = np.setdiff1d(all_dofs, boundary_dofs).astype(np.int32)
    interior_dofs = np.sort(interior_dofs)

    if rank == 0:
        print(f"  Total DOFs: {len(all_dofs)}")
        print(f"  Interior DOFs (per block): {len(interior_dofs)}")

    # Create restrictions - same for both blocks
    restriction_0 = multiphenicsx.fem.DofMapRestriction(V.dofmap, interior_dofs)
    restriction_1 = multiphenicsx.fem.DofMapRestriction(V.dofmap, interior_dofs)
    restrictions = [restriction_0, restriction_1]

    # Create forms
    u = TrialFunction(V)
    v = TestFunction(V)

    # Coupled block structure:
    # [ Laplacian   Mass     ]
    # [ Mass        Laplacian ]
    laplacian = inner(grad(u), grad(v)) * dx
    mass = inner(u, v) * dx

    forms = [
        [dfx.fem.form(laplacian), dfx.fem.form(mass)],
        [dfx.fem.form(mass), dfx.fem.form(laplacian)]
    ]

    # Assemble with PETSc via multiphenicsx
    A_petsc = multiphenicsx.fem.petsc.assemble_matrix_block(
        forms, bcs=[], restriction=(restrictions, restrictions)
    )
    A_petsc.assemble()

    # Assemble with native assembly
    coo_rows, coo_cols, coo_vals, global_size, row_ranges = assemble_block_to_coo(
        forms, restrictions, [], comm
    )

    if rank == 0:
        print(f"  PETSc matrix size: {A_petsc.getSize()}")
        print(f"  Native global size: {global_size}")
        print(f"  Native COO entries: {len(coo_vals)}")

    # Gather COO data to rank 0
    all_coo_rows = comm.gather(coo_rows, root=0)
    all_coo_cols = comm.gather(coo_cols, root=0)
    all_coo_vals = comm.gather(coo_vals, root=0)

    # Gather PETSc matrix data from all ranks
    m_global, n_global = A_petsc.getSize()
    row_start, row_end = A_petsc.getOwnershipRange()
    local_rows = []
    local_cols = []
    local_vals = []
    for i in range(row_start, row_end):
        cols, vals = A_petsc.getRow(i)
        for c, v in zip(cols, vals):
            local_rows.append(i)
            local_cols.append(c)
            local_vals.append(v)

    all_petsc_rows = comm.gather(local_rows, root=0)
    all_petsc_cols = comm.gather(local_cols, root=0)
    all_petsc_vals = comm.gather(local_vals, root=0)

    if rank == 0:
        all_rows = np.concatenate(all_coo_rows)
        all_cols = np.concatenate(all_coo_cols)
        all_vals = np.concatenate(all_coo_vals)

        from scipy.sparse import coo_matrix
        A_native = coo_matrix((all_vals, (all_rows, all_cols)), shape=(global_size, global_size))
        A_native = A_native.tocsr()

        print(f"  Native matrix nnz: {A_native.nnz}")

        # Build PETSc matrix as scipy sparse
        petsc_rows = np.concatenate(all_petsc_rows)
        petsc_cols = np.concatenate(all_petsc_cols)
        petsc_vals = np.concatenate(all_petsc_vals)
        A_petsc_scipy = coo_matrix((petsc_vals, (petsc_rows, petsc_cols)), shape=(m_global, n_global))
        A_petsc_scipy = A_petsc_scipy.tocsr()

        A_native_dense = A_native.toarray()
        A_petsc_dense = A_petsc_scipy.toarray()

        # Compare with relative tolerance
        passed, max_rel_diff, max_abs_diff, diff_indices = compare_matrices(
            A_petsc_dense, A_native_dense, rtol=1e-12, atol=1e-14
        )
        print(f"  Max absolute diff: {max_abs_diff:.2e}, max relative diff: {max_rel_diff:.2e}")

        if passed:
            print("  PASS: Matrices match!")
            return True
        else:
            print("  FAIL: Matrices differ!")
            print(f"  Differences at {len(diff_indices[0])} entries")
            for idx in range(min(5, len(diff_indices[0]))):
                i, j = diff_indices[0][idx], diff_indices[1][idx]
                print(f"    ({i}, {j}): PETSc={A_petsc_dense[i,j]:.6e}, Native={A_native_dense[i,j]:.6e}")
            return False

    return True


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(f"Running on {comm.size} MPI rank(s)")
        print()

    passed = True
    passed &= test_single_block()
    passed &= test_two_blocks()
    passed &= test_coupled_blocks()

    if comm.rank == 0:
        print("\n" + "=" * 60)
        if passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("=" * 60)
