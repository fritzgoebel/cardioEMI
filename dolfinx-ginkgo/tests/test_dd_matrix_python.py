#!/usr/bin/env python3
"""
Test DdMatrix Python bindings.

Run with: mpirun -n 2 python test_dd_matrix_python.py
"""

import numpy as np
from mpi4py import MPI

# Add build and python directories to path
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, "..", "build")
python_dir = os.path.join(script_dir, "..", "python")
# Insert build dir first so compiled _cpp module takes precedence
sys.path.insert(0, build_dir)
sys.path.insert(1, python_dir)

from dolfinx_ginkgo import _cpp


def test_dd_matrix_creation():
    """Test DdMatrix creation from local COO data."""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        print("=== Test: DdMatrix Python Bindings ===")
        print(f"MPI processes: {size}")

    if size != 2:
        if rank == 0:
            print("Skipping (requires exactly 2 ranks)")
        return

    # Create executor and communicator
    exec_ = _cpp.create_executor(_cpp.Backend.OMP, 0)
    gko_comm = _cpp.create_communicator(comm)

    if rank == 0:
        print("\n--- Test 1: DdMatrix Creation ---")

    # Use the documentation example: 3x3 matrix
    global_size = 3
    row_ranges = np.array([0, 2, 3], dtype=np.int64)

    if rank == 0:
        # Rank 0's local contribution:
        # |  4 -2  0 |
        # | -2  2  0 |
        # |  0  0  0 |
        row_indices = np.array([0, 0, 1, 1], dtype=np.int64)
        col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
        values = np.array([4.0, -2.0, -2.0, 2.0], dtype=np.float64)
    else:
        # Rank 1's local contribution:
        # |  0  0  0 |
        # |  0  2 -2 |
        # |  0 -2  4 |
        row_indices = np.array([1, 1, 2, 2], dtype=np.int64)
        col_indices = np.array([1, 2, 1, 2], dtype=np.int64)
        values = np.array([2.0, -2.0, -2.0, 4.0], dtype=np.float64)

    # Create DdMatrix
    dd_mat = _cpp.create_dd_matrix_from_local_coo(
        exec_, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size, row_ranges
    )

    assert dd_mat is not None
    print(f"  Rank {rank}: DdMatrix created [OK]")

    if rank == 0:
        print("\n--- Test 2: Compare with Regular Matrix ---")

    # Create regular distributed matrix with same data
    regular_mat = _cpp.create_distributed_matrix_from_local_coo(
        exec_, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size, row_ranges
    )

    assert regular_mat is not None
    print(f"  Rank {rank}: Regular matrix created [OK]")

    comm.Barrier()
    if rank == 0:
        print("\n=== All Python tests passed! ===")


def test_ginkgo_solver_dd():
    """Test GinkgoSolver with DdMatrix."""
    from dolfinx_ginkgo import GinkgoSolver

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if size != 2:
        return

    if rank == 0:
        print("\n--- Test 3: GinkgoSolver with DdMatrix ---")

    # Same 3x3 example
    global_size = 3
    row_ranges = np.array([0, 2, 3], dtype=np.int64)

    if rank == 0:
        row_indices = np.array([0, 0, 1, 1], dtype=np.int64)
        col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
        values = np.array([4.0, -2.0, -2.0, 2.0], dtype=np.float64)
    else:
        row_indices = np.array([1, 1, 2, 2], dtype=np.int64)
        col_indices = np.array([1, 2, 1, 2], dtype=np.int64)
        values = np.array([2.0, -2.0, -2.0, 4.0], dtype=np.float64)

    # Create solver (no preconditioner for DdMatrix compatibility test)
    solver = GinkgoSolver(
        comm=comm,
        backend="omp",
        solver="cg",
        preconditioner="none",
        verbose=False,
    )

    # Set DdMatrix operator
    solver.set_operator_dd_from_local_coo(
        row_indices, col_indices, values,
        global_size, global_size, row_ranges
    )

    print(f"  Rank {rank}: GinkgoSolver with DdMatrix created [OK]")

    comm.Barrier()
    if rank == 0:
        print("\n=== GinkgoSolver DdMatrix test passed! ===")


def test_ginkgo_solver_bddc():
    """Test GinkgoSolver with BDDC preconditioner."""
    from dolfinx_ginkgo import GinkgoSolver

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if size != 2:
        return

    if rank == 0:
        print("\n--- Test 4: GinkgoSolver with BDDC Preconditioner ---")

    # Same 3x3 example
    global_size = 3
    row_ranges = np.array([0, 2, 3], dtype=np.int64)

    if rank == 0:
        row_indices = np.array([0, 0, 1, 1], dtype=np.int64)
        col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
        values = np.array([4.0, -2.0, -2.0, 2.0], dtype=np.float64)
    else:
        row_indices = np.array([1, 1, 2, 2], dtype=np.int64)
        col_indices = np.array([1, 2, 1, 2], dtype=np.int64)
        values = np.array([2.0, -2.0, -2.0, 4.0], dtype=np.float64)

    # Create solver with BDDC preconditioner
    solver = GinkgoSolver(
        comm=comm,
        backend="omp",
        solver="cg",
        preconditioner="bddc",
        bddc_config={
            "vertices": True,
            "edges": True,
            "faces": True,
            "local_solver": "direct",
            "coarse_solver": "cg",
            "coarse_max_iterations": 100,
        },
        verbose=True,
    )

    # Set DdMatrix operator
    solver.set_operator_dd_from_local_coo(
        row_indices, col_indices, values,
        global_size, global_size, row_ranges
    )

    print(f"  Rank {rank}: GinkgoSolver with BDDC created [OK]")

    comm.Barrier()
    if rank == 0:
        print("\n=== BDDC preconditioner test passed! ===")


if __name__ == "__main__":
    test_dd_matrix_creation()
    test_ginkgo_solver_dd()
    test_ginkgo_solver_bddc()
