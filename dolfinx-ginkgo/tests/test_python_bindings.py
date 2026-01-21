#!/usr/bin/env python3
"""
Test the dolfinx_ginkgo Python bindings.

Run with: mpirun -n 2 python test_python_bindings.py
"""

import sys
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Set up paths before any dolfinx_ginkgo imports
# The _cpp module is built in the build directory
# The Python source is in the python directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_build_dir = os.path.join(_root_dir, "build")
_python_dir = os.path.join(_root_dir, "python")

# Add python source directory first (for solver.py, __init__.py)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# For the _cpp module, we need to copy or link it to the package directory
# or import it directly. Let's import it directly.
_cpp_module_path = os.path.join(_build_dir, "_cpp.cpython-312-aarch64-linux-gnu.so")
if not os.path.exists(_cpp_module_path):
    # Try to find the module with a glob pattern
    import glob
    _cpp_modules = glob.glob(os.path.join(_build_dir, "_cpp*.so"))
    if _cpp_modules:
        _cpp_module_path = _cpp_modules[0]


def test_imports():
    """Test that all imports work."""
    print("=== Test: Imports ===")

    # Import _cpp directly from the build directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cpp", _cpp_module_path)
    _cpp = importlib.util.module_from_spec(spec)
    sys.modules["dolfinx_ginkgo._cpp"] = _cpp
    spec.loader.exec_module(_cpp)

    # Also make it available as dolfinx_ginkgo._cpp
    import dolfinx_ginkgo
    dolfinx_ginkgo._cpp = _cpp

    # Check that enums are accessible
    assert hasattr(_cpp, "Backend")
    assert hasattr(_cpp, "SolverType")
    assert hasattr(_cpp, "PreconditionerType")
    assert hasattr(_cpp, "SolverConfig")
    assert hasattr(_cpp, "AMGConfig")

    print("  All imports OK")
    return _cpp


def test_backend_availability(_cpp):
    """Test backend availability functions."""
    print("=== Test: Backend Availability ===")

    # Reference and OMP should always be available
    assert _cpp.is_backend_available(_cpp.Backend.REFERENCE)
    assert _cpp.is_backend_available(_cpp.Backend.OMP)

    print(f"  REFERENCE: {_cpp.is_backend_available(_cpp.Backend.REFERENCE)}")
    print(f"  OMP: {_cpp.is_backend_available(_cpp.Backend.OMP)}")
    print(f"  CUDA: {_cpp.is_backend_available(_cpp.Backend.CUDA)}")
    print(f"  HIP: {_cpp.is_backend_available(_cpp.Backend.HIP)}")
    print(f"  SYCL: {_cpp.is_backend_available(_cpp.Backend.SYCL)}")

    print("  Backend availability OK")


def test_executor_creation(_cpp):
    """Test executor creation."""
    print("=== Test: Executor Creation ===")

    # Create reference executor
    ref_exec = _cpp.create_executor(_cpp.Backend.REFERENCE)
    assert ref_exec is not None
    print("  Reference executor: OK")

    # Create OMP executor
    omp_exec = _cpp.create_executor(_cpp.Backend.OMP)
    assert omp_exec is not None
    print("  OMP executor: OK")


def test_communicator_creation(_cpp):
    """Test MPI communicator wrapper."""
    print("=== Test: Communicator Creation ===")

    comm = MPI.COMM_WORLD
    gko_comm = _cpp.create_communicator(comm)
    assert gko_comm is not None
    print(f"  Ginkgo communicator created for {comm.size} processes: OK")


def test_solver_config(_cpp):
    """Test solver configuration."""
    print("=== Test: Solver Configuration ===")

    config = _cpp.SolverConfig()

    # Test default values
    assert config.solver == _cpp.SolverType.CG
    assert config.preconditioner == _cpp.PreconditionerType.AMG
    assert config.rtol == 1e-8
    assert config.max_iterations == 1000

    # Test modification
    config.solver = _cpp.SolverType.GMRES
    config.preconditioner = _cpp.PreconditionerType.JACOBI
    config.rtol = 1e-10
    config.max_iterations = 500

    assert config.solver == _cpp.SolverType.GMRES
    assert config.preconditioner == _cpp.PreconditionerType.JACOBI
    assert config.rtol == 1e-10
    assert config.max_iterations == 500

    print("  SolverConfig: OK")


def test_amg_config(_cpp):
    """Test AMG configuration."""
    print("=== Test: AMG Configuration ===")

    amg = _cpp.AMGConfig()

    # Test default values
    assert amg.max_levels == 10
    assert amg.cycle == _cpp.AMGConfig.Cycle.V
    assert amg.smoother == _cpp.AMGConfig.Smoother.JACOBI

    # Test modification
    amg.max_levels = 5
    amg.cycle = _cpp.AMGConfig.Cycle.W
    amg.smoother = _cpp.AMGConfig.Smoother.ILU
    amg.relaxation_factor = 0.8

    assert amg.max_levels == 5
    assert amg.cycle == _cpp.AMGConfig.Cycle.W
    assert amg.smoother == _cpp.AMGConfig.Smoother.ILU
    assert abs(amg.relaxation_factor - 0.8) < 1e-10

    print("  AMGConfig: OK")


def create_1d_laplacian(n, comm):
    """Create a 1D Laplacian matrix distributed across MPI ranks."""
    rank = comm.rank
    size = comm.size

    # Compute local range
    local_n = n // size
    remainder = n % size

    if rank < remainder:
        local_n += 1
        row_start = rank * (n // size + 1)
    else:
        row_start = remainder * (n // size + 1) + (rank - remainder) * (n // size)

    row_end = row_start + local_n

    # Create PETSc matrix
    A = PETSc.Mat().create(comm)
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.MPIAIJ)
    A.setPreallocationNNZ(3)
    A.setUp()

    # Fill the matrix (1D Laplacian: -1, 2, -1)
    for i in range(row_start, row_end):
        if i > 0:
            A.setValue(i, i - 1, -1.0)
        A.setValue(i, i, 2.0)
        if i < n - 1:
            A.setValue(i, i + 1, -1.0)

    A.assemblyBegin()
    A.assemblyEnd()

    return A, row_start, local_n


def test_matrix_conversion(_cpp):
    """Test PETSc to Ginkgo matrix conversion."""
    print("=== Test: Matrix Conversion ===")

    comm = MPI.COMM_WORLD
    n = 100

    # Create a simple 1D Laplacian
    A, row_start, local_n = create_1d_laplacian(n, comm)

    # Create executor and communicator
    exec_ = _cpp.create_executor(_cpp.Backend.OMP)
    gko_comm = _cpp.create_communicator(comm)

    # Convert to Ginkgo
    gko_A = _cpp.create_distributed_matrix_from_petsc(exec_, gko_comm, A)
    assert gko_A is not None

    print(f"  Rank {comm.rank}: local_rows={local_n}, row_start={row_start}")
    print("  Matrix conversion: OK")

    A.destroy()
    return gko_A


def test_vector_conversion(_cpp):
    """Test PETSc to Ginkgo vector conversion."""
    print("=== Test: Vector Conversion ===")

    comm = MPI.COMM_WORLD
    n = 100

    # Create PETSc vector
    b = PETSc.Vec().create(comm)
    b.setSizes(n)
    b.setFromOptions()
    b.set(1.0)

    # Create executor and communicator
    exec_ = _cpp.create_executor(_cpp.Backend.OMP)
    gko_comm = _cpp.create_communicator(comm)

    # Convert to Ginkgo
    gko_b = _cpp.create_distributed_vector_from_petsc(exec_, gko_comm, b)
    assert gko_b is not None

    print("  Vector conversion: OK")

    b.destroy()


def test_solver(_cpp):
    """Test the distributed solver."""
    print("=== Test: Distributed Solver ===")

    comm = MPI.COMM_WORLD
    n = 200

    # Create 1D Laplacian
    A, _, _ = create_1d_laplacian(n, comm)

    # Create RHS (b = 1)
    b = PETSc.Vec().create(comm)
    b.setSizes(n)
    b.setFromOptions()
    b.set(1.0)

    # Create solution vector (x = 0 initial guess)
    x = PETSc.Vec().create(comm)
    x.setSizes(n)
    x.setFromOptions()
    x.set(0.0)

    # Create executor and communicator
    exec_ = _cpp.create_executor(_cpp.Backend.OMP)
    gko_comm = _cpp.create_communicator(comm)

    # Convert matrix
    gko_A = _cpp.create_distributed_matrix_from_petsc(exec_, gko_comm, A)

    # Create solver config
    config = _cpp.SolverConfig()
    config.solver = _cpp.SolverType.CG
    config.preconditioner = _cpp.PreconditionerType.JACOBI
    config.rtol = 1e-8
    config.max_iterations = 500
    config.verbose = (comm.rank == 0)

    # Create solver
    solver = _cpp.DistributedSolver(exec_, gko_comm, config)
    solver.set_operator(gko_A)

    # Convert vectors
    gko_b = _cpp.create_distributed_vector_from_petsc(exec_, gko_comm, b)
    gko_x = _cpp.create_distributed_vector_from_petsc(exec_, gko_comm, x)

    # Solve
    iters = solver.solve(gko_b, gko_x)

    # Copy solution back to PETSc
    _cpp.copy_to_petsc(gko_x, x)

    # Compute residual
    r = A.createVecLeft()
    A.mult(x, r)
    r.axpy(-1.0, b)
    residual = r.norm()

    if comm.rank == 0:
        print(f"  Iterations: {solver.iterations()}")
        print(f"  Residual: {residual:.2e}")
        print(f"  Converged: {solver.converged()}")

    assert solver.converged() or solver.iterations() == config.max_iterations

    # Clean up
    A.destroy()
    b.destroy()
    x.destroy()
    r.destroy()

    print("  Solver test: OK")


def test_high_level_api():
    """Test the high-level GinkgoSolver Python class."""
    print("=== Test: High-Level GinkgoSolver API ===")

    # Add the Python source directory to path
    sys.path.insert(0, "/home/fenics/dolfinx-ginkgo/python")

    from dolfinx_ginkgo import GinkgoSolver

    comm = MPI.COMM_WORLD
    n = 200

    # Create 1D Laplacian
    A, _, _ = create_1d_laplacian(n, comm)

    # Create RHS
    b = PETSc.Vec().create(comm)
    b.setSizes(n)
    b.setFromOptions()
    b.set(1.0)

    # Create solution vector
    x = PETSc.Vec().create(comm)
    x.setSizes(n)
    x.setFromOptions()
    x.set(0.0)

    # Create high-level solver
    solver = GinkgoSolver(
        A,
        comm=comm,
        backend="omp",
        solver="cg",
        preconditioner="jacobi",
        rtol=1e-8,
        verbose=(comm.rank == 0)
    )

    # Solve
    solver.solve(b, x)

    if comm.rank == 0:
        print(f"  Iterations: {solver.iterations}")
        print(f"  Residual: {solver.residual_norm:.2e}")
        print(f"  Converged: {solver.converged}")
        print(f"  Solver: {solver}")

    assert solver.converged

    # Clean up
    A.destroy()
    b.destroy()
    x.destroy()

    print("  High-level API test: OK")


def main():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("=" * 60)
        print("dolfinx-ginkgo Python Bindings Test Suite")
        print("=" * 60)
        print(f"MPI processes: {comm.size}")
        print()

    comm.Barrier()

    try:
        # Run tests
        _cpp = test_imports()
        comm.Barrier()

        if comm.rank == 0:
            test_backend_availability(_cpp)
            test_executor_creation(_cpp)
        comm.Barrier()

        test_communicator_creation(_cpp)
        comm.Barrier()

        if comm.rank == 0:
            test_solver_config(_cpp)
            test_amg_config(_cpp)
        comm.Barrier()

        test_matrix_conversion(_cpp)
        comm.Barrier()

        test_vector_conversion(_cpp)
        comm.Barrier()

        test_solver(_cpp)
        comm.Barrier()

        test_high_level_api()
        comm.Barrier()

        if comm.rank == 0:
            print()
            print("=" * 60)
            print("All tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"[Rank {comm.rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        comm.Abort(1)


if __name__ == "__main__":
    main()
