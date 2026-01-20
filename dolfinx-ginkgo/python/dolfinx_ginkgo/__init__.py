"""
dolfinx-ginkgo: Ginkgo linear solver backend for DOLFINx

This package provides GPU-accelerated distributed linear solvers for DOLFINx
using the Ginkgo library. It supports CUDA, HIP, and SYCL backends.

Example usage:

    from dolfinx_ginkgo import GinkgoSolver

    # Create solver with AMG preconditioner on GPU
    solver = GinkgoSolver(
        A,
        comm=comm,
        backend="cuda",
        solver="cg",
        preconditioner="amg",
        rtol=1e-8
    )

    # Solve Ax = b
    solver.solve(b, x)
"""

from dolfinx_ginkgo.solver import GinkgoSolver
from dolfinx_ginkgo._cpp import (
    Backend,
    SolverType,
    PreconditionerType,
    SolverConfig,
    AMGConfig,
    create_executor,
    create_communicator,
    is_backend_available,
    get_device_count,
)

__version__ = "0.1.0"

__all__ = [
    "GinkgoSolver",
    "Backend",
    "SolverType",
    "PreconditionerType",
    "SolverConfig",
    "AMGConfig",
    "create_executor",
    "create_communicator",
    "is_backend_available",
    "get_device_count",
]
