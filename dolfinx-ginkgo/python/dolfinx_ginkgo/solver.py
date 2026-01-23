"""
High-level Python interface for Ginkgo-based distributed solvers.

This module provides a user-friendly interface for solving linear systems
with Ginkgo's GPU-accelerated solvers, integrated with DOLFINx/PETSc.
"""

from __future__ import annotations

from typing import Optional, Literal, Any
from mpi4py import MPI


class GinkgoSolver:
    """
    Ginkgo-based distributed linear solver for DOLFINx/PETSc systems.

    This solver wraps Ginkgo's distributed Krylov solvers and preconditioners,
    enabling GPU-accelerated linear algebra with CUDA, HIP, or SYCL backends.

    Parameters
    ----------
    A : PETSc.Mat
        System matrix (distributed MPIAIJ or sequential SEQAIJ)
    comm : MPI.Comm, optional
        MPI communicator. Default: MPI.COMM_WORLD
    backend : {"cuda", "hip", "omp", "reference"}, optional
        Compute backend. Default: "cuda"
    device_id : int, optional
        GPU device ID (for CUDA/HIP backends). Default: 0
    solver : {"cg", "fcg", "gmres", "bicgstab", "cgs"}, optional
        Krylov solver type. Default: "cg"
    preconditioner : {"none", "jacobi", "block_jacobi", "ilu", "ic", "isai", "amg"}, optional
        Preconditioner type. Default: "amg"
    rtol : float, optional
        Relative tolerance. Default: 1e-8
    atol : float, optional
        Absolute tolerance. Default: 1e-12
    max_iter : int, optional
        Maximum iterations. Default: 1000
    krylov_dim : int, optional
        Krylov dimension (GMRES only). Default: 30
    jacobi_block_size : int, optional
        Block size for block Jacobi preconditioner. Default: 32
    amg_config : dict, optional
        AMG-specific configuration. See AMGConfig for options.
    verbose : bool, optional
        Print convergence info. Default: False

    Examples
    --------
    Basic usage with CG and AMG:

    >>> solver = GinkgoSolver(A, backend="cuda", solver="cg", preconditioner="amg")
    >>> solver.solve(b, x)
    >>> print(f"Converged in {solver.iterations} iterations")

    With custom AMG configuration:

    >>> solver = GinkgoSolver(
    ...     A,
    ...     backend="cuda",
    ...     preconditioner="amg",
    ...     amg_config={
    ...         "max_levels": 10,
    ...         "cycle": "v",
    ...         "smoother": "jacobi",
    ...         "coarse_solver": "direct"
    ...     }
    ... )

    GMRES for non-symmetric systems:

    >>> solver = GinkgoSolver(
    ...     A,
    ...     solver="gmres",
    ...     preconditioner="ilu",
    ...     krylov_dim=50
    ... )
    """

    # Backend name to enum mapping
    _backend_map = {
        "reference": "REFERENCE",
        "ref": "REFERENCE",
        "omp": "OMP",
        "openmp": "OMP",
        "cuda": "CUDA",
        "hip": "HIP",
        "rocm": "HIP",
        "sycl": "SYCL",
        "dpcpp": "SYCL",
    }

    # Solver name to enum mapping
    _solver_map = {
        "cg": "CG",
        "fcg": "FCG",
        "gmres": "GMRES",
        "bicgstab": "BICGSTAB",
        "cgs": "CGS",
    }

    # Preconditioner name to enum mapping
    _precond_map = {
        "none": "NONE",
        "jacobi": "JACOBI",
        "block_jacobi": "BLOCK_JACOBI",
        "ilu": "ILU",
        "ic": "IC",
        "isai": "ISAI",
        "amg": "AMG",
    }

    def __init__(
        self,
        A: Any = None,  # PETSc.Mat (optional for backward compatibility)
        comm: MPI.Comm = MPI.COMM_WORLD,
        backend: Literal["reference", "omp", "cuda", "hip", "sycl"] = "omp",
        device_id: int = 0,
        solver: Literal["cg", "fcg", "gmres", "bicgstab", "cgs"] = "cg",
        preconditioner: Literal["none", "jacobi", "block_jacobi", "ilu", "ic", "isai", "amg"] = "jacobi",
        rtol: float = 1e-8,
        atol: float = 1e-12,
        max_iter: int = 1000,
        krylov_dim: int = 30,
        jacobi_block_size: int = 32,
        amg_config: Optional[dict] = None,
        verbose: bool = False,
    ):
        # Import C++ bindings
        from dolfinx_ginkgo import _cpp

        # Validate backend
        backend_lower = backend.lower()
        if backend_lower not in self._backend_map:
            raise ValueError(f"Unknown backend: {backend}. "
                           f"Available: {list(self._backend_map.keys())}")

        backend_enum = getattr(_cpp.Backend, self._backend_map[backend_lower])

        # Check backend availability
        if not _cpp.is_backend_available(backend_enum):
            raise RuntimeError(
                f"Backend '{backend}' is not available. "
                f"Either it was not compiled or no devices are present."
            )

        # Validate solver
        solver_lower = solver.lower()
        if solver_lower not in self._solver_map:
            raise ValueError(f"Unknown solver: {solver}. "
                           f"Available: {list(self._solver_map.keys())}")

        solver_enum = getattr(_cpp.SolverType, self._solver_map[solver_lower])

        # Validate preconditioner
        precond_lower = preconditioner.lower()
        if precond_lower not in self._precond_map:
            raise ValueError(f"Unknown preconditioner: {preconditioner}. "
                           f"Available: {list(self._precond_map.keys())}")

        precond_enum = getattr(_cpp.PreconditionerType, self._precond_map[precond_lower])

        # Create executor
        self._exec = _cpp.create_executor(backend_enum, device_id)

        # Create Ginkgo MPI communicator
        self._gko_comm = _cpp.create_communicator(comm)

        # Build solver config
        config = _cpp.SolverConfig()
        config.solver = solver_enum
        config.preconditioner = precond_enum
        config.rtol = rtol
        config.atol = atol
        config.max_iterations = max_iter
        config.krylov_dim = krylov_dim
        config.jacobi_block_size = jacobi_block_size
        config.verbose = verbose

        # Configure AMG if selected
        if preconditioner.lower() == "amg" and amg_config is not None:
            self._configure_amg(config.amg, amg_config, _cpp)

        # Store references
        self._config = config
        self._comm = comm
        self._cpp = _cpp
        self._A_gko = None
        self._A_petsc = None
        self._solver = None
        self._row_ranges = None

        # If matrix provided, set it (backward compatibility)
        if A is not None:
            self.set_operator_from_petsc(A)

    def set_operator_from_petsc(self, A: Any) -> None:
        """
        Set the system matrix from a PETSc Mat.

        Parameters
        ----------
        A : PETSc.Mat
            System matrix (distributed MPIAIJ or sequential SEQAIJ)
        """
        _cpp = self._cpp

        self._A_gko = _cpp.create_distributed_matrix_from_petsc(
            self._exec, self._gko_comm, A
        )
        self._A_petsc = A

        if self._solver is None:
            self._solver = _cpp.DistributedSolver(self._exec, self._gko_comm, self._config)
        self._solver.set_operator(self._A_gko)

    def set_operator_from_local_coo(
        self,
        row_indices,  # np.ndarray[int64]
        col_indices,  # np.ndarray[int64]
        values,       # np.ndarray[float64]
        global_rows: int,
        global_cols: int,
        row_ranges,   # np.ndarray[int64]
    ) -> None:
        """
        Set the system matrix from local COO data in global numbering.

        Each rank provides its local contributions. Entries for the same
        (row, col) from different ranks are automatically summed by Ginkgo.

        Parameters
        ----------
        row_indices : ndarray[int64]
            Global row indices of non-zeros
        col_indices : ndarray[int64]
            Global column indices of non-zeros
        values : ndarray[float64]
            Non-zero values
        global_rows : int
            Total number of rows in the global matrix
        global_cols : int
            Total number of columns in the global matrix
        row_ranges : ndarray[int64]
            Partition ranges [r0, r1, ..., r_nprocs] defining row ownership.
            Rank i owns rows [row_ranges[i], row_ranges[i+1]).
        """
        import numpy as np
        _cpp = self._cpp

        row_indices = np.ascontiguousarray(row_indices, dtype=np.int64)
        col_indices = np.ascontiguousarray(col_indices, dtype=np.int64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        row_ranges = np.ascontiguousarray(row_ranges, dtype=np.int64)

        self._A_gko = _cpp.create_distributed_matrix_from_local_coo(
            self._exec, self._gko_comm,
            row_indices, col_indices, values,
            global_rows, global_cols, row_ranges
        )
        self._row_ranges = row_ranges
        self._A_petsc = None

        if self._solver is None:
            self._solver = _cpp.DistributedSolver(self._exec, self._gko_comm, self._config)
        self._solver.set_operator(self._A_gko)

    def _configure_amg(self, amg, cfg: dict, _cpp) -> None:
        """Apply user AMG configuration."""
        if "max_levels" in cfg:
            amg.max_levels = cfg["max_levels"]

        if "min_coarse_rows" in cfg:
            amg.min_coarse_rows = cfg["min_coarse_rows"]

        if "deterministic" in cfg:
            amg.deterministic = cfg["deterministic"]

        if "cycle" in cfg:
            cycle_map = {"v": "V", "w": "W", "f": "F"}
            cycle = cfg["cycle"].lower()
            if cycle not in cycle_map:
                raise ValueError(f"Unknown AMG cycle: {cfg['cycle']}. Available: v, w, f")
            amg.cycle = getattr(_cpp.AMGConfig.Cycle, cycle_map[cycle])

        if "smoother" in cfg:
            smoother_map = {"jacobi": "JACOBI", "gauss_seidel": "GAUSS_SEIDEL", "ilu": "ILU"}
            smoother = cfg["smoother"].lower()
            if smoother not in smoother_map:
                raise ValueError(f"Unknown smoother: {cfg['smoother']}. "
                               f"Available: jacobi, gauss_seidel, ilu")
            amg.smoother = getattr(_cpp.AMGConfig.Smoother, smoother_map[smoother])

        if "pre_smooth_steps" in cfg:
            amg.pre_smooth_steps = cfg["pre_smooth_steps"]

        if "post_smooth_steps" in cfg:
            amg.post_smooth_steps = cfg["post_smooth_steps"]

        if "relaxation_factor" in cfg:
            amg.relaxation_factor = cfg["relaxation_factor"]

        if "coarse_solver" in cfg:
            coarse_map = {"direct": "DIRECT", "cg": "CG", "gmres": "GMRES"}
            coarse = cfg["coarse_solver"].lower()
            if coarse not in coarse_map:
                raise ValueError(f"Unknown coarse solver: {cfg['coarse_solver']}. "
                               f"Available: direct, cg, gmres")
            amg.coarse_solver = getattr(_cpp.AMGConfig.CoarseSolver, coarse_map[coarse])

        if "coarse_max_iterations" in cfg:
            amg.coarse_max_iterations = cfg["coarse_max_iterations"]

        if "coarse_tolerance" in cfg:
            amg.coarse_tolerance = cfg["coarse_tolerance"]

        if "use_mixed_precision" in cfg:
            amg.use_mixed_precision = cfg["use_mixed_precision"]

        if "mixed_precision_level" in cfg:
            amg.mixed_precision_level = cfg["mixed_precision_level"]

    def solve(self, b: Any, x: Any) -> int:
        """
        Solve the linear system Ax = b.

        Parameters
        ----------
        b : PETSc.Vec
            Right-hand side vector
        x : PETSc.Vec
            Solution vector (modified in place)

        Returns
        -------
        int
            Number of iterations. Negative if not converged.
        """
        if self._solver is None:
            raise RuntimeError("No matrix set. Call set_operator_from_petsc() or "
                             "set_operator_from_local_coo() first.")

        _cpp = self._cpp

        # Convert to Ginkgo vectors
        b_gko = _cpp.create_distributed_vector_from_petsc(self._exec, self._gko_comm, b)
        x_gko = _cpp.create_distributed_vector_from_petsc(self._exec, self._gko_comm, x)

        # Solve
        iters = self._solver.solve(b_gko, x_gko)

        # Copy solution back to PETSc
        _cpp.copy_to_petsc(x_gko, x)

        return iters

    def update_operator_from_petsc(self, A: Any) -> None:
        """
        Update the system matrix from PETSc (reuse structure if possible).

        Parameters
        ----------
        A : PETSc.Mat
            New system matrix
        """
        _cpp = self._cpp
        _cpp.update_matrix_values_from_petsc(self._A_gko, A)
        self._A_petsc = A

    def update_operator_from_local_coo(
        self,
        row_indices,
        col_indices,
        values,
        row_ranges=None,
    ) -> None:
        """
        Update the system matrix from local COO data.

        Parameters
        ----------
        row_indices : ndarray[int64]
            Global row indices of non-zeros
        col_indices : ndarray[int64]
            Global column indices of non-zeros
        values : ndarray[float64]
            Non-zero values
        row_ranges : ndarray[int64], optional
            Partition ranges. If None, uses ranges from initial set_operator call.
        """
        import numpy as np
        _cpp = self._cpp

        row_indices = np.ascontiguousarray(row_indices, dtype=np.int64)
        col_indices = np.ascontiguousarray(col_indices, dtype=np.int64)
        values = np.ascontiguousarray(values, dtype=np.float64)

        if row_ranges is None:
            row_ranges = self._row_ranges
        else:
            row_ranges = np.ascontiguousarray(row_ranges, dtype=np.int64)

        _cpp.update_matrix_from_local_coo(
            self._A_gko, row_indices, col_indices, values, row_ranges
        )

    # Backward compatibility alias
    def update_operator(self, A: Any) -> None:
        """Alias for update_operator_from_petsc (backward compatibility)."""
        self.update_operator_from_petsc(A)

    def set_tolerance(self, rtol: float, atol: float = 1e-12) -> None:
        """Update convergence tolerances."""
        self._solver.set_tolerance(rtol, atol)

    def set_max_iterations(self, max_iter: int) -> None:
        """Update maximum iterations."""
        self._solver.set_max_iterations(max_iter)

    @property
    def iterations(self) -> int:
        """Number of iterations from last solve."""
        return self._solver.iterations()

    @property
    def residual_norm(self) -> float:
        """Final residual norm from last solve."""
        return self._solver.residual_norm()

    @property
    def converged(self) -> bool:
        """Whether last solve converged."""
        return self._solver.converged()

    def __repr__(self) -> str:
        return (
            f"GinkgoSolver("
            f"solver={self._config.solver}, "
            f"preconditioner={self._config.preconditioner}, "
            f"rtol={self._config.rtol})"
        )
