# Ginkgo Integration Design for DOLFINx

## Executive Summary

This document outlines strategies for integrating Ginkgo as an alternative linear algebra backend in DOLFINx, enabling GPU-accelerated sparse linear solvers (CUDA, HIP, SYCL) for the CardioEMI simulation framework.

---

## 1. DOLFINx Architecture Overview

### 1.1 Linear Algebra Layer

DOLFINx has a **backend-agnostic** linear algebra design:

```
cpp/dolfinx/la/
├── Vector.h          # Templated distributed vector
├── MatrixCSR.h       # Templated CSR matrix
├── SparsityPattern.h # Symbolic sparsity structure
├── petsc.h           # Optional PETSc backend (compile flag)
└── utils.h           # Norms, inner products
```

Key characteristics:
- `la::MatrixCSR<T, Container>` is templated on storage container (enables GPU containers)
- `la::Vector<T, Container>` similarly templated
- PETSc is optional via `DOLFINX_ENABLE_PETSC` compile flag
- Recent v0.10.0 added GPU support for cast-copy operations

### 1.2 Assembly-Solve Separation

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   UFL Forms     │ ──► │  DOLFINx        │ ──► │  Backend        │
│   (symbolic)    │     │  Assembly       │     │  (PETSc/Ginkgo) │
└─────────────────┘     │  la::MatrixCSR  │     └─────────────────┘
                        │  la::Vector     │
                        └─────────────────┘
```

This separation is the key integration point.

### 1.3 PETSc Backend Pattern

Location: `cpp/dolfinx/la/petsc.h`

```cpp
namespace dolfinx::la::petsc {
  // Creation from DOLFINx structures
  Vec create_vector(const common::IndexMap& map, int bs);
  Mat create_matrix(const SparsityPattern& sp);

  // Wrapper classes with RAII
  class Vector { Vec _x; ... };
  class Matrix { Mat _A; ... };

  // Solver wrapper
  class KrylovSolver { KSP _ksp; ... };
}
```

---

## 2. Ginkgo Distributed Computing Support

### 2.1 Ginkgo's Distributed Infrastructure

Ginkgo provides distributed matrix and vector support via `gko::experimental::distributed`:

```cpp
#include <ginkgo/ginkgo.hpp>

namespace gko_dist = gko::experimental::distributed;

// Distributed matrix (CSR format per partition)
using dist_mtx = gko_dist::Matrix<double, int32_t, int64_t>;

// Distributed vector
using dist_vec = gko_dist::Vector<double>;

// Partition describes data distribution
using partition_type = gko::experimental::distributed::Partition<int32_t, int64_t>;
```

### 2.2 Key Components

| Component | Description |
|-----------|-------------|
| `Partition` | Describes how global indices map to MPI ranks |
| `distributed::Matrix` | Stores local CSR blocks with communication metadata |
| `distributed::Vector` | Local data + ghost handling |
| `distributed::preconditioner::Schwarz` | Wraps local preconditioners for distributed use |
| `solver::Multigrid` | Distributed AMG with PGM coarsening (Ginkgo 1.8.0+) |

### 2.3 Krylov Solvers

Standard Ginkgo Krylov solvers work directly with distributed matrices:
- `gko::solver::Cg` - Conjugate Gradient
- `gko::solver::Gmres` - GMRES
- `gko::solver::Bicgstab` - BiCGSTAB
- `gko::solver::Fcg` - Flexible CG

### 2.4 Preconditioners

#### Local Preconditioners (wrapped in Schwarz for distributed use)

```cpp
// Local preconditioner (runs on each rank's local matrix)
auto local_precond = gko::preconditioner::Jacobi<>::build()
    .with_max_block_size(32u)
    .on(exec);

// Wrap in Schwarz for distributed matrix
auto schwarz = gko_dist::preconditioner::Schwarz<double, int32_t, int64_t>::build()
    .with_local_solver(local_precond)
    .on(exec);
```

Available local preconditioners for Schwarz wrapping:
- `Jacobi` (block Jacobi)
- `Ilu` (incomplete LU)
- `Ic` (incomplete Cholesky)
- `Isai` (approximate sparse inverse)

#### Distributed AMG Preconditioner (Ginkgo 1.8.0+)

Ginkgo 1.8.0 introduced distributed multigrid with the PGM (Parallel Graph Match) coarsening method:

```cpp
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>

// Build PGM-based AMG preconditioner for distributed matrix
auto pgm_factory = gko::multigrid::Pgm<double, int32_t>::build()
    .with_deterministic(true)
    .on(exec);

// Smoother: typically Jacobi or Gauss-Seidel
auto smoother_factory = gko::solver::Ir<double>::build()
    .with_solver(gko::preconditioner::Jacobi<double, int32_t>::build()
        .with_max_block_size(1u))
    .with_relaxation_factor(0.9)
    .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
    .on(exec);

// Coarse solver: direct or iterative
auto coarse_factory = gko::experimental::solver::Direct<double, int32_t>::build()
    .with_factorization(gko::experimental::factorization::Lu<double, int32_t>::build())
    .on(exec);

// Build multigrid
auto mg_factory = gko::solver::Multigrid::build()
    .with_mg_level(pgm_factory)
    .with_pre_smoother(smoother_factory)
    .with_post_smoother(smoother_factory)
    .with_coarsest_solver(coarse_factory)
    .with_max_levels(10u)
    .with_min_coarse_rows(100u)
    .with_cycle(gko::solver::multigrid::cycle::v)
    .on(exec);

// Use as preconditioner in CG
auto solver_factory = gko::solver::Cg<double>::build()
    .with_preconditioner(mg_factory)
    .with_criteria(...)
    .on(exec);
```

#### Mixed-Precision AMG

Ginkgo supports mixed-precision AMG for additional performance:

```cpp
// Use single precision on coarser levels
auto mg_factory = gko::solver::Multigrid::build()
    .with_mg_level(pgm_factory)
    .with_level_selector([](auto level, auto) {
        return level > 2;  // Switch to float32 after level 2
    })
    // ...
```

---

## 3. Integration Strategy

### 3.1 Recommended Approach: Solver-Level Integration

**Approach**: Keep DOLFINx/multiphenicsx assembly, convert to Ginkgo distributed matrices for solving.

```
DOLFINx Assembly → PETSc Matrix → Extract CSR → Ginkgo dist::Matrix → Ginkgo Solver
                   (per rank)      (per rank)     (distributed)
```

### 3.2 Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         MPI Rank 0                               │
├──────────────────────────────────────────────────────────────────┤
│  multiphenicsx    →   Local PETSc   →   Extract    →   Ginkgo   │
│  assemble_block       Matrix            CSR data       dist::Mat │
└──────────────────────────────────────────────────────────────────┘
                              ↕ MPI Communication ↕
┌──────────────────────────────────────────────────────────────────┐
│                         MPI Rank 1                               │
├──────────────────────────────────────────────────────────────────┤
│  multiphenicsx    →   Local PETSc   →   Extract    →   Ginkgo   │
│  assemble_block       Matrix            CSR data       dist::Mat │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Plan

### Phase 1: Core Library (dolfinx-ginkgo)

#### 4.1 Directory Structure

```
dolfinx-ginkgo/
├── CMakeLists.txt
├── cpp/
│   └── dolfinx_ginkgo/
│       ├── ginkgo.h              # Main header
│       ├── Partition.h           # DOLFINx IndexMap → Ginkgo Partition
│       ├── DistributedMatrix.h   # Distributed matrix wrapper
│       ├── DistributedVector.h   # Distributed vector wrapper
│       ├── Solver.h              # Solver abstraction
│       ├── AMG.h                 # AMG preconditioner setup
│       └── convert.h             # PETSc/DOLFINx ↔ Ginkgo conversions
├── python/
│   └── dolfinx_ginkgo/
│       ├── __init__.py
│       ├── _cpp.cpp              # nanobind bindings
│       └── solver.py             # High-level Python API
└── tests/
```

#### 4.2 Core C++ Interface

```cpp
// cpp/dolfinx_ginkgo/ginkgo.h
#pragma once

#include <ginkgo/ginkgo.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <mpi.h>

namespace dolfinx_ginkgo {

namespace gko_dist = gko::experimental::distributed;

// ============================================================================
// Executor Management
// ============================================================================

enum class Backend { REFERENCE, OMP, CUDA, HIP, SYCL };

/// Create Ginkgo executor for specified backend
std::shared_ptr<gko::Executor>
create_executor(Backend backend, int device_id = 0);

/// Create Ginkgo MPI communicator wrapper
std::shared_ptr<gko::experimental::mpi::communicator>
create_communicator(MPI_Comm comm);

// ============================================================================
// Partition Creation from DOLFINx IndexMap
// ============================================================================

/// Create Ginkgo partition from DOLFINx IndexMap
template<typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Partition<LocalIndexType, GlobalIndexType>>
create_partition(std::shared_ptr<gko::Executor> exec,
                 std::shared_ptr<gko::experimental::mpi::communicator> comm,
                 const dolfinx::common::IndexMap& index_map);

// ============================================================================
// Distributed Matrix
// ============================================================================

/// Create Ginkgo distributed matrix from DOLFINx MatrixCSR
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_distributed_matrix(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> comm,
    const dolfinx::la::MatrixCSR<ValueType>& A,
    const dolfinx::common::IndexMap& row_map,
    const dolfinx::common::IndexMap& col_map);

/// Create Ginkgo distributed matrix from PETSc Mat (extracts CSR)
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_distributed_matrix_from_petsc(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> comm,
    Mat petsc_mat);

// ============================================================================
// Distributed Vector
// ============================================================================

/// Create Ginkgo distributed vector from DOLFINx Vector
template<typename ValueType = double>
std::shared_ptr<gko_dist::Vector<ValueType>>
create_distributed_vector(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> comm,
    const dolfinx::la::Vector<ValueType>& v);

/// Create Ginkgo distributed vector from PETSc Vec
template<typename ValueType = double>
std::shared_ptr<gko_dist::Vector<ValueType>>
create_distributed_vector_from_petsc(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> comm,
    Vec petsc_vec);

/// Copy Ginkgo distributed vector back to DOLFINx Vector
template<typename ValueType = double>
void copy_to_dolfinx(const gko_dist::Vector<ValueType>& gko_vec,
                     dolfinx::la::Vector<ValueType>& dfx_vec);

/// Copy Ginkgo distributed vector back to PETSc Vec
template<typename ValueType = double>
void copy_to_petsc(const gko_dist::Vector<ValueType>& gko_vec,
                   Vec petsc_vec);

// ============================================================================
// Solver Configuration
// ============================================================================

enum class SolverType { CG, FCG, GMRES, BICGSTAB, CGS };

enum class PreconditionerType {
    NONE,
    JACOBI,
    BLOCK_JACOBI,
    ILU,
    IC,
    ISAI,
    AMG          // Distributed AMG with PGM coarsening
};

/// AMG-specific configuration
struct AMGConfig {
    // Coarsening
    unsigned int max_levels = 10;
    unsigned int min_coarse_rows = 100;
    bool deterministic = true;

    // Cycle type
    enum class Cycle { V, W, F } cycle = Cycle::V;

    // Smoother
    enum class Smoother { JACOBI, GAUSS_SEIDEL, ILU } smoother = Smoother::JACOBI;
    unsigned int pre_smooth_steps = 1;
    unsigned int post_smooth_steps = 1;
    double relaxation_factor = 0.9;

    // Coarse solver
    enum class CoarseSolver { DIRECT, CG, GMRES } coarse_solver = CoarseSolver::DIRECT;
    int coarse_max_iterations = 100;
    double coarse_tolerance = 1e-10;

    // Mixed precision (optional)
    bool use_mixed_precision = false;
    unsigned int mixed_precision_level = 2;  // Switch precision after this level
};

struct SolverConfig {
    SolverType solver = SolverType::CG;
    PreconditionerType preconditioner = PreconditionerType::AMG;

    // Convergence criteria
    double rtol = 1e-8;
    double atol = 1e-12;
    int max_iterations = 1000;

    // GMRES specific
    int krylov_dim = 30;

    // Jacobi specific
    unsigned int jacobi_block_size = 32;

    // ILU specific
    int ilu_fill_level = 0;

    // AMG configuration
    AMGConfig amg;

    // Verbose output
    bool verbose = false;
};

// ============================================================================
// Linear Solver
// ============================================================================

template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
class DistributedSolver {
public:
    using matrix_type = gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using vector_type = gko_dist::Vector<ValueType>;

    /// Constructor
    DistributedSolver(std::shared_ptr<gko::Executor> exec,
                      std::shared_ptr<gko::experimental::mpi::communicator> comm,
                      const SolverConfig& config = SolverConfig{});

    /// Set the system matrix (triggers preconditioner setup)
    void set_operator(std::shared_ptr<matrix_type> A);

    /// Update convergence criteria
    void set_tolerance(ValueType rtol, ValueType atol = 1e-12);
    void set_max_iterations(int max_iter);

    /// Solve Ax = b
    /// Returns number of iterations (negative if not converged)
    int solve(const vector_type& b, vector_type& x);

    /// Get iteration count from last solve
    int iterations() const;

    /// Get final residual norm from last solve
    ValueType residual_norm() const;

    /// Check if last solve converged
    bool converged() const;

private:
    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<gko::experimental::mpi::communicator> comm_;
    SolverConfig config_;

    std::shared_ptr<matrix_type> A_;
    std::shared_ptr<gko::LinOp> solver_;

    int last_iterations_ = 0;
    ValueType last_residual_ = 0;
    bool last_converged_ = false;

    void build_solver();
    std::shared_ptr<gko::LinOp> build_preconditioner();
    std::shared_ptr<gko::LinOpFactory> build_amg_factory();
};

} // namespace dolfinx_ginkgo
```

#### 4.3 AMG Preconditioner Implementation

```cpp
// cpp/dolfinx_ginkgo/AMG.h
#pragma once

#include <ginkgo/ginkgo.hpp>
#include "ginkgo.h"

namespace dolfinx_ginkgo {

template<typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<gko::LinOpFactory>
DistributedSolver<ValueType, LocalIndexType, GlobalIndexType>::build_amg_factory()
{
    const auto& amg_cfg = config_.amg;

    // Build PGM coarsening factory
    auto pgm_factory = gko::multigrid::Pgm<ValueType, LocalIndexType>::build()
        .with_deterministic(amg_cfg.deterministic)
        .on(exec_);

    // Build smoother factory
    std::shared_ptr<gko::LinOpFactory> smoother_factory;
    switch (amg_cfg.smoother) {
        case AMGConfig::Smoother::JACOBI: {
            auto jacobi = gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                .with_max_block_size(1u)
                .on(exec_);
            smoother_factory = gko::solver::Ir<ValueType>::build()
                .with_solver(jacobi)
                .with_relaxation_factor(amg_cfg.relaxation_factor)
                .with_criteria(gko::stop::Iteration::build()
                    .with_max_iters(static_cast<gko::size_type>(amg_cfg.pre_smooth_steps))
                    .on(exec_))
                .on(exec_);
            break;
        }
        case AMGConfig::Smoother::GAUSS_SEIDEL: {
            // Gauss-Seidel via lower triangular solve
            smoother_factory = gko::solver::LowerTrs<ValueType, LocalIndexType>::build()
                .on(exec_);
            break;
        }
        case AMGConfig::Smoother::ILU: {
            auto ilu = gko::factorization::ParIlu<ValueType, LocalIndexType>::build()
                .on(exec_);
            smoother_factory = gko::solver::Ir<ValueType>::build()
                .with_solver(ilu)
                .with_criteria(gko::stop::Iteration::build()
                    .with_max_iters(static_cast<gko::size_type>(amg_cfg.pre_smooth_steps))
                    .on(exec_))
                .on(exec_);
            break;
        }
    }

    // Build coarse solver factory
    std::shared_ptr<gko::LinOpFactory> coarse_factory;
    switch (amg_cfg.coarse_solver) {
        case AMGConfig::CoarseSolver::DIRECT: {
            coarse_factory = gko::experimental::solver::Direct<ValueType, LocalIndexType>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<ValueType, LocalIndexType>::build()
                        .on(exec_))
                .on(exec_);
            break;
        }
        case AMGConfig::CoarseSolver::CG: {
            coarse_factory = gko::solver::Cg<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build()
                        .with_max_iters(static_cast<gko::size_type>(amg_cfg.coarse_max_iterations))
                        .on(exec_),
                    gko::stop::RelativeResidualNorm<ValueType>::build()
                        .with_tolerance(amg_cfg.coarse_tolerance)
                        .on(exec_))
                .on(exec_);
            break;
        }
        case AMGConfig::CoarseSolver::GMRES: {
            coarse_factory = gko::solver::Gmres<ValueType>::build()
                .with_krylov_dim(30u)
                .with_criteria(
                    gko::stop::Iteration::build()
                        .with_max_iters(static_cast<gko::size_type>(amg_cfg.coarse_max_iterations))
                        .on(exec_),
                    gko::stop::RelativeResidualNorm<ValueType>::build()
                        .with_tolerance(amg_cfg.coarse_tolerance)
                        .on(exec_))
                .on(exec_);
            break;
        }
    }

    // Determine cycle type
    gko::solver::multigrid::cycle cycle_type;
    switch (amg_cfg.cycle) {
        case AMGConfig::Cycle::V:
            cycle_type = gko::solver::multigrid::cycle::v;
            break;
        case AMGConfig::Cycle::W:
            cycle_type = gko::solver::multigrid::cycle::w;
            break;
        case AMGConfig::Cycle::F:
            cycle_type = gko::solver::multigrid::cycle::f;
            break;
    }

    // Build multigrid factory
    auto mg_builder = gko::solver::Multigrid::build()
        .with_mg_level(pgm_factory)
        .with_pre_smoother(smoother_factory)
        .with_post_smoother(smoother_factory)
        .with_coarsest_solver(coarse_factory)
        .with_max_levels(static_cast<gko::size_type>(amg_cfg.max_levels))
        .with_min_coarse_rows(static_cast<gko::size_type>(amg_cfg.min_coarse_rows))
        .with_cycle(cycle_type);

    // Optional: Mixed precision
    if (amg_cfg.use_mixed_precision) {
        mg_builder.with_level_selector([level = amg_cfg.mixed_precision_level](auto l, auto) {
            return l >= level;
        });
    }

    return mg_builder.on(exec_);
}

} // namespace dolfinx_ginkgo
```

#### 4.4 Solver Implementation with Schwarz and AMG

```cpp
// cpp/dolfinx_ginkgo/Solver.h (implementation details)

template<typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<gko::LinOp>
DistributedSolver<ValueType, LocalIndexType, GlobalIndexType>::build_preconditioner()
{
    using schwarz_type = gko_dist::preconditioner::Schwarz<ValueType, LocalIndexType, GlobalIndexType>;

    // AMG is handled separately (already distributed-aware)
    if (config_.preconditioner == PreconditionerType::AMG) {
        auto amg_factory = build_amg_factory();
        return amg_factory->generate(A_);
    }

    // For other preconditioners, wrap in Schwarz
    std::shared_ptr<gko::LinOpFactory> local_factory;

    switch (config_.preconditioner) {
        case PreconditionerType::NONE:
            return nullptr;

        case PreconditionerType::JACOBI:
            local_factory = gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                .with_max_block_size(1u)
                .on(exec_);
            break;

        case PreconditionerType::BLOCK_JACOBI:
            local_factory = gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                .with_max_block_size(config_.jacobi_block_size)
                .on(exec_);
            break;

        case PreconditionerType::ILU:
            local_factory = gko::factorization::ParIlu<ValueType, LocalIndexType>::build()
                .on(exec_);
            break;

        case PreconditionerType::IC:
            local_factory = gko::factorization::ParIc<ValueType, LocalIndexType>::build()
                .on(exec_);
            break;

        case PreconditionerType::ISAI:
            local_factory = gko::preconditioner::Isai<
                gko::preconditioner::LowerIsai, ValueType, LocalIndexType>::build()
                .on(exec_);
            break;

        default:
            return nullptr;
    }

    // Wrap local preconditioner in Schwarz for distributed use
    return schwarz_type::build()
        .with_local_solver(local_factory)
        .on(exec_)
        ->generate(A_);
}

template<typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DistributedSolver<ValueType, LocalIndexType, GlobalIndexType>::build_solver()
{
    // Build convergence criteria
    auto iter_stop = gko::stop::Iteration::build()
        .with_max_iters(static_cast<gko::size_type>(config_.max_iterations))
        .on(exec_);

    auto rel_stop = gko::stop::RelativeResidualNorm<ValueType>::build()
        .with_tolerance(config_.rtol)
        .on(exec_);

    auto abs_stop = gko::stop::AbsoluteResidualNorm<ValueType>::build()
        .with_tolerance(config_.atol)
        .on(exec_);

    // Build preconditioner
    auto precond = build_preconditioner();

    // Build solver factory
    std::shared_ptr<gko::LinOpFactory> solver_factory;

    auto build_with_precond = [&](auto builder) {
        if (precond) {
            return builder.with_preconditioner(
                gko::share(precond)).on(exec_);
        } else {
            return builder.on(exec_);
        }
    };

    switch (config_.solver) {
        case SolverType::CG:
            solver_factory = build_with_precond(
                gko::solver::Cg<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop));
            break;

        case SolverType::FCG:
            solver_factory = build_with_precond(
                gko::solver::Fcg<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop));
            break;

        case SolverType::GMRES:
            solver_factory = build_with_precond(
                gko::solver::Gmres<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop)
                    .with_krylov_dim(static_cast<gko::size_type>(config_.krylov_dim)));
            break;

        case SolverType::BICGSTAB:
            solver_factory = build_with_precond(
                gko::solver::Bicgstab<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop));
            break;

        case SolverType::CGS:
            solver_factory = build_with_precond(
                gko::solver::Cgs<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop));
            break;
    }

    solver_ = solver_factory->generate(A_);
}
```

#### 4.5 Python Interface

```python
# python/dolfinx_ginkgo/solver.py
"""High-level Python interface for Ginkgo-based distributed solvers."""

from __future__ import annotations
from typing import Optional, Literal
from mpi4py import MPI
import numpy as np

from dolfinx_ginkgo._cpp import (
    Backend,
    SolverType,
    PreconditionerType,
    SolverConfig,
    AMGConfig,
    create_executor,
    create_communicator,
    create_distributed_matrix_from_petsc,
    create_distributed_vector_from_petsc,
    copy_to_petsc,
    DistributedSolver as _DistributedSolver,
)


class GinkgoSolver:
    """
    Ginkgo-based distributed linear solver for DOLFINx/PETSc systems.

    Parameters
    ----------
    A : PETSc.Mat
        System matrix (distributed)
    comm : MPI.Comm
        MPI communicator
    backend : str
        Compute backend: "cuda", "hip", "omp", or "reference"
    device_id : int
        GPU device ID (for CUDA/HIP backends)
    solver : str
        Solver type: "cg", "fcg", "gmres", "bicgstab", "cgs"
    preconditioner : str
        Preconditioner: "none", "jacobi", "block_jacobi", "ilu", "ic", "isai", "amg"
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_iter : int
        Maximum iterations
    krylov_dim : int
        Krylov dimension (GMRES only)
    jacobi_block_size : int
        Block size for block Jacobi preconditioner
    amg_config : dict, optional
        AMG-specific configuration (see AMGConfig)
    verbose : bool
        Print convergence info
    """

    _backend_map = {
        "reference": Backend.REFERENCE,
        "omp": Backend.OMP,
        "cuda": Backend.CUDA,
        "hip": Backend.HIP,
        "sycl": Backend.SYCL,
    }

    _solver_map = {
        "cg": SolverType.CG,
        "fcg": SolverType.FCG,
        "gmres": SolverType.GMRES,
        "bicgstab": SolverType.BICGSTAB,
        "cgs": SolverType.CGS,
    }

    _precond_map = {
        "none": PreconditionerType.NONE,
        "jacobi": PreconditionerType.JACOBI,
        "block_jacobi": PreconditionerType.BLOCK_JACOBI,
        "ilu": PreconditionerType.ILU,
        "ic": PreconditionerType.IC,
        "isai": PreconditionerType.ISAI,
        "amg": PreconditionerType.AMG,
    }

    def __init__(
        self,
        A,  # PETSc.Mat
        comm: MPI.Comm = MPI.COMM_WORLD,
        backend: Literal["reference", "omp", "cuda", "hip", "sycl"] = "cuda",
        device_id: int = 0,
        solver: Literal["cg", "fcg", "gmres", "bicgstab", "cgs"] = "cg",
        preconditioner: Literal["none", "jacobi", "block_jacobi", "ilu", "ic", "isai", "amg"] = "amg",
        rtol: float = 1e-8,
        atol: float = 1e-12,
        max_iter: int = 1000,
        krylov_dim: int = 30,
        jacobi_block_size: int = 32,
        amg_config: Optional[dict] = None,
        verbose: bool = False,
    ):
        # Create executor
        self._exec = create_executor(self._backend_map[backend], device_id)

        # Create Ginkgo MPI communicator
        self._gko_comm = create_communicator(comm)

        # Build solver config
        config = SolverConfig()
        config.solver = self._solver_map[solver]
        config.preconditioner = self._precond_map[preconditioner]
        config.rtol = rtol
        config.atol = atol
        config.max_iterations = max_iter
        config.krylov_dim = krylov_dim
        config.jacobi_block_size = jacobi_block_size
        config.verbose = verbose

        # Configure AMG if selected
        if preconditioner == "amg" and amg_config is not None:
            self._configure_amg(config.amg, amg_config)

        # Create Ginkgo distributed matrix from PETSc
        self._A_gko = create_distributed_matrix_from_petsc(
            self._exec, self._gko_comm, A
        )

        # Create solver
        self._solver = _DistributedSolver(self._exec, self._gko_comm, config)
        self._solver.set_operator(self._A_gko)

        # Store config
        self._config = config
        self._comm = comm

    def _configure_amg(self, amg: AMGConfig, cfg: dict):
        """Apply user AMG configuration."""
        if "max_levels" in cfg:
            amg.max_levels = cfg["max_levels"]
        if "min_coarse_rows" in cfg:
            amg.min_coarse_rows = cfg["min_coarse_rows"]
        if "cycle" in cfg:
            amg.cycle = {"v": AMGConfig.Cycle.V,
                         "w": AMGConfig.Cycle.W,
                         "f": AMGConfig.Cycle.F}[cfg["cycle"].lower()]
        if "smoother" in cfg:
            amg.smoother = {"jacobi": AMGConfig.Smoother.JACOBI,
                           "gauss_seidel": AMGConfig.Smoother.GAUSS_SEIDEL,
                           "ilu": AMGConfig.Smoother.ILU}[cfg["smoother"].lower()]
        if "pre_smooth_steps" in cfg:
            amg.pre_smooth_steps = cfg["pre_smooth_steps"]
        if "post_smooth_steps" in cfg:
            amg.post_smooth_steps = cfg["post_smooth_steps"]
        if "relaxation_factor" in cfg:
            amg.relaxation_factor = cfg["relaxation_factor"]
        if "coarse_solver" in cfg:
            amg.coarse_solver = {"direct": AMGConfig.CoarseSolver.DIRECT,
                                 "cg": AMGConfig.CoarseSolver.CG,
                                 "gmres": AMGConfig.CoarseSolver.GMRES}[cfg["coarse_solver"].lower()]
        if "use_mixed_precision" in cfg:
            amg.use_mixed_precision = cfg["use_mixed_precision"]
        if "mixed_precision_level" in cfg:
            amg.mixed_precision_level = cfg["mixed_precision_level"]

    def solve(self, b, x) -> int:
        """
        Solve Ax = b.

        Parameters
        ----------
        b : PETSc.Vec
            Right-hand side vector
        x : PETSc.Vec
            Solution vector (modified in place)

        Returns
        -------
        int
            Number of iterations (negative if not converged)
        """
        # Convert to Ginkgo vectors
        b_gko = create_distributed_vector_from_petsc(self._exec, self._gko_comm, b)
        x_gko = create_distributed_vector_from_petsc(self._exec, self._gko_comm, x)

        # Solve
        iters = self._solver.solve(b_gko, x_gko)

        # Copy solution back to PETSc
        copy_to_petsc(x_gko, x)

        return iters

    def set_tolerance(self, rtol: float, atol: float = 1e-12):
        """Update convergence tolerances."""
        self._solver.set_tolerance(rtol, atol)

    def set_max_iterations(self, max_iter: int):
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
```

---

## 5. Integration with CardioEMI

### 5.1 Modified main.py Usage

```python
# In main.py, replace PETSc KSP with Ginkgo solver

from dolfinx_ginkgo import GinkgoSolver

# After assembling matrix with multiphenicsx
A = multiphenicsx.fem.petsc.assemble_matrix_block(a, bcs=bcs, restriction=(restriction, restriction))
A.assemble()

# Create Ginkgo solver with AMG preconditioner (setup phase - done once)
gko_solver = GinkgoSolver(
    A,
    comm=comm,
    backend="cuda",  # or "hip", "omp"
    solver="cg",
    preconditioner="amg",
    rtol=params["ksp_rtol"],
    amg_config={
        "max_levels": 10,
        "cycle": "v",
        "smoother": "jacobi",
        "coarse_solver": "direct",
    },
    verbose=params["verbose"],
)

# In time loop (solve phase - every timestep)
for time_step in range(params["time_steps"]):
    # ... assemble b ...

    # Solve with Ginkgo
    iters = gko_solver.solve(b, sol_vec)
    ksp_iterations.append(iters)

    # ... extract solution ...
```

### 5.2 Configuration Extension

```yaml
# input_pepe36_colored.yml

# Solver configuration
solver_backend: "ginkgo"  # or "petsc"
ginkgo_backend: "cuda"    # or "hip", "omp", "reference"
ginkgo_device_id: 0
ginkgo_solver: "cg"
ginkgo_preconditioner: "amg"

# AMG configuration
ginkgo_amg:
  max_levels: 10
  min_coarse_rows: 100
  cycle: "v"              # v, w, or f
  smoother: "jacobi"      # jacobi, gauss_seidel, or ilu
  pre_smooth_steps: 1
  post_smooth_steps: 1
  relaxation_factor: 0.9
  coarse_solver: "direct" # direct, cg, or gmres
  use_mixed_precision: false
  mixed_precision_level: 2
```

---

## 6. Preconditioner Comparison

| Preconditioner | Distributed | GPU | Setup Cost | Per-Iter Cost | Best For |
|----------------|-------------|-----|------------|---------------|----------|
| Jacobi | Schwarz | Yes | Low | Low | Simple problems |
| Block Jacobi | Schwarz | Yes | Low | Low | Block-structured |
| ILU | Schwarz | Yes | Medium | Low | General SPD |
| IC | Schwarz | Yes | Medium | Low | SPD systems |
| ISAI | Schwarz | Yes | High | Low | Sparse inverses |
| **AMG** | Native | Yes | High | Medium | **Large-scale, scalable** |

For CardioEMI with large meshes, **AMG is recommended** due to its mesh-independent convergence.

---

## 7. Performance Considerations

### 7.1 Memory Transfer Optimization

| Operation | Transfer | Frequency | Optimization |
|-----------|----------|-----------|--------------|
| Matrix A | Host → Device | Once (setup) | Acceptable overhead |
| AMG hierarchy | Computed on device | Once (setup) | GPU-accelerated |
| RHS b | Host → Device | Every timestep | Keep on GPU* |
| Solution x | Device → Host | Every timestep | Keep on GPU* |

*Future optimization: Assemble directly on GPU to eliminate transfers.

### 7.2 Expected Performance Profile

| Phase | CPU (PETSc) | GPU (Ginkgo+AMG) | Notes |
|-------|-------------|------------------|-------|
| Assembly | 100% | 100% | Still on CPU |
| AMG Setup | 100% | 50-80% | GPU coarsening |
| Transfer b | 0% | ~5% | Small overhead |
| Solve | 100% | 10-30% | GPU speedup |
| Transfer x | 0% | ~5% | Small overhead |
| **Total** | 100% | 20-40% | For solver-bound problems |

---

## 8. Dependencies

### Required

- Ginkgo >= 1.8.0 (distributed AMG support)
- DOLFINx >= 0.9.0
- PETSc (for assembly, can be CPU-only)
- nanobind (Python bindings)
- CMake >= 3.19

### Optional (GPU backends)

- CUDA Toolkit >= 11.0 (NVIDIA)
- ROCm >= 4.5 (AMD)
- oneAPI >= 2023.1 (Intel)

---

## 9. Implementation Roadmap

| Phase | Description | Deliverables |
|-------|-------------|--------------|
| **1** | Core infrastructure | Executor, partition, matrix/vector conversion |
| **2** | Solver wrapper | CG, GMRES with Schwarz-wrapped preconditioners |
| **3** | AMG integration | Distributed multigrid with PGM coarsening |
| **4** | Python bindings | nanobind wrappers, high-level API |
| **5** | CardioEMI integration | Modified main.py, benchmarks |
| **6** | Optimization | Persistent vectors, async transfers |
| **7** | Documentation | Usage guide, API docs |

---

## 10. References

- [Ginkgo Documentation](https://ginkgo-project.github.io/)
- [Ginkgo 1.8.0 Release Notes](https://github.com/ginkgo-project/ginkgo/releases/tag/v1.8.0) - Distributed AMG
- [Sparse Day 2025 - Ginkgo Advances](https://sparsedays.cerfacs.fr/wp-content/uploads/sites/72/2025/07/2025_SparseDay_YHMTsai_Ginkgo.pdf)
- [Three-precision AMG on GPUs](https://www.sciencedirect.com/science/article/abs/pii/S0167739X23002741)
- [pyGinkgo](https://arxiv.org/html/2510.08230)
- [DOLFINx la module](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.la.html)
- [cuda-dolfinx](https://github.com/bpachev/cuda-dolfinx)
