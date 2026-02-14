// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file ginkgo.h
/// @brief Main header for dolfinx-ginkgo library
///
/// This library provides Ginkgo-based distributed linear solvers for DOLFINx,
/// enabling GPU-accelerated sparse linear algebra with CUDA, HIP, and SYCL backends.

#include <ginkgo/ginkgo.hpp>
#include <mpi.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace dolfinx_ginkgo {

/// Namespace alias for Ginkgo distributed types
namespace gko_dist = gko::experimental::distributed;

// ============================================================================
// Backend Configuration
// ============================================================================

/// Available compute backends
enum class Backend {
    REFERENCE,  ///< Reference (sequential CPU)
    OMP,        ///< OpenMP (parallel CPU)
    CUDA,       ///< NVIDIA CUDA
    HIP,        ///< AMD HIP (ROCm)
    SYCL        ///< Intel SYCL (oneAPI)
};

/// Convert backend enum to string
inline std::string backend_to_string(Backend backend) {
    switch (backend) {
        case Backend::REFERENCE: return "reference";
        case Backend::OMP: return "omp";
        case Backend::CUDA: return "cuda";
        case Backend::HIP: return "hip";
        case Backend::SYCL: return "sycl";
        default: return "unknown";
    }
}

/// Parse backend from string
inline Backend string_to_backend(const std::string& str) {
    if (str == "reference" || str == "ref") return Backend::REFERENCE;
    if (str == "omp" || str == "openmp") return Backend::OMP;
    if (str == "cuda") return Backend::CUDA;
    if (str == "hip" || str == "rocm") return Backend::HIP;
    if (str == "sycl" || str == "dpcpp") return Backend::SYCL;
    throw std::invalid_argument("Unknown backend: " + str);
}

// ============================================================================
// Executor Management
// ============================================================================

/// Create a Ginkgo executor for the specified backend
///
/// @param backend The compute backend to use
/// @param device_id GPU device ID (ignored for CPU backends)
/// @return Shared pointer to the Ginkgo executor
///
/// @note For GPU backends, the reference executor is automatically set as
///       the master executor for CPU fallback operations.
inline std::shared_ptr<gko::Executor>
create_executor(Backend backend, int device_id = 0) {
    auto ref = gko::ReferenceExecutor::create();

    switch (backend) {
        case Backend::REFERENCE:
            return ref;

        case Backend::OMP:
            return gko::OmpExecutor::create();

#ifdef DOLFINX_GINKGO_ENABLE_CUDA
        case Backend::CUDA:
            return gko::CudaExecutor::create(device_id, ref);
#endif

#ifdef DOLFINX_GINKGO_ENABLE_HIP
        case Backend::HIP:
            return gko::HipExecutor::create(device_id, ref);
#endif

#ifdef DOLFINX_GINKGO_ENABLE_SYCL
        case Backend::SYCL:
            return gko::DpcppExecutor::create(device_id, ref);
#endif

        default:
            throw std::runtime_error(
                "Backend " + backend_to_string(backend) + " not enabled at compile time"
            );
    }
}

/// Create a Ginkgo MPI communicator wrapper
///
/// @param comm MPI communicator
/// @return Shared pointer to the Ginkgo MPI communicator
inline std::shared_ptr<gko::experimental::mpi::communicator>
create_communicator(MPI_Comm comm) {
    return std::make_shared<gko::experimental::mpi::communicator>(comm);
}

/// Get the number of available devices for a backend
///
/// @param backend The compute backend to query
/// @return Number of available devices (1 for CPU backends)
inline int get_device_count(Backend backend) {
    switch (backend) {
        case Backend::REFERENCE:
        case Backend::OMP:
            return 1;

#ifdef DOLFINX_GINKGO_ENABLE_CUDA
        case Backend::CUDA: {
            int count = 0;
            cudaGetDeviceCount(&count);
            return count;
        }
#endif

#ifdef DOLFINX_GINKGO_ENABLE_HIP
        case Backend::HIP: {
            int count = 0;
            hipGetDeviceCount(&count);
            return count;
        }
#endif

#ifdef DOLFINX_GINKGO_ENABLE_SYCL
        case Backend::SYCL: {
            // SYCL device enumeration is more complex
            // Return 1 as a placeholder
            return 1;
        }
#endif

        default:
            return 0;
    }
}

/// Check if a backend is available (compiled and has devices)
///
/// @param backend The compute backend to check
/// @return true if the backend is available
inline bool is_backend_available(Backend backend) {
    switch (backend) {
        case Backend::REFERENCE:
        case Backend::OMP:
            return true;

        case Backend::CUDA:
#ifdef DOLFINX_GINKGO_ENABLE_CUDA
            return get_device_count(Backend::CUDA) > 0;
#else
            return false;
#endif

        case Backend::HIP:
#ifdef DOLFINX_GINKGO_ENABLE_HIP
            return get_device_count(Backend::HIP) > 0;
#else
            return false;
#endif

        case Backend::SYCL:
#ifdef DOLFINX_GINKGO_ENABLE_SYCL
            return true;  // Assume available if compiled
#else
            return false;
#endif

        default:
            return false;
    }
}

// ============================================================================
// Solver Configuration Types
// ============================================================================

/// Krylov solver types
enum class SolverType {
    CG,        ///< Conjugate Gradient (for SPD systems)
    FCG,       ///< Flexible Conjugate Gradient
    GMRES,     ///< Generalized Minimal Residual
    BICGSTAB,  ///< Bi-Conjugate Gradient Stabilized
    CGS        ///< Conjugate Gradient Squared
};

/// Preconditioner types
enum class PreconditionerType {
    NONE,         ///< No preconditioner
    JACOBI,       ///< Point Jacobi (diagonal scaling)
    BLOCK_JACOBI, ///< Block Jacobi
    ILU,          ///< Incomplete LU factorization
    IC,           ///< Incomplete Cholesky factorization
    ISAI,         ///< Approximate sparse inverse
    AMG,          ///< Algebraic multigrid with PGM coarsening
    BDDC          ///< Balancing Domain Decomposition by Constraints (requires DdMatrix)
};

/// AMG-specific configuration
struct AMGConfig {
    // Coarsening parameters
    unsigned int max_levels = 10;         ///< Maximum number of multigrid levels
    unsigned int min_coarse_rows = 100;   ///< Stop coarsening when matrix has fewer rows
    bool deterministic = true;            ///< Use deterministic coarsening (reproducible)

    // Cycle type
    enum class Cycle { V, W, F };
    Cycle cycle = Cycle::V;               ///< Multigrid cycle type

    // Smoother configuration
    enum class Smoother { JACOBI, GAUSS_SEIDEL, ILU };
    Smoother smoother = Smoother::JACOBI; ///< Smoother type
    unsigned int pre_smooth_steps = 1;    ///< Pre-smoothing iterations
    unsigned int post_smooth_steps = 1;   ///< Post-smoothing iterations
    double relaxation_factor = 0.9;       ///< Smoother relaxation factor

    // Coarse solver configuration
    enum class CoarseSolver { DIRECT, CG, GMRES };
    CoarseSolver coarse_solver = CoarseSolver::DIRECT;  ///< Coarse level solver
    int coarse_max_iterations = 100;      ///< Max iterations for iterative coarse solver
    double coarse_tolerance = 1e-10;      ///< Tolerance for iterative coarse solver

    // Mixed precision (optional)
    bool use_mixed_precision = false;     ///< Enable mixed precision AMG
    unsigned int mixed_precision_level = 2;  ///< Switch precision after this level
};

/// BDDC-specific configuration
struct BDDCConfig {
    // Constraint types (which DOFs to use as primal constraints)
    bool vertices = true;     ///< Use vertex constraints
    bool edges = true;        ///< Use edge average constraints
    bool faces = true;        ///< Use face average constraints

    // Scaling type for weighted averages at interfaces
    enum class Scaling { STIFFNESS, DELUXE };
    Scaling scaling = Scaling::STIFFNESS;

    // Local solver configuration
    enum class LocalSolver { DIRECT, ILU, IC, AMG };
    LocalSolver local_solver = LocalSolver::DIRECT;
    int local_max_iterations = 100;       ///< Max iterations for iterative local solver
    double local_tolerance = 1e-12;       ///< Tolerance for iterative local solver

    // Local AMG configuration (used when local_solver = AMG)
    struct LocalAMGConfig {
        unsigned int max_levels = 10;
        unsigned int min_coarse_rows = 50;

        enum class Smoother { JACOBI, GAUSS_SEIDEL, ILU };
        Smoother smoother = Smoother::JACOBI;
        unsigned int smooth_steps = 1;
        double relaxation_factor = 0.9;

        enum class CoarseSolver { DIRECT, CG, GMRES };
        CoarseSolver coarse_solver = CoarseSolver::DIRECT;
        int coarse_max_iterations = 100;
    };
    LocalAMGConfig local_amg;

    // Coarse solver configuration (distributed - no direct solver)
    enum class CoarseSolver { CG, GMRES, BDDC };
    CoarseSolver coarse_solver = CoarseSolver::CG;
    int coarse_max_iterations = 100;      ///< Max iterations for iterative coarse solver
    double coarse_tolerance = 1e-10;      ///< Tolerance for iterative coarse solver

    // Nested BDDC configuration (used when coarse_solver = BDDC)
    LocalSolver coarse_bddc_local_solver = LocalSolver::DIRECT;  ///< Local solver for nested BDDC

    // Advanced options
    bool repartition_coarse = true;       ///< Repartition coarse problem for load balance
    bool constant_nullspace = false;      ///< Handle constant nullspace (pure Neumann BC)
};

/// Complete solver configuration
struct SolverConfig {
    // Solver selection
    SolverType solver = SolverType::CG;
    PreconditionerType preconditioner = PreconditionerType::AMG;

    // Convergence criteria
    double rtol = 1e-8;                   ///< Relative tolerance
    double atol = 1e-12;                  ///< Absolute tolerance
    int max_iterations = 1000;            ///< Maximum iterations

    // Solver-specific parameters
    int krylov_dim = 30;                  ///< GMRES Krylov dimension

    // Preconditioner-specific parameters
    unsigned int jacobi_block_size = 32;  ///< Block Jacobi block size
    int ilu_fill_level = 0;               ///< ILU fill level

    // AMG configuration
    AMGConfig amg;

    // BDDC configuration
    BDDCConfig bddc;

    // Output
    bool verbose = false;                 ///< Print convergence info
};

// ============================================================================
// Type Aliases
// ============================================================================

/// Default value type for matrices and vectors
using DefaultValueType = double;

/// Default local index type (matches PETSc local indices)
using DefaultLocalIndexType = std::int32_t;

/// Default global index type (matches PETSc global indices)
using DefaultGlobalIndexType = std::int64_t;

/// Distributed matrix type alias
template<typename ValueType = DefaultValueType,
         typename LocalIndexType = DefaultLocalIndexType,
         typename GlobalIndexType = DefaultGlobalIndexType>
using DistMatrix = gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>;

/// Domain decomposition matrix type alias (for BDDC and related methods)
template<typename ValueType = DefaultValueType,
         typename LocalIndexType = DefaultLocalIndexType,
         typename GlobalIndexType = DefaultGlobalIndexType>
using DdMatrix = gko_dist::DdMatrix<ValueType, LocalIndexType, GlobalIndexType>;

/// Distributed vector type alias
template<typename ValueType = DefaultValueType>
using DistVector = gko_dist::Vector<ValueType>;

/// Partition type alias
template<typename LocalIndexType = DefaultLocalIndexType,
         typename GlobalIndexType = DefaultGlobalIndexType>
using Partition = gko_dist::Partition<LocalIndexType, GlobalIndexType>;

} // namespace dolfinx_ginkgo

// Include other headers
#include "Partition.h"
#include "convert.h"
#include "DistributedMatrix.h"
#include "DistributedVector.h"
#include "Solver.h"
