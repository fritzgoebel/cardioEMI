// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file Solver.h
/// @brief Distributed Krylov solvers with preconditioners for DOLFINx

#include <ginkgo/ginkgo.hpp>
#include <mpi.h>

#include <memory>
#include <stdexcept>
#include <iostream>

#include "ginkgo.h"

namespace dolfinx_ginkgo {

/// Distributed linear solver using Ginkgo
///
/// This class wraps Ginkgo's Krylov solvers and preconditioners for use
/// with distributed matrices from DOLFINx/PETSc.
///
/// Supported solvers: CG, FCG, GMRES, BiCGSTAB, CGS
/// Supported preconditioners: Jacobi, Block Jacobi, ILU, IC, ISAI, AMG
///
/// @tparam ValueType Matrix/vector value type (default: double)
/// @tparam LocalIndexType Local index type (default: int32_t)
/// @tparam GlobalIndexType Global index type (default: int64_t)
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
class DistributedSolver {
public:
    using matrix_type = gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using vector_type = gko_dist::Vector<ValueType>;
    using local_matrix_type = gko::matrix::Csr<ValueType, LocalIndexType>;
    using schwarz_type = gko_dist::preconditioner::Schwarz<ValueType, LocalIndexType, GlobalIndexType>;

    /// Constructor
    ///
    /// @param exec Ginkgo executor (determines where computation runs)
    /// @param comm Ginkgo MPI communicator wrapper
    /// @param config Solver configuration
    DistributedSolver(std::shared_ptr<gko::Executor> exec,
                      std::shared_ptr<gko::experimental::mpi::communicator> comm,
                      const SolverConfig& config = SolverConfig{})
        : exec_(exec), comm_(comm), config_(config) {}

    /// Set the system matrix
    ///
    /// This triggers preconditioner and solver setup. Call this once
    /// before solving, or when the matrix sparsity pattern changes.
    ///
    /// @param A Ginkgo distributed matrix
    void set_operator(std::shared_ptr<matrix_type> A) {
        A_ = A;
        build_solver();
    }

    /// Update convergence tolerances
    ///
    /// @param rtol Relative tolerance
    /// @param atol Absolute tolerance (default: 1e-12)
    void set_tolerance(ValueType rtol, ValueType atol = 1e-12) {
        config_.rtol = rtol;
        config_.atol = atol;
        if (A_) {
            build_solver();  // Rebuild with new tolerances
        }
    }

    /// Update maximum iterations
    ///
    /// @param max_iter Maximum number of iterations
    void set_max_iterations(int max_iter) {
        config_.max_iterations = max_iter;
        if (A_) {
            build_solver();
        }
    }

    /// Solve Ax = b
    ///
    /// @param b Right-hand side vector
    /// @param x Solution vector (initial guess on entry, solution on exit)
    /// @return Number of iterations (negative if not converged)
    int solve(const vector_type& b, vector_type& x) {
        if (!solver_) {
            throw std::runtime_error("Solver not initialized. Call set_operator() first.");
        }

        // Reset convergence state
        last_converged_ = false;
        last_iterations_ = 0;
        last_residual_ = 0;

        // Create a copy of x as initial guess (Ginkgo modifies in place)
        auto x_copy = x.clone();

        // Apply the solver
        solver_->apply(&b, x_copy.get());

        // Copy result back to x
        x.copy_from(x_copy.get());

        // Extract convergence information from the logger
        if (logger_) {
            last_iterations_ = static_cast<int>(logger_->get_num_iterations());

            // Check if converged (iteration count < max means converged)
            last_converged_ = (last_iterations_ < config_.max_iterations);

            // Get residual norm if available
            auto residual_norms = logger_->get_residual_norm();
            if (residual_norms && residual_norms->get_size()[0] > 0) {
                auto host_residual = gko::clone(exec_->get_master(), residual_norms);
                auto dense_residual = gko::as<gko::matrix::Dense<ValueType>>(host_residual.get());
                last_residual_ = dense_residual->at(0, 0);
            }
        }

        if (config_.verbose && comm_->rank() == 0) {
            std::cout << "Ginkgo solver: " << last_iterations_ << " iterations, "
                      << "residual = " << last_residual_
                      << (last_converged_ ? " (converged)" : " (NOT converged)")
                      << std::endl;
        }

        return last_converged_ ? last_iterations_ : -last_iterations_;
    }

    /// Get iteration count from last solve
    int iterations() const { return last_iterations_; }

    /// Get final residual norm from last solve
    ValueType residual_norm() const { return last_residual_; }

    /// Check if last solve converged
    bool converged() const { return last_converged_; }

    /// Get the solver configuration
    const SolverConfig& config() const { return config_; }

private:
    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<gko::experimental::mpi::communicator> comm_;
    SolverConfig config_;

    std::shared_ptr<matrix_type> A_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<ValueType>> logger_;

    int last_iterations_ = 0;
    ValueType last_residual_ = 0;
    bool last_converged_ = false;

    /// Build the preconditioner based on configuration
    std::shared_ptr<gko::LinOpFactory> build_preconditioner_factory() {
        switch (config_.preconditioner) {
            case PreconditionerType::NONE:
                return nullptr;

            case PreconditionerType::JACOBI: {
                auto local_factory = gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                    .with_max_block_size(1u)
                    .on(exec_);
                return schwarz_type::build()
                    .with_local_solver(std::move(local_factory))
                    .on(exec_);
            }

            case PreconditionerType::BLOCK_JACOBI: {
                auto local_factory = gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                    .with_max_block_size(config_.jacobi_block_size)
                    .on(exec_);
                return schwarz_type::build()
                    .with_local_solver(std::move(local_factory))
                    .on(exec_);
            }

            case PreconditionerType::ILU: {
                auto local_factory = gko::factorization::ParIlu<ValueType, LocalIndexType>::build()
                    .on(exec_);
                return schwarz_type::build()
                    .with_local_solver(std::move(local_factory))
                    .on(exec_);
            }

            case PreconditionerType::IC: {
                auto local_factory = gko::factorization::ParIc<ValueType, LocalIndexType>::build()
                    .on(exec_);
                return schwarz_type::build()
                    .with_local_solver(std::move(local_factory))
                    .on(exec_);
            }

            case PreconditionerType::ISAI: {
                auto local_factory = gko::preconditioner::Isai<
                    gko::preconditioner::isai_type::lower, ValueType, LocalIndexType>::build()
                    .on(exec_);
                return schwarz_type::build()
                    .with_local_solver(std::move(local_factory))
                    .on(exec_);
            }

            case PreconditionerType::AMG:
                return build_amg_factory();

            default:
                return nullptr;
        }
    }

    /// Build AMG preconditioner factory for distributed matrices
    /// Based on Ginkgo's distributed-multigrid-preconditioned-solver example
    std::shared_ptr<gko::LinOpFactory> build_amg_factory() {
        const auto& amg_cfg = config_.amg;

        // Build local smoother factory (wrapped in Schwarz for distributed)
        std::shared_ptr<gko::LinOpFactory> schwarz_smoother;
        switch (amg_cfg.smoother) {
            case AMGConfig::Smoother::JACOBI: {
                schwarz_smoother = gko::share(schwarz_type::build()
                    .with_local_solver(
                        gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
                            .with_max_block_size(1u))
                    .on(exec_));
                break;
            }
            case AMGConfig::Smoother::GAUSS_SEIDEL: {
                schwarz_smoother = gko::share(schwarz_type::build()
                    .with_local_solver(
                        gko::solver::LowerTrs<ValueType, LocalIndexType>::build())
                    .on(exec_));
                break;
            }
            case AMGConfig::Smoother::ILU: {
                schwarz_smoother = gko::share(schwarz_type::build()
                    .with_local_solver(
                        gko::factorization::ParIlu<ValueType, LocalIndexType>::build())
                    .on(exec_));
                break;
            }
        }

        // Use build_smoother helper for proper smoother setup
        auto smoother_factory = gko::share(gko::solver::build_smoother(
            schwarz_smoother,
            static_cast<gko::size_type>(amg_cfg.pre_smooth_steps),
            static_cast<ValueType>(amg_cfg.relaxation_factor)));

        // Build coarse solver factory - must work with distributed matrices
        // Direct solvers don't work, so we use iterative solvers with limited iterations
        std::shared_ptr<gko::LinOpFactory> coarse_factory;
        switch (amg_cfg.coarse_solver) {
            case AMGConfig::CoarseSolver::DIRECT:
                // Direct solver not supported for distributed - fall back to CG
                [[fallthrough]];
            case AMGConfig::CoarseSolver::CG: {
                coarse_factory = gko::share(gko::solver::Cg<ValueType>::build()
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(static_cast<gko::size_type>(amg_cfg.coarse_max_iterations))
                            .on(exec_))
                    .on(exec_));
                break;
            }
            case AMGConfig::CoarseSolver::GMRES: {
                coarse_factory = gko::share(gko::solver::Gmres<ValueType>::build()
                    .with_krylov_dim(30u)
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(static_cast<gko::size_type>(amg_cfg.coarse_max_iterations))
                            .on(exec_))
                    .on(exec_));
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
        // When used as preconditioner, typically run 1 iteration
        return gko::solver::Multigrid::build()
            .with_mg_level(gko::multigrid::Pgm<ValueType, LocalIndexType>::build()
                .with_deterministic(amg_cfg.deterministic))
            .with_pre_smoother(smoother_factory)
            .with_coarsest_solver(coarse_factory)
            .with_max_levels(static_cast<gko::size_type>(amg_cfg.max_levels))
            .with_min_coarse_rows(static_cast<gko::size_type>(amg_cfg.min_coarse_rows))
            .with_cycle(cycle_type)
            .with_criteria(gko::stop::Iteration::build()
                .with_max_iters(1u)
                .on(exec_))
            .on(exec_);
    }

    /// Build the complete solver with preconditioner
    void build_solver() {
        if (!A_) {
            throw std::runtime_error("Matrix not set. Call set_operator() first.");
        }

        // Build convergence criteria (use gko::share to allow reuse)
        auto iter_stop = gko::share(gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(config_.max_iterations))
            .on(exec_));

        auto rel_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
            .with_baseline(gko::stop::mode::rhs_norm)
            .with_reduction_factor(config_.rtol)
            .on(exec_));

        auto abs_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
            .with_baseline(gko::stop::mode::absolute)
            .with_reduction_factor(config_.atol)
            .on(exec_));

        // Create convergence logger
        logger_ = gko::log::Convergence<ValueType>::create();

        // Build preconditioner factory
        auto precond_factory = build_preconditioner_factory();

        // Build solver factory based on type
        std::shared_ptr<gko::LinOpFactory> solver_factory;

        switch (config_.solver) {
            case SolverType::CG: {
                auto builder = gko::solver::Cg<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop);
                if (precond_factory) {
                    solver_factory = builder.with_preconditioner(precond_factory).on(exec_);
                } else {
                    solver_factory = builder.on(exec_);
                }
                break;
            }

            case SolverType::FCG: {
                auto builder = gko::solver::Fcg<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop);
                if (precond_factory) {
                    solver_factory = builder.with_preconditioner(precond_factory).on(exec_);
                } else {
                    solver_factory = builder.on(exec_);
                }
                break;
            }

            case SolverType::GMRES: {
                auto builder = gko::solver::Gmres<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop)
                    .with_krylov_dim(static_cast<gko::size_type>(config_.krylov_dim));
                if (precond_factory) {
                    solver_factory = builder.with_preconditioner(precond_factory).on(exec_);
                } else {
                    solver_factory = builder.on(exec_);
                }
                break;
            }

            case SolverType::BICGSTAB: {
                auto builder = gko::solver::Bicgstab<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop);
                if (precond_factory) {
                    solver_factory = builder.with_preconditioner(precond_factory).on(exec_);
                } else {
                    solver_factory = builder.on(exec_);
                }
                break;
            }

            case SolverType::CGS: {
                auto builder = gko::solver::Cgs<ValueType>::build()
                    .with_criteria(iter_stop, rel_stop, abs_stop);
                if (precond_factory) {
                    solver_factory = builder.with_preconditioner(precond_factory).on(exec_);
                } else {
                    solver_factory = builder.on(exec_);
                }
                break;
            }
        }

        // Generate the solver from the matrix
        solver_ = solver_factory->generate(A_);

        // Add convergence logger to the solver
        solver_->add_logger(logger_);
    }
};

} // namespace dolfinx_ginkgo
