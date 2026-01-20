// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file test_solver.cpp
/// @brief Integration test for distributed solver
///
/// Run with: mpirun -n 2 ./test_solver

#include <dolfinx_ginkgo/ginkgo.h>

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <mpi.h>

#include <iostream>
#include <cmath>
#include <cassert>

namespace dgko = dolfinx_ginkgo;

/// Create SPD test matrix: tridiagonal with 2 on diagonal, -1 off-diagonal
/// This is the 1D Laplacian discretization (SPD for CG)
Mat create_test_matrix(MPI_Comm comm, PetscInt global_size)
{
    Mat A;
    PetscInt local_start, local_end;

    MatCreate(comm, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, global_size, global_size);
    MatSetType(A, MATMPIAIJ);
    MatMPIAIJSetPreallocation(A, 3, nullptr, 2, nullptr);
    MatSetUp(A);

    MatGetOwnershipRange(A, &local_start, &local_end);

    for (PetscInt i = local_start; i < local_end; ++i) {
        // Diagonal
        PetscScalar diag = 2.0;
        MatSetValue(A, i, i, diag, INSERT_VALUES);

        // Off-diagonal
        if (i > 0) {
            MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES);
        }
        if (i < global_size - 1) {
            MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    return A;
}

/// Create RHS vector such that solution x = [1, 1, ..., 1]
/// For the tridiagonal matrix, b = A * ones = [1, 0, 0, ..., 0, 0, 1]
Vec create_rhs_vector(MPI_Comm comm, PetscInt global_size)
{
    Vec b;
    PetscInt local_start, local_end;

    VecCreate(comm, &b);
    VecSetSizes(b, PETSC_DECIDE, global_size);
    VecSetType(b, VECMPI);
    VecSetUp(b);

    VecGetOwnershipRange(b, &local_start, &local_end);

    // b = [1, 0, 0, ..., 0, 0, 1] for x = [1, 1, ..., 1]
    for (PetscInt i = local_start; i < local_end; ++i) {
        PetscScalar val = (i == 0 || i == global_size - 1) ? 1.0 : 0.0;
        VecSetValue(b, i, val, INSERT_VALUES);
    }

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    return b;
}

/// Compute solution error: ||x - x_exact||_inf where x_exact = [1, 1, ..., 1]
double compute_error(Vec x)
{
    PetscInt local_start, local_end;
    VecGetOwnershipRange(x, &local_start, &local_end);

    PetscScalar* arr;
    VecGetArray(x, &arr);

    double local_max = 0.0;
    for (PetscInt i = 0; i < local_end - local_start; ++i) {
        double err = std::abs(arr[i] - 1.0);
        local_max = std::max(local_max, err);
    }

    VecRestoreArray(x, &arr);

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_max;
}

/// Test CG solver with Jacobi preconditioner
void test_cg_jacobi(MPI_Comm comm, Mat A, Vec b, Vec x)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n--- Test: CG + Jacobi ---" << std::endl;
    }

    // Create Ginkgo executor and communicator
    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    // Convert matrix and vectors to Ginkgo format
    auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
    auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

    // Create solution vector (zero initial guess)
    VecSet(x, 0.0);
    auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);

    // Configure solver
    dgko::SolverConfig config;
    config.solver = dgko::SolverType::CG;
    config.preconditioner = dgko::PreconditionerType::JACOBI;
    config.rtol = 1e-10;
    config.max_iterations = 500;
    config.verbose = (rank == 0);

    // Create and run solver
    dgko::DistributedSolver<> solver(exec, gko_comm, config);
    solver.set_operator(A_gko);
    int iters = solver.solve(*b_gko, *x_gko);

    // Copy solution back to PETSc
    dgko::copy_to_petsc(*x_gko, x);

    // Compute error
    double error = compute_error(x);

    if (rank == 0) {
        std::cout << "  Iterations: " << std::abs(iters) << std::endl;
        std::cout << "  Converged: " << (solver.converged() ? "yes" : "no") << std::endl;
        std::cout << "  Solution error: " << error << std::endl;
    }

    assert(solver.converged());
    assert(error < 1e-8);

    if (rank == 0) {
        std::cout << "  [OK]" << std::endl;
    }
}

/// Test CG solver with Block Jacobi preconditioner
void test_cg_block_jacobi(MPI_Comm comm, Mat A, Vec b, Vec x)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n--- Test: CG + Block Jacobi ---" << std::endl;
    }

    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
    auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

    VecSet(x, 0.0);
    auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);

    dgko::SolverConfig config;
    config.solver = dgko::SolverType::CG;
    config.preconditioner = dgko::PreconditionerType::BLOCK_JACOBI;
    config.jacobi_block_size = 8;
    config.rtol = 1e-10;
    config.max_iterations = 500;
    config.verbose = (rank == 0);

    dgko::DistributedSolver<> solver(exec, gko_comm, config);
    solver.set_operator(A_gko);
    int iters = solver.solve(*b_gko, *x_gko);

    dgko::copy_to_petsc(*x_gko, x);
    double error = compute_error(x);

    if (rank == 0) {
        std::cout << "  Iterations: " << std::abs(iters) << std::endl;
        std::cout << "  Converged: " << (solver.converged() ? "yes" : "no") << std::endl;
        std::cout << "  Solution error: " << error << std::endl;
    }

    assert(solver.converged());
    assert(error < 1e-8);

    if (rank == 0) {
        std::cout << "  [OK]" << std::endl;
    }
}

/// Test GMRES solver with ILU preconditioner
void test_gmres_ilu(MPI_Comm comm, Mat A, Vec b, Vec x)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n--- Test: GMRES + ILU ---" << std::endl;
    }

    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
    auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

    VecSet(x, 0.0);
    auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);

    dgko::SolverConfig config;
    config.solver = dgko::SolverType::GMRES;
    config.preconditioner = dgko::PreconditionerType::ILU;
    config.krylov_dim = 50;
    config.rtol = 1e-10;
    config.max_iterations = 500;
    config.verbose = (rank == 0);

    dgko::DistributedSolver<> solver(exec, gko_comm, config);
    solver.set_operator(A_gko);
    int iters = solver.solve(*b_gko, *x_gko);

    dgko::copy_to_petsc(*x_gko, x);
    double error = compute_error(x);

    if (rank == 0) {
        std::cout << "  Iterations: " << std::abs(iters) << std::endl;
        std::cout << "  Converged: " << (solver.converged() ? "yes" : "no") << std::endl;
        std::cout << "  Solution error: " << error << std::endl;
    }

    assert(solver.converged());
    assert(error < 1e-8);

    if (rank == 0) {
        std::cout << "  [OK]" << std::endl;
    }
}

/// Test CG solver without preconditioner
void test_cg_none(MPI_Comm comm, Mat A, Vec b, Vec x)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n--- Test: CG (no preconditioner) ---" << std::endl;
    }

    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
    auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

    VecSet(x, 0.0);
    auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);

    dgko::SolverConfig config;
    config.solver = dgko::SolverType::CG;
    config.preconditioner = dgko::PreconditionerType::NONE;
    config.rtol = 1e-10;
    config.max_iterations = 1000;
    config.verbose = (rank == 0);

    dgko::DistributedSolver<> solver(exec, gko_comm, config);
    solver.set_operator(A_gko);
    int iters = solver.solve(*b_gko, *x_gko);

    dgko::copy_to_petsc(*x_gko, x);
    double error = compute_error(x);

    if (rank == 0) {
        std::cout << "  Iterations: " << std::abs(iters) << std::endl;
        std::cout << "  Converged: " << (solver.converged() ? "yes" : "no") << std::endl;
        std::cout << "  Solution error: " << error << std::endl;
    }

    assert(solver.converged());
    assert(error < 1e-8);

    if (rank == 0) {
        std::cout << "  [OK]" << std::endl;
    }
}

/// Compare Ginkgo solver to PETSc KSP
void test_vs_petsc(MPI_Comm comm, Mat A, Vec b, Vec x)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n--- Test: Ginkgo vs PETSc comparison ---" << std::endl;
    }

    // Solve with PETSc
    KSP ksp;
    PC pc;

    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCJACOBI);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 500);
    KSPSetUp(ksp);

    Vec x_petsc;
    VecDuplicate(b, &x_petsc);
    VecSet(x_petsc, 0.0);

    KSPSolve(ksp, b, x_petsc);

    PetscInt petsc_iters;
    KSPGetIterationNumber(ksp, &petsc_iters);
    double petsc_error = compute_error(x_petsc);

    // Solve with Ginkgo
    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
    auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

    VecSet(x, 0.0);
    auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);

    dgko::SolverConfig config;
    config.solver = dgko::SolverType::CG;
    config.preconditioner = dgko::PreconditionerType::JACOBI;
    config.rtol = 1e-10;
    config.max_iterations = 500;

    dgko::DistributedSolver<> solver(exec, gko_comm, config);
    solver.set_operator(A_gko);
    solver.solve(*b_gko, *x_gko);
    dgko::copy_to_petsc(*x_gko, x);

    int gko_iters = solver.iterations();
    double gko_error = compute_error(x);

    if (rank == 0) {
        std::cout << "  PETSc:  " << petsc_iters << " iterations, error = " << petsc_error << std::endl;
        std::cout << "  Ginkgo: " << gko_iters << " iterations, error = " << gko_error << std::endl;
    }

    // Both should converge to the same solution
    assert(petsc_error < 1e-8);
    assert(gko_error < 1e-8);

    // Iteration counts should be similar (not necessarily identical due to implementation differences)
    assert(std::abs(petsc_iters - gko_iters) < 20);

    if (rank == 0) {
        std::cout << "  [OK]" << std::endl;
    }

    VecDestroy(&x_petsc);
    KSPDestroy(&ksp);
}

int main(int argc, char* argv[])
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const PetscInt global_size = 200;

    if (rank == 0) {
        std::cout << "=== Test: Distributed Solver ===" << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
        std::cout << "Global size: " << global_size << std::endl;
    }

    // Create test problem
    Mat A = create_test_matrix(comm, global_size);
    Vec b = create_rhs_vector(comm, global_size);
    Vec x;
    VecDuplicate(b, &x);

    // Run tests
    test_cg_none(comm, A, b, x);
    test_cg_jacobi(comm, A, b, x);
    test_cg_block_jacobi(comm, A, b, x);
    test_gmres_ilu(comm, A, b, x);
    test_vs_petsc(comm, A, b, x);

    // Cleanup
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);

    MPI_Barrier(comm);

    if (rank == 0) {
        std::cout << "\n=== All solver tests passed! ===" << std::endl;
    }

    PetscFinalize();
    return 0;
}
