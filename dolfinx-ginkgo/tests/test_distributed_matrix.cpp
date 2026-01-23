// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file test_distributed_matrix.cpp
/// @brief Integration test for distributed matrix conversion
///
/// Run with: mpirun -n 2 ./test_distributed_matrix

#include <dolfinx_ginkgo/ginkgo.h>

#include <petscmat.h>
#include <petscvec.h>
#include <mpi.h>

#include <iostream>
#include <cmath>
#include <cassert>

namespace dgko = dolfinx_ginkgo;

/// Create a simple test matrix: tridiagonal with -1, 2, -1
/// This is the 1D Laplacian discretization
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

/// Create a test vector with values 1, 2, 3, ...
Vec create_test_vector(MPI_Comm comm, PetscInt global_size)
{
    Vec v;
    PetscInt local_start, local_end;

    VecCreate(comm, &v);
    VecSetSizes(v, PETSC_DECIDE, global_size);
    VecSetType(v, VECMPI);
    VecSetUp(v);

    VecGetOwnershipRange(v, &local_start, &local_end);

    for (PetscInt i = local_start; i < local_end; ++i) {
        VecSetValue(v, i, static_cast<PetscScalar>(i + 1), INSERT_VALUES);
    }

    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    return v;
}

int main(int argc, char* argv[])
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const PetscInt global_size = 100;

    if (rank == 0) {
        std::cout << "=== Test: Distributed Matrix Conversion ===" << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
        std::cout << "Global size: " << global_size << std::endl;
    }

    // Create PETSc matrix and vector
    Mat A = create_test_matrix(comm, global_size);
    Vec b = create_test_vector(comm, global_size);
    Vec x;
    VecDuplicate(b, &x);
    VecSet(x, 0.0);

    // Get local sizes for verification
    PetscInt local_rows, local_cols;
    MatGetLocalSize(A, &local_rows, &local_cols);

    if (rank == 0) {
        std::cout << "\nPETSc matrix created successfully" << std::endl;
    }

    // Create Ginkgo executor and communicator
    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    if (rank == 0) {
        std::cout << "Ginkgo executor created (OpenMP backend)" << std::endl;
    }

    // Test 1: Extract CSR data from PETSc matrix
    {
        if (rank == 0) {
            std::cout << "\n--- Test 1: CSR Extraction ---" << std::endl;
        }

        auto csr = dgko::extract_petsc_csr<double, std::int32_t, std::int64_t>(A);

        // Verify dimensions
        assert(csr.local_rows == local_rows);
        assert(csr.global_rows == global_size);
        assert(csr.global_cols == global_size);

        // Verify row pointers size
        assert(static_cast<PetscInt>(csr.row_ptrs.size()) == local_rows + 1);

        // Verify NNZ (each row has 2-3 entries)
        PetscInt expected_nnz = 0;
        PetscInt row_start, row_end;
        MatGetOwnershipRange(A, &row_start, &row_end);
        for (PetscInt i = row_start; i < row_end; ++i) {
            expected_nnz += (i == 0 || i == global_size - 1) ? 2 : 3;
        }
        assert(static_cast<PetscInt>(csr.values.size()) == expected_nnz);

        std::cout << "  Rank " << rank << ": local_rows=" << csr.local_rows
                  << ", nnz=" << csr.values.size() << " [OK]" << std::endl;
    }

    // Test 2: Create Ginkgo distributed matrix
    {
        if (rank == 0) {
            std::cout << "\n--- Test 2: Ginkgo Matrix Creation ---" << std::endl;
        }

        auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);

        // Verify matrix was created
        assert(A_gko != nullptr);

        // Get local matrix and verify dimensions
        auto local_mat = A_gko->get_local_matrix();
        auto local_dim = local_mat->get_size();

        assert(static_cast<PetscInt>(local_dim[0]) == local_rows);

        std::cout << "  Rank " << rank << ": Ginkgo matrix created, local size="
                  << local_dim[0] << "x" << local_dim[1] << " [OK]" << std::endl;
    }

    // Test 3: Create Ginkgo distributed vector
    {
        if (rank == 0) {
            std::cout << "\n--- Test 3: Ginkgo Vector Creation ---" << std::endl;
        }

        auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

        // Verify vector was created
        assert(b_gko != nullptr);

        // Get local vector and verify size
        auto local_vec = b_gko->get_local_vector();
        auto local_size = local_vec->get_size()[0];

        PetscInt petsc_local_size;
        VecGetLocalSize(b, &petsc_local_size);

        assert(static_cast<PetscInt>(local_size) == petsc_local_size);

        std::cout << "  Rank " << rank << ": Ginkgo vector created, local size="
                  << local_size << " [OK]" << std::endl;
    }

    // Test 4: Vector round-trip (PETSc -> Ginkgo -> PETSc)
    {
        if (rank == 0) {
            std::cout << "\n--- Test 4: Vector Round-Trip ---" << std::endl;
        }

        // Create Ginkgo vector from PETSc
        auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

        // Create destination PETSc vector
        Vec b_copy;
        VecDuplicate(b, &b_copy);
        VecSet(b_copy, 0.0);

        // Copy back
        dgko::copy_to_petsc(*b_gko, b_copy);

        // Verify values match
        PetscScalar *b_arr, *copy_arr;
        VecGetArray(b, &b_arr);
        VecGetArray(b_copy, &copy_arr);

        PetscInt local_size;
        VecGetLocalSize(b, &local_size);

        double max_diff = 0.0;
        for (PetscInt i = 0; i < local_size; ++i) {
            double diff = std::abs(b_arr[i] - copy_arr[i]);
            max_diff = std::max(max_diff, diff);
        }

        VecRestoreArray(b, &b_arr);
        VecRestoreArray(b_copy, &copy_arr);
        VecDestroy(&b_copy);

        assert(max_diff < 1e-14);

        std::cout << "  Rank " << rank << ": Round-trip max diff=" << max_diff << " [OK]" << std::endl;
    }

    // Test 5: Create Ginkgo matrix from local COO data (new assembly path)
    {
        if (rank == 0) {
            std::cout << "\n--- Test 5: Local COO Assembly (communicate mode) ---" << std::endl;
        }

        // Build local COO data for the same tridiagonal matrix
        // Each rank contributes its local rows in global numbering
        PetscInt row_start, row_end;
        MatGetOwnershipRange(A, &row_start, &row_end);

        std::vector<std::int64_t> row_indices;
        std::vector<std::int64_t> col_indices;
        std::vector<double> values;

        for (std::int64_t i = row_start; i < row_end; ++i) {
            // Diagonal
            row_indices.push_back(i);
            col_indices.push_back(i);
            values.push_back(2.0);

            // Off-diagonal
            if (i > 0) {
                row_indices.push_back(i);
                col_indices.push_back(i - 1);
                values.push_back(-1.0);
            }
            if (i < global_size - 1) {
                row_indices.push_back(i);
                col_indices.push_back(i + 1);
                values.push_back(-1.0);
            }
        }

        // Build row_ranges for the partition
        std::vector<std::int64_t> row_ranges(size + 1);
        std::int64_t my_start = row_start;
        MPI_Allgather(&my_start, 1, MPI_INT64_T, row_ranges.data(), 1, MPI_INT64_T, comm);
        row_ranges[size] = global_size;

        // Create matrix using the new local COO function
        auto A_coo = dgko::create_distributed_matrix_from_local_coo<>(
            exec, gko_comm,
            row_indices, col_indices, values,
            static_cast<std::int64_t>(global_size),
            static_cast<std::int64_t>(global_size),
            row_ranges
        );

        // Verify matrix was created
        assert(A_coo != nullptr);

        // Verify dimensions match
        auto local_mat = A_coo->get_local_matrix();
        auto local_dim = local_mat->get_size();
        assert(static_cast<PetscInt>(local_dim[0]) == local_rows);

        std::cout << "  Rank " << rank << ": COO matrix created, local size="
                  << local_dim[0] << "x" << local_dim[1]
                  << ", local nnz=" << values.size() << " [OK]" << std::endl;

        // Test 5b: Verify SpMV gives same result as PETSc matrix
        // Create test vectors
        auto A_petsc = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);
        auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);

        // Result vectors
        auto y_petsc = x_gko->clone();
        auto y_coo = x_gko->clone();

        // SpMV with both matrices
        A_petsc->apply(x_gko, y_petsc);
        A_coo->apply(x_gko, y_coo);

        // Compare results
        auto y_petsc_local = y_petsc->get_local_vector();
        auto y_coo_local = y_coo->get_local_vector();

        auto y_petsc_vals = y_petsc_local->get_const_values();
        auto y_coo_vals = y_coo_local->get_const_values();

        double max_diff = 0.0;
        for (size_t i = 0; i < y_petsc_local->get_size()[0]; ++i) {
            double diff = std::abs(y_petsc_vals[i] - y_coo_vals[i]);
            max_diff = std::max(max_diff, diff);
        }

        assert(max_diff < 1e-12);

        std::cout << "  Rank " << rank << ": SpMV comparison max diff=" << max_diff << " [OK]" << std::endl;
    }

    // Cleanup
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);

    MPI_Barrier(comm);

    if (rank == 0) {
        std::cout << "\n=== All tests passed! ===" << std::endl;
    }

    PetscFinalize();
    return 0;
}
