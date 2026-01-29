// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file test_dd_matrix.cpp
/// @brief Integration test for DdMatrix (domain decomposition matrix)
///
/// Run with: mpirun -n 2 ./test_dd_matrix
///
/// DdMatrix differs from the regular distributed matrix:
/// - Regular Matrix: Each global DOF is owned by exactly one rank
/// - DdMatrix: Interface DOFs can have contributions from multiple ranks,
///             with restriction/prolongation operators handling the overlap

#include <dolfinx_ginkgo/ginkgo.h>

#include <mpi.h>

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

namespace dgko = dolfinx_ginkgo;

/// Test 1: Create a simple DdMatrix and verify structure
///
/// We create a 2D Poisson-like matrix where each subdomain has overlapping
/// contributions at the interfaces.
void test_dd_matrix_creation(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    MPI_Comm comm,
    int rank,
    int size)
{
    if (rank == 0) {
        std::cout << "\n--- Test 1: DdMatrix Creation ---" << std::endl;
    }

    // Global matrix size
    const std::int64_t global_size = 10;

    // Build partition: divide rows evenly
    std::vector<std::int64_t> row_ranges(size + 1);
    std::int64_t rows_per_rank = global_size / size;
    for (int i = 0; i <= size; ++i) {
        row_ranges[i] = std::min(static_cast<std::int64_t>(i) * rows_per_rank, global_size);
    }
    row_ranges[size] = global_size;

    std::int64_t row_start = row_ranges[rank];
    std::int64_t row_end = row_ranges[rank + 1];
    std::int64_t local_rows = row_end - row_start;

    // Build local COO data for a tridiagonal matrix
    // In domain decomposition, each subdomain can contribute to interface DOFs
    std::vector<std::int64_t> row_indices;
    std::vector<std::int64_t> col_indices;
    std::vector<double> values;

    // Each rank contributes its local rows
    for (std::int64_t i = row_start; i < row_end; ++i) {
        // Diagonal: 2.0
        row_indices.push_back(i);
        col_indices.push_back(i);
        values.push_back(2.0);

        // Off-diagonal: -1.0
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

    // Create DdMatrix
    auto dd_mat = dgko::create_dd_matrix_from_local_coo<>(
        exec, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size,
        row_ranges
    );

    // Verify matrix was created
    assert(dd_mat != nullptr);

    // Check global dimensions
    auto dims = dd_mat->get_size();
    assert(dims[0] == static_cast<size_t>(global_size));
    assert(dims[1] == static_cast<size_t>(global_size));

    // Check local matrix exists
    auto local_mat = dd_mat->get_local_matrix();
    assert(local_mat != nullptr);

    // Check restriction operator exists
    auto restriction = dd_mat->get_restriction();
    assert(restriction != nullptr);

    // Check prolongation operator exists
    auto prolongation = dd_mat->get_prolongation();
    assert(prolongation != nullptr);

    std::cout << "  Rank " << rank << ": DdMatrix created, global size="
              << dims[0] << "x" << dims[1]
              << ", local nnz=" << values.size() << " [OK]" << std::endl;
}

/// Test 2: DdMatrix SpMV operation using the documentation example
///
/// From the DdMatrix documentation:
/// ```
/// Local Contribution on       Globally Assembled Matrix A
/// Rank 0        Rank 1
/// |  4 -2  0 |  |  0  0  0 |  |  4 -2  0 |
/// | -2  2  0 |  |  0  2 -2 |  | -2  4 -2 |
/// |  0  0  0 |  |  0 -2  4 |  |  0 -2  4 |
/// ```
/// With partition where rank 0 owns rows 0-1 and rank 1 owns row 2.
/// The interface DOF (row 1) has contributions from both ranks that sum to 4.
void test_dd_matrix_spmv(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    MPI_Comm comm,
    int rank,
    int size)
{
    if (rank == 0) {
        std::cout << "\n--- Test 2: DdMatrix SpMV (documentation example) ---" << std::endl;
    }

    if (size != 2) {
        if (rank == 0) {
            std::cout << "  Skipping (requires exactly 2 ranks)" << std::endl;
        }
        return;
    }

    // 3x3 global matrix, partition: rank 0 owns rows 0-1, rank 1 owns row 2
    const std::int64_t global_size = 3;
    std::vector<std::int64_t> row_ranges = {0, 2, 3};

    // Build local contributions according to the documentation example
    std::vector<std::int64_t> row_indices;
    std::vector<std::int64_t> col_indices;
    std::vector<double> values;

    if (rank == 0) {
        // Rank 0's local contribution (3x3 but only rows 0-1 have values):
        // |  4 -2  0 |
        // | -2  2  0 |
        // |  0  0  0 |
        // Row 0: (0,0)=4, (0,1)=-2
        row_indices.push_back(0); col_indices.push_back(0); values.push_back(4.0);
        row_indices.push_back(0); col_indices.push_back(1); values.push_back(-2.0);
        // Row 1: (1,0)=-2, (1,1)=2
        row_indices.push_back(1); col_indices.push_back(0); values.push_back(-2.0);
        row_indices.push_back(1); col_indices.push_back(1); values.push_back(2.0);
    } else {
        // Rank 1's local contribution (3x3 but only rows 1-2 have values):
        // |  0  0  0 |
        // |  0  2 -2 |
        // |  0 -2  4 |
        // Row 1: (1,1)=2, (1,2)=-2
        row_indices.push_back(1); col_indices.push_back(1); values.push_back(2.0);
        row_indices.push_back(1); col_indices.push_back(2); values.push_back(-2.0);
        // Row 2: (2,1)=-2, (2,2)=4
        row_indices.push_back(2); col_indices.push_back(1); values.push_back(-2.0);
        row_indices.push_back(2); col_indices.push_back(2); values.push_back(4.0);
    }

    // Create DdMatrix
    auto dd_mat = dgko::create_dd_matrix_from_local_coo<>(
        exec, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size,
        row_ranges
    );

    // Create test vector x = [1, 1, 1]
    auto partition = dgko::create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec->get_master(), row_ranges);

    auto x = dgko::gko_dist::Vector<double>::create(exec, *gko_comm);
    x->read_distributed(
        gko::matrix_data<double, std::int64_t>{
            gko::dim<2>{static_cast<size_t>(global_size), 1}},
        partition);
    x->fill(1.0);

    // Create result vector y
    auto y = dgko::gko_dist::Vector<double>::create(exec, *gko_comm);
    y->read_distributed(
        gko::matrix_data<double, std::int64_t>{
            gko::dim<2>{static_cast<size_t>(global_size), 1}},
        partition);

    // Apply: y = A * x
    // For globally assembled matrix A with x = [1, 1, 1]:
    // y[0] = 4*1 - 2*1 + 0*1 = 2
    // y[1] = -2*1 + 4*1 - 2*1 = 0
    // y[2] = 0*1 - 2*1 + 4*1 = 2
    dd_mat->apply(x, y);

    // Verify results
    auto y_local = y->get_local_vector();
    std::int64_t row_start = row_ranges[rank];
    std::int64_t row_end = row_ranges[rank + 1];

    double expected_values[] = {2.0, 0.0, 2.0};
    bool correct = true;

    for (std::int64_t i = row_start; i < row_end; ++i) {
        size_t local_idx = static_cast<size_t>(i - row_start);
        double expected = expected_values[i];
        double actual = y_local->at(local_idx, 0);

        if (std::abs(actual - expected) > 1e-10) {
            std::cerr << "  Rank " << rank << ": y[" << i << "] = " << actual
                      << ", expected " << expected << std::endl;
            correct = false;
        }
    }

    assert(correct);
    std::cout << "  Rank " << rank << ": SpMV verification [OK]" << std::endl;
}

/// Test 3: Compare DdMatrix with regular distributed Matrix
///
/// Both matrix types receive the same local COO data on each rank.
/// - Regular Matrix uses read_distributed with communicate mode (sums contributions)
/// - DdMatrix uses read_distributed (handles DD structure via R^T A_local R)
/// The SpMV results should be identical.
void test_dd_vs_regular_matrix(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    MPI_Comm comm,
    int rank,
    int size)
{
    if (rank == 0) {
        std::cout << "\n--- Test 3: DdMatrix vs Regular Matrix ---" << std::endl;
    }

    if (size != 2) {
        if (rank == 0) {
            std::cout << "  Skipping (requires exactly 2 ranks)" << std::endl;
        }
        return;
    }

    // Use the same 3x3 example from the documentation
    const std::int64_t global_size = 3;
    std::vector<std::int64_t> row_ranges = {0, 2, 3};

    // Build local contributions - same data on each rank as test 2
    std::vector<std::int64_t> row_indices;
    std::vector<std::int64_t> col_indices;
    std::vector<double> values;

    if (rank == 0) {
        // Rank 0's local contribution:
        // |  4 -2  0 |
        // | -2  2  0 |
        // |  0  0  0 |
        row_indices.push_back(0); col_indices.push_back(0); values.push_back(4.0);
        row_indices.push_back(0); col_indices.push_back(1); values.push_back(-2.0);
        row_indices.push_back(1); col_indices.push_back(0); values.push_back(-2.0);
        row_indices.push_back(1); col_indices.push_back(1); values.push_back(2.0);
    } else {
        // Rank 1's local contribution:
        // |  0  0  0 |
        // |  0  2 -2 |
        // |  0 -2  4 |
        row_indices.push_back(1); col_indices.push_back(1); values.push_back(2.0);
        row_indices.push_back(1); col_indices.push_back(2); values.push_back(-2.0);
        row_indices.push_back(2); col_indices.push_back(1); values.push_back(-2.0);
        row_indices.push_back(2); col_indices.push_back(2); values.push_back(4.0);
    }

    // Create DdMatrix using create_dd_matrix_from_local_coo
    auto dd_mat = dgko::create_dd_matrix_from_local_coo<>(
        exec, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size,
        row_ranges
    );

    // Create regular Matrix using create_distributed_matrix_from_local_coo
    // (uses assembly_mode::communicate internally)
    auto regular_mat = dgko::create_distributed_matrix_from_local_coo<>(
        exec, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size,
        row_ranges
    );

    auto partition = dgko::create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec->get_master(), row_ranges);

    // Create test vector x = [1, 1, 1]
    auto x = dgko::gko_dist::Vector<double>::create(exec, *gko_comm);
    x->read_distributed(
        gko::matrix_data<double, std::int64_t>{
            gko::dim<2>{static_cast<size_t>(global_size), 1}},
        partition);
    x->fill(1.0);

    // Create result vectors
    auto y_dd = x->clone();
    auto y_regular = x->clone();

    // Apply both matrices
    dd_mat->apply(x, y_dd);
    regular_mat->apply(x, y_regular);

    // Compare results
    auto y_dd_local = y_dd->get_local_vector();
    auto y_regular_local = y_regular->get_local_vector();

    double max_diff = 0.0;
    for (size_t i = 0; i < y_dd_local->get_size()[0]; ++i) {
        double diff = std::abs(y_dd_local->at(i, 0) - y_regular_local->at(i, 0));
        max_diff = std::max(max_diff, diff);
    }

    assert(max_diff < 1e-10);
    std::cout << "  Rank " << rank << ": DdMatrix vs Matrix max diff="
              << max_diff << " [OK]" << std::endl;
}

/// Test 4: DdMatrix with larger example
///
/// Tests a 5x5 tridiagonal matrix with proper DD local contributions.
/// Partition: rank 0 owns rows 0-2, rank 1 owns rows 3-4.
/// Interface is at row 2/3 boundary.
void test_dd_matrix_overlap(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    MPI_Comm comm,
    int rank,
    int size)
{
    if (rank == 0) {
        std::cout << "\n--- Test 4: DdMatrix with Larger Example ---" << std::endl;
    }

    if (size != 2) {
        if (rank == 0) {
            std::cout << "  Skipping (requires exactly 2 ranks)" << std::endl;
        }
        return;
    }

    // 5x5 global tridiagonal matrix: -1, 2, -1
    // Globally assembled:
    // |  2 -1  0  0  0 |
    // | -1  2 -1  0  0 |
    // |  0 -1  2 -1  0 |  <- interface row
    // |  0  0 -1  2 -1 |  <- interface row
    // |  0  0  0 -1  2 |
    //
    // Partition: rank 0 owns rows 0-2, rank 1 owns rows 3-4
    // Interface: rows 2 and 3 are coupled
    //
    // Local contributions following the DD pattern:
    // Rank 0 contributes to rows 0-2 (including interface coupling to row 3)
    // Rank 1 contributes to rows 2-4 (including interface coupling to row 2)
    const std::int64_t global_size = 5;
    std::vector<std::int64_t> row_ranges = {0, 3, 5};

    std::vector<std::int64_t> row_indices;
    std::vector<std::int64_t> col_indices;
    std::vector<double> values;

    if (rank == 0) {
        // Rank 0's local contribution (rows 0-2, but row 2 is interface):
        // Row 0: diag=2, off=-1
        row_indices.push_back(0); col_indices.push_back(0); values.push_back(2.0);
        row_indices.push_back(0); col_indices.push_back(1); values.push_back(-1.0);
        // Row 1: off=-1, diag=2, off=-1
        row_indices.push_back(1); col_indices.push_back(0); values.push_back(-1.0);
        row_indices.push_back(1); col_indices.push_back(1); values.push_back(2.0);
        row_indices.push_back(1); col_indices.push_back(2); values.push_back(-1.0);
        // Row 2 (interface): partial contribution, diag=1 (half of 2), off=-1 to row 1
        row_indices.push_back(2); col_indices.push_back(1); values.push_back(-1.0);
        row_indices.push_back(2); col_indices.push_back(2); values.push_back(1.0);
    } else {
        // Rank 1's local contribution (rows 3-4, but row 3 couples to row 2):
        // Row 2 (interface): partial contribution, diag=1 (half of 2), off=-1 to row 3
        row_indices.push_back(2); col_indices.push_back(2); values.push_back(1.0);
        row_indices.push_back(2); col_indices.push_back(3); values.push_back(-1.0);
        // Row 3: off=-1, diag=2, off=-1
        row_indices.push_back(3); col_indices.push_back(2); values.push_back(-1.0);
        row_indices.push_back(3); col_indices.push_back(3); values.push_back(2.0);
        row_indices.push_back(3); col_indices.push_back(4); values.push_back(-1.0);
        // Row 4: off=-1, diag=2
        row_indices.push_back(4); col_indices.push_back(3); values.push_back(-1.0);
        row_indices.push_back(4); col_indices.push_back(4); values.push_back(2.0);
    }

    // Create DdMatrix
    auto dd_mat = dgko::create_dd_matrix_from_local_coo<>(
        exec, gko_comm,
        row_indices, col_indices, values,
        global_size, global_size,
        row_ranges
    );

    // Verify the restriction and prolongation operators exist
    auto restriction = dd_mat->get_restriction();
    auto prolongation = dd_mat->get_prolongation();

    assert(restriction != nullptr);
    assert(prolongation != nullptr);

    // Test SpMV with x = [1, 1, 1, 1, 1]
    // Expected y = A*x:
    // y[0] = 2*1 - 1*1 = 1
    // y[1] = -1*1 + 2*1 - 1*1 = 0
    // y[2] = -1*1 + 2*1 - 1*1 = 0
    // y[3] = -1*1 + 2*1 - 1*1 = 0
    // y[4] = -1*1 + 2*1 = 1
    auto partition = dgko::create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec->get_master(), row_ranges);

    auto x = dgko::gko_dist::Vector<double>::create(exec, *gko_comm);
    x->read_distributed(
        gko::matrix_data<double, std::int64_t>{
            gko::dim<2>{static_cast<size_t>(global_size), 1}},
        partition);
    x->fill(1.0);

    auto y = x->clone();
    dd_mat->apply(x, y);

    // Verify results
    std::int64_t row_start = row_ranges[rank];
    std::int64_t row_end = row_ranges[rank + 1];
    auto y_local = y->get_local_vector();

    double expected_values[] = {1.0, 0.0, 0.0, 0.0, 1.0};
    bool correct = true;

    for (std::int64_t i = row_start; i < row_end; ++i) {
        size_t local_idx = static_cast<size_t>(i - row_start);
        double expected = expected_values[i];
        double actual = y_local->at(local_idx, 0);

        if (std::abs(actual - expected) > 1e-10) {
            std::cerr << "  Rank " << rank << ": y[" << i << "] = " << actual
                      << ", expected " << expected << std::endl;
            correct = false;
        }
    }

    assert(correct);
    std::cout << "  Rank " << rank << ": Larger example test [OK]" << std::endl;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "=== Test: DdMatrix (Domain Decomposition Matrix) ===" << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
    }

    // Create Ginkgo executor and communicator
    auto exec = dgko::create_executor(dgko::Backend::OMP, 0);
    auto gko_comm = dgko::create_communicator(comm);

    if (rank == 0) {
        std::cout << "Ginkgo executor created (OpenMP backend)" << std::endl;
    }

    // Run tests
    test_dd_matrix_creation(exec, gko_comm, comm, rank, size);
    test_dd_matrix_spmv(exec, gko_comm, comm, rank, size);
    test_dd_vs_regular_matrix(exec, gko_comm, comm, rank, size);
    test_dd_matrix_overlap(exec, gko_comm, comm, rank, size);

    MPI_Barrier(comm);

    if (rank == 0) {
        std::cout << "\n=== All DdMatrix tests passed! ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
