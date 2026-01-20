// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file DistributedMatrix.h
/// @brief Ginkgo distributed matrix creation from PETSc matrices

#include <ginkgo/ginkgo.hpp>
#include <petscmat.h>
#include <mpi.h>

#include <memory>

#include "convert.h"
#include "Partition.h"

namespace dolfinx_ginkgo {

namespace gko_dist = gko::experimental::distributed;

/// Create a Ginkgo distributed matrix from a PETSc MPIAIJ matrix
///
/// This function:
/// 1. Extracts CSR data from the PETSc matrix (diagonal + off-diagonal blocks)
/// 2. Creates a Ginkgo partition from the row distribution
/// 3. Builds a Ginkgo distributed::Matrix with the CSR data
///
/// The resulting matrix can be used with Ginkgo's distributed Krylov solvers.
///
/// @tparam ValueType Matrix value type (typically double)
/// @tparam LocalIndexType Local index type (typically int32_t)
/// @tparam GlobalIndexType Global index type (typically int64_t)
///
/// @param exec Ginkgo executor (determines where computation happens)
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param petsc_mat PETSc matrix (must be assembled, type MPIAIJ or SEQAIJ)
///
/// @return Shared pointer to the Ginkgo distributed matrix
///
/// @note The matrix data is copied to Ginkgo's internal storage. The original
///       PETSc matrix can be safely modified or destroyed after this call.
///
/// @note For GPU executors, the matrix data is transferred to device memory.
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_distributed_matrix_from_petsc(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    Mat petsc_mat)
{
    using matrix_type = gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using partition_type = gko_dist::Partition<LocalIndexType, GlobalIndexType>;

    MPI_Comm comm = gko_comm->get();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Extract CSR data from PETSc matrix
    auto csr_data = extract_petsc_csr<ValueType, LocalIndexType, GlobalIndexType>(petsc_mat);

    // Build partition from row distribution
    std::vector<GlobalIndexType> ranges(size + 1);
    GlobalIndexType my_row_start = csr_data.row_start;

    MPI_Allgather(&my_row_start, 1, MPI_INT64_T,
                  ranges.data(), 1, MPI_INT64_T, comm);
    ranges[size] = csr_data.global_rows;

    auto row_partition = create_partition_from_ranges<LocalIndexType, GlobalIndexType>(
        exec->get_master(), ranges
    );
    auto col_partition = row_partition;  // Square matrix

    // Convert CSR to COO format (triplets) for read_distributed
    // Build matrix_data with global row indices
    std::vector<GlobalIndexType> coo_rows;
    std::vector<GlobalIndexType> coo_cols;
    std::vector<ValueType> coo_vals;

    size_t nnz = csr_data.values.size();
    coo_rows.reserve(nnz);
    coo_cols.reserve(nnz);
    coo_vals.reserve(nnz);

    for (LocalIndexType local_row = 0; local_row < csr_data.local_rows; ++local_row) {
        GlobalIndexType global_row = csr_data.row_start + local_row;
        for (LocalIndexType j = csr_data.row_ptrs[local_row]; j < csr_data.row_ptrs[local_row + 1]; ++j) {
            coo_rows.push_back(global_row);
            coo_cols.push_back(csr_data.col_idxs[j]);
            coo_vals.push_back(csr_data.values[j]);
        }
    }

    // Create matrix_data structure
    gko::matrix_data<ValueType, GlobalIndexType> mat_data{
        gko::dim<2>{static_cast<size_t>(csr_data.global_rows),
                   static_cast<size_t>(csr_data.global_cols)}
    };
    mat_data.nonzeros.reserve(nnz);
    for (size_t i = 0; i < nnz; ++i) {
        mat_data.nonzeros.emplace_back(coo_rows[i], coo_cols[i], coo_vals[i]);
    }

    // Create empty distributed matrix and fill via read_distributed
    auto dist_mat = matrix_type::create(exec, *gko_comm);
    dist_mat->read_distributed(mat_data, row_partition, col_partition);

    return dist_mat;
}

/// Create a Ginkgo distributed matrix from CSR data
///
/// This is a lower-level function for when you already have CSR data
/// (e.g., from DOLFINx's native MatrixCSR).
///
/// @tparam ValueType Matrix value type
/// @tparam LocalIndexType Local index type
/// @tparam GlobalIndexType Global index type
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param row_ptrs CSR row pointers (local, size local_rows+1)
/// @param col_idxs CSR column indices (global)
/// @param values CSR values
/// @param local_rows Number of local rows
/// @param global_rows Total number of rows
/// @param global_cols Total number of columns
/// @param row_start First global row index on this rank
///
/// @return Shared pointer to the Ginkgo distributed matrix
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_distributed_matrix_from_csr(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    const std::vector<LocalIndexType>& row_ptrs,
    const std::vector<GlobalIndexType>& col_idxs,
    const std::vector<ValueType>& values,
    LocalIndexType local_rows,
    GlobalIndexType global_rows,
    GlobalIndexType global_cols,
    GlobalIndexType row_start)
{
    using matrix_type = gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>;

    MPI_Comm comm = gko_comm->get();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Build partition from row distribution
    std::vector<GlobalIndexType> ranges(size + 1);
    MPI_Allgather(&row_start, 1, MPI_INT64_T,
                  ranges.data(), 1, MPI_INT64_T, comm);
    ranges[size] = global_rows;

    auto row_partition = create_partition_from_ranges<LocalIndexType, GlobalIndexType>(
        exec->get_master(), ranges
    );
    auto col_partition = row_partition;  // Square matrix

    // Convert CSR to COO format (triplets) for read_distributed
    std::vector<GlobalIndexType> coo_rows;
    std::vector<GlobalIndexType> coo_cols;
    std::vector<ValueType> coo_vals;

    size_t nnz = values.size();
    coo_rows.reserve(nnz);
    coo_cols.reserve(nnz);
    coo_vals.reserve(nnz);

    for (LocalIndexType local_row = 0; local_row < local_rows; ++local_row) {
        GlobalIndexType global_row = row_start + local_row;
        for (LocalIndexType j = row_ptrs[local_row]; j < row_ptrs[local_row + 1]; ++j) {
            coo_rows.push_back(global_row);
            coo_cols.push_back(col_idxs[j]);
            coo_vals.push_back(values[j]);
        }
    }

    // Create matrix_data structure
    gko::matrix_data<ValueType, GlobalIndexType> mat_data{
        gko::dim<2>{static_cast<size_t>(global_rows), static_cast<size_t>(global_cols)}
    };
    mat_data.nonzeros.reserve(nnz);
    for (size_t i = 0; i < nnz; ++i) {
        mat_data.nonzeros.emplace_back(coo_rows[i], coo_cols[i], coo_vals[i]);
    }

    // Create empty distributed matrix and fill via read_distributed
    auto dist_mat = matrix_type::create(exec, *gko_comm);
    dist_mat->read_distributed(mat_data, row_partition, col_partition);

    return dist_mat;
}

/// Update values of an existing Ginkgo distributed matrix from PETSc
///
/// This is more efficient than recreating the matrix when only values change
/// (same sparsity pattern). Useful in time-stepping where the matrix structure
/// is fixed but values are updated.
///
/// @tparam ValueType Matrix value type
/// @tparam LocalIndexType Local index type
/// @tparam GlobalIndexType Global index type
///
/// @param gko_mat Ginkgo distributed matrix to update
/// @param petsc_mat Source PETSc matrix with new values
///
/// @note The sparsity pattern must match exactly. This function does not
///       verify pattern compatibility.
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
void update_matrix_values_from_petsc(
    std::shared_ptr<gko_dist::Matrix<ValueType, LocalIndexType, GlobalIndexType>> gko_mat,
    Mat petsc_mat)
{
    // Extract CSR data
    auto csr_data = extract_petsc_csr<ValueType, LocalIndexType, GlobalIndexType>(petsc_mat);

    // Get the local matrix from the distributed matrix
    auto local_mat = gko_mat->get_local_matrix();
    auto exec = gko_mat->get_executor();

    // Update values array
    // Note: This assumes the sparsity pattern hasn't changed
    auto values_view = local_mat->get_values();
    exec->copy_from(exec->get_master(),
                    csr_data.values.size(),
                    csr_data.values.data(),
                    values_view);
}

} // namespace dolfinx_ginkgo
