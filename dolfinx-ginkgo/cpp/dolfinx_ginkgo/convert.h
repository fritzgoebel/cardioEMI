// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file convert.h
/// @brief Conversion utilities between PETSc and Ginkgo data structures
///
/// This file provides functions to extract CSR data from PETSc matrices and
/// vectors, enabling conversion to Ginkgo distributed data structures.

#include <ginkgo/ginkgo.hpp>
#include <petscmat.h>
#include <petscvec.h>
#include <mpi.h>

#include <memory>
#include <vector>
#include <stdexcept>
#include <cassert>

namespace dolfinx_ginkgo {

namespace gko_dist = gko::experimental::distributed;

// ============================================================================
// PETSc Matrix CSR Extraction
// ============================================================================

/// Structure holding CSR data extracted from a PETSc matrix
///
/// For distributed (MPIAIJ) matrices, PETSc stores:
/// - Diagonal block: rows and columns owned by this rank
/// - Off-diagonal block: rows owned by this rank, columns owned by other ranks
///
/// We merge these into a single CSR structure with global column indices.
template<typename ValueType, typename LocalIndexType, typename GlobalIndexType>
struct PetscCSRData {
    std::vector<LocalIndexType> row_ptrs;    ///< Row pointers (size: local_rows + 1)
    std::vector<GlobalIndexType> col_idxs;   ///< Global column indices
    std::vector<ValueType> values;           ///< Non-zero values

    LocalIndexType local_rows = 0;           ///< Number of local rows
    GlobalIndexType global_rows = 0;         ///< Total number of rows
    GlobalIndexType global_cols = 0;         ///< Total number of columns
    GlobalIndexType row_start = 0;           ///< First global row index on this rank
};

/// Extract CSR data from a PETSc MPIAIJ matrix
///
/// PETSc stores MPIAIJ matrices as two blocks:
/// - A (diagonal): owns columns in [col_start, col_end)
/// - B (off-diagonal): owns columns outside that range
///
/// This function merges them into a single CSR with global column indices,
/// sorted by column index within each row.
///
/// @tparam ValueType Matrix value type (double)
/// @tparam LocalIndexType Local index type (int32_t)
/// @tparam GlobalIndexType Global index type (int64_t)
///
/// @param petsc_mat PETSc matrix (must be assembled, type MPIAIJ or SEQAIJ)
///
/// @return CSR data structure with global column indices
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
PetscCSRData<ValueType, LocalIndexType, GlobalIndexType>
extract_petsc_csr(Mat petsc_mat)
{
    PetscErrorCode ierr;
    PetscCSRData<ValueType, LocalIndexType, GlobalIndexType> csr;

    // Get matrix info
    PetscInt M, N, m, n;
    ierr = MatGetSize(petsc_mat, &M, &N);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = MatGetLocalSize(petsc_mat, &m, &n);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    csr.global_rows = static_cast<GlobalIndexType>(M);
    csr.global_cols = static_cast<GlobalIndexType>(N);
    csr.local_rows = static_cast<LocalIndexType>(m);

    // Get ownership range
    PetscInt row_start, row_end;
    ierr = MatGetOwnershipRange(petsc_mat, &row_start, &row_end);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    csr.row_start = static_cast<GlobalIndexType>(row_start);

    // Check matrix type
    MatType mat_type;
    ierr = MatGetType(petsc_mat, &mat_type);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    bool is_mpiaij = (strcmp(mat_type, MATMPIAIJ) == 0);
    bool is_seqaij = (strcmp(mat_type, MATSEQAIJ) == 0);

    if (!is_mpiaij && !is_seqaij) {
        throw std::runtime_error(
            "extract_petsc_csr: Unsupported matrix type. Expected MPIAIJ or SEQAIJ, got: " +
            std::string(mat_type)
        );
    }

    csr.row_ptrs.resize(m + 1);
    csr.row_ptrs[0] = 0;

    if (is_seqaij) {
        // Sequential matrix - simple extraction
        const PetscInt* ai;
        const PetscInt* aj;
        const PetscScalar* aa;
        PetscBool done;

        ierr = MatGetRowIJ(petsc_mat, 0, PETSC_FALSE, PETSC_FALSE, &m, &ai, &aj, &done);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        if (!done) {
            throw std::runtime_error("MatGetRowIJ failed for SEQAIJ matrix");
        }

        ierr = MatSeqAIJGetArrayRead(petsc_mat, &aa);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        PetscInt nnz = ai[m];
        csr.col_idxs.resize(nnz);
        csr.values.resize(nnz);

        for (PetscInt i = 0; i <= m; ++i) {
            csr.row_ptrs[i] = static_cast<LocalIndexType>(ai[i]);
        }

        for (PetscInt i = 0; i < nnz; ++i) {
            csr.col_idxs[i] = static_cast<GlobalIndexType>(aj[i]);
            csr.values[i] = static_cast<ValueType>(aa[i]);
        }

        ierr = MatSeqAIJRestoreArrayRead(petsc_mat, &aa);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        ierr = MatRestoreRowIJ(petsc_mat, 0, PETSC_FALSE, PETSC_FALSE, &m, &ai, &aj, &done);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

    } else {
        // MPIAIJ matrix - merge diagonal and off-diagonal blocks
        Mat A_diag, B_offdiag;
        const PetscInt* garray;  // Maps local off-diag columns to global indices

        ierr = MatMPIAIJGetSeqAIJ(petsc_mat, &A_diag, &B_offdiag, &garray);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        // Get column ownership range (for diagonal block)
        PetscInt col_start, col_end;
        ierr = MatGetOwnershipRangeColumn(petsc_mat, &col_start, &col_end);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        // Get CSR data from diagonal block
        const PetscInt *ai_d, *aj_d;
        const PetscScalar* aa_d;
        PetscBool done_d;
        PetscInt m_d;

        ierr = MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &m_d, &ai_d, &aj_d, &done_d);
        CHKERRABORT(PETSC_COMM_SELF, ierr);
        ierr = MatSeqAIJGetArrayRead(A_diag, &aa_d);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        // Get CSR data from off-diagonal block
        const PetscInt *ai_o, *aj_o;
        const PetscScalar* aa_o;
        PetscBool done_o;
        PetscInt m_o;

        ierr = MatGetRowIJ(B_offdiag, 0, PETSC_FALSE, PETSC_FALSE, &m_o, &ai_o, &aj_o, &done_o);
        CHKERRABORT(PETSC_COMM_SELF, ierr);
        ierr = MatSeqAIJGetArrayRead(B_offdiag, &aa_o);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        assert(m_d == m && m_o == m);

        // Count total non-zeros per row and allocate
        GlobalIndexType total_nnz = 0;
        for (PetscInt i = 0; i < m; ++i) {
            PetscInt nnz_row = (ai_d[i + 1] - ai_d[i]) + (ai_o[i + 1] - ai_o[i]);
            csr.row_ptrs[i + 1] = csr.row_ptrs[i] + static_cast<LocalIndexType>(nnz_row);
            total_nnz += nnz_row;
        }

        csr.col_idxs.resize(total_nnz);
        csr.values.resize(total_nnz);

        // Merge diagonal and off-diagonal entries for each row
        // We need to sort by global column index
        for (PetscInt i = 0; i < m; ++i) {
            LocalIndexType row_ptr = csr.row_ptrs[i];

            // Collect all entries for this row with global column indices
            std::vector<std::pair<GlobalIndexType, ValueType>> entries;
            entries.reserve(csr.row_ptrs[i + 1] - row_ptr);

            // Diagonal block entries (local columns map to [col_start, col_end))
            for (PetscInt k = ai_d[i]; k < ai_d[i + 1]; ++k) {
                GlobalIndexType global_col = static_cast<GlobalIndexType>(col_start + aj_d[k]);
                entries.emplace_back(global_col, static_cast<ValueType>(aa_d[k]));
            }

            // Off-diagonal block entries (use garray to map to global columns)
            for (PetscInt k = ai_o[i]; k < ai_o[i + 1]; ++k) {
                GlobalIndexType global_col = static_cast<GlobalIndexType>(garray[aj_o[k]]);
                entries.emplace_back(global_col, static_cast<ValueType>(aa_o[k]));
            }

            // Sort by column index
            std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            // Store in CSR arrays
            for (size_t k = 0; k < entries.size(); ++k) {
                csr.col_idxs[row_ptr + k] = entries[k].first;
                csr.values[row_ptr + k] = entries[k].second;
            }
        }

        // Restore arrays
        ierr = MatSeqAIJRestoreArrayRead(A_diag, &aa_d);
        CHKERRABORT(PETSC_COMM_SELF, ierr);
        ierr = MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &m_d, &ai_d, &aj_d, &done_d);
        CHKERRABORT(PETSC_COMM_SELF, ierr);

        ierr = MatSeqAIJRestoreArrayRead(B_offdiag, &aa_o);
        CHKERRABORT(PETSC_COMM_SELF, ierr);
        ierr = MatRestoreRowIJ(B_offdiag, 0, PETSC_FALSE, PETSC_FALSE, &m_o, &ai_o, &aj_o, &done_o);
        CHKERRABORT(PETSC_COMM_SELF, ierr);
    }

    return csr;
}

// ============================================================================
// PETSc Vector Extraction
// ============================================================================

/// Structure holding vector data extracted from PETSc
template<typename ValueType>
struct PetscVectorData {
    std::vector<ValueType> values;           ///< Local values
    std::int64_t local_size = 0;             ///< Local size
    std::int64_t global_size = 0;            ///< Global size
    std::int64_t local_start = 0;            ///< First global index on this rank
};

/// Extract local values from a PETSc vector
///
/// @tparam ValueType Vector value type
///
/// @param petsc_vec PETSc vector
///
/// @return Vector data structure with local values
template<typename ValueType = double>
PetscVectorData<ValueType> extract_petsc_vector(Vec petsc_vec)
{
    PetscErrorCode ierr;
    PetscVectorData<ValueType> vec_data;

    // Get sizes
    PetscInt local_size, global_size;
    ierr = VecGetLocalSize(petsc_vec, &local_size);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
    ierr = VecGetSize(petsc_vec, &global_size);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    // Get ownership range
    PetscInt local_start, local_end;
    ierr = VecGetOwnershipRange(petsc_vec, &local_start, &local_end);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    vec_data.local_size = local_size;
    vec_data.global_size = global_size;
    vec_data.local_start = local_start;

    // Get array
    const PetscScalar* array;
    ierr = VecGetArrayRead(petsc_vec, &array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    vec_data.values.resize(local_size);
    for (PetscInt i = 0; i < local_size; ++i) {
        vec_data.values[i] = static_cast<ValueType>(array[i]);
    }

    ierr = VecRestoreArrayRead(petsc_vec, &array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    return vec_data;
}

/// Copy values back to a PETSc vector
///
/// @tparam ValueType Vector value type
///
/// @param values Local values to copy
/// @param petsc_vec Destination PETSc vector
template<typename ValueType = double>
void copy_to_petsc_vector(const std::vector<ValueType>& values, Vec petsc_vec)
{
    PetscErrorCode ierr;

    PetscInt local_size;
    ierr = VecGetLocalSize(petsc_vec, &local_size);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    if (static_cast<PetscInt>(values.size()) != local_size) {
        throw std::runtime_error(
            "copy_to_petsc_vector: Size mismatch. Expected " +
            std::to_string(local_size) + ", got " + std::to_string(values.size())
        );
    }

    PetscScalar* array;
    ierr = VecGetArray(petsc_vec, &array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);

    for (PetscInt i = 0; i < local_size; ++i) {
        array[i] = static_cast<PetscScalar>(values[i]);
    }

    ierr = VecRestoreArray(petsc_vec, &array);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
}

// ============================================================================
// Ginkgo Array Creation
// ============================================================================

/// Create a Ginkgo array from a std::vector (copies data)
///
/// @tparam T Value type
///
/// @param exec Ginkgo executor
/// @param data Source vector
///
/// @return Ginkgo array on the specified executor
template<typename T>
gko::array<T> create_gko_array(std::shared_ptr<gko::Executor> exec,
                                const std::vector<T>& data)
{
    gko::array<T> arr(exec, data.size());
    exec->copy_from(exec->get_master(), data.size(), data.data(), arr.get_data());
    return arr;
}

/// Create a Ginkgo array from a std::vector (moves data to master, then copies to device)
///
/// This is more efficient when the source vector can be moved.
///
/// @tparam T Value type
///
/// @param exec Ginkgo executor
/// @param data Source vector (will be moved)
///
/// @return Ginkgo array on the specified executor
template<typename T>
gko::array<T> create_gko_array(std::shared_ptr<gko::Executor> exec,
                                std::vector<T>&& data)
{
    gko::array<T> arr(exec, data.size());
    auto master = exec->get_master();

    // If target is the master executor, just wrap the data
    if (exec == master) {
        std::copy(data.begin(), data.end(), arr.get_data());
    } else {
        // Copy to master first, then to device
        exec->copy_from(master, data.size(), data.data(), arr.get_data());
    }

    return arr;
}

} // namespace dolfinx_ginkgo
