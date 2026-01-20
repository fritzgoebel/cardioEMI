// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file DistributedVector.h
/// @brief Ginkgo distributed vector creation from PETSc vectors

#include <ginkgo/ginkgo.hpp>
#include <petscvec.h>
#include <mpi.h>

#include <memory>
#include <vector>

#include "convert.h"
#include "Partition.h"

namespace dolfinx_ginkgo {

namespace gko_dist = gko::experimental::distributed;

/// Create a Ginkgo distributed vector from a PETSc vector
///
/// This function:
/// 1. Extracts local values from the PETSc vector
/// 2. Creates a Ginkgo partition from the distribution
/// 3. Builds a Ginkgo distributed::Vector with the values
///
/// @tparam ValueType Vector value type (typically double)
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param petsc_vec PETSc vector
///
/// @return Shared pointer to the Ginkgo distributed vector
///
/// @note The vector data is copied to Ginkgo's internal storage.
template<typename ValueType = double>
std::shared_ptr<gko_dist::Vector<ValueType>>
create_distributed_vector_from_petsc(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    Vec petsc_vec)
{
    using vector_type = gko_dist::Vector<ValueType>;
    using local_vector_type = gko::matrix::Dense<ValueType>;
    using partition_type = gko_dist::Partition<std::int32_t, std::int64_t>;

    MPI_Comm comm = gko_comm->get();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Extract data from PETSc vector
    auto vec_data = extract_petsc_vector<ValueType>(petsc_vec);

    // Build partition from distribution
    std::vector<std::int64_t> ranges(size + 1);
    std::int64_t my_start = vec_data.local_start;
    MPI_Allgather(&my_start, 1, MPI_INT64_T,
                  ranges.data(), 1, MPI_INT64_T, comm);
    ranges[size] = vec_data.global_size;

    auto partition = create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec->get_master(), ranges
    );

    // Create local Dense vector (column vector: N x 1)
    auto local_values = create_gko_array(exec, std::move(vec_data.values));
    auto local_vec = local_vector_type::create(
        exec,
        gko::dim<2>{static_cast<size_t>(vec_data.local_size), 1},
        std::move(local_values),
        1  // stride
    );

    // Create distributed vector
    // Ginkgo 1.11 API: create(exec, comm, global_dim, local_vec)
    gko::dim<2> global_dim{static_cast<size_t>(vec_data.global_size), 1};
    return vector_type::create(exec, *gko_comm, global_dim, std::move(local_vec));
}

/// Create an empty Ginkgo distributed vector with given partition
///
/// @tparam ValueType Vector value type
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param partition Ginkgo partition describing distribution
///
/// @return Shared pointer to the Ginkgo distributed vector (zero-initialized)
template<typename ValueType = double,
         typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Vector<ValueType>>
create_distributed_vector(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    std::shared_ptr<gko_dist::Partition<LocalIndexType, GlobalIndexType>> partition)
{
    using vector_type = gko_dist::Vector<ValueType>;
    using local_vector_type = gko::matrix::Dense<ValueType>;

    int rank = gko_comm->rank();
    const auto* range_bounds = partition->get_range_bounds();
    auto local_size = range_bounds[rank + 1] - range_bounds[rank];
    auto global_size = range_bounds[partition->get_num_parts()];

    // Create zero-initialized local vector
    auto local_vec = local_vector_type::create(
        exec,
        gko::dim<2>{static_cast<size_t>(local_size), 1}
    );

    // Ginkgo 1.11 API: create(exec, comm, global_dim, local_vec)
    gko::dim<2> global_dim{static_cast<size_t>(global_size), 1};
    return vector_type::create(exec, *gko_comm, global_dim, std::move(local_vec));
}

/// Create a Ginkgo distributed vector from local values
///
/// @tparam ValueType Vector value type
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param local_values Local values for this rank
/// @param global_size Total vector size
///
/// @return Shared pointer to the Ginkgo distributed vector
template<typename ValueType = double>
std::shared_ptr<gko_dist::Vector<ValueType>>
create_distributed_vector_from_local(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    const std::vector<ValueType>& local_values,
    std::int64_t global_size)
{
    using vector_type = gko_dist::Vector<ValueType>;
    using local_vector_type = gko::matrix::Dense<ValueType>;
    using partition_type = gko_dist::Partition<std::int32_t, std::int64_t>;

    MPI_Comm comm = gko_comm->get();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Compute local start via prefix sum
    std::int64_t local_size = local_values.size();
    std::int64_t local_start = 0;
    MPI_Exscan(&local_size, &local_start, 1, MPI_INT64_T, MPI_SUM, comm);

    // Build partition
    std::vector<std::int64_t> ranges(size + 1);
    MPI_Allgather(&local_start, 1, MPI_INT64_T,
                  ranges.data(), 1, MPI_INT64_T, comm);
    ranges[size] = global_size;

    auto partition = create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec->get_master(), ranges
    );

    // Create local Dense vector
    auto gko_values = create_gko_array(exec, local_values);
    auto local_vec = local_vector_type::create(
        exec,
        gko::dim<2>{static_cast<size_t>(local_size), 1},
        std::move(gko_values),
        1
    );

    // Ginkgo 1.11 API: create(exec, comm, global_dim, local_vec)
    gko::dim<2> global_dim{static_cast<size_t>(global_size), 1};
    return vector_type::create(exec, *gko_comm, global_dim, std::move(local_vec));
}

/// Copy values from Ginkgo distributed vector to PETSc vector
///
/// @tparam ValueType Vector value type
///
/// @param gko_vec Source Ginkgo distributed vector
/// @param petsc_vec Destination PETSc vector
///
/// @note The vectors must have compatible sizes and distributions.
template<typename ValueType = double>
void copy_to_petsc(
    const gko_dist::Vector<ValueType>& gko_vec,
    Vec petsc_vec)
{
    // Get local vector from Ginkgo
    const auto* local_vec = gko_vec.get_local_vector();
    auto exec = gko_vec.get_executor();
    auto master = exec->get_master();

    // Get local size
    auto local_size = local_vec->get_size()[0];

    // Copy to host if needed
    std::vector<ValueType> host_values(local_size);
    if (exec != master) {
        auto host_vec = local_vec->clone(master);
        const auto* data = host_vec->get_const_values();
        std::copy(data, data + local_size, host_values.begin());
    } else {
        const auto* data = local_vec->get_const_values();
        std::copy(data, data + local_size, host_values.begin());
    }

    // Copy to PETSc
    copy_to_petsc_vector(host_values, petsc_vec);
}

/// Copy values from PETSc vector to existing Ginkgo distributed vector
///
/// @tparam ValueType Vector value type
///
/// @param petsc_vec Source PETSc vector
/// @param gko_vec Destination Ginkgo distributed vector
///
/// @note The vectors must have compatible sizes and distributions.
template<typename ValueType = double>
void copy_from_petsc(
    Vec petsc_vec,
    gko_dist::Vector<ValueType>& gko_vec)
{
    // Extract data from PETSc
    auto vec_data = extract_petsc_vector<ValueType>(petsc_vec);

    // Get local vector from Ginkgo
    auto local_vec = gko_vec.get_local_vector();
    auto exec = gko_vec.get_executor();

    // Copy values to Ginkgo
    exec->copy_from(exec->get_master(),
                    vec_data.values.size(),
                    vec_data.values.data(),
                    local_vec->get_values());
}

/// Compute the 2-norm of a Ginkgo distributed vector
///
/// @tparam ValueType Vector value type
///
/// @param gko_vec Ginkgo distributed vector
///
/// @return The 2-norm (Euclidean norm)
template<typename ValueType = double>
ValueType compute_norm2(const gko_dist::Vector<ValueType>& gko_vec)
{
    auto exec = gko_vec.get_executor();

    // Create result array on master
    auto result = gko::array<ValueType>(exec->get_master(), 1);

    // Compute norm
    gko_vec.compute_norm2(result.get_data());

    return result.get_const_data()[0];
}

/// Compute the inner product of two Ginkgo distributed vectors
///
/// @tparam ValueType Vector value type
///
/// @param x First vector
/// @param y Second vector
///
/// @return The inner product <x, y>
template<typename ValueType = double>
ValueType compute_dot(const gko_dist::Vector<ValueType>& x,
                      const gko_dist::Vector<ValueType>& y)
{
    auto exec = x.get_executor();

    // Create result array on master
    auto result = gko::array<ValueType>(exec->get_master(), 1);

    // Compute dot product
    x.compute_dot(&y, result.get_data());

    return result.get_const_data()[0];
}

} // namespace dolfinx_ginkgo
