// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

#pragma once

/// @file Partition.h
/// @brief Utilities for creating Ginkgo partitions from DOLFINx IndexMaps

#include <ginkgo/ginkgo.hpp>
#include <dolfinx/common/IndexMap.h>
#include <mpi.h>

#include <memory>
#include <vector>

namespace dolfinx_ginkgo {

namespace gko_dist = gko::experimental::distributed;

/// Create a Ginkgo partition from a DOLFINx IndexMap
///
/// The partition describes how global indices are distributed across MPI ranks.
/// DOLFINx IndexMap provides this information, which we translate to Ginkgo's
/// partition format.
///
/// @tparam LocalIndexType Local index type (typically int32_t)
/// @tparam GlobalIndexType Global index type (typically int64_t)
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param index_map DOLFINx index map describing the distribution
///
/// @return Shared pointer to the Ginkgo partition
///
/// @note The partition is built from range information where each rank owns
///       a contiguous block of global indices.
template<typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Partition<LocalIndexType, GlobalIndexType>>
create_partition_from_index_map(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    const dolfinx::common::IndexMap& index_map)
{
    MPI_Comm comm = gko_comm->get();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get local and global sizes from IndexMap
    const GlobalIndexType local_size = index_map.size_local();
    const GlobalIndexType global_size = index_map.size_global();

    // Get the local range [local_start, local_end) owned by this rank
    const auto local_range = index_map.local_range();
    const GlobalIndexType local_start = local_range[0];

    // Gather all starting offsets to build partition ranges
    // ranges[i] gives the first global index owned by rank i
    // ranges[size] gives the total global size
    std::vector<GlobalIndexType> ranges(size + 1);

    // Gather local_start from all ranks
    MPI_Allgather(&local_start, 1, MPI_INT64_T,
                  ranges.data(), 1, MPI_INT64_T, comm);

    // The last entry is the global size
    ranges[size] = global_size;

    // Create Ginkgo array from ranges
    auto ranges_array = gko::array<GlobalIndexType>::view(
        exec->get_master(), size + 1, ranges.data()
    );

    // Build partition from consecutive ranges
    // Each rank owns indices [ranges[rank], ranges[rank+1])
    return gko_dist::Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges_array
    );
}

/// Create a Ginkgo partition from explicit range boundaries
///
/// This is useful when the partition is already known or when creating
/// custom partitions for testing.
///
/// @tparam LocalIndexType Local index type
/// @tparam GlobalIndexType Global index type
///
/// @param exec Ginkgo executor
/// @param ranges Vector of range boundaries [r0, r1, ..., rn] where rank i
///               owns indices [ri, ri+1)
///
/// @return Shared pointer to the Ginkgo partition
template<typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Partition<LocalIndexType, GlobalIndexType>>
create_partition_from_ranges(
    std::shared_ptr<const gko::Executor> exec,
    const std::vector<GlobalIndexType>& ranges)
{
    auto ranges_array = gko::array<GlobalIndexType>::view(
        exec->get_master(), ranges.size(), const_cast<GlobalIndexType*>(ranges.data())
    );

    return gko_dist::Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges_array
    );
}

/// Create a Ginkgo partition for a simple uniform distribution
///
/// This creates a partition where global indices are evenly distributed
/// across all ranks. The last rank gets any remainder.
///
/// @tparam LocalIndexType Local index type
/// @tparam GlobalIndexType Global index type
///
/// @param exec Ginkgo executor
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param global_size Total number of indices
///
/// @return Shared pointer to the Ginkgo partition
template<typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
std::shared_ptr<gko_dist::Partition<LocalIndexType, GlobalIndexType>>
create_uniform_partition(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    GlobalIndexType global_size)
{
    int size = gko_comm->size();

    std::vector<GlobalIndexType> ranges(size + 1);
    GlobalIndexType chunk_size = global_size / size;
    GlobalIndexType remainder = global_size % size;

    ranges[0] = 0;
    for (int i = 0; i < size; ++i) {
        // First 'remainder' ranks get one extra element
        ranges[i + 1] = ranges[i] + chunk_size + (i < remainder ? 1 : 0);
    }

    return create_partition_from_ranges<LocalIndexType, GlobalIndexType>(exec, ranges);
}

/// Get partition information for the current rank
///
/// @tparam LocalIndexType Local index type
/// @tparam GlobalIndexType Global index type
///
/// @param partition The Ginkgo partition
/// @param gko_comm Ginkgo MPI communicator wrapper
/// @param[out] local_size Number of indices owned by this rank
/// @param[out] global_start First global index owned by this rank
template<typename LocalIndexType = std::int32_t,
         typename GlobalIndexType = std::int64_t>
void get_local_partition_info(
    const gko_dist::Partition<LocalIndexType, GlobalIndexType>& partition,
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
    LocalIndexType& local_size,
    GlobalIndexType& global_start)
{
    int rank = gko_comm->rank();

    // Get the range offsets array from the partition
    const auto* range_bounds = partition.get_range_bounds();

    global_start = range_bounds[rank];
    local_size = static_cast<LocalIndexType>(range_bounds[rank + 1] - range_bounds[rank]);
}

} // namespace dolfinx_ginkgo
