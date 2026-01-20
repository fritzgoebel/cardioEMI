// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file test_partition.cpp
/// @brief Unit tests for partition creation

#include <gtest/gtest.h>
#include <dolfinx_ginkgo/ginkgo.h>
#include <mpi.h>

namespace dgko = dolfinx_ginkgo;

class PartitionTest : public ::testing::Test {
protected:
    void SetUp() override {
        exec_ = dgko::create_executor(dgko::Backend::REFERENCE, 0);
        gko_comm_ = dgko::create_communicator(MPI_COMM_WORLD);
    }

    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<gko::experimental::mpi::communicator> gko_comm_;
};

TEST_F(PartitionTest, CreateFromRanges) {
    // Create partition for 100 elements across ranks
    // Assume 2 ranks: [0, 50) and [50, 100)
    std::vector<std::int64_t> ranges = {0, 50, 100};

    auto partition = dgko::create_partition_from_ranges<std::int32_t, std::int64_t>(
        exec_, ranges
    );

    ASSERT_NE(partition, nullptr);

    // Check global size
    EXPECT_EQ(partition->get_size(), 100);

    // Check number of parts
    EXPECT_EQ(partition->get_num_parts(), 2);
}

TEST_F(PartitionTest, CreateUniform) {
    const std::int64_t global_size = 100;

    auto partition = dgko::create_uniform_partition<std::int32_t, std::int64_t>(
        exec_, gko_comm_, global_size
    );

    ASSERT_NE(partition, nullptr);

    // Check global size
    EXPECT_EQ(partition->get_size(), global_size);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
