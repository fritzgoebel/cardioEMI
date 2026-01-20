// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file test_convert.cpp
/// @brief Unit tests for PETSc to Ginkgo conversions

#include <gtest/gtest.h>
#include <dolfinx_ginkgo/ginkgo.h>
#include <petscmat.h>
#include <petscvec.h>

namespace dgko = dolfinx_ginkgo;

class ConvertTest : public ::testing::Test {
protected:
    void SetUp() override {
        PetscInitializeNoArguments();
        exec_ = dgko::create_executor(dgko::Backend::REFERENCE, 0);
    }

    void TearDown() override {
        PetscFinalize();
    }

    std::shared_ptr<gko::Executor> exec_;
};

TEST_F(ConvertTest, ExtractSeqAIJCSR) {
    // Create a small sequential matrix
    Mat A;
    MatCreate(PETSC_COMM_SELF, &A);
    MatSetSizes(A, 3, 3, 3, 3);
    MatSetType(A, MATSEQAIJ);
    MatSetUp(A);

    // Set values: simple diagonal matrix
    for (int i = 0; i < 3; ++i) {
        PetscScalar val = static_cast<PetscScalar>(i + 1);
        MatSetValue(A, i, i, val, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Extract CSR
    auto csr = dgko::extract_petsc_csr<double, std::int32_t, std::int64_t>(A);

    // Verify
    EXPECT_EQ(csr.local_rows, 3);
    EXPECT_EQ(csr.global_rows, 3);
    EXPECT_EQ(csr.global_cols, 3);
    EXPECT_EQ(csr.values.size(), 3u);

    // Check diagonal values
    EXPECT_DOUBLE_EQ(csr.values[0], 1.0);
    EXPECT_DOUBLE_EQ(csr.values[1], 2.0);
    EXPECT_DOUBLE_EQ(csr.values[2], 3.0);

    MatDestroy(&A);
}

TEST_F(ConvertTest, ExtractVectorData) {
    // Create a small vector
    Vec v;
    VecCreate(PETSC_COMM_SELF, &v);
    VecSetSizes(v, 5, 5);
    VecSetType(v, VECSEQ);
    VecSetUp(v);

    for (int i = 0; i < 5; ++i) {
        VecSetValue(v, i, static_cast<PetscScalar>(i * 2.0), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    // Extract data
    auto vec_data = dgko::extract_petsc_vector<double>(v);

    // Verify
    EXPECT_EQ(vec_data.local_size, 5);
    EXPECT_EQ(vec_data.global_size, 5);
    EXPECT_EQ(vec_data.values.size(), 5u);

    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(vec_data.values[i], i * 2.0);
    }

    VecDestroy(&v);
}

TEST_F(ConvertTest, CopyToPetscVector) {
    // Create vector data
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Create PETSc vector
    Vec v;
    VecCreate(PETSC_COMM_SELF, &v);
    VecSetSizes(v, 5, 5);
    VecSetType(v, VECSEQ);
    VecSetUp(v);
    VecSet(v, 0.0);

    // Copy values
    dgko::copy_to_petsc_vector(values, v);

    // Verify
    const PetscScalar* arr;
    VecGetArrayRead(v, &arr);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], values[i]);
    }
    VecRestoreArrayRead(v, &arr);

    VecDestroy(&v);
}

TEST_F(ConvertTest, CreateGkoArray) {
    std::vector<double> values = {1.0, 2.0, 3.0};

    // Test copy version
    auto arr_copy = dgko::create_gko_array(exec_, values);
    EXPECT_EQ(arr_copy.get_size(), 3u);

    // Test move version
    std::vector<double> values_move = {4.0, 5.0, 6.0};
    auto arr_move = dgko::create_gko_array(exec_, std::move(values_move));
    EXPECT_EQ(arr_move.get_size(), 3u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
