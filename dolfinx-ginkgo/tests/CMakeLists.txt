# Tests for dolfinx-ginkgo

find_package(GTest QUIET)

if(GTest_FOUND)
    # Unit tests
    add_executable(test_partition test_partition.cpp)
    target_link_libraries(test_partition PRIVATE dolfinx_ginkgo GTest::gtest_main)
    add_test(NAME test_partition COMMAND test_partition)

    add_executable(test_convert test_convert.cpp)
    target_link_libraries(test_convert PRIVATE dolfinx_ginkgo GTest::gtest_main)
    add_test(NAME test_convert COMMAND test_convert)
else()
    message(STATUS "GTest not found, skipping unit tests")
endif()

# MPI integration tests
add_executable(test_distributed_matrix test_distributed_matrix.cpp)
target_link_libraries(test_distributed_matrix PRIVATE dolfinx_ginkgo)
add_test(NAME test_distributed_matrix
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_distributed_matrix>
)

add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE dolfinx_ginkgo)
add_test(NAME test_solver
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_solver>
)
