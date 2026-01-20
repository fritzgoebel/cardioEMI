# Examples for dolfinx-ginkgo

# Simple Poisson solver example
add_executable(poisson_example poisson.cpp)
target_link_libraries(poisson_example PRIVATE dolfinx_ginkgo)

# Add MPI run target
add_custom_target(run_poisson
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:poisson_example>
    DEPENDS poisson_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Poisson example with 4 MPI ranks"
)
