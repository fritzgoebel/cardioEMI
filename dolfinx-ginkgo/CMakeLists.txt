cmake_minimum_required(VERSION 3.19)

project(dolfinx_ginkgo
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Ginkgo linear solver backend for DOLFINx"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(DOLFINX_GINKGO_ENABLE_CUDA "Enable CUDA backend" OFF)
option(DOLFINX_GINKGO_ENABLE_HIP "Enable HIP backend" OFF)
option(DOLFINX_GINKGO_ENABLE_SYCL "Enable SYCL backend" OFF)
option(DOLFINX_GINKGO_BUILD_PYTHON "Build Python bindings" ON)
option(DOLFINX_GINKGO_BUILD_TESTS "Build tests" ON)
option(DOLFINX_GINKGO_BUILD_EXAMPLES "Build examples" ON)

# Find required packages
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(Ginkgo 1.8.0 REQUIRED)

# Find DOLFINx (note: package name is DOLFINX with uppercase X)
find_package(DOLFINX REQUIRED)

# Find PETSc (required for matrix conversion)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

# Create interface library (header-only for now)
add_library(dolfinx_ginkgo INTERFACE)
add_library(dolfinx_ginkgo::dolfinx_ginkgo ALIAS dolfinx_ginkgo)

target_include_directories(dolfinx_ginkgo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>
    $<INSTALL_INTERFACE:include>
)

# Find METIS and ParMETIS (via scotch) from PETSc installation
set(PETSC_LIB_HINTS
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib
    /usr/local/petsc/linux-gnu-real64-32/lib
    /usr/local/petsc/linux-gnu-complex64-32/lib
)
find_library(METIS_LIBRARY NAMES metis HINTS ${PETSC_LIB_HINTS})
find_library(PTSCOTCHPARMETIS_LIBRARY NAMES ptscotchparmetisv3 HINTS ${PETSC_LIB_HINTS})
find_library(SCOTCH_LIBRARY NAMES scotch HINTS ${PETSC_LIB_HINTS})
find_library(SCOTCHERR_LIBRARY NAMES scotcherr HINTS ${PETSC_LIB_HINTS})

target_link_libraries(dolfinx_ginkgo INTERFACE
    Ginkgo::ginkgo
    dolfinx
    MPI::MPI_CXX
    PkgConfig::PETSC
)

# Add METIS/ParMETIS if found
if(METIS_LIBRARY)
    target_link_libraries(dolfinx_ginkgo INTERFACE ${METIS_LIBRARY})
endif()
if(PTSCOTCHPARMETIS_LIBRARY)
    target_link_libraries(dolfinx_ginkgo INTERFACE ${PTSCOTCHPARMETIS_LIBRARY})
endif()
if(SCOTCH_LIBRARY)
    target_link_libraries(dolfinx_ginkgo INTERFACE ${SCOTCH_LIBRARY})
endif()
if(SCOTCHERR_LIBRARY)
    target_link_libraries(dolfinx_ginkgo INTERFACE ${SCOTCHERR_LIBRARY})
endif()

# DOLFINx target may be named differently
if(TARGET DOLFINX::dolfinx)
    target_link_libraries(dolfinx_ginkgo INTERFACE DOLFINX::dolfinx)
endif()

# Set compile definitions based on enabled backends
if(DOLFINX_GINKGO_ENABLE_CUDA)
    target_compile_definitions(dolfinx_ginkgo INTERFACE DOLFINX_GINKGO_ENABLE_CUDA)
endif()

if(DOLFINX_GINKGO_ENABLE_HIP)
    target_compile_definitions(dolfinx_ginkgo INTERFACE DOLFINX_GINKGO_ENABLE_HIP)
endif()

if(DOLFINX_GINKGO_ENABLE_SYCL)
    target_compile_definitions(dolfinx_ginkgo INTERFACE DOLFINX_GINKGO_ENABLE_SYCL)
endif()

# Python bindings
if(DOLFINX_GINKGO_BUILD_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # Try to find nanobind
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        RESULT_VARIABLE nanobind_RESULT
    )

    if(nanobind_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
        find_package(nanobind CONFIG REQUIRED)

        # Find petsc4py include directory
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import petsc4py; print(petsc4py.get_include())"
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
            RESULT_VARIABLE PETSC4PY_RESULT
        )
        if(NOT PETSC4PY_RESULT EQUAL 0)
            message(FATAL_ERROR "petsc4py not found. Install with: pip install petsc4py")
        endif()
        message(STATUS "petsc4py include: ${PETSC4PY_INCLUDE_DIR}")

        # Find mpi4py include directory
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
            RESULT_VARIABLE MPI4PY_RESULT
        )
        if(NOT MPI4PY_RESULT EQUAL 0)
            message(FATAL_ERROR "mpi4py not found. Install with: pip install mpi4py")
        endif()
        message(STATUS "mpi4py include: ${MPI4PY_INCLUDE_DIR}")

        nanobind_add_module(_cpp
            python/dolfinx_ginkgo/_cpp.cpp
        )

        target_link_libraries(_cpp PRIVATE dolfinx_ginkgo)
        target_include_directories(_cpp PRIVATE
            ${PETSC4PY_INCLUDE_DIR}
            ${MPI4PY_INCLUDE_DIR}
        )

        install(TARGETS _cpp LIBRARY DESTINATION dolfinx_ginkgo)
    else()
        message(WARNING "nanobind not found, skipping Python bindings")
    endif()
endif()

# Tests
if(DOLFINX_GINKGO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(DOLFINX_GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install headers
install(DIRECTORY cpp/dolfinx_ginkgo
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Install CMake config
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_ginkgo-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dolfinx_ginkgo-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_ginkgo-config.cmake"
    INSTALL_DESTINATION lib/cmake/dolfinx_ginkgo
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_ginkgo-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_ginkgo-config-version.cmake"
    DESTINATION lib/cmake/dolfinx_ginkgo
)

install(TARGETS dolfinx_ginkgo
    EXPORT dolfinx_ginkgo-targets
    INCLUDES DESTINATION include
)

install(EXPORT dolfinx_ginkgo-targets
    NAMESPACE dolfinx_ginkgo::
    DESTINATION lib/cmake/dolfinx_ginkgo
)

# Print configuration summary
message(STATUS "")
message(STATUS "dolfinx-ginkgo configuration summary")
message(STATUS "====================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "CUDA backend: ${DOLFINX_GINKGO_ENABLE_CUDA}")
message(STATUS "HIP backend: ${DOLFINX_GINKGO_ENABLE_HIP}")
message(STATUS "SYCL backend: ${DOLFINX_GINKGO_ENABLE_SYCL}")
message(STATUS "Python bindings: ${DOLFINX_GINKGO_BUILD_PYTHON}")
message(STATUS "Tests: ${DOLFINX_GINKGO_BUILD_TESTS}")
message(STATUS "Examples: ${DOLFINX_GINKGO_BUILD_EXAMPLES}")
message(STATUS "")
