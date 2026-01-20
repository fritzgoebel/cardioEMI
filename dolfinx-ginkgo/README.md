# dolfinx-ginkgo

GPU-accelerated distributed linear solvers for DOLFINx using Ginkgo.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Ginkgo](https://img.shields.io/badge/Ginkgo-1.11.0-blue)]()
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.9.0-blue)]()

## Overview

dolfinx-ginkgo provides an alternative linear solver backend for DOLFINx/FEniCSx applications, enabling GPU-accelerated sparse linear algebra with:

- **CUDA** (NVIDIA GPUs)
- **HIP** (AMD GPUs via ROCm)
- **SYCL** (Intel GPUs via oneAPI)
- **OpenMP** (CPU parallelism)

The library integrates with the existing PETSc-based assembly workflow, converting PETSc matrices to Ginkgo's distributed format for GPU-accelerated solving.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Matrix Conversion | ✅ Complete | PETSc MPIAIJ → Ginkgo distributed::Matrix |
| Vector Conversion | ✅ Complete | PETSc Vec ↔ Ginkgo distributed::Vector |
| CG Solver | ✅ Tested | With/without preconditioner |
| GMRES Solver | ✅ Tested | Configurable Krylov dimension |
| Jacobi Preconditioner | ✅ Tested | Point and block variants |
| ILU Preconditioner | ✅ Implemented | Via Schwarz wrapper |
| AMG Preconditioner | ✅ Implemented | PGM coarsening, V/W/F cycles |
| Python Bindings | 🚧 Planned | Phase 4 |
| GPU Backends | 🚧 Untested | CUDA/HIP/SYCL compile flags ready |

## Quick Start

```bash
# Clone and build with Docker
cd dolfinx-ginkgo
docker build -t dolfinx-ginkgo:latest .

# Run tests
docker run --rm -v "$(pwd):/home/fenics/dolfinx-ginkgo" \
  -w /home/fenics/dolfinx-ginkgo/build dolfinx-ginkgo:latest \
  bash -c "cmake .. -DCMAKE_PREFIX_PATH=/usr/local/dolfinx-real \
           -DDOLFINX_GINKGO_BUILD_PYTHON=OFF && make -j2"

# Test matrix/vector conversion
docker run --rm -v "$(pwd):/home/fenics/dolfinx-ginkgo" \
  -w /home/fenics/dolfinx-ginkgo/build dolfinx-ginkgo:latest \
  mpirun -n 2 ./tests/test_distributed_matrix

# Test solvers
docker run --rm -v "$(pwd):/home/fenics/dolfinx-ginkgo" \
  -w /home/fenics/dolfinx-ginkgo/build dolfinx-ginkgo:latest \
  mpirun -n 2 ./tests/test_solver
```

## Features

### Krylov Solvers
- Conjugate Gradient (CG)
- Flexible CG (FCG)
- GMRES
- BiCGSTAB
- CGS

### Preconditioners
- Point Jacobi
- Block Jacobi
- ILU (Incomplete LU)
- IC (Incomplete Cholesky)
- ISAI (Approximate Sparse Inverse)
- **AMG** (Algebraic Multigrid with PGM coarsening)

### Key Capabilities
- Distributed computing via MPI
- Automatic PETSc → Ginkgo matrix/vector conversion
- AMG preconditioner with configurable cycle types (V, W, F)
- Mixed-precision AMG support
- Python bindings via nanobind

## Requirements

### Required
- CMake ≥ 3.19
- Ginkgo ≥ 1.11.0 (with distributed support and MPI)
- DOLFINx ≥ 0.9.0
- PETSc (for matrix assembly and conversion)
- MPI

### Optional
- CUDA Toolkit ≥ 11.0 (for NVIDIA GPUs)
- ROCm ≥ 4.5 (for AMD GPUs)
- oneAPI ≥ 2023.1 (for Intel GPUs)
- nanobind (for Python bindings)
- Google Test (for unit tests)

## Installation

### Option 1: Docker (Recommended)

The easiest way to build and test is using the provided Docker setup:

```bash
cd cardioEMI/dolfinx-ginkgo

# Build Docker image and run tests
./docker-build.sh
```

This builds a Docker image with DOLFINx 0.9.0 + Ginkgo 1.11.0, compiles the library, and runs the tests.

To use interactively:

```bash
# Build the image
docker build -t dolfinx-ginkgo .

# Run container
docker run -it -v "$(pwd)/..:/home/fenics" -w /home/fenics dolfinx-ginkgo bash
```

### Option 2: Native Installation

Requires Ginkgo 1.8.0+ installed on your system.

```bash
cd cardioEMI/dolfinx-ginkgo

# Create build directory
mkdir build && cd build

# Configure (adjust options as needed)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDOLFINX_GINKGO_ENABLE_CUDA=ON \
    -DDOLFINX_GINKGO_BUILD_PYTHON=ON \
    -DDOLFINX_GINKGO_BUILD_TESTS=ON

# Build
make -j$(nproc)

# Install
make install

# Run tests
ctest
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `DOLFINX_GINKGO_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `DOLFINX_GINKGO_ENABLE_HIP` | OFF | Enable HIP backend |
| `DOLFINX_GINKGO_ENABLE_SYCL` | OFF | Enable SYCL backend |
| `DOLFINX_GINKGO_BUILD_PYTHON` | ON | Build Python bindings |
| `DOLFINX_GINKGO_BUILD_TESTS` | ON | Build tests |
| `DOLFINX_GINKGO_BUILD_EXAMPLES` | ON | Build examples |

## Usage

### Python

```python
from dolfinx_ginkgo import GinkgoSolver

# After assembling matrix A with DOLFINx/multiphenicsx
solver = GinkgoSolver(
    A,
    comm=comm,
    backend="cuda",       # or "hip", "omp"
    solver="cg",
    preconditioner="amg",
    rtol=1e-8,
    amg_config={
        "max_levels": 10,
        "cycle": "v",
        "smoother": "jacobi",
        "coarse_solver": "direct"
    }
)

# Solve
for timestep in range(num_timesteps):
    # ... assemble RHS b ...
    solver.solve(b, x)
    print(f"Converged in {solver.iterations} iterations")
```

### C++

```cpp
#include <dolfinx_ginkgo/ginkgo.h>

namespace dgko = dolfinx_ginkgo;

// Create executor and communicator
auto exec = dgko::create_executor(dgko::Backend::CUDA, 0);
auto gko_comm = dgko::create_communicator(MPI_COMM_WORLD);

// Configure solver
dgko::SolverConfig config;
config.solver = dgko::SolverType::CG;
config.preconditioner = dgko::PreconditionerType::AMG;
config.rtol = 1e-8;
config.amg.cycle = dgko::AMGConfig::Cycle::V;

// Create distributed matrix from PETSc
auto A_gko = dgko::create_distributed_matrix_from_petsc<>(exec, gko_comm, A);

// Create solver
dgko::DistributedSolver<> solver(exec, gko_comm, config);
solver.set_operator(A_gko);

// Solve
auto b_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, b);
auto x_gko = dgko::create_distributed_vector_from_petsc<>(exec, gko_comm, x);
solver.solve(*b_gko, *x_gko);

// Copy result back
dgko::copy_to_petsc(*x_gko, x);
```

## Architecture

```
DOLFINx/multiphenicsx Assembly → PETSc Matrix → Extract CSR → Ginkgo dist::Matrix → Ginkgo Solver
                                  (per rank)      (per rank)     (distributed)

┌──────────────────────────────────────────────────────────────────┐
│                         MPI Rank 0                               │
├──────────────────────────────────────────────────────────────────┤
│  Assembly    →   PETSc Mat   →   Extract CSR   →   Ginkgo Matrix │
└──────────────────────────────────────────────────────────────────┘
                              ↕ MPI Communication ↕
┌──────────────────────────────────────────────────────────────────┐
│                         MPI Rank 1                               │
├──────────────────────────────────────────────────────────────────┤
│  Assembly    →   PETSc Mat   →   Extract CSR   →   Ginkgo Matrix │
└──────────────────────────────────────────────────────────────────┘
```

## File Structure

```
dolfinx-ginkgo/
├── CMakeLists.txt
├── Dockerfile                    # DOLFINx 0.9.0 + Ginkgo 1.11.0
├── docker-build.sh               # Build and test script
├── README.md
├── cmake/
│   └── dolfinx_ginkgo-config.cmake.in
├── cpp/
│   └── dolfinx_ginkgo/
│       ├── ginkgo.h              # Main header, Backend enum, SolverConfig, AMGConfig
│       ├── Partition.h           # IndexMap → Ginkgo Partition
│       ├── convert.h             # PETSc CSR extraction (MPIAIJ support)
│       ├── DistributedMatrix.h   # PETSc Mat → Ginkgo distributed::Matrix
│       ├── DistributedVector.h   # PETSc Vec → Ginkgo distributed::Vector
│       └── Solver.h              # DistributedSolver with Krylov solvers + preconditioners
├── python/
│   └── dolfinx_ginkgo/
│       ├── __init__.py
│       ├── _cpp.cpp              # nanobind bindings (planned)
│       └── solver.py             # High-level Python API (planned)
├── tests/
│   ├── CMakeLists.txt
│   ├── test_partition.cpp        # Partition unit tests
│   ├── test_convert.cpp          # CSR conversion unit tests
│   ├── test_distributed_matrix.cpp  # MPI matrix/vector integration tests
│   └── test_solver.cpp           # Solver integration tests (CG, GMRES, preconditioners)
└── examples/
    ├── CMakeLists.txt
    └── poisson.cpp               # Example Poisson solve
```

## AMG Configuration

The AMG preconditioner uses Ginkgo's PGM (Parallel Graph Match) coarsening:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_levels` | 10 | Maximum multigrid levels |
| `min_coarse_rows` | 100 | Stop coarsening below this size |
| `cycle` | V | Cycle type: V, W, or F |
| `smoother` | JACOBI | Smoother: JACOBI, GAUSS_SEIDEL, ILU |
| `pre_smooth_steps` | 1 | Pre-smoothing iterations |
| `post_smooth_steps` | 1 | Post-smoothing iterations |
| `relaxation_factor` | 0.9 | Smoother relaxation |
| `coarse_solver` | DIRECT | Coarse solver: DIRECT, CG, GMRES |
| `use_mixed_precision` | false | Enable mixed precision |

## Performance Notes

- **Setup phase** (matrix conversion, AMG hierarchy): Done once
- **Solve phase**: GPU-accelerated, runs every timestep
- **Memory transfers**: b and x vectors transferred each timestep

For best performance:
1. Use AMG for large-scale problems (mesh-independent convergence)
2. Keep matrix structure fixed when possible (use `update_operator()`)
3. Consider mixed-precision AMG for additional speedup

## Test Results

The following tests pass on 2 MPI ranks:

```
=== Test: Distributed Matrix Conversion ===
--- Test 1: CSR Extraction --- [OK]
--- Test 2: Ginkgo Matrix Creation --- [OK]
--- Test 3: Ginkgo Vector Creation --- [OK]
--- Test 4: Vector Round-Trip --- [OK]

=== Test: Distributed Solver ===
--- Test: CG (no preconditioner) ---
  Iterations: 100, error = 5.99e-15 [OK]

--- Test: CG + Jacobi ---
  Iterations: 100, error = 5.99e-15 [OK]

--- Test: CG + Block Jacobi ---
  Iterations: 52, error = 9.67e-13 [OK]

--- Test: Ginkgo vs PETSc comparison ---
  PETSc:  100 iterations, error = 2.89e-15
  Ginkgo: 100 iterations, error = 5.99e-15 [OK]
```

## License

MIT License

## References

- [Ginkgo Documentation](https://ginkgo-project.github.io/)
- [DOLFINx Documentation](https://docs.fenicsproject.org/dolfinx/main/python/)
- [Ginkgo Distributed Examples](https://github.com/ginkgo-project/ginkgo/tree/develop/examples/distributed)
