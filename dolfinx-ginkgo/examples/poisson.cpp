// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file poisson.cpp
/// @brief Example: Solve Poisson equation with Ginkgo
///
/// This example demonstrates using dolfinx-ginkgo to solve a simple
/// Poisson equation: -Δu = f with Dirichlet BCs.
///
/// Compile: See CMakeLists.txt
/// Run: mpirun -n 4 ./poisson_example

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx_ginkgo/ginkgo.h>

#include <cmath>
#include <iostream>

using namespace dolfinx;
namespace dgko = dolfinx_ginkgo;

int main(int argc, char* argv[])
{
    // Initialize MPI and PETSc
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    {
        MPI_Comm comm = MPI_COMM_WORLD;
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (rank == 0) {
            std::cout << "=== Poisson Example with Ginkgo Solver ===" << std::endl;
            std::cout << "Running with " << size << " MPI processes" << std::endl;
        }

        // =====================================================================
        // Create mesh and function space
        // =====================================================================

        // Unit square mesh
        auto mesh = std::make_shared<mesh::Mesh<double>>(
            mesh::create_rectangle<double>(
                comm,
                {{{0.0, 0.0}, {1.0, 1.0}}},
                {64, 64},
                mesh::CellType::triangle
            )
        );

        // P1 Lagrange element
        auto element = basix::create_element<double>(
            basix::element::family::P,
            basix::cell::type::triangle,
            1,
            basix::element::lagrange_variant::unset,
            basix::element::dpc_variant::unset,
            false
        );

        auto V = std::make_shared<fem::FunctionSpace<double>>(
            fem::create_functionspace(mesh, element, {})
        );

        if (rank == 0) {
            std::cout << "Mesh: " << mesh->topology()->index_map(0)->size_global()
                      << " vertices, " << mesh->topology()->index_map(2)->size_global()
                      << " cells" << std::endl;
            std::cout << "DOFs: " << V->dofmap()->index_map->size_global() << std::endl;
        }

        // =====================================================================
        // Define variational problem
        // =====================================================================

        // Source term f = 2π² sin(πx) sin(πy)
        auto f = std::make_shared<fem::Function<double>>(V);
        f->interpolate([](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>> {
            std::vector<double> vals(x.extent(1));
            for (std::size_t i = 0; i < x.extent(1); ++i) {
                double px = x(0, i);
                double py = x(1, i);
                vals[i] = 2.0 * M_PI * M_PI * std::sin(M_PI * px) * std::sin(M_PI * py);
            }
            return {vals, {vals.size()}};
        });

        // Bilinear form a(u,v) = ∫ ∇u·∇v dx
        // Linear form L(v) = ∫ f*v dx
        // (Forms would be defined via UFL/FFCx in real code)

        // For this example, we'll use DOLFINx's built-in assembly
        // In practice, you would use multiphenicsx or DOLFINx forms

        // =====================================================================
        // Boundary conditions
        // =====================================================================

        // Dirichlet BC: u = 0 on boundary
        auto boundary_facets = mesh::locate_entities_boundary(
            *mesh, 1,
            [](auto x) -> std::vector<std::int8_t> {
                std::vector<std::int8_t> marker(x.extent(1), false);
                for (std::size_t i = 0; i < x.extent(1); ++i) {
                    double px = x(0, i);
                    double py = x(1, i);
                    if (std::abs(px) < 1e-10 || std::abs(px - 1.0) < 1e-10 ||
                        std::abs(py) < 1e-10 || std::abs(py - 1.0) < 1e-10) {
                        marker[i] = true;
                    }
                }
                return marker;
            }
        );

        auto bc_dofs = fem::locate_dofs_topological(
            *V->mesh()->topology(), *V->dofmap(), 1, boundary_facets
        );

        auto u0 = std::make_shared<fem::Function<double>>(V);
        u0->x()->set(0.0);

        // =====================================================================
        // Assemble system (placeholder - would use forms in real code)
        // =====================================================================

        // In a real implementation, you would:
        // 1. Define UFL forms
        // 2. Compile with FFCx
        // 3. Assemble with dolfinx::fem::petsc::assemble_matrix/vector

        // For demonstration, we create a simple placeholder
        if (rank == 0) {
            std::cout << "\n[Note: This example shows the Ginkgo solver setup." << std::endl;
            std::cout << "In practice, use DOLFINx/multiphenicsx for form assembly.]" << std::endl;
        }

        // =====================================================================
        // Solve with Ginkgo
        // =====================================================================

        // Determine backend
        dgko::Backend backend = dgko::Backend::OMP;  // Default to OpenMP

#ifdef DOLFINX_GINKGO_ENABLE_CUDA
        if (dgko::is_backend_available(dgko::Backend::CUDA)) {
            backend = dgko::Backend::CUDA;
            if (rank == 0) {
                std::cout << "\nUsing CUDA backend with "
                          << dgko::get_device_count(dgko::Backend::CUDA)
                          << " device(s)" << std::endl;
            }
        }
#endif

#ifdef DOLFINX_GINKGO_ENABLE_HIP
        if (backend == dgko::Backend::OMP &&
            dgko::is_backend_available(dgko::Backend::HIP)) {
            backend = dgko::Backend::HIP;
            if (rank == 0) {
                std::cout << "\nUsing HIP backend" << std::endl;
            }
        }
#endif

        if (backend == dgko::Backend::OMP && rank == 0) {
            std::cout << "\nUsing OpenMP backend (no GPU available)" << std::endl;
        }

        // Create executor and communicator
        auto exec = dgko::create_executor(backend, 0);
        auto gko_comm = dgko::create_communicator(comm);

        // Configure solver
        dgko::SolverConfig config;
        config.solver = dgko::SolverType::CG;
        config.preconditioner = dgko::PreconditionerType::AMG;
        config.rtol = 1e-8;
        config.atol = 1e-12;
        config.max_iterations = 500;
        config.verbose = (rank == 0);

        // AMG configuration
        config.amg.max_levels = 10;
        config.amg.cycle = dgko::AMGConfig::Cycle::V;
        config.amg.smoother = dgko::AMGConfig::Smoother::JACOBI;
        config.amg.coarse_solver = dgko::AMGConfig::CoarseSolver::DIRECT;

        if (rank == 0) {
            std::cout << "\nSolver configuration:" << std::endl;
            std::cout << "  Solver: CG" << std::endl;
            std::cout << "  Preconditioner: AMG (PGM coarsening)" << std::endl;
            std::cout << "  Relative tolerance: " << config.rtol << std::endl;
            std::cout << "  Max iterations: " << config.max_iterations << std::endl;
        }

        // NOTE: In a complete example, you would:
        // 1. Create PETSc matrix A from assembled DOLFINx forms
        // 2. Create Ginkgo matrix: dgko::create_distributed_matrix_from_petsc(exec, gko_comm, A)
        // 3. Create solver: dgko::DistributedSolver<>(exec, gko_comm, config)
        // 4. Set operator: solver.set_operator(A_gko)
        // 5. Convert vectors: b_gko, x_gko
        // 6. Solve: solver.solve(b_gko, x_gko)
        // 7. Copy back: dgko::copy_to_petsc(x_gko, x)

        if (rank == 0) {
            std::cout << "\n[Solver setup complete - ready for matrix/vector input]" << std::endl;
            std::cout << "\n=== Example finished ===" << std::endl;
        }
    }

    PetscFinalize();
    return 0;
}
