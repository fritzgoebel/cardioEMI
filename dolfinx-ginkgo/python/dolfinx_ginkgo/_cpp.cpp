// Copyright (c) 2026 CardioEMI Project
// SPDX-License-Identifier: MIT

/// @file _cpp.cpp
/// @brief nanobind Python bindings for dolfinx-ginkgo

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <dolfinx_ginkgo/ginkgo.h>

#include <petsc4py/petsc4py.h>
#include <mpi4py/mpi4py.h>

namespace nb = nanobind;
using namespace dolfinx_ginkgo;

// Type aliases for commonly used Ginkgo types
using GkoExecutor = gko::Executor;
using GkoCommunicator = gko::experimental::mpi::communicator;
using GkoDistMatrix = gko_dist::Matrix<double, std::int32_t, std::int64_t>;
using GkoDdMatrix = gko_dist::DdMatrix<double, std::int32_t, std::int64_t>;
using GkoDistVector = gko_dist::Vector<double>;

// Helper to convert mpi4py communicator to MPI_Comm
MPI_Comm get_mpi_comm(nb::object comm_obj) {
    PyObject* py_comm = comm_obj.ptr();
    MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm);
    if (!comm_ptr) {
        throw std::runtime_error("Invalid MPI communicator");
    }
    return *comm_ptr;
}

// Helper to convert petsc4py Mat to PETSc Mat
Mat get_petsc_mat(nb::object mat_obj) {
    PyObject* py_mat = mat_obj.ptr();
    Mat mat = nullptr;
    // petsc4py provides PetscMat_Get function
    if (PyObject_TypeCheck(py_mat, &PyPetscMat_Type)) {
        mat = PyPetscMat_Get(py_mat);
    }
    if (!mat) {
        throw std::runtime_error("Invalid PETSc Mat");
    }
    return mat;
}

// Helper to convert petsc4py Vec to PETSc Vec
Vec get_petsc_vec(nb::object vec_obj) {
    PyObject* py_vec = vec_obj.ptr();
    Vec vec = nullptr;
    if (PyObject_TypeCheck(py_vec, &PyPetscVec_Type)) {
        vec = PyPetscVec_Get(py_vec);
    }
    if (!vec) {
        throw std::runtime_error("Invalid PETSc Vec");
    }
    return vec;
}

NB_MODULE(_cpp, m) {
    // Initialize petsc4py and mpi4py
    if (import_petsc4py() < 0) {
        throw std::runtime_error("Failed to import petsc4py");
    }
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }

    m.doc() = "C++ bindings for dolfinx-ginkgo";

    // =========================================================================
    // Enums
    // =========================================================================

    nb::enum_<Backend>(m, "Backend", "Compute backend selection")
        .value("REFERENCE", Backend::REFERENCE, "Reference (sequential CPU)")
        .value("OMP", Backend::OMP, "OpenMP (parallel CPU)")
        .value("CUDA", Backend::CUDA, "NVIDIA CUDA")
        .value("HIP", Backend::HIP, "AMD HIP (ROCm)")
        .value("SYCL", Backend::SYCL, "Intel SYCL (oneAPI)")
        .export_values();

    nb::enum_<SolverType>(m, "SolverType", "Krylov solver type")
        .value("CG", SolverType::CG, "Conjugate Gradient")
        .value("FCG", SolverType::FCG, "Flexible Conjugate Gradient")
        .value("GMRES", SolverType::GMRES, "GMRES")
        .value("BICGSTAB", SolverType::BICGSTAB, "BiCGSTAB")
        .value("CGS", SolverType::CGS, "CGS")
        .export_values();

    nb::enum_<PreconditionerType>(m, "PreconditionerType", "Preconditioner type")
        .value("NONE", PreconditionerType::NONE, "No preconditioner")
        .value("JACOBI", PreconditionerType::JACOBI, "Point Jacobi")
        .value("BLOCK_JACOBI", PreconditionerType::BLOCK_JACOBI, "Block Jacobi")
        .value("ILU", PreconditionerType::ILU, "Incomplete LU")
        .value("IC", PreconditionerType::IC, "Incomplete Cholesky")
        .value("ISAI", PreconditionerType::ISAI, "Approximate sparse inverse")
        .value("AMG", PreconditionerType::AMG, "Algebraic multigrid")
        .value("BDDC", PreconditionerType::BDDC, "BDDC (requires DdMatrix)")
        .export_values();

    // =========================================================================
    // Ginkgo Type Wrappers (opaque handles)
    // =========================================================================

    // These are opaque handles - users don't interact with them directly,
    // they just pass them to functions. nanobind handles shared_ptr automatically.
    nb::class_<GkoExecutor>(m, "Executor",
        "Ginkgo executor (opaque handle)");

    nb::class_<GkoCommunicator>(m, "Communicator",
        "Ginkgo MPI communicator (opaque handle)");

    nb::class_<GkoDistMatrix>(m, "DistributedMatrix",
        "Ginkgo distributed matrix (opaque handle)");

    nb::class_<GkoDdMatrix>(m, "DdMatrix",
        "Ginkgo domain decomposition matrix (opaque handle)");

    nb::class_<GkoDistVector>(m, "DistributedVector",
        "Ginkgo distributed vector (opaque handle)");

    // =========================================================================
    // AMG Configuration
    // =========================================================================

    nb::class_<AMGConfig> amg_config(m, "AMGConfig", "AMG preconditioner configuration");

    nb::enum_<AMGConfig::Cycle>(amg_config, "Cycle", "Multigrid cycle type")
        .value("V", AMGConfig::Cycle::V)
        .value("W", AMGConfig::Cycle::W)
        .value("F", AMGConfig::Cycle::F)
        .export_values();

    nb::enum_<AMGConfig::Smoother>(amg_config, "Smoother", "Smoother type")
        .value("JACOBI", AMGConfig::Smoother::JACOBI)
        .value("GAUSS_SEIDEL", AMGConfig::Smoother::GAUSS_SEIDEL)
        .value("ILU", AMGConfig::Smoother::ILU)
        .export_values();

    nb::enum_<AMGConfig::CoarseSolver>(amg_config, "CoarseSolver", "Coarse level solver")
        .value("DIRECT", AMGConfig::CoarseSolver::DIRECT)
        .value("CG", AMGConfig::CoarseSolver::CG)
        .value("GMRES", AMGConfig::CoarseSolver::GMRES)
        .export_values();

    amg_config
        .def(nb::init<>())
        .def_rw("max_levels", &AMGConfig::max_levels)
        .def_rw("min_coarse_rows", &AMGConfig::min_coarse_rows)
        .def_rw("deterministic", &AMGConfig::deterministic)
        .def_rw("cycle", &AMGConfig::cycle)
        .def_rw("smoother", &AMGConfig::smoother)
        .def_rw("pre_smooth_steps", &AMGConfig::pre_smooth_steps)
        .def_rw("post_smooth_steps", &AMGConfig::post_smooth_steps)
        .def_rw("relaxation_factor", &AMGConfig::relaxation_factor)
        .def_rw("coarse_solver", &AMGConfig::coarse_solver)
        .def_rw("coarse_max_iterations", &AMGConfig::coarse_max_iterations)
        .def_rw("coarse_tolerance", &AMGConfig::coarse_tolerance)
        .def_rw("use_mixed_precision", &AMGConfig::use_mixed_precision)
        .def_rw("mixed_precision_level", &AMGConfig::mixed_precision_level);

    // =========================================================================
    // BDDC Configuration
    // =========================================================================

    nb::class_<BDDCConfig> bddc_config(m, "BDDCConfig", "BDDC preconditioner configuration");

    nb::enum_<BDDCConfig::Scaling>(bddc_config, "Scaling", "Scaling type for interface weights")
        .value("STIFFNESS", BDDCConfig::Scaling::STIFFNESS)
        .value("DELUXE", BDDCConfig::Scaling::DELUXE)
        .export_values();

    nb::enum_<BDDCConfig::LocalSolver>(bddc_config, "LocalSolver", "Local subdomain solver")
        .value("DIRECT", BDDCConfig::LocalSolver::DIRECT)
        .value("ILU", BDDCConfig::LocalSolver::ILU)
        .value("IC", BDDCConfig::LocalSolver::IC)
        .value("AMG", BDDCConfig::LocalSolver::AMG)
        .export_values();

    nb::enum_<BDDCConfig::CoarseSolver>(bddc_config, "CoarseSolver", "Coarse level solver")
        .value("CG", BDDCConfig::CoarseSolver::CG)
        .value("GMRES", BDDCConfig::CoarseSolver::GMRES)
        .value("BDDC", BDDCConfig::CoarseSolver::BDDC)
        .export_values();

    // Local AMG configuration (nested in BDDCConfig)
    nb::class_<BDDCConfig::LocalAMGConfig> local_amg_config(bddc_config, "LocalAMGConfig",
        "AMG configuration for local subdomain solver");

    nb::enum_<BDDCConfig::LocalAMGConfig::Smoother>(local_amg_config, "Smoother", "Smoother type")
        .value("JACOBI", BDDCConfig::LocalAMGConfig::Smoother::JACOBI)
        .value("GAUSS_SEIDEL", BDDCConfig::LocalAMGConfig::Smoother::GAUSS_SEIDEL)
        .value("ILU", BDDCConfig::LocalAMGConfig::Smoother::ILU)
        .export_values();

    nb::enum_<BDDCConfig::LocalAMGConfig::CoarseSolver>(local_amg_config, "CoarseSolver",
        "Coarsest level solver for local AMG")
        .value("DIRECT", BDDCConfig::LocalAMGConfig::CoarseSolver::DIRECT)
        .value("CG", BDDCConfig::LocalAMGConfig::CoarseSolver::CG)
        .value("GMRES", BDDCConfig::LocalAMGConfig::CoarseSolver::GMRES)
        .export_values();

    local_amg_config
        .def(nb::init<>())
        .def_rw("max_levels", &BDDCConfig::LocalAMGConfig::max_levels)
        .def_rw("min_coarse_rows", &BDDCConfig::LocalAMGConfig::min_coarse_rows)
        .def_rw("smoother", &BDDCConfig::LocalAMGConfig::smoother)
        .def_rw("smooth_steps", &BDDCConfig::LocalAMGConfig::smooth_steps)
        .def_rw("relaxation_factor", &BDDCConfig::LocalAMGConfig::relaxation_factor)
        .def_rw("coarse_solver", &BDDCConfig::LocalAMGConfig::coarse_solver)
        .def_rw("coarse_max_iterations", &BDDCConfig::LocalAMGConfig::coarse_max_iterations);

    bddc_config
        .def(nb::init<>())
        .def_rw("vertices", &BDDCConfig::vertices)
        .def_rw("edges", &BDDCConfig::edges)
        .def_rw("faces", &BDDCConfig::faces)
        .def_rw("scaling", &BDDCConfig::scaling)
        .def_rw("local_solver", &BDDCConfig::local_solver)
        .def_rw("local_max_iterations", &BDDCConfig::local_max_iterations)
        .def_rw("local_tolerance", &BDDCConfig::local_tolerance)
        .def_rw("local_amg", &BDDCConfig::local_amg)
        .def_rw("coarse_solver", &BDDCConfig::coarse_solver)
        .def_rw("coarse_max_iterations", &BDDCConfig::coarse_max_iterations)
        .def_rw("coarse_tolerance", &BDDCConfig::coarse_tolerance)
        .def_rw("coarse_bddc_local_solver", &BDDCConfig::coarse_bddc_local_solver)
        .def_rw("repartition_coarse", &BDDCConfig::repartition_coarse)
        .def_rw("constant_nullspace", &BDDCConfig::constant_nullspace);

    // =========================================================================
    // Solver Configuration
    // =========================================================================

    nb::class_<SolverConfig>(m, "SolverConfig", "Solver configuration")
        .def(nb::init<>())
        .def_rw("solver", &SolverConfig::solver)
        .def_rw("preconditioner", &SolverConfig::preconditioner)
        .def_rw("rtol", &SolverConfig::rtol)
        .def_rw("atol", &SolverConfig::atol)
        .def_rw("max_iterations", &SolverConfig::max_iterations)
        .def_rw("krylov_dim", &SolverConfig::krylov_dim)
        .def_rw("jacobi_block_size", &SolverConfig::jacobi_block_size)
        .def_rw("ilu_fill_level", &SolverConfig::ilu_fill_level)
        .def_rw("amg", &SolverConfig::amg)
        .def_rw("bddc", &SolverConfig::bddc)
        .def_rw("verbose", &SolverConfig::verbose);

    // =========================================================================
    // Executor Functions
    // =========================================================================

    m.def("create_executor", &create_executor,
          nb::arg("backend"), nb::arg("device_id") = 0,
          "Create a Ginkgo executor for the specified backend");

    m.def("create_communicator",
          [](nb::object comm_obj) {
              MPI_Comm comm = get_mpi_comm(comm_obj);
              return create_communicator(comm);
          },
          nb::arg("comm"),
          "Create a Ginkgo MPI communicator wrapper");

    m.def("is_backend_available", &is_backend_available,
          nb::arg("backend"),
          "Check if a backend is available");

    m.def("get_device_count", &get_device_count,
          nb::arg("backend"),
          "Get the number of available devices for a backend");

    // =========================================================================
    // Matrix/Vector Conversion Functions
    // =========================================================================

    m.def("create_distributed_matrix_from_petsc",
          [](std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
             nb::object mat_obj) {
              Mat petsc_mat = get_petsc_mat(mat_obj);
              return create_distributed_matrix_from_petsc<>(exec, gko_comm, petsc_mat);
          },
          nb::arg("exec"), nb::arg("comm"), nb::arg("mat"),
          "Create Ginkgo distributed matrix from PETSc Mat");

    m.def("create_distributed_vector_from_petsc",
          [](std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
             nb::object vec_obj) {
              Vec petsc_vec = get_petsc_vec(vec_obj);
              return create_distributed_vector_from_petsc<>(exec, gko_comm, petsc_vec);
          },
          nb::arg("exec"), nb::arg("comm"), nb::arg("vec"),
          "Create Ginkgo distributed vector from PETSc Vec");

    m.def("copy_to_petsc",
          [](const gko_dist::Vector<double>& gko_vec, nb::object vec_obj) {
              Vec petsc_vec = get_petsc_vec(vec_obj);
              copy_to_petsc(gko_vec, petsc_vec);
          },
          nb::arg("gko_vec"), nb::arg("petsc_vec"),
          "Copy Ginkgo distributed vector to PETSc Vec");

    m.def("update_matrix_values_from_petsc",
          [](std::shared_ptr<gko_dist::Matrix<double, std::int32_t, std::int64_t>> gko_mat,
             nb::object mat_obj) {
              Mat petsc_mat = get_petsc_mat(mat_obj);
              update_matrix_values_from_petsc(gko_mat, petsc_mat);
          },
          nb::arg("gko_mat"), nb::arg("petsc_mat"),
          "Update Ginkgo matrix values from PETSc Mat");

    // =========================================================================
    // Native Assembly Functions (bypass PETSc)
    // =========================================================================

    m.def("create_distributed_matrix_from_local_coo",
          [](std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_indices,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> col_indices,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> values,
             std::int64_t global_rows,
             std::int64_t global_cols,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_ranges) {
              // Convert ndarray to std::vector
              std::vector<std::int64_t> rows(row_indices.data(),
                                              row_indices.data() + row_indices.size());
              std::vector<std::int64_t> cols(col_indices.data(),
                                              col_indices.data() + col_indices.size());
              std::vector<double> vals(values.data(),
                                       values.data() + values.size());
              std::vector<std::int64_t> ranges(row_ranges.data(),
                                                row_ranges.data() + row_ranges.size());

              return create_distributed_matrix_from_local_coo<>(
                  exec, gko_comm, rows, cols, vals, global_rows, global_cols, ranges);
          },
          nb::arg("exec"), nb::arg("comm"),
          nb::arg("row_indices"), nb::arg("col_indices"), nb::arg("values"),
          nb::arg("global_rows"), nb::arg("global_cols"), nb::arg("row_ranges"),
          "Create Ginkgo distributed matrix from local COO data with communication");

    m.def("update_matrix_from_local_coo",
          [](std::shared_ptr<gko_dist::Matrix<double, std::int32_t, std::int64_t>> gko_mat,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_indices,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> col_indices,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> values,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_ranges) {
              std::vector<std::int64_t> rows(row_indices.data(),
                                              row_indices.data() + row_indices.size());
              std::vector<std::int64_t> cols(col_indices.data(),
                                              col_indices.data() + col_indices.size());
              std::vector<double> vals(values.data(),
                                       values.data() + values.size());
              std::vector<std::int64_t> ranges(row_ranges.data(),
                                                row_ranges.data() + row_ranges.size());

              update_matrix_from_local_coo(gko_mat, rows, cols, vals, ranges);
          },
          nb::arg("gko_mat"),
          nb::arg("row_indices"), nb::arg("col_indices"), nb::arg("values"),
          nb::arg("row_ranges"),
          "Update Ginkgo matrix values from local COO data with communication");

    // =========================================================================
    // DdMatrix (Domain Decomposition Matrix) Functions
    // =========================================================================

    m.def("create_dd_matrix_from_local_coo",
          [](std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_indices,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> col_indices,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> values,
             std::int64_t global_rows,
             std::int64_t global_cols,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_ranges) {
              std::vector<std::int64_t> rows(row_indices.data(),
                                              row_indices.data() + row_indices.size());
              std::vector<std::int64_t> cols(col_indices.data(),
                                              col_indices.data() + col_indices.size());
              std::vector<double> vals(values.data(),
                                       values.data() + values.size());
              std::vector<std::int64_t> ranges(row_ranges.data(),
                                                row_ranges.data() + row_ranges.size());

              return create_dd_matrix_from_local_coo<>(
                  exec, gko_comm, rows, cols, vals, global_rows, global_cols, ranges);
          },
          nb::arg("exec"), nb::arg("comm"),
          nb::arg("row_indices"), nb::arg("col_indices"), nb::arg("values"),
          nb::arg("global_rows"), nb::arg("global_cols"), nb::arg("row_ranges"),
          "Create Ginkgo DdMatrix from local COO data for domain decomposition");

    m.def("update_dd_matrix_from_local_coo",
          [](std::shared_ptr<gko_dist::DdMatrix<double, std::int32_t, std::int64_t>> dd_mat,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_indices,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> col_indices,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> values,
             nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig> row_ranges) {
              std::vector<std::int64_t> rows(row_indices.data(),
                                              row_indices.data() + row_indices.size());
              std::vector<std::int64_t> cols(col_indices.data(),
                                              col_indices.data() + col_indices.size());
              std::vector<double> vals(values.data(),
                                       values.data() + values.size());
              std::vector<std::int64_t> ranges(row_ranges.data(),
                                                row_ranges.data() + row_ranges.size());

              update_dd_matrix_from_local_coo(dd_mat, rows, cols, vals, ranges);
          },
          nb::arg("dd_mat"),
          nb::arg("row_indices"), nb::arg("col_indices"), nb::arg("values"),
          nb::arg("row_ranges"),
          "Update Ginkgo DdMatrix values from local COO data");

    // =========================================================================
    // Apply (SpMV) Functions
    // =========================================================================

    m.def("apply_distributed",
          [](std::shared_ptr<GkoDistMatrix> A,
             std::shared_ptr<GkoDistVector> x,
             std::shared_ptr<GkoDistVector> y) {
              A->apply(x.get(), y.get());
          },
          nb::arg("A"), nb::arg("x"), nb::arg("y"),
          "Compute y = A * x for a distributed matrix");

    m.def("apply_dd",
          [](std::shared_ptr<GkoDdMatrix> A,
             std::shared_ptr<GkoDistVector> x,
             std::shared_ptr<GkoDistVector> y) {
              A->apply(x.get(), y.get());
          },
          nb::arg("A"), nb::arg("x"), nb::arg("y"),
          "Compute y = A * x for a DdMatrix");

    m.def("create_distributed_vector_from_local",
          [](std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<gko::experimental::mpi::communicator> gko_comm,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> local_values,
             std::int64_t global_size) {
              std::vector<double> vals(local_values.data(),
                                       local_values.data() + local_values.size());
              return create_distributed_vector_from_local<>(
                  exec, gko_comm, vals, global_size);
          },
          nb::arg("exec"), nb::arg("comm"),
          nb::arg("local_values"), nb::arg("global_size"),
          "Create Ginkgo distributed vector from local numpy array");

    // =========================================================================
    // Distributed Solver
    // =========================================================================

    using Solver = DistributedSolver<double, std::int32_t, std::int64_t>;

    nb::class_<Solver>(m, "DistributedSolver", "Ginkgo distributed linear solver")
        .def(nb::init<std::shared_ptr<gko::Executor>,
                      std::shared_ptr<gko::experimental::mpi::communicator>,
                      const SolverConfig&>(),
             nb::arg("exec"), nb::arg("comm"), nb::arg("config") = SolverConfig{})
        .def("set_operator",
             static_cast<void (Solver::*)(std::shared_ptr<GkoDistMatrix>)>(&Solver::set_operator),
             nb::arg("A"), "Set the system matrix (DistributedMatrix)")
        .def("set_operator_dd",
             [](Solver& solver, std::shared_ptr<GkoDdMatrix> A) {
                 solver.set_operator(std::static_pointer_cast<gko::LinOp>(A));
             },
             nb::arg("A"), "Set the system matrix (DdMatrix)")
        .def("set_tolerance", &Solver::set_tolerance,
             nb::arg("rtol"), nb::arg("atol") = 1e-12,
             "Update convergence tolerances")
        .def("set_max_iterations", &Solver::set_max_iterations,
             nb::arg("max_iter"), "Update maximum iterations")
        .def("solve",
             [](Solver& solver,
                std::shared_ptr<gko_dist::Vector<double>> b,
                std::shared_ptr<gko_dist::Vector<double>> x) {
                 return solver.solve(*b, *x);
             },
             nb::arg("b"), nb::arg("x"),
             "Solve Ax = b")
        .def("iterations", &Solver::iterations,
             "Get iteration count from last solve")
        .def("residual_norm", &Solver::residual_norm,
             "Get final residual norm from last solve")
        .def("converged", &Solver::converged,
             "Check if last solve converged");
}
