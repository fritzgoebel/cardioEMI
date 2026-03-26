// app-solver.js - Solver settings UI and config loading from YAML

App.prototype.setupSolverSettings = function() {
    const backendSelect = document.getElementById('solver-backend');
    const petscOptions = document.getElementById('petsc-options');
    const ginkgoOptions = document.getElementById('ginkgo-options');
    const amgOptions = document.getElementById('amg-options');
    const bddcOptions = document.getElementById('bddc-options');
    const ginkgoPrecond = document.getElementById('ginkgo-precond');
    const petscKspType = document.getElementById('petsc-ksp-type');
    const petscPcType = document.getElementById('petsc-pc-type');
    const petscPcRow = document.getElementById('petsc-pc-row');

    // Initialize solver config state
    this.solverConfig = {
        backend: 'petsc',
        petsc: {
            kspType: 'preonly',
            pcType: 'lu',
            bddc: {
                scaling: 'stiffness',
                localSolver: 'mumps',
                coarseSolver: 'mumps',
                coarsePcType: 'lu',
                useVertices: true,
                useEdges: true,
                useFaces: false
            }
        },
        ginkgo: {
            nativeAssembly: true,
            ddMatrix: false,
            backend: 'omp',
            solver: 'cg',
            preconditioner: 'jacobi',
            amg: { cycle: 'v', smoother: 'jacobi', maxLevels: 10 },
            bddc: {
                localSolver: 'direct',
                localMaxIterations: 100,
                localTolerance: 1e-12,
                coarseSolver: 'cg',
                coarseMaxIterations: 100,
                coarseBddcLocalSolver: 'direct',
                vertices: true,
                edges: true,
                faces: true,
                repartitionCoarse: true,
                localAmg: {
                    coarsening: 'pgm',
                    strengthThreshold: 0.25,
                    cycle: 'v',
                    smoother: 'jacobi',
                    smoothSteps: 1,
                    maxLevels: 10,
                    coarseSolver: 'direct',
                    relaxationFactor: 0.9
                },
                localHypre: {
                    cycleType: 1,
                    coarseningType: 10,
                    strengthThreshold: 0.25,
                    smootherType: 6,
                    numSweeps: 1,
                    interpolationType: 0,
                    maxLevels: 25
                }
            }
        },
        rtol: '1e-8',
        atol: '1e-12',
        maxIterations: 1000
    };

    // Get DOM elements for DD matrix logic
    const ddMatrixRow = document.getElementById('dd-matrix-row');
    const ddMatrixCheckbox = document.getElementById('ginkgo-dd-matrix');
    const precondRow = ginkgoPrecond.closest('.param-row');

    // Backend selection
    backendSelect.addEventListener('change', () => {
        this.solverConfig.backend = backendSelect.value;
        if (backendSelect.value === 'petsc') {
            petscOptions.style.display = 'block';
            ginkgoOptions.style.display = 'none';
        } else {
            petscOptions.style.display = 'none';
            ginkgoOptions.style.display = 'block';
        }
    });

    // PETSc KSP type
    petscKspType.addEventListener('change', () => {
        this.solverConfig.petsc.kspType = petscKspType.value;
        if (petscKspType.value === 'preonly') {
            petscPcRow.style.display = 'none';
            petscPcType.value = 'lu';
            this.solverConfig.petsc.pcType = 'lu';
        } else {
            petscPcRow.style.display = 'flex';
        }
    });

    // PETSc preconditioner
    const petscBddcOptions = document.getElementById('petsc-bddc-options');
    petscPcType.addEventListener('change', () => {
        this.solverConfig.petsc.pcType = petscPcType.value;
        petscBddcOptions.style.display = petscPcType.value === 'bddc' ? 'block' : 'none';
    });

    // PETSc BDDC options
    document.getElementById('petsc-bddc-scaling').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.scaling = e.target.value;
    });
    document.getElementById('petsc-bddc-local-solver').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.localSolver = e.target.value;
    });
    document.getElementById('petsc-bddc-coarse-solver').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.coarseSolver = e.target.value;
    });
    document.getElementById('petsc-bddc-coarse-pc').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.coarsePcType = e.target.value;
    });
    document.getElementById('petsc-bddc-vertices').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.useVertices = e.target.checked;
    });
    document.getElementById('petsc-bddc-edges').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.useEdges = e.target.checked;
    });
    document.getElementById('petsc-bddc-faces').addEventListener('change', (e) => {
        this.solverConfig.petsc.bddc.useFaces = e.target.checked;
    });

    // Ginkgo native assembly
    document.getElementById('ginkgo-native-assembly').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.nativeAssembly = e.target.checked;
        if (!e.target.checked) {
            ddMatrixCheckbox.checked = false;
            this.solverConfig.ginkgo.ddMatrix = false;
            ddMatrixRow.style.display = 'none';
            precondRow.style.display = 'flex';
        } else {
            ddMatrixRow.style.display = 'flex';
        }
    });

    // Ginkgo DD matrix
    ddMatrixCheckbox.addEventListener('change', (e) => {
        this.solverConfig.ginkgo.ddMatrix = e.target.checked;
        if (e.target.checked) {
            this.solverConfig.ginkgo.preconditioner = 'bddc';
            ginkgoPrecond.value = 'bddc';
            amgOptions.style.display = 'none';
            bddcOptions.style.display = 'block';
        } else {
            this.solverConfig.ginkgo.preconditioner = 'jacobi';
            ginkgoPrecond.value = 'jacobi';
            bddcOptions.style.display = 'none';
        }
    });

    // Ginkgo backend
    document.getElementById('ginkgo-backend').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.backend = e.target.value;
    });

    // Ginkgo solver
    document.getElementById('ginkgo-solver').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.solver = e.target.value;
    });

    // Ginkgo preconditioner
    ginkgoPrecond.addEventListener('change', () => {
        this.solverConfig.ginkgo.preconditioner = ginkgoPrecond.value;
        amgOptions.style.display = ginkgoPrecond.value === 'amg' ? 'block' : 'none';
        bddcOptions.style.display = ginkgoPrecond.value === 'bddc' ? 'block' : 'none';

        if (ginkgoPrecond.value === 'bddc') {
            ddMatrixCheckbox.checked = true;
            this.solverConfig.ginkgo.ddMatrix = true;
        }
    });

    // AMG options
    document.getElementById('amg-cycle').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.amg.cycle = e.target.value;
    });
    document.getElementById('amg-smoother').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.amg.smoother = e.target.value;
    });
    document.getElementById('amg-max-levels').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.amg.maxLevels = parseInt(e.target.value);
    });

    // BDDC options
    const bddcLocalAmgOptions = document.getElementById('bddc-local-amg-options');
    const bddcLocalHypreOptions = document.getElementById('bddc-local-hypre-options');
    const bddcLocalStoppingOptions = document.getElementById('bddc-local-stopping-options');
    document.getElementById('bddc-local-solver').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localSolver = e.target.value;
        bddcLocalAmgOptions.style.display = e.target.value === 'amg' ? 'block' : 'none';
        bddcLocalHypreOptions.style.display = e.target.value === 'hypre' ? 'block' : 'none';
        bddcLocalStoppingOptions.style.display = (e.target.value !== 'direct' && e.target.value !== 'direct_lu') ? 'block' : 'none';
    });

    document.getElementById('bddc-local-max-iter').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localMaxIterations = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-tolerance').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localTolerance = parseFloat(e.target.value);
    });

    const bddcCoarseBddcOptions = document.getElementById('bddc-coarse-bddc-options');
    document.getElementById('bddc-coarse-solver').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.coarseSolver = e.target.value;
        bddcCoarseBddcOptions.style.display = e.target.value === 'bddc' ? 'flex' : 'none';
    });
    document.getElementById('bddc-coarse-max-iter').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.coarseMaxIterations = parseInt(e.target.value);
    });

    document.getElementById('bddc-vertices').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.vertices = e.target.checked;
    });
    document.getElementById('bddc-edges').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.edges = e.target.checked;
    });
    document.getElementById('bddc-faces').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.faces = e.target.checked;
    });
    document.getElementById('bddc-repartition-coarse').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.repartitionCoarse = e.target.checked;
    });

    // BDDC local AMG options
    const bddcLocalAmgHmisOptions = document.getElementById('bddc-local-amg-hmis-options');
    document.getElementById('bddc-local-amg-coarsening').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.coarsening = e.target.value;
        bddcLocalAmgHmisOptions.style.display = e.target.value === 'hmis' ? 'block' : 'none';
    });
    document.getElementById('bddc-local-amg-strength-threshold').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.strengthThreshold = parseFloat(e.target.value);
    });
    document.getElementById('bddc-local-amg-cycle').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.cycle = e.target.value;
    });
    document.getElementById('bddc-local-amg-smoother').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.smoother = e.target.value;
    });
    document.getElementById('bddc-local-amg-smooth-steps').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.smoothSteps = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-amg-max-levels').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.maxLevels = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-amg-coarse-solver').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.coarseSolver = e.target.value;
    });
    document.getElementById('bddc-local-amg-relaxation').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localAmg.relaxationFactor = parseFloat(e.target.value);
    });
    document.getElementById('bddc-coarse-bddc-local-solver').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.coarseBddcLocalSolver = e.target.value;
    });

    // BDDC local Hypre BoomerAMG options
    document.getElementById('bddc-local-hypre-cycle').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.cycleType = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-hypre-coarsening').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.coarseningType = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-hypre-strength').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.strengthThreshold = parseFloat(e.target.value);
    });
    document.getElementById('bddc-local-hypre-smoother').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.smootherType = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-hypre-sweeps').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.numSweeps = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-hypre-interpolation').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.interpolationType = parseInt(e.target.value);
    });
    document.getElementById('bddc-local-hypre-max-levels').addEventListener('change', (e) => {
        this.solverConfig.ginkgo.bddc.localHypre.maxLevels = parseInt(e.target.value);
    });

    // Tolerance and max iterations
    document.getElementById('solver-rtol').addEventListener('change', (e) => {
        this.solverConfig.rtol = e.target.value;
    });
    document.getElementById('solver-atol').addEventListener('change', (e) => {
        this.solverConfig.atol = e.target.value;
    });
    document.getElementById('solver-max-iter').addEventListener('change', (e) => {
        this.solverConfig.maxIterations = parseInt(e.target.value);
    });

    this.initSolverSettingsUI();
};

App.prototype.initSolverSettingsUI = function() {
    const backendSelect = document.getElementById('solver-backend');
    const petscOptions = document.getElementById('petsc-options');
    const ginkgoOptions = document.getElementById('ginkgo-options');
    const amgOptions = document.getElementById('amg-options');
    const bddcOptions = document.getElementById('bddc-options');
    const bddcLocalAmgOptions = document.getElementById('bddc-local-amg-options');
    const petscKspType = document.getElementById('petsc-ksp-type');
    const petscPcRow = document.getElementById('petsc-pc-row');
    const ginkgoPrecond = document.getElementById('ginkgo-precond');
    const nativeAssemblyCheckbox = document.getElementById('ginkgo-native-assembly');
    const ddMatrixCheckbox = document.getElementById('ginkgo-dd-matrix');
    const ddMatrixRow = document.getElementById('dd-matrix-row');
    const bddcLocalSolver = document.getElementById('bddc-local-solver');

    if (backendSelect.value === 'ginkgo') {
        petscOptions.style.display = 'none';
        ginkgoOptions.style.display = 'block';
    } else {
        petscOptions.style.display = 'block';
        ginkgoOptions.style.display = 'none';
    }

    if (petscKspType.value === 'preonly') {
        petscPcRow.style.display = 'none';
    } else {
        petscPcRow.style.display = 'flex';
    }

    const petscBddcOptions = document.getElementById('petsc-bddc-options');
    const petscPcType = document.getElementById('petsc-pc-type');
    petscBddcOptions.style.display = petscPcType.value === 'bddc' ? 'block' : 'none';

    if (!nativeAssemblyCheckbox.checked) {
        ddMatrixRow.style.display = 'none';
    } else {
        ddMatrixRow.style.display = 'flex';
    }

    amgOptions.style.display = ginkgoPrecond.value === 'amg' ? 'block' : 'none';
    bddcOptions.style.display = ginkgoPrecond.value === 'bddc' ? 'block' : 'none';

    bddcLocalAmgOptions.style.display = bddcLocalSolver.value === 'amg' ? 'block' : 'none';
    document.getElementById('bddc-local-hypre-options').style.display = bddcLocalSolver.value === 'hypre' ? 'block' : 'none';
    document.getElementById('bddc-local-stopping-options').style.display = (bddcLocalSolver.value !== 'direct' && bddcLocalSolver.value !== 'direct_lu') ? 'block' : 'none';

    const bddcCoarseSolver = document.getElementById('bddc-coarse-solver');
    document.getElementById('bddc-coarse-bddc-options').style.display = bddcCoarseSolver.value === 'bddc' ? 'flex' : 'none';

    this.solverConfig.backend = backendSelect.value;
    this.solverConfig.petsc.kspType = petscKspType.value;
    this.solverConfig.petsc.pcType = document.getElementById('petsc-pc-type').value;
    this.solverConfig.ginkgo.nativeAssembly = nativeAssemblyCheckbox.checked;
    this.solverConfig.ginkgo.ddMatrix = ddMatrixCheckbox.checked;
    this.solverConfig.ginkgo.backend = document.getElementById('ginkgo-backend').value;
    this.solverConfig.ginkgo.solver = document.getElementById('ginkgo-solver').value;
    this.solverConfig.ginkgo.preconditioner = ginkgoPrecond.value;
};

App.prototype.loadConfigFromYaml = async function() {
    try {
        const config = await this.configManager.getConfig();
        if (!config || config.error) return;

        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el && val !== undefined && val !== null) {
                el.value = String(val);
            }
        };
        const setChecked = (id, val) => {
            const el = document.getElementById(id);
            if (el && val !== undefined && val !== null) {
                el.checked = Boolean(val);
            }
        };

        // Top-level solver settings
        setVal('solver-backend', config.solver_backend || 'petsc');
        setVal('petsc-ksp-type', config.ksp_type || 'preonly');
        setVal('petsc-pc-type', config.pc_type || 'lu');
        setVal('solver-rtol', config.ksp_rtol || '1e-8');
        setVal('solver-atol', config.ksp_atol || '1e-12');
        setVal('solver-max-iter', config.max_iterations || 1000);

        // Simulation parameters
        if (config.dt !== undefined) {
            setVal('dt-input', config.dt);
            this.dt = parseFloat(config.dt);
        }
        if (config.time_steps !== undefined) {
            setVal('timesteps-input', config.time_steps);
            this.timeSteps = parseInt(config.time_steps);
        }

        // PETSc BDDC options
        const petscBddc = config.petsc_bddc || {};
        if (petscBddc.scaling) setVal('petsc-bddc-scaling', petscBddc.scaling);
        if (petscBddc.local_solver) setVal('petsc-bddc-local-solver', petscBddc.local_solver);
        if (petscBddc.coarse_solver) setVal('petsc-bddc-coarse-solver', petscBddc.coarse_solver);
        if (petscBddc.coarse_pc_type) setVal('petsc-bddc-coarse-pc', petscBddc.coarse_pc_type);
        if (petscBddc.use_vertices !== undefined) setChecked('petsc-bddc-vertices', petscBddc.use_vertices);
        if (petscBddc.use_edges !== undefined) setChecked('petsc-bddc-edges', petscBddc.use_edges);
        if (petscBddc.use_faces !== undefined) setChecked('petsc-bddc-faces', petscBddc.use_faces);

        // Ginkgo options
        const gko = config.ginkgo || {};
        if (gko.native_assembly !== undefined) setChecked('ginkgo-native-assembly', gko.native_assembly);
        if (gko.dd_matrix !== undefined) setChecked('ginkgo-dd-matrix', gko.dd_matrix);
        if (gko.backend) setVal('ginkgo-backend', gko.backend);
        if (gko.solver) setVal('ginkgo-solver', gko.solver);
        if (gko.preconditioner) setVal('ginkgo-precond', gko.preconditioner);

        // Ginkgo AMG options
        const amg = gko.amg || {};
        if (amg.cycle) setVal('amg-cycle', amg.cycle);
        if (amg.smoother) setVal('amg-smoother', amg.smoother);
        if (amg.max_levels) setVal('amg-max-levels', amg.max_levels);

        // Ginkgo BDDC options
        const bddc = gko.bddc || {};
        if (bddc.local_solver) setVal('bddc-local-solver', bddc.local_solver);
        if (bddc.local_max_iterations) setVal('bddc-local-max-iter', bddc.local_max_iterations);
        if (bddc.local_tolerance) setVal('bddc-local-tolerance', bddc.local_tolerance);
        if (bddc.coarse_solver) setVal('bddc-coarse-solver', bddc.coarse_solver);
        if (bddc.coarse_max_iterations) setVal('bddc-coarse-max-iter', bddc.coarse_max_iterations);
        if (bddc.coarse_bddc_local_solver) setVal('bddc-coarse-bddc-local-solver', bddc.coarse_bddc_local_solver);
        if (bddc.vertices !== undefined) setChecked('bddc-vertices', bddc.vertices);
        if (bddc.edges !== undefined) setChecked('bddc-edges', bddc.edges);
        if (bddc.faces !== undefined) setChecked('bddc-faces', bddc.faces);
        if (bddc.repartition_coarse !== undefined) setChecked('bddc-repartition-coarse', bddc.repartition_coarse);

        // Ginkgo BDDC Local AMG options
        const localAmg = bddc.local_amg || {};
        if (localAmg.coarsening) setVal('bddc-local-amg-coarsening', localAmg.coarsening);
        if (localAmg.strength_threshold) setVal('bddc-local-amg-strength-threshold', localAmg.strength_threshold);
        if (localAmg.cycle) setVal('bddc-local-amg-cycle', localAmg.cycle);
        if (localAmg.smoother) setVal('bddc-local-amg-smoother', localAmg.smoother);
        if (localAmg.smooth_steps) setVal('bddc-local-amg-smooth-steps', localAmg.smooth_steps);
        if (localAmg.max_levels) setVal('bddc-local-amg-max-levels', localAmg.max_levels);
        if (localAmg.coarse_solver) setVal('bddc-local-amg-coarse-solver', localAmg.coarse_solver);
        if (localAmg.relaxation_factor) setVal('bddc-local-amg-relaxation', localAmg.relaxation_factor);

        // Ginkgo BDDC Local Hypre BoomerAMG options
        const localHypre = bddc.local_hypre || {};
        if (localHypre.cycle_type) setVal('bddc-local-hypre-cycle', localHypre.cycle_type);
        if (localHypre.coarsening_type !== undefined) setVal('bddc-local-hypre-coarsening', localHypre.coarsening_type);
        if (localHypre.strength_threshold) setVal('bddc-local-hypre-strength', localHypre.strength_threshold);
        if (localHypre.smoother_type !== undefined) setVal('bddc-local-hypre-smoother', localHypre.smoother_type);
        if (localHypre.num_sweeps) setVal('bddc-local-hypre-sweeps', localHypre.num_sweeps);
        if (localHypre.interpolation_type !== undefined) setVal('bddc-local-hypre-interpolation', localHypre.interpolation_type);
        if (localHypre.max_levels) setVal('bddc-local-hypre-max-levels', localHypre.max_levels);

        this.initSolverSettingsUI();

        console.log('Config loaded from YAML:', this.configManager.configFile);
    } catch (error) {
        console.warn('Could not load config from YAML:', error.message);
    }
};
