// app-simulation.js - Simulation execution (local + remote cluster), conditions snapshot

App.prototype.runSimulation = async function() {
    const statusEl = document.getElementById('simulation-status');
    const outputEl = document.getElementById('simulation-output');
    const runBtn = document.getElementById('run-simulation');

    // First save the configuration
    try {
        statusEl.className = 'status visible running';
        statusEl.textContent = 'Saving configuration...';

        const expr = this.generateVinitExpression();
        const vinitValue = expr.slice(1, -1);

        const solverBackend = document.getElementById('solver-backend').value;
        const kspType = document.getElementById('petsc-ksp-type').value;
        const pcType = document.getElementById('petsc-pc-type').value;
        const rtol = document.getElementById('solver-rtol').value;
        const atol = document.getElementById('solver-atol').value;

        const trackIterResEl = document.getElementById('track-iter-residuals');
        const trackIterRes = !!(trackIterResEl && trackIterResEl.checked);

        const randomRhsEl = document.getElementById('random-rhs');
        const randomRhs = !!(randomRhsEl && randomRhsEl.checked);

        const configUpdates = {
            v_init: vinitValue,
            dt: this.dt,
            time_steps: this.timeSteps,
            save_interval: 10,
            solver_backend: solverBackend,
            ksp_type: kspType,
            pc_type: pcType,
            ksp_rtol: rtol,
            ksp_atol: atol,
            bc_type: this.bcType,
            partition_mode: this.partitionMode,
            component_granularity: this.componentGranularity,
            track_iter_residuals: trackIterRes,
            random_rhs: randomRhs
        };

        this.solverConfig.backend = solverBackend;
        this.solverConfig.petsc.kspType = kspType;
        this.solverConfig.petsc.pcType = pcType;
        this.solverConfig.rtol = rtol;
        this.solverConfig.atol = atol;

        await this.configManager.updateConfig(configUpdates);

        // Update scar tissue config
        const scarConfig = this.getScarConfig();
        const scarPayload = scarConfig || { regions: [], healthy: { sigma_i: 4.0, sigma_e: 20.0 } };
        scarPayload.conversionFactor = this.conversionFactor;
        scarPayload.file = this.configManager.configFile;
        await fetch('/api/config/scar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(scarPayload),
        });

        // PETSc BDDC config
        if (solverBackend === 'petsc' && pcType === 'bddc') {
            const petscBddcConfig = {
                scaling: document.getElementById('petsc-bddc-scaling').value,
                localSolver: document.getElementById('petsc-bddc-local-solver').value,
                coarseSolver: document.getElementById('petsc-bddc-coarse-solver').value,
                coarsePcType: document.getElementById('petsc-bddc-coarse-pc').value,
                useVertices: document.getElementById('petsc-bddc-vertices').checked,
                useEdges: document.getElementById('petsc-bddc-edges').checked,
                useFaces: document.getElementById('petsc-bddc-faces').checked
            };
            if (petscBddcConfig.localSolver === 'hypre') {
                petscBddcConfig.localHypre = {
                    cycleType: document.getElementById('petsc-bddc-local-hypre-cycle').value,
                    coarsenType: document.getElementById('petsc-bddc-local-hypre-coarsen').value,
                    strongThreshold: parseFloat(document.getElementById('petsc-bddc-local-hypre-strength').value),
                    relaxType: document.getElementById('petsc-bddc-local-hypre-relax').value,
                    numSweeps: parseInt(document.getElementById('petsc-bddc-local-hypre-sweeps').value),
                    interpType: document.getElementById('petsc-bddc-local-hypre-interp').value,
                    maxLevels: parseInt(document.getElementById('petsc-bddc-local-hypre-max-levels').value)
                };
            }
            await this.configManager.updatePetscBddcConfig(petscBddcConfig);
        }

        // Ginkgo config
        if (solverBackend === 'ginkgo') {
            const ginkgoConfig = {
                nativeAssembly: document.getElementById('ginkgo-native-assembly').checked,
                ddMatrix: document.getElementById('ginkgo-dd-matrix').checked,
                backend: document.getElementById('ginkgo-backend').value,
                solver: document.getElementById('ginkgo-solver').value,
                preconditioner: document.getElementById('ginkgo-precond').value,
                rtol: rtol,
                atol: atol,
                maxIterations: parseInt(document.getElementById('solver-max-iter').value),
                amg: {
                    cycle: document.getElementById('amg-cycle').value,
                    smoother: document.getElementById('amg-smoother').value,
                    maxLevels: parseInt(document.getElementById('amg-max-levels').value)
                },
                bddc: {
                    localSolver: document.getElementById('bddc-local-solver').value,
                    innerSolver: document.getElementById('bddc-inner-solver').value,
                    localMaxIterations: parseInt(document.getElementById('bddc-local-max-iter').value),
                    localTolerance: parseFloat(document.getElementById('bddc-local-tolerance').value),
                    coarseSolver: document.getElementById('bddc-coarse-solver').value,
                    coarseMaxIterations: parseInt(document.getElementById('bddc-coarse-max-iter').value),
                    coarseBddcLocalSolver: document.getElementById('bddc-coarse-bddc-local-solver').value,
                    vertices: document.getElementById('bddc-vertices').checked,
                    edges: document.getElementById('bddc-edges').checked,
                    faces: document.getElementById('bddc-faces').checked,
                    distributedCoarse: document.getElementById('bddc-coarse-placement').value === 'distributed',
                    unanimousConnectivity: document.getElementById('bddc-unanimous-connectivity').checked,
                    writeInterfaces: document.getElementById('bddc-write-interfaces').checked,
                    localAmg: {
                        coarsening: document.getElementById('bddc-local-amg-coarsening').value,
                        strengthThreshold: parseFloat(document.getElementById('bddc-local-amg-strength-threshold').value),
                        cycle: document.getElementById('bddc-local-amg-cycle').value,
                        smoother: document.getElementById('bddc-local-amg-smoother').value,
                        smoothSteps: parseInt(document.getElementById('bddc-local-amg-smooth-steps').value),
                        maxLevels: parseInt(document.getElementById('bddc-local-amg-max-levels').value),
                        coarseSolver: document.getElementById('bddc-local-amg-coarse-solver').value,
                        relaxationFactor: parseFloat(document.getElementById('bddc-local-amg-relaxation').value)
                    },
                    localHypre: {
                        cycleType: parseInt(document.getElementById('bddc-local-hypre-cycle').value),
                        coarseningType: parseInt(document.getElementById('bddc-local-hypre-coarsening').value),
                        strengthThreshold: parseFloat(document.getElementById('bddc-local-hypre-strength').value),
                        smootherType: parseInt(document.getElementById('bddc-local-hypre-smoother').value),
                        numSweeps: parseInt(document.getElementById('bddc-local-hypre-sweeps').value),
                        interpolationType: parseInt(document.getElementById('bddc-local-hypre-interpolation').value),
                        maxLevels: parseInt(document.getElementById('bddc-local-hypre-max-levels').value)
                    },
                    innerMaxIterations: parseInt(document.getElementById('bddc-inner-max-iter').value),
                    innerTolerance: parseFloat(document.getElementById('bddc-inner-tolerance').value),
                    innerAmg: {
                        coarsening: document.getElementById('bddc-inner-amg-coarsening').value,
                        strengthThreshold: parseFloat(document.getElementById('bddc-inner-amg-strength-threshold').value),
                        cycle: document.getElementById('bddc-inner-amg-cycle').value,
                        smoother: document.getElementById('bddc-inner-amg-smoother').value,
                        smoothSteps: parseInt(document.getElementById('bddc-inner-amg-smooth-steps').value),
                        maxLevels: parseInt(document.getElementById('bddc-inner-amg-max-levels').value),
                        coarseSolver: document.getElementById('bddc-inner-amg-coarse-solver').value,
                        relaxationFactor: parseFloat(document.getElementById('bddc-inner-amg-relaxation').value)
                    },
                    innerHypre: {
                        cycleType: parseInt(document.getElementById('bddc-inner-hypre-cycle').value),
                        coarseningType: parseInt(document.getElementById('bddc-inner-hypre-coarsening').value),
                        strengthThreshold: parseFloat(document.getElementById('bddc-inner-hypre-strength').value),
                        smootherType: parseInt(document.getElementById('bddc-inner-hypre-smoother').value),
                        numSweeps: parseInt(document.getElementById('bddc-inner-hypre-sweeps').value),
                        interpolationType: parseInt(document.getElementById('bddc-inner-hypre-interpolation').value),
                        maxLevels: parseInt(document.getElementById('bddc-inner-hypre-max-levels').value)
                    }
                }
            };
            await this.configManager.updateGinkgoConfig(ginkgoConfig);
        }
    } catch (error) {
        statusEl.className = 'status visible error';
        statusEl.textContent = 'Failed to save configuration: ' + error.message;
        return;
    }

    // Save conditions for local runs
    if (!this.isRemote()) {
        try {
            const meshName = this.meshLoader.currentMesh || 'unknown';
            const outName = meshName + '_sim';
            await fetch('/api/config/conditions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ out_name: outName, conditions: this.getConditionsSnapshot() }),
            });
        } catch (e) {
            console.warn('Failed to save conditions:', e);
        }
    }

    if (this.isRemote()) {
        await this.runSimulationRemote(statusEl, outputEl, runBtn);
    } else {
        await this.runSimulationLocal(statusEl, outputEl, runBtn);
    }
};

App.prototype.runSimulationLocal = async function(statusEl, outputEl, runBtn) {
    const cancelBtn = document.getElementById('cancel-simulation');

    try {
        runBtn.disabled = true;
        cancelBtn.style.display = 'inline-block';
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Cancel';
        statusEl.className = 'status visible running';
        statusEl.textContent = 'Simulation running...';
        outputEl.textContent = '';

        this.clearIterationsChart();
        this.initIterationsChartAxis(this.timeSteps);
        this.showIterationsChart();

        this.clearResidualChart();
        this.initResidualChartAxis(this.timeSteps);
        this.showResidualChart();

        let lastWasProgress = false;

        await this.simulationRunner.run(
            (output) => {
                if (lastWasProgress) {
                    outputEl.textContent += '\n';
                    lastWasProgress = false;
                }
                outputEl.textContent += output;
                outputEl.scrollTop = outputEl.scrollHeight;
            },
            (percent, message) => {
                const lines = outputEl.textContent.split('\n');
                const progressLine = `Time stepping: ${message}`;

                if (lastWasProgress && lines.length > 0) {
                    lines[lines.length - 1] = progressLine;
                    outputEl.textContent = lines.join('\n');
                } else {
                    outputEl.textContent += progressLine;
                }
                lastWasProgress = true;
                outputEl.scrollTop = outputEl.scrollHeight;
            },
            (step, count) => {
                this.addIterationPoint(step, count);
            },
            (step, absRes, relRes) => {
                this.addResidualPoint(step, absRes, relRes);
            }
        );

        if (lastWasProgress) {
            outputEl.textContent += '\n';
        }

        statusEl.className = 'status visible success';
        statusEl.textContent = 'Simulation completed successfully!';

        await this.loadSimulationList();
    } catch (error) {
        statusEl.className = 'status visible error';
        statusEl.textContent = 'Simulation failed: ' + error.message;
    } finally {
        runBtn.disabled = false;
        cancelBtn.style.display = 'none';
    }
};

App.prototype.runSimulationRemote = async function(statusEl, outputEl, runBtn) {
    outputEl.style.display = 'none';
    const jobSection = document.getElementById('cluster-job-section');

    try {
        // One job per (checked mesh, rank count); the config just saved for
        // the shown mesh is the template for all of them.
        const { meshes } = this.buildBatchRequest();
        const nJobs = meshes.reduce((n, m) => n + m.ranks.length, 0);

        statusEl.className = 'status visible running';
        statusEl.textContent = `Submitting ${nJobs} job${nJobs === 1 ? '' : 's'} to ${this.clusterLabel()}...`;

        const conditions = this.getConditionsSnapshot();
        const result = await this.clusterRunner.submitBatch({
            config: this.configManager.configFile || 'input_pepe36_colored.yml',
            meshes,
            max_tasks_per_node: this.batchMaxTasksPerNode(),
            walltime: document.getElementById('cluster-walltime').value || '01:00:00',
            // Empty values fall back to the cluster's configured defaults server-side
            partition: document.getElementById('cluster-partition').value || '',
            account: document.getElementById('cluster-account').value || '',
            solver_backend: document.getElementById('solver-backend').value || 'petsc',
            conditions,
            folder: this.batchFolderName(),
        });
        const jobs = result.jobs || [];

        if (jobs.length) jobSection.style.display = 'block';
        for (const job of jobs) {
            const jobId = job.job_id;
            this.clusterJobs[jobId] = {
                ...job,
                conditions_hash: job.conditions_hash,
                mesh_name: job.mesh,
                solver_backend: conditions.solver,
                preconditioner: conditions.preconditioner,
                localSolver: conditions.localSolver,
            };
            // The stored entry, not the server's: only it carries solver/precond.
            this.renderJobEntry(this.clusterJobs[jobId]);
            this.ensureMeshInFilter(job.mesh);
            this.clusterRunner.startPolling(jobId, (data) => {
                this.updateJobStatus(jobId, data);
            }, job.out_name);
        }
        this.saveClusterJobs();
        this.renderMeshFilter();
        this.applyMeshFilter();
        this.loadRunIndex({ refresh: true });  // new run folders, filed into their folder

        const failed = result.failed || [];
        statusEl.className = 'status visible ' + (failed.length ? 'error' : 'success');
        statusEl.textContent = result.message
            || `${jobs.length} job(s) submitted: ${jobs.map(j => j.job_id).join(', ')}`;

    } catch (error) {
        statusEl.className = 'status visible error';
        statusEl.textContent = 'Submission failed: ' + error.message;
    }
};

App.prototype.cancelSimulation = async function() {
    const cancelBtn = document.getElementById('cancel-simulation');
    const statusEl = document.getElementById('simulation-status');

    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Cancelling...';

    try {
        const response = await fetch('/api/simulation/stop', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            statusEl.className = 'status visible error';
            statusEl.textContent = 'Simulation cancelled by user';
        }
    } catch (error) {
        console.error('Failed to cancel simulation:', error);
    }
};

App.prototype.getConditionsSnapshot = function() {
    const solverBackend = document.getElementById('solver-backend').value;
    const precond = solverBackend === 'ginkgo'
        ? document.getElementById('ginkgo-precond').value
        : document.getElementById('petsc-pc-type').value;
    let localSolver = null;
    let reordering = null;
    if (solverBackend === 'ginkgo' && precond === 'bddc') {
        localSolver = document.getElementById('bddc-local-solver').value;
        reordering = document.getElementById('bddc-reordering').value;
    } else if (solverBackend === 'petsc' && precond === 'bddc') {
        localSolver = document.getElementById('petsc-bddc-local-solver').value;
    }

    // Cluster batches overwrite mesh/nRanks/boundingBox per job server-side.
    const mpiRanksEl = document.getElementById('mpi-ranks');
    const nRanks = mpiRanksEl ? parseInt(mpiRanksEl.value) || 1 : 1;

    const conditions = {
        mesh: this.meshLoader.currentMesh,
        solver: solverBackend,
        preconditioner: precond,
        localSolver: localSolver,
        reordering: reordering,
        nRanks: nRanks,
        boundingBox: { ...this.boundingBox },
        vExcited: this.vExcited,
        vResting: this.vResting,
        scarEnabled: this.scarEnabled,
    };
    if (this.scarEnabled) {
        conditions.scarBox = { ...this.scarBox };
        conditions.scarMargin = this.scarMargin;
        conditions.scarConductivities = {
            dense: {
                sigma_i: parseFloat(document.getElementById('scar-si-dense').value),
                sigma_e: parseFloat(document.getElementById('scar-se-dense').value),
            },
            border: {
                sigma_i: parseFloat(document.getElementById('scar-si-border').value),
                sigma_e: parseFloat(document.getElementById('scar-se-border').value),
            },
        };
    }
    return conditions;
};
