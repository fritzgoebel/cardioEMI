// app.js - Main application entry point

class App {
    constructor() {
        this.meshLoader = new MeshLoader();
        this.viewer = null;
        this.configManager = new ConfigManager();
        this.simulationRunner = new SimulationRunner();

        // Bounding box state (in micrometers - raw mesh coordinates)
        this.boundingBox = {
            xMin: -62, xMax: 15,
            yMin: -19, yMax: 68,
            zMin: -20, zMax: 118
        };

        // Simulation parameters
        this.dt = 0.001;
        this.timeSteps = 1000;
        this.bcType = 'one_corner';  // Boundary condition type

        // Voltage parameters
        this.vExcited = 0;    // mV for excited region
        this.vResting = -80;  // mV for resting region

        this.meshBounds = null;
        this.conversionFactor = 0.0001;

        // Results data
        this.resultsData = null;
        this.resultsTimeSteps = [];

        // Selected simulation for results/video
        this.selectedSimulation = null;

        // Iterations chart
        this.iterationsChart = null;
        this.iterationsData = [];
        this.currentTimeIndex = 0;

        // Residual chart
        this.residualChart = null;
        this.residualAbsData = [];
        this.residualRelData = [];

        // MPI partition data
        this.ranksData = null;
        this.numRanks = null;
        this.ecsRanksData = null;
        this.cutRanksData = null;
        this.rankCentroids = null;
        this.globalCentroid = null;

        // Interface data for BDDC visualization
        this.interfaceData = null;
        this.interfaceDofTypes = null;  // DOF -> 'vertex' | 'edge' | 'face'
        this.visibleRanks = new Set();
        this.showInterfaces = false;
        this.showInterfaceVertices = true;
        this.showInterfaceEdges = true;
        this.showInterfaceFaces = true;
    }

    async init() {
        const container = document.getElementById('viewer-container');
        const colorbar = document.getElementById('colorbar');

        // Show loading state
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.textContent = 'Loading mesh data';
        container.appendChild(loadingDiv);

        try {
            // Setup mesh selector first (before loading mesh)
            await this.setupMeshSelector();

            // Load mesh data using current mesh
            const meshData = await this.meshLoader.load();
            this.meshBounds = meshData.metadata.bounds;
            this.conversionFactor = meshData.metadata.mesh_conversion_factor;

            // Remove loading message
            loadingDiv.remove();

            // Initialize 3D viewer
            this.viewer = new Viewer('viewer-container');
            await this.viewer.init(meshData);

            // Setup UI controls
            this.setupSliders();
            this.setupSimulationParams();
            this.setupSolverSettings();
            this.setupMpiRanks();
            this.setupVoltageControls();
            this.setupButtons();
            this.setupCheckboxes();
            this.setupColormapSelector();
            this.setupResultsControls();
            this.setupVideoExport();
            this.setupIterationsChart();
            this.setupResidualChart();

            // Initial update
            this.updateVinitExpression();
            this.updateBoundingBoxVisualization();
            this.updateColorbar();

            console.log('Application ready');
        } catch (error) {
            console.error('Failed to initialize:', error);
            loadingDiv.style.color = '#e94560';
            loadingDiv.textContent = `Error: ${error.message}`;
        }
    }

    async setupMeshSelector() {
        const selector = document.getElementById('mesh-selector');
        const statusEl = document.getElementById('mesh-status');
        const convertBtn = document.getElementById('convert-mesh');
        const progressBar = document.getElementById('conversion-progress');

        // Fetch available meshes
        try {
            const response = await fetch('/api/meshes');
            const data = await response.json();

            // Store mesh info for later lookup
            this.meshesInfo = data.meshes;

            // Populate dropdown
            selector.innerHTML = '';
            data.meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh.name;
                option.textContent = mesh.name + (mesh.converted ? '' : ' (not converted)');
                if (mesh.name === data.current) {
                    option.selected = true;
                }
                selector.appendChild(option);
            });

            // Set current mesh in loader
            this.meshLoader.setMesh(data.current);

            // Set current config file
            if (data.currentConfig) {
                this.configManager.setConfigFile(data.currentConfig);
                this.simulationRunner.setConfigFile(data.currentConfig);
                console.log(`Using config: ${data.currentConfig}`);
            }

            // Update status
            this.updateMeshStatus(data.current, data.meshes);

        } catch (error) {
            console.error('Failed to load mesh list:', error);
            selector.innerHTML = '<option value="">Error loading meshes</option>';
        }

        // Handle mesh selection change
        selector.addEventListener('change', async () => {
            const meshName = selector.value;
            await this.onMeshSelected(meshName);
        });

        // Handle convert button
        convertBtn.addEventListener('click', async () => {
            const meshName = selector.value;
            await this.convertMesh(meshName);
        });
    }

    async onMeshSelected(meshName) {
        const statusEl = document.getElementById('mesh-status');
        const convertBtn = document.getElementById('convert-mesh');

        // Check if mesh is converted
        const response = await fetch('/api/meshes');
        const data = await response.json();
        const meshInfo = data.meshes.find(m => m.name === meshName);

        if (!meshInfo) return;

        if (meshInfo.converted) {
            // Select the mesh
            await this.selectMesh(meshName);
        } else {
            // Show convert button
            this.updateMeshStatus(meshName, data.meshes);
        }
    }

    updateMeshStatus(meshName, meshes) {
        const statusEl = document.getElementById('mesh-status');
        const convertBtn = document.getElementById('convert-mesh');
        const meshInfo = meshes.find(m => m.name === meshName);

        if (!meshInfo) return;

        if (meshInfo.converted) {
            statusEl.className = 'mesh-status converted';
            statusEl.textContent = 'Ready to use';
            statusEl.style.display = 'block';
            convertBtn.style.display = 'none';
        } else {
            statusEl.className = 'mesh-status pending';
            statusEl.textContent = 'Mesh needs conversion before use';
            statusEl.style.display = 'block';
            convertBtn.style.display = 'block';
        }
    }

    async convertMesh(meshName) {
        const convertBtn = document.getElementById('convert-mesh');
        const progressBar = document.getElementById('conversion-progress');
        const progressFill = progressBar.querySelector('.progress-fill');
        const progressText = progressBar.querySelector('.progress-text');
        const statusEl = document.getElementById('mesh-status');

        convertBtn.disabled = true;
        progressBar.style.display = 'block';
        statusEl.style.display = 'none';

        try {
            // Start conversion via POST, then listen for SSE
            const response = await fetch('/api/meshes/convert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mesh: meshName })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));

                        if (data.type === 'progress') {
                            progressFill.style.width = `${data.percent}%`;
                            progressText.textContent = data.message || `${data.percent}%`;
                        } else if (data.type === 'complete') {
                            progressFill.style.width = '100%';
                            progressText.textContent = 'Complete!';

                            // Auto-select the mesh
                            setTimeout(() => this.selectMesh(meshName), 500);
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        }
                    }
                }
            }
        } catch (error) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Conversion failed: ${error.message}`;
            statusEl.style.display = 'block';
        } finally {
            convertBtn.disabled = false;
            setTimeout(() => {
                progressBar.style.display = 'none';
            }, 1000);
        }
    }

    async selectMesh(meshName) {
        const statusEl = document.getElementById('mesh-status');
        const convertBtn = document.getElementById('convert-mesh');

        // Find the config file for this mesh
        const meshInfo = this.meshesInfo?.find(m => m.name === meshName);
        const configFile = meshInfo?.configFile;

        try {
            const response = await fetch('/api/meshes/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mesh: meshName, configFile: configFile })
            });

            const data = await response.json();

            if (data.success) {
                statusEl.className = 'mesh-status converted';
                statusEl.textContent = `Mesh selected (config: ${data.configFile}) - reloading...`;
                statusEl.style.display = 'block';
                convertBtn.style.display = 'none';

                // Reload the page to use new mesh
                window.location.reload();
            } else {
                throw new Error(data.error || 'Failed to select mesh');
            }
        } catch (error) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Selection failed: ${error.message}`;
            statusEl.style.display = 'block';
        }
    }

    setupSliders() {
        const axes = ['x', 'y', 'z'];

        axes.forEach(axis => {
            const minSlider = document.getElementById(`${axis}-min`);
            const maxSlider = document.getElementById(`${axis}-max`);
            const minVal = document.getElementById(`${axis}-min-val`);
            const maxVal = document.getElementById(`${axis}-max-val`);

            const bounds = this.meshBounds[axis];
            const range = bounds[1] - bounds[0];

            // Configure sliders with mesh bounds
            minSlider.min = bounds[0];
            minSlider.max = bounds[1];
            minSlider.step = range / 200;
            minSlider.value = this.boundingBox[`${axis}Min`];

            maxSlider.min = bounds[0];
            maxSlider.max = bounds[1];
            maxSlider.step = range / 200;
            maxSlider.value = this.boundingBox[`${axis}Max`];

            // Update display
            minVal.textContent = parseFloat(minSlider.value).toFixed(1);
            maxVal.textContent = parseFloat(maxSlider.value).toFixed(1);

            // Event listeners
            minSlider.addEventListener('input', () => {
                const val = parseFloat(minSlider.value);
                this.boundingBox[`${axis}Min`] = val;
                minVal.textContent = val.toFixed(1);
                this.onBoundingBoxChange();
            });

            maxSlider.addEventListener('input', () => {
                const val = parseFloat(maxSlider.value);
                this.boundingBox[`${axis}Max`] = val;
                maxVal.textContent = val.toFixed(1);
                this.onBoundingBoxChange();
            });
        });
    }

    setupSimulationParams() {
        const dtInput = document.getElementById('dt');
        const stepsInput = document.getElementById('time-steps');
        const totalTimeSpan = document.getElementById('total-time');
        const bcTypeSelect = document.getElementById('bc-type');

        const updateTotalTime = () => {
            this.dt = parseFloat(dtInput.value);
            this.timeSteps = parseInt(stepsInput.value);
            const total = (this.dt * this.timeSteps).toFixed(3);
            totalTimeSpan.textContent = total;
        };

        dtInput.addEventListener('input', updateTotalTime);
        stepsInput.addEventListener('input', updateTotalTime);
        updateTotalTime();

        // Boundary condition type
        this.bcType = bcTypeSelect.value;
        bcTypeSelect.addEventListener('change', (e) => {
            this.bcType = e.target.value;
        });
    }

    setupSolverSettings() {
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
            petsc: { kspType: 'preonly', pcType: 'lu' },
            ginkgo: {
                nativeAssembly: true,  // Default to native assembly
                ddMatrix: false,       // Domain decomposition matrix format
                backend: 'omp',
                solver: 'cg',
                preconditioner: 'jacobi',
                amg: { cycle: 'v', smoother: 'jacobi', maxLevels: 10 },
                bddc: {
                    localSolver: 'direct',
                    coarseSolver: 'cg',
                    coarseMaxIterations: 100,
                    vertices: true,
                    edges: true,
                    faces: true,
                    localAmg: {
                        smoother: 'jacobi',
                        smoothSteps: 1,
                        maxLevels: 10,
                        coarseSolver: 'direct',
                        relaxationFactor: 0.9
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

        // PETSc KSP type - show/hide preconditioner for direct solver
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
        petscPcType.addEventListener('change', () => {
            this.solverConfig.petsc.pcType = petscPcType.value;
        });

        // Ginkgo native assembly - also controls DD matrix visibility
        document.getElementById('ginkgo-native-assembly').addEventListener('change', (e) => {
            this.solverConfig.ginkgo.nativeAssembly = e.target.checked;
            // DD matrix requires native assembly
            if (!e.target.checked) {
                ddMatrixCheckbox.checked = false;
                this.solverConfig.ginkgo.ddMatrix = false;
                ddMatrixRow.style.display = 'none';
                // Restore preconditioner options
                precondRow.style.display = 'flex';
            } else {
                ddMatrixRow.style.display = 'flex';
            }
        });

        // Ginkgo DD matrix - limits preconditioner to 'none' or 'bddc'
        ddMatrixCheckbox.addEventListener('change', (e) => {
            this.solverConfig.ginkgo.ddMatrix = e.target.checked;
            if (e.target.checked) {
                // DD matrix works with 'none' or 'bddc' preconditioner
                // Default to BDDC when DD matrix is enabled
                this.solverConfig.ginkgo.preconditioner = 'bddc';
                ginkgoPrecond.value = 'bddc';
                amgOptions.style.display = 'none';
                bddcOptions.style.display = 'block';
            } else {
                // Switching off DD matrix - reset to jacobi
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

        // Ginkgo preconditioner - show AMG/BDDC options when selected
        ginkgoPrecond.addEventListener('change', () => {
            this.solverConfig.ginkgo.preconditioner = ginkgoPrecond.value;
            amgOptions.style.display = ginkgoPrecond.value === 'amg' ? 'block' : 'none';
            bddcOptions.style.display = ginkgoPrecond.value === 'bddc' ? 'block' : 'none';

            // BDDC requires DD matrix
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
        document.getElementById('bddc-local-solver').addEventListener('change', (e) => {
            this.solverConfig.ginkgo.bddc.localSolver = e.target.value;
            // Show/hide local AMG options
            bddcLocalAmgOptions.style.display = e.target.value === 'amg' ? 'block' : 'none';
        });

        document.getElementById('bddc-coarse-solver').addEventListener('change', (e) => {
            this.solverConfig.ginkgo.bddc.coarseSolver = e.target.value;
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

        // BDDC local AMG options
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

        // Initialize UI state based on current form values (handles browser auto-fill)
        this.initSolverSettingsUI();
    }

    initSolverSettingsUI() {
        // Sync UI visibility with current form values on page load
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

        // Backend visibility
        if (backendSelect.value === 'ginkgo') {
            petscOptions.style.display = 'none';
            ginkgoOptions.style.display = 'block';
        } else {
            petscOptions.style.display = 'block';
            ginkgoOptions.style.display = 'none';
        }

        // PETSc KSP type -> preconditioner row
        if (petscKspType.value === 'preonly') {
            petscPcRow.style.display = 'none';
        } else {
            petscPcRow.style.display = 'flex';
        }

        // Native assembly -> DD matrix row
        if (!nativeAssemblyCheckbox.checked) {
            ddMatrixRow.style.display = 'none';
        } else {
            ddMatrixRow.style.display = 'flex';
        }

        // Preconditioner -> AMG/BDDC options
        amgOptions.style.display = ginkgoPrecond.value === 'amg' ? 'block' : 'none';
        bddcOptions.style.display = ginkgoPrecond.value === 'bddc' ? 'block' : 'none';

        // BDDC local solver -> local AMG options
        bddcLocalAmgOptions.style.display = bddcLocalSolver.value === 'amg' ? 'block' : 'none';

        // Update config state from form values
        this.solverConfig.backend = backendSelect.value;
        this.solverConfig.petsc.kspType = petscKspType.value;
        this.solverConfig.petsc.pcType = document.getElementById('petsc-pc-type').value;
        this.solverConfig.ginkgo.nativeAssembly = nativeAssemblyCheckbox.checked;
        this.solverConfig.ginkgo.ddMatrix = ddMatrixCheckbox.checked;
        this.solverConfig.ginkgo.backend = document.getElementById('ginkgo-backend').value;
        this.solverConfig.ginkgo.solver = document.getElementById('ginkgo-solver').value;
        this.solverConfig.ginkgo.preconditioner = ginkgoPrecond.value;
    }

    setupMpiRanks() {
        const input = document.getElementById('mpi-ranks');
        const cpuInfo = document.getElementById('cpu-info');

        // Fetch system info for recommendations
        fetch('/api/system/info')
            .then(r => r.json())
            .then(info => {
                cpuInfo.textContent = `(${info.cpu_count} CPUs)`;
                input.max = info.max_ranks;
                input.value = info.recommended_ranks;
                this.simulationRunner.setMpiRanks(info.recommended_ranks);
            })
            .catch(err => {
                console.warn('Could not fetch system info:', err);
            });

        input.addEventListener('change', () => {
            this.simulationRunner.setMpiRanks(parseInt(input.value));
        });
    }

    setupVoltageControls() {
        const vExcitedSlider = document.getElementById('v-excited');
        const vExcitedVal = document.getElementById('v-excited-val');
        const vRestingSlider = document.getElementById('v-resting');
        const vRestingVal = document.getElementById('v-resting-val');

        vExcitedSlider.addEventListener('input', () => {
            this.vExcited = parseInt(vExcitedSlider.value);
            vExcitedVal.textContent = this.vExcited;
            this.updateVinitExpression();
            this.updateColorbar();
            this.onBoundingBoxChange();
        });

        vRestingSlider.addEventListener('input', () => {
            this.vResting = parseInt(vRestingSlider.value);
            vRestingVal.textContent = this.vResting;
            this.updateVinitExpression();
            this.updateColorbar();
            this.onBoundingBoxChange();
        });
    }

    setupResultsControls() {
        const loadBtn = document.getElementById('load-results');
        const timeSlider = document.getElementById('result-time');
        const timeVal = document.getElementById('result-time-val');
        const simSelector = document.getElementById('simulation-selector');

        // Load available simulations
        this.loadSimulationList();

        loadBtn.addEventListener('click', () => this.loadResults());

        timeSlider.addEventListener('input', () => {
            if (this.resultsData && this.resultsTimeSteps.length > 0) {
                const idx = parseInt(timeSlider.value);
                const time = this.resultsTimeSteps[idx];
                timeVal.textContent = time.toFixed(3);
                this.showResultsAtTime(idx);
                // Update iterations chart highlight
                this.highlightIterationStep(idx, this.resultsTimeSteps.length);
            }
        });

        // Store selected simulation for use in video export
        simSelector.addEventListener('change', () => {
            this.selectedSimulation = simSelector.value;
        });
    }

    async loadSimulationList() {
        const simSelector = document.getElementById('simulation-selector');

        try {
            const response = await fetch('/api/simulations');
            const data = await response.json();

            simSelector.innerHTML = '<option value="">Select simulation...</option>';

            data.simulations.forEach(sim => {
                const option = document.createElement('option');
                option.value = sim.name;
                option.textContent = sim.name + (sim.has_viz_data ? '' : ' (will generate viz data)');
                simSelector.appendChild(option);
            });

            // Select first simulation by default if available
            if (data.simulations.length > 0) {
                simSelector.value = data.simulations[0].name;
                this.selectedSimulation = data.simulations[0].name;
            }
        } catch (error) {
            console.error('Failed to load simulation list:', error);
        }
    }

    setupButtons() {
        document.getElementById('run-simulation').addEventListener('click', () => {
            this.runSimulation();
        });

        document.getElementById('cancel-simulation').addEventListener('click', () => {
            this.cancelSimulation();
        });

        document.getElementById('reset-camera').addEventListener('click', () => {
            this.viewer.resetCamera();
        });
    }

    async cancelSimulation() {
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
    }

    setupColormapSelector() {
        const selector = document.getElementById('colormap-selector');

        selector.addEventListener('change', () => {
            const colormap = selector.value;
            this.viewer.setColormap(colormap);

            // Update colorbar gradient
            this.updateColorbarGradient();

            // Re-render with new colormap
            const showPartition = document.getElementById('show-partition').checked;
            if (!showPartition) {
                if (this.resultsData) {
                    // If viewing results, re-render current timestep
                    const timeSlider = document.getElementById('result-time');
                    const idx = parseInt(timeSlider.value);
                    this.viewer.updateVoltageColors(this.resultsData[idx]);
                } else {
                    // Otherwise update the excited highlight
                    this.viewer.updateBoundingBox(this.boundingBox);
                }
            }
        });

        // Initialize colorbar gradient
        this.updateColorbarGradient();
    }

    updateColorbarGradient() {
        const gradient = this.viewer.getColormapGradient();
        const gradientEl = document.querySelector('.colorbar-gradient');
        if (gradientEl) {
            gradientEl.style.background = gradient;
        }
    }

    setupCheckboxes() {
        document.getElementById('show-box').addEventListener('change', (e) => {
            this.viewer.setBoundingBoxVisible(e.target.checked);
        });

        document.getElementById('show-excited').addEventListener('change', (e) => {
            this.viewer.setExcitedRegionHighlight(e.target.checked);
            if (e.target.checked) {
                this.viewer.updateBoundingBox(this.boundingBox);
            }
        });

        // Partition toggle
        document.getElementById('show-partition').addEventListener('change', (e) => {
            this.onPartitionToggle(e.target.checked);
        });

        // ECS visibility toggle
        document.getElementById('show-ecs').addEventListener('change', (e) => {
            this.viewer.setEcsVisible(e.target.checked);
            // Color ECS by rank when shown, and refresh rank visibility to apply interface highlighting
            if (e.target.checked && this.ecsRanksData) {
                // Trigger rank visibility update which handles both rank colors and interface highlighting
                if (this.visibleRanks && this.visibleRanks.size > 0) {
                    this.viewer.setVisibleRanks(this.visibleRanks);
                } else {
                    this.viewer.updateEcsRankColors(this.ecsRanksData);
                }
            }
        });

        // Explosion slider
        document.getElementById('explosion-slider').addEventListener('input', (e) => {
            const factor = parseFloat(e.target.value);
            document.getElementById('explosion-value').textContent = factor.toFixed(2);
            this.viewer.setExplosionFactor(factor);
        });

        // Show interfaces toggle
        document.getElementById('show-interfaces').addEventListener('change', (e) => {
            this.showInterfaces = e.target.checked;
            // Show/hide interface type controls
            document.getElementById('interface-type-controls').style.display = this.showInterfaces ? 'block' : 'none';
            this.updateInterfaceHighlight();
            // Also update ECS visibility to apply interface highlighting
            if (document.getElementById('show-ecs').checked && this.ecsRanksData && this.visibleRanks) {
                this.viewer.setVisibleRanks(this.visibleRanks);
            }
        });

        // Interface type toggles
        document.getElementById('show-interface-vertices').addEventListener('change', (e) => {
            this.showInterfaceVertices = e.target.checked;
            this.updateInterfaceHighlight();
        });
        document.getElementById('show-interface-edges').addEventListener('change', (e) => {
            this.showInterfaceEdges = e.target.checked;
            this.updateInterfaceHighlight();
        });
        document.getElementById('show-interface-faces').addEventListener('change', (e) => {
            this.showInterfaceFaces = e.target.checked;
            this.updateInterfaceHighlight();
        });

        // Rank selection buttons
        document.getElementById('select-all-ranks').addEventListener('click', () => {
            this.selectAllRanks(true);
        });

        document.getElementById('select-no-ranks').addEventListener('click', () => {
            this.selectAllRanks(false);
        });
    }

    selectAllRanks(selectAll) {
        const checkboxes = document.querySelectorAll('#rank-checkboxes input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.checked = selectAll;
        });
        this.onRankSelectionChange();
    }

    onRankSelectionChange() {
        // Get selected ranks from checkboxes
        const checkboxes = document.querySelectorAll('#rank-checkboxes input[type="checkbox"]');
        this.visibleRanks = new Set();
        checkboxes.forEach(cb => {
            if (cb.checked) {
                this.visibleRanks.add(parseInt(cb.dataset.rank));
            }
        });

        // Update viewer
        this.viewer.setVisibleRanks(this.visibleRanks);

        // Update interface highlight based on visible ranks
        this.updateInterfaceHighlight();
    }

    updateInterfaceHighlight() {
        if (!this.interfaceData || !this.showInterfaces) {
            this.viewer.clearInterfaceHighlight();
            return;
        }

        // Build interface map: DOF index -> global interface index
        // This gives each interface a unique color
        // Filter by interface type (vertex/edge/face) based on toggle state
        const interfaceMap = new Map();
        let globalInterfaceIdx = 0;
        let skippedByType = { vertex: 0, edge: 0, face: 0 };

        for (const rank of this.visibleRanks) {
            const rankInterfaces = this.interfaceData[rank];
            if (rankInterfaces) {
                for (const interfaceList of rankInterfaces) {
                    // Each interface (line in IF_*.txt) gets its own color
                    for (const dof of interfaceList) {
                        // Check if this DOF's type is visible
                        if (this.interfaceDofTypes) {
                            const dofType = this.interfaceDofTypes[dof];
                            if (dofType === 'vertex' && !this.showInterfaceVertices) {
                                skippedByType.vertex++;
                                continue;
                            }
                            if (dofType === 'edge' && !this.showInterfaceEdges) {
                                skippedByType.edge++;
                                continue;
                            }
                            if (dofType === 'face' && !this.showInterfaceFaces) {
                                skippedByType.face++;
                                continue;
                            }
                        }
                        // If a DOF is in multiple interfaces, keep the first assignment
                        // (interfaces may share vertices at corners)
                        if (!interfaceMap.has(dof)) {
                            interfaceMap.set(dof, globalInterfaceIdx);
                        }
                    }
                    globalInterfaceIdx++;
                }
            }
        }

        console.log(`Highlighting ${interfaceMap.size} interface DOFs across ${globalInterfaceIdx} interfaces (skipped: ${skippedByType.vertex} vertices, ${skippedByType.edge} edges, ${skippedByType.face} faces)`);
        this.viewer.setHighlightedInterfaceDofs(interfaceMap);
    }

    async loadInterfaceData() {
        try {
            const response = await fetch('/api/interfaces');
            const data = await response.json();

            if (data.interfaces && Object.keys(data.interfaces).length > 0) {
                // Convert string keys to integers
                this.interfaceData = {};
                for (const [rank, interfaces] of Object.entries(data.interfaces)) {
                    this.interfaceData[parseInt(rank)] = interfaces;
                }
                // Store DOF type classifications (vertex/edge/face)
                if (data.dofTypes) {
                    this.interfaceDofTypes = {};
                    for (const [dof, dofType] of Object.entries(data.dofTypes)) {
                        this.interfaceDofTypes[parseInt(dof)] = dofType;
                    }
                    // Count by type
                    const typeCounts = { vertex: 0, edge: 0, face: 0 };
                    for (const t of Object.values(this.interfaceDofTypes)) {
                        typeCounts[t]++;
                    }
                    console.log(`Interface DOF types: ${typeCounts.vertex} vertices, ${typeCounts.edge} edges, ${typeCounts.face} faces`);
                }
                console.log(`Loaded interface data: ${data.totalInterfaces} interfaces across ${data.numRanks} ranks`);
                return true;
            }
        } catch (error) {
            console.warn('Could not load interface data:', error);
        }
        this.interfaceData = null;
        this.interfaceDofTypes = null;
        return false;
    }

    onPartitionToggle(showPartition) {
        const colorbar = document.getElementById('colorbar');
        const rankLegend = document.getElementById('rank-legend');
        const partitionControls = document.getElementById('partition-controls');
        const rankSelector = document.getElementById('rank-selector');

        if (showPartition && this.ranksData) {
            // Show partition coloring
            this.viewer.updateRankColors(this.ranksData);
            colorbar.style.display = 'none';
            rankLegend.style.display = 'flex';
            partitionControls.style.display = 'flex';
            rankSelector.style.display = 'flex';

            // Initialize viewer with all ranks visible
            this.viewer.setVisibleRanks(this.visibleRanks);

            // Color ECS by rank if visible
            if (document.getElementById('show-ecs').checked && this.ecsRanksData) {
                this.viewer.updateEcsRankColors(this.ecsRanksData);
            }

            // Show and color partition cut mesh
            if (this.cutRanksData) {
                this.viewer.updateCutRankColors(this.cutRanksData);
                this.viewer.setCutVisible(true);
            }

            // Update interface highlight if enabled
            if (this.showInterfaces) {
                this.updateInterfaceHighlight();
            }
        } else if (this.resultsData) {
            // Restore full mesh (all ranks) before switching to voltage view
            this.viewer.restoreFullMesh();

            // Restore voltage coloring
            const timeSlider = document.getElementById('result-time');
            const idx = parseInt(timeSlider.value);
            this.viewer.updateVoltageColors(this.resultsData[idx]);
            colorbar.style.display = 'flex';
            rankLegend.style.display = 'none';
            partitionControls.style.display = 'none';
            rankSelector.style.display = 'none';

            // Hide ECS, cut mesh and reset explosion when leaving partition mode
            this.viewer.setEcsVisible(false);
            this.viewer.setCutVisible(false);
            this.viewer.setExplosionFactor(0);
            this.viewer.resetEcsColors();
            this.viewer.clearInterfaceHighlight();
            document.getElementById('show-ecs').checked = false;
            document.getElementById('show-interfaces').checked = false;
            document.getElementById('interface-type-controls').style.display = 'none';
            document.getElementById('explosion-slider').value = 0;
            document.getElementById('explosion-value').textContent = '0';
            this.showInterfaces = false;
        }
    }

    showPartitionOption(numRanks) {
        // Show the partition toggle option and build the legend
        const label = document.getElementById('show-partition-label');
        const legendItems = document.getElementById('rank-legend-items');
        const rankCheckboxes = document.getElementById('rank-checkboxes');

        label.style.display = 'flex';

        // Initialize visible ranks to all
        this.visibleRanks = new Set();
        for (let i = 0; i < numRanks; i++) {
            this.visibleRanks.add(i);
        }

        // Build legend items
        legendItems.innerHTML = '';
        for (let i = 0; i < numRanks; i++) {
            const color = this.viewer.rankToColor(i);
            const item = document.createElement('span');
            item.className = 'rank-legend-item';
            item.innerHTML = `
                <span class="rank-legend-color" style="background-color: rgb(${Math.round(color.r*255)}, ${Math.round(color.g*255)}, ${Math.round(color.b*255)})"></span>
                <span>${i}</span>
            `;
            legendItems.appendChild(item);
        }

        // Build rank checkboxes
        rankCheckboxes.innerHTML = '';
        for (let i = 0; i < numRanks; i++) {
            const color = this.viewer.rankToColor(i);
            const item = document.createElement('label');
            item.className = 'rank-checkbox-item';
            item.innerHTML = `
                <input type="checkbox" data-rank="${i}" checked>
                <span class="rank-color" style="background-color: rgb(${Math.round(color.r*255)}, ${Math.round(color.g*255)}, ${Math.round(color.b*255)})"></span>
                <span>${i}</span>
            `;
            item.querySelector('input').addEventListener('change', () => this.onRankSelectionChange());
            rankCheckboxes.appendChild(item);
        }

        // Load interface data
        this.loadInterfaceData();
    }

    hidePartitionOption() {
        const label = document.getElementById('show-partition-label');
        const rankLegend = document.getElementById('rank-legend');
        const partitionControls = document.getElementById('partition-controls');
        const rankSelector = document.getElementById('rank-selector');
        const checkbox = document.getElementById('show-partition');

        label.style.display = 'none';
        rankLegend.style.display = 'none';
        partitionControls.style.display = 'none';
        rankSelector.style.display = 'none';
        checkbox.checked = false;

        // Reset ECS, interfaces, and explosion
        document.getElementById('show-ecs').checked = false;
        document.getElementById('show-interfaces').checked = false;
        document.getElementById('interface-type-controls').style.display = 'none';
        document.getElementById('explosion-slider').value = 0;
        document.getElementById('explosion-value').textContent = '0';
        this.showInterfaces = false;
        this.interfaceData = null;
        this.interfaceDofTypes = null;
    }

    onBoundingBoxChange() {
        this.updateVinitExpression();
        this.updateBoundingBoxVisualization();
    }

    updateVinitExpression() {
        const expr = this.generateVinitExpression();
        document.getElementById('v-init-preview').textContent = expr;
    }

    generateVinitExpression() {
        // Convert micrometers to cm (scaled coordinates)
        const cf = this.conversionFactor;

        // Format numbers, removing unnecessary trailing zeros
        const fmt = (v) => {
            const scaled = v * cf;
            // Use enough precision to be accurate
            return scaled.toPrecision(6).replace(/\.?0+$/, '');
        };

        const xMin = fmt(this.boundingBox.xMin);
        const xMax = fmt(this.boundingBox.xMax);
        const yMin = fmt(this.boundingBox.yMin);
        const yMax = fmt(this.boundingBox.yMax);
        const zMin = fmt(this.boundingBox.zMin);
        const zMax = fmt(this.boundingBox.zMax);

        // Generate condition: inside box = vExcited, outside = vResting
        // v = vResting + (vExcited - vResting) * inside
        const inside = `((x[0] >= ${xMin}) * (x[0] <= ${xMax}) * (x[1] >= ${yMin}) * (x[1] <= ${yMax}) * (x[2] >= ${zMin}) * (x[2] <= ${zMax}))`;

        const vDiff = this.vExcited - this.vResting;
        return `"(${this.vResting}.0) + (${vDiff}.0) * ${inside}"`;
    }

    updateColorbar() {
        document.getElementById('colorbar-max').textContent = `${this.vExcited} mV`;
        document.getElementById('colorbar-min').textContent = `${this.vResting} mV`;
        document.getElementById('colorbar-mid').textContent = `${Math.round((this.vExcited + this.vResting) / 2)} mV`;

        // Update viewer voltage range
        if (this.viewer) {
            this.viewer.setVoltageRange(this.vResting, this.vExcited);
            // Update colorbar gradient (in case colormap changed)
            this.updateColorbarGradient();
        }
    }

    updateBoundingBoxVisualization() {
        if (this.viewer) {
            this.viewer.updateBoundingBox(this.boundingBox);
        }
    }

    async loadResults() {
        const loadBtn = document.getElementById('load-results');
        const statusEl = document.getElementById('results-status');
        const originalText = loadBtn.textContent;

        const simName = this.selectedSimulation || document.getElementById('simulation-selector').value;
        if (!simName) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = 'Please select a simulation first';
            statusEl.style.display = 'block';
            return;
        }

        try {
            loadBtn.disabled = true;
            loadBtn.textContent = 'Loading...';
            statusEl.style.display = 'none';

            // Check if regenerate checkbox is checked
            const regenerate = document.getElementById('regenerate-viz').checked;

            // Fetch results metadata and data from server
            let url = `/api/results?dir=${encodeURIComponent(simName)}`;
            if (regenerate) {
                url += '&regenerate=true';
                loadBtn.textContent = 'Regenerating viz data...';
            }
            const response = await fetch(url);
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to load results');
            }

            const data = await response.json();

            // Reload mesh from simulation's viz data if different from current
            if (data.vizDataDir && data.vizDataDir !== this.meshLoader.currentMesh) {
                loadBtn.textContent = 'Reloading mesh...';
                this.meshLoader.setMesh(data.vizDataDir);
                const meshData = await this.meshLoader.load();
                this.meshBounds = meshData.metadata.bounds;
                this.conversionFactor = meshData.metadata.mesh_conversion_factor;
                await this.viewer.reloadMesh(meshData);
            }

            this.resultsData = data.voltages;
            this.resultsTimeSteps = data.times;

            // Update UI
            const timeSlider = document.getElementById('result-time');
            timeSlider.max = this.resultsTimeSteps.length - 1;
            timeSlider.value = 0;
            document.getElementById('result-time-val').textContent = this.resultsTimeSteps[0].toFixed(3);

            // Update voltage range display
            document.getElementById('v-min-result').textContent = Math.round(data.vMin);
            document.getElementById('v-max-result').textContent = Math.round(data.vMax);

            // Update colorbar for results
            this.viewer.setVoltageRange(data.vMin, data.vMax);
            document.getElementById('colorbar-max').textContent = `${Math.round(data.vMax)} mV`;
            document.getElementById('colorbar-min').textContent = `${Math.round(data.vMin)} mV`;
            document.getElementById('colorbar-mid').textContent = `${Math.round((data.vMax + data.vMin) / 2)} mV`;

            // Load iterations data if available
            if (data.iterations && data.iterations.length > 0) {
                this.setIterationsData(data.iterations);
                this.showIterationsChart();
                // Highlight initial step
                this.highlightIterationStep(0, this.resultsTimeSteps.length);
            } else {
                this.hideIterationsChart();
            }

            // Load residuals data if available
            if (data.residuals && data.residuals.abs && data.residuals.abs.length > 0) {
                this.setResidualData(data.residuals.abs, data.residuals.rel);
                this.showResidualChart();
            } else {
                this.hideResidualChart();
            }

            // Load MPI rank data if available
            if (data.ranks && data.numRanks) {
                this.ranksData = data.ranks;
                this.numRanks = data.numRanks;
                this.ecsRanksData = data.ecsRanks;
                this.cutRanksData = data.cutRanks;
                this.rankCentroids = data.rankCentroids;
                this.globalCentroid = data.globalCentroid;

                this.viewer.setNumRanks(data.numRanks);
                this.viewer.setExplosionData(
                    data.ranks,
                    data.ecsRanks,
                    data.cutRanks,
                    data.rankCentroids,
                    data.globalCentroid
                );

                // Pass DOF indices for interface highlighting
                if (data.dofIndices) {
                    this.viewer.setDofIndices(data.dofIndices);
                }
                if (data.ecsDofIndices) {
                    this.viewer.setEcsDofIndices(data.ecsDofIndices);
                }

                this.showPartitionOption(data.numRanks);
            } else {
                this.ranksData = null;
                this.numRanks = null;
                this.ecsRanksData = null;
                this.cutRanksData = null;
                this.rankCentroids = null;
                this.globalCentroid = null;
                this.hidePartitionOption();
            }

            // Show first timestep
            this.showResultsAtTime(0);

            // Uncheck regenerate to prevent accidental re-regeneration
            document.getElementById('regenerate-viz').checked = false;

            loadBtn.textContent = 'Reload Results';
        } catch (error) {
            alert('Failed to load results: ' + error.message);
            loadBtn.textContent = originalText;
        } finally {
            loadBtn.disabled = false;
        }
    }

    showResultsAtTime(timeIndex) {
        if (!this.resultsData || !this.viewer) return;

        // Only update voltage colors if not in partition mode
        const showPartition = document.getElementById('show-partition').checked;
        if (!showPartition) {
            const voltages = this.resultsData[timeIndex];
            this.viewer.updateVoltageColors(voltages);
        }
    }

    async runSimulation() {
        const statusEl = document.getElementById('simulation-status');
        const outputEl = document.getElementById('simulation-output');
        const runBtn = document.getElementById('run-simulation');

        // First save the configuration
        try {
            statusEl.className = 'status visible running';
            statusEl.textContent = 'Saving configuration...';

            const expr = this.generateVinitExpression();
            const vinitValue = expr.slice(1, -1);

            // Build config updates including solver settings
            // Read values directly from DOM to ensure we capture user selections
            const solverBackend = document.getElementById('solver-backend').value;
            const kspType = document.getElementById('petsc-ksp-type').value;
            const pcType = document.getElementById('petsc-pc-type').value;
            const rtol = document.getElementById('solver-rtol').value;
            const atol = document.getElementById('solver-atol').value;

            const configUpdates = {
                v_init: vinitValue,
                dt: this.dt,
                time_steps: this.timeSteps,
                solver_backend: solverBackend,
                ksp_type: kspType,
                pc_type: pcType,
                ksp_rtol: rtol,
                ksp_atol: atol,
                bc_type: this.bcType
            };

            // Update local state to match
            this.solverConfig.backend = solverBackend;
            this.solverConfig.petsc.kspType = kspType;
            this.solverConfig.petsc.pcType = pcType;
            this.solverConfig.rtol = rtol;
            this.solverConfig.atol = atol;

            // Update config via API
            await this.configManager.updateConfig(configUpdates);

            // If using Ginkgo, update ginkgo config via special endpoint
            if (solverBackend === 'ginkgo') {
                // Read Ginkgo values from DOM
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
                        coarseSolver: document.getElementById('bddc-coarse-solver').value,
                        coarseMaxIterations: parseInt(document.getElementById('bddc-coarse-max-iter').value),
                        vertices: document.getElementById('bddc-vertices').checked,
                        edges: document.getElementById('bddc-edges').checked,
                        faces: document.getElementById('bddc-faces').checked,
                        localAmg: {
                            smoother: document.getElementById('bddc-local-amg-smoother').value,
                            smoothSteps: parseInt(document.getElementById('bddc-local-amg-smooth-steps').value),
                            maxLevels: parseInt(document.getElementById('bddc-local-amg-max-levels').value),
                            coarseSolver: document.getElementById('bddc-local-amg-coarse-solver').value,
                            relaxationFactor: parseFloat(document.getElementById('bddc-local-amg-relaxation').value)
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

        const cancelBtn = document.getElementById('cancel-simulation');

        try {
            runBtn.disabled = true;
            cancelBtn.style.display = 'inline-block';
            cancelBtn.disabled = false;
            cancelBtn.textContent = 'Cancel';
            statusEl.className = 'status visible running';
            statusEl.textContent = 'Simulation running...';
            outputEl.textContent = '';

            // Clear and show charts for real-time updates
            this.clearIterationsChart();
            this.initIterationsChartAxis(this.timeSteps);
            this.showIterationsChart();

            this.clearResidualChart();
            this.initResidualChartAxis(this.timeSteps);
            this.showResidualChart();

            // Track if last output was progress to enable line replacement
            let lastWasProgress = false;

            await this.simulationRunner.run(
                // Output callback
                (output) => {
                    if (lastWasProgress) {
                        // Add newline after progress before regular output
                        outputEl.textContent += '\n';
                        lastWasProgress = false;
                    }
                    outputEl.textContent += output;
                    outputEl.scrollTop = outputEl.scrollHeight;
                },
                // Progress callback - replace the progress line in output
                (percent, message) => {
                    const lines = outputEl.textContent.split('\n');
                    const progressLine = `Time stepping: ${message}`;

                    if (lastWasProgress && lines.length > 0) {
                        // Replace the last line
                        lines[lines.length - 1] = progressLine;
                        outputEl.textContent = lines.join('\n');
                    } else {
                        // First progress update - append it
                        outputEl.textContent += progressLine;
                    }
                    lastWasProgress = true;
                    outputEl.scrollTop = outputEl.scrollHeight;
                },
                // Iterations callback - update chart in real-time
                (step, count) => {
                    this.addIterationPoint(step, count);
                },
                // Residual callback - update residual chart in real-time
                (step, absRes, relRes) => {
                    this.addResidualPoint(step, absRes, relRes);
                }
            );

            // Add newline after final progress
            if (lastWasProgress) {
                outputEl.textContent += '\n';
            }

            statusEl.className = 'status visible success';
            statusEl.textContent = 'Simulation completed successfully!';

            // Refresh simulation list to include new output
            await this.loadSimulationList();
        } catch (error) {
            statusEl.className = 'status visible error';
            statusEl.textContent = 'Simulation failed: ' + error.message;
        } finally {
            runBtn.disabled = false;
            cancelBtn.style.display = 'none';
        }
    }

    setupVideoExport() {
        const exportBtn = document.getElementById('export-video');
        const progressBar = document.getElementById('video-progress');
        const progressFill = progressBar.querySelector('.progress-fill');
        const progressText = progressBar.querySelector('.progress-text');
        const statusEl = document.getElementById('video-status');
        const downloadLink = document.getElementById('video-download');

        exportBtn.addEventListener('click', async () => {
            const resolutionSelect = document.getElementById('video-resolution');
            const fpsInput = document.getElementById('video-fps');

            const resolution = resolutionSelect.value.split('x');
            const width = parseInt(resolution[0]);
            const height = parseInt(resolution[1]);
            const fps = parseInt(fpsInput.value);

            // Get current camera state
            const cameraState = this.viewer.getCameraState();

            exportBtn.disabled = true;
            progressBar.style.display = 'block';
            progressFill.style.width = '0%';
            progressText.textContent = 'Starting...';
            statusEl.style.display = 'none';
            downloadLink.style.display = 'none';

            try {
                const simName = this.selectedSimulation || document.getElementById('simulation-selector').value;
                if (!simName) {
                    throw new Error('Please select a simulation first');
                }

                const response = await fetch('/api/video/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        output_dir: simName,
                        camera: cameraState,
                        width: width,
                        height: height,
                        fps: fps
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const text = decoder.decode(value);
                    const lines = text.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));

                                if (data.type === 'progress') {
                                    progressFill.style.width = `${data.percent}%`;
                                    progressText.textContent = data.message || `${data.percent}%`;
                                } else if (data.type === 'complete') {
                                    progressFill.style.width = '100%';
                                    progressText.textContent = 'Complete!';

                                    statusEl.className = 'mesh-status converted';
                                    statusEl.textContent = 'Video exported successfully!';
                                    statusEl.style.display = 'block';

                                    downloadLink.href = `/api/video/download/${data.filename}`;
                                    downloadLink.textContent = `Download ${data.filename}`;
                                    downloadLink.style.display = 'block';
                                } else if (data.type === 'error') {
                                    throw new Error(data.message);
                                }
                            } catch (e) {
                                if (e.message !== 'Unexpected end of JSON input') {
                                    throw e;
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = `Export failed: ${error.message}`;
                statusEl.style.display = 'block';
            } finally {
                exportBtn.disabled = false;
                setTimeout(() => {
                    progressBar.style.display = 'none';
                }, 2000);
            }
        });
    }

    setupIterationsChart() {
        const ctx = document.getElementById('iterations-chart').getContext('2d');

        this.iterationsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Solver Iterations',
                    data: [],
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }, {
                    label: 'Current',
                    data: [],
                    borderColor: '#4ade80',
                    backgroundColor: '#4ade80',
                    borderWidth: 0,
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    showLine: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#16213e',
                        titleColor: '#fff',
                        bodyColor: '#ccc',
                        borderColor: '#e94560',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time Step',
                            color: '#888'
                        },
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Iterations',
                            color: '#888'
                        },
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        beginAtZero: true
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    showIterationsChart() {
        const container = document.getElementById('iterations-chart-container');
        container.style.display = 'block';
    }

    hideIterationsChart() {
        const container = document.getElementById('iterations-chart-container');
        container.style.display = 'none';
    }

    clearIterationsChart() {
        this.iterationsData = [];
        if (this.iterationsChart) {
            this.iterationsChart.data.labels = [];
            this.iterationsChart.data.datasets[0].data = [];
            this.iterationsChart.data.datasets[1].data = [];
            this.iterationsChart.update('none');
        }
    }

    initIterationsChartAxis(totalSteps) {
        // Pre-populate x-axis with full time range
        if (this.iterationsChart) {
            this.iterationsChart.data.labels = Array.from({ length: totalSteps }, (_, i) => i);
            this.iterationsChart.data.datasets[0].data = new Array(totalSteps).fill(null);
            this.iterationsChart.data.datasets[1].data = [];
            this.iterationsChart.update('none');
        }
    }

    addIterationPoint(step, count) {
        // Store the data
        while (this.iterationsData.length <= step) {
            this.iterationsData.push(null);
        }
        this.iterationsData[step] = { step, count };

        // Update chart data at the specific index
        if (this.iterationsChart && step < this.iterationsChart.data.datasets[0].data.length) {
            this.iterationsChart.data.datasets[0].data[step] = count;
            this.iterationsChart.update('none');
        }
    }

    setIterationsData(iterations) {
        this.iterationsData = iterations.map((count, i) => ({ step: i, count }));
        if (this.iterationsChart) {
            this.iterationsChart.data.labels = iterations.map((_, i) => i);
            this.iterationsChart.data.datasets[0].data = iterations;
            this.iterationsChart.data.datasets[1].data = [];
            this.iterationsChart.update('none');
        }
    }

    highlightIterationStep(timeIndex, totalResultSteps) {
        if (!this.iterationsChart || this.iterationsData.length === 0) return;

        // Map result time index to iteration index
        // Results may be sampled (e.g., 50 timesteps out of 1000)
        const iterationIndex = Math.round(timeIndex * (this.iterationsData.length - 1) / (totalResultSteps - 1));

        // Update current marker dataset
        const markerData = new Array(this.iterationsData.length).fill(null);
        if (iterationIndex >= 0 && iterationIndex < this.iterationsData.length) {
            markerData[iterationIndex] = this.iterationsData[iterationIndex].count;
        }

        this.iterationsChart.data.datasets[1].data = markerData;
        this.iterationsChart.update('none');
    }

    // ==================== Residual Chart Methods ====================

    setupResidualChart() {
        const ctx = document.getElementById('residual-chart').getContext('2d');

        this.residualChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Absolute Residual',
                    data: [],
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    yAxisID: 'y'
                }, {
                    label: 'Relative Residual',
                    data: [],
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    yAxisID: 'y'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#888',
                            boxWidth: 12,
                            padding: 8,
                            font: { size: 10 }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#16213e',
                        titleColor: '#fff',
                        bodyColor: '#ccc',
                        borderColor: '#e94560',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toExponential(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time Step',
                            color: '#888'
                        },
                        ticks: { color: '#888' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Residual Norm',
                            color: '#888'
                        },
                        ticks: {
                            color: '#888',
                            callback: function(value) {
                                return value.toExponential(0);
                            }
                        },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    showResidualChart() {
        const container = document.getElementById('residual-chart-container');
        container.style.display = 'block';
    }

    hideResidualChart() {
        const container = document.getElementById('residual-chart-container');
        container.style.display = 'none';
    }

    clearResidualChart() {
        this.residualAbsData = [];
        this.residualRelData = [];
        if (this.residualChart) {
            this.residualChart.data.labels = [];
            this.residualChart.data.datasets[0].data = [];
            this.residualChart.data.datasets[1].data = [];
            this.residualChart.update('none');
        }
    }

    initResidualChartAxis(totalSteps) {
        if (this.residualChart) {
            this.residualChart.data.labels = Array.from({ length: totalSteps }, (_, i) => i);
            this.residualChart.data.datasets[0].data = new Array(totalSteps).fill(null);
            this.residualChart.data.datasets[1].data = new Array(totalSteps).fill(null);
            this.residualChart.update('none');
        }
    }

    addResidualPoint(step, absRes, relRes) {
        // Store the data
        while (this.residualAbsData.length <= step) {
            this.residualAbsData.push(null);
            this.residualRelData.push(null);
        }
        this.residualAbsData[step] = absRes;
        this.residualRelData[step] = relRes;

        // Update chart data at the specific index
        if (this.residualChart && step < this.residualChart.data.datasets[0].data.length) {
            this.residualChart.data.datasets[0].data[step] = absRes;
            this.residualChart.data.datasets[1].data[step] = relRes;
            this.residualChart.update('none');
        }
    }

    setResidualData(absData, relData) {
        this.residualAbsData = absData;
        this.residualRelData = relData;
        if (this.residualChart) {
            this.residualChart.data.labels = absData.map((_, i) => i);
            this.residualChart.data.datasets[0].data = absData;
            this.residualChart.data.datasets[1].data = relData;
            this.residualChart.update('none');
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});
