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
            this.setupMpiRanks();
            this.setupVoltageControls();
            this.setupButtons();
            this.setupCheckboxes();
            this.setupResultsControls();
            this.setupVideoExport();

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

        const updateTotalTime = () => {
            this.dt = parseFloat(dtInput.value);
            this.timeSteps = parseInt(stepsInput.value);
            const total = (this.dt * this.timeSteps).toFixed(3);
            totalTimeSpan.textContent = total;
        };

        dtInput.addEventListener('input', updateTotalTime);
        stepsInput.addEventListener('input', updateTotalTime);
        updateTotalTime();
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

        document.getElementById('reset-camera').addEventListener('click', () => {
            this.viewer.resetCamera();
        });
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

            // Fetch results metadata and data from server
            const response = await fetch(`/api/results?dir=${encodeURIComponent(simName)}`);
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

            // Show first timestep
            this.showResultsAtTime(0);

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

        const voltages = this.resultsData[timeIndex];
        this.viewer.updateVoltageColors(voltages);
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
            await this.configManager.updateConfig({
                v_init: vinitValue,
                dt: this.dt,
                time_steps: this.timeSteps
            });
        } catch (error) {
            statusEl.className = 'status visible error';
            statusEl.textContent = 'Failed to save configuration: ' + error.message;
            return;
        }

        try {
            runBtn.disabled = true;
            statusEl.className = 'status visible running';
            statusEl.textContent = 'Simulation running...';
            outputEl.textContent = '';

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
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});
