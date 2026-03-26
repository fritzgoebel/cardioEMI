// app.js - Main application entry point

class App {
    constructor() {
        this.meshLoader = new MeshLoader();
        this.viewer = null;
        this.configManager = new ConfigManager();
        this.simulationRunner = new SimulationRunner();
        this.karolinaRunner = new KarolinaRunner();
        this.runTarget = sessionStorage.getItem('runTarget') || 'local'; // 'local' or 'karolina'

        // Bounding box state (in micrometers - raw mesh coordinates)
        this.boundingBox = {
            xMin: -62, xMax: 15,
            yMin: -19, yMax: 68,
            zMin: -20, zMax: 118
        };

        // Scar tissue state (in micrometers)
        this.scarEnabled = false;
        this.scarBox = { xMin: 0, xMax: 0, yMin: 0, yMax: 0, zMin: 0, zMax: 0 };
        this.scarMargin = 10; // um

        // Simulation parameters
        this.dt = 0.001;
        this.timeSteps = 1000;
        this.bcType = 'one_corner';  // Boundary condition type
        this.partitionMode = 'default';  // Partition mode: 'default' or 'component'

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

        // Voltage time-series plot
        this.voltagePlotChart = null;
        this.pickedVertexIndex = null;
        this._pickedVertexSeries = null;

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
            this.setupRunTarget();
            this.setupKarolinaOptions();
            this.setupSolverSettings();
            this.setupMpiRanks();
            this.setupVoltageControls();
            this.setupButtons();
            this.setupCheckboxes();
            this.setupColormapSelector();
            this.setupScarControls();
            this.setupResultsControls();
            this.setupVideoExport();
            this.setupIterationsChart();
            this.setupResidualChart();
            this.setupVoltagePlot();

            // Setup vertex picking
            this.viewer.setupPickingHandler();
            this.viewer.onVertexPicked = (vertexIdx, worldPos) => {
                if (!this.resultsVizDir) return;
                this.pickedVertexIndex = vertexIdx;
                this.showVoltagePlot(vertexIdx, worldPos);
            };

            // Load config from YAML and populate form
            await this.loadConfigFromYaml();

            // Restore per-mesh IC/scar config from localStorage
            this.loadMeshConfig();

            // Initial update
            this.updateVinitExpression();
            this.updateBoundingBoxVisualization();
            this.updateColorbar();

            // Check for existing iteration data to show compare selector
            this.updateCompareSelector();

            console.log('Application ready');
        } catch (error) {
            console.error('Failed to initialize:', error);
            loadingDiv.style.color = '#e94560';
            loadingDiv.textContent = `Error: ${error.message}`;
        }
    }

    async setupMeshSelector() {
        const selector = document.getElementById('mesh-selector');
        const convertBtn = document.getElementById('convert-mesh');
        const refreshBtn = document.getElementById('refresh-mesh-list');

        // Initial mesh list load (always local to get current mesh for viewer)
        const savedTarget = this.runTarget;
        this.runTarget = 'local';
        await this.refreshMeshList();
        this.runTarget = savedTarget;

        // Handle mesh selection change
        selector.addEventListener('change', async () => {
            const meshName = selector.value;
            await this.onMeshSelected(meshName);
        });

        // Handle convert button (for local viz conversion)
        convertBtn.addEventListener('click', async () => {
            const meshName = selector.value;
            await this.convertMesh(meshName);
        });

        // Handle refresh button (visible in Karolina mode)
        refreshBtn.addEventListener('click', () => {
            this.refreshMeshList();
        });
    }

    async refreshMeshList() {
        const selector = document.getElementById('mesh-selector');
        const remoteConvertArea = document.getElementById('remote-convert-area');

        // Always fetch local meshes
        let localMeshes = [];
        let currentMesh = null;
        let currentConfig = null;
        try {
            const response = await fetch('/api/meshes');
            const data = await response.json();
            localMeshes = data.meshes;
            currentMesh = data.current;
            currentConfig = data.currentConfig;
            this.meshesInfo = data.meshes;
        } catch (error) {
            console.error('Failed to load local mesh list:', error);
            selector.innerHTML = '<option value="">Error loading meshes</option>';
            return;
        }

        if (this.runTarget === 'karolina') {
            // Karolina mode: fetch remote meshes and populate dropdown
            selector.innerHTML = '<option value="">Loading remote meshes...</option>';
            remoteConvertArea.innerHTML = '';
            remoteConvertArea.style.display = 'none';

            try {
                const families = await this.karolinaRunner.listRemoteMeshes();
                const localNames = new Set(localMeshes.map(m => m.name));

                selector.innerHTML = '';
                let hasOptions = false;

                // Build list of converted remote meshes for the dropdown
                // and unconverted ones for convert buttons
                const unconverted = [];

                for (const family of families) {
                    for (const mesh of family.meshes) {
                        // Plain converted mesh
                        if (mesh.converted) {
                            const opt = document.createElement('option');
                            opt.value = mesh.name;
                            const isLocal = localNames.has(mesh.name);
                            const localInfo = localMeshes.find(m => m.name === mesh.name);
                            const vizReady = localInfo && localInfo.converted;
                            let label = mesh.name;
                            if (vizReady) label += ' (ready)';
                            else if (isLocal) label += ' (local, needs viz convert)';
                            else label += ' (remote)';
                            opt.textContent = label;
                            if (mesh.name === currentMesh) opt.selected = true;
                            selector.appendChild(opt);
                            hasOptions = true;
                        } else {
                            unconverted.push({ family: family.family, mesh, color: false });
                        }

                        // Colored variant
                        if (mesh.converted_colored) {
                            const colorName = mesh.name + '_colored';
                            const opt = document.createElement('option');
                            opt.value = colorName;
                            const isLocal = localNames.has(colorName);
                            const localInfo = localMeshes.find(m => m.name === colorName);
                            const vizReady = localInfo && localInfo.converted;
                            let label = colorName;
                            if (vizReady) label += ' (ready)';
                            else if (isLocal) label += ' (local, needs viz convert)';
                            else label += ' (remote)';
                            opt.textContent = label;
                            if (colorName === currentMesh) opt.selected = true;
                            selector.appendChild(opt);
                            hasOptions = true;
                        } else {
                            unconverted.push({ family: family.family, mesh, color: true });
                        }
                    }
                }

                if (!hasOptions) {
                    selector.innerHTML = '<option value="">No converted meshes on Karolina</option>';
                }

                // Show unconverted meshes as dropdown with convert button
                if (unconverted.length > 0) {
                    remoteConvertArea.style.display = 'block';
                    remoteConvertArea.innerHTML = '';
                    const row = document.createElement('div');
                    row.style.cssText = 'display:flex; align-items:center; gap:6px; font-size:0.85em;';
                    const label = document.createElement('label');
                    label.textContent = 'Unconverted:';
                    label.style.color = '#888';
                    row.appendChild(label);
                    const sel = document.createElement('select');
                    sel.className = 'mesh-dropdown';
                    sel.id = 'unconverted-mesh-selector';
                    sel.style.flex = '1';
                    for (const item of unconverted) {
                        const opt = document.createElement('option');
                        const displayName = item.color ? item.mesh.name + '_colored' : item.mesh.name;
                        opt.value = JSON.stringify({ family: item.family, mesh: item.mesh, color: item.color });
                        opt.textContent = displayName;
                        sel.appendChild(opt);
                    }
                    row.appendChild(sel);
                    const btn = document.createElement('button');
                    btn.textContent = 'Convert';
                    btn.className = 'btn-small';
                    btn.addEventListener('click', () => {
                        const val = JSON.parse(sel.value);
                        this.convertRemoteMeshAndRefresh(val.family, val.mesh, val.color);
                    });
                    row.appendChild(btn);
                    remoteConvertArea.appendChild(row);
                }
            } catch (e) {
                selector.innerHTML = '<option value="">Failed to load remote meshes</option>';
                console.error('Failed to load remote meshes:', e);
            }
        } else {
            // Local mode: populate with local meshes
            remoteConvertArea.style.display = 'none';
            selector.innerHTML = '';
            localMeshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh.name;
                option.textContent = mesh.name + (mesh.converted ? '' : ' (not converted)');
                if (mesh.name === currentMesh) {
                    option.selected = true;
                }
                selector.appendChild(option);
            });
        }

        // Set current mesh in loader
        if (currentMesh) {
            this.meshLoader.setMesh(currentMesh);
        }

        // Set current config file
        if (currentConfig) {
            this.configManager.setConfigFile(currentConfig);
            this.simulationRunner.setConfigFile(currentConfig);
        }

        // Update status for currently selected mesh
        const selected = selector.value;
        if (selected && this.runTarget === 'local') {
            this.updateMeshStatus(selected, localMeshes);
        }
    }

    async onMeshSelected(meshName) {
        const statusEl = document.getElementById('mesh-status');

        if (this.runTarget === 'karolina') {
            // Karolina mode: auto-download if needed, auto-convert, then select
            await this.onKarolinaMeshSelected(meshName);
        } else {
            // Local mode: check if converted
            const response = await fetch('/api/meshes');
            const data = await response.json();
            const meshInfo = data.meshes.find(m => m.name === meshName);

            if (!meshInfo) return;

            if (meshInfo.converted) {
                await this.selectMesh(meshName);
            } else {
                this.updateMeshStatus(meshName, data.meshes);
            }
        }
    }

    async onKarolinaMeshSelected(meshName) {
        const statusEl = document.getElementById('mesh-status');
        const convertBtn = document.getElementById('convert-mesh');

        statusEl.style.display = 'block';
        convertBtn.style.display = 'none';

        // Step 1: Check if mesh data exists locally
        let localInfo = this.meshesInfo?.find(m => m.name === meshName);

        if (!localInfo) {
            // Need to download from Karolina
            statusEl.className = 'mesh-status pending';
            statusEl.textContent = `Downloading ${meshName} from Karolina...`;

            try {
                await this.karolinaRunner.downloadMeshData(meshName);
                statusEl.textContent = `Downloaded ${meshName}. Checking conversion...`;

                // Refresh local mesh info
                const resp = await fetch('/api/meshes');
                const data = await resp.json();
                this.meshesInfo = data.meshes;
                localInfo = data.meshes.find(m => m.name === meshName);
            } catch (e) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = `Download failed: ${e.message}`;
                return;
            }
        }

        if (!localInfo) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Mesh ${meshName} not found locally after download`;
            return;
        }

        // Step 2: Check if viz conversion exists
        if (!localInfo.converted) {
            statusEl.className = 'mesh-status pending';
            statusEl.textContent = `Converting ${meshName} for visualization...`;

            try {
                await this.convertMesh(meshName);
                // convertMesh handles auto-select on completion
                return;
            } catch (e) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = `Conversion failed: ${e.message}`;
                return;
            }
        }

        // Step 3: Already local and converted - just select it
        await this.selectMesh(meshName);
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

    setupScarControls() {
        const enabledCb = document.getElementById('scar-enabled');
        const controls = document.getElementById('scar-controls');
        const showBoxCb = document.getElementById('show-scar-box');
        const marginSlider = document.getElementById('scar-margin');
        const marginVal = document.getElementById('scar-margin-val');

        enabledCb.addEventListener('change', () => {
            this.scarEnabled = enabledCb.checked;
            controls.style.display = this.scarEnabled ? 'block' : 'none';
            showBoxCb.checked = this.scarEnabled;
            this.onScarChange();
        });

        showBoxCb.addEventListener('change', () => {
            if (this.viewer) {
                this.viewer.setScarBoxVisible(showBoxCb.checked);
            }
        });

        marginSlider.addEventListener('input', () => {
            this.scarMargin = parseFloat(marginSlider.value);
            marginVal.textContent = this.scarMargin;
            this.onScarChange();
        });

        // Scar bounding box sliders (same pattern as excitation box)
        const axes = ['x', 'y', 'z'];
        axes.forEach(axis => {
            const minSlider = document.getElementById(`scar-${axis}-min`);
            const maxSlider = document.getElementById(`scar-${axis}-max`);
            const minValEl = document.getElementById(`scar-${axis}-min-val`);
            const maxValEl = document.getElementById(`scar-${axis}-max-val`);

            const bounds = this.meshBounds[axis];
            const range = bounds[1] - bounds[0];
            const center = (bounds[0] + bounds[1]) / 2;
            const quarter = range / 4;

            minSlider.min = bounds[0];
            minSlider.max = bounds[1];
            minSlider.step = range / 200;
            // Default to center quarter of mesh
            this.scarBox[`${axis}Min`] = center - quarter;
            this.scarBox[`${axis}Max`] = center + quarter;
            minSlider.value = this.scarBox[`${axis}Min`];

            maxSlider.min = bounds[0];
            maxSlider.max = bounds[1];
            maxSlider.step = range / 200;
            maxSlider.value = this.scarBox[`${axis}Max`];

            minValEl.textContent = parseFloat(minSlider.value).toFixed(1);
            maxValEl.textContent = parseFloat(maxSlider.value).toFixed(1);

            minSlider.addEventListener('input', () => {
                const val = parseFloat(minSlider.value);
                this.scarBox[`${axis}Min`] = val;
                minValEl.textContent = val.toFixed(1);
                this.onScarChange();
            });
            maxSlider.addEventListener('input', () => {
                const val = parseFloat(maxSlider.value);
                this.scarBox[`${axis}Max`] = val;
                maxValEl.textContent = val.toFixed(1);
                this.onScarChange();
            });
        });

        // Conductivity inputs trigger preview update
        ['scar-si-dense', 'scar-se-dense', 'scar-si-border', 'scar-se-border'].forEach(id => {
            document.getElementById(id).addEventListener('input', () => this.onScarChange());
        });
    }

    onScarChange() {
        this.saveMeshConfig();
        if (this.viewer) {
            this.viewer.updateScarBox(this.scarBox, this.scarMargin, this.scarEnabled);
            // Update scar zone mask for desaturation
            this.viewer.setScarZones(this.scarBox, this.scarMargin, this.scarEnabled);

            // Re-render current results with desaturation, or preview on resting mesh
            if (this.resultsVizDir) {
                const idx = parseInt(document.getElementById('result-time').value);
                this.showResultsAtTime(idx);
            } else if (this.scarEnabled) {
                this.viewer.highlightScarZones(this.scarBox, this.scarMargin);
            }
        }
    }

    _updateScarSlidersFromConfig(region) {
        const box = region.box;
        const margin = region.margin || 10;

        // Update sliders and display values
        ['x', 'y', 'z'].forEach(axis => {
            const minSlider = document.getElementById(`scar-${axis}-min`);
            const maxSlider = document.getElementById(`scar-${axis}-max`);
            const minValEl = document.getElementById(`scar-${axis}-min-val`);
            const maxValEl = document.getElementById(`scar-${axis}-max-val`);

            minSlider.value = box[`${axis}Min`];
            maxSlider.value = box[`${axis}Max`];
            minValEl.textContent = box[`${axis}Min`].toFixed(1);
            maxValEl.textContent = box[`${axis}Max`].toFixed(1);
        });

        document.getElementById('scar-margin').value = margin;
        document.getElementById('scar-margin-val').textContent = margin;

        // Update conductivity inputs if present
        const dense = region.dense;
        const border = region.border;
        if (dense) {
            document.getElementById('scar-si-dense').value = dense.sigma_i;
            document.getElementById('scar-se-dense').value = dense.sigma_e;
        }
        if (border) {
            document.getElementById('scar-si-border').value = border.sigma_i;
            document.getElementById('scar-se-border').value = border.sigma_e;
        }
    }

    saveMeshConfig() {
        const meshName = this.meshLoader.currentMesh;
        if (!meshName) return;
        const config = {
            boundingBox: { ...this.boundingBox },
            vExcited: this.vExcited,
            vResting: this.vResting,
            scarEnabled: this.scarEnabled,
            scarBox: { ...this.scarBox },
            scarMargin: this.scarMargin,
            scarConductivities: {
                dense: {
                    sigma_i: parseFloat(document.getElementById('scar-si-dense').value),
                    sigma_e: parseFloat(document.getElementById('scar-se-dense').value),
                },
                border: {
                    sigma_i: parseFloat(document.getElementById('scar-si-border').value),
                    sigma_e: parseFloat(document.getElementById('scar-se-border').value),
                },
            },
        };
        localStorage.setItem(`meshConfig_${meshName}`, JSON.stringify(config));
    }

    loadMeshConfig() {
        const meshName = this.meshLoader.currentMesh;
        if (!meshName) return;
        const raw = localStorage.getItem(`meshConfig_${meshName}`);
        if (!raw) return;

        let config;
        try { config = JSON.parse(raw); } catch { return; }

        // Restore bounding box
        if (config.boundingBox) {
            Object.assign(this.boundingBox, config.boundingBox);
            ['x', 'y', 'z'].forEach(axis => {
                const minSlider = document.getElementById(`${axis}-min`);
                const maxSlider = document.getElementById(`${axis}-max`);
                minSlider.value = this.boundingBox[`${axis}Min`];
                maxSlider.value = this.boundingBox[`${axis}Max`];
                document.getElementById(`${axis}-min-val`).textContent = parseFloat(minSlider.value).toFixed(1);
                document.getElementById(`${axis}-max-val`).textContent = parseFloat(maxSlider.value).toFixed(1);
            });
        }

        // Restore voltage controls
        if (config.vExcited !== undefined) {
            this.vExcited = config.vExcited;
            const el = document.getElementById('v-excited');
            el.value = this.vExcited;
            document.getElementById('v-excited-val').textContent = this.vExcited;
        }
        if (config.vResting !== undefined) {
            this.vResting = config.vResting;
            const el = document.getElementById('v-resting');
            el.value = this.vResting;
            document.getElementById('v-resting-val').textContent = this.vResting;
        }

        // Restore scar config
        if (config.scarEnabled !== undefined) {
            this.scarEnabled = config.scarEnabled;
            document.getElementById('scar-enabled').checked = this.scarEnabled;
            document.getElementById('scar-controls').style.display = this.scarEnabled ? 'block' : 'none';
        }
        if (config.scarBox) {
            Object.assign(this.scarBox, config.scarBox);
        }
        if (config.scarMargin !== undefined) {
            this.scarMargin = config.scarMargin;
        }
        if (config.scarBox || config.scarMargin !== undefined) {
            this._updateScarSlidersFromConfig({
                box: this.scarBox,
                margin: this.scarMargin,
                dense: config.scarConductivities?.dense,
                border: config.scarConductivities?.border,
            });
        }
    }

    getScarConfig() {
        if (!this.scarEnabled) return null;
        return {
            regions: [{
                box: { ...this.scarBox },
                margin: this.scarMargin,
                dense: {
                    sigma_i: parseFloat(document.getElementById('scar-si-dense').value),
                    sigma_e: parseFloat(document.getElementById('scar-se-dense').value),
                },
                border: {
                    sigma_i: parseFloat(document.getElementById('scar-si-border').value),
                    sigma_e: parseFloat(document.getElementById('scar-se-border').value),
                },
            }],
            healthy: { sigma_i: 4.0, sigma_e: 20.0 },
            conversionFactor: this.conversionFactor,
        };
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

        // Partition mode
        const partitionModeSelect = document.getElementById('partition-mode');
        if (partitionModeSelect) {
            this.partitionMode = partitionModeSelect.value;
            partitionModeSelect.addEventListener('change', (e) => {
                this.partitionMode = e.target.value;
            });
        }
    }

    setupRunTarget() {
        const selector = document.getElementById('run-target');
        const karolinaOptions = document.getElementById('karolina-options');
        const containerStatusRow = document.getElementById('karolina-container-status-row');
        const statusDot = document.getElementById('karolina-status-dot');
        const refreshBtn = document.getElementById('refresh-mesh-list');
        const mpiRanksRow = document.getElementById('mpi-ranks').closest('.param-row');

        const applyTarget = async (target) => {
            if (target === 'karolina') {
                karolinaOptions.style.display = 'block';
                containerStatusRow.style.display = 'flex';
                statusDot.style.display = 'inline';
                refreshBtn.style.display = 'inline-block';
                mpiRanksRow.style.display = 'none';
                // Check connectivity + containers
                statusDot.textContent = '...';
                statusDot.title = 'Checking SSH...';
                try {
                    const result = await this.karolinaRunner.checkConnectivity();
                    const ok = result.available;
                    statusDot.textContent = '\u25CF';
                    statusDot.style.color = ok ? '#4ade80' : '#e94560';
                    statusDot.title = ok ? 'SSH connected' : 'SSH unreachable';
                    // Show container status
                    if (ok && result.containers) {
                        const c = result.containers;
                        const parts = [];
                        parts.push(`DOLFINx: ${c.dolfinx ? 'ready' : 'missing'}`);
                        parts.push(`Ginkgo: ${c.ginkgo ? 'ready' : 'missing'}`);
                        const containerEl = document.getElementById('karolina-container-status');
                        containerEl.textContent = parts.join(' | ');
                        containerEl.style.color = c.dolfinx ? '#4ade80' : '#e94560';
                    }
                    // Refresh mesh list with remote meshes
                    if (ok) this.refreshMeshList();
                } catch (e) {
                    statusDot.textContent = '\u25CF';
                    statusDot.style.color = '#e94560';
                    statusDot.title = 'SSH check failed';
                }
                document.getElementById('download-karolina-results').style.display = 'block';
                // Refresh sim list to include remote sims
                this.loadSimulationList();
            } else {
                karolinaOptions.style.display = 'none';
                containerStatusRow.style.display = 'none';
                statusDot.style.display = 'none';
                refreshBtn.style.display = 'none';
                mpiRanksRow.style.display = 'flex';
                document.getElementById('download-karolina-results').style.display = 'none';
                // Refresh mesh list with local meshes
                this.refreshMeshList();
            }
        };

        selector.addEventListener('change', async () => {
            this.runTarget = selector.value;
            sessionStorage.setItem('runTarget', this.runTarget);
            await applyTarget(this.runTarget);
        });

        // Restore persisted run target on page load
        if (this.runTarget !== 'local') {
            selector.value = this.runTarget;
            applyTarget(this.runTarget);
        }
    }

    async convertRemoteMeshAndRefresh(family, mesh, color) {
        const statusEl = document.getElementById('remote-mesh-convert-status');
        const outputEl = document.getElementById('remote-mesh-convert-output');
        const outputPrefix = color ? mesh.name + '_colored' : mesh.name;

        statusEl.className = 'mesh-status pending';
        statusEl.textContent = `Converting ${outputPrefix} on Karolina...`;
        statusEl.style.display = 'block';
        outputEl.style.display = 'block';
        outputEl.textContent = '';

        try {
            await this.karolinaRunner.convertRemoteMesh(
                family, mesh.pts, mesh.elem, outputPrefix, color,
                (text) => {
                    outputEl.textContent += text;
                    outputEl.scrollTop = outputEl.scrollHeight;
                }
            );

            statusEl.className = 'mesh-status converted';
            statusEl.textContent = `Conversion of ${outputPrefix} complete!`;

            // Refresh mesh list to show the newly converted mesh in the dropdown
            await this.refreshMeshList();
        } catch (e) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Conversion failed: ${e.message}`;
        }
    }

    setupKarolinaOptions() {
        const nodesInput = document.getElementById('karolina-nodes');
        const ntasksInput = document.getElementById('karolina-ntasks');
        const totalRanksSpan = document.getElementById('karolina-total-ranks');

        const updateTotal = () => {
            const nodes = parseInt(nodesInput.value) || 1;
            const ntasks = parseInt(ntasksInput.value) || 128;
            totalRanksSpan.textContent = nodes * ntasks;
        };

        nodesInput.addEventListener('input', updateTotal);
        ntasksInput.addEventListener('input', updateTotal);
        updateTotal();

        // Track active jobs
        this.karolinaJobs = {};
        this.compareDatasets = [];  // extra datasets from comparison runs
    }

    renderJobEntry(jobInfo) {
        const container = document.getElementById('karolina-jobs-list');
        const jobId = jobInfo.job_id;
        const entryId = `karolina-job-${jobId}`;

        // Don't duplicate
        if (document.getElementById(entryId)) return;

        const entry = document.createElement('div');
        entry.id = entryId;
        entry.style.cssText = 'border:1px solid #444; border-radius:6px; padding:8px; margin-bottom:8px; background:#1a1a1a;';
        entry.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:bold; color:#ccc;">${jobInfo.label || jobId}</span>
                <span style="font-size:0.8em; color:#888;">ID: ${jobId}</span>
            </div>
            <div class="param-row" style="margin:4px 0;">
                <label>Status:</label>
                <span class="job-status" style="font-weight:bold;">PENDING</span>
            </div>
            <div style="margin-top:6px;">
                <button class="btn btn-danger btn-cancel" style="font-size:0.75em; padding:2px 8px;">Cancel</button>
                <button class="btn btn-success btn-download" style="font-size:0.75em; padding:2px 8px; display:none;">Download</button>
                <button class="btn btn-toggle-log" style="font-size:0.75em; padding:2px 8px; background:#555; color:#ccc;">Log</button>
            </div>
            <pre class="job-log output-console" style="max-height:150px; display:none; margin-top:6px; font-size:0.75em;"></pre>
        `;

        // Cancel button
        entry.querySelector('.btn-cancel').addEventListener('click', async () => {
            try {
                await this.karolinaRunner.cancel(jobId);
                this.karolinaRunner.stopPolling(jobId);
                entry.querySelector('.job-status').textContent = 'CANCELLED';
                entry.querySelector('.job-status').style.color = '#e94560';
                entry.querySelector('.btn-cancel').style.display = 'none';
            } catch (e) {
                alert('Cancel failed: ' + e.message);
            }
        });

        // Download button
        entry.querySelector('.btn-download').addEventListener('click', async () => {
            const btn = entry.querySelector('.btn-download');
            btn.disabled = true;
            btn.textContent = 'Downloading...';
            try {
                const outName = this.karolinaJobs[jobId]?.out_name;
                await this.karolinaRunner.downloadResults(outName, () => {});
                btn.textContent = 'Downloaded';
                await this.loadSimulationList();
                this.updateCompareSelector();
            } catch (e) {
                btn.textContent = 'Download Failed';
            } finally {
                setTimeout(() => { btn.textContent = 'Download'; btn.disabled = false; }, 3000);
            }
        });

        // Toggle log
        entry.querySelector('.btn-toggle-log').addEventListener('click', () => {
            const log = entry.querySelector('.job-log');
            log.style.display = log.style.display === 'none' ? 'block' : 'none';
        });

        container.prepend(entry);
    }

    updateJobStatus(jobId, data) {
        const entry = document.getElementById(`karolina-job-${jobId}`);
        if (!entry) return;

        const statusEl = entry.querySelector('.job-status');
        const cancelBtn = entry.querySelector('.btn-cancel');
        const downloadBtn = entry.querySelector('.btn-download');
        const logEl = entry.querySelector('.job-log');

        statusEl.textContent = data.status || '-';
        if (data.log) {
            logEl.textContent = data.log;
            logEl.scrollTop = logEl.scrollHeight;
        }

        const s = data.status;
        if (s === 'RUNNING') {
            statusEl.style.color = '#4ade80';
        } else if (s === 'PENDING') {
            statusEl.style.color = '#fbbf24';
        } else if (s === 'COMPLETED') {
            statusEl.style.color = '#4ade80';
            cancelBtn.style.display = 'none';
            downloadBtn.style.display = 'inline-block';
            // Auto-download iterations for comparison
            this.autoDownloadIterations(jobId);
        } else if (['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'].includes(s)) {
            statusEl.style.color = '#e94560';
            cancelBtn.style.display = 'none';
        }
    }

    async autoDownloadIterations(jobId) {
        const job = this.karolinaJobs[jobId];
        if (!job?.out_name || job._iterationsDownloaded) return;
        job._iterationsDownloaded = true;

        try {
            await this.karolinaRunner.downloadIterations(job.out_name);
            await this.loadSimulationList();
            this.updateCompareSelector();
        } catch (e) {
            console.warn(`Auto-download iterations for job ${jobId} failed:`, e);
        }
    }

    async downloadKarolinaSimulation() {
        const simName = document.getElementById('simulation-selector').value;
        if (!simName) {
            alert('Please select a simulation first');
            return;
        }

        const btn = document.getElementById('download-karolina-results');
        const statusEl = document.getElementById('results-status');
        const progressEl = document.getElementById('results-download-progress');
        const progressBar = document.getElementById('results-download-bar');
        const progressText = document.getElementById('results-download-text');

        btn.disabled = true;
        btn.textContent = 'Downloading...';
        statusEl.style.display = 'none';
        progressEl.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting download...';

        try {
            const result = await this.karolinaRunner.downloadResults(simName, (data) => {
                const pct = data.bytes_total > 0
                    ? Math.round(100 * data.bytes_done / data.bytes_total)
                    : 0;
                progressBar.style.width = pct + '%';
                const doneMB = (data.bytes_done / 1048576).toFixed(1);
                const totalMB = (data.bytes_total / 1048576).toFixed(1);
                progressText.textContent = data.file
                    ? `${doneMB} / ${totalMB} MB — ${data.file}`
                    : `${doneMB} / ${totalMB} MB`;
            });
            progressBar.style.width = '100%';
            progressText.textContent = 'Done';
            statusEl.className = 'mesh-status converted';
            statusEl.textContent = result.message;
            statusEl.style.display = 'block';
            await this.loadSimulationList();
        } catch (e) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = 'Download failed: ' + e.message;
            statusEl.style.display = 'block';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Download from Karolina';
            setTimeout(() => { progressEl.style.display = 'none'; }, 2000);
        }
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
                nativeAssembly: true,  // Default to native assembly
                ddMatrix: false,       // Domain decomposition matrix format
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
        const bddcLocalHypreOptions = document.getElementById('bddc-local-hypre-options');
        const bddcLocalStoppingOptions = document.getElementById('bddc-local-stopping-options');
        document.getElementById('bddc-local-solver').addEventListener('change', (e) => {
            this.solverConfig.ginkgo.bddc.localSolver = e.target.value;
            // Show/hide local AMG options
            bddcLocalAmgOptions.style.display = e.target.value === 'amg' ? 'block' : 'none';
            // Show/hide local Hypre options
            bddcLocalHypreOptions.style.display = e.target.value === 'hypre' ? 'block' : 'none';
            // Show/hide local stopping criteria (for iterative solvers: ilu, ic, amg, hypre)
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
            // Show/hide coarse BDDC local solver options
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

        // PETSc PC type -> BDDC options
        const petscBddcOptions = document.getElementById('petsc-bddc-options');
        const petscPcType = document.getElementById('petsc-pc-type');
        petscBddcOptions.style.display = petscPcType.value === 'bddc' ? 'block' : 'none';

        // Native assembly -> DD matrix row
        if (!nativeAssemblyCheckbox.checked) {
            ddMatrixRow.style.display = 'none';
        } else {
            ddMatrixRow.style.display = 'flex';
        }

        // Preconditioner -> AMG/BDDC options
        amgOptions.style.display = ginkgoPrecond.value === 'amg' ? 'block' : 'none';
        bddcOptions.style.display = ginkgoPrecond.value === 'bddc' ? 'block' : 'none';

        // BDDC local solver -> local AMG/Hypre options and stopping criteria
        bddcLocalAmgOptions.style.display = bddcLocalSolver.value === 'amg' ? 'block' : 'none';
        document.getElementById('bddc-local-hypre-options').style.display = bddcLocalSolver.value === 'hypre' ? 'block' : 'none';
        document.getElementById('bddc-local-stopping-options').style.display = (bddcLocalSolver.value !== 'direct' && bddcLocalSolver.value !== 'direct_lu') ? 'block' : 'none';

        // BDDC coarse solver -> coarse BDDC local solver options
        const bddcCoarseSolver = document.getElementById('bddc-coarse-solver');
        document.getElementById('bddc-coarse-bddc-options').style.display = bddcCoarseSolver.value === 'bddc' ? 'flex' : 'none';

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

    async loadConfigFromYaml() {
        try {
            const config = await this.configManager.getConfig();
            if (!config || config.error) return;

            // Helper to set a select/input value
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

            // Now sync UI visibility and internal state with loaded form values
            this.initSolverSettingsUI();

            console.log('Config loaded from YAML:', this.configManager.configFile);
        } catch (error) {
            console.warn('Could not load config from YAML:', error.message);
        }
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

        // Karolina download button in results section
        const dlBtn = document.getElementById('download-karolina-results');
        dlBtn.addEventListener('click', () => this.downloadKarolinaSimulation());

        timeSlider.addEventListener('input', () => {
            if (this.resultsVizDir && this.resultsTimeSteps && this.resultsTimeSteps.length > 0) {
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
            // Fetch local simulations
            const response = await fetch('/api/simulations');
            const data = await response.json();
            const localNames = new Set(data.simulations.map(s => s.name));

            simSelector.innerHTML = '<option value="">Select simulation...</option>';

            data.simulations.forEach(sim => {
                const option = document.createElement('option');
                option.value = sim.name;
                option.textContent = sim.name + (sim.has_viz_data ? '' : ' (will generate viz data)');
                simSelector.appendChild(option);
            });

            // If in Karolina mode, also fetch remote simulations
            if (this.runTarget === 'karolina') {
                try {
                    const remoteResp = await fetch('/api/karolina/remote-simulations');
                    const remoteData = await remoteResp.json();
                    if (remoteData.simulations) {
                        for (const name of remoteData.simulations) {
                            if (!localNames.has(name)) {
                                const option = document.createElement('option');
                                option.value = name;
                                option.textContent = name + ' (remote)';
                                simSelector.appendChild(option);
                            }
                        }
                    }
                } catch (e) {
                    console.warn('Failed to load remote simulations:', e);
                }
            }

            // Select first simulation by default if available
            if (data.simulations.length > 0) {
                simSelector.value = data.simulations[0].name;
                this.selectedSimulation = data.simulations[0].name;
            }

            // Update comparison selector
            this.updateCompareSelector();
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
                    // Pass DOF types to viewer for differentiated rendering
                    this.viewer.setInterfaceDofTypes(this.interfaceDofTypes);
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
        this.viewer.setInterfaceDofTypes(null);
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
        this.viewer.setInterfaceDofTypes(null);
    }

    onBoundingBoxChange() {
        this.updateVinitExpression();
        this.updateBoundingBoxVisualization();
        this.saveMeshConfig();
    }

    updateVinitExpression() {
        // Expression is generated on demand when running simulation
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

    async _generateVizData(simName, statusEl) {
        /**
         * Run viz data generation via SSE endpoint, showing progress.
         * Returns true on success, throws on error.
         */
        return new Promise((resolve, reject) => {
            // EventSource doesn't support POST, so use fetch + ReadableStream
            fetch('/api/generate-viz', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dir: simName }),
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                const pump = () => {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            reject(new Error('Stream ended without completion'));
                            return;
                        }
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // keep incomplete line

                        for (const line of lines) {
                            if (!line.startsWith('data: ')) continue;
                            try {
                                const event = JSON.parse(line.slice(6));
                                if (event.type === 'progress') {
                                    statusEl.className = 'mesh-status';
                                    statusEl.textContent = `Generating viz data... ${event.percent}% - ${event.message}`;
                                    statusEl.style.display = 'block';
                                } else if (event.type === 'complete') {
                                    resolve(true);
                                    return;
                                } else if (event.type === 'error') {
                                    reject(new Error(event.message));
                                    return;
                                }
                            } catch (e) { /* skip malformed */ }
                        }
                        pump();
                    }).catch(reject);
                };
                pump();
            }).catch(reject);
        });
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
            const url = `/api/results?dir=${encodeURIComponent(simName)}`;

            // If regenerate requested, generate first
            if (regenerate) {
                loadBtn.textContent = 'Generating viz data...';
                statusEl.className = 'mesh-status';
                statusEl.textContent = 'Starting viz data generation...';
                statusEl.style.display = 'block';
                await this._generateVizData(simName, statusEl);
                statusEl.textContent = 'Viz data generated, loading results...';
                loadBtn.textContent = 'Loading results...';
            }

            // Load results
            let response = await fetch(url);

            // If viz data doesn't exist yet, generate it
            if (!response.ok) {
                const errData = await response.json();
                if (errData.error && errData.error.includes('not found')) {
                    loadBtn.textContent = 'Generating viz data...';
                    statusEl.className = 'mesh-status';
                    statusEl.textContent = 'Starting viz data generation...';
                    statusEl.style.display = 'block';
                    await this._generateVizData(simName, statusEl);
                    statusEl.textContent = 'Viz data generated, loading results...';
                    loadBtn.textContent = 'Loading results...';
                    response = await fetch(url);
                    if (!response.ok) {
                        const err2 = await response.json();
                        throw new Error(err2.error || 'Failed to load results after generation');
                    }
                } else {
                    throw new Error(errData.error || 'Failed to load results');
                }
            }

            const data = await response.json();
            const vizDir = data.vizDataDir;

            // Reload mesh from simulation's viz data if different from current
            if (vizDir && vizDir !== this.meshLoader.currentMesh) {
                loadBtn.textContent = 'Reloading mesh...';
                this.meshLoader.setMesh(vizDir);
                const meshData = await this.meshLoader.load();
                this.meshBounds = meshData.metadata.bounds;
                this.conversionFactor = meshData.metadata.mesh_conversion_factor;
                await this.viewer.reloadMesh(meshData);
            }

            // Store viz dir for binary fetches; voltage data loaded on demand
            this.resultsVizDir = vizDir;
            this.resultsData = null; // voltages loaded per-timestep now
            this._voltageCache = {};
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

            // Apply scar config from the simulation's YAML if present
            if (data.scarConfig && data.scarConfig.regions && data.scarConfig.regions.length > 0) {
                const region = data.scarConfig.regions[0];
                this.scarBox = { ...region.box };
                this.scarMargin = region.margin || 10;
                this.scarEnabled = true;
                document.getElementById('scar-enabled').checked = true;
                document.getElementById('scar-controls').style.display = 'block';
                // Update scar sliders to match
                this._updateScarSlidersFromConfig(region);
                // Set scar zone mask for desaturation
                this.viewer.setScarZones(this.scarBox, this.scarMargin, true);
                this.viewer.updateScarBox(this.scarBox, this.scarMargin, true);
            } else {
                this.viewer.setScarZones(null, 0, false);
            }

            // Load iterations data if available
            if (data.iterations && data.iterations.length > 0) {
                this.setIterationsData(data.iterations);
                this.showIterationsChart();
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

            // Load MPI rank data as binary if available
            if (data.hasRankData && data.numRanks) {
                loadBtn.textContent = 'Loading rank data...';
                const binBase = `/api/results/binary/${encodeURIComponent(vizDir)}`;

                // Fetch rank binary data in parallel
                const [ranksResp, ecsRanksResp, cutRanksResp, dofResp, ecsDofResp] = await Promise.all([
                    fetch(`${binBase}/dof_ranks.bin`),
                    data.hasEcsRanks ? fetch(`${binBase}/ecs_ranks.bin`) : Promise.resolve(null),
                    data.hasCutRanks ? fetch(`${binBase}/cut_ranks.bin`) : Promise.resolve(null),
                    data.hasDofIndices ? fetch(`${binBase}/facet_orig_vertices.bin`) : Promise.resolve(null),
                    data.hasEcsDofIndices ? fetch(`${binBase}/ecs_orig_vertices.bin`) : Promise.resolve(null),
                ]);

                this.ranksData = new Int32Array(await ranksResp.arrayBuffer());
                this.numRanks = data.numRanks;
                this.rankCentroids = data.rankCentroids;
                this.globalCentroid = data.globalCentroid;

                this.ecsRanksData = ecsRanksResp ? new Int32Array(await ecsRanksResp.arrayBuffer()) : null;
                this.cutRanksData = cutRanksResp ? new Int32Array(await cutRanksResp.arrayBuffer()) : null;

                this.viewer.setNumRanks(data.numRanks);
                this.viewer.setExplosionData(
                    this.ranksData,
                    this.ecsRanksData,
                    this.cutRanksData,
                    data.rankCentroids,
                    data.globalCentroid
                );

                if (dofResp) {
                    this.viewer.setDofIndices(new Uint32Array(await dofResp.arrayBuffer()));
                }
                if (ecsDofResp) {
                    this.viewer.setEcsDofIndices(new Uint32Array(await ecsDofResp.arrayBuffer()));
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

            // Show first timestep (loads voltage binary on demand)
            await this.showResultsAtTime(0);

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

    async showResultsAtTime(timeIndex) {
        if (!this.resultsVizDir || !this.viewer) return;

        // Only update voltage colors if not in partition mode
        const showPartition = document.getElementById('show-partition').checked;
        if (!showPartition) {
            // Fetch voltage binary on demand with caching
            if (!this._voltageCache[timeIndex]) {
                const url = `/api/results/binary/${encodeURIComponent(this.resultsVizDir)}/${timeIndex}.bin`;
                const resp = await fetch(url);
                if (resp.ok) {
                    this._voltageCache[timeIndex] = new Float32Array(await resp.arrayBuffer());
                }
            }
            const voltages = this._voltageCache[timeIndex];
            if (voltages) {
                this.viewer.updateVoltageColors(voltages);
            }
        }

        this.updateVoltagePlotTimeMarker(timeIndex);
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
                bc_type: this.bcType,
                partition_mode: this.partitionMode
            };

            // Update local state to match
            this.solverConfig.backend = solverBackend;
            this.solverConfig.petsc.kspType = kspType;
            this.solverConfig.petsc.pcType = pcType;
            this.solverConfig.rtol = rtol;
            this.solverConfig.atol = atol;

            // Update config via API
            await this.configManager.updateConfig(configUpdates);

            // Update scar tissue config (writes sigma_i/sigma_e expressions)
            const scarConfig = this.getScarConfig();
            const scarPayload = scarConfig || { regions: [], healthy: { sigma_i: 4.0, sigma_e: 20.0 } };
            scarPayload.conversionFactor = this.conversionFactor;
            scarPayload.file = this.configManager.configFile;
            await fetch('/api/config/scar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(scarPayload),
            });

            // If using PETSc BDDC, update petsc_bddc config via special endpoint
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
                await this.configManager.updatePetscBddcConfig(petscBddcConfig);
            }

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
                        localMaxIterations: parseInt(document.getElementById('bddc-local-max-iter').value),
                        localTolerance: parseFloat(document.getElementById('bddc-local-tolerance').value),
                        coarseSolver: document.getElementById('bddc-coarse-solver').value,
                        coarseMaxIterations: parseInt(document.getElementById('bddc-coarse-max-iter').value),
                        coarseBddcLocalSolver: document.getElementById('bddc-coarse-bddc-local-solver').value,
                        vertices: document.getElementById('bddc-vertices').checked,
                        edges: document.getElementById('bddc-edges').checked,
                        faces: document.getElementById('bddc-faces').checked,
                        repartitionCoarse: document.getElementById('bddc-repartition-coarse').checked,
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

        // Save conditions for local runs (Karolina jobs handle this in submit)
        if (this.runTarget !== 'karolina') {
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

        // Branch on run target
        if (this.runTarget === 'karolina') {
            await this.runSimulationKarolina(statusEl, outputEl, runBtn);
        } else {
            await this.runSimulationLocal(statusEl, outputEl, runBtn);
        }
    }

    async runSimulationLocal(statusEl, outputEl, runBtn) {
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

    getConditionsSnapshot() {
        const solverBackend = document.getElementById('solver-backend').value;
        const precond = solverBackend === 'ginkgo'
            ? document.getElementById('ginkgo-precond').value
            : document.getElementById('petsc-pc-type').value;
        let localSolver = null;
        if (solverBackend === 'ginkgo' && precond === 'bddc') {
            localSolver = document.getElementById('bddc-local-solver').value;
        } else if (solverBackend === 'petsc' && precond === 'bddc') {
            localSolver = document.getElementById('petsc-bddc-local-solver').value;
        }

        const nodesEl = document.getElementById('karolina-nodes');
        const ntasksEl = document.getElementById('karolina-ntasks');
        const mpiRanksEl = document.getElementById('mpi-ranks');
        const nRanks = nodesEl && ntasksEl
            ? (parseInt(nodesEl.value) || 1) * (parseInt(ntasksEl.value) || 1)
            : (mpiRanksEl ? parseInt(mpiRanksEl.value) || 1 : 1);

        const conditions = {
            mesh: this.meshLoader.currentMesh,
            solver: solverBackend,
            preconditioner: precond,
            localSolver: localSolver,
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
    }

    async runSimulationKarolina(statusEl, outputEl, runBtn) {
        outputEl.style.display = 'none';
        const jobSection = document.getElementById('karolina-job-section');

        try {
            statusEl.className = 'status visible running';
            statusEl.textContent = 'Submitting to Karolina...';

            const configFile = this.configManager.configFile || 'input_pepe36_colored.yml';
            const label = document.getElementById('karolina-job-label').value.trim() || undefined;
            const conditions = this.getConditionsSnapshot();

            const options = {
                config: configFile,
                nodes: parseInt(document.getElementById('karolina-nodes').value) || 1,
                ntasks_per_node: parseInt(document.getElementById('karolina-ntasks').value) || 128,
                walltime: document.getElementById('karolina-walltime').value || '01:00:00',
                partition: document.getElementById('karolina-partition').value || 'qcpu_exp',
                account: document.getElementById('karolina-account').value || 'eu-26-11',
                solver_backend: document.getElementById('solver-backend').value || 'petsc',
                label,
                conditions,
            };

            const result = await this.karolinaRunner.submit(options);
            const jobId = result.job_id;

            // Store job info
            this.karolinaJobs[jobId] = {
                ...result,
                conditions_hash: result.conditions_hash,
            };

            // Show job section and render entry
            jobSection.style.display = 'block';
            this.renderJobEntry(result);

            statusEl.className = 'status visible success';
            statusEl.textContent = `Job ${jobId} submitted!`;

            // Start polling this job
            this.karolinaRunner.startPolling(jobId, (data) => {
                this.updateJobStatus(jobId, data);
                if (data.out_name) {
                    this.karolinaJobs[jobId].out_name = data.out_name;
                }
            });

        } catch (error) {
            statusEl.className = 'status visible error';
            statusEl.textContent = 'Submission failed: ' + error.message;
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
            const fps = parseInt(document.getElementById('video-fps').value);

            if (!this.resultsVizDir || !this.resultsTimeSteps || this.resultsTimeSteps.length === 0) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = 'Load results first before exporting video';
                statusEl.style.display = 'block';
                return;
            }

            exportBtn.disabled = true;
            progressBar.style.display = 'block';
            progressFill.style.width = '0%';
            progressText.textContent = 'Starting capture session...';
            statusEl.style.display = 'none';
            downloadLink.style.display = 'none';

            try {
                // 1. Start capture session on server
                const startResp = await fetch('/api/video/start-capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fps })
                });
                const { session_id } = await startResp.json();

                const numFrames = this.resultsTimeSteps.length;

                // 2. Capture each frame from the viewer
                for (let i = 0; i < numFrames; i++) {
                    // Update voltage display (same as moving the time slider)
                    await this.showResultsAtTime(i);

                    // Force render
                    this.viewer.renderer.render(this.viewer.scene, this.viewer.camera);

                    // Composite frame with overlays
                    const blob = await this._captureViewerFrame(i);

                    // Send frame to server
                    await fetch(`/api/video/frame/${session_id}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'image/jpeg' },
                        body: blob
                    });

                    const pct = Math.round(((i + 1) / numFrames) * 80);
                    progressFill.style.width = `${pct}%`;
                    progressText.textContent = `Capturing frame ${i + 1}/${numFrames}`;
                }

                // 3. Finalize: tell server to encode video
                progressText.textContent = 'Encoding video...';
                progressFill.style.width = '85%';

                const finishResp = await fetch(`/api/video/finish-capture/${session_id}`, {
                    method: 'POST'
                });
                const result = await finishResp.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                progressFill.style.width = '100%';
                progressText.textContent = 'Complete!';

                statusEl.className = 'mesh-status converted';
                statusEl.textContent = 'Video exported successfully!';
                statusEl.style.display = 'block';

                downloadLink.href = `/api/video/download/${result.filename}`;
                downloadLink.textContent = `Download ${result.filename}`;
                downloadLink.style.display = 'block';

            } catch (error) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = `Export failed: ${error.message}`;
                statusEl.style.display = 'block';
            } finally {
                exportBtn.disabled = false;
                setTimeout(() => { progressBar.style.display = 'none'; }, 2000);
            }
        });
    }

    /**
     * Capture the current viewer as a JPEG blob, compositing the 3D canvas
     * with colorbar and voltage plot overlays.
     */
    _captureViewerFrame(timeIndex) {
        return new Promise((resolve) => {
            const glCanvas = this.viewer.renderer.domElement;
            const w = glCanvas.width;
            const h = glCanvas.height;

            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');

            // Draw 3D scene
            ctx.drawImage(glCanvas, 0, 0, w, h);

            // Draw time label (top-left)
            const time = this.resultsTimeSteps[timeIndex];
            ctx.font = 'bold 18px -apple-system, sans-serif';
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(10, 10, ctx.measureText(`t = ${time.toFixed(3)} ms`).width + 16, 30);
            ctx.fillStyle = '#fff';
            ctx.fillText(`t = ${time.toFixed(3)} ms`, 18, 32);

            // Draw colorbar (top-right)
            const cbW = 30, cbH = 150, cbPad = 20;
            const cbX = w - cbW - cbPad - 70, cbY = cbPad;

            // Background
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(cbX - 8, cbY - 8, cbW + 80, cbH + 16);

            // Gradient
            const cm = this.viewer.colormaps[this.viewer.colormap];
            const grad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
            for (let i = 0; i < cm.colors.length; i++) {
                const [r, g, b] = cm.colors[i];
                const pos = 1 - cm.positions[i]; // Invert for top-to-bottom
                grad.addColorStop(pos, `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`);
            }
            ctx.fillStyle = grad;
            ctx.fillRect(cbX, cbY, cbW, cbH);

            // Labels
            ctx.font = '12px -apple-system, sans-serif';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'left';
            ctx.fillText(`${Math.round(this.viewer.vMax)} mV`, cbX + cbW + 6, cbY + 10);
            ctx.fillText(`${Math.round((this.viewer.vMax + this.viewer.vMin) / 2)} mV`, cbX + cbW + 6, cbY + cbH / 2 + 4);
            ctx.fillText(`${Math.round(this.viewer.vMin)} mV`, cbX + cbW + 6, cbY + cbH);

            // Draw voltage plot if a vertex is picked
            const plotPanel = document.getElementById('voltage-plot-panel');
            if (plotPanel && plotPanel.style.display !== 'none') {
                const plotCanvas = document.getElementById('voltage-plot-canvas');
                if (plotCanvas) {
                    const plotW = plotCanvas.width;
                    const plotH = plotCanvas.height;
                    // Draw background
                    ctx.fillStyle = 'rgba(10, 10, 26, 0.92)';
                    ctx.fillRect(15, h - plotH - 50, plotW + 20, plotH + 40);
                    // Draw coords text
                    const coordsText = document.getElementById('voltage-plot-coords').textContent;
                    ctx.font = '10px monospace';
                    ctx.fillStyle = '#aaa';
                    ctx.fillText(coordsText, 25, h - plotH - 18);
                    // Draw chart canvas
                    ctx.drawImage(plotCanvas, 25, h - plotH - 10, plotW, plotH);
                }
            }

            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92);
        });
    }

    // ==================== Voltage Time-Series Plot ====================

    setupVoltagePlot() {
        const ctx = document.getElementById('voltage-plot-canvas').getContext('2d');
        this.voltagePlotChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Vm',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231,76,60,0.08)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.3,
                    },
                    {
                        label: 'Now',
                        data: [],
                        borderColor: '#f1c40f',
                        backgroundColor: '#f1c40f',
                        borderWidth: 0,
                        pointRadius: 5,
                        pointHoverRadius: 5,
                        showLine: false,
                    }
                ]
            },
            options: {
                animation: false,
                responsive: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: (items) => `${items[0].parsed.x.toFixed(3)} ms`,
                            label: (item) => {
                                if (item.datasetIndex === 0) return `Vm: ${item.parsed.y.toFixed(2)} mV`;
                                return null;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Time (ms)', color: '#aaa', font: { size: 10 } },
                        ticks: { color: '#aaa', font: { size: 9 }, maxTicksLimit: 6 },
                        grid: { color: '#2a2a3a' },
                    },
                    y: {
                        title: { display: true, text: 'Vm (mV)', color: '#aaa', font: { size: 10 } },
                        ticks: { color: '#aaa', font: { size: 9 }, maxTicksLimit: 5 },
                        grid: { color: '#2a2a3a' },
                    }
                }
            }
        });

        document.getElementById('voltage-plot-close').addEventListener('click', () => {
            this.hideVoltagePlot();
        });
    }

    async showVoltagePlot(vertexIdx, worldPos) {
        if (!this.resultsVizDir || !this.voltagePlotChart) return;

        const times = this.resultsTimeSteps;

        // Load all timestep voltages for this vertex from binary cache
        const series = [];
        for (let i = 0; i < times.length; i++) {
            if (!this._voltageCache[i]) {
                const url = `/api/results/binary/${encodeURIComponent(this.resultsVizDir)}/${i}.bin`;
                const resp = await fetch(url);
                if (resp.ok) {
                    this._voltageCache[i] = new Float32Array(await resp.arrayBuffer());
                }
            }
            const voltages = this._voltageCache[i];
            series.push(voltages ? voltages[vertexIdx] : null);
        }

        this.voltagePlotChart.data.labels = times;
        this.voltagePlotChart.data.datasets[0].data = series;

        const currentIdx = parseInt(document.getElementById('result-time').value);
        const markerData = new Array(times.length).fill(null);
        if (currentIdx >= 0 && currentIdx < times.length) {
            markerData[currentIdx] = series[currentIdx];
        }
        this.voltagePlotChart.data.datasets[1].data = markerData;
        this.voltagePlotChart.update('none');

        // Store loaded series for time marker updates
        this._pickedVertexSeries = series;

        const x = worldPos.x.toFixed(1), y = worldPos.y.toFixed(1), z = worldPos.z.toFixed(1);
        document.getElementById('voltage-plot-coords').textContent = `(${x}, ${y}, ${z}) μm`;

        document.getElementById('voltage-plot-panel').style.display = 'block';
    }

    updateVoltagePlotTimeMarker(timeIndex) {
        if (!this.voltagePlotChart || this.pickedVertexIndex === null || !this._pickedVertexSeries) return;

        const times = this.resultsTimeSteps;
        const series = this._pickedVertexSeries;
        const markerData = new Array(times.length).fill(null);
        if (timeIndex >= 0 && timeIndex < times.length) {
            markerData[timeIndex] = series[timeIndex];
        }
        this.voltagePlotChart.data.datasets[1].data = markerData;
        this.voltagePlotChart.update('none');
    }

    hideVoltagePlot() {
        document.getElementById('voltage-plot-panel').style.display = 'none';
        this.pickedVertexIndex = null;
        this._pickedVertexSeries = null;
        if (this.viewer) this.viewer.clearPickMarker();
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
                        display: true,
                        labels: { color: '#ccc', font: { size: 10 }, filter: (item) => item.text !== 'Current' }
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
        this.updateCompareSelector();
    }

    hideIterationsChart() {
        const container = document.getElementById('iterations-chart-container');
        container.style.display = 'none';
    }

    clearIterationsChart() {
        this.iterationsData = [];
        this.compareDatasets = [];
        if (this.iterationsChart) {
            this.iterationsChart.data.labels = [];
            // Keep only the two base datasets (solver iterations + current marker)
            this.iterationsChart.data.datasets = this.iterationsChart.data.datasets.slice(0, 2);
            this.iterationsChart.data.datasets[0].data = [];
            this.iterationsChart.data.datasets[1].data = [];
            this.iterationsChart.update('none');
        }
    }

    initIterationsChartAxis(totalSteps) {
        // Pre-populate x-axis with full time range, keep comparison datasets
        if (this.iterationsChart) {
            const maxLen = Math.max(totalSteps, ...this.compareDatasets.map(d => d.data.length));
            this.iterationsChart.data.labels = Array.from({ length: maxLen }, (_, i) => i);
            this.iterationsChart.data.datasets = [
                { ...this.iterationsChart.data.datasets[0], data: new Array(totalSteps).fill(null) },
                { ...this.iterationsChart.data.datasets[1], data: [] },
                ...this.compareDatasets,
            ];
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
            const maxLen = Math.max(iterations.length, ...this.compareDatasets.map(d => d.data.length));
            this.iterationsChart.data.labels = Array.from({ length: maxLen }, (_, i) => i);
            // Rebuild datasets: base + current marker + comparisons
            this.iterationsChart.data.datasets = [
                { ...this.iterationsChart.data.datasets[0], data: iterations },
                { ...this.iterationsChart.data.datasets[1], data: [] },
                ...this.compareDatasets,
            ];
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

    // ---- Iteration Comparison (overlaid on main chart) ----

    async updateCompareSelector() {
        const container = document.getElementById('compare-sim-selector');
        if (!container) return;

        let sims = [];
        try {
            const resp = await fetch('/api/simulations/with-iterations');
            const data = await resp.json();
            sims = data.simulations || [];
        } catch (e) {
            console.warn('Failed to fetch simulations with iterations:', e);
        }

        // Remember currently checked items
        const checked = new Set();
        container.querySelectorAll('input:checked').forEach(cb => checked.add(cb.value));

        container.innerHTML = '';

        if (sims.length === 0) return;

        // Show chart container if it's hidden (so user can compare even without active run)
        const chartContainer = document.getElementById('iterations-chart-container');
        if (chartContainer.style.display === 'none') {
            chartContainer.style.display = 'block';
        }

        // Group by mesh
        const groups = {};
        for (const sim of sims) {
            const match = sim.name.match(/^(.+?)_sim/);
            const mesh = match ? match[1] : 'other';
            if (!groups[mesh]) groups[mesh] = [];
            groups[mesh].push(sim);
        }

        for (const [mesh, meshSims] of Object.entries(groups).sort((a, b) => a[0].localeCompare(b[0]))) {
            const group = document.createElement('details');
            group.style.cssText = 'margin-bottom: 4px;';
            const summary = document.createElement('summary');
            summary.style.cssText = 'cursor:pointer; font-size:0.8em; font-weight:bold; color:#888; padding:2px 0;';
            summary.textContent = `${mesh} (${meshSims.length})`;
            group.appendChild(summary);

            for (const sim of meshSims) {
                const label = document.createElement('label');
                label.style.cssText = 'display:block; margin:2px 0 2px 16px; font-size:0.75em; cursor:pointer;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.value = sim.name;
                cb.style.marginRight = '6px';
                if (checked.has(sim.name)) cb.checked = true;
                cb.addEventListener('change', () => this.onCompareSelectionChange());
                label.appendChild(cb);

                let displayLabel = '';
                if (sim.solver || sim.preconditioner) {
                    const parts = [sim.preconditioner || sim.solver];
                    if (sim.localSolver) parts[0] += `(${sim.localSolver})`;
                    if (sim.nRanks) parts.push(`${sim.nRanks}r`);
                    displayLabel = parts.join(' ');
                } else {
                    displayLabel = sim.name.replace(/^.+?_sim_?/, '') || 'default';
                }
                label.appendChild(document.createTextNode(displayLabel));
                group.appendChild(label);
            }

            container.appendChild(group);
        }
    }

    _getCompareLabel(simName) {
        const cb = document.querySelector(`#compare-sim-selector input[value="${simName}"]`);
        if (cb && cb.parentElement) {
            const text = cb.parentElement.textContent.trim();
            if (text) return text;
        }
        return simName;
    }

    async onCompareSelectionChange() {
        const checkboxes = document.querySelectorAll('#compare-sim-selector input:checked');
        const selected = Array.from(checkboxes).map(cb => cb.value);

        // Remove old comparison datasets, keep first 2 (solver iterations + current marker)
        if (this.iterationsChart) {
            this.iterationsChart.data.datasets = this.iterationsChart.data.datasets.slice(0, 2);
        }
        this.compareDatasets = [];

        if (selected.length === 0) {
            document.getElementById('compare-conditions-warning').style.display = 'none';
            if (this.iterationsChart) this.iterationsChart.update('none');
            return;
        }

        // Use distinct colors that differ from the primary red (#e94560)
        const colors = ['#4a9de9', '#4ade80', '#e9c74a', '#c74ae9', '#e9844a', '#4ae9c7', '#e94ac7', '#9de94a'];
        const conditionsHashes = new Map();

        for (let i = 0; i < selected.length; i++) {
            try {
                const data = await this.karolinaRunner.fetchIterations(selected[i]);
                const iters = data.iterations || [];

                const ds = {
                    label: this._getCompareLabel(selected[i]),
                    data: iters,
                    borderColor: colors[i % colors.length],
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                };
                this.compareDatasets.push(ds);

                if (data.conditions) {
                    // Hash only physics conditions (mesh, IC, scar) — not solver config
                    const phys = {
                        mesh: data.conditions.mesh,
                        boundingBox: data.conditions.boundingBox,
                        vExcited: data.conditions.vExcited,
                        vResting: data.conditions.vResting,
                        scarEnabled: data.conditions.scarEnabled,
                        scarBox: data.conditions.scarBox,
                        scarMargin: data.conditions.scarMargin,
                        scarConductivities: data.conditions.scarConductivities,
                    };
                    conditionsHashes.set(selected[i], JSON.stringify(phys, Object.keys(phys).sort()));
                }
            } catch (e) {
                console.warn(`Failed to fetch iterations for ${selected[i]}:`, e);
            }
        }

        // Conditions mismatch warning
        const warningEl = document.getElementById('compare-conditions-warning');
        const uniqueHashes = new Set(conditionsHashes.values());
        if (conditionsHashes.size >= 2 && uniqueHashes.size > 1) {
            warningEl.textContent = 'Different conditions';
            warningEl.style.display = 'inline-block';
        } else {
            warningEl.style.display = 'none';
        }

        // Add comparison datasets to the main chart
        if (this.iterationsChart) {
            // Ensure x-axis is long enough for all datasets
            const maxCompareLen = Math.max(0, ...this.compareDatasets.map(d => d.data.length));
            const currentLen = this.iterationsChart.data.labels.length;
            if (maxCompareLen > currentLen) {
                this.iterationsChart.data.labels = Array.from({ length: maxCompareLen }, (_, i) => i);
            }
            for (const ds of this.compareDatasets) {
                this.iterationsChart.data.datasets.push(ds);
            }
            this.iterationsChart.update('none');
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});
