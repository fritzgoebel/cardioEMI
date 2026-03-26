// app.js - Main application class definition and entry point
// Methods are split across app-*.js files, attached to App.prototype

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
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});
