// app-ui.js - Sliders, controls, buttons, checkboxes, colormap

App.prototype.setupSliders = function() {
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
};

App.prototype.setupSimulationParams = function() {
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
};

App.prototype.setupMpiRanks = function() {
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
};

App.prototype.setupVoltageControls = function() {
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
};

App.prototype.setupButtons = function() {
    document.getElementById('run-simulation').addEventListener('click', () => {
        this.runSimulation();
    });

    document.getElementById('cancel-simulation').addEventListener('click', () => {
        this.cancelSimulation();
    });

    document.getElementById('reset-camera').addEventListener('click', () => {
        this.viewer.resetCamera();
    });
};

App.prototype.setupCheckboxes = function() {
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
        if (e.target.checked && this.ecsRanksData) {
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
        document.getElementById('interface-type-controls').style.display = this.showInterfaces ? 'block' : 'none';
        this.updateInterfaceHighlight();
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
};

App.prototype.setupColormapSelector = function() {
    const selector = document.getElementById('colormap-selector');

    selector.addEventListener('change', () => {
        const colormap = selector.value;
        this.viewer.setColormap(colormap);
        this.updateColorbarGradient();

        const showPartition = document.getElementById('show-partition').checked;
        if (!showPartition) {
            if (this.resultsVizDir && this._voltageCache) {
                const timeSlider = document.getElementById('result-time');
                const idx = parseInt(timeSlider.value);
                const voltages = this._voltageCache[idx];
                if (voltages) {
                    this.viewer.updateVoltageColors(voltages);
                }
            } else {
                this.viewer.updateBoundingBox(this.boundingBox);
            }
        }
    });

    this.updateColorbarGradient();
};

App.prototype.updateColorbarGradient = function() {
    const gradient = this.viewer.getColormapGradient();
    const gradientEl = document.querySelector('.colorbar-gradient');
    if (gradientEl) {
        gradientEl.style.background = gradient;
    }
};

App.prototype.setupResultsControls = function() {
    const loadBtn = document.getElementById('load-results');
    const timeSlider = document.getElementById('result-time');
    const timeVal = document.getElementById('result-time-val');
    const simSelector = document.getElementById('simulation-selector');

    this.loadSimulationList();

    loadBtn.addEventListener('click', () => this.loadResults());

    const dlBtn = document.getElementById('download-karolina-results');
    dlBtn.addEventListener('click', () => this.downloadKarolinaSimulation());

    timeSlider.addEventListener('input', () => {
        if (this.resultsVizDir && this.resultsTimeSteps && this.resultsTimeSteps.length > 0) {
            const idx = parseInt(timeSlider.value);
            const time = this.resultsTimeSteps[idx];
            timeVal.textContent = time.toFixed(3);
            this.showResultsAtTime(idx);
            this.highlightIterationStep(idx, this.resultsTimeSteps.length);
        }
    });

    simSelector.addEventListener('change', () => {
        this.selectedSimulation = simSelector.value;
    });
};

App.prototype.onBoundingBoxChange = function() {
    this.updateVinitExpression();
    this.updateBoundingBoxVisualization();
    this.saveMeshConfig();
};

App.prototype.updateVinitExpression = function() {
    // Expression is generated on demand when running simulation
};

App.prototype.generateVinitExpression = function() {
    const cf = this.conversionFactor;

    const fmt = (v) => {
        const scaled = v * cf;
        return scaled.toPrecision(6).replace(/\.?0+$/, '');
    };

    const xMin = fmt(this.boundingBox.xMin);
    const xMax = fmt(this.boundingBox.xMax);
    const yMin = fmt(this.boundingBox.yMin);
    const yMax = fmt(this.boundingBox.yMax);
    const zMin = fmt(this.boundingBox.zMin);
    const zMax = fmt(this.boundingBox.zMax);

    const inside = `((x[0] >= ${xMin}) * (x[0] <= ${xMax}) * (x[1] >= ${yMin}) * (x[1] <= ${yMax}) * (x[2] >= ${zMin}) * (x[2] <= ${zMax}))`;

    const vDiff = this.vExcited - this.vResting;
    return `"(${this.vResting}.0) + (${vDiff}.0) * ${inside}"`;
};

App.prototype.updateColorbar = function() {
    document.getElementById('colorbar-max').textContent = `${this.vExcited} mV`;
    document.getElementById('colorbar-min').textContent = `${this.vResting} mV`;
    document.getElementById('colorbar-mid').textContent = `${Math.round((this.vExcited + this.vResting) / 2)} mV`;

    if (this.viewer) {
        this.viewer.setVoltageRange(this.vResting, this.vExcited);
        this.updateColorbarGradient();
    }
};

App.prototype.updateBoundingBoxVisualization = function() {
    if (this.viewer) {
        this.viewer.updateBoundingBox(this.boundingBox);
    }
};
