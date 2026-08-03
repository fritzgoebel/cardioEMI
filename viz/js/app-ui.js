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
        this.refreshInterfaceView();
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

    // Per-interface inspector: enable/disable all rows at once
    document.getElementById('interface-list-all').addEventListener('click', () => {
        this.setAllInterfaces(true);
    });
    document.getElementById('interface-list-none').addEventListener('click', () => {
        this.setAllInterfaces(false);
    });

    // Rank selection buttons
    document.getElementById('select-all-ranks').addEventListener('click', () => {
        this.selectAllRanks(true);
    });

    document.getElementById('select-no-ranks').addEventListener('click', () => {
        this.selectAllRanks(false);
    });

    // Cross-section mode toggle
    document.getElementById('cross-section-mode').addEventListener('change', async (e) => {
        document.getElementById('cross-section-controls').style.display = e.target.checked ? 'block' : 'none';
        document.getElementById('colorbar').style.display = e.target.checked ? 'none' : '';
        document.getElementById('colorbar-intra').style.display = e.target.checked ? 'block' : 'none';
        document.getElementById('colorbar-extra').style.display = e.target.checked ? 'block' : 'none';
        try {
            await this.setCrossSectionMode(e.target.checked);
        } catch (err) {
            console.error(err);
        }
        this._updatePlanePreview();
    });

    // Plane axis preset toggles the custom normal inputs.
    document.getElementById('cs-plane-axis').addEventListener('change', (e) => {
        document.getElementById('cs-custom-normal').style.display =
            e.target.value === 'custom' ? 'inline-block' : 'none';
        // Reasonable default offset = midpoint of the chosen axis bounds.
        this._updateCsOffsetRange();
        this._updatePlanePreview();
    });

    document.getElementById('cs-offset-slider').addEventListener('input', (e) => {
        const t = parseFloat(e.target.value);
        const range = this._csOffsetRange || { min: 0, max: 1 };
        const val = range.min + t * (range.max - range.min);
        document.getElementById('cs-offset').value = val.toFixed(3);
        this._updatePlanePreview();
    });
    document.getElementById('cs-offset').addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        const range = this._csOffsetRange || { min: 0, max: 1 };
        const span = (range.max - range.min) || 1;
        const t = Math.min(1, Math.max(0, (val - range.min) / span));
        document.getElementById('cs-offset-slider').value = String(t);
        this._updatePlanePreview();
    });
    ['cs-nx', 'cs-ny', 'cs-nz'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => this._updatePlanePreview());
    });

    document.getElementById('cs-apply').addEventListener('click', async () => {
        const statusEl = document.getElementById('cs-status');
        const { normal, offset } = this._currentCsPlane();
        statusEl.textContent = 'Slicing ECS volume…';
        try {
            const result = await this.applyCrossSectionPlane(normal, offset);
            const warn = result.warning ? ` — WARNING: ${result.warning}` : '';
            statusEl.textContent = `Cap: ${result.cap_facet_count} triangles, ${result.num_timesteps} steps${warn}`;
            this.markPlaneApplied(normal, offset);
        } catch (err) {
            statusEl.textContent = `Failed: ${err.message}`;
        }
    });

    // Intra/extra colormap & range
    const updateIntraGradient = () => {
        const grad = this.viewer.getColormapGradientFor(this.viewer.colormapIntra);
        const el = document.querySelector('#colorbar-intra .colorbar-gradient');
        if (el) el.style.background = grad;
    };
    const updateExtraGradient = () => {
        const grad = this.viewer.getColormapGradientFor(this.viewer.colormapExtra);
        const el = document.querySelector('#colorbar-extra .colorbar-gradient');
        if (el) el.style.background = grad;
    };
    const updateIntraLabels = () => {
        const mn = this.viewer.vMinIntra, mx = this.viewer.vMaxIntra;
        document.getElementById('colorbar-intra-min').textContent = mn.toFixed(0);
        document.getElementById('colorbar-intra-mid').textContent = ((mn + mx) / 2).toFixed(0);
        document.getElementById('colorbar-intra-max').textContent = mx.toFixed(0);
    };
    const updateExtraLabels = () => {
        const mn = this.viewer.vMinExtra, mx = this.viewer.vMaxExtra;
        document.getElementById('colorbar-extra-min').textContent = mn.toFixed(2);
        document.getElementById('colorbar-extra-mid').textContent = ((mn + mx) / 2).toFixed(2);
        document.getElementById('colorbar-extra-max').textContent = mx.toFixed(2);
    };
    this._refreshCsLabels = () => { updateIntraLabels(); updateExtraLabels(); };
    this._refreshCsGradients = () => { updateIntraGradient(); updateExtraGradient(); };

    document.getElementById('cs-intra-colormap').addEventListener('change', (e) => {
        this.viewer.setIntraColormap(e.target.value);
        updateIntraGradient();
        this.refreshCrossSectionColors();
    });
    document.getElementById('cs-extra-colormap').addEventListener('change', (e) => {
        this.viewer.setExtraColormap(e.target.value);
        updateExtraGradient();
        this.refreshCrossSectionColors();
    });
    const onIntraRangeChange = () => {
        const mn = parseFloat(document.getElementById('cs-intra-vmin').value);
        const mx = parseFloat(document.getElementById('cs-intra-vmax').value);
        if (Number.isFinite(mn) && Number.isFinite(mx) && mx > mn) {
            this.viewer.setIntraRange(mn, mx);
            updateIntraLabels();
            this.refreshCrossSectionColors();
        }
    };
    const onExtraRangeChange = () => {
        const mn = parseFloat(document.getElementById('cs-extra-vmin').value);
        const mx = parseFloat(document.getElementById('cs-extra-vmax').value);
        if (Number.isFinite(mn) && Number.isFinite(mx) && mx > mn) {
            this.viewer.setExtraRange(mn, mx);
            updateExtraLabels();
            this.refreshCrossSectionColors();
        }
    };
    document.getElementById('cs-intra-vmin').addEventListener('input', onIntraRangeChange);
    document.getElementById('cs-intra-vmax').addEventListener('input', onIntraRangeChange);
    document.getElementById('cs-extra-vmin').addEventListener('input', onExtraRangeChange);
    document.getElementById('cs-extra-vmax').addEventListener('input', onExtraRangeChange);

    const autofitBtn = document.getElementById('cs-autofit');
    if (autofitBtn) {
        autofitBtn.addEventListener('click', () => {
            this.autofitCrossSectionRanges();
        });
    }
};

App.prototype._currentCsPlane = function() {
    const axis = document.getElementById('cs-plane-axis').value;
    let normal;
    if (axis === 'x') normal = [1, 0, 0];
    else if (axis === 'y') normal = [0, 1, 0];
    else if (axis === 'z') normal = [0, 0, 1];
    else {
        normal = [
            parseFloat(document.getElementById('cs-nx').value) || 0,
            parseFloat(document.getElementById('cs-ny').value) || 0,
            parseFloat(document.getElementById('cs-nz').value) || 0,
        ];
    }
    const offset = parseFloat(document.getElementById('cs-offset').value);
    return { normal, offset };
};

App.prototype._planeMatchesApplied = function(normal, offset) {
    const a = this._appliedPlane;
    if (!a) return false;
    if (!Number.isFinite(offset) || Math.abs(offset - a.offset) > 1e-6) return false;
    for (let i = 0; i < 3; i++) {
        if (Math.abs((normal[i] || 0) - a.normal[i]) > 1e-6) return false;
    }
    return true;
};

App.prototype.markPlaneApplied = function(normal, offset) {
    this._appliedPlane = { normal: normal.slice(), offset };
    this._updatePlanePreview();
};

App.prototype._updatePlanePreview = function() {
    if (!this.viewer || !this.meshBounds) return;
    const { normal, offset } = this._currentCsPlane();
    if (!Number.isFinite(offset)) return;
    const sx = (this.meshBounds.x[1] - this.meshBounds.x[0]);
    const sy = (this.meshBounds.y[1] - this.meshBounds.y[0]);
    const sz = (this.meshBounds.z[1] - this.meshBounds.z[0]);
    const size = Math.max(sx, sy, sz);
    this.viewer.setPlanePreviewSize(size);
    // Show the preview only when (a) we're in cross-section mode and
    // (b) the current config differs from the last-applied plane.
    const dirty = !this._planeMatchesApplied(normal, offset);
    this.viewer.setPlanePreview(normal, offset, this.crossSectionMode && dirty);
};

App.prototype._updateCsOffsetRange = function() {
    if (!this.meshBounds) return;
    const axis = document.getElementById('cs-plane-axis').value;
    let mn, mx;
    if (axis === 'x' || axis === 'custom') { mn = this.meshBounds.x[0]; mx = this.meshBounds.x[1]; }
    if (axis === 'y') { mn = this.meshBounds.y[0]; mx = this.meshBounds.y[1]; }
    if (axis === 'z') { mn = this.meshBounds.z[0]; mx = this.meshBounds.z[1]; }
    if (axis === 'custom') {
        const allMin = Math.min(this.meshBounds.x[0], this.meshBounds.y[0], this.meshBounds.z[0]);
        const allMax = Math.max(this.meshBounds.x[1], this.meshBounds.y[1], this.meshBounds.z[1]);
        mn = allMin; mx = allMax;
    }
    this._csOffsetRange = { min: mn, max: mx };
    const offsetInput = document.getElementById('cs-offset');
    const slider = document.getElementById('cs-offset-slider');
    if (!offsetInput.value || parseFloat(offsetInput.value) < mn || parseFloat(offsetInput.value) > mx) {
        const mid = (mn + mx) / 2;
        offsetInput.value = mid.toFixed(3);
        slider.value = '0.5';
    } else {
        const span = (mx - mn) || 1;
        slider.value = String(Math.min(1, Math.max(0, (parseFloat(offsetInput.value) - mn) / span)));
    }
};

App.prototype.showCrossSectionOption = function(available) {
    // New sim or new mesh → drop the "last applied plane" memory so the
    // preview shows again for the first plane configuration.
    this._appliedPlane = null;
    document.getElementById('cross-section-label').style.display = available ? '' : 'none';
    if (!available) {
        document.getElementById('cross-section-mode').checked = false;
        document.getElementById('cross-section-controls').style.display = 'none';
        document.getElementById('colorbar').style.display = '';
        document.getElementById('colorbar-intra').style.display = 'none';
        document.getElementById('colorbar-extra').style.display = 'none';
        this.crossSectionMode = false;
        if (this.viewer) {
            this.viewer.setCrossSectionVisible(false);
            this.viewer.setPlanePreview([0, 0, 1], 0, false);
        }
    } else {
        this._updateCsOffsetRange();
        if (this._refreshCsGradients) this._refreshCsGradients();
        if (this._refreshCsLabels) this._refreshCsLabels();
        this._updatePlanePreview();
    }
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

    const deleteAllBtn = document.getElementById('delete-all-results');
    if (deleteAllBtn) {
        deleteAllBtn.addEventListener('click', () => this.deleteAllResults());
    }

    timeSlider.addEventListener('input', () => {
        if (this.resultsVizDir && this.resultsTimeSteps && this.resultsTimeSteps.length > 0) {
            const idx = parseInt(timeSlider.value);
            const time = this.resultsTimeSteps[idx];
            timeVal.textContent = time.toFixed(3);
            this.showResultsAtTime(idx);
            this.highlightIterationStep(idx, this.resultsTimeSteps.length);
            this.setResidualHistoryForStep(idx);
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
