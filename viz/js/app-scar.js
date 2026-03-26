// app-scar.js - Scar tissue controls and per-mesh config persistence

App.prototype.setupScarControls = function() {
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

    // Scar bounding box sliders
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
};

App.prototype.onScarChange = function() {
    this.saveMeshConfig();
    if (this.viewer) {
        this.viewer.updateScarBox(this.scarBox, this.scarMargin, this.scarEnabled);
        this.viewer.setScarZones(this.scarBox, this.scarMargin, this.scarEnabled);

        if (this.resultsVizDir) {
            const idx = parseInt(document.getElementById('result-time').value);
            this.showResultsAtTime(idx);
        } else if (this.scarEnabled) {
            this.viewer.highlightScarZones(this.scarBox, this.scarMargin);
        }
    }
};

App.prototype._updateScarSlidersFromConfig = function(region) {
    const box = region.box;
    const margin = region.margin || 10;

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
};

App.prototype.saveMeshConfig = function() {
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
};

App.prototype.loadMeshConfig = function() {
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
};

App.prototype.getScarConfig = function() {
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
};
