// app-charts.js - Iterations, residual, voltage charts + iteration comparison

// ==================== Iterations Chart ====================

App.prototype.setupIterationsChart = function() {
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
};

App.prototype.showIterationsChart = function() {
    const container = document.getElementById('iterations-chart-container');
    container.style.display = 'block';
    this.updateCompareSelector();
};

App.prototype.hideIterationsChart = function() {
    const container = document.getElementById('iterations-chart-container');
    container.style.display = 'none';
};

App.prototype.clearIterationsChart = function() {
    this.iterationsData = [];
    this.compareDatasets = [];
    if (this.iterationsChart) {
        this.iterationsChart.data.labels = [];
        this.iterationsChart.data.datasets = this.iterationsChart.data.datasets.slice(0, 2);
        this.iterationsChart.data.datasets[0].data = [];
        this.iterationsChart.data.datasets[1].data = [];
        this.iterationsChart.update('none');
    }
};

App.prototype.initIterationsChartAxis = function(totalSteps) {
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
};

App.prototype.addIterationPoint = function(step, count) {
    while (this.iterationsData.length <= step) {
        this.iterationsData.push(null);
    }
    this.iterationsData[step] = { step, count };

    if (this.iterationsChart && step < this.iterationsChart.data.datasets[0].data.length) {
        this.iterationsChart.data.datasets[0].data[step] = count;
        this.iterationsChart.update('none');
    }
};

App.prototype.setIterationsData = function(iterations) {
    this.iterationsData = iterations.map((count, i) => ({ step: i, count }));
    if (this.iterationsChart) {
        const maxLen = Math.max(iterations.length, ...this.compareDatasets.map(d => d.data.length));
        this.iterationsChart.data.labels = Array.from({ length: maxLen }, (_, i) => i);
        this.iterationsChart.data.datasets = [
            { ...this.iterationsChart.data.datasets[0], data: iterations },
            { ...this.iterationsChart.data.datasets[1], data: [] },
            ...this.compareDatasets,
        ];
        this.iterationsChart.update('none');
    }
};

App.prototype.highlightIterationStep = function(timeIndex, totalResultSteps) {
    if (!this.iterationsChart || this.iterationsData.length === 0) return;

    const iterationIndex = Math.round(timeIndex * (this.iterationsData.length - 1) / (totalResultSteps - 1));

    const markerData = new Array(this.iterationsData.length).fill(null);
    if (iterationIndex >= 0 && iterationIndex < this.iterationsData.length) {
        markerData[iterationIndex] = this.iterationsData[iterationIndex].count;
    }

    this.iterationsChart.data.datasets[1].data = markerData;
    this.iterationsChart.update('none');
};

// ==================== Iteration Comparison ====================

App.prototype.updateCompareSelector = async function() {
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

    // Show chart container if it's hidden
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
};

App.prototype._getCompareLabel = function(simName) {
    const cb = document.querySelector(`#compare-sim-selector input[value="${simName}"]`);
    if (cb && cb.parentElement) {
        const text = cb.parentElement.textContent.trim();
        if (text) return text;
    }
    return simName;
};

App.prototype.onCompareSelectionChange = async function() {
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
                // Hash only physics conditions (mesh, IC, scar) -- not solver config
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
};

// ==================== Residual Chart ====================

App.prototype.setupResidualChart = function() {
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
};

App.prototype.showResidualChart = function() {
    document.getElementById('residual-chart-container').style.display = 'block';
};

App.prototype.hideResidualChart = function() {
    document.getElementById('residual-chart-container').style.display = 'none';
};

App.prototype.clearResidualChart = function() {
    this.residualAbsData = [];
    this.residualRelData = [];
    if (this.residualChart) {
        this.residualChart.data.labels = [];
        this.residualChart.data.datasets[0].data = [];
        this.residualChart.data.datasets[1].data = [];
        this.residualChart.update('none');
    }
};

App.prototype.initResidualChartAxis = function(totalSteps) {
    if (this.residualChart) {
        this.residualChart.data.labels = Array.from({ length: totalSteps }, (_, i) => i);
        this.residualChart.data.datasets[0].data = new Array(totalSteps).fill(null);
        this.residualChart.data.datasets[1].data = new Array(totalSteps).fill(null);
        this.residualChart.update('none');
    }
};

App.prototype.addResidualPoint = function(step, absRes, relRes) {
    while (this.residualAbsData.length <= step) {
        this.residualAbsData.push(null);
        this.residualRelData.push(null);
    }
    this.residualAbsData[step] = absRes;
    this.residualRelData[step] = relRes;

    if (this.residualChart && step < this.residualChart.data.datasets[0].data.length) {
        this.residualChart.data.datasets[0].data[step] = absRes;
        this.residualChart.data.datasets[1].data[step] = relRes;
        this.residualChart.update('none');
    }
};

App.prototype.setResidualData = function(absData, relData) {
    this.residualAbsData = absData;
    this.residualRelData = relData;
    if (this.residualChart) {
        this.residualChart.data.labels = absData.map((_, i) => i);
        this.residualChart.data.datasets[0].data = absData;
        this.residualChart.data.datasets[1].data = relData;
        this.residualChart.update('none');
    }
};

// ==================== Voltage Time-Series Plot ====================

App.prototype.setupVoltagePlot = function() {
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
};

App.prototype.showVoltagePlot = async function(vertexIdx, worldPos) {
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

    this._pickedVertexSeries = series;

    const x = worldPos.x.toFixed(1), y = worldPos.y.toFixed(1), z = worldPos.z.toFixed(1);
    document.getElementById('voltage-plot-coords').textContent = `(${x}, ${y}, ${z}) μm`;

    document.getElementById('voltage-plot-panel').style.display = 'block';
};

App.prototype.updateVoltagePlotTimeMarker = function(timeIndex) {
    if (!this.voltagePlotChart || this.pickedVertexIndex === null || !this._pickedVertexSeries) return;

    const times = this.resultsTimeSteps;
    const series = this._pickedVertexSeries;
    const markerData = new Array(times.length).fill(null);
    if (timeIndex >= 0 && timeIndex < times.length) {
        markerData[timeIndex] = series[timeIndex];
    }
    this.voltagePlotChart.data.datasets[1].data = markerData;
    this.voltagePlotChart.update('none');
};

App.prototype.hideVoltagePlot = function() {
    document.getElementById('voltage-plot-panel').style.display = 'none';
    this.pickedVertexIndex = null;
    this._pickedVertexSeries = null;
    if (this.viewer) this.viewer.clearPickMarker();
};
