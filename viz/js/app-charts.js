// app-charts.js - Iterations, residual, voltage charts + iteration comparison

// ==================== Shared chart styling / helpers ====================

// Typography shared by the comparison plots (kept large enough to stay readable
// when a plot is exported as a figure).
const CHART_TYPO = {
    legend: { size: 16 },
    axisTitle: { size: 16, weight: '600' },
    tick: { size: 14 },
    title: { size: 15 },
};

// A dataset only earns a legend entry once it actually carries a value.
function chartDatasetHasData(ds) {
    if (!ds || !Array.isArray(ds.data)) return false;
    return ds.data.some(v => {
        if (v === null || v === undefined) return false;
        if (typeof v === 'number') return isFinite(v);
        return true;
    });
}

// The two datasets the residual-history chart shows for a single run
// (comparison mode swaps in one dataset per selected run instead).
function residualHistorySingleRunDatasets() {
    return [
        { label: 'true ||b - A·x||', data: [], borderColor: '#e94560',
          borderWidth: 2, fill: false, pointRadius: 0, pointHoverRadius: 3, tension: 0 },
        { label: 'implicit', data: [], borderColor: '#4ade80', borderDash: [4, 3],
          borderWidth: 2, fill: false, pointRadius: 0, pointHoverRadius: 3, tension: 0 },
    ];
}

// Legend filter: drop empty or toggled-off datasets, plus any label listed in `hidden`.
function chartLegendFilter(hidden) {
    const hide = new Set(hidden || []);
    return (item, data) => {
        if (hide.has(item.text)) return false;
        const ds = data.datasets[item.datasetIndex];
        if (ds && ds.hidden) return false;
        return chartDatasetHasData(ds);
    };
}

// ==================== Draggable / editable chart legends ====================
// Chart.js draws its legend on the canvas, which can neither be dragged out of
// the way nor edited in place. Every compare chart gets a floating HTML legend
// instead (the canvas legend is turned off via plugins.legend.display:false).
// Position is remembered per chart (localStorage, session-independent like the
// chart-panel sizes). Renaming a legend entry - where it maps to an actual run
// or folder/mesh group - reuses the same persisted-name plumbing as the
// pencil icons in the Runs browser, so it's a global rename, not a local relabel.

const CHART_LEGEND_POS_KEY = 'chartLegendPositions';

function loadChartLegendPositions() {
    try {
        return JSON.parse(localStorage.getItem(CHART_LEGEND_POS_KEY) || '{}') || {};
    } catch (e) {
        return {};
    }
}

function saveChartLegendPosition(name, pos) {
    const all = loadChartLegendPositions();
    all[name] = pos;
    try {
        localStorage.setItem(CHART_LEGEND_POS_KEY, JSON.stringify(all));
    } catch (e) { /* non-fatal */ }
}

// Creates the floating legend box inside `box` (a .chart-resize-box, already
// position:relative) and wires dragging. Returns the <div> to render items into.
App.prototype._createChartLegend = function(name, box) {
    const el = document.createElement('div');
    el.className = 'chart-legend-overlay';
    const list = document.createElement('div');
    list.className = 'chart-legend-list';
    el.appendChild(list);
    box.appendChild(el);

    const saved = loadChartLegendPositions()[name];
    if (saved && saved.left != null && saved.top != null) {
        el.style.left = saved.left;
        el.style.top = saved.top;
        el.style.right = 'auto';
    }

    let drag = null;
    el.addEventListener('mousedown', (e) => {
        if (e.target.closest('.chart-legend-item')) return; // items handle their own clicks
        const boxRect = box.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        // Switch from the default CSS right/top anchor to an explicit left/top
        // anchor at the box's current on-screen position, so it doesn't jump.
        el.style.left = `${elRect.left - boxRect.left}px`;
        el.style.top = `${elRect.top - boxRect.top}px`;
        el.style.right = 'auto';
        drag = {
            offsetX: e.clientX - elRect.left,
            offsetY: e.clientY - elRect.top,
            boxRect,
        };
        e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
        if (!drag) return;
        let left = e.clientX - drag.boxRect.left - drag.offsetX;
        let top = e.clientY - drag.boxRect.top - drag.offsetY;
        left = Math.max(0, Math.min(left, drag.boxRect.width - el.offsetWidth));
        top = Math.max(0, Math.min(top, drag.boxRect.height - el.offsetHeight));
        el.style.left = `${left}px`;
        el.style.top = `${top}px`;
    });
    window.addEventListener('mouseup', () => {
        if (!drag) return;
        drag = null;
        saveChartLegendPosition(name, { left: el.style.left, top: el.style.top });
    });

    return list;
};

// Renders one row per legend item: a colour swatch + label that toggles that
// dataset's visibility on click, plus a ✎ button when the item is renameable.
App.prototype._renderLegendItems = function(listEl, items) {
    listEl.innerHTML = '';
    for (const item of items) {
        const row = document.createElement('div');
        row.className = 'chart-legend-item';
        if (item.hidden) row.classList.add('chart-legend-item-hidden');

        const swatch = document.createElement('span');
        swatch.className = 'chart-legend-swatch';
        swatch.style.background = item.color || '#888';
        row.appendChild(swatch);

        const label = document.createElement('span');
        label.className = 'chart-legend-label';
        label.textContent = item.text;
        row.appendChild(label);

        row.addEventListener('click', () => item.onToggle && item.onToggle());

        if (item.onRename) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'chart-legend-rename';
            btn.textContent = '✎';
            btn.title = 'Rename (applies everywhere this run/folder is shown)';
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                item.onRename();
            });
            row.appendChild(btn);
        }

        listEl.appendChild(row);
    }
};

// Shared prompt+persist flow for a legend rename button - same shape as the
// pencil-icon options used in the compare tree (_addRenameButton).
App.prototype._promptChartRename = async function(opts) {
    const next = window.prompt(opts.prompt, opts.current || '');
    if (next === null) return;
    await opts.onSave(next.trim());
};

// Rebuilds a chart's floating legend from its current datasets.
// getRename(ds, index) -> {prompt, current, onSave} or null/undefined if that
// entry doesn't map to a renameable run/folder.
// opts.toggle:false disables click-to-toggle (residual charts already have
// their own abs/rel checkboxes driving visibility; letting the legend also
// call setDatasetVisibility there would fight that mechanism for control).
// opts.requireData:false keeps structurally-fixed entries (e.g. "Absolute
// Residual") in the legend even before any data has arrived.
App.prototype._syncChartLegend = function(chart, listEl, opts) {
    if (!chart || !listEl) return;
    opts = opts || {};
    const hidden = opts.hiddenLabels || new Set();
    const toggle = opts.toggle !== false;
    const requireData = opts.requireData !== false;
    const items = [];
    chart.data.datasets.forEach((ds, index) => {
        if (!ds || hidden.has(ds.label)) return;
        if (requireData && !chartDatasetHasData(ds)) return;
        const renameOpts = opts.getRename ? opts.getRename(ds, index) : null;
        items.push({
            text: ds.label,
            color: ds.borderColor || ds.backgroundColor || '#888',
            hidden: !chart.isDatasetVisible(index),
            onToggle: toggle ? () => {
                chart.setDatasetVisibility(index, !chart.isDatasetVisible(index));
                chart.update();
                this._syncChartLegend(chart, listEl, opts);
            } : null,
            onRename: renameOpts ? () => this._promptChartRename(renameOpts) : null,
        });
    });
    this._renderLegendItems(listEl, items);
};

// ==================== Iterations Chart ====================

App.prototype.setupIterationsChart = function() {
    const ctx = document.getElementById('iterations-chart').getContext('2d');
    const self = this;

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
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#16213e',
                    titleColor: '#fff',
                    bodyColor: '#ccc',
                    borderColor: '#e94560',
                    borderWidth: 1,
                    titleFont: CHART_TYPO.legend,
                    bodyFont: CHART_TYPO.tick
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Step',
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: { color: '#888', font: CHART_TYPO.tick },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Iterations',
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: { color: '#888', font: CHART_TYPO.tick },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    beginAtZero: true
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            onHover: (event, elements) => {
                if (elements && elements.length > 0) {
                    self.setCompareResidualHistoryForStep(elements[0].index);
                }
            }
        }
    });

    this.wireChartExport('iterations-chart-export', () => this.iterationsChart, 'iterations');

    const box = document.querySelector('#iterations-chart-container .chart-resize-box');
    this._iterationsLegendList = this._createChartLegend('iterations', box);
    this._refreshIterationsLegend();
};

// Legend entry 0 is the loaded single run, 'Current' is the hover marker
// (never shown in the legend), and the rest are the comparison datasets -
// each carries the sim name it came from (see onCompareSelectionChange).
App.prototype._refreshIterationsLegend = function() {
    this._syncChartLegend(this.iterationsChart, this._iterationsLegendList, {
        hiddenLabels: new Set(['Current']),
        getRename: (ds, index) => {
            const simName = index === 0 ? this.loadedSimName : (ds && ds.name);
            if (!simName) return null;
            return {
                prompt: `Name for this run (empty: default)\n${simName}`,
                current: this._customRunName(simName),
                onSave: (value) => this._saveCompareNames({ runs: { [simName]: { label: value } } }),
            };
        },
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
    this.compareLabels = {};
    this.loadedSimName = null;
    if (this.iterationsChart) {
        this.iterationsChart.data.labels = [];
        this.iterationsChart.data.datasets = this.iterationsChart.data.datasets.slice(0, 2);
        this.iterationsChart.data.datasets[0].label = 'Solver Iterations';
        this.iterationsChart.data.datasets[0].data = [];
        this.iterationsChart.data.datasets[1].data = [];
        this.iterationsChart.update('none');
        this._refreshIterationsLegend();
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

// Name the loaded run in the legend the same way compared runs are named,
// falling back to the generic label until its metadata is known.
App.prototype._applyLoadedRunLabel = function() {
    if (!this.iterationsChart) return;
    const meta = this.loadedSimName && this.compareSimMeta
        ? this.compareSimMeta[this.loadedSimName] : null;
    const label = meta ? this._formatRunLabel(this.loadedSimName, meta, {}) : 'Solver Iterations';
    const ds = this.iterationsChart.data.datasets[0];
    if (ds && ds.label !== label) {
        ds.label = label;
        this.iterationsChart.update('none');
        this._refreshIterationsLegend();
    }
};

App.prototype.setIterationsData = function(iterations, simName) {
    if (simName !== undefined) this.loadedSimName = simName;
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
        this._applyLoadedRunLabel();
        this._refreshIterationsLegend();
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

// ==================== Scaling Chart ====================
// Avg. iterations (over all timesteps) vs. a weak-scaling geometry metric,
// one line per folder (falls back to mesh group for unfiled runs).
// Only meaningful for plus_<nx>x<ny>x<nz>_n<n>_L<L> weak-scaling meshes, whose
// nSubdomains / hRatio the server derives from the mesh name (see
// /api/simulations/with-iterations); runs on other meshes are simply omitted.

App.prototype.setupScalingChart = function() {
    const ctx = document.getElementById('scaling-chart').getContext('2d');

    this.scalingChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            parsing: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#16213e',
                    titleColor: '#fff',
                    bodyColor: '#ccc',
                    borderColor: '#e94560',
                    borderWidth: 1,
                    titleFont: CHART_TYPO.legend,
                    bodyFont: CHART_TYPO.tick,
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(1)} iters @ ${context.parsed.x}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Number of subdomains',
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: { color: '#888', font: CHART_TYPO.tick },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Avg. iterations',
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: { color: '#888', font: CHART_TYPO.tick },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    beginAtZero: true
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: true
            }
        }
    });

    this.wireChartExport('scaling-chart-export', () => this.scalingChart, 'scaling');

    const box = document.querySelector('#scaling-chart-container .chart-resize-box');
    this._scalingLegendList = this._createChartLegend('scaling', box);

    const xModeSelect = document.getElementById('scaling-x-mode');
    this.scalingXMode = (xModeSelect && xModeSelect.value) || 'nSubdomains';
    if (xModeSelect) {
        xModeSelect.addEventListener('change', () => {
            this.scalingXMode = xModeSelect.value;
            this._updateScalingChart();
        });
    }

    const logXCb = document.getElementById('scaling-log-x');
    this.scalingLogX = !!(logXCb && logXCb.checked);
    if (logXCb) {
        logXCb.addEventListener('change', () => {
            this.scalingLogX = logXCb.checked;
            this._updateScalingChart();
        });
    }

    const logYCb = document.getElementById('scaling-log-y');
    this.scalingLogY = !!(logYCb && logYCb.checked);
    if (logYCb) {
        logYCb.addEventListener('change', () => {
            this.scalingLogY = logYCb.checked;
            this._updateScalingChart();
        });
    }
};

// Rebuilds the scaling chart from this._scalingRunData (populated by
// onCompareSelectionChange) using whichever x-axis metric is selected.
App.prototype._updateScalingChart = function() {
    const container = document.getElementById('scaling-chart-container');
    if (!container || !this.scalingChart) return;

    const xField = this.scalingXMode || 'nSubdomains';
    const runs = (this._scalingRunData || [])
        .filter(r => r.avgIterations != null && r[xField] != null);

    if (runs.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    const groups = {};
    for (const r of runs) {
        if (!groups[r.groupKey]) groups[r.groupKey] = { label: r.groupLabel, points: [], names: [] };
        groups[r.groupKey].points.push({ x: r[xField], y: r.avgIterations });
        groups[r.groupKey].names.push(r.name);
    }

    const colors = ['#4a9de9', '#4ade80', '#e9c74a', '#c74ae9', '#e9844a', '#4ae9c7', '#e94ac7', '#9de94a'];
    const groupKeys = Object.keys(groups).sort((a, b) => groups[a].label.localeCompare(groups[b].label));

    this.scalingChart.data.datasets = groupKeys.map((key, i) => {
        const g = groups[key];
        g.points.sort((a, b) => a.x - b.x);
        return {
            label: g.label,
            data: g.points,
            names: g.names,                                          // for legend rename
            groupKind: key.startsWith('folder:') ? 'folder' : 'mesh',
            groupValue: key.slice(key.indexOf(':') + 1),
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length],
            borderWidth: 2,
            fill: false,
            tension: 0,
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: true
        };
    });

    this.scalingChart.options.scales.x.title.text =
        xField === 'hRatio' ? 'H / h' : 'Number of subdomains';
    this.scalingChart.options.scales.x.type = this.scalingLogX ? 'logarithmic' : 'linear';
    this.scalingChart.options.scales.y.type = this.scalingLogY ? 'logarithmic' : 'linear';
    this.scalingChart.update('none');
    this._refreshScalingLegend();
};

// A scaling-chart dataset is a folder (rename moves the folder in the Runs
// browser) or a mesh fallback group (rename just relabels that heading).
App.prototype._refreshScalingLegend = function() {
    this._syncChartLegend(this.scalingChart, this._scalingLegendList, {
        getRename: (ds) => {
            if (!ds || !ds.names || ds.names.length === 0) return null;
            if (ds.groupKind === 'folder') {
                return {
                    prompt: `Rename folder "${ds.groupValue}" (a path; / nests)`,
                    current: ds.groupValue,
                    onSave: (value) => (value && value !== ds.groupValue
                        ? this.moveFolder(ds.groupValue, value) : null),
                };
            }
            return {
                prompt: `Name for mesh group "${ds.groupValue}" (empty: use the mesh name)`,
                current: this._categoryTitle(`mesh:${ds.groupValue}`, ''),
                onSave: (value) => this._saveCompareNames({
                    categories: { [`mesh:${ds.groupValue}`]: value },
                }),
            };
        },
    });
};

// ==================== Iteration Comparison ====================

// ---- user-assigned names -------------------------------------------------
// Runs and mesh headings in the Runs browser can be renamed; the names live in
// viz/run_labels.json (see /api/simulations/labels) so they survive restarts.
// A run's folder groups it instead of its mesh - that is how two campaigns on
// the same mesh stay apart, in the browser and in the chart legends.

// The run's folder path in the Runs browser ('' = unfiled).
App.prototype._collectionOf = function(simName) {
    const entry = (this.runNames || {})[simName];
    return (entry && entry.folder) || '';
};

App.prototype._customRunName = function(simName) {
    const entry = (this.runNames || {})[simName];
    return (entry && entry.label) || '';
};

App.prototype._categoryTitle = function(key, fallback) {
    return (this.categoryNames || {})[key] || fallback;
};

App.prototype._meshOf = function(simName) {
    const match = simName.match(/^(.+?)_sim/);
    return match ? match[1] : 'other';
};

// Same grouping as the Runs browser (folder, else mesh group), as a
// {key, label} pair - used by the scaling chart to draw one line per group.
App.prototype._runGroupKeyAndLabel = function(simName) {
    const folder = this._collectionOf(simName);
    if (folder) return { key: `folder:${folder}`, label: folder };
    const mesh = this._meshOf(simName);
    return { key: `mesh:${mesh}`, label: this._categoryTitle(`mesh:${mesh}`, mesh) };
};

// Small pencil that prompts for a new name and persists whatever comes back.
App.prototype._addRenameButton = function(parent, opts) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = '✎';
    btn.title = opts.prompt;
    btn.style.cssText = 'font-size:0.75em; line-height:1; padding:0 3px; background:transparent; color:#666; border:none; cursor:pointer;';
    btn.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const next = window.prompt(opts.prompt, opts.current || '');
        if (next === null) return;
        await opts.onSave(next.trim());
    });
    parent.appendChild(btn);
};

App.prototype._saveCompareNames = async function(patch) {
    try {
        const resp = await fetch('/api/simulations/labels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patch),
        });
        const saved = await resp.json();
        if (!resp.ok) {
            alert('Rename failed: ' + (saved.error || 'unknown error'));
            return;
        }
        this.runNames = saved.runs || {};
        this.categoryNames = saved.categories || {};
    } catch (err) {
        alert('Rename request failed: ' + err.message);
        return;
    }
    await this.updateCompareSelector();
    // Redraw so the legend of anything plotted picks up the new name
    await this.onCompareSelectionChange();
};

// Reflects the checked/unchecked/mixed state of a group's runs onto its
// select-all checkbox (checked = all selected, indeterminate = some selected).
App.prototype._syncGroupCheckbox = function(groupCb, rowCheckboxes) {
    const checkedCount = rowCheckboxes.filter(cb => cb.checked).length;
    groupCb.checked = rowCheckboxes.length > 0 && checkedCount === rowCheckboxes.length;
    groupCb.indeterminate = checkedCount > 0 && checkedCount < rowCheckboxes.length;
};

// Describe one run as "<preconditioner> (<local solver>), <ranks>", e.g. "bddc (hypre), 16".
// The backend is prefixed only when the comparison spans more than one backend.
App.prototype._formatRunLabel = function(simName, meta, opts) {
    meta = meta || {};
    opts = opts || {};

    // A name the user typed for this run is used as-is.
    const custom = this._customRunName(simName);
    if (custom) return custom;

    const clean = v => {
        const s = (v || '').toString().trim();
        return s.toLowerCase() === 'none' ? '' : s;
    };
    const backend = clean(meta.solver);
    const pc = clean(meta.preconditioner);
    const ls = clean(meta.localSolver);

    let label;
    if (!backend && !pc && !meta.nRanks) {
        label = simName.replace(/^.+?_sim_?/, '') || simName;
    } else {
        label = pc || backend || 'default';
        if (ls) label += ` (${ls})`;
        // Prefix the backend only when it disambiguates and isn't already the label
        if (opts.withBackend && backend && pc && backend.toLowerCase() !== pc.toLowerCase()) {
            label = `${backend}/${label}`;
        }
        if (meta.nRanks) label += `, ${meta.nRanks}`;
    }

    // Name the folder when the comparison spans more than one, so runs of
    // the same config from different campaigns stay apart in the legend.
    if (opts.withFolder) {
        const folder = this._collectionOf(simName);
        if (folder) label = `${folder}: ${label}`;
    }
    return label;
};

// Build simName -> legend label for the current selection, disambiguating runs
// that describe identically (same config, different run) by their timestamp.
App.prototype._buildCompareLabels = function(names, metas) {
    const backends = new Set(metas.map(m => ((m && m.solver) || '').toLowerCase()));
    const folders = new Set(names.map(n => this._collectionOf(n)));
    const opts = { withBackend: backends.size > 1, withFolder: folders.size > 1 };

    const raw = names.map((n, i) => this._formatRunLabel(n, metas[i], opts));
    const counts = {};
    raw.forEach(l => { counts[l] = (counts[l] || 0) + 1; });

    const seen = {};
    const labels = {};
    names.forEach((n, i) => {
        let label = raw[i];
        if (counts[label] > 1 && !this._customRunName(n)) {
            seen[label] = (seen[label] || 0) + 1;
            const ts = n.match(/_\d{8}_(\d{6})/);
            label += ts ? ` [${ts[1]}]` : ` [${seen[label]}]`;
        }
        labels[n] = label;
    });
    return labels;
};

App.prototype._getCompareLabel = function(simName) {
    const cached = this.compareLabels && this.compareLabels[simName];
    if (cached) return cached;
    const meta = this.compareSimMeta && this.compareSimMeta[simName];
    return this._formatRunLabel(simName, meta, {});
};

App.prototype.onCompareSelectionChange = async function() {
    const selected = this.getCompareSelection ? this.getCompareSelection() : [];
    if (selected.length) {
        const chartContainer = document.getElementById('iterations-chart-container');
        if (chartContainer.style.display === 'none') chartContainer.style.display = 'block';
    }

    // Remove old comparison datasets, keep first 2 (solver iterations + current marker)
    if (this.iterationsChart) {
        this.iterationsChart.data.datasets = this.iterationsChart.data.datasets.slice(0, 2);
    }
    this.compareDatasets = [];
    this.compareIterHistory = {};        // simName -> per-timestep iter_history array
    this.compareIterHistoryOrder = [];   // preserve selection order for colour mapping
    this.compareIterHistoryColors = {};
    this.compareBNorms = {};             // simName -> array of ||b|| per timestep (for relative residual)
    this.compareLabels = {};             // simName -> legend label for the current selection
    this._scalingRunData = [];           // one entry per selected run, for the scaling chart

    if (selected.length === 0) {
        document.getElementById('compare-conditions-warning').style.display = 'none';
        if (this.iterationsChart) this.iterationsChart.update('none');
        this._refreshIterationsLegend();
        const c = document.getElementById('residual-history-chart-container');
        if (c) c.style.display = 'none';
        this._updateScalingChart();
        return;
    }

    // Use distinct colors that differ from the primary red (#e94560)
    const colors = ['#4a9de9', '#4ade80', '#e9c74a', '#c74ae9', '#e9844a', '#4ae9c7', '#e94ac7', '#9de94a'];
    const conditionsHashes = new Map();

    // Fetch first so legend labels can be built from the full selection
    // (backend prefix / timestamp suffix only appear when needed to disambiguate).
    const fetched = [];
    for (const name of selected) {
        try {
            fetched.push({ name, data: await this.clusterRunner.fetchIterations(name) });
        } catch (e) {
            console.warn(`Failed to fetch iterations for ${name}:`, e);
        }
    }

    const names = fetched.map(f => f.name);
    const metas = fetched.map(f => {
        const c = f.data.conditions;
        if (c) {
            return {
                solver: c.solver,
                preconditioner: c.preconditioner,
                localSolver: c.localSolver,
                nRanks: c.nRanks,
            };
        }
        return (this.compareSimMeta && this.compareSimMeta[f.name]) || {};
    });
    this.compareLabels = this._buildCompareLabels(names, metas);

    for (let i = 0; i < fetched.length; i++) {
        const name = fetched[i].name;
        const data = fetched[i].data;
        const iters = data.iterations || [];

        const finiteIters = iters.filter(v => typeof v === 'number' && isFinite(v));
        const avgIterations = finiteIters.length
            ? finiteIters.reduce((a, b) => a + b, 0) / finiteIters.length
            : null;
        const scalingMeta = (this.compareSimMeta && this.compareSimMeta[name]) || {};
        const group = this._runGroupKeyAndLabel(name);
        this._scalingRunData.push({
            name,
            avgIterations,
            nSubdomains: scalingMeta.nSubdomains != null ? scalingMeta.nSubdomains : null,
            hRatio: scalingMeta.hRatio != null ? scalingMeta.hRatio : null,
            groupKey: group.key,
            groupLabel: group.label,
        });

        this.compareDatasets.push({
            label: this._getCompareLabel(name),
            name,                                  // sim name, for legend rename
            data: iters,
            borderColor: colors[i % colors.length],
            borderWidth: 1.5,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
        });

        // Cache iter_history (true ||b-Ax|| per Krylov iter, per timestep) for hover-driven sub-plot
        const ih = data.residuals && data.residuals.iter_history;
        if (Array.isArray(ih) && ih.length > 0) {
            this.compareIterHistory[name] = ih;
            this.compareIterHistoryOrder.push(name);
            this.compareIterHistoryColors[name] = colors[i % colors.length];
            // Recover ||b|| per timestep from final residuals (abs / rel) so we can
            // scale the per-iter explicit residual into a relative one.
            const absArr = (data.residuals && data.residuals.abs) || [];
            const relArr = (data.residuals && data.residuals.rel) || [];
            this.compareBNorms[name] = absArr.map((a, t) => {
                const r = relArr[t];
                return (r && isFinite(r) && r > 0) ? (a / r) : null;
            });
        }

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
            conditionsHashes.set(name, JSON.stringify(phys, Object.keys(phys).sort()));
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
        this._refreshIterationsLegend();
    }

    // Show/hide per-iter sub-plot based on whether any selected run carries iter_history
    const subContainer = document.getElementById('residual-history-chart-container');
    if (subContainer) {
        subContainer.style.display = this.compareIterHistoryOrder.length ? 'block' : 'none';
    }
    this.setCompareResidualHistoryForStep(0);

    this._updateScalingChart();
};

App.prototype.setCompareResidualHistoryForStep = function(stepIdx) {
    if (!this.residualHistoryChart) return;
    const order = this.compareIterHistoryOrder || [];
    if (order.length === 0) return;

    // Build one dataset per selected sim showing its true ||b - A·x|| at this timestep.
    const sanitize = arr => (arr || []).map(v =>
        (v === null || v === undefined || !isFinite(v) || v <= 0) ? null : v);

    let maxIters = 0;
    const datasets = [];
    for (const simName of order) {
        const ih = this.compareIterHistory[simName];
        const entry = ih && ih[stepIdx];
        if (!entry || !Array.isArray(entry.iters) || entry.iters.length === 0) continue;
        maxIters = Math.max(maxIters, entry.iters.length);
        const color = this.compareIterHistoryColors[simName] || '#aaa';
        const label = this._getCompareLabel(simName);
        const absExplicit = sanitize(entry.abs_explicit);
        datasets.push({
            label: label + ' (abs)', name: simName, residualMode: 'abs', data: absExplicit,
            borderColor: color, borderWidth: 2, fill: false,
            pointRadius: 0, pointHoverRadius: 3, tension: 0,
        });
        // Relative residual = abs / ||b|| at this timestep — dotted, same colour.
        const bNorms = this.compareBNorms[simName] || [];
        const bNorm = bNorms[stepIdx];
        if (bNorm && isFinite(bNorm) && bNorm > 0) {
            const relExplicit = absExplicit.map(v => v == null ? null : v / bNorm);
            datasets.push({
                label: label + ' (rel)', name: simName, residualMode: 'rel', data: relExplicit,
                borderColor: color, borderDash: [2, 3], borderWidth: 1.5, fill: false,
                pointRadius: 0, pointHoverRadius: 3, tension: 0,
            });
        }
    }

    this.residualHistoryChart.data.labels = Array.from({ length: maxIters }, (_, i) => i);
    this.residualHistoryChart.data.datasets = datasets;
    this.residualHistoryChart.options.plugins.title = {
        display: true, color: '#aaa', font: CHART_TYPO.title,
        text: `Residual norms vs Krylov iteration — timestep ${stepIdx}`,
    };
    this._residualHistoryStep = stepIdx;
    this._applyResidualModesTo(this.residualHistoryChart);
    this.syncResidualToggleVisibility();
    this.residualHistoryChart.update('none');
    this._refreshResidualHistoryLegend();
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
                residualMode: 'abs',
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
                residualMode: 'rel',
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
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#16213e',
                    titleColor: '#fff',
                    bodyColor: '#ccc',
                    borderColor: '#e94560',
                    borderWidth: 1,
                    titleFont: CHART_TYPO.legend,
                    bodyFont: CHART_TYPO.tick,
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
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: { color: '#888', font: CHART_TYPO.tick },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Residual Norm',
                        color: '#888',
                        font: CHART_TYPO.axisTitle
                    },
                    ticks: {
                        color: '#888',
                        font: CHART_TYPO.tick,
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

    this.wireChartExport('residual-chart-export', () => this.residualChart, 'residual');

    // Fixed abs/rel pair - content never changes, so one sync is enough. Visibility
    // is driven by the abs/rel checkboxes above the chart, not by this legend.
    const box = document.querySelector('#residual-chart-container .chart-resize-box');
    this._residualLegendList = this._createChartLegend('residual', box);
    this._syncChartLegend(this.residualChart, this._residualLegendList, {
        toggle: false, requireData: false,
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

// ==================== Residual History (per-timestep, per Krylov iter) ====================
// Driven by the timestep slider via setResidualHistoryForStep(stepIdx).
// Data source: residuals.pickle["iter_history"][stepIdx] -> {iters, abs_explicit, abs_implicit?}

App.prototype.setupResidualHistoryChart = function() {
    const el = document.getElementById('residual-history-chart');
    if (!el) return;
    const ctx = el.getContext('2d');

    this.residualHistoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: residualHistorySingleRunDatasets()
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index', intersect: false,
                    backgroundColor: '#16213e', titleColor: '#fff', bodyColor: '#ccc',
                    titleFont: CHART_TYPO.legend, bodyFont: CHART_TYPO.tick,
                    callbacks: {
                        label: c => `${c.dataset.label}: ${c.parsed.y.toExponential(2)}`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Krylov iteration', color: '#888',
                              font: CHART_TYPO.axisTitle },
                     ticks: { color: '#888', font: CHART_TYPO.tick },
                     grid: { color: 'rgba(255,255,255,0.1)' } },
                y: { type: 'logarithmic',
                     title: { display: true, text: 'Residual Norm', color: '#888',
                              font: CHART_TYPO.axisTitle },
                     ticks: { color: '#888', font: CHART_TYPO.tick,
                              callback: v => v.toExponential(0) },
                     grid: { color: 'rgba(255,255,255,0.1)' } }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }
        }
    });

    this.wireChartExport('residual-history-chart-export',
                         () => this.residualHistoryChart,
                         () => `residual_history_step${this._residualHistoryStep || 0}`);

    const box = document.querySelector('#residual-history-chart-container .chart-resize-box');
    this._residualHistoryLegendList = this._createChartLegend('residual-history', box);
    this._refreshResidualHistoryLegend();
};

// Single-run mode shows the fixed explicit/implicit pair (no rename target);
// comparison mode swaps in one (abs, rel) pair per selected run, each carrying
// the sim name it came from (see setCompareResidualHistoryForStep) and so
// renameable like any other run. Visibility stays with the abs/rel checkboxes.
App.prototype._refreshResidualHistoryLegend = function() {
    this._syncChartLegend(this.residualHistoryChart, this._residualHistoryLegendList, {
        toggle: false,
        getRename: (ds) => {
            if (!ds || !ds.name) return null;
            const simName = ds.name;
            return {
                prompt: `Name for this run (empty: default)\n${simName}`,
                current: this._customRunName(simName),
                onSave: (value) => this._saveCompareNames({ runs: { [simName]: { label: value } } }),
            };
        },
    });
};

App.prototype.setIterHistoryCache = function(history) {
    // history: array (per-timestep) of {iters, abs_explicit, abs_implicit?} or null/missing
    this.iterHistoryCache = Array.isArray(history) ? history : [];
    const container = document.getElementById('residual-history-chart-container');
    if (!container) return;
    const hasAny = this.iterHistoryCache.some(e => e && Array.isArray(e.iters) && e.iters.length > 0);
    container.style.display = hasAny ? 'block' : 'none';
};

App.prototype.setResidualHistoryForStep = function(stepIdx) {
    if (!this.residualHistoryChart || !this.iterHistoryCache) return;
    const entry = this.iterHistoryCache[stepIdx];
    const sanitize = arr => (arr || []).map(v =>
        (v === null || v === undefined || !isFinite(v) || v <= 0) ? null : v);

    const iters = (entry && Array.isArray(entry.iters)) ? entry.iters : [];
    const expl = sanitize(entry && entry.abs_explicit);
    const impl = sanitize(entry && entry.abs_implicit);

    // Comparison mode replaces the dataset list wholesale, so restore the
    // single-run pair before writing into it.
    const ds = this.residualHistoryChart.data.datasets;
    if (ds.length !== 2 || ds[0].label !== 'true ||b - A·x||') {
        this.residualHistoryChart.data.datasets = residualHistorySingleRunDatasets();
    }

    this.residualHistoryChart.data.labels = iters;
    this.residualHistoryChart.data.datasets[0].data = expl;
    this.residualHistoryChart.data.datasets[1].data = impl;
    this._residualHistoryStep = stepIdx;
    this.residualHistoryChart.update('none');
    this._refreshResidualHistoryLegend();
};

// ==================== Residual mode (abs / rel) toggle ====================
// Datasets tagged with `residualMode` are shown only while that mode is enabled.
// Untagged datasets (e.g. the explicit/implicit pair of a single run) are unaffected.

const RESIDUAL_MODES_KEY = 'chartResidualModes';

App.prototype.setupResidualModeToggles = function() {
    this.residualModes = { abs: true, rel: true };
    try {
        const saved = JSON.parse(localStorage.getItem(RESIDUAL_MODES_KEY) || 'null');
        if (saved && (saved.abs || saved.rel)) {
            this.residualModes = { abs: !!saved.abs, rel: !!saved.rel };
        }
    } catch (e) { /* keep defaults */ }

    document.querySelectorAll('.residual-mode-toggle input[data-residual-mode]').forEach(cb => {
        cb.addEventListener('change', () => {
            const mode = cb.dataset.residualMode;
            const next = { ...this.residualModes, [mode]: cb.checked };
            // Keep at least one series on screen
            if (!next.abs && !next.rel) {
                cb.checked = true;
                return;
            }
            this.residualModes = next;
            try {
                localStorage.setItem(RESIDUAL_MODES_KEY, JSON.stringify(next));
            } catch (e) { /* non-fatal */ }
            this.syncResidualModeToggles();
            this.applyResidualModes();
        });
    });

    this.syncResidualModeToggles();
    this.applyResidualModes();
};

App.prototype.syncResidualModeToggles = function() {
    document.querySelectorAll('.residual-mode-toggle input[data-residual-mode]').forEach(cb => {
        cb.checked = !!this.residualModes[cb.dataset.residualMode];
    });
};

App.prototype._applyResidualModesTo = function(chart) {
    if (!chart) return;
    const modes = this.residualModes || { abs: true, rel: true };
    for (const ds of chart.data.datasets) {
        if (ds.residualMode) ds.hidden = !modes[ds.residualMode];
    }
};

App.prototype.applyResidualModes = function() {
    for (const chart of [this.residualChart, this.residualHistoryChart]) {
        if (!chart) continue;
        this._applyResidualModesTo(chart);
        chart.update('none');
    }
    this.syncResidualToggleVisibility();
    // Keep the floating legends' dimmed/active styling in sync with the toggle.
    if (this.residualChart) {
        this._syncChartLegend(this.residualChart, this._residualLegendList, {
            toggle: false, requireData: false,
        });
    }
    this._refreshResidualHistoryLegend();
};

// Hide the toggle on a chart whose series aren't abs/rel pairs — the single-run
// residual history plots explicit vs implicit norms, which the modes don't split.
App.prototype.syncResidualToggleVisibility = function() {
    const panels = [
        ['residual-chart-container', this.residualChart],
        ['residual-history-chart-container', this.residualHistoryChart],
    ];
    for (const [containerId, chart] of panels) {
        const toggle = document.querySelector(`#${containerId} .residual-mode-toggle`);
        if (!toggle) continue;
        const applies = !!chart && chart.data.datasets.some(ds => ds.residualMode);
        toggle.style.display = applies ? 'flex' : 'none';
    }
};

// ==================== Resizable chart panels ====================

const CHART_SIZES_KEY = 'chartPanelSizes';

App.prototype.setupChartPanels = function() {
    let saved = {};
    try {
        saved = JSON.parse(localStorage.getItem(CHART_SIZES_KEY) || '{}') || {};
    } catch (e) { /* ignore */ }

    const persist = () => {
        const sizes = {};
        document.querySelectorAll('.chart-resize-box').forEach(box => {
            const name = box.dataset.chartBox;
            if (name && box.style.width) {
                sizes[name] = { width: box.style.width, height: box.style.height };
            }
        });
        try {
            localStorage.setItem(CHART_SIZES_KEY, JSON.stringify(sizes));
        } catch (e) { /* non-fatal */ }
    };

    let persistTimer = null;
    const observer = (typeof ResizeObserver !== 'undefined')
        ? new ResizeObserver(() => {
            clearTimeout(persistTimer);
            persistTimer = setTimeout(persist, 300);
        })
        : null;

    document.querySelectorAll('.chart-resize-box').forEach(box => {
        const size = saved[box.dataset.chartBox];
        if (size && size.width) box.style.width = size.width;
        if (size && size.height) box.style.height = size.height;
        if (observer) observer.observe(box);
    });
};

// ==================== PNG export ====================
// Chart.js never paints the canvas background, so toDataURL() yields a PNG with a
// transparent background. Exports are re-rendered for print first: dark ink, dark
// grid, and series colours darkened enough to read on white paper (Overleaf).
// Shift-click exports the on-screen light-on-dark styling instead.

const EXPORT_INK = '#1a1a1a';
const EXPORT_GRID = 'rgba(0,0,0,0.18)';
const EXPORT_AXIS = 'rgba(0,0,0,0.55)';
// Max relative luminance a series colour may have on white (~3.5:1 contrast).
const EXPORT_MAX_LUMINANCE = 0.24;
// Backing-store scale for the exported bitmap, so figures survive LaTeX scaling.
const EXPORT_PIXEL_RATIO = 3;

function parseCssColor(color) {
    if (typeof color !== 'string') return null;
    const hex = color.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
        let h = hex[1];
        if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        return {
            r: parseInt(h.slice(0, 2), 16),
            g: parseInt(h.slice(2, 4), 16),
            b: parseInt(h.slice(4, 6), 16),
            a: 1,
        };
    }
    const rgb = color.trim().match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,/\s]+([\d.]+))?\s*\)$/i);
    if (rgb) {
        return {
            r: parseFloat(rgb[1]), g: parseFloat(rgb[2]), b: parseFloat(rgb[3]),
            a: rgb[4] === undefined ? 1 : parseFloat(rgb[4]),
        };
    }
    return null;
}

function relativeLuminance(c) {
    const lin = v => {
        const s = v / 255;
        return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
}

// Scale a colour towards black until it is dark enough to read on white,
// keeping its hue. Colours already dark enough are returned unchanged.
function darkenForPrint(color, maxLuminance) {
    const c = parseCssColor(color);
    if (!c) return color;
    if (relativeLuminance(c) <= maxLuminance) return color;

    let lo = 0, hi = 1;
    for (let i = 0; i < 24; i++) {
        const mid = (lo + hi) / 2;
        const scaled = { r: c.r * mid, g: c.g * mid, b: c.b * mid };
        if (relativeLuminance(scaled) > maxLuminance) hi = mid; else lo = mid;
    }
    const f = lo;
    const r = Math.round(c.r * f), g = Math.round(c.g * f), b = Math.round(c.b * f);
    return c.a >= 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${c.a})`;
}

App.prototype.wireChartExport = function(btnId, getChart, baseName) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener('click', async (e) => {
        const name = (typeof baseName === 'function') ? baseName() : baseName;
        await this.exportChartPNG(getChart(), name, !e.shiftKey);
    });
};

// Restyle a chart for printing on white: dark ink for text, dark grid/axis lines,
// and series colours darkened where they'd be too faint.
// Returns a function that puts every original value back.
App.prototype._applyExportInk = function(chart) {
    const undo = [];
    // Chart.js resolves some option sub-objects (e.g. a scale's merged defaults)
    // through a Proxy; writing to one of those can throw a proxy-invariant
    // TypeError ("trap reported non-configurability...") depending on how that
    // chart's scale was configured. Skip that one property rather than letting
    // it abort the whole export.
    const set = (obj, key, val) => {
        if (!obj) return;
        try {
            const had = Object.prototype.hasOwnProperty.call(obj, key);
            undo.push([obj, key, obj[key], had]);
            obj[key] = val;
        } catch (err) {
            console.warn(`Chart export: could not restyle "${key}" for print`, err);
        }
    };
    // Chart.js leaves some sub-options undefined; create them so we can colour them.
    const ensure = (parent, key) => {
        if (!parent) return null;
        try {
            if (!parent[key]) set(parent, key, {});
            return parent[key];
        } catch (err) {
            console.warn(`Chart export: could not restyle "${key}" for print`, err);
            return null;
        }
    };

    const scales = chart.options.scales || {};
    for (const key of Object.keys(scales)) {
        const scale = scales[key];
        if (!scale) continue;
        set(ensure(scale, 'ticks'), 'color', EXPORT_INK);
        if (scale.title) set(scale.title, 'color', EXPORT_INK);
        set(ensure(scale, 'grid'), 'color', EXPORT_GRID);
        set(ensure(scale, 'border'), 'color', EXPORT_AXIS);
    }

    const plugins = chart.options.plugins || {};
    if (plugins.legend) set(ensure(plugins.legend, 'labels'), 'color', EXPORT_INK);
    if (plugins.title) set(plugins.title, 'color', EXPORT_INK);

    for (const ds of chart.data.datasets) {
        for (const key of ['borderColor', 'backgroundColor', 'pointBackgroundColor']) {
            if (typeof ds[key] !== 'string') continue;
            const dark = darkenForPrint(ds[key], EXPORT_MAX_LUMINANCE);
            if (dark !== ds[key]) set(ds, key, dark);
        }
    }

    return () => undo.reverse().forEach(([obj, key, val, had]) => {
        if (had) obj[key] = val; else delete obj[key];
    });
};

// The floating legend is a sibling HTML element, not part of the canvas, so a
// plain toDataURL() export never includes it. Locate it (via the canvas's own
// .chart-resize-box ancestor - works for any chart, no per-chart bookkeeping)
// and report its position as fractions of the box, so it can be redrawn at the
// matching spot on a canvas of any size (the export canvas is 3x for print).
App.prototype._getChartLegendGeometry = function(chart) {
    const box = chart.canvas.closest('.chart-resize-box');
    if (!box) return null;
    const overlay = box.querySelector('.chart-legend-overlay');
    if (!overlay || getComputedStyle(overlay).display === 'none') return null;
    if (overlay.querySelectorAll('.chart-legend-item').length === 0) return null;

    const boxRect = box.getBoundingClientRect();
    const elRect = overlay.getBoundingClientRect();
    if (!boxRect.width || !boxRect.height) return null;
    return {
        overlay,
        xFrac: (elRect.left - boxRect.left) / boxRect.width,
        yFrac: (elRect.top - boxRect.top) / boxRect.height,
        wFrac: elRect.width / boxRect.width,
        hFrac: elRect.height / boxRect.height,
    };
};

// Rounded-rect path, falling back to a manual path on browsers without the
// native CanvasRenderingContext2D.roundRect (Safari < 16).
function tracePillRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
        return;
    }
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

// Draws the floating legend (as currently shown: position, entries, dimmed
// state) onto a flattened copy of the chart's exported bitmap. Returns a data
// URL - the chart's own toDataURL() unchanged if there's no legend to draw, or
// on any failure, so compositing can never block the export itself.
App.prototype._compositeChartWithLegend = function(chart, forPrint) {
    const chartUrl = chart.canvas.toDataURL('image/png');
    let geo;
    try {
        geo = this._getChartLegendGeometry(chart);
    } catch (err) {
        console.warn('Chart export: could not locate legend', err);
        return Promise.resolve(chartUrl);
    }
    if (!geo) return Promise.resolve(chartUrl);

    const items = Array.from(geo.overlay.querySelectorAll('.chart-legend-item')).map(row => {
        const swatch = row.querySelector('.chart-legend-swatch');
        const label = row.querySelector('.chart-legend-label');
        return {
            text: label ? label.textContent : '',
            color: swatch ? getComputedStyle(swatch).backgroundColor : '#888',
            dimmed: row.classList.contains('chart-legend-item-hidden'),
        };
    }).filter(item => item.text);
    if (items.length === 0) return Promise.resolve(chartUrl);

    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                const x = geo.xFrac * canvas.width;
                const y = geo.yFrac * canvas.height;
                const w = geo.wFrac * canvas.width;
                const h = geo.hFrac * canvas.height;
                const scale = canvas.width / 800; // reference: an ~800px-wide chart panel

                const ink = forPrint ? '#1a1a1a' : '#cccccc';
                const bg = forPrint ? 'rgba(255,255,255,0.92)' : 'rgba(22,33,62,0.88)';
                const border = forPrint ? 'rgba(0,0,0,0.35)' : '#0f3460';

                ctx.save();
                tracePillRect(ctx, x, y, w, h, 5 * scale);
                ctx.fillStyle = bg;
                ctx.fill();
                ctx.lineWidth = Math.max(1, scale);
                ctx.strokeStyle = border;
                ctx.stroke();

                const pad = 8 * scale;
                const rowH = (h - 2 * pad) / items.length;
                const fontSize = Math.max(9 * scale, rowH * 0.55);
                ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
                ctx.textBaseline = 'middle';

                items.forEach((item, i) => {
                    const rowY = y + pad + rowH * i + rowH / 2;
                    const swatchSize = fontSize * 0.85;
                    ctx.globalAlpha = item.dimmed ? 0.4 : 1;
                    ctx.fillStyle = item.color;
                    ctx.fillRect(x + pad, rowY - swatchSize / 2, swatchSize, swatchSize);
                    ctx.fillStyle = ink;
                    ctx.fillText(item.text, x + pad + swatchSize + pad * 0.6, rowY);
                });
                ctx.restore();

                resolve(canvas.toDataURL('image/png'));
            } catch (err) {
                console.warn('Chart export: could not draw legend onto export', err);
                resolve(chartUrl);
            }
        };
        img.onerror = () => resolve(chartUrl);
        img.src = chartUrl;
    });
};

App.prototype.exportChartPNG = async function(chart, baseName, forPrint) {
    if (!chart) return;

    let restore = null;
    if (forPrint) {
        try {
            restore = this._applyExportInk(chart);
        } catch (err) {
            // Never let print restyling block the export itself.
            console.warn('Chart export: print restyling failed, exporting as-is', err);
        }
    }
    const savedRatio = chart.options.devicePixelRatio;
    chart.options.devicePixelRatio = EXPORT_PIXEL_RATIO;

    let dataUrl;
    try {
        chart.resize();         // grow the backing store to the export resolution
        chart.update('none');   // draw synchronously (animations are off) before grabbing it
        dataUrl = await this._compositeChartWithLegend(chart, forPrint);
    } finally {
        if (savedRatio === undefined) delete chart.options.devicePixelRatio;
        else chart.options.devicePixelRatio = savedRatio;
        if (restore) restore();
        chart.resize();
        chart.update('none');
    }

    const stamp = new Date().toISOString().replace(/\D/g, '').slice(0, 14);
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `${baseName}_${stamp}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};
