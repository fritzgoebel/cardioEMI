// app-results.js - Results loading, viz data generation, time stepping, simulation list

App.prototype.loadSimulationList = async function() {
    const simSelector = document.getElementById('simulation-selector');

    try {
        const response = await fetch('/api/simulations');
        const data = await response.json();
        const localNames = new Set(data.simulations.map(s => s.name));

        simSelector.innerHTML = '<option value="">Select simulation...</option>';

        data.simulations.forEach(sim => {
            const option = document.createElement('option');
            option.value = sim.name;
            const label = (sim.solver || sim.mesh)
                ? this.formatSimulationLabelInline(sim)
                : sim.name;
            option.textContent = label + (sim.has_viz_data ? '' : ' (will generate viz data)');
            option.title = sim.name;
            simSelector.appendChild(option);
        });

        // If in Karolina mode, also fetch remote simulations
        if (this.runTarget === 'karolina') {
            try {
                const remoteResp = await fetch('/api/karolina/remote-simulations');
                const remoteData = await remoteResp.json();
                if (remoteData.simulations) {
                    for (const entry of remoteData.simulations) {
                        // Backwards-compat: tolerate legacy bare-string entries.
                        const sim = (typeof entry === 'string') ? { name: entry } : entry;
                        if (!sim.name) continue;
                        if (localNames.has(sim.name)) continue;
                        const option = document.createElement('option');
                        option.value = sim.name;
                        const label = (sim.solver || sim.mesh)
                            ? this.formatSimulationLabelInline(sim)
                            : sim.name;
                        option.textContent = label + ' (remote)';
                        option.title = sim.name;
                        simSelector.appendChild(option);
                    }
                }
            } catch (e) {
                console.warn('Failed to load remote simulations:', e);
            }
        }

        if (data.simulations.length > 0) {
            simSelector.value = data.simulations[0].name;
            this.selectedSimulation = data.simulations[0].name;
        }

        this.updateCompareSelector();
    } catch (error) {
        console.error('Failed to load simulation list:', error);
    }
};

App.prototype.deleteAllResults = async function() {
    const first = window.confirm(
        'Delete ALL simulation results?\n\n' +
        'This wipes every local *_sim* directory, every viz/data cache, ' +
        'AND every remote *_sim* directory on Karolina.\n\n' +
        'This cannot be undone.'
    );
    if (!first) return;
    const second = window.confirm('Really delete everything? Final confirmation.');
    if (!second) return;

    const btn = document.getElementById('delete-all-results');
    const originalText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = 'Deleting...'; }
    try {
        const resp = await fetch('/api/simulations/delete_all', { method: 'POST' });
        const result = await resp.json();
        if (!resp.ok) {
            alert('Delete failed: ' + (result.error || 'unknown error'));
            return;
        }
        const summary = [
            `Local removed: ${(result.removed_local || []).length}`,
            `viz cache cleared: ${(result.viz_cleared || []).length}`,
            `Remote cleared: ${result.remote_cleared ? 'yes' : 'no'}`,
        ].join('  •  ');
        if (result.errors && result.errors.length) {
            console.warn('Partial delete failure:', result.errors);
            alert(summary + '\n\nErrors:\n' + result.errors.join('\n'));
        } else {
            console.log('delete_all:', summary);
        }
        await this.loadSimulationList();
        if (this.updateCompareSelector) await this.updateCompareSelector();
    } catch (err) {
        alert('Delete request failed: ' + err.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = originalText; }
    }
};

App.prototype._generateVizData = async function(simName, statusEl) {
    return new Promise((resolve, reject) => {
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
                    buffer = lines.pop();

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
};

App.prototype.loadResults = async function() {
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

        // Per-iter residual history loads from a separate endpoint that doesn't
        // require voltage data — so the chart shows even if voltage gen fails.
        try {
            const ihResp = await fetch(`/api/results/iterations/${encodeURIComponent(simName)}`);
            if (ihResp.ok) {
                const ihData = await ihResp.json();
                const ih = ihData.residuals && ihData.residuals.iter_history;
                this.setIterHistoryCache(ih);
                this.setResidualHistoryForStep(0);
            }
        } catch (e) { console.warn('iter_history fetch failed:', e); }

        const regenerate = document.getElementById('regenerate-viz').checked;
        const url = `/api/results?dir=${encodeURIComponent(simName)}`;

        if (regenerate) {
            loadBtn.textContent = 'Generating viz data...';
            statusEl.className = 'mesh-status';
            statusEl.textContent = 'Starting viz data generation...';
            statusEl.style.display = 'block';
            await this._generateVizData(simName, statusEl);
            statusEl.textContent = 'Viz data generated, loading results...';
            loadBtn.textContent = 'Loading results...';
        }

        let response = await fetch(url);

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

        if (vizDir && vizDir !== this.meshLoader.currentMesh) {
            loadBtn.textContent = 'Reloading mesh...';
            this.meshLoader.setMesh(vizDir);
            const meshData = await this.meshLoader.load((msg) => {
                loadBtn.textContent = msg;
            });
            loadBtn.textContent = 'Building 3D scene...';
            this.meshBounds = meshData.metadata.bounds;
            this.conversionFactor = meshData.metadata.mesh_conversion_factor;
            this._latestMetadata = meshData.metadata;
            await this.viewer.reloadMesh(meshData);
        }

        // Reveal cross-section toggle if volume data was exported
        if (this._latestMetadata && this._latestMetadata.cross_section) {
            this._applyCrossSectionDefaults(this._latestMetadata.cross_section);
            this.showCrossSectionOption(true);
        } else {
            this.showCrossSectionOption(false);
        }

        this.resultsVizDir = vizDir;
        this.resultsData = null;
        this._voltageCache = {};
        this.resultsTimeSteps = data.times;

        // Update UI
        const timeSlider = document.getElementById('result-time');
        timeSlider.max = this.resultsTimeSteps.length - 1;
        timeSlider.value = 0;
        document.getElementById('result-time-val').textContent = this.resultsTimeSteps[0].toFixed(3);

        // Clamp to physiological voltage range (AP model can overshoot)
        const vMin = Math.max(data.vMin, -80);
        const vMax = Math.min(data.vMax, 30);

        document.getElementById('v-min-result').textContent = Math.round(vMin);
        document.getElementById('v-max-result').textContent = Math.round(vMax);

        this.viewer.setVoltageRange(vMin, vMax);
        document.getElementById('colorbar-max').textContent = `${Math.round(vMax)} mV`;
        document.getElementById('colorbar-min').textContent = `${Math.round(vMin)} mV`;
        document.getElementById('colorbar-mid').textContent = `${Math.round((vMax + vMin) / 2)} mV`;

        // Apply scar config from the simulation's YAML
        if (data.scarConfig && data.scarConfig.regions && data.scarConfig.regions.length > 0) {
            const region = data.scarConfig.regions[0];
            this.scarBox = { ...region.box };
            this.scarMargin = region.margin || 10;
            this.scarEnabled = true;
            document.getElementById('scar-enabled').checked = true;
            document.getElementById('scar-controls').style.display = 'block';
            this._updateScarSlidersFromConfig(region);
            this.viewer.setScarZones(this.scarBox, this.scarMargin, true);
            this.viewer.updateScarBox(this.scarBox, this.scarMargin, true);
        } else {
            this.viewer.setScarZones(null, 0, false);
        }

        // Iterations data
        if (data.iterations && data.iterations.length > 0) {
            this.setIterationsData(data.iterations);
            this.showIterationsChart();
            this.highlightIterationStep(0, this.resultsTimeSteps.length);
        } else {
            this.hideIterationsChart();
        }

        // Residuals data
        if (data.residuals && data.residuals.abs && data.residuals.abs.length > 0) {
            this.setResidualData(data.residuals.abs, data.residuals.rel);
            this.showResidualChart();
        } else {
            this.hideResidualChart();
        }

        // Per-iteration residual history (opt-in via track_iter_residuals)
        this.setIterHistoryCache(data.residuals && data.residuals.iter_history);
        this.setResidualHistoryForStep(0);

        // Load MPI rank data as binary if available
        if (data.hasRankData && data.numRanks) {
            loadBtn.textContent = 'Loading rank data...';
            const binBase = `/api/results/binary/${encodeURIComponent(vizDir)}`;

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

        await this.showResultsAtTime(0);

        document.getElementById('regenerate-viz').checked = false;
        loadBtn.textContent = 'Reload Results';
    } catch (error) {
        alert('Failed to load results: ' + error.message);
        loadBtn.textContent = originalText;
    } finally {
        loadBtn.disabled = false;
    }
};

App.prototype.showResultsAtTime = async function(timeIndex) {
    if (!this.resultsVizDir || !this.viewer) return;

    const showPartition = document.getElementById('show-partition').checked;
    if (this.crossSectionMode) {
        await this._applyCrossSectionAtTime(timeIndex);
    } else if (!showPartition) {
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
};

// ---------------------- Cross-section mode ----------------------

App.prototype._initCrossSectionCaches = function() {
    if (!this._intraCache) this._intraCache = {};
    if (!this._extraCache) this._extraCache = {};
};

App.prototype._fetchPhiI = async function(tag, timeIndex) {
    this._initCrossSectionCaches();
    const cacheKey = `${tag}:${timeIndex}`;
    if (this._intraCache[cacheKey]) return this._intraCache[cacheKey];
    const url = `/api/cross-section/binary/${encodeURIComponent(this.resultsVizDir)}/phi_i/${tag}/${timeIndex}.bin`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const arr = new Float32Array(await resp.arrayBuffer());
    this._intraCache[cacheKey] = arr;
    return arr;
};

App.prototype._fetchPhiECap = async function(timeIndex) {
    this._initCrossSectionCaches();
    if (this._extraCache[timeIndex]) return this._extraCache[timeIndex];
    const url = `/api/cross-section/binary/${encodeURIComponent(this.resultsVizDir)}/cap/phi_e/${timeIndex}.bin`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const arr = new Float32Array(await resp.arrayBuffer());
    this._extraCache[timeIndex] = arr;
    return arr;
};

App.prototype._fetchPhiEShell = async function(timeIndex) {
    if (!this._shellCache) this._shellCache = {};
    if (this._shellCache[timeIndex]) return this._shellCache[timeIndex];
    const url = `/api/cross-section/binary/${encodeURIComponent(this.resultsVizDir)}/phi_e_shell/${timeIndex}.bin`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const arr = new Float32Array(await resp.arrayBuffer());
    this._shellCache[timeIndex] = arr;
    return arr;
};

App.prototype._applyCrossSectionAtTime = async function(timeIndex) {
    if (!this.viewer) return;
    const cellTags = Array.from(this.viewer.cellMeshes.keys());
    const fetches = cellTags.map(tag => this._fetchPhiI(tag, timeIndex).then(arr => [tag, arr]));
    const results = await Promise.all(fetches);
    const phiByTag = new Map();
    for (const [tag, arr] of results) {
        if (arr) phiByTag.set(tag, arr);
    }
    this._lastIntraPhi = phiByTag;
    this.viewer.updatePhiIColors(phiByTag);

    if (this.viewer.capMeshObject) {
        const cap = await this._fetchPhiECap(timeIndex);
        if (cap) {
            this._lastExtraPhi = cap;
            this.viewer.updatePhiECapColors(cap);
        }
    }
    // ECS shell coloring (φ_e on the volume below the clip plane)
    const shell = await this._fetchPhiEShell(timeIndex);
    if (shell) {
        this._lastShellPhi = shell;
        this.viewer.updatePhiEShellColors(shell);
    }
    this._updateCrossSectionStats();
};

App.prototype._arrayStats = function(arr) {
    if (!arr || !arr.length) return null;
    let mn = arr[0], mx = arr[0];
    for (let i = 1; i < arr.length; i++) {
        const v = arr[i];
        if (v < mn) mn = v;
        else if (v > mx) mx = v;
    }
    return { min: mn, max: mx };
};

App.prototype._computeCrossSectionStats = function() {
    let intraMin = Infinity, intraMax = -Infinity;
    if (this._lastIntraPhi) {
        for (const arr of this._lastIntraPhi.values()) {
            const s = this._arrayStats(arr);
            if (s) {
                if (s.min < intraMin) intraMin = s.min;
                if (s.max > intraMax) intraMax = s.max;
            }
        }
    }
    const capStats = this._arrayStats(this._lastExtraPhi);
    const shellStats = this._arrayStats(this._lastShellPhi);
    let extraMin = Infinity, extraMax = -Infinity;
    for (const s of [capStats, shellStats]) {
        if (!s) continue;
        if (s.min < extraMin) extraMin = s.min;
        if (s.max > extraMax) extraMax = s.max;
    }
    return {
        intra: Number.isFinite(intraMin) ? { min: intraMin, max: intraMax } : null,
        cap: capStats,
        shell: shellStats,
        extra: Number.isFinite(extraMin) ? { min: extraMin, max: extraMax } : null,
    };
};

App.prototype._updateCrossSectionStats = function() {
    const el = document.getElementById('cs-stats');
    if (!el) return;
    const s = this._computeCrossSectionStats();
    const parts = [];
    if (s.intra) parts.push(`φᵢ cells: [${s.intra.min.toFixed(2)}, ${s.intra.max.toFixed(2)}]`);
    if (s.cap)   parts.push(`φₑ cap: [${s.cap.min.toFixed(3)}, ${s.cap.max.toFixed(3)}]`);
    if (s.shell) parts.push(`φₑ shell: [${s.shell.min.toFixed(3)}, ${s.shell.max.toFixed(3)}]`);
    el.textContent = parts.length ? `Current step — ${parts.join('  •  ')}` : '';
};

App.prototype.autofitCrossSectionRanges = function() {
    const s = this._computeCrossSectionStats();
    const pad = (lo, hi) => {
        if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
        if (hi === lo) {
            const eps = Math.max(Math.abs(lo) * 1e-3, 1e-6);
            return [lo - eps, hi + eps];
        }
        const span = (hi - lo) * 0.05;
        return [lo - span, hi + span];
    };
    if (s.intra) {
        const padded = pad(s.intra.min, s.intra.max);
        if (padded) {
            const [lo, hi] = padded;
            this.viewer.setIntraRange(lo, hi);
            const minIn = document.getElementById('cs-intra-vmin');
            const maxIn = document.getElementById('cs-intra-vmax');
            if (minIn) minIn.value = lo.toFixed(3);
            if (maxIn) maxIn.value = hi.toFixed(3);
        }
    }
    // Fit φₑ to the cap's range (boundary shell often sits at a Dirichlet
    // value and dominates a combined scale, hiding all cap variation). Fall
    // back to the shell range if no cap data is available.
    const phiERef = s.cap || s.shell;
    if (phiERef) {
        const padded = pad(phiERef.min, phiERef.max);
        if (padded) {
            const [lo, hi] = padded;
            this.viewer.setExtraRange(lo, hi);
            const minIn = document.getElementById('cs-extra-vmin');
            const maxIn = document.getElementById('cs-extra-vmax');
            if (minIn) minIn.value = lo.toFixed(4);
            if (maxIn) maxIn.value = hi.toFixed(4);
        }
    }
    if (this._refreshCsLabels) this._refreshCsLabels();
    this.refreshCrossSectionColors();
};

App.prototype.setCrossSectionMode = async function(enabled) {
    this.crossSectionMode = !!enabled;
    if (this.viewer) {
        this.viewer.setCrossSectionVisible(this.crossSectionMode);
    }
    if (this.crossSectionMode) {
        const timeIndex = parseInt(document.getElementById('result-time').value, 10) || 0;
        await this._applyCrossSectionAtTime(timeIndex);
    }
};

App.prototype.applyCrossSectionPlane = async function(normal, offset) {
    if (!this.resultsVizDir) {
        throw new Error('Load results first');
    }
    const resp = await fetch('/api/cross-section/slice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sim_name: this.resultsVizDir, normal, offset }),
    });
    const data = await resp.json();
    if (!resp.ok) {
        throw new Error(data.error || 'Failed to slice ECS volume');
    }
    // Cap geometry / per-timestep φ_e changed → drop cap cache, refetch geometry.
    this._extraCache = {};
    const capData = await this.meshLoader.loadCap();
    this.viewer.createCapMesh(capData);

    // Clip the ECS shell so the volume below the plane stays visible.
    this.viewer.setClippingPlane(normal, offset);

    // Auto-tune φ_e range to the actual cap values (cap is typically a much
    // narrower band than the full volume range, so the metadata default looks
    // washed out / mono-coloured).
    if (Array.isArray(data.phi_e_range) && data.phi_e_range.length === 2) {
        const [lo, hi] = data.phi_e_range;
        if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
            this.viewer.setExtraRange(lo, hi);
            const minIn = document.getElementById('cs-extra-vmin');
            const maxIn = document.getElementById('cs-extra-vmax');
            if (minIn) minIn.value = lo.toFixed(3);
            if (maxIn) maxIn.value = hi.toFixed(3);
            if (this._refreshCsLabels) this._refreshCsLabels();
        }
    }

    if (this.crossSectionMode) {
        const timeIndex = parseInt(document.getElementById('result-time').value, 10) || 0;
        await this._applyCrossSectionAtTime(timeIndex);
        // Tighten φₑ to the cap's own range so the slice variation is visible
        // even when the shell sits at a saturating Dirichlet value.
        this.autofitCrossSectionRanges();
    }
    return data;
};

App.prototype.refreshCrossSectionColors = function() {
    if (this._lastIntraPhi) this.viewer.refreshIntraColors(this._lastIntraPhi);
    if (this._lastExtraPhi) this.viewer.refreshExtraColors(this._lastExtraPhi);
    if (this._lastShellPhi) this.viewer.refreshShellColors(this._lastShellPhi);
};

App.prototype._applyCrossSectionDefaults = function(meta) {
    if (!this.viewer || !meta) return;
    const padRange = (lo, hi, padFrac = 0.05) => {
        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo, hi];
        const span = (hi - lo) * padFrac;
        return [lo - span, hi + span];
    };
    if (Array.isArray(meta.phi_i_range) && meta.phi_i_range.length === 2) {
        const [lo, hi] = padRange(meta.phi_i_range[0], meta.phi_i_range[1]);
        this.viewer.setIntraRange(lo, hi);
        const minIn = document.getElementById('cs-intra-vmin');
        const maxIn = document.getElementById('cs-intra-vmax');
        if (minIn) minIn.value = lo.toFixed(2);
        if (maxIn) maxIn.value = hi.toFixed(2);
    }
    if (Array.isArray(meta.phi_e_range) && meta.phi_e_range.length === 2) {
        const [lo, hi] = padRange(meta.phi_e_range[0], meta.phi_e_range[1]);
        this.viewer.setExtraRange(lo, hi);
        const minIn = document.getElementById('cs-extra-vmin');
        const maxIn = document.getElementById('cs-extra-vmax');
        if (minIn) minIn.value = lo.toFixed(3);
        if (maxIn) maxIn.value = hi.toFixed(3);
    }
};
