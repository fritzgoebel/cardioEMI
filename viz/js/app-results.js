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

        if (data.simulations.length > 0) {
            simSelector.value = data.simulations[0].name;
            this.selectedSimulation = data.simulations[0].name;
        }

        this.updateCompareSelector();
    } catch (error) {
        console.error('Failed to load simulation list:', error);
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
            const meshData = await this.meshLoader.load();
            this.meshBounds = meshData.metadata.bounds;
            this.conversionFactor = meshData.metadata.mesh_conversion_factor;
            await this.viewer.reloadMesh(meshData);
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

        document.getElementById('v-min-result').textContent = Math.round(data.vMin);
        document.getElementById('v-max-result').textContent = Math.round(data.vMax);

        this.viewer.setVoltageRange(data.vMin, data.vMax);
        document.getElementById('colorbar-max').textContent = `${Math.round(data.vMax)} mV`;
        document.getElementById('colorbar-min').textContent = `${Math.round(data.vMin)} mV`;
        document.getElementById('colorbar-mid').textContent = `${Math.round((data.vMax + data.vMin) / 2)} mV`;

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
    if (!showPartition) {
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
