// app-partition.js - MPI ranks, interfaces, and partition visualization

App.prototype.selectAllRanks = function(selectAll) {
    const checkboxes = document.querySelectorAll('#rank-checkboxes input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = selectAll;
    });
    this.onRankSelectionChange();
};

App.prototype.onRankSelectionChange = function() {
    const checkboxes = document.querySelectorAll('#rank-checkboxes input[type="checkbox"]');
    this.visibleRanks = new Set();
    checkboxes.forEach(cb => {
        if (cb.checked) {
            this.visibleRanks.add(parseInt(cb.dataset.rank));
        }
    });

    this.viewer.setVisibleRanks(this.visibleRanks);
    this.refreshInterfaceView();
};

// Decide whether the per-interface inspector panel is shown (exactly one
// subdomain selected + interfaces on), (re)build it when the selected rank
// changes, then refresh the 3D highlight.
App.prototype.refreshInterfaceView = function() {
    const panel = document.getElementById('interface-list-panel');
    const selected = [...this.visibleRanks];

    if (!this.interfaceData || !this.showInterfaces || selected.length !== 1) {
        if (panel) panel.style.display = 'none';
        this._panelRank = null;
        this.updateInterfaceHighlight();
        return;
    }

    const rank = selected[0];
    if (this._panelRank !== rank) {
        this.buildInterfacePanel(rank);
        this._panelRank = rank;
    }
    if (panel) panel.style.display = 'block';
    this.updateInterfaceHighlight();
};

// Build the three-column (Vertices / Edges / Faces) list of this rank's
// interfaces, one toggle row per interface, labeled by the ranks sharing it.
App.prototype.buildInterfacePanel = function(rank) {
    const cols = {
        vertex: document.getElementById('interface-col-vertex'),
        edge: document.getElementById('interface-col-edge'),
        face: document.getElementById('interface-col-face'),
    };
    Object.values(cols).forEach(c => { if (c) c.innerHTML = ''; });
    document.getElementById('interface-list-rank').textContent = rank;

    // All interfaces start enabled when the panel rank changes.
    this.enabledInterfaces = new Set();
    const info = (this.interfaceInfo && this.interfaceInfo[rank]) || [];
    const counts = { vertex: 0, edge: 0, face: 0 };

    info.forEach((iface, idx) => {
        this.enabledInterfaces.add(idx);
        const type = iface.type || 'face';
        const col = cols[type];
        if (!col) return;
        counts[type]++;
        const color = this.viewer.interfaceToColor(idx);
        const rgb = `rgb(${Math.round(color.r*255)}, ${Math.round(color.g*255)}, ${Math.round(color.b*255)})`;
        const label = (iface.ranks && iface.ranks.length)
            ? `{${iface.ranks.join(',')}}` : '{?}';
        const item = document.createElement('label');
        item.className = 'interface-item';
        item.title = `${type} shared by ranks ${label}`;
        item.innerHTML = `
            <input type="checkbox" data-idx="${idx}" checked>
            <span class="interface-swatch" style="background-color: ${rgb}"></span>
            <span>${label}</span>
        `;
        item.querySelector('input').addEventListener('change', (e) => {
            const i = parseInt(e.target.dataset.idx);
            if (e.target.checked) this.enabledInterfaces.add(i);
            else this.enabledInterfaces.delete(i);
            this.updateInterfaceHighlight();
        });
        col.appendChild(item);
    });

    for (const [type, col] of Object.entries(cols)) {
        if (col && counts[type] === 0) {
            col.innerHTML = '<span class="interface-item-empty">none</span>';
        }
    }
};

// Toggle every interface row in the panel on/off.
App.prototype.setAllInterfaces = function(enable) {
    const boxes = document.querySelectorAll('#interface-list-columns input[type="checkbox"]');
    this.enabledInterfaces = new Set();
    boxes.forEach(b => {
        b.checked = enable;
        if (enable) this.enabledInterfaces.add(parseInt(b.dataset.idx));
    });
    this.updateInterfaceHighlight();
};

App.prototype.updateInterfaceHighlight = function() {
    if (!this.interfaceData || !this.showInterfaces) {
        this.viewer.clearInterfaceHighlight();
        return;
    }

    const interfaceMap = new Map();
    const selected = [...this.visibleRanks];

    // Single-subdomain mode: highlight only the interfaces enabled in the panel
    // (still respecting the column-level Vertices/Edges/Faces master toggles).
    // Each interface keeps its own colour index so it matches its panel swatch.
    if (selected.length === 1 && this.interfaceInfo && this._panelRank === selected[0]) {
        const rank = selected[0];
        const rankInterfaces = this.interfaceData[rank] || [];
        const info = this.interfaceInfo[rank] || [];
        rankInterfaces.forEach((interfaceList, idx) => {
            if (!this.enabledInterfaces || !this.enabledInterfaces.has(idx)) return;
            const type = (info[idx] && info[idx].type) || 'face';
            if (type === 'vertex' && !this.showInterfaceVertices) return;
            if (type === 'edge' && !this.showInterfaceEdges) return;
            if (type === 'face' && !this.showInterfaceFaces) return;
            for (const dof of interfaceList) {
                if (!interfaceMap.has(dof)) interfaceMap.set(dof, idx);
            }
        });
        this.viewer.setHighlightedInterfaceDofs(interfaceMap);
        return;
    }

    // Multi-subdomain (or no panel) mode: aggregate all interfaces of the
    // visible ranks, filtered by DOF type.
    let globalInterfaceIdx = 0;
    let skippedByType = { vertex: 0, edge: 0, face: 0 };

    for (const rank of this.visibleRanks) {
        const rankInterfaces = this.interfaceData[rank];
        if (rankInterfaces) {
            for (const interfaceList of rankInterfaces) {
                for (const dof of interfaceList) {
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
};

App.prototype.loadInterfaceData = async function() {
    try {
        const simName = this.resultsVizDir || this.selectedSimulation;
        const url = simName
            ? `/api/interfaces?sim=${encodeURIComponent(simName)}`
            : '/api/interfaces';
        const response = await fetch(url);
        const data = await response.json();

        if (data.interfaces && Object.keys(data.interfaces).length > 0) {
            this.interfaceData = {};
            for (const [rank, interfaces] of Object.entries(data.interfaces)) {
                this.interfaceData[parseInt(rank)] = interfaces;
            }
            this.interfaceInfo = {};
            if (data.interfaceInfo) {
                for (const [rank, info] of Object.entries(data.interfaceInfo)) {
                    this.interfaceInfo[parseInt(rank)] = info;
                }
            }
            if (data.dofTypes) {
                this.interfaceDofTypes = {};
                for (const [dof, dofType] of Object.entries(data.dofTypes)) {
                    this.interfaceDofTypes[parseInt(dof)] = dofType;
                }
                this.viewer.setInterfaceDofTypes(this.interfaceDofTypes);
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
    this.interfaceInfo = null;
    this.interfaceDofTypes = null;
    this.viewer.setInterfaceDofTypes(null);
    return false;
};

App.prototype.onPartitionToggle = function(showPartition) {
    const colorbar = document.getElementById('colorbar');
    const rankLegend = document.getElementById('rank-legend');
    const partitionControls = document.getElementById('partition-controls');
    const rankSelector = document.getElementById('rank-selector');

    if (showPartition && this.ranksData) {
        this.viewer.updateRankColors(this.ranksData);
        colorbar.style.display = 'none';
        rankLegend.style.display = 'flex';
        partitionControls.style.display = 'flex';
        rankSelector.style.display = 'flex';

        this.viewer.setVisibleRanks(this.visibleRanks);

        if (document.getElementById('show-ecs').checked && this.ecsRanksData) {
            this.viewer.updateEcsRankColors(this.ecsRanksData);
        }

        if (this.cutRanksData) {
            this.viewer.updateCutRankColors(this.cutRanksData);
            this.viewer.setCutVisible(true);
        }

        if (this.showInterfaces) {
            this.refreshInterfaceView();
        }
    } else if (this.resultsVizDir) {
        this.viewer.restoreFullMesh();

        const timeSlider = document.getElementById('result-time');
        const idx = parseInt(timeSlider.value);
        const voltages = this._voltageCache && this._voltageCache[idx];
        if (voltages) {
            this.viewer.updateVoltageColors(voltages);
        }
        colorbar.style.display = 'flex';
        rankLegend.style.display = 'none';
        partitionControls.style.display = 'none';
        rankSelector.style.display = 'none';

        this.viewer.setEcsVisible(false);
        this.viewer.setCutVisible(false);
        this.viewer.setExplosionFactor(0);
        this.viewer.resetEcsColors();
        this.viewer.clearInterfaceHighlight();
        document.getElementById('show-ecs').checked = false;
        document.getElementById('show-interfaces').checked = false;
        document.getElementById('interface-type-controls').style.display = 'none';
        document.getElementById('interface-list-panel').style.display = 'none';
        this._panelRank = null;
        document.getElementById('explosion-slider').value = 0;
        document.getElementById('explosion-value').textContent = '0';
        this.showInterfaces = false;
    }
};

App.prototype.showPartitionOption = function(numRanks) {
    const label = document.getElementById('show-partition-label');
    const legendItems = document.getElementById('rank-legend-items');
    const rankCheckboxes = document.getElementById('rank-checkboxes');

    label.style.display = 'flex';

    this.visibleRanks = new Set();
    for (let i = 0; i < numRanks; i++) {
        this.visibleRanks.add(i);
    }

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

    this.loadInterfaceData();
};

App.prototype.hidePartitionOption = function() {
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

    document.getElementById('show-ecs').checked = false;
    document.getElementById('show-interfaces').checked = false;
    document.getElementById('interface-type-controls').style.display = 'none';
    document.getElementById('interface-list-panel').style.display = 'none';
    document.getElementById('explosion-slider').value = 0;
    document.getElementById('explosion-value').textContent = '0';
    this.showInterfaces = false;
    this.interfaceData = null;
    this.interfaceInfo = null;
    this.interfaceDofTypes = null;
    this._panelRank = null;
    this.viewer.setInterfaceDofTypes(null);
};
