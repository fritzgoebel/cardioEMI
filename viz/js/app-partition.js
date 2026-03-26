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
    this.updateInterfaceHighlight();
};

App.prototype.updateInterfaceHighlight = function() {
    if (!this.interfaceData || !this.showInterfaces) {
        this.viewer.clearInterfaceHighlight();
        return;
    }

    const interfaceMap = new Map();
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
        const response = await fetch('/api/interfaces');
        const data = await response.json();

        if (data.interfaces && Object.keys(data.interfaces).length > 0) {
            this.interfaceData = {};
            for (const [rank, interfaces] of Object.entries(data.interfaces)) {
                this.interfaceData[parseInt(rank)] = interfaces;
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
            this.updateInterfaceHighlight();
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
    document.getElementById('explosion-slider').value = 0;
    document.getElementById('explosion-value').textContent = '0';
    this.showInterfaces = false;
    this.interfaceData = null;
    this.interfaceDofTypes = null;
    this.viewer.setInterfaceDofTypes(null);
};
