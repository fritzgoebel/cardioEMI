// app-mesh.js - Mesh selection, conversion, and status management

App.prototype.setupMeshSelector = async function() {
    const selector = document.getElementById('mesh-selector');
    const convertBtn = document.getElementById('convert-mesh');
    const refreshBtn = document.getElementById('refresh-mesh-list');

    this.setupMeshBatch();

    // Initial mesh list load (always local to get current mesh for viewer)
    const savedTarget = this.runTarget;
    this.runTarget = 'local';
    await this.refreshMeshList();
    this.runTarget = savedTarget;

    // Handle mesh selection change
    selector.addEventListener('change', async () => {
        const meshName = selector.value;
        await this.onMeshSelected(meshName);
    });

    // Handle convert button (for local viz conversion)
    convertBtn.addEventListener('click', async () => {
        const meshName = selector.value;
        await this.convertMesh(meshName);
    });

    // Handle refresh button (visible in remote mode)
    refreshBtn.addEventListener('click', () => {
        this.refreshMeshList();
    });
};

App.prototype.refreshMeshList = async function() {
    const selector = document.getElementById('mesh-selector');
    const remoteConvertArea = document.getElementById('remote-convert-area');

    // Always fetch local meshes
    let localMeshes = [];
    let currentMesh = null;
    let currentConfig = null;
    try {
        const response = await fetch('/api/meshes');
        const data = await response.json();
        localMeshes = data.meshes;
        currentMesh = data.current;
        currentConfig = data.currentConfig;
        this.meshesInfo = data.meshes;
    } catch (error) {
        console.error('Failed to load local mesh list:', error);
        selector.innerHTML = '<option value="">Error loading meshes</option>';
        return;
    }

    // Cluster targets pick meshes from the batch checklist (app-mesh-batch.js);
    // the dropdown stays as the hidden holder of the focused mesh.
    selector.style.display = this.isRemote() ? 'none' : '';
    document.getElementById('mesh-batch').style.display = this.isRemote() ? 'block' : 'none';

    if (this.isRemote()) {
        // Remote mode: fetch remote meshes and populate dropdown
        selector.innerHTML = '<option value="">Loading remote meshes...</option>';
        remoteConvertArea.innerHTML = '';
        remoteConvertArea.style.display = 'none';

        try {
            const families = await this.clusterRunner.listRemoteMeshes();
            const localNames = new Set(localMeshes.map(m => m.name));

            selector.innerHTML = '';
            let hasOptions = false;

            // Build list of converted remote meshes for the dropdown
            // and unconverted ones for convert buttons
            const unconverted = [];

            for (const family of families) {
                for (const mesh of family.meshes) {
                    // Plain converted mesh
                    if (mesh.converted) {
                        const opt = document.createElement('option');
                        opt.value = mesh.name;
                        const isLocal = localNames.has(mesh.name);
                        const localInfo = localMeshes.find(m => m.name === mesh.name);
                        const vizReady = localInfo && localInfo.converted;
                        let label = mesh.name;
                        if (vizReady) label += ' (ready)';
                        else if (isLocal) label += ' (local, needs viz convert)';
                        else label += ' (remote)';
                        opt.textContent = label;
                        if (mesh.name === currentMesh) opt.selected = true;
                        selector.appendChild(opt);
                        hasOptions = true;
                    } else {
                        unconverted.push({ family: family.family, mesh, color: false });
                    }

                    // Colored variant
                    if (mesh.converted_colored) {
                        const colorName = mesh.name + '_colored';
                        const opt = document.createElement('option');
                        opt.value = colorName;
                        const isLocal = localNames.has(colorName);
                        const localInfo = localMeshes.find(m => m.name === colorName);
                        const vizReady = localInfo && localInfo.converted;
                        let label = colorName;
                        if (vizReady) label += ' (ready)';
                        else if (isLocal) label += ' (local, needs viz convert)';
                        else label += ' (remote)';
                        opt.textContent = label;
                        if (colorName === currentMesh) opt.selected = true;
                        selector.appendChild(opt);
                        hasOptions = true;
                    } else {
                        unconverted.push({ family: family.family, mesh, color: true });
                    }
                }
            }

            if (!hasOptions) {
                selector.innerHTML = '<option value="">No converted meshes on the cluster</option>';
            }
            this.setBatchMeshes([...selector.options]
                .filter(o => o.value)
                .map(o => ({ name: o.value, label: (o.textContent.match(/\(([^)]*)\)$/) || [])[1] || '' })));

            // Show unconverted meshes as dropdown with convert button
            if (unconverted.length > 0) {
                remoteConvertArea.style.display = 'block';
                remoteConvertArea.innerHTML = '';
                const row = document.createElement('div');
                row.style.cssText = 'display:flex; align-items:center; gap:6px; font-size:0.85em;';
                const label = document.createElement('label');
                label.textContent = 'Unconverted:';
                label.style.color = '#888';
                row.appendChild(label);
                const sel = document.createElement('select');
                sel.className = 'mesh-dropdown';
                sel.id = 'unconverted-mesh-selector';
                sel.style.flex = '1';
                for (const item of unconverted) {
                    const opt = document.createElement('option');
                    const displayName = item.color ? item.mesh.name + '_colored' : item.mesh.name;
                    opt.value = JSON.stringify({ family: item.family, mesh: item.mesh, color: item.color });
                    opt.textContent = displayName;
                    sel.appendChild(opt);
                }
                row.appendChild(sel);
                const btn = document.createElement('button');
                btn.textContent = 'Convert';
                btn.className = 'btn-small';
                btn.addEventListener('click', () => {
                    const val = JSON.parse(sel.value);
                    this.convertRemoteMeshAndRefresh(val.family, val.mesh, val.color);
                });
                row.appendChild(btn);
                remoteConvertArea.appendChild(row);
            }
        } catch (e) {
            selector.innerHTML = '<option value="">Failed to load remote meshes</option>';
            this.setBatchMeshes([]);
            console.error('Failed to load remote meshes:', e);
        }
    } else {
        // Local mode: populate with local meshes
        remoteConvertArea.style.display = 'none';
        selector.innerHTML = '';
        localMeshes.forEach(mesh => {
            const option = document.createElement('option');
            option.value = mesh.name;
            option.textContent = mesh.name + (mesh.converted ? '' : ' (not converted)');
            if (mesh.name === currentMesh) {
                option.selected = true;
            }
            selector.appendChild(option);
        });
    }

    // Set current mesh in loader
    if (currentMesh) {
        this.meshLoader.setMesh(currentMesh);
    }

    // Set current config file
    if (currentConfig) {
        this.configManager.setConfigFile(currentConfig);
        this.simulationRunner.setConfigFile(currentConfig);
    }

    // Update status for currently selected mesh
    const selected = selector.value;
    if (selected && this.runTarget === 'local') {
        this.updateMeshStatus(selected, localMeshes);
        const selectedInfo = localMeshes.find(m => m.name === selected);
        this.updateMeshTagInfo(selectedInfo?.numTags, selectedInfo?.numComponents, selectedInfo?.numOriginalTags);
    }
};

App.prototype.onMeshSelected = async function(meshName) {
    if (this.isRemote()) {
        await this.onRemoteMeshSelected(meshName);
    } else {
        const response = await fetch('/api/meshes');
        const data = await response.json();
        const meshInfo = data.meshes.find(m => m.name === meshName);

        if (!meshInfo) return;

        this.updateMeshTagInfo(meshInfo.numTags, meshInfo.numComponents, meshInfo.numOriginalTags);

        if (meshInfo.converted) {
            await this.selectMesh(meshName);
        } else {
            this.updateMeshStatus(meshName, data.meshes);
        }
    }
};

// Number of volume tags / mesh-partitioning units for the selected mesh.
// numOriginalTags/numComponents come from the original uncolored mesh (only
// known for a `_colored` mesh) - see get_mesh_tag_counts server-side:
//   numOriginalTags -> target rank count for component_granularity "tag"
//   numComponents   -> target rank count for component_granularity "component" (default)
App.prototype.updateMeshTagInfo = function(numTags, numComponents, numOriginalTags) {
    this.currentMeshNumComponents = numComponents || null;
    this.currentMeshNumOriginalTags = numOriginalTags || null;

    const infoEl = document.getElementById('mesh-tags-info');
    if (numTags == null) {
        infoEl.style.display = 'none';
    } else {
        infoEl.style.display = 'block';
        infoEl.className = 'mesh-status';
        infoEl.textContent = numOriginalTags
            ? `Tags: ${numOriginalTags} (${numComponents} ECS+cell pairs)`
            : `Tags: ${numTags}`;
    }

    const canMatch = !!(numComponents || numOriginalTags);
    const matchBtn = document.getElementById('match-ranks-to-tags');
    if (matchBtn) matchBtn.style.display = canMatch ? 'inline-block' : 'none';

    this.updateMatchButtonLabels();
};

// Rank target for the current mesh + component_granularity choice - or null
// if 'Tag based' partitioning isn't applicable to the selected mesh.
App.prototype.getPartitionTargetCount = function() {
    return this.componentGranularity === 'tag'
        ? this.currentMeshNumOriginalTags
        : this.currentMeshNumComponents;
};

// Keep the "= N" match-button labels showing the count they'll actually set,
// since that depends on both the selected mesh and component_granularity.
App.prototype.updateMatchButtonLabels = function() {
    const target = this.getPartitionTargetCount();
    const label = target ? `= ${target}` : '= components';
    const el = document.getElementById('match-ranks-to-tags');
    if (el) el.textContent = label;
};

App.prototype.onRemoteMeshSelected = async function(meshName) {
    const statusEl = document.getElementById('mesh-status');
    const convertBtn = document.getElementById('convert-mesh');

    statusEl.style.display = 'block';
    convertBtn.style.display = 'none';

    // Step 1: Check if mesh data exists locally
    let localInfo = this.meshesInfo?.find(m => m.name === meshName);

    if (localInfo) {
        this.updateMeshTagInfo(localInfo.numTags, localInfo.numComponents, localInfo.numOriginalTags);
    }

    if (!localInfo) {
        // Mesh not local — fetch metadata (bounds) from the cluster instead of downloading
        statusEl.className = 'mesh-status pending';
        statusEl.textContent = `Fetching bounds for ${meshName} from ${this.clusterLabel()}...`;

        try {
            const metadata = await this.clusterRunner.fetchMeshMetadata(meshName);
            this.applyRemoteMeshMetadata(meshName, metadata);
            statusEl.className = 'mesh-status pending';
            statusEl.textContent = `Remote mesh: ${meshName} — building membrane preview on ${this.clusterLabel()}…`;
            statusEl.style.display = 'block';
            await this.loadRemoteMeshPreview(meshName);
            return;
        } catch (e) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Failed to fetch metadata: ${e.message}`;
            return;
        }
    }

    // Step 2: Check if viz conversion exists
    if (!localInfo.converted) {
        statusEl.className = 'mesh-status pending';
        statusEl.textContent = `Converting ${meshName} for visualization...`;

        try {
            await this.convertMesh(meshName);
            return;
        } catch (e) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Conversion failed: ${e.message}`;
            return;
        }
    }

    // Step 3: Already local and converted - just select it
    await this.selectMesh(meshName);
};

// Show a remote-only mesh's membranes instead of just its bounding box. The
// preview is built next to the mesh, coarsened to a triangle budget if needed,
// and cached on both sides (see /api/cluster/<id>/meshes/preview).
App.prototype.loadRemoteMeshPreview = async function(meshName) {
    const statusEl = document.getElementById('mesh-status');
    const seq = this._meshPreviewSeq = (this._meshPreviewSeq || 0) + 1;
    try {
        const res = await this.clusterRunner.fetchMeshPreview(meshName);
        const meshData = await this.meshLoader.loadFrom(res.path);
        // The user may have picked another mesh meanwhile.
        if (seq !== this._meshPreviewSeq || this.meshLoader.currentMesh !== meshName) return;
        await this.viewer.reloadMesh(meshData);
        this.viewer.hideBoundsOutline();
        this.updateBoundingBoxVisualization();
        const p = res.metadata.preview || {};
        const res_note = p.cluster_size
            ? `coarsened to ${res.metadata.facet_count.toLocaleString()} of ${p.full_facets.toLocaleString()} triangles (${p.cluster_size.toFixed(1)} µm grid)`
            : `${res.metadata.facet_count.toLocaleString()} triangles, full resolution`;
        statusEl.className = 'mesh-status converted';
        statusEl.textContent = `Remote mesh: ${meshName} — membrane preview, ${res_note}`;
    } catch (e) {
        if (seq !== this._meshPreviewSeq) return;
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Remote mesh: ${meshName} — bounds only (preview failed: ${e.message})`;
    }
    statusEl.style.display = 'block';
};

App.prototype.applyRemoteMeshMetadata = function(meshName, metadata) {
    // Store bounds and conversion factor from remote metadata
    this.meshBounds = metadata.bounds;
    this.conversionFactor = metadata.mesh_conversion_factor;
    this.remoteMeshName = meshName;
    this.updateMeshTagInfo(metadata.num_tags, metadata.num_components, metadata.num_original_tags);

    // Track this as the current mesh (used by conditions snapshot, v_init, etc.)
    this.meshLoader.setMesh(meshName);

    // Update config manager with the mesh name for YAML generation
    const configFile = `input_${meshName}.yml`;
    this.configManager.setConfigFile(configFile);
    this.simulationRunner.setConfigFile(configFile);

    // Re-initialize sliders with new bounds
    this.setupSliders();

    // Restore per-mesh IC/scar config from localStorage
    this.loadMeshConfig();

    // Update 3D viewer: clear old mesh, show bounds outline
    if (this.viewer) {
        this.viewer.clearAllMeshes();
        this.viewer.showBoundsOutline(metadata.bounds);
        this.updateBoundingBoxVisualization();
    }

    this.updateVinitExpression();
    this.updateColorbar();
};

App.prototype.updateMeshStatus = function(meshName, meshes) {
    const statusEl = document.getElementById('mesh-status');
    const convertBtn = document.getElementById('convert-mesh');
    const meshInfo = meshes.find(m => m.name === meshName);

    if (!meshInfo) return;

    if (meshInfo.converted) {
        statusEl.className = 'mesh-status converted';
        statusEl.textContent = 'Ready to use';
        statusEl.style.display = 'block';
        convertBtn.style.display = 'none';
    } else {
        statusEl.className = 'mesh-status pending';
        statusEl.textContent = 'Mesh needs conversion before use';
        statusEl.style.display = 'block';
        convertBtn.style.display = 'block';
    }
};

App.prototype.convertMesh = async function(meshName) {
    const convertBtn = document.getElementById('convert-mesh');
    const progressBar = document.getElementById('conversion-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');
    const statusEl = document.getElementById('mesh-status');

    convertBtn.disabled = true;
    progressBar.style.display = 'block';
    statusEl.style.display = 'none';

    try {
        const response = await fetch('/api/meshes/convert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mesh: meshName })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));

                    if (data.type === 'progress') {
                        progressFill.style.width = `${data.percent}%`;
                        progressText.textContent = data.message || `${data.percent}%`;
                    } else if (data.type === 'complete') {
                        progressFill.style.width = '100%';
                        progressText.textContent = 'Complete!';
                        setTimeout(() => this.selectMesh(meshName), 500);
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                }
            }
        }
    } catch (error) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Conversion failed: ${error.message}`;
        statusEl.style.display = 'block';
    } finally {
        convertBtn.disabled = false;
        setTimeout(() => {
            progressBar.style.display = 'none';
        }, 1000);
    }
};

App.prototype.selectMesh = async function(meshName) {
    const statusEl = document.getElementById('mesh-status');
    const convertBtn = document.getElementById('convert-mesh');

    const meshInfo = this.meshesInfo?.find(m => m.name === meshName);
    const configFile = meshInfo?.configFile;

    try {
        const response = await fetch('/api/meshes/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mesh: meshName, configFile: configFile })
        });

        const data = await response.json();

        if (data.success) {
            statusEl.className = 'mesh-status converted';
            statusEl.textContent = `Mesh selected (config: ${data.configFile}) - reloading...`;
            statusEl.style.display = 'block';
            convertBtn.style.display = 'none';
            window.location.reload();
        } else {
            throw new Error(data.error || 'Failed to select mesh');
        }
    } catch (error) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Selection failed: ${error.message}`;
        statusEl.style.display = 'block';
    }
};

App.prototype.convertRemoteMeshAndRefresh = async function(family, mesh, color) {
    const statusEl = document.getElementById('remote-mesh-convert-status');
    const outputEl = document.getElementById('remote-mesh-convert-output');
    const outputPrefix = color ? mesh.name + '_colored' : mesh.name;

    statusEl.className = 'mesh-status pending';
    statusEl.textContent = `Converting ${outputPrefix} on ${this.clusterLabel()}...`;
    statusEl.style.display = 'block';
    outputEl.style.display = 'block';
    outputEl.textContent = '';

    try {
        await this.clusterRunner.convertRemoteMesh(
            family, mesh.pts, mesh.elem, outputPrefix, color,
            (text) => {
                outputEl.textContent += text;
                outputEl.scrollTop = outputEl.scrollHeight;
            }
        );

        statusEl.className = 'mesh-status converted';
        statusEl.textContent = `Conversion of ${outputPrefix} complete!`;
        await this.refreshMeshList();
    } catch (e) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Conversion failed: ${e.message}`;
    }
};
