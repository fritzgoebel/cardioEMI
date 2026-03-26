// app-mesh.js - Mesh selection, conversion, and status management

App.prototype.setupMeshSelector = async function() {
    const selector = document.getElementById('mesh-selector');
    const convertBtn = document.getElementById('convert-mesh');
    const refreshBtn = document.getElementById('refresh-mesh-list');

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

    // Handle refresh button (visible in Karolina mode)
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

    if (this.runTarget === 'karolina') {
        // Karolina mode: fetch remote meshes and populate dropdown
        selector.innerHTML = '<option value="">Loading remote meshes...</option>';
        remoteConvertArea.innerHTML = '';
        remoteConvertArea.style.display = 'none';

        try {
            const families = await this.karolinaRunner.listRemoteMeshes();
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
                selector.innerHTML = '<option value="">No converted meshes on Karolina</option>';
            }

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
    }
};

App.prototype.onMeshSelected = async function(meshName) {
    if (this.runTarget === 'karolina') {
        await this.onKarolinaMeshSelected(meshName);
    } else {
        const response = await fetch('/api/meshes');
        const data = await response.json();
        const meshInfo = data.meshes.find(m => m.name === meshName);

        if (!meshInfo) return;

        if (meshInfo.converted) {
            await this.selectMesh(meshName);
        } else {
            this.updateMeshStatus(meshName, data.meshes);
        }
    }
};

App.prototype.onKarolinaMeshSelected = async function(meshName) {
    const statusEl = document.getElementById('mesh-status');
    const convertBtn = document.getElementById('convert-mesh');

    statusEl.style.display = 'block';
    convertBtn.style.display = 'none';

    // Step 1: Check if mesh data exists locally
    let localInfo = this.meshesInfo?.find(m => m.name === meshName);

    if (!localInfo) {
        // Need to download from Karolina
        statusEl.className = 'mesh-status pending';
        statusEl.textContent = `Downloading ${meshName} from Karolina...`;

        try {
            await this.karolinaRunner.downloadMeshData(meshName);
            statusEl.textContent = `Downloaded ${meshName}. Checking conversion...`;

            // Refresh local mesh info
            const resp = await fetch('/api/meshes');
            const data = await resp.json();
            this.meshesInfo = data.meshes;
            localInfo = data.meshes.find(m => m.name === meshName);
        } catch (e) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Download failed: ${e.message}`;
            return;
        }
    }

    if (!localInfo) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Mesh ${meshName} not found locally after download`;
        return;
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
    statusEl.textContent = `Converting ${outputPrefix} on Karolina...`;
    statusEl.style.display = 'block';
    outputEl.style.display = 'block';
    outputEl.textContent = '';

    try {
        await this.karolinaRunner.convertRemoteMesh(
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
