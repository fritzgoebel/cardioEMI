// app-weakscaling.js - Weak-scaling (3D-plus-cell) mesh generation tab

App.prototype.setupWeakScaling = function() {
    const tabStd = document.getElementById('tab-standard');
    const tabWs = document.getElementById('tab-weak-scaling');
    if (!tabStd || !tabWs) return;

    tabStd.addEventListener('click', () => this.switchMeshTab('standard'));
    tabWs.addEventListener('click', () => this.switchMeshTab('weak-scaling'));

    // Live estimate + validation on any parameter change
    ['ws-nx', 'ws-ny', 'ws-nz', 'ws-n', 'ws-L', 'ws-pad'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => this.updateWeakScalingEstimate());
    });

    // Selecting an existing config fills the inputs
    const existing = document.getElementById('ws-existing');
    if (existing) {
        existing.addEventListener('change', () => {
            const opt = existing.selectedOptions[0];
            if (!opt || !opt.dataset.nx) return;
            document.getElementById('ws-nx').value = opt.dataset.nx;
            document.getElementById('ws-ny').value = opt.dataset.ny;
            document.getElementById('ws-nz').value = opt.dataset.nz;
            document.getElementById('ws-n').value = opt.dataset.n;
            document.getElementById('ws-L').value = opt.dataset.l;
            document.getElementById('ws-pad').value = opt.dataset.pad;
            this.updateWeakScalingEstimate();
        });
    }

    const genBtn = document.getElementById('ws-generate');
    if (genBtn) genBtn.addEventListener('click', () => this.generateOrLoadWeakScaling());

    // Restore the last-active mesh tab (survives the reload done after a local
    // generation, so the UI stays on Weak scaling instead of resetting).
    if (sessionStorage.getItem('ws-mesh-tab') === 'weak-scaling') {
        this.switchMeshTab('weak-scaling');
    }

    this.refreshWeakScalingList();
    this.updateWeakScalingEstimate();
};

App.prototype.switchMeshTab = function(which) {
    const stdPanel = document.getElementById('standard-mesh-panel');
    const wsPanel = document.getElementById('weak-scaling-panel');
    const tabStd = document.getElementById('tab-standard');
    const tabWs = document.getElementById('tab-weak-scaling');

    const std = which === 'standard';
    stdPanel.style.display = std ? 'block' : 'none';
    wsPanel.style.display = std ? 'none' : 'block';
    tabStd.classList.toggle('active', std);
    tabWs.classList.toggle('active', !std);
    sessionStorage.setItem('ws-mesh-tab', which);

    if (!std) this.refreshWeakScalingList();
};

App.prototype.readWeakScalingParams = function() {
    return {
        nx: parseInt(document.getElementById('ws-nx').value, 10),
        ny: parseInt(document.getElementById('ws-ny').value, 10),
        nz: parseInt(document.getElementById('ws-nz').value, 10),
        n:  parseInt(document.getElementById('ws-n').value, 10),
        L:  parseFloat(document.getElementById('ws-L').value),
        pad: parseInt(document.getElementById('ws-pad').value, 10),
    };
};

App.prototype.weakScalingValidation = function(p) {
    if (![p.nx, p.ny, p.nz, p.n, p.pad].every(Number.isInteger) || Number.isNaN(p.L))
        return 'Enter valid numbers.';
    if (Math.min(p.nx, p.ny, p.nz) < 1) return 'Cubes per dimension must be >= 1.';
    if (p.n < 4 || p.n % 4 !== 0) return 'Resolution n must be a multiple of 4 (>= 4).';
    if (p.L <= 0) return 'Cube size L must be > 0.';
    if (p.pad < 0) return 'ECS padding must be >= 0.';
    return null;
};

App.prototype.updateWeakScalingEstimate = function() {
    const estEl = document.getElementById('ws-estimate');
    const genBtn = document.getElementById('ws-generate');
    if (!estEl) return;

    const p = this.readWeakScalingParams();
    const err = this.weakScalingValidation(p);
    if (err) {
        estEl.textContent = err;
        estEl.className = 'ws-estimate warn';
        if (genBtn) genBtn.disabled = true;
        return;
    }

    const cells = p.nx * p.ny * p.nz;
    const h = p.L / p.n;                              // voxel edge length
    const Gx = p.nx * p.n + 2 * p.pad;
    const Gy = p.ny * p.n + 2 * p.pad;
    const Gz = p.nz * p.n + 2 * p.pad;
    const tets = Gx * Gy * Gz * 6;
    const verts = (Gx + 1) * (Gy + 1) * (Gz + 1);
    const domain = [Gx * h, Gy * h, Gz * h];

    // Does this exact configuration already exist?
    const existing = document.getElementById('ws-existing');
    let ready = false;
    if (existing) {
        for (const opt of existing.options) {
            if (opt.dataset.nx &&
                +opt.dataset.nx === p.nx && +opt.dataset.ny === p.ny &&
                +opt.dataset.nz === p.nz && +opt.dataset.n === p.n &&
                +opt.dataset.pad === p.pad &&
                Math.abs(+opt.dataset.l - p.L) < 1e-9) {
                ready = opt.dataset.converted === 'true';
                break;
            }
        }
    }

    estEl.className = 'ws-estimate';
    estEl.innerHTML =
        `${cells.toLocaleString()} cells · ${tets.toLocaleString()} tets · ` +
        `${verts.toLocaleString()} vertices<br>` +
        `domain ${domain.map(d => d.toFixed(1)).join(' × ')} µm` +
        (p.pad ? ` · ${(p.pad * h).toFixed(2)} µm ECS shell` : '') +
        (ready ? ' · <span class="ws-ready">already generated</span>' : '');

    if (genBtn) {
        genBtn.disabled = false;
        genBtn.textContent = ready ? 'Load' : 'Generate';
    }
};

App.prototype.refreshWeakScalingList = async function() {
    const existing = document.getElementById('ws-existing');
    if (!existing) return;
    try {
        const resp = await fetch('/api/weak-scaling/list');
        const data = await resp.json();
        existing.innerHTML = '';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = data.meshes.length
            ? '— select a generated config —'
            : '— none generated yet —';
        existing.appendChild(placeholder);

        for (const m of data.meshes) {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.dataset.nx = m.nx; opt.dataset.ny = m.ny; opt.dataset.nz = m.nz;
            opt.dataset.n = m.n; opt.dataset.l = m.L; opt.dataset.pad = m.pad;
            opt.dataset.converted = m.converted;
            opt.textContent = `${m.nx}×${m.ny}×${m.nz}  (n=${m.n}, L=${m.L}, pad=${m.pad})` +
                (m.converted ? '' : ' — needs viz convert');
            existing.appendChild(opt);
        }
        this.updateWeakScalingEstimate();
    } catch (e) {
        console.error('Failed to load weak-scaling mesh list:', e);
    }
};

App.prototype.generateOrLoadWeakScaling = async function() {
    const p = this.readWeakScalingParams();
    const err = this.weakScalingValidation(p);
    const statusEl = document.getElementById('ws-status');

    if (err) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = err;
        statusEl.style.display = 'block';
        return;
    }

    // Karolina mode: generate on the cluster so the mesh lands in the remote
    // data/ dir and shows up under remote meshes (rather than only locally).
    if (this.runTarget === 'karolina') {
        return this.generateWeakScalingRemote(p);
    }

    const genBtn = document.getElementById('ws-generate');
    const progressBar = document.getElementById('ws-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');

    genBtn.disabled = true;
    statusEl.style.display = 'none';
    progressBar.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting...';

    try {
        const response = await fetch('/api/weak-scaling/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(p),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let meshName = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();  // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.substring(6));
                if (data.type === 'progress') {
                    progressFill.style.width = `${data.percent}%`;
                    progressText.textContent = data.message || `${data.percent}%`;
                } else if (data.type === 'complete') {
                    meshName = data.name;
                    progressFill.style.width = '100%';
                    progressText.textContent = 'Complete!';
                } else if (data.type === 'error') {
                    throw new Error(data.message);
                }
            }
        }

        if (meshName) {
            await this.refreshWeakScalingList();
            statusEl.className = 'mesh-status converted';
            statusEl.textContent = `Ready: ${meshName} — reloading...`;
            statusEl.style.display = 'block';
            setTimeout(() => this.selectMesh(meshName), 400);
        }
    } catch (error) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Failed: ${error.message}`;
        statusEl.style.display = 'block';
    } finally {
        genBtn.disabled = false;
        setTimeout(() => { progressBar.style.display = 'none'; }, 1500);
    }
};

// Generate a weak-scaling mesh on Karolina, then switch to the Standard tab and
// select it from the (refreshed) remote mesh list.
App.prototype.generateWeakScalingRemote = async function(p) {
    const statusEl = document.getElementById('ws-status');
    const genBtn = document.getElementById('ws-generate');
    const progressBar = document.getElementById('ws-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');

    genBtn.disabled = true;
    statusEl.style.display = 'none';
    progressBar.style.display = 'block';
    progressFill.style.width = '40%';
    progressText.textContent = 'Generating on Karolina...';

    try {
        const { name } = await this.karolinaRunner.generateWeakScaling(p, (text) => {
            const last = text.trim().split('\n').filter(Boolean).pop();
            if (last) progressText.textContent = last.slice(0, 80);
        });

        if (!name) throw new Error('Generation finished without a mesh name');

        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';

        // Register the new mesh in the (hidden) Standard selector and select it,
        // without leaving the Weak scaling tab.
        await this.refreshMeshList();
        const selector = document.getElementById('mesh-selector');
        if (selector) {
            selector.value = name;
            await this.onMeshSelected(name);
        }

        statusEl.className = 'mesh-status converted';
        statusEl.textContent = `Generated on Karolina: ${name}`;
        statusEl.style.display = 'block';
    } catch (error) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Failed: ${error.message}`;
        statusEl.style.display = 'block';
    } finally {
        genBtn.disabled = false;
        setTimeout(() => { progressBar.style.display = 'none'; }, 1500);
    }
};
