// app-weakscaling.js - Weak-scaling (3D-plus-cell) mesh generation tab

App.prototype.setupWeakScaling = function() {
    const tabStd = document.getElementById('tab-standard');
    const tabWs = document.getElementById('tab-weak-scaling');
    if (!tabStd || !tabWs) return;

    tabStd.addEventListener('click', () => this.switchMeshTab('standard'));
    tabWs.addEventListener('click', () => this.switchMeshTab('weak-scaling'));

    // Live estimate + validation on any parameter change
    ['ws-nx', 'ws-ny', 'ws-nz', 'ws-n', 'ws-L', 'ws-pad', 'ws-ax', 'ws-slabs',
     'ws-latx', 'ws-latxz', 'ws-angle', 'ws-latr'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => this.updateWeakScalingEstimate());
    });
    const shapeEl = document.getElementById('ws-shape');
    if (shapeEl) shapeEl.addEventListener('change', () => this.updateWeakScalingEstimate());
    if (this.initWeakScalingPreview) {
        this.initWeakScalingPreview();
        const wsTab = sessionStorage.getItem('ws-mesh-tab') === 'weak-scaling';
        this.showWeakScalingPreview(wsTab);
    }

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
            if (document.getElementById('ws-shape'))
                document.getElementById('ws-shape').value = opt.dataset.shape;
            if (document.getElementById('ws-ax'))
                document.getElementById('ws-ax').value = opt.dataset.ax;
            if (document.getElementById('ws-slabs'))
                document.getElementById('ws-slabs').value = opt.dataset.slabs;
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

    if (this.showWeakScalingPreview) this.showWeakScalingPreview(!std);
    if (!std) this.refreshWeakScalingList();
};

// Common denominator and numerators for the two lateral shifts -- the JS twin
// of CellShape.resolve. Each direction's shift is Lx - 2d, and both must be
// expressed over ONE denominator, because a box's x offset is
// (J*step_y + K*step_z)/slabs of the box length.
App.prototype.weakScalingShift = function(ax, dY, dZ, maxSlabs) {
    const want = [(ax - 2 * dY) / ax, (ax - 2 * dZ) / ax];
    let best = { slabs: 2, step_y: 1, step_z: 1 }, bestErr = Infinity;
    for (let q = 2; q <= maxSlabs; q++) {
        const ps = want.map(w => Math.min(q - 1, Math.max(1, Math.round(w * q))));
        const err = Math.max(...want.map((w, i) => Math.abs(w - ps[i] / q)));
        if (err < bestErr - 1e-12) {
            bestErr = err;
            best = { slabs: q, step_y: ps[0], step_z: ps[1] };
        }
    }
    return best;
};

// The interface inset must clear the x face by the connector's own radius, and
// a stub must not run into its opposite number: it leans `leanX` in x on its way
// to the body, so d + leanX + latR < ax - d. Both the width and the angle move
// that ceiling, which is why the slider is held as a fraction of the range
// rather than an absolute distance -- changing either keeps it valid.
App.prototype.weakScalingInset = function(ax, latR, leanX, pct) {
    const eps = 0.04;
    const lo = latR + eps;
    const hi = 0.5 * (ax - leanX - latR) - eps;
    if (hi <= lo) return 0.5 * (lo + hi);
    return lo + (hi - lo) * (pct / 100);
};

// x travel of a stub between its face and the body axis.
App.prototype.weakScalingLeanX = function(deg) {
    return 0.5 / Math.tan(Math.min(89.9, Math.max(5, deg)) * Math.PI / 180);
};

App.prototype.readWeakScalingParams = function() {
    return {
        nx: parseInt(document.getElementById('ws-nx').value, 10),
        ny: parseInt(document.getElementById('ws-ny').value, 10),
        nz: parseInt(document.getElementById('ws-nz').value, 10),
        n:  parseInt(document.getElementById('ws-n').value, 10),
        L:  parseFloat(document.getElementById('ws-L').value),
        pad: parseInt(document.getElementById('ws-pad').value, 10),
        shape: (document.getElementById('ws-shape') || {}).value || 'cell',
        ax: parseInt((document.getElementById('ws-ax') || {}).value, 10) || 1,
        angleDeg: parseFloat((document.getElementById('ws-angle') || {}).value) || 63,
        latRPct: parseFloat((document.getElementById('ws-latr') || {}).value) || 68,
        latPct: parseFloat((document.getElementById('ws-latx') || {}).value),
        latPctZ: parseFloat((document.getElementById('ws-latxz') || {}).value),
    };
};

App.prototype.weakScalingValidation = function(p) {
    if (![p.nx, p.ny, p.nz, p.n, p.pad].every(Number.isInteger) || Number.isNaN(p.L))
        return 'Enter valid numbers.';
    if (Math.min(p.nx, p.ny, p.nz) < 1) return 'Boxes per dimension must be >= 1.';
    if (p.L <= 0) return 'Box size L must be > 0.';
    if (p.pad < 0) return 'ECS padding must be >= 0.';
    if (p.shape === 'plus') {
        if (p.n < 4 || p.n % 4 !== 0) return 'The plus shape needs n a multiple of 4 (>= 4).';
    } else {
        if (p.n < 4) return 'The cell shape needs n >= 4 elements per L.';
        // Both the near-face and far-face connectors must fit along x.
        if (!Number.isInteger(p.ax) || p.ax < 2) return 'The cell shape needs an x aspect >= 2.';
        if (p.slabs && p.slabs < 2) return 'Slabs per box must be ≥ 2 (the lattice shift is Lx/slabs).';
    }
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
        this._wsLastParams = null;      // never generate from a stale resolve
        return;
    }

    const cells = p.nx * p.ny * p.nz;
    const h = p.L / p.n;                              // element size
    const ax = p.shape === 'plus' ? 1 : p.ax;
    // The angle slider drives everything: it fixes the shift, which fixes the
    // slab count, which is also the zig-zag period.
    const BODY_R = 0.38;                                // CellShape default
    // The interface insets drive the lattice: s = Lx - 2d in each direction.
    // The angle is independent -- it only sets how a stub slants on its way in.
    // Cap the width twice over. At exactly body_r a stub's far cap is tangent to
    // the body surface and the OCC boolean degenerates; and a fat stub meeting
    // the body at a shallow angle produces slivers the 3D mesher rejects, so the
    // ceiling has to fall as the lean gets harder (fitted to a sweep over the
    // slider grid: the ceiling is ~0.6 of the body radius at 25 deg, ~0.75 at
    // 40 deg, and 0.9 from 55 deg up).
    const wCap = Math.min(0.90, 0.30 + p.angleDeg / 90);
    const latR = Math.min(wCap * BODY_R,
                          BODY_R * ((Number.isNaN(p.latRPct) ? 68 : p.latRPct) / 100));
    const leanX = this.weakScalingLeanX(p.angleDeg);
    const dY = this.weakScalingInset(ax, latR, leanX,
                                     Number.isNaN(p.latPct) ? 50 : p.latPct);
    const dZ = this.weakScalingInset(ax, latR, leanX,
                                     Number.isNaN(p.latPctZ) ? 50 : p.latPctZ);
    const rat = this.weakScalingShift(ax, dY, dZ, 24);
    const slabs = p.shape === 'plus' ? 0 : rat.slabs;
    const step = p.shape === 'plus' ? 1 : rat.step_y;
    const stepZ = p.shape === 'plus' ? 1 : rat.step_z;
    p.slabs = slabs; p.step_y = step; p.step_z = stepZ;
    let tets, verts, domain;
    if (p.shape === 'plus') {
        const Gx = p.nx * p.n + 2 * p.pad;
        const Gy = p.ny * p.n + 2 * p.pad;
        const Gz = p.nz * p.n + 2 * p.pad;
        tets = Gx * Gy * Gz * 6;
        verts = (Gx + 1) * (Gy + 1) * (Gz + 1);
        domain = [Gx * h, Gy * h, Gz * h];
    } else {
        // ~6.5 tets per h-cube is what the unstructured mesher averages here.
        // Padding is a boundary layer, so it scales with surface, not volume.
        const bx = p.nx, by = p.ny, bz = p.nz;
        const core = bx * by * bz * ax * Math.pow(p.n, 3) * 6.5;
        const faces = 2 * (by * bz + ax * bx * bz + ax * bx * by) * p.n * p.n;
        tets = Math.round(core + 3 * p.pad * 2 * faces);
        verts = Math.round(tets / 5.7);
        // The lattice is shifted by Lx/slabs per step in y and z, so the block
        // is a staircase and its bounding box is longer in x than the cells.
        // Only the offsets this block actually reaches count: (J+K) runs to
        // ny+nz-2, so a small block never sees the whole wrap period.
        const seen = new Set();
        for (let J = 0; J < by; J++)
            for (let K = 0; K < bz; K++) seen.add((J * step + K * stepZ) % slabs);
        const spread = (Math.max(...seen) - Math.min(...seen)) * (ax / slabs);
        domain = [(bx * ax + spread) * p.L + 2 * p.pad * h,
                  by * p.L + 2 * p.pad * h, bz * p.L + 2 * p.pad * h];
    }

    // Does this exact configuration already exist?
    const existing = document.getElementById('ws-existing');
    let ready = false;
    if (existing) {
        for (const opt of existing.options) {
            if (opt.dataset.nx &&
                +opt.dataset.nx === p.nx && +opt.dataset.ny === p.ny &&
                +opt.dataset.nz === p.nz && +opt.dataset.n === p.n &&
                +opt.dataset.pad === p.pad && opt.dataset.shape === p.shape &&
                +opt.dataset.ax === ax && +opt.dataset.slabs === slabs &&
                (+opt.dataset.stepy || 1) === step &&
                (+opt.dataset.stepz || 1) === stepZ &&
                Math.abs(+opt.dataset.l - p.L) < 1e-9) {
                ready = opt.dataset.converted === 'true';
                break;
            }
        }
    }

    // Resolve the sliders into what the generator takes.
    const set = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
    };
    if (p.shape === 'plus') {
        p.d_y = p.d_z = p.lat_r = p.lean = 0;
        ['ws-angle-readout', 'ws-latr-readout', 'ws-latx-readout',
         'ws-latxz-readout'].forEach(id => set(id, 'cell shape only'));
    } else {
        // Snap the insets onto the shifts actually realised, so the readout and
        // the preview show what will be built.
        p.lat_r = Math.round(latR * 1000) / 1000;
        p.lean = p.angleDeg;
        p.d_y = 0.5 * (ax - ax * step / slabs);
        p.d_z = 0.5 * (ax - ax * stepZ / slabs);
        set('ws-angle-readout', `${p.angleDeg}° off the cell axis`
            + (p.angleDeg >= 89 ? ' (straight out)' : ''));
        const capped = latR < BODY_R * (p.latRPct / 100) - 1e-9;
        set('ws-latr-readout', `${(2 * latR * p.L).toFixed(1)} µm across, cell is `
            + `${(2 * BODY_R * p.L).toFixed(1)}`
            + (capped ? ' (capped — a fat stub at this angle grazes the body)' : ''));
        const Lxum = ax * p.L;
        [['ws-latx-readout', p.d_y, step], ['ws-latxz-readout', p.d_z, stepZ]]
            .forEach(([id, dd, st]) => set(id,
                `${(dd * p.L).toFixed(1)} µm from each end `
                + `(interfaces at ${(dd * p.L).toFixed(1)} and `
                + `${(Lxum - dd * p.L).toFixed(1)} of ${Lxum.toFixed(0)}); `
                + `shift ${st}/${slabs}`));
    }
    this._wsLastParams = p;
    if (this.updateWeakScalingPreview) this.updateWeakScalingPreview();

    estEl.className = 'ws-estimate';
    estEl.innerHTML =
        `${cells.toLocaleString()} cells · ~${tets.toLocaleString()} tets · ` +
        `~${verts.toLocaleString()} vertices<br>` +
        `box ${(ax * p.L).toFixed(1)} × ${p.L.toFixed(1)} × ${p.L.toFixed(1)} µm` +
        (p.shape === 'plus' ? '' :
            ` · connectors lean ${(Math.atan(ax / slabs) * 180 / Math.PI).toFixed(0)}°,` +
            ` interfaces at ${(100 / (2 * slabs)).toFixed(0)}% and` +
            ` ${(300 / (2 * slabs)).toFixed(0)}% along the cell`) +
        `<br>bounding box ${domain.map(d => d.toFixed(0)).join(' × ')} µm` +
        (p.shape === 'plus' ? '' : ' (zig-zag block)') +
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
            opt.dataset.shape = m.shape; opt.dataset.ax = m.ax;
            opt.dataset.slabs = m.slabs || 0;
            opt.dataset.stepy = m.step_y || 1;
            opt.dataset.stepz = m.step_z || 1;
            opt.dataset.converted = m.converted;
            opt.textContent = `${m.shape} ${m.nx}×${m.ny}×${m.nz}  ` +
                `(n=${m.n}, L=${m.L}, pad=${m.pad}` +
                (m.ax > 1 ? `, ax=${m.ax}` : '') + ')' +
                (m.converted ? '' : ' — needs viz convert');
            existing.appendChild(opt);
        }
        this.updateWeakScalingEstimate();
    } catch (e) {
        console.error('Failed to load weak-scaling mesh list:', e);
    }
};

App.prototype.generateOrLoadWeakScaling = async function() {
    this.updateWeakScalingEstimate();            // resolves lat_x from the slider
    const p = this._wsLastParams || this.readWeakScalingParams();
    const err = this.weakScalingValidation(p);
    const statusEl = document.getElementById('ws-status');

    if (err) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = err;
        statusEl.style.display = 'block';
        return;
    }

    // Remote mode: generate on the cluster so the mesh lands in the remote
    // data/ dir and shows up under remote meshes (rather than only locally).
    if (this.isRemote()) {
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
                    // Real mesh from here on -- drop the preview overlay so the
                    // viewer shows what was actually generated.
                    if (this.showWeakScalingPreview) this.showWeakScalingPreview(false);
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

// Generate a weak-scaling mesh on the cluster, then switch to the Standard tab and
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
    progressText.textContent = `Generating on ${this.clusterLabel()}...`;

    try {
        const { name } = await this.clusterRunner.generateWeakScaling(p, (text) => {
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
        this.setBatchChecked([name], true);

        statusEl.className = 'mesh-status converted';
        statusEl.textContent = `Generated on ${this.clusterLabel()}: ${name}`;
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
