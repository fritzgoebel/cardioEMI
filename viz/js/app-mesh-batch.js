// app-mesh-batch.js - Batch submission over several meshes (cluster targets).
//
// With a cluster run target the mesh panel shows a checklist instead of the
// single dropdown: checkboxes choose the batch, clicking a name focuses that
// mesh (viewer, sliders, stimulus box - same as picking it in the dropdown).
// Every checked mesh gets a row with its subdomain count and rank count(s);
// submission sends one job per (mesh, rank count), all built from the focused
// mesh's config. Each mesh's stimulus box comes from the batch-wide box
// (stimulusBoxFor in app-ui.js): ticked ends at that mesh's own bounds, the
// others at their absolute slider positions, clipped to the mesh.
//
// The selection survives the page reload that focusing a local mesh triggers
// (sessionStorage), but not the session.

const MESH_BATCH_STORAGE_KEY = 'meshBatch';

App.prototype.setupMeshBatch = function() {
    this.meshBatchState = this.loadMeshBatchState();  // clusterId -> {checked, rows}
    this.meshBatchInfo = {};       // clusterId -> {mesh: info | {error}}
    this.meshBatchPending = new Set();
    this.meshBatchMeshes = [];     // [{name, label}] converted meshes on the active cluster

    const filterEl = document.getElementById('mesh-batch-filter');
    filterEl.value = sessionStorage.getItem('meshBatchFilter') || '';
    filterEl.addEventListener('input', () => {
        sessionStorage.setItem('meshBatchFilter', filterEl.value);
        this.renderMeshChecklist();
        this.updateBatchFolderDefault();
    });

    document.getElementById('mesh-batch-all').addEventListener('click', () => {
        this.setBatchChecked(this.filteredBatchMeshes().map(m => m.name), true);
    });
    document.getElementById('mesh-batch-none').addEventListener('click', () => {
        this.setBatchChecked(this.filteredBatchMeshes().map(m => m.name), false);
    });

    const modeEl = document.getElementById('cluster-rank-mode');
    const kEl = document.getElementById('cluster-rank-k');
    const syncK = () => { kEl.style.display = modeEl.value === 'divide' ? 'inline-block' : 'none'; };
    modeEl.addEventListener('change', () => { syncK(); this.renderBatchRows(); });
    syncK();

    // Anything the rows derive from re-renders them.
    ['cluster-rank-k', 'cluster-ntasks', 'cluster-walltime'].forEach(id =>
        document.getElementById(id).addEventListener('input', () => this.renderBatchRows()));
    ['partition-mode', 'component-granularity'].forEach(id =>
        document.getElementById(id).addEventListener('change', () => this.renderBatchRows()));

    // Folder path follows the filter until the user edits it; clearing it
    // leaves the batch's runs unfiled.
    const folderEl = document.getElementById('cluster-folder');
    folderEl.dataset.auto = '1';
    folderEl.addEventListener('input', () => { folderEl.dataset.auto = '0'; });
    this.updateBatchFolderDefault();
};

// --------------------- State ---------------------

App.prototype.loadMeshBatchState = function() {
    try {
        return JSON.parse(sessionStorage.getItem(MESH_BATCH_STORAGE_KEY)) || {};
    } catch (e) {
        return {};
    }
};

App.prototype.saveMeshBatchState = function() {
    try {
        sessionStorage.setItem(MESH_BATCH_STORAGE_KEY, JSON.stringify(this.meshBatchState));
    } catch (e) { /* storage unavailable - selection just won't survive a reload */ }
};

App.prototype.batchState = function() {
    const cid = this.runTarget;
    if (!this.meshBatchState[cid]) this.meshBatchState[cid] = { checked: [], rows: {} };
    return this.meshBatchState[cid];
};

App.prototype.batchChecked = function() {
    return this.isRemote() ? this.batchState().checked : [];
};

App.prototype.batchRow = function(mesh) {
    const rows = this.batchState().rows;
    if (!rows[mesh]) rows[mesh] = { ranks: '', walltime: '' };
    return rows[mesh];
};

App.prototype.batchInfo = function(mesh) {
    return (this.meshBatchInfo[this.runTarget] || {})[mesh];
};

App.prototype.setBatchChecked = function(names, on) {
    const state = this.batchState();
    const set = new Set(state.checked);
    names.forEach(n => on ? set.add(n) : set.delete(n));
    // Keep the list order, so rows appear in the same order as the checklist.
    state.checked = this.meshBatchMeshes.map(m => m.name).filter(n => set.has(n));
    this.saveMeshBatchState();
    this.renderMeshChecklist();
    this.renderBatchRows();
    this.ensureBatchInfo();
};

// --------------------- Checklist ---------------------

App.prototype.filteredBatchMeshes = function() {
    const raw = document.getElementById('mesh-batch-filter').value.trim();
    if (!raw) return this.meshBatchMeshes;
    // '*' is a wildcard; without one the filter is a plain substring match.
    const esc = raw.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
    const re = new RegExp(raw.includes('*') ? `^${esc}$` : esc, 'i');
    return this.meshBatchMeshes.filter(m => re.test(m.name));
};

// Called by refreshMeshList (remote branch) with the cluster's converted meshes.
App.prototype.setBatchMeshes = function(meshes) {
    this.meshBatchMeshes = meshes;
    const state = this.batchState();
    const names = new Set(meshes.map(m => m.name));
    state.checked = state.checked.filter(n => names.has(n));
    // A refresh is the retry path for meshes whose info failed to load.
    const info = this.meshBatchInfo[this.runTarget] || {};
    for (const [name, entry] of Object.entries(info)) {
        if (entry.error) delete info[name];
    }
    this.saveMeshBatchState();
    this.renderMeshChecklist();
    this.renderBatchRows();
    this.ensureBatchInfo();
};

App.prototype.renderMeshChecklist = function() {
    const list = document.getElementById('mesh-batch-list');
    const checked = new Set(this.batchChecked());
    const focused = this.meshLoader.currentMesh;
    const shown = this.filteredBatchMeshes();
    list.innerHTML = '';

    if (this.meshBatchMeshes.length === 0) {
        list.innerHTML = '<div class="mesh-batch-empty">No converted meshes on this cluster</div>';
        return;
    }
    if (shown.length === 0) {
        list.innerHTML = '<div class="mesh-batch-empty">No mesh matches the filter</div>';
        return;
    }

    for (const mesh of shown) {
        const item = document.createElement('div');
        item.className = 'mesh-batch-item' + (mesh.name === focused ? ' focused' : '');

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = checked.has(mesh.name);
        cb.title = 'Include in the batch';
        cb.addEventListener('change', () => this.setBatchChecked([mesh.name], cb.checked));

        const name = document.createElement('span');
        name.className = 'mesh-batch-name';
        name.textContent = mesh.name;
        name.title = `Show ${mesh.name} (its stimulus box becomes the reference)`;
        name.addEventListener('click', () => this.focusBatchMesh(mesh.name));

        const tag = document.createElement('span');
        tag.className = 'mesh-batch-tag';
        tag.textContent = mesh.label;

        item.append(cb, name, tag);
        list.appendChild(item);
    }
};

App.prototype.focusBatchMesh = async function(meshName) {
    const selector = document.getElementById('mesh-selector');
    selector.value = meshName;
    await this.onMeshSelected(meshName);
    this.renderMeshChecklist();
    this.renderBatchRows();
};

App.prototype.updateBatchFolderDefault = function() {
    const folderEl = document.getElementById('cluster-folder');
    if (!folderEl || folderEl.dataset.auto !== '1') return;
    const filter = document.getElementById('mesh-batch-filter').value.trim();
    const date = new Date().toISOString().slice(0, 10);
    folderEl.value = `${filter || 'batch'} ${date}`;
};

App.prototype.batchFolderName = function() {
    return document.getElementById('cluster-folder').value.trim();
};

// --------------------- Mesh info (bounds + subdomain counts) ---------------------

App.prototype.ensureBatchInfo = async function() {
    if (!this.isRemote()) return;
    const cid = this.runTarget;
    const info = this.meshBatchInfo[cid] || (this.meshBatchInfo[cid] = {});
    const missing = this.batchChecked().filter(m => !info[m] && !this.meshBatchPending.has(`${cid}:${m}`));
    if (missing.length === 0) return;

    missing.forEach(m => this.meshBatchPending.add(`${cid}:${m}`));
    this.renderBatchRows();
    try {
        const fetched = await this.getRunner(cid).fetchBatchMeshInfo(missing);
        Object.assign(info, fetched);
    } catch (e) {
        missing.forEach(m => { info[m] = { error: `Mesh info failed: ${e.message}` }; });
    } finally {
        missing.forEach(m => this.meshBatchPending.delete(`${cid}:${m}`));
    }
    if (this.runTarget === cid) this.renderBatchRows();
};

// Partition units the solver config's partitioning will create on this mesh.
// {n, unit} | {n: null} (default partitioner - no notion of subdomains) | {error}
App.prototype.batchSubdomains = function(info) {
    if (this.partitionMode === 'cube') {
        return info.cube_subdomains
            ? { n: info.cube_subdomains, unit: 'subdomains' }
            : { error: 'Cube partitioning needs a weak-scaling mesh name (<cell|plus>_<nx>x<ny>x<nz>_...)' };
    }
    if (this.partitionMode === 'component') {
        const perTag = this.componentGranularity === 'tag';
        const n = perTag ? info.num_original_tags : info.num_components;
        return n
            ? { n, unit: perTag ? 'tags' : 'ECS+cell pairs' }
            : { error: 'Tag-based partitioning needs a _colored mesh with its original mesh on the cluster' };
    }
    return { n: null };
};

// --------------------- Per-mesh rows ---------------------

App.prototype.batchMaxTasksPerNode = function() {
    return Math.max(1, parseInt(document.getElementById('cluster-ntasks').value) || 128);
};

// Mirrors Cluster.pack_ranks: fewest nodes at <= cap tasks/node, tasks spread evenly.
App.prototype.packRanks = function(ranks, cap) {
    const nodes = Math.max(1, Math.ceil(ranks / cap));
    return { nodes, tasks: Math.ceil(ranks / nodes) };
};

// What this mesh's row would submit: {pending} or {ranks, sub, errors, warnings}.
App.prototype.batchRowPlan = function(mesh) {
    const info = this.batchInfo(mesh);
    if (!info) return { pending: true, errors: [], warnings: [] };
    if (info.error) return { ranks: [], errors: [info.error], warnings: [] };

    const row = this.batchRow(mesh);
    const sub = this.batchSubdomains(info);
    const errors = sub.error ? [sub.error] : [];
    const warnings = [];
    let ranks = [];

    if (row.ranks.trim()) {
        const parts = row.ranks.split(',').map(s => s.trim()).filter(Boolean);
        const bad = parts.filter(p => !/^\d+$/.test(p) || parseInt(p) < 1);
        if (bad.length) errors.push(`Not a rank count: ${bad.join(', ')}`);
        ranks = [...new Set(parts.filter(p => !bad.includes(p)).map(Number))].sort((a, b) => a - b);
    } else {
        const mode = document.getElementById('cluster-rank-mode').value;
        if (mode === 'custom') {
            errors.push('Enter rank count(s)');
        } else if (!sub.n) {
            if (!sub.error) errors.push('Default partitioning has no subdomains - enter rank count(s)');
        } else if (mode === 'divide') {
            const k = parseInt(document.getElementById('cluster-rank-k').value) || 0;
            if (k < 1) errors.push('k must be a positive integer');
            else if (sub.n % k) errors.push(`k = ${k} does not divide ${sub.n} ${sub.unit}`);
            else ranks = [sub.n / k];
        } else {
            ranks = [sub.n];
        }
    }

    if (info.bounds && this.stimulusBoxFor(info.bounds).empty) {
        errors.push('The stimulus box misses this mesh - widen it or tick min/max');
    }

    // The same rules main.py / mesh_partition.py enforce at startup.
    if (sub.n) {
        for (const r of ranks) {
            if (this.partitionMode === 'cube' && sub.n % r) {
                errors.push(`${r} ranks do not divide ${sub.n} subdomains`);
            } else if (this.partitionMode === 'component' && this.componentGranularity === 'tag' && r !== sub.n) {
                errors.push(`Per-tag partitioning needs exactly ${sub.n} ranks (got ${r})`);
            } else if (this.partitionMode === 'component' && r > sub.n) {
                warnings.push(`${r} ranks > ${sub.n} ${sub.unit}: ${r - sub.n} rank(s) stay empty`);
            }
        }
    }
    return { ranks, sub, errors, warnings };
};

App.prototype.renderBatchRows = function() {
    const container = document.getElementById('mesh-batch-rows');
    if (!container) return;
    const checked = this.batchChecked();
    const keep = new Set(checked);

    // Reuse row elements so typing in one row never loses focus to a re-render.
    for (const el of [...container.children]) {
        if (!keep.has(el.dataset.mesh)) el.remove();
    }
    checked.forEach((mesh, i) => {
        let el = container.querySelector(`.batch-row[data-mesh="${CSS.escape(mesh)}"]`);
        if (!el) el = this.createBatchRow(mesh);
        if (container.children[i] !== el) container.insertBefore(el, container.children[i] || null);
        this.updateBatchRow(mesh, el);
    });
    this.updateBatchSummary();
};

App.prototype.createBatchRow = function(mesh) {
    const el = document.createElement('div');
    el.className = 'batch-row';
    el.dataset.mesh = mesh;
    el.innerHTML = `
        <div class="batch-row-head">
            <span class="batch-row-name"></span>
            <span class="batch-row-sub"></span>
        </div>
        <div class="batch-row-body">
            <label>Ranks</label>
            <input type="text" class="batch-ranks" title="Rank count(s) for this mesh, e.g. 64,128 (one job each). Empty = the rank mode below.">
            <button type="button" class="btn-small batch-reset" title="Back to the rank mode">&#8634;</button>
            <span class="batch-pack" title="nodes x tasks/node"></span>
            <input type="text" class="batch-walltime" title="Walltime for this mesh's jobs (empty = default)">
        </div>
        <div class="batch-row-stim"></div>
        <div class="batch-row-msg"></div>`;
    el.querySelector('.batch-row-name').textContent = mesh;
    el.querySelector('.batch-row-name').title = mesh;

    const ranksEl = el.querySelector('.batch-ranks');
    ranksEl.value = this.batchRow(mesh).ranks;
    ranksEl.addEventListener('input', () => {
        this.batchRow(mesh).ranks = ranksEl.value;
        this.saveMeshBatchState();
        this.updateBatchRow(mesh, el);
        this.updateBatchSummary();
    });
    el.querySelector('.batch-reset').addEventListener('click', () => {
        ranksEl.value = '';
        ranksEl.dispatchEvent(new Event('input'));
    });

    const wallEl = el.querySelector('.batch-walltime');
    wallEl.value = this.batchRow(mesh).walltime;
    wallEl.addEventListener('input', () => {
        this.batchRow(mesh).walltime = wallEl.value.trim();
        this.saveMeshBatchState();
    });
    return el;
};

App.prototype.updateBatchRow = function(mesh, el) {
    const plan = this.batchRowPlan(mesh);
    const row = this.batchRow(mesh);
    const subEl = el.querySelector('.batch-row-sub');
    const ranksEl = el.querySelector('.batch-ranks');
    const packEl = el.querySelector('.batch-pack');
    const msgEl = el.querySelector('.batch-row-msg');

    el.classList.toggle('focused', mesh === this.meshLoader.currentMesh);
    el.classList.toggle('has-error', plan.errors.length > 0);

    if (plan.pending) {
        subEl.textContent = 'loading…';
    } else if (plan.sub && plan.sub.n) {
        subEl.textContent = `${plan.sub.n} ${plan.sub.unit}`;
    } else {
        subEl.textContent = '';
    }

    const auto = !row.ranks.trim() && !plan.pending && plan.ranks.length ? String(plan.ranks[0]) : '';
    ranksEl.placeholder = auto ? `auto: ${auto}` : 'e.g. 64,128';
    el.querySelector('.batch-reset').style.visibility = row.ranks.trim() ? 'visible' : 'hidden';

    const cap = this.batchMaxTasksPerNode();
    packEl.textContent = (plan.ranks || []).map(r => {
        const p = this.packRanks(r, cap);
        return `${p.nodes}×${p.tasks}`;
    }).join(', ');

    el.querySelector('.batch-walltime').placeholder =
        document.getElementById('cluster-walltime').value || '01:00:00';

    const info = this.batchInfo(mesh);
    const stimEl = el.querySelector('.batch-row-stim');
    if (info && info.bounds && this.boundingBox) {
        const { box } = this.stimulusBoxFor(info.bounds);
        const fmt = (a) => {
            const full = box[`${a}Min`] <= info.bounds[a][0] && box[`${a}Max`] >= info.bounds[a][1];
            return full ? `${a} full` : `${a} ${box[`${a}Min`].toFixed(1)}–${box[`${a}Max`].toFixed(1)}`;
        };
        stimEl.textContent = `stimulus: ${['x', 'y', 'z'].map(fmt).join(', ')} µm`;
    } else {
        stimEl.textContent = '';
    }

    msgEl.innerHTML = '';
    for (const [cls, text] of [...plan.errors.map(t => ['error', t]), ...plan.warnings.map(t => ['warning', t])]) {
        const line = document.createElement('div');
        line.className = cls;
        line.textContent = text;
        msgEl.appendChild(line);
    }
};

App.prototype.updateBatchSummary = function() {
    const el = document.getElementById('mesh-batch-summary');
    if (!el) return;
    const checked = this.batchChecked();
    if (checked.length === 0) {
        el.textContent = this.meshBatchMeshes.length ? 'Tick meshes to add them to the batch.' : '';
        el.className = 'mesh-batch-summary';
        return;
    }
    const cap = this.batchMaxTasksPerNode();
    let jobs = 0, nodes = 0, bad = 0, pending = 0;
    for (const mesh of checked) {
        const plan = this.batchRowPlan(mesh);
        if (plan.pending) { pending++; continue; }
        if (plan.errors.length) { bad++; continue; }
        jobs += plan.ranks.length;
        plan.ranks.forEach(r => { nodes += this.packRanks(r, cap).nodes; });
    }
    const parts = [`${checked.length} mesh${checked.length === 1 ? '' : 'es'}`,
                   `${jobs} job${jobs === 1 ? '' : 's'}`, `${nodes} node${nodes === 1 ? '' : 's'}`];
    if (pending) parts.push(`${pending} loading`);
    if (bad) parts.push(`${bad} with errors`);
    el.textContent = parts.join(' · ');
    el.className = 'mesh-batch-summary' + (bad ? ' error' : '');
};

// --------------------- Submission payload ---------------------

// -> {meshes: [...] } for /submit-batch, or throws with every problem listed.
App.prototype.buildBatchRequest = function() {
    const checked = this.batchChecked();
    if (checked.length === 0) throw new Error('Tick at least one mesh in the mesh panel');

    const focused = this.meshLoader.currentMesh;
    if (this.scarEnabled && !(checked.length === 1 && checked[0] === focused)) {
        throw new Error('Scar regions only apply to the shown mesh - disable scar, or submit just that mesh');
    }

    const problems = [];
    const meshes = [];
    for (const mesh of checked) {
        const plan = this.batchRowPlan(mesh);
        if (plan.pending) { problems.push(`${mesh}: mesh info still loading`); continue; }
        if (plan.errors.length) { problems.push(`${mesh}: ${plan.errors.join('; ')}`); continue; }

        const info = this.batchInfo(mesh);
        const cf = info.mesh_conversion_factor;
        const { box } = this.stimulusBoxFor(info.bounds);
        meshes.push({
            mesh,
            ranks: plan.ranks,
            walltime: this.batchRow(mesh).walltime || null,
            config_overrides: {
                v_init: this.generateVinitExpression(box, cf).slice(1, -1),
                mesh_conversion_factor: cf,
            },
            conditions_overrides: { boundingBox: box },
        });
    }
    if (problems.length) throw new Error(problems.join('\n'));
    return { meshes };
};
