// app-runs.js - Runs browser: every simulation run, local and on every
// cluster, in user-made virtual folders (see viz/run_index.py).
//
// One tree replaces the old results dropdown and the compare-runs tree:
//   - the checkbox puts a run into the comparison charts (a remote-only run's
//     iterations are fetched on demand) and doubles as the selection for
//     Move to... / Delete;
//   - clicking a run makes it the current run for Load Results, Download and
//     video export; double-click loads it.
// Folders nest arbitrarily and are only names in run_labels.json, so filing,
// renaming and moving never touch data. Runs not filed anywhere sit under
// "Unfiled", grouped by mesh. Deleting removes a run everywhere (local folder,
// viz cache, every cluster copy, its label and job-list entry).

const RUNS_EXPANDED_KEY = 'runsExpanded';
const RUNS_CHECKED_KEY = 'runsChecked';
const RUNS_CURRENT_KEY = 'runsCurrent';

App.prototype.setupRunBrowser = function() {
    this.runIndex = [];
    this.runByName = {};
    this.runFolders = [];
    this.runClusters = {};
    this._runIndexSeq = 0;
    this.runChecked = new Set(this._loadJson(sessionStorage, RUNS_CHECKED_KEY, []));
    this.currentRun = sessionStorage.getItem(RUNS_CURRENT_KEY) || null;
    this.selectedSimulation = this.currentRun;
    this.runExpanded = new Set(this._loadJson(localStorage, RUNS_EXPANDED_KEY, []));

    const filterEl = document.getElementById('runs-filter');
    filterEl.addEventListener('input', () => this.renderRunBrowser());
    document.getElementById('runs-refresh').addEventListener('click',
        () => this.loadRunIndex({ refresh: true }));
    document.getElementById('runs-new-folder').addEventListener('click', () => this.promptNewFolder(''));
    document.getElementById('runs-move').addEventListener('click', () => this.openMovePopover());
    document.getElementById('runs-move-go').addEventListener('click', () => this.submitMovePopover());
    document.getElementById('runs-move-cancel').addEventListener('click', () => {
        document.getElementById('runs-move-popover').style.display = 'none';
    });
    document.getElementById('runs-move-target').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.submitMovePopover();
        if (e.key === 'Escape') document.getElementById('runs-move-popover').style.display = 'none';
    });
    document.getElementById('runs-delete').addEventListener('click', () => this.deleteCheckedRuns());
    document.getElementById('runs-delete-failed').addEventListener('click', () => this.deleteFailedRuns());

    // Instant render from the cache, then re-list the clusters.
    this.loadRunIndex().then(() => this.loadRunIndex({ refresh: true }));
};

App.prototype._loadJson = function(storage, key, fallback) {
    try {
        const v = JSON.parse(storage.getItem(key));
        return v == null ? fallback : v;
    } catch (e) {
        return fallback;
    }
};

App.prototype._saveRunState = function() {
    try {
        sessionStorage.setItem(RUNS_CHECKED_KEY, JSON.stringify([...this.runChecked]));
        if (this.currentRun) sessionStorage.setItem(RUNS_CURRENT_KEY, this.currentRun);
        localStorage.setItem(RUNS_EXPANDED_KEY, JSON.stringify([...this.runExpanded]));
    } catch (e) { /* storage unavailable: state just won't survive a reload */ }
};

// Old entry points, kept so their many callers refresh the browser instead.
App.prototype.loadSimulationList = function() { return this.loadRunIndex(); };
App.prototype.updateCompareSelector = function() { return this.loadRunIndex(); };

// --------------------- index ---------------------

// Fetch /api/runs (refresh: also re-list every reachable cluster). Overlapping
// calls are fine: only the newest response is rendered. While a refresh is in
// flight, a plain reload waits for it instead - otherwise its quicker answer
// from the old cache would be newer and the refresh result thrown away.
App.prototype.loadRunIndex = function(opts = {}) {
    if (opts.refresh) {
        const p = this._loadRunIndex(opts).finally(() => {
            if (this._runRefresh === p) this._runRefresh = null;
        });
        this._runRefresh = p;
        return p;
    }
    return this._runRefresh || this._loadRunIndex(opts);
};

App.prototype._loadRunIndex = async function(opts) {
    const seq = ++this._runIndexSeq;
    if (opts.refresh) this._setRunStatus('Listing runs on the clusters…');
    let data;
    try {
        const resp = await fetch('/api/runs' + (opts.refresh ? '?refresh=1' : ''));
        data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
    } catch (e) {
        if (seq === this._runIndexSeq) this._setRunStatus(`Could not load runs: ${e.message}`, true);
        return;
    }
    if (seq !== this._runIndexSeq) return;

    this.runIndex = data.runs || [];
    this.runByName = Object.fromEntries(this.runIndex.map(r => [r.name, r]));
    this.runFolders = data.folders || [];
    this.runClusters = data.clusters || {};
    this.runNames = data.labels || {};
    this.categoryNames = data.categories || {};
    this.compareSimMeta = this.compareSimMeta || {};
    for (const r of this.runIndex) {
        this.compareSimMeta[r.name] = {
            solver: r.solver, preconditioner: r.preconditioner, localSolver: r.localSolver,
            nRanks: r.nRanks, nSubdomains: r.nSubdomains, hRatio: r.hRatio,
        };
    }
    for (const n of [...this.runChecked]) {
        if (!this.runByName[n]) this.runChecked.delete(n);
    }
    if (this.currentRun && !this.runByName[this.currentRun]) this.currentRun = null;
    this._saveRunState();

    this.renderRunBrowser();
    this.updateCurrentRunUI();
    this._applyLoadedRunLabel && this._applyLoadedRunLabel();
};

// Coalesce refresh requests (e.g. several jobs finishing at once).
App.prototype.requestRunRefresh = function() {
    clearTimeout(this._runRefreshTimer);
    this._runRefreshTimer = setTimeout(() => this.loadRunIndex({ refresh: true }), 3000);
};

App.prototype._setRunStatus = function(text, isError) {
    const el = document.getElementById('runs-status');
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('error', !!isError);
};

App.prototype._clusterStatusText = function() {
    const parts = [];
    for (const [cid, c] of Object.entries(this.runClusters)) {
        if (c.error) {
            parts.push(`${c.label}: ${c.updated ? 'last listing ' + this._ago(c.updated) : 'never listed'} (unreachable)`);
        } else if (c.updated) {
            parts.push(`${c.label}: ${this._ago(c.updated)}`);
        }
    }
    return parts.join(' · ');
};

App.prototype._ago = function(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return 'just now';
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} d ago`;
};

// --------------------- classification ---------------------

App.prototype._jobForRun = function(name) {
    return Object.values(this.clusterJobs || {}).find(j => j.out_name === name) || null;
};

// A run is failed when it holds nothing anywhere and no job for it is queued
// or running. While its job is still in the job list it stays (so the log can
// be read); dismissing the job deletes it.
App.prototype._runIsFailed = function(r) {
    if (r.hasData || r.active) return false;
    const job = this._jobForRun(r.name);
    if (job && !CLUSTER_TERMINAL_STATES.includes(job.status)) return false;
    return true;
};

// Runs the "Delete failed" button may remove: failed, not kept by the job
// list, and every cluster holding it was checked for active jobs. Plus empty
// cluster copies of runs that have data elsewhere (old sync artifacts).
App.prototype._failedCleanup = function() {
    const kept = new Set(Object.values(this.clusterJobs || {}).map(j => j.out_name));
    const runs = [];
    const copies = [];
    for (const r of this.runIndex) {
        const checked = Object.keys(r.remote).every(cid => (this.runClusters[cid] || {}).active_ok);
        if (this._runIsFailed(r)) {
            if (!kept.has(r.name) && checked) runs.push(r.name);
        } else if (r.hasData) {
            for (const [cid, rem] of Object.entries(r.remote)) {
                if (!rem.iterations && !rem.results && !rem.active
                        && (this.runClusters[cid] || {}).active_ok) {
                    copies.push({ cluster: cid, name: r.name });
                }
            }
        }
    }
    return { runs, copies };
};

App.prototype._runSize = function(r) {
    let size = (r.local && r.local.size) || 0;
    for (const rem of Object.values(r.remote)) size += rem.size || 0;
    return size;
};

App.prototype._fmtSize = function(bytes) {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
    return `${Math.max(1, Math.round(bytes / 1e3))} kB`;
};

// --------------------- tree ---------------------

App.prototype._runFilter = function() {
    const raw = document.getElementById('runs-filter').value.trim();
    if (!raw) return null;
    const esc = raw.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
    return new RegExp(esc, 'i');
};

App.prototype._runMatches = function(r, re) {
    if (!re) return true;
    return [r.name, r.label, r.mesh, r.folder, r.solver, r.preconditioner]
        .some(v => v && re.test(String(v)));
};

App.prototype._buildRunTree = function(re) {
    const node = (path) => ({ path, name: path.split('/').pop(), folders: new Map(), runs: [] });
    const root = node('');
    const ensure = (path) => {
        let cur = root;
        let acc = '';
        for (const part of path.split('/')) {
            acc = acc ? `${acc}/${part}` : part;
            if (!cur.folders.has(part)) cur.folders.set(part, node(acc));
            cur = cur.folders.get(part);
        }
        return cur;
    };
    this.runFolders.forEach(ensure);
    const unfiled = new Map();
    for (const r of this.runIndex) {
        if (!this._runMatches(r, re)) continue;
        if (r.folder) {
            ensure(r.folder).runs.push(r);
        } else {
            if (!unfiled.has(r.mesh)) unfiled.set(r.mesh, []);
            unfiled.get(r.mesh).push(r);
        }
    }
    return { root, unfiled };
};

App.prototype._allRunsUnder = function(node) {
    const out = [...node.runs];
    for (const child of node.folders.values()) out.push(...this._allRunsUnder(child));
    return out;
};

App.prototype._sortRuns = function(runs) {
    return [...runs].sort((a, b) => (a.mesh || '').localeCompare(b.mesh || '')
        || (a.nRanks || 0) - (b.nRanks || 0)
        || (a.timestamp || '').localeCompare(b.timestamp || ''));
};

App.prototype.renderRunBrowser = function() {
    const tree = document.getElementById('runs-tree');
    if (!tree) return;
    const re = this._runFilter();
    const { root, unfiled } = this._buildRunTree(re);
    const scroll = tree.scrollTop;
    tree.innerHTML = '';

    const folders = [...root.folders.values()].sort((a, b) => a.name.localeCompare(b.name));
    for (const f of folders) this._renderFolder(tree, f, 0, re);

    // Unfiled: one pseudo-folder (drop target for "unfile"), grouped by mesh.
    const unfiledRuns = [...unfiled.values()].flat();
    const uKey = 'u';
    const uOpen = !!re || this.runExpanded.has(uKey);
    const uRow = this._folderRow({
        label: 'Unfiled', count: unfiledRuns.length, depth: 0, open: uOpen,
        runs: unfiledRuns, dropPath: '', key: uKey, special: true,
    });
    tree.appendChild(uRow);
    if (uOpen) {
        for (const mesh of [...unfiled.keys()].sort()) {
            const runs = unfiled.get(mesh);
            const key = `m:${mesh}`;
            const open = !!re || this.runExpanded.has(key);
            tree.appendChild(this._folderRow({
                label: this._categoryTitle(`mesh:${mesh}`, mesh), count: runs.length, depth: 1,
                open, runs, key, special: true, title: mesh,
            }));
            if (open) {
                for (const r of this._sortRuns(runs)) tree.appendChild(this._runRow(r, 2, false));
            }
        }
    }
    if (this.runIndex.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'rb-empty';
        empty.textContent = 'No runs yet';
        tree.appendChild(empty);
    }
    tree.scrollTop = scroll;

    // Toolbar state + folder suggestions for Move to...
    const { runs: failed, copies } = this._failedCleanup();
    const failedBtn = document.getElementById('runs-delete-failed');
    const n = failed.length + copies.length;
    failedBtn.style.display = n ? '' : 'none';
    failedBtn.textContent = `Delete failed (${n})`;
    document.getElementById('runs-folder-list').innerHTML =
        this.runFolders.map(f => `<option value="${f.replace(/"/g, '&quot;')}"></option>`).join('');
    const checked = this.runChecked.size;
    document.getElementById('runs-move').disabled = !checked;
    document.getElementById('runs-delete').disabled = !checked;
    document.getElementById('runs-delete').textContent = checked ? `Delete (${checked})` : 'Delete';
    this._setRunStatus(this._clusterStatusText());
};

App.prototype._renderFolder = function(parent, node, depth, re) {
    const runs = this._allRunsUnder(node);
    if (re && runs.length === 0) return;  // filtering: hide folders without matches
    const key = `f:${node.path}`;
    const open = !!re || this.runExpanded.has(key);
    parent.appendChild(this._folderRow({
        label: node.name, count: runs.length, depth, open, runs,
        dropPath: node.path, key, path: node.path,
    }));
    if (!open) return;
    const children = [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name));
    for (const child of children) this._renderFolder(parent, child, depth + 1, re);
    for (const r of this._sortRuns(node.runs)) parent.appendChild(this._runRow(r, depth + 1, true));
};

App.prototype._iconButton = function(text, title, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'rb-icon';
    btn.textContent = text;
    btn.title = title;
    btn.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
    return btn;
};

// One folder-like row: a real folder (opts.path), Unfiled, or a mesh group.
App.prototype._folderRow = function(opts) {
    const row = document.createElement('div');
    row.className = 'rb-folder' + (opts.special ? ' rb-special' : '');
    row.style.paddingLeft = `${opts.depth * 14 + 2}px`;
    if (opts.title) row.title = opts.title;

    const twisty = document.createElement('span');
    twisty.className = 'rb-twisty';
    twisty.textContent = opts.open ? '▾' : '▸';
    row.appendChild(twisty);

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.title = `Select all ${opts.count} run(s)`;
    const n = opts.runs.filter(r => this.runChecked.has(r.name)).length;
    cb.checked = opts.runs.length > 0 && n === opts.runs.length;
    cb.indeterminate = n > 0 && n < opts.runs.length;
    cb.addEventListener('click', (e) => e.stopPropagation());
    cb.addEventListener('change', () => this.setRunsChecked(opts.runs.map(r => r.name), cb.checked));
    row.appendChild(cb);

    const name = document.createElement('span');
    name.className = 'rb-folder-name';
    name.textContent = opts.label;
    row.appendChild(name);
    const count = document.createElement('span');
    count.className = 'rb-count';
    count.textContent = opts.count;
    row.appendChild(count);

    const actions = document.createElement('span');
    actions.className = 'rb-actions';
    if (opts.path) {
        actions.append(
            this._iconButton('✎', 'Rename folder', () => this.promptRenameFolder(opts.path)),
            this._iconButton('＋', 'New subfolder', () => this.promptNewFolder(opts.path)),
            this._iconButton('🗑', 'Delete this folder and every run in it, everywhere',
                () => this.deleteFolder(opts.path, opts.runs)),
        );
    } else if (opts.key.startsWith('m:')) {
        const mesh = opts.key.slice(2);
        actions.append(this._iconButton('✎', `Name for mesh group "${mesh}" (display only)`, async () => {
            const v = window.prompt(`Name for mesh group "${mesh}" (empty: use the mesh name)`,
                this._categoryTitle(`mesh:${mesh}`, ''));
            if (v !== null) await this._saveCompareNames({ categories: { [`mesh:${mesh}`]: v.trim() } });
        }));
    }
    row.appendChild(actions);

    row.addEventListener('click', () => {
        if (this.runExpanded.has(opts.key)) this.runExpanded.delete(opts.key);
        else this.runExpanded.add(opts.key);
        this._saveRunState();
        this.renderRunBrowser();
    });

    if (opts.path) {
        row.draggable = true;
        row.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('application/x-cardioemi-folder', opts.path);
            e.dataTransfer.effectAllowed = 'move';
        });
    }
    if (opts.dropPath !== undefined) this._makeDropTarget(row, opts.dropPath);
    return row;
};

App.prototype._shortMesh = function(mesh) {
    const m = (mesh || '').match(/^(cell|plus)_\d+x\d+x\d+_n\d+_L[0-9p]+/);
    return m ? m[0] : (mesh || '?');
};

App.prototype._runDisplayName = function(r, withMesh) {
    if (r.label) return r.label;
    const parts = [];
    if (withMesh) parts.push(this._shortMesh(r.mesh));
    if (r.nRanks) parts.push(`${r.nRanks}r`);
    const solver = [r.solver, r.preconditioner].filter(Boolean).join('/');
    if (solver) parts.push(solver);
    if (r.timestamp) {
        const today = new Date().toISOString().slice(0, 10);
        const [d, t] = r.timestamp.split(' ');
        parts.push(d === today ? t.slice(0, 5) : `${d.slice(5)} ${t.slice(0, 5)}`);
    }
    return parts.join(' · ') || r.name;
};

App.prototype._badge = function(text, cls, title) {
    const b = document.createElement('span');
    b.className = `rb-badge ${cls}`;
    b.textContent = text;
    b.title = title;
    return b;
};

App.prototype._runBadges = function(r) {
    const out = [];
    if (r.local) {
        if (r.local.results) out.push(this._badge('local', 'rb-full', 'Full results on this machine'));
        else if (r.local.iterations) out.push(this._badge('local·it', 'rb-it', 'Only iterations/residuals on this machine'));
    }
    for (const [cid, rem] of Object.entries(r.remote)) {
        const c = this.runClusters[cid] || { label: cid };
        const stale = c.error ? ' (from the last listing - cluster unreachable)' : '';
        if (rem.active) {
            out.push(this._badge(`${c.label} ${rem.active[1].toLowerCase()}`, 'rb-active',
                `Job ${rem.active[0]} is ${rem.active[1]} on ${c.label}${stale}`));
        } else if (rem.results) {
            out.push(this._badge(c.label, 'rb-remote' + (c.error ? ' rb-stale' : ''), `Full results on ${c.label}${stale}`));
        } else if (rem.iterations) {
            out.push(this._badge(`${c.label}·it`, 'rb-it' + (c.error ? ' rb-stale' : ''), `Only iterations on ${c.label}${stale}`));
        } else if (r.hasData) {
            out.push(this._badge(`${c.label} ∅`, 'rb-none', `Empty copy on ${c.label} (a sync artifact; "Delete failed" removes it)${stale}`));
        }
    }
    if (this._runIsFailed(r)) {
        const job = this._jobForRun(r.name);
        out.push(this._badge('failed', 'rb-failed', job
            ? 'No results. Kept while its job is in the job list - dismissing the job deletes it.'
            : 'No results anywhere. "Delete failed" removes it.'));
    }
    return out;
};

App.prototype._runRow = function(r, depth, withMesh) {
    const row = document.createElement('div');
    row.className = 'rb-run'
        + (r.name === this.currentRun ? ' rb-current' : '')
        + (this._runIsFailed(r) ? ' rb-failed-run' : '');
    row.style.paddingLeft = `${depth * 14 + 2}px`;
    row.title = `${r.name}\n${this._fmtSize(this._runSize(r))}`;
    row.dataset.name = r.name;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = this.runChecked.has(r.name);
    cb.title = 'Compare in the charts / select for Move and Delete';
    cb.addEventListener('click', (e) => e.stopPropagation());
    cb.addEventListener('change', () => this.setRunsChecked([r.name], cb.checked));
    row.appendChild(cb);

    const name = document.createElement('span');
    name.className = 'rb-run-name';
    name.textContent = this._runDisplayName(r, withMesh);
    row.appendChild(name);

    const badges = document.createElement('span');
    badges.className = 'rb-badges';
    this._runBadges(r).forEach(b => badges.appendChild(b));
    row.appendChild(badges);

    row.appendChild(this._iconButton('✎', 'Rename run (display only)', async () => {
        const v = window.prompt(`Name for this run (empty: automatic)\n${r.name}`, r.label || '');
        if (v !== null) await this._saveCompareNames({ runs: { [r.name]: { label: v.trim() } } });
    }));

    row.addEventListener('click', () => this.setCurrentRun(r.name));
    row.addEventListener('dblclick', () => { this.setCurrentRun(r.name); this.loadResults(); });

    row.draggable = true;
    row.addEventListener('dragstart', (e) => {
        // Dragging a checked run drags the whole selection.
        const names = this.runChecked.has(r.name) ? [...this.runChecked] : [r.name];
        e.dataTransfer.setData('application/x-cardioemi-runs', JSON.stringify(names));
        e.dataTransfer.effectAllowed = 'move';
    });
    return row;
};

App.prototype._makeDropTarget = function(row, path) {
    const accepts = (e) => e.dataTransfer.types.includes('application/x-cardioemi-runs')
        || e.dataTransfer.types.includes('application/x-cardioemi-folder');
    row.addEventListener('dragover', (e) => {
        if (!accepts(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        row.classList.add('rb-drop');
    });
    row.addEventListener('dragleave', () => row.classList.remove('rb-drop'));
    row.addEventListener('drop', async (e) => {
        e.preventDefault();
        row.classList.remove('rb-drop');
        const runs = e.dataTransfer.getData('application/x-cardioemi-runs');
        const folder = e.dataTransfer.getData('application/x-cardioemi-folder');
        if (runs) {
            await this.moveRuns(JSON.parse(runs), path);
        } else if (folder) {
            const leaf = folder.split('/').pop();
            const target = path ? `${path}/${leaf}` : leaf;
            if (target !== folder) await this.moveFolder(folder, target);
        }
    });
};

// --------------------- selection + current run ---------------------

App.prototype.setRunsChecked = async function(names, on) {
    names.forEach(n => on ? this.runChecked.add(n) : this.runChecked.delete(n));
    this._saveRunState();
    this.renderRunBrowser();
    if (on) await this.ensureIterationsLocal(names);
    await this.onCompareSelectionChange();
};

// Checked runs whose iterations are on this machine, in tree order - what the
// comparison charts plot.
App.prototype.getCompareSelection = function() {
    return this.runIndex
        .filter(r => this.runChecked.has(r.name) && r.local && r.local.iterations)
        .map(r => r.name);
};

// Fetch iterations for checked runs that only exist on a cluster.
App.prototype.ensureIterationsLocal = async function(names) {
    const todo = [];
    for (const n of names) {
        const r = this.runByName[n];
        if (!r || (r.local && r.local.iterations)) continue;
        const cid = Object.keys(r.remote).find(c => r.remote[c].iterations
            && !(this.runClusters[c] || {}).error);
        if (cid) todo.push([n, cid]);
    }
    if (!todo.length) return;
    this._setRunStatus(`Fetching iterations for ${todo.length} run(s)…`);
    for (const [n, cid] of todo) {
        try {
            await this.getRunner(cid).downloadIterations(n);
        } catch (e) {
            console.warn(`Fetching iterations for ${n} from ${cid} failed:`, e);
        }
    }
    await this.loadRunIndex();
};

App.prototype.setCurrentRun = function(name) {
    this.currentRun = name;
    this.selectedSimulation = name;
    this._saveRunState();
    this.renderRunBrowser();
    this.updateCurrentRunUI();
};

// Expand the folders above a run, select it and scroll to it (job list click).
App.prototype.revealRun = function(name) {
    const r = this.runByName[name];
    if (!r) return;
    if (r.folder) {
        let acc = '';
        for (const part of r.folder.split('/')) {
            acc = acc ? `${acc}/${part}` : part;
            this.runExpanded.add(`f:${acc}`);
        }
    } else {
        this.runExpanded.add('u');
        this.runExpanded.add(`m:${r.mesh}`);
    }
    this.setCurrentRun(name);
    const row = document.querySelector(`#runs-tree .rb-run[data-name="${CSS.escape(name)}"]`);
    if (row) row.scrollIntoView({ block: 'nearest' });
};

// Results section: which run Load/Download/Video act on, and where its data is.
App.prototype.updateCurrentRunUI = function() {
    const labelEl = document.getElementById('current-run-label');
    const dlBtn = document.getElementById('download-cluster-results');
    const r = this.currentRun && this.runByName[this.currentRun];
    this._currentRunCluster = null;
    if (!labelEl) return;
    if (!r) {
        labelEl.textContent = 'click a run in Runs above';
        labelEl.title = '';
        if (dlBtn) dlBtn.style.display = 'none';
        return;
    }
    labelEl.textContent = this._runDisplayName(r, true);
    labelEl.title = r.name;
    // Download from a cluster that has the full results - the active run
    // target first - unless they are already here.
    const withResults = Object.keys(r.remote).filter(c => r.remote[c].results);
    withResults.sort((a, b) => (b === this.runTarget) - (a === this.runTarget));
    if (dlBtn) {
        if (withResults.length && !(r.local && r.local.results)) {
            this._currentRunCluster = withResults[0];
            const c = this.runClusters[withResults[0]] || { label: withResults[0] };
            dlBtn.textContent = `Download from ${c.label}`;
            dlBtn.style.display = 'block';
        } else {
            dlBtn.style.display = 'none';
        }
    }
};

// --------------------- folders ---------------------

App.prototype._runsApi = async function(path, body) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    return data;
};

App.prototype.promptNewFolder = async function(parent) {
    const name = window.prompt(parent ? `New folder inside "${parent}"` : 'New folder (use / for nesting, e.g. weak scaling/L30)');
    if (!name || !name.trim()) return;
    const path = parent ? `${parent}/${name.trim()}` : name.trim();
    try {
        await this._runsApi('/api/runs/folder', { op: 'create', path });
        if (parent) this.runExpanded.add(`f:${parent}`);
        this._saveRunState();
        await this.loadRunIndex();
    } catch (e) {
        alert('Could not create folder: ' + e.message);
    }
};

App.prototype.promptRenameFolder = async function(path) {
    const parts = path.split('/');
    const name = window.prompt(`Rename folder "${path}"`, parts[parts.length - 1]);
    if (!name || !name.trim() || name.trim() === parts[parts.length - 1]) return;
    parts[parts.length - 1] = name.trim();
    await this.moveFolder(path, parts.join('/'));
};

App.prototype.moveFolder = async function(path, newPath) {
    try {
        await this._runsApi('/api/runs/folder', { op: 'move', path, new_path: newPath });
        // Keep the moved folders open where they were open.
        for (const key of [...this.runExpanded]) {
            if (key === `f:${path}` || key.startsWith(`f:${path}/`)) {
                this.runExpanded.delete(key);
                this.runExpanded.add(`f:${newPath}${key.slice(2 + path.length)}`);
            }
        }
        this._saveRunState();
        await this.loadRunIndex();
        await this.onCompareSelectionChange();  // legends name folders
    } catch (e) {
        alert('Could not move folder: ' + e.message);
    }
};

App.prototype.moveRuns = async function(names, folder) {
    if (!names.length) return;
    try {
        await this._runsApi('/api/runs/move', { names, folder });
        if (folder) this.runExpanded.add(`f:${folder}`);
        this._saveRunState();
        await this.loadRunIndex();
        await this.onCompareSelectionChange();
    } catch (e) {
        alert('Could not move runs: ' + e.message);
    }
};

App.prototype.openMovePopover = function() {
    if (!this.runChecked.size) return;
    const pop = document.getElementById('runs-move-popover');
    pop.style.display = 'flex';
    const input = document.getElementById('runs-move-target');
    input.value = '';
    input.placeholder = `Folder for ${this.runChecked.size} run(s), e.g. weak scaling/L30 (empty: Unfiled)`;
    input.focus();
};

App.prototype.submitMovePopover = async function() {
    const input = document.getElementById('runs-move-target');
    document.getElementById('runs-move-popover').style.display = 'none';
    await this.moveRuns([...this.runChecked], input.value.trim());
};

// --------------------- delete ---------------------

// Delete runs everywhere. `body` goes to /api/runs/delete as is.
App.prototype._deleteRuns = async function(body) {
    let result;
    try {
        result = await this._runsApi('/api/runs/delete', body);
    } catch (e) {
        alert('Delete failed: ' + e.message);
        return null;
    }
    const removed = new Set(result.removed || []);
    removed.forEach(n => this.runChecked.delete(n));
    this._dropJobsForRuns(removed);
    const notes = [];
    if (result.skipped && result.skipped.length) {
        notes.push(`Skipped ${result.skipped.length} run(s) whose job is still queued or running.`);
    }
    if (result.errors && result.errors.length) {
        notes.push('Errors:\n' + result.errors.join('\n'));
    }
    if (notes.length) alert(notes.join('\n\n'));
    await this.loadRunIndex();
    await this.onCompareSelectionChange();
    return result;
};

// Remove job-list entries whose run is gone.
App.prototype._dropJobsForRuns = function(names) {
    let changed = false;
    for (const [jobId, job] of Object.entries(this.clusterJobs || {})) {
        if (!names.has(job.out_name)) continue;
        this.getRunner(job.cluster || 'karolina').stopPolling(jobId);
        delete this.clusterJobs[jobId];
        const entry = document.getElementById(`cluster-job-${jobId}`);
        if (entry) entry.remove();
        changed = true;
    }
    if (changed) {
        this.saveClusterJobs();
        this.renderMeshFilter();
    }
};

App.prototype._describeRuns = function(names) {
    const runs = names.map(n => this.runByName[n]).filter(Boolean);
    const size = runs.reduce((s, r) => s + this._runSize(r), 0);
    const where = new Set();
    runs.forEach(r => {
        if (r.local) where.add('this machine');
        Object.keys(r.remote).forEach(c => where.add((this.runClusters[c] || { label: c }).label));
    });
    return `${runs.length} run(s), ${this._fmtSize(size)} on ${[...where].join(', ') || 'nowhere'}`;
};

App.prototype.deleteCheckedRuns = async function() {
    const names = [...this.runChecked];
    if (!names.length) return;
    if (!window.confirm(`Delete ${this._describeRuns(names)}?\n\n`
            + 'This removes the run folders, viz caches and every cluster copy. It cannot be undone.')) return;
    await this._deleteRuns({ names });
};

App.prototype.deleteFolder = async function(path, runs) {
    const what = runs.length ? this._describeRuns(runs.map(r => r.name)) : 'no runs';
    if (!window.confirm(`Delete folder "${path}" with ${what}?\n\n`
            + 'Every run in it (and its subfolders) is removed everywhere. It cannot be undone.')) return;
    await this._deleteRuns({ folders: [path] });
};

App.prototype.deleteFailedRuns = async function() {
    const { runs, copies } = this._failedCleanup();
    if (!runs.length && !copies.length) return;
    const lines = [];
    if (runs.length) {
        lines.push(`${runs.length} run(s) without any results:`);
        runs.slice(0, 8).forEach(n => lines.push('  ' + n));
        if (runs.length > 8) lines.push(`  … and ${runs.length - 8} more`);
    }
    if (copies.length) {
        lines.push(`${copies.length} empty cluster copies of runs that have data elsewhere (left over from code syncs).`);
    }
    if (!window.confirm(`Delete failed runs?\n\n${lines.join('\n')}\n\n`
            + 'Runs still in the job list are kept. Only folders without iterations or results are removed.')) return;
    await this._deleteRuns({ names: runs, copies, only_if_empty: true });
};

// Job-list dismissal of a finished job: its run goes too if it produced
// nothing (checked again server-side, where the data lives).
App.prototype.deleteRunsIfEmpty = async function(names) {
    if (!names.length) return;
    try {
        const result = await this._runsApi('/api/runs/delete', { names, only_if_empty: true });
        (result.removed || []).forEach(n => this.runChecked.delete(n));
    } catch (e) {
        console.warn('Deleting failed runs after dismissal failed:', e);
    }
    await this.loadRunIndex();
};
