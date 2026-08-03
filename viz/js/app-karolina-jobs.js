// app-karolina-jobs.js - Persistent Karolina jobs list, mesh filter, and removal
//
// Owns localStorage persistence for `app.karolinaJobs` and `app.meshFilter`,
// the "Show meshes" multi-select filter UI, and the per-entry remove flow
// (with optional remote+local data deletion).

const KAROLINA_TERMINAL_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];

// Format a job/simulation as three concise lines for compact display.
// Accepts either a Karolina job dict (mesh_name, num_ranks, solver_backend, ...)
// or a conditions-derived dict (mesh, nRanks, solver, ...).
App.prototype.formatSimulationLabel = function(info) {
    const mesh = info.mesh_name || info.mesh || '';
    const nRanks = info.num_ranks || info.nRanks;
    const solver = info.solver_backend || info.solver || '';
    const precond = info.preconditioner || '';
    const localSolver = info.localSolver || '';

    const line1 = nRanks ? `${mesh || '?'} · ${nRanks}r` : (mesh || '?');
    const line2 = solver || '?';
    let line3 = precond;
    if (localSolver) line3 = line3 ? `${line3} / ${localSolver}` : localSolver;
    return { line1, line2, line3 };
};

// Single-line variant for dropdown options.
App.prototype.formatSimulationLabelInline = function(info) {
    const { line1, line2, line3 } = this.formatSimulationLabel(info);
    const solverPart = line3 ? `${line2} / ${line3}` : line2;
    return solverPart && solverPart !== '?' ? `${line1} — ${solverPart}` : line1;
};

App.prototype.setupKarolinaJobsPersistence = function() {
    this.meshFilter = null;  // null = show all; otherwise array of visible mesh names

    const filterBtn = document.getElementById('karolina-jobs-mesh-filter-btn');
    const filterPanel = document.getElementById('karolina-jobs-mesh-filter-panel');
    filterBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        filterPanel.style.display = filterPanel.style.display === 'none' ? 'block' : 'none';
    });
    document.addEventListener('click', (e) => {
        if (!document.getElementById('karolina-jobs-mesh-filter').contains(e.target)) {
            filterPanel.style.display = 'none';
        }
    });

    const clearBtn = document.getElementById('karolina-jobs-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => this.clearKarolinaJobsList());
    }

    this.loadKarolinaJobs();
};

App.prototype.clearKarolinaJobsList = function() {
    const jobIds = Object.keys(this.karolinaJobs || {});
    if (jobIds.length === 0) return;
    const ok = window.confirm(
        `Remove all ${jobIds.length} job(s) from the list?\n\n` +
        'This only clears the local list — it does NOT cancel jobs on Karolina ' +
        'or delete any data (remote dirs or downloaded results stay put).'
    );
    if (!ok) return;
    for (const jobId of jobIds) {
        this.karolinaRunner.stopPolling(jobId);
    }
    this.karolinaJobs = {};
    const list = document.getElementById('karolina-jobs-list');
    if (list) list.innerHTML = '';
    this.saveKarolinaJobs();
    this.renderMeshFilter();
};

App.prototype.saveKarolinaJobs = function() {
    try {
        const stripped = {};
        for (const [jobId, job] of Object.entries(this.karolinaJobs)) {
            const copy = { ...job };
            delete copy._iterationsDownloaded;
            stripped[jobId] = copy;
        }
        localStorage.setItem('karolinaJobs', JSON.stringify(stripped));
    } catch (e) {
        console.warn('Failed to save Karolina jobs:', e);
    }
};

App.prototype.saveMeshFilter = function() {
    try {
        localStorage.setItem('karolinaJobsMeshFilter',
            this.meshFilter === null ? 'null' : JSON.stringify(this.meshFilter));
    } catch (e) {
        console.warn('Failed to save mesh filter:', e);
    }
};

App.prototype.loadKarolinaJobs = function() {
    let jobs = {};
    try {
        const raw = localStorage.getItem('karolinaJobs');
        if (raw) jobs = JSON.parse(raw) || {};
    } catch (e) {
        console.warn('Failed to load persisted Karolina jobs:', e);
        jobs = {};
    }

    let filter = null;
    try {
        const rawFilter = localStorage.getItem('karolinaJobsMeshFilter');
        if (rawFilter !== null && rawFilter !== 'null') {
            filter = JSON.parse(rawFilter);
            if (!Array.isArray(filter)) filter = null;
        }
    } catch (e) {
        filter = null;
    }
    this.meshFilter = filter;

    this.karolinaJobs = jobs;

    if (Object.keys(jobs).length === 0) {
        this.renderMeshFilter();
        return;
    }

    if (this.runTarget === 'karolina') {
        document.getElementById('karolina-job-section').style.display = 'block';
    }

    for (const jobId of Object.keys(jobs)) {
        const job = jobs[jobId];
        this.renderJobEntry(job);
        this._applyStatusToDom(jobId, { status: job.status });
        if (!KAROLINA_TERMINAL_STATES.includes(job.status)) {
            this.karolinaRunner.startPolling(jobId, (data) => {
                this.updateJobStatus(jobId, data);
                if (data.out_name) this.karolinaJobs[jobId].out_name = data.out_name;
            });
        }
    }

    this.renderMeshFilter();
    this.applyMeshFilter();
};

App.prototype.getDistinctMeshNames = function() {
    const names = new Set();
    for (const job of Object.values(this.karolinaJobs)) {
        if (job.mesh_name) names.add(job.mesh_name);
    }
    return Array.from(names).sort();
};

App.prototype.isMeshVisible = function(meshName) {
    if (this.meshFilter === null) return true;
    if (!meshName) return true;  // jobs without mesh_name always visible (e.g. legacy)
    return this.meshFilter.includes(meshName);
};

App.prototype.renderMeshFilter = function() {
    const btn = document.getElementById('karolina-jobs-mesh-filter-btn');
    const panel = document.getElementById('karolina-jobs-mesh-filter-panel');
    const meshes = this.getDistinctMeshNames();

    if (meshes.length === 0) {
        btn.textContent = 'All';
        panel.innerHTML = '<div style="color:#888; font-size:0.85em;">No jobs yet</div>';
        return;
    }

    if (this.meshFilter !== null) {
        const pruned = this.meshFilter.filter(m => meshes.includes(m));
        if (pruned.length !== this.meshFilter.length) {
            this.meshFilter = pruned;
            this.saveMeshFilter();
        }
    }

    const visible = meshes.filter(m => this.isMeshVisible(m));
    btn.textContent = (visible.length === meshes.length)
        ? `All (${meshes.length})`
        : `${visible.length} / ${meshes.length}`;

    const allChecked = (this.meshFilter === null || this.meshFilter.length === meshes.length);
    let html = `
        <label style="display:block; padding:2px 0; color:#eee; font-size:0.85em; cursor:pointer;">
            <input type="checkbox" class="mesh-filter-all" ${allChecked ? 'checked' : ''}> All
        </label>
        <hr style="border:0; border-top:1px solid #444; margin:4px 0;">
    `;
    for (const m of meshes) {
        const checked = this.isMeshVisible(m);
        html += `
            <label style="display:block; padding:2px 0; color:#eee; font-size:0.85em; cursor:pointer;">
                <input type="checkbox" class="mesh-filter-item" data-mesh="${m}" ${checked ? 'checked' : ''}> ${m}
            </label>
        `;
    }
    panel.innerHTML = html;

    panel.querySelector('.mesh-filter-all').addEventListener('change', (e) => {
        this.meshFilter = e.target.checked ? null : [];
        this.saveMeshFilter();
        this.renderMeshFilter();
        this.applyMeshFilter();
    });
    panel.querySelectorAll('.mesh-filter-item').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const mesh = e.target.dataset.mesh;
            const all = this.getDistinctMeshNames();
            let current = this.meshFilter === null ? [...all] : [...this.meshFilter];
            if (e.target.checked) {
                if (!current.includes(mesh)) current.push(mesh);
            } else {
                current = current.filter(x => x !== mesh);
            }
            this.meshFilter = (current.length === all.length) ? null : current;
            this.saveMeshFilter();
            this.renderMeshFilter();
            this.applyMeshFilter();
        });
    });
};

App.prototype.applyMeshFilter = function() {
    for (const [jobId, job] of Object.entries(this.karolinaJobs)) {
        const entry = document.getElementById(`karolina-job-${jobId}`);
        if (!entry) continue;
        entry.style.display = this.isMeshVisible(job.mesh_name) ? '' : 'none';
    }
};

App.prototype.ensureMeshInFilter = function(meshName) {
    if (!meshName || this.meshFilter === null) return;
    if (!this.meshFilter.includes(meshName)) {
        this.meshFilter.push(meshName);
        const all = this.getDistinctMeshNames();
        if (all.length > 0 && this.meshFilter.length === all.length) {
            this.meshFilter = null;
        }
        this.saveMeshFilter();
    }
};

// Apply a status object to an existing entry's DOM without triggering side
// effects (no auto-download, no save). Used to restore last-known status on
// page load before live polling overwrites it.
App.prototype._applyStatusToDom = function(jobId, data) {
    const entry = document.getElementById(`karolina-job-${jobId}`);
    if (!entry || !data.status) return;
    const statusEl = entry.querySelector('.job-status');
    const cancelBtn = entry.querySelector('.btn-cancel');
    const downloadBtn = entry.querySelector('.btn-download');
    statusEl.textContent = data.status;
    const s = data.status;
    if (s === 'RUNNING') {
        statusEl.style.color = '#4ade80';
    } else if (s === 'PENDING') {
        statusEl.style.color = '#fbbf24';
    } else if (s === 'COMPLETED') {
        statusEl.style.color = '#4ade80';
        cancelBtn.style.display = 'none';
        downloadBtn.style.display = 'inline-block';
    } else if (KAROLINA_TERMINAL_STATES.includes(s) && s !== 'COMPLETED') {
        statusEl.style.color = '#e94560';
        cancelBtn.style.display = 'none';
    }
};

// Wrap renderJobEntry so this module owns the ✕ button + remove popover. The
// base renderJobEntry (in app-karolina.js) stays focused on Cancel/Download/Log
// — anything tied to *removal* lives here.
const _baseRenderJobEntry = App.prototype.renderJobEntry;
App.prototype.renderJobEntry = function(jobInfo) {
    const existed = !!document.getElementById(`karolina-job-${jobInfo.job_id}`);
    _baseRenderJobEntry.call(this, jobInfo);
    if (existed) return;
    this._injectRemoveControls(jobInfo.job_id);
};

App.prototype._injectRemoveControls = function(jobId) {
    const entry = document.getElementById(`karolina-job-${jobId}`);
    if (!entry) return;

    const header = entry.querySelector('.job-entry-header');
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn btn-remove';
    removeBtn.title = 'Remove';
    removeBtn.innerHTML = '&times;';
    removeBtn.style.cssText = 'font-size:0.85em; padding:0 6px; background:transparent; color:#888; border:1px solid #444; flex-shrink:0;';
    header.appendChild(removeBtn);

    // Popover with two confirmation actions, inserted just before the log <pre>
    const popover = document.createElement('div');
    popover.className = 'remove-popover';
    popover.style.cssText = 'display:none; margin-top:6px; padding:6px; background:#222; border:1px solid #555; border-radius:4px;';
    popover.innerHTML = `
        <div style="font-size:0.8em; color:#ccc; margin-bottom:4px;">Remove this job?</div>
        <button class="btn btn-remove-list" style="font-size:0.75em; padding:2px 8px; background:#555; color:#eee;">From list only</button>
        <button class="btn btn-remove-data" style="font-size:0.75em; padding:2px 8px; background:#a23030; color:#fff;">+ Delete data</button>
        <button class="btn btn-remove-cancel" style="font-size:0.75em; padding:2px 8px; background:transparent; color:#888;">Cancel</button>
    `;
    const logEl = entry.querySelector('.job-log');
    entry.insertBefore(popover, logEl);

    removeBtn.addEventListener('click', () => {
        popover.style.display = popover.style.display === 'none' ? 'block' : 'none';
    });
    popover.querySelector('.btn-remove-cancel').addEventListener('click', () => {
        popover.style.display = 'none';
    });
    popover.querySelector('.btn-remove-list').addEventListener('click', () => {
        popover.style.display = 'none';
        this.removeJob(jobId, false);
    });
    popover.querySelector('.btn-remove-data').addEventListener('click', () => {
        popover.style.display = 'none';
        this.removeJob(jobId, true);
    });
};

App.prototype.removeJob = async function(jobId, deleteData) {
    const job = this.karolinaJobs[jobId];
    if (!job) return;

    if (deleteData) {
        const ok = window.confirm(
            `Delete remote dir and local download for job ${jobId}? This cannot be undone.`
        );
        if (!ok) return;
        try {
            const resp = await fetch('/api/karolina/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, out_name: job.out_name })
            });
            const result = await resp.json();
            if (!resp.ok) {
                alert('Delete failed: ' + (result.error || 'unknown error'));
                return;
            }
            if (result.errors && result.errors.length) {
                console.warn('Partial delete failure:', result.errors);
            }
        } catch (e) {
            alert('Delete request failed: ' + e.message);
            return;
        }
    }

    this.karolinaRunner.stopPolling(jobId);
    delete this.karolinaJobs[jobId];
    const entry = document.getElementById(`karolina-job-${jobId}`);
    if (entry) entry.remove();
    this.saveKarolinaJobs();
    this.renderMeshFilter();
};
