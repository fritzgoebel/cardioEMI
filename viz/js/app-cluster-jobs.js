// app-cluster-jobs.js - Persistent remote-cluster jobs list, mesh filter, and removal
//
// Owns localStorage persistence for `app.clusterJobs` and `app.meshFilter`,
// the "Show meshes" multi-select filter UI, and the per-entry remove flow
// (with optional remote+local data deletion).

const CLUSTER_TERMINAL_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];

// Format a job/simulation as three concise lines for compact display.
// Accepts either a cluster job dict (mesh_name, num_ranks, solver_backend, ...)
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

App.prototype.setupClusterJobsPersistence = function() {
    this.meshFilter = null;  // null = show all; otherwise array of visible mesh names

    const filterBtn = document.getElementById('cluster-jobs-mesh-filter-btn');
    const filterPanel = document.getElementById('cluster-jobs-mesh-filter-panel');
    filterBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        filterPanel.style.display = filterPanel.style.display === 'none' ? 'block' : 'none';
    });
    document.addEventListener('click', (e) => {
        if (!document.getElementById('cluster-jobs-mesh-filter').contains(e.target)) {
            filterPanel.style.display = 'none';
        }
    });

    const clearBtn = document.getElementById('cluster-jobs-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => this.clearClusterJobsList());
    }

    this.loadClusterJobs();
};

App.prototype.clearClusterJobsList = function() {
    const jobIds = Object.keys(this.clusterJobs || {});
    if (jobIds.length === 0) return;
    const ok = window.confirm(
        `Remove all ${jobIds.length} job(s) from the list?\n\n` +
        'This does NOT cancel jobs on the cluster and keeps every run with results. ' +
        'Runs of finished jobs that produced no results (failed) are deleted - ' +
        'the job list was what kept them around.'
    );
    if (!ok) return;
    const finished = [];
    for (const jobId of jobIds) {
        const job = this.clusterJobs[jobId];
        this.getRunner(job?.cluster || 'karolina').stopPolling(jobId);
        if (job?.out_name && CLUSTER_TERMINAL_STATES.includes(job.status)) finished.push(job.out_name);
    }
    this.clusterJobs = {};
    const list = document.getElementById('cluster-jobs-list');
    if (list) list.innerHTML = '';
    this.saveClusterJobs();
    this.renderMeshFilter();
    this.deleteRunsIfEmpty(finished);
};

App.prototype.saveClusterJobs = function() {
    try {
        const stripped = {};
        for (const [jobId, job] of Object.entries(this.clusterJobs)) {
            const copy = { ...job };
            delete copy._iterationsDownloaded;
            delete copy._logFetched;
            stripped[jobId] = copy;
        }
        localStorage.setItem('clusterJobs', JSON.stringify(stripped));
    } catch (e) {
        console.warn('Failed to save cluster jobs:', e);
    }
};

App.prototype.saveMeshFilter = function() {
    try {
        localStorage.setItem('clusterJobsMeshFilter',
            this.meshFilter === null ? 'null' : JSON.stringify(this.meshFilter));
    } catch (e) {
        console.warn('Failed to save mesh filter:', e);
    }
};

App.prototype.loadClusterJobs = function() {
    let jobs = {};
    try {
        // Migration: jobs were stored under 'karolinaJobs' before multi-cluster
        // support; tag those with their (only possible) cluster.
        let raw = localStorage.getItem('clusterJobs');
        if (!raw) {
            raw = localStorage.getItem('karolinaJobs');
            if (raw) {
                jobs = JSON.parse(raw) || {};
                for (const job of Object.values(jobs)) job.cluster = job.cluster || 'karolina';
                localStorage.setItem('clusterJobs', JSON.stringify(jobs));
                localStorage.removeItem('karolinaJobs');
                raw = null;
            }
        }
        if (raw) jobs = JSON.parse(raw) || {};
        for (const job of Object.values(jobs)) job.cluster = job.cluster || 'karolina';
    } catch (e) {
        console.warn('Failed to load persisted cluster jobs:', e);
        jobs = {};
    }

    let filter = null;
    try {
        const rawFilter = localStorage.getItem('clusterJobsMeshFilter')
            ?? localStorage.getItem('karolinaJobsMeshFilter');
        if (rawFilter !== null && rawFilter !== 'null') {
            filter = JSON.parse(rawFilter);
            if (!Array.isArray(filter)) filter = null;
        }
    } catch (e) {
        filter = null;
    }
    this.meshFilter = filter;

    this.clusterJobs = jobs;

    if (Object.keys(jobs).length === 0) {
        this.renderMeshFilter();
        return;
    }

    if (this.isRemote()) {
        document.getElementById('cluster-job-section').style.display = 'block';
    }

    for (const jobId of Object.keys(jobs)) {
        const job = jobs[jobId];
        this.renderJobEntry(job);
        this._applyStatusToDom(jobId, { status: job.status, log: job.log });
        if (!CLUSTER_TERMINAL_STATES.includes(job.status)) {
            this.getRunner(job.cluster || 'karolina').startPolling(jobId, (data) => {
                this.updateJobStatus(jobId, data);
            }, job.out_name);
        }
    }

    this.renderMeshFilter();
    this.applyMeshFilter();
};

App.prototype.getDistinctMeshNames = function() {
    const names = new Set();
    for (const job of Object.values(this.clusterJobs)) {
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
    const btn = document.getElementById('cluster-jobs-mesh-filter-btn');
    const panel = document.getElementById('cluster-jobs-mesh-filter-panel');
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
    for (const [jobId, job] of Object.entries(this.clusterJobs)) {
        const entry = document.getElementById(`cluster-job-${jobId}`);
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
    const entry = document.getElementById(`cluster-job-${jobId}`);
    if (!entry || !data.status) return;
    if (data.log) entry.querySelector('.job-log').textContent = data.log;
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
    } else if (CLUSTER_TERMINAL_STATES.includes(s) && s !== 'COMPLETED') {
        statusEl.style.color = '#e94560';
        cancelBtn.style.display = 'none';
    }
};

// Wrap renderJobEntry so this module owns the ✕ button + remove popover. The
// base renderJobEntry (in app-cluster.js) stays focused on Cancel/Download/Log
// — anything tied to *removal* lives here.
const _baseRenderJobEntry = App.prototype.renderJobEntry;
App.prototype.renderJobEntry = function(jobInfo) {
    const existed = !!document.getElementById(`cluster-job-${jobInfo.job_id}`);
    _baseRenderJobEntry.call(this, jobInfo);
    if (existed) return;
    this._injectRemoveControls(jobInfo.job_id);
};

App.prototype._injectRemoveControls = function(jobId) {
    const entry = document.getElementById(`cluster-job-${jobId}`);
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
        <button class="btn btn-remove-list" style="font-size:0.75em; padding:2px 8px; background:#555; color:#eee;" title="Keeps the run if it has results; a finished run without results (failed) is deleted">From list only</button>
        <button class="btn btn-remove-data" style="font-size:0.75em; padding:2px 8px; background:#a23030; color:#fff;" title="Delete the run everywhere (local, viz cache, every cluster copy)">+ Delete data</button>
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
    const job = this.clusterJobs[jobId];
    if (!job) return;

    if (deleteData) {
        const ok = window.confirm(
            `Delete the run of job ${jobId} everywhere (local folder, viz cache, every cluster copy)? ` +
            'A still running job is cancelled first. This cannot be undone.'
        );
        if (!ok) return;
        if (!CLUSTER_TERMINAL_STATES.includes(job.status)) {
            try {
                await this.getRunner(job.cluster || 'karolina').cancel(jobId);
                // The cached listing still marks the run active, which the
                // delete would skip - re-list first.
                await this.loadRunIndex({ refresh: true });
            } catch (e) {
                console.warn(`Cancelling job ${jobId} failed:`, e);
            }
        }
    }

    this.getRunner(job.cluster || 'karolina').stopPolling(jobId);
    delete this.clusterJobs[jobId];
    const entry = document.getElementById(`cluster-job-${jobId}`);
    if (entry) entry.remove();
    this.saveClusterJobs();
    this.renderMeshFilter();

    if (!job.out_name) return;
    if (deleteData) {
        await this._deleteRuns({ names: [job.out_name] });
    } else if (CLUSTER_TERMINAL_STATES.includes(job.status)) {
        // The job list was what kept a failed run around; it goes with it.
        await this.deleteRunsIfEmpty([job.out_name]);
    }
};
