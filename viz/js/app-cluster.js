// app-cluster.js - remote cluster run target: registry, connectivity (incl.
// OTP/2FA connect flow), container install, SLURM defaults, job entries and
// result downloads. Clusters are defined in viz/clusters.yml (editable via
// the "Clusters..." settings dialog here).

// Runner for the currently selected run target (falls back to the first
// configured cluster for cluster-agnostic uses like chart fetches).
Object.defineProperty(App.prototype, 'clusterRunner', {
    get() {
        const cid = (this.runTarget && this.runTarget !== 'local')
            ? this.runTarget
            : (this.clusters?.[0]?.id || 'karolina');
        return this.getRunner(cid);
    }
});

App.prototype.getRunner = function(clusterId) {
    if (!this.clusterRunners) this.clusterRunners = {};
    if (!this.clusterRunners[clusterId]) {
        this.clusterRunners[clusterId] = new ClusterRunner(clusterId);
    }
    return this.clusterRunners[clusterId];
};

App.prototype.isRemote = function() {
    return this.runTarget && this.runTarget !== 'local';
};

App.prototype.activeCluster = function() {
    if (!this.isRemote()) return null;
    return (this.clusters || []).find(c => c.id === this.runTarget) || null;
};

App.prototype.clusterLabel = function() {
    const c = this.activeCluster();
    return c ? c.label : 'cluster';
};

// --------------------- Run target setup ---------------------

App.prototype.setupRunTarget = async function() {
    const selector = document.getElementById('run-target');

    await this.refreshClusterList();

    document.getElementById('cluster-settings-btn').addEventListener('click',
        () => this.openClusterSettings());
    document.getElementById('cluster-connect-btn').addEventListener('click',
        () => this.startClusterConnect());
    document.getElementById('cluster-install-btn').addEventListener('click',
        () => this.startClusterInstall());

    selector.addEventListener('change', async () => {
        this.runTarget = selector.value;
        sessionStorage.setItem('runTarget', this.runTarget);
        await this.applyRunTarget();
    });

    if (this.runTarget !== 'local') {
        // Persisted target may have been deleted from the registry
        if (!(this.clusters || []).some(c => c.id === this.runTarget)) {
            this.runTarget = 'local';
            sessionStorage.setItem('runTarget', 'local');
        }
        selector.value = this.runTarget;
    }
    if (this.runTarget !== 'local') {
        this.applyRunTarget();
    }
};

App.prototype.refreshClusterList = async function() {
    const selector = document.getElementById('run-target');
    try {
        const resp = await fetch('/api/clusters');
        const data = await resp.json();
        this.clusters = (data.clusters || []).filter(c => !c.error);
    } catch (e) {
        console.error('Failed to load cluster registry:', e);
        this.clusters = [];
    }
    // Rebuild dropdown: local + one entry per cluster
    const current = selector.value;
    selector.innerHTML = '<option value="local">Local (Docker)</option>';
    for (const c of this.clusters) {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.label;
        selector.appendChild(opt);
    }
    if ([...selector.options].some(o => o.value === current)) selector.value = current;
};

App.prototype.applyRunTarget = async function(skipChecks = false) {
    const target = this.runTarget;
    const clusterOptions = document.getElementById('cluster-options');
    const containerStatusRow = document.getElementById('cluster-container-status-row');
    const statusDot = document.getElementById('cluster-status-dot');
    const refreshBtn = document.getElementById('refresh-mesh-list');
    const mpiRanksRow = document.getElementById('mpi-ranks').closest('.param-row');
    const jobSection = document.getElementById('cluster-job-section');
    const connectBtn = document.getElementById('cluster-connect-btn');
    const installBtn = document.getElementById('cluster-install-btn');

    if (this.isRemote()) {
        const cluster = this.activeCluster();
        clusterOptions.style.display = 'block';
        containerStatusRow.style.display = 'flex';
        statusDot.style.display = 'inline';
        refreshBtn.style.display = 'inline-block';
        mpiRanksRow.style.display = 'none';
        this.applyClusterSlurmDefaults(cluster);
        if (this.clusterJobs && Object.keys(this.clusterJobs).length > 0) {
            jobSection.style.display = 'block';
        }
        document.getElementById('export-video-remote').style.display = 'block';

        if (!skipChecks) await this.checkClusterConnectivity();
        this.loadSimulationList();
    } else {
        clusterOptions.style.display = 'none';
        containerStatusRow.style.display = 'none';
        statusDot.style.display = 'none';
        connectBtn.style.display = 'none';
        installBtn.style.display = 'none';
        refreshBtn.style.display = 'none';
        mpiRanksRow.style.display = 'flex';
        jobSection.style.display = 'none';
        document.getElementById('export-video-remote').style.display = 'none';
        this.refreshMeshList();
    }
};

App.prototype.applyClusterSlurmDefaults = function(cluster) {
    if (!cluster || !cluster.slurm) return;
    const s = cluster.slurm;
    const partSel = document.getElementById('cluster-partition');
    partSel.innerHTML = '';
    for (const p of (s.partitions && s.partitions.length ? s.partitions : [s.partition])) {
        if (!p) continue;
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        partSel.appendChild(opt);
    }
    if (s.partition) partSel.value = s.partition;
    document.getElementById('cluster-account').value = s.account || '';
    document.getElementById('cluster-walltime').value = s.walltime || '01:00:00';
    const ntasks = document.getElementById('cluster-ntasks');
    ntasks.value = s.ntasks_per_node || 128;
    ntasks.max = s.cores_per_node || 128;
    ntasks.dispatchEvent(new Event('input'));
    document.getElementById('cluster-walltime').dispatchEvent(new Event('input'));
};

App.prototype.checkClusterConnectivity = async function() {
    const statusDot = document.getElementById('cluster-status-dot');
    const connectBtn = document.getElementById('cluster-connect-btn');
    const installBtn = document.getElementById('cluster-install-btn');
    const containerEl = document.getElementById('cluster-container-status');
    const cluster = this.activeCluster();
    if (!cluster) return;

    statusDot.textContent = '...';
    statusDot.style.color = '#888';
    statusDot.title = 'Checking SSH...';
    connectBtn.style.display = 'none';
    installBtn.style.display = 'none';
    containerEl.textContent = '-';

    try {
        const result = await this.clusterRunner.checkConnectivity();
        const ok = result.available;
        statusDot.textContent = '●';
        statusDot.style.color = ok ? '#4ade80' : '#e94560';
        statusDot.title = ok ? 'SSH connected' : (result.needs_otp
            ? 'Not connected — click Connect to authenticate (OTP)'
            : 'SSH unreachable');

        if (!ok && result.needs_otp) {
            connectBtn.style.display = 'inline-block';
        }
        if (ok && result.containers) {
            const c = result.containers;
            const parts = [];
            parts.push(`DOLFINx: ${c.dolfinx ? 'ready' : 'missing'}`);
            parts.push(`Ginkgo: ${c.ginkgo ? 'ready' : 'missing'}`);
            containerEl.textContent = parts.join(' | ');
            containerEl.style.color = c.dolfinx ? '#4ade80' : '#e94560';
            if (!c.dolfinx || !c.ginkgo) {
                installBtn.style.display = 'inline-block';
            }
        } else if (!ok) {
            containerEl.textContent = result.needs_otp ? 'connect first' : 'unreachable';
        }
        if (ok) this.refreshMeshList();
    } catch (e) {
        statusDot.textContent = '●';
        statusDot.style.color = '#e94560';
        statusDot.title = 'SSH check failed';
    }
};

// --------------------- Interactive connect (OTP/2FA) ---------------------

App.prototype.startClusterConnect = async function() {
    const modal = document.getElementById('cluster-connect-modal');
    const promptEl = document.getElementById('cluster-connect-prompt');
    const inputEl = document.getElementById('cluster-connect-input');
    const sendBtn = document.getElementById('cluster-connect-send');
    const cancelBtn = document.getElementById('cluster-connect-cancel');
    const titleEl = document.getElementById('cluster-connect-title');
    const runner = this.clusterRunner;

    titleEl.textContent = `Connect to ${this.clusterLabel()}`;
    promptEl.textContent = 'Opening SSH connection...';
    inputEl.value = '';
    inputEl.style.display = 'none';
    sendBtn.style.display = 'none';
    modal.style.display = 'flex';

    let cancelled = false;
    let answered = false;

    const cleanup = () => {
        modal.style.display = 'none';
        sendBtn.onclick = null;
        cancelBtn.onclick = null;
        inputEl.onkeydown = null;
    };
    cancelBtn.onclick = () => {
        cancelled = true;
        cleanup();
        runner.connectCancel().catch(() => {});
    };

    const submit = async () => {
        const text = inputEl.value;
        inputEl.value = '';
        inputEl.style.display = 'none';
        sendBtn.style.display = 'none';
        promptEl.textContent = 'Authenticating...';
        answered = true;
        await runner.connectInput(text);
    };
    sendBtn.onclick = submit;
    inputEl.onkeydown = (e) => { if (e.key === 'Enter') submit(); };

    try {
        let state = await runner.connect();
        while (!cancelled) {
            if (state.phase === 'connected') {
                cleanup();
                await this.checkClusterConnectivity();
                return;
            }
            if (state.phase === 'failed') {
                promptEl.textContent = 'Connection failed: ' + (state.error || 'unknown error');
                inputEl.style.display = 'none';
                sendBtn.style.display = 'none';
                return;  // leave modal open so the user can read the error
            }
            if (state.phase === 'prompt' && !answered) {
                promptEl.textContent = state.prompt || 'Input required:';
                inputEl.style.display = 'block';
                sendBtn.style.display = 'inline-block';
                inputEl.focus();
            }
            if (state.phase !== 'prompt') answered = false;
            await new Promise(r => setTimeout(r, 700));
            state = await runner.connectState();
        }
    } catch (e) {
        promptEl.textContent = 'Connect failed: ' + e.message;
    }
};

// --------------------- Install ---------------------

App.prototype.startClusterInstall = async function() {
    const label = this.clusterLabel();
    const ok = window.confirm(
        `Install cardioEMI on ${label}?\n\n` +
        'This creates the remote directory layout, syncs the project code, ' +
        'and installs the container images (uploaded, streamed from another ' +
        'cluster, or pulled from the Docker registry). Container transfer can ' +
        'take a while.');
    if (!ok) return;

    const installBtn = document.getElementById('cluster-install-btn');
    const outputEl = document.getElementById('cluster-install-output');
    installBtn.disabled = true;
    installBtn.textContent = 'Installing...';
    outputEl.style.display = 'block';
    outputEl.textContent = '';

    try {
        const result = await this.clusterRunner.install((text) => {
            outputEl.textContent += text;
            outputEl.scrollTop = outputEl.scrollHeight;
        });
        outputEl.textContent += result.success
            ? '\nInstall finished.\n'
            : '\nInstall did not complete.\n';
        await this.checkClusterConnectivity();
    } catch (e) {
        outputEl.textContent += '\nInstall failed: ' + e.message + '\n';
    } finally {
        installBtn.disabled = false;
        installBtn.textContent = 'Install';
    }
};

// --------------------- Cluster settings dialog ---------------------

App.prototype.openClusterSettings = function(editId = null) {
    const modal = document.getElementById('cluster-settings-modal');
    const listEl = document.getElementById('cluster-settings-list');
    modal.style.display = 'flex';

    const renderList = () => {
        listEl.innerHTML = '';
        for (const c of this.clusters || []) {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex; align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid #333;';
            row.innerHTML = `
                <span style="flex:1; color:#eee;">${c.label} <span style="color:#777; font-size:0.85em;">(${c.user ? c.user + '@' : ''}${c.host})</span></span>
                <button class="btn btn-edit-cluster" data-id="${c.id}" style="font-size:0.75em; padding:2px 8px;">Edit</button>
                <button class="btn btn-del-cluster" data-id="${c.id}" style="font-size:0.75em; padding:2px 8px; background:#a23030; color:#fff;">Delete</button>
            `;
            listEl.appendChild(row);
        }
        listEl.querySelectorAll('.btn-edit-cluster').forEach(b =>
            b.addEventListener('click', () => showForm(b.dataset.id)));
        listEl.querySelectorAll('.btn-del-cluster').forEach(b =>
            b.addEventListener('click', async () => {
                if (!window.confirm(`Remove cluster "${b.dataset.id}" from the registry? ` +
                                    '(No remote data is touched.)')) return;
                await fetch(`/api/clusters/${b.dataset.id}`, { method: 'DELETE' });
                await this.refreshClusterList();
                renderList();
            }));
    };

    const formEl = document.getElementById('cluster-settings-form');
    const el = (name) => document.getElementById('cluster-field-' + name);

    const showForm = (cid) => {
        formEl.style.display = 'block';
        const c = cid ? (this.clusters || []).find(x => x.id === cid) : null;
        el('id').value = c ? c.id : '';
        el('id').disabled = !!c;
        el('label').value = c ? c.label : '';
        el('host').value = c ? c.host : '';
        el('user').value = c?.user || '';
        el('identity_file').value = c?.identity_file || '';
        el('needs_otp').checked = !!c?.needs_otp;
        el('remote_path').value = c?.remote_path || '';
        el('account').value = c?.slurm?.account || '';
        el('partition').value = c?.slurm?.partition || '';
        el('partitions').value = (c?.slurm?.partitions || []).join(', ');
        el('ntasks_per_node').value = c?.slurm?.ntasks_per_node || 128;
        el('cores_per_node').value = c?.slurm?.cores_per_node || 128;
        el('walltime').value = c?.slurm?.walltime || '01:00:00';
    };

    document.getElementById('cluster-settings-add').onclick = () => showForm(null);
    document.getElementById('cluster-settings-close').onclick = () => {
        modal.style.display = 'none';
        formEl.style.display = 'none';
    };
    document.getElementById('cluster-field-cancel').onclick = () => {
        formEl.style.display = 'none';
    };
    document.getElementById('cluster-field-save').onclick = async () => {
        const id = el('id').value.trim().toLowerCase();
        const partitions = el('partitions').value.split(',')
            .map(s => s.trim()).filter(Boolean);
        const cfg = {
            label: el('label').value.trim() || id,
            host: el('host').value.trim(),
            user: el('user').value.trim() || null,
            identity_file: el('identity_file').value.trim() || null,
            needs_otp: el('needs_otp').checked,
            remote_path: el('remote_path').value.trim(),
            slurm: {
                account: el('account').value.trim(),
                partition: el('partition').value.trim(),
                partitions: partitions.length ? partitions : [el('partition').value.trim()],
                ntasks_per_node: parseInt(el('ntasks_per_node').value) || 128,
                cores_per_node: parseInt(el('cores_per_node').value) || 128,
                walltime: el('walltime').value.trim() || '01:00:00',
            },
        };
        try {
            const resp = await fetch('/api/clusters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, cfg }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || 'save failed');
            formEl.style.display = 'none';
            await this.refreshClusterList();
            renderList();
            if (this.runTarget === id) this.applyRunTarget();
        } catch (e) {
            alert('Save failed: ' + e.message);
        }
    };

    renderList();
    if (editId) showForm(editId);
};

// --------------------- SLURM sizing options ---------------------

App.prototype.setupClusterOptions = function() {
    // Rank counts are per mesh now (batch rows in the mesh panel, see
    // app-mesh-batch.js); this section only holds the shared SLURM fields.
    this.clusterJobs = {};
    this.compareDatasets = [];

    // Persistence + mesh filter + remove flow live in app-cluster-jobs.js
    this.setupClusterJobsPersistence();
};

// --------------------- Job entries ---------------------

App.prototype.renderJobEntry = function(jobInfo) {
    const container = document.getElementById('cluster-jobs-list');
    const jobId = jobInfo.job_id;
    const entryId = `cluster-job-${jobId}`;

    if (document.getElementById(entryId)) return;

    const entry = document.createElement('div');
    entry.id = entryId;
    entry.style.cssText = 'border:1px solid #444; border-radius:6px; padding:8px; margin-bottom:8px; background:#1a1a1a;';
    const lbl = this.formatSimulationLabel(jobInfo);
    const clusterTag = jobInfo.cluster
        ? `<span style="font-size:0.75em; color:#7aa2f7; border:1px solid #3b4261; border-radius:3px; padding:0 4px; margin-left:4px;">${jobInfo.cluster}</span>`
        : '';
    entry.innerHTML = `
        <div class="job-entry-header" style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
            <div class="job-entry-title" style="min-width:0; flex:1;" title="${jobInfo.out_name || jobId}">
                <div style="font-weight:bold; color:#ccc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${lbl.line1}${clusterTag}</div>
                <div style="font-size:0.85em; color:#aaa;">${lbl.line2}</div>
                <div style="font-size:0.85em; color:#888;">${lbl.line3 || ''}</div>
            </div>
        </div>
        <div class="param-row" style="margin:4px 0;">
            <label>Status:</label>
            <span class="job-status" style="font-weight:bold;">PENDING</span>
        </div>
        <div class="job-actions" style="margin-top:6px;">
            <button class="btn btn-danger btn-cancel" style="font-size:0.75em; padding:2px 8px;">Cancel</button>
            <button class="btn btn-success btn-download" style="font-size:0.75em; padding:2px 8px; display:none;">Download</button>
            <button class="btn btn-toggle-log" style="font-size:0.75em; padding:2px 8px; background:#555; color:#ccc;">Log</button>
        </div>
        <pre class="job-log output-console" style="max-height:150px; display:none; margin-top:6px; font-size:0.75em;"></pre>
    `;

    const runner = this.getRunner(jobInfo.cluster || 'karolina');

    entry.querySelector('.btn-cancel').addEventListener('click', async () => {
        try {
            await runner.cancel(jobId);
            runner.stopPolling(jobId);
            entry.querySelector('.job-status').textContent = 'CANCELLED';
            entry.querySelector('.job-status').style.color = '#e94560';
            entry.querySelector('.btn-cancel').style.display = 'none';
        } catch (e) {
            alert('Cancel failed: ' + e.message);
        }
    });

    entry.querySelector('.btn-download').addEventListener('click', async () => {
        const btn = entry.querySelector('.btn-download');
        btn.disabled = true;
        btn.textContent = 'Downloading...';
        try {
            const outName = this.clusterJobs[jobId]?.out_name;
            await runner.downloadResults(outName, () => {});
            btn.textContent = 'Downloaded';
            await this.loadSimulationList();
            this.updateCompareSelector();
        } catch (e) {
            btn.textContent = 'Download Failed';
        } finally {
            setTimeout(() => { btn.textContent = 'Download'; btn.disabled = false; }, 3000);
        }
    });

    const title = entry.querySelector('.job-entry-title');
    title.style.cursor = 'pointer';
    title.addEventListener('click', () => {
        const outName = this.clusterJobs[jobId]?.out_name || jobInfo.out_name;
        if (outName) this.revealRun(outName);
    });

    entry.querySelector('.btn-toggle-log').addEventListener('click', async () => {
        const log = entry.querySelector('.job-log');
        const opening = log.style.display === 'none';
        log.style.display = opening ? 'block' : 'none';
        // A finished job isn't polled any more: fetch its log once on demand.
        const job = this.clusterJobs[jobId];
        if (opening && job && CLUSTER_TERMINAL_STATES.includes(job.status) && !job._logFetched) {
            if (!log.textContent) log.textContent = 'Fetching log…';
            try {
                const data = await runner.fetchStatus(jobId, job.out_name);
                job._logFetched = true;
                if (data.log) {
                    log.textContent = data.log;
                    job.log = data.log;
                    this.saveClusterJobs();
                } else if (log.textContent === 'Fetching log…') {
                    log.textContent = data.error ? `Log unavailable: ${data.error}` : '(no log found on the cluster)';
                }
                log.scrollTop = log.scrollHeight;
            } catch (e) {
                if (log.textContent === 'Fetching log…') log.textContent = `Log unavailable: ${e.message}`;
            }
        }
    });

    container.prepend(entry);
};

App.prototype.updateJobStatus = function(jobId, data) {
    const entry = document.getElementById(`cluster-job-${jobId}`);
    if (!entry) return;

    const statusEl = entry.querySelector('.job-status');
    const cancelBtn = entry.querySelector('.btn-cancel');
    const downloadBtn = entry.querySelector('.btn-download');
    const logEl = entry.querySelector('.job-log');

    statusEl.textContent = data.status || '-';
    if (data.log) {
        logEl.textContent = data.log;
        logEl.scrollTop = logEl.scrollHeight;
    }

    const s = data.status;
    if (s === 'RUNNING') {
        statusEl.style.color = '#4ade80';
    } else if (s === 'PENDING') {
        statusEl.style.color = '#fbbf24';
    } else if (s === 'COMPLETED') {
        statusEl.style.color = '#4ade80';
        cancelBtn.style.display = 'none';
        downloadBtn.style.display = 'inline-block';
        this.autoDownloadIterations(jobId);
    } else if (['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'].includes(s)) {
        statusEl.style.color = '#e94560';
        cancelBtn.style.display = 'none';
        if (this.clusterJobs[jobId] && this.clusterJobs[jobId].status !== s) this.requestRunRefresh();
    }

    if (this.clusterJobs[jobId] && s) {
        this.clusterJobs[jobId].status = s;
        // Keep the last log with the job: finished jobs aren't polled after a
        // reload, and a cluster may be unreachable then (Vega without OTP).
        if (data.log) this.clusterJobs[jobId].log = data.log;
        // The job's run name is fixed at submission; never take another.
        if (data.out_name && !this.clusterJobs[jobId].out_name) {
            this.clusterJobs[jobId].out_name = data.out_name;
        }
        this.saveClusterJobs();
    }
};

App.prototype.autoDownloadIterations = async function(jobId) {
    const job = this.clusterJobs[jobId];
    if (!job?.out_name || job._iterationsDownloaded) return;
    job._iterationsDownloaded = true;

    try {
        await this.getRunner(job.cluster || 'karolina').downloadIterations(job.out_name);
        this.requestRunRefresh();
    } catch (e) {
        console.warn(`Auto-download iterations for job ${jobId} failed:`, e);
    }
};

App.prototype.downloadClusterSimulation = async function() {
    // The current run, from whichever cluster holds its results (updateCurrentRunUI).
    const simName = this.selectedSimulation;
    const clusterId = this._currentRunCluster;
    if (!simName || !clusterId) {
        alert('Click a run with results on a cluster first');
        return;
    }

    const btn = document.getElementById('download-cluster-results');
    const statusEl = document.getElementById('results-status');
    const progressEl = document.getElementById('results-download-progress');
    const progressBar = document.getElementById('results-download-bar');
    const progressText = document.getElementById('results-download-text');

    btn.disabled = true;
    btn.textContent = 'Downloading...';
    statusEl.style.display = 'none';
    progressEl.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Starting download...';

    try {
        const result = await this.getRunner(clusterId).downloadResults(simName, (data) => {
            const pct = data.bytes_total > 0
                ? Math.round(100 * data.bytes_done / data.bytes_total)
                : 0;
            progressBar.style.width = pct + '%';
            const doneMB = (data.bytes_done / 1048576).toFixed(1);
            const totalMB = (data.bytes_total / 1048576).toFixed(1);
            progressText.textContent = data.file
                ? `${doneMB} / ${totalMB} MB — ${data.file}`
                : `${doneMB} / ${totalMB} MB`;
        });
        progressBar.style.width = '100%';
        progressText.textContent = 'Done';
        statusEl.className = 'mesh-status converted';
        statusEl.textContent = result.message;
        statusEl.style.display = 'block';
        await this.loadSimulationList();
    } catch (e) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = 'Download failed: ' + e.message;
        statusEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        this.updateCurrentRunUI();
        setTimeout(() => { progressEl.style.display = 'none'; }, 2000);
    }
};
