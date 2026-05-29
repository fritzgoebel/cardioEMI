// app-karolina.js - Karolina job management, run target, and downloads

App.prototype.setupRunTarget = function() {
    const selector = document.getElementById('run-target');
    const karolinaOptions = document.getElementById('karolina-options');
    const containerStatusRow = document.getElementById('karolina-container-status-row');
    const statusDot = document.getElementById('karolina-status-dot');
    const refreshBtn = document.getElementById('refresh-mesh-list');
    const mpiRanksRow = document.getElementById('mpi-ranks').closest('.param-row');

    const applyTarget = async (target) => {
        const jobSection = document.getElementById('karolina-job-section');
        if (target === 'karolina') {
            karolinaOptions.style.display = 'block';
            containerStatusRow.style.display = 'flex';
            statusDot.style.display = 'inline';
            refreshBtn.style.display = 'inline-block';
            mpiRanksRow.style.display = 'none';
            if (this.karolinaJobs && Object.keys(this.karolinaJobs).length > 0) {
                jobSection.style.display = 'block';
            }
            statusDot.textContent = '...';
            statusDot.title = 'Checking SSH...';
            try {
                const result = await this.karolinaRunner.checkConnectivity();
                const ok = result.available;
                statusDot.textContent = '\u25CF';
                statusDot.style.color = ok ? '#4ade80' : '#e94560';
                statusDot.title = ok ? 'SSH connected' : 'SSH unreachable';
                if (ok && result.containers) {
                    const c = result.containers;
                    const parts = [];
                    parts.push(`DOLFINx: ${c.dolfinx ? 'ready' : 'missing'}`);
                    parts.push(`Ginkgo: ${c.ginkgo ? 'ready' : 'missing'}`);
                    const containerEl = document.getElementById('karolina-container-status');
                    containerEl.textContent = parts.join(' | ');
                    containerEl.style.color = c.dolfinx ? '#4ade80' : '#e94560';
                }
                if (ok) this.refreshMeshList();
            } catch (e) {
                statusDot.textContent = '\u25CF';
                statusDot.style.color = '#e94560';
                statusDot.title = 'SSH check failed';
            }
            document.getElementById('download-karolina-results').style.display = 'block';
            document.getElementById('export-video-remote').style.display = 'block';
            this.loadSimulationList();
        } else {
            karolinaOptions.style.display = 'none';
            containerStatusRow.style.display = 'none';
            statusDot.style.display = 'none';
            refreshBtn.style.display = 'none';
            mpiRanksRow.style.display = 'flex';
            jobSection.style.display = 'none';
            document.getElementById('download-karolina-results').style.display = 'none';
            document.getElementById('export-video-remote').style.display = 'none';
            this.refreshMeshList();
        }
    };

    selector.addEventListener('change', async () => {
        this.runTarget = selector.value;
        sessionStorage.setItem('runTarget', this.runTarget);
        await applyTarget(this.runTarget);
    });

    if (this.runTarget !== 'local') {
        selector.value = this.runTarget;
        applyTarget(this.runTarget);
    }
};

App.prototype.setupKarolinaOptions = function() {
    const nodesInput = document.getElementById('karolina-nodes');
    const ntasksInput = document.getElementById('karolina-ntasks');
    const totalRanksSpan = document.getElementById('karolina-total-ranks');

    const updateTotal = () => {
        const nodes = parseInt(nodesInput.value) || 1;
        const ntasks = parseInt(ntasksInput.value) || 128;
        totalRanksSpan.textContent = nodes * ntasks;
    };

    nodesInput.addEventListener('input', updateTotal);
    ntasksInput.addEventListener('input', updateTotal);
    updateTotal();

    this.karolinaJobs = {};
    this.compareDatasets = [];

    // Persistence + mesh filter + remove flow live in app-karolina-jobs.js
    this.setupKarolinaJobsPersistence();
};

App.prototype.renderJobEntry = function(jobInfo) {
    const container = document.getElementById('karolina-jobs-list');
    const jobId = jobInfo.job_id;
    const entryId = `karolina-job-${jobId}`;

    if (document.getElementById(entryId)) return;

    const entry = document.createElement('div');
    entry.id = entryId;
    entry.style.cssText = 'border:1px solid #444; border-radius:6px; padding:8px; margin-bottom:8px; background:#1a1a1a;';
    const lbl = this.formatSimulationLabel(jobInfo);
    entry.innerHTML = `
        <div class="job-entry-header" style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
            <div class="job-entry-title" style="min-width:0; flex:1;" title="${jobInfo.out_name || jobId}">
                <div style="font-weight:bold; color:#ccc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${lbl.line1}</div>
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

    entry.querySelector('.btn-cancel').addEventListener('click', async () => {
        try {
            await this.karolinaRunner.cancel(jobId);
            this.karolinaRunner.stopPolling(jobId);
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
            const outName = this.karolinaJobs[jobId]?.out_name;
            await this.karolinaRunner.downloadResults(outName, () => {});
            btn.textContent = 'Downloaded';
            await this.loadSimulationList();
            this.updateCompareSelector();
        } catch (e) {
            btn.textContent = 'Download Failed';
        } finally {
            setTimeout(() => { btn.textContent = 'Download'; btn.disabled = false; }, 3000);
        }
    });

    entry.querySelector('.btn-toggle-log').addEventListener('click', () => {
        const log = entry.querySelector('.job-log');
        log.style.display = log.style.display === 'none' ? 'block' : 'none';
    });

    container.prepend(entry);
};

App.prototype.updateJobStatus = function(jobId, data) {
    const entry = document.getElementById(`karolina-job-${jobId}`);
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
    }

    if (this.karolinaJobs[jobId] && s) {
        this.karolinaJobs[jobId].status = s;
        this.saveKarolinaJobs();
    }
};

App.prototype.autoDownloadIterations = async function(jobId) {
    const job = this.karolinaJobs[jobId];
    if (!job?.out_name || job._iterationsDownloaded) return;
    job._iterationsDownloaded = true;

    try {
        await this.karolinaRunner.downloadIterations(job.out_name);
        await this.loadSimulationList();
        this.updateCompareSelector();
    } catch (e) {
        console.warn(`Auto-download iterations for job ${jobId} failed:`, e);
    }
};

App.prototype.downloadKarolinaSimulation = async function() {
    const simName = document.getElementById('simulation-selector').value;
    if (!simName) {
        alert('Please select a simulation first');
        return;
    }

    const btn = document.getElementById('download-karolina-results');
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
        const result = await this.karolinaRunner.downloadResults(simName, (data) => {
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
        btn.textContent = 'Download from Karolina';
        setTimeout(() => { progressEl.style.display = 'none'; }, 2000);
    }
};
