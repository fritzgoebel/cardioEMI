// cluster-runner.js - remote HPC cluster job management via polling.
// One instance per cluster; all endpoints live under /api/cluster/<id>/.

class ClusterRunner {
    constructor(clusterId) {
        this.clusterId = clusterId;
        this.apiBase = `/api/cluster/${clusterId}`;
        this.pollIntervals = {};  // jobId -> intervalId
        this.pollIntervalMs = 5000;
    }

    async checkConnectivity() {
        const response = await fetch(`${this.apiBase}/check`);
        const data = await response.json();
        return data;  // { available, containers: { dolfinx, ginkgo } }
    }

    // One job per (mesh, rank count); see /submit-batch in server.py.
    async submitBatch(options) {
        const response = await fetch(`${this.apiBase}/submit-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Submission failed');
        }
        return data;  // { jobs, failed, message }
    }

    async fetchBatchMeshInfo(meshNames) {
        const response = await fetch(`${this.apiBase}/meshes/batch-info`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ meshes: meshNames })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to fetch mesh info');
        return data.meshes || {};
    }

    async listJobs() {
        const response = await fetch(`${this.apiBase}/jobs`);
        const data = await response.json();
        return data.jobs || [];
    }

    // outName: the job's run folder, so the server finds its log even after a
    // restart (when it no longer knows the job).
    startPolling(jobId, onStatusUpdate, outName) {
        this.stopPolling(jobId);
        const poll = () => this._poll(jobId, onStatusUpdate, outName);
        poll();
        this.pollIntervals[jobId] = setInterval(poll, this.pollIntervalMs);
    }

    async fetchStatus(jobId, outName) {
        const q = outName ? `?out_name=${encodeURIComponent(outName)}` : '';
        const response = await fetch(`${this.apiBase}/status/${jobId}${q}`);
        return await response.json();
    }

    async _poll(jobId, onStatusUpdate, outName) {
        try {
            const data = await this.fetchStatus(jobId, outName);
            onStatusUpdate(data);

            const terminal = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];
            if (data.status && terminal.includes(data.status)) {
                this.stopPolling(jobId);
            }
        } catch (error) {
            console.error(`[${this.clusterId}] status poll failed for job ${jobId}:`, error);
        }
    }

    stopPolling(jobId) {
        if (jobId && this.pollIntervals[jobId]) {
            clearInterval(this.pollIntervals[jobId]);
            delete this.pollIntervals[jobId];
        }
    }

    stopAllPolling() {
        for (const jobId of Object.keys(this.pollIntervals)) {
            clearInterval(this.pollIntervals[jobId]);
        }
        this.pollIntervals = {};
    }

    async cancel(jobId) {
        const response = await fetch(`${this.apiBase}/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Cancel failed');
        }
        return data;
    }

    async downloadResults(remoteDir, onProgress) {
        const response = await fetch(`${this.apiBase}/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ remote_dir: remoteDir })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let result = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            for (const line of text.split('\n')) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.substring(6));
                    if (data.type === 'progress' && onProgress) {
                        onProgress(data);
                    } else if (data.type === 'complete') {
                        result = data;
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (e) {
                    if (e.message && !e.message.includes('Unexpected end of JSON')) throw e;
                }
            }
        }
        return result || { message: 'Download complete' };
    }

    async downloadIterations(remoteDir) {
        const response = await fetch(`${this.apiBase}/download-iterations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ remote_dir: remoteDir })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to download iterations');
        }
        return data;
    }

    async fetchIterations(simName) {
        const response = await fetch(`/api/results/iterations/${simName}`);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch iterations');
        }
        return data;
    }

    async listRemoteMeshes() {
        const response = await fetch(`${this.apiBase}/meshes`);
        const data = await response.json();
        return data.families || [];
    }

    async convertRemoteMesh(family, pts, elem, outputPrefix, color, onOutput) {
        const response = await fetch(`${this.apiBase}/meshes/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ family, pts, elem, output_prefix: outputPrefix, color })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            for (const line of text.split('\n')) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.type === 'output' && onOutput) {
                            onOutput(data.text);
                        } else if (data.type === 'complete') {
                            return { success: data.success };
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes('Unexpected end of JSON')) throw e;
                    }
                }
            }
        }
        return { success: true };
    }

    async generateWeakScaling(params, onOutput) {
        const response = await fetch(`${this.apiBase}/weak-scaling/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let name = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            for (const line of text.split('\n')) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.type === 'output' && onOutput) {
                            onOutput(data.text);
                        } else if (data.type === 'complete') {
                            name = data.name;
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes('Unexpected end of JSON')) throw e;
                    }
                }
            }
        }
        return { success: true, name };
    }

    // Membrane-only (possibly coarsened) preview of a cluster mesh; resolves
    // to {path, metadata, cached} with path loadable by MeshLoader.loadFrom.
    async fetchMeshPreview(meshName, opts = {}) {
        const response = await fetch(`${this.apiBase}/meshes/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mesh: meshName, ...opts })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Preview failed');
        return data;
    }

    async fetchMeshMetadata(meshName) {
        const response = await fetch(`${this.apiBase}/meshes/metadata/${encodeURIComponent(meshName)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to fetch mesh metadata');
        return data;
    }

    async downloadMeshData(meshName) {
        const response = await fetch(`${this.apiBase}/meshes/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mesh_name: meshName })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Download failed');
        }
        return data;
    }

    async generateRemoteVideo(options) {
        const response = await fetch(`${this.apiBase}/video/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Video generation failed');
        return data;
    }

    async checkVideoStatus(jobId) {
        const response = await fetch(`${this.apiBase}/video/status/${jobId}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Status check failed');
        return data;
    }

    async downloadRemoteVideo(jobId) {
        const response = await fetch(`${this.apiBase}/video/download/${jobId}`, {
            method: 'POST'
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Download failed');
        return data;
    }

    startVideoPolling(jobId, onStatusUpdate) {
        const poll = async () => {
            try {
                const status = await this.checkVideoStatus(jobId);
                onStatusUpdate(status);
                const terminal = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];
                if (status.status && terminal.includes(status.status)) {
                    clearInterval(this._videoPollInterval);
                    this._videoPollInterval = null;
                }
            } catch (e) {
                console.error('Video status poll failed:', e);
            }
        };
        poll();
        this._videoPollInterval = setInterval(poll, 5000);
    }

    stopVideoPolling() {
        if (this._videoPollInterval) {
            clearInterval(this._videoPollInterval);
            this._videoPollInterval = null;
        }
    }

    // --- Remote viz data generation ---

    async generateRemoteViz(simName) {
        const response = await fetch(`${this.apiBase}/viz/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sim_name: simName })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Viz generation failed');
        return data;
    }

    async checkVizStatus(jobId) {
        const response = await fetch(`${this.apiBase}/viz/status/${jobId}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Status check failed');
        return data;
    }

    startVizPolling(jobId, onStatusUpdate) {
        const poll = async () => {
            try {
                const status = await this.checkVizStatus(jobId);
                onStatusUpdate(status);
                const terminal = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];
                if (status.status && terminal.includes(status.status)) {
                    clearInterval(this._vizPollInterval);
                    this._vizPollInterval = null;
                }
            } catch (e) {
                console.error('Viz status poll failed:', e);
            }
        };
        poll();
        this._vizPollInterval = setInterval(poll, 5000);
    }

    async downloadVizData(simName, onProgress) {
        const response = await fetch(`${this.apiBase}/viz/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sim_name: simName })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.type === 'progress' && onProgress) {
                        onProgress(data);
                    } else if (data.type === 'complete') {
                        return data;
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (e) {
                    if (e.message && !e.message.includes('Unexpected')) throw e;
                }
            }
        }
        return { message: 'Download complete' };
    }

    // --- Connection (persistent ControlMaster; OTP for 2FA clusters) ---

    async connect() {
        const response = await fetch(`${this.apiBase}/connect`, { method: 'POST' });
        return await response.json();  // {phase, prompt, error}
    }

    async connectState() {
        const response = await fetch(`${this.apiBase}/connect/state`);
        return await response.json();
    }

    async connectInput(text) {
        const response = await fetch(`${this.apiBase}/connect/input`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return await response.json();
    }

    // Abandon a login in progress (OTP window closed), so the next Connect
    // starts a fresh one instead of reusing a prompt the server has dropped.
    async connectCancel() {
        const response = await fetch(`${this.apiBase}/connect/cancel`, { method: 'POST' });
        return await response.json();
    }

    async disconnect() {
        const response = await fetch(`${this.apiBase}/disconnect`, { method: 'POST' });
        return await response.json();
    }

    // --- Install (dirs + code sync + container SIFs), SSE stream ---

    async install(onOutput) {
        const response = await fetch(`${this.apiBase}/install`, { method: 'POST' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.type === 'output' && onOutput) {
                        onOutput(data.text);
                    } else if (data.type === 'complete') {
                        result = data;
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (e) {
                    if (e.message && !e.message.includes('Unexpected')) throw e;
                }
            }
        }
        return result || { success: false };
    }
}

