// karolina-runner.js - Karolina supercomputer job management via polling

class KarolinaRunner {
    constructor(apiBase = '/api/karolina') {
        this.apiBase = apiBase;
        this.pollIntervals = {};  // jobId -> intervalId
        this.pollIntervalMs = 5000;
    }

    async checkConnectivity() {
        const response = await fetch(`${this.apiBase}/check`);
        const data = await response.json();
        return data;  // { available, containers: { dolfinx, ginkgo } }
    }

    async submit(options) {
        const response = await fetch(`${this.apiBase}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Submission failed');
        }
        return data;
    }

    async listJobs() {
        const response = await fetch(`${this.apiBase}/jobs`);
        const data = await response.json();
        return data.jobs || [];
    }

    startPolling(jobId, onStatusUpdate) {
        this.stopPolling(jobId);
        const poll = () => this._poll(jobId, onStatusUpdate);
        poll();
        this.pollIntervals[jobId] = setInterval(poll, this.pollIntervalMs);
    }

    async _poll(jobId, onStatusUpdate) {
        try {
            const response = await fetch(`${this.apiBase}/status/${jobId}`);
            const data = await response.json();
            onStatusUpdate(data);

            const terminal = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'];
            if (data.status && terminal.includes(data.status)) {
                this.stopPolling(jobId);
            }
        } catch (error) {
            console.error(`Karolina status poll failed for job ${jobId}:`, error);
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

    async listRemoteSimulations() {
        const response = await fetch(`${this.apiBase}/remote-simulations`);
        const data = await response.json();
        return data.simulations || [];
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
}
