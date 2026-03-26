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
}
