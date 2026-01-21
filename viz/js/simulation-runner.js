// simulation-runner.js - Execute Docker simulation command

class SimulationRunner {
    constructor(apiBase = '/api') {
        this.apiBase = apiBase;
        this.eventSource = null;
        this.mpiRanks = 8;
        this.configFile = 'input_pepe36_colored.yml';
    }

    setMpiRanks(ranks) {
        this.mpiRanks = Math.max(1, Math.min(32, ranks));
    }

    setConfigFile(configFile) {
        this.configFile = configFile;
    }

    async run(onOutput, onProgress, onIterations, onResidual) {
        // Close any existing connection
        if (this.eventSource) {
            this.eventSource.close();
        }

        return new Promise((resolve, reject) => {
            const url = new URL(`${this.apiBase}/simulation/run`, window.location.origin);
            url.searchParams.set('ranks', this.mpiRanks);
            url.searchParams.set('config', this.configFile);

            this.eventSource = new EventSource(url.toString());

            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'output') {
                        onOutput(data.text);
                    } else if (data.type === 'progress') {
                        if (onProgress) {
                            onProgress(data.percent, data.message);
                        }
                    } else if (data.type === 'iterations') {
                        if (onIterations) {
                            onIterations(data.step, data.count);
                        }
                    } else if (data.type === 'residual') {
                        if (onResidual) {
                            onResidual(data.step, data.abs, data.rel);
                        }
                    } else if (data.type === 'complete') {
                        this.eventSource.close();
                        this.eventSource = null;
                        if (data.success) {
                            resolve(data);
                        } else {
                            reject(new Error(`Simulation failed with exit code ${data.returncode}`));
                        }
                    } else if (data.type === 'error') {
                        this.eventSource.close();
                        this.eventSource = null;
                        reject(new Error(data.message));
                    }
                } catch (e) {
                    console.error('Failed to parse SSE message:', e);
                }
            };

            this.eventSource.onerror = (error) => {
                this.eventSource.close();
                this.eventSource = null;
                reject(new Error('Connection to server lost'));
            };
        });
    }

    async stop() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        const response = await fetch(`${this.apiBase}/simulation/stop`, {
            method: 'POST'
        });
        return await response.json();
    }

    async getStatus() {
        const response = await fetch(`${this.apiBase}/simulation/status`);
        return await response.json();
    }
}
