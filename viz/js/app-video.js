// app-video.js - Video export (local + remote Karolina)

App.prototype.setupVideoExport = function() {
    const exportBtn = document.getElementById('export-video');
    const remoteBtn = document.getElementById('export-video-remote');
    const progressBar = document.getElementById('video-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');
    const statusEl = document.getElementById('video-status');
    const downloadLink = document.getElementById('video-download');

    // Show remote button when in Karolina mode
    if (this.runTarget === 'karolina') {
        remoteBtn.style.display = 'block';
    }

    // Remote video generation
    remoteBtn.addEventListener('click', () => this.exportVideoRemote());

    exportBtn.addEventListener('click', async () => {
        const fps = parseInt(document.getElementById('video-fps').value);

        if (!this.resultsVizDir || !this.resultsTimeSteps || this.resultsTimeSteps.length === 0) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = 'Load results first before exporting video';
            statusEl.style.display = 'block';
            return;
        }

        exportBtn.disabled = true;
        progressBar.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting capture session...';
        statusEl.style.display = 'none';
        downloadLink.style.display = 'none';

        try {
            // 1. Start capture session on server
            const startResp = await fetch('/api/video/start-capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fps })
            });
            const { session_id } = await startResp.json();

            const numFrames = this.resultsTimeSteps.length;

            // 2. Capture each frame from the viewer
            for (let i = 0; i < numFrames; i++) {
                await this.showResultsAtTime(i);
                this.viewer.renderer.render(this.viewer.scene, this.viewer.camera);

                const blob = await this._captureViewerFrame(i);

                await fetch(`/api/video/frame/${session_id}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'image/jpeg' },
                    body: blob
                });

                const pct = Math.round(((i + 1) / numFrames) * 80);
                progressFill.style.width = `${pct}%`;
                progressText.textContent = `Capturing frame ${i + 1}/${numFrames}`;
            }

            // 3. Finalize: tell server to encode video
            progressText.textContent = 'Encoding video...';
            progressFill.style.width = '85%';

            const finishResp = await fetch(`/api/video/finish-capture/${session_id}`, {
                method: 'POST'
            });
            const result = await finishResp.json();

            if (result.error) {
                throw new Error(result.error);
            }

            progressFill.style.width = '100%';
            progressText.textContent = 'Complete!';

            statusEl.className = 'mesh-status converted';
            statusEl.textContent = 'Video exported successfully!';
            statusEl.style.display = 'block';

            downloadLink.href = `/api/video/download/${result.filename}`;
            downloadLink.textContent = `Download ${result.filename}`;
            downloadLink.style.display = 'block';

        } catch (error) {
            statusEl.className = 'mesh-status error';
            statusEl.textContent = `Export failed: ${error.message}`;
            statusEl.style.display = 'block';
        } finally {
            exportBtn.disabled = false;
            setTimeout(() => { progressBar.style.display = 'none'; }, 2000);
        }
    });
};

App.prototype.exportVideoRemote = async function() {
    const remoteBtn = document.getElementById('export-video-remote');
    const exportBtn = document.getElementById('export-video');
    const progressBar = document.getElementById('video-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');
    const statusEl = document.getElementById('video-status');
    const downloadLink = document.getElementById('video-download');

    const simName = this.selectedSimulation || document.getElementById('simulation-selector').value;
    if (!simName) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = 'Select a simulation first';
        statusEl.style.display = 'block';
        return;
    }

    remoteBtn.disabled = true;
    exportBtn.disabled = true;
    progressBar.style.display = 'block';
    progressFill.style.width = '0%';
    statusEl.style.display = 'none';
    downloadLink.style.display = 'none';

    try {
        // Step 1: Check if viz data already exists locally
        const checkResp = await fetch(`/api/results?dir=${encodeURIComponent(simName)}`);
        const vizExists = checkResp.ok;

        if (!vizExists) {
            // Step 1a: Generate viz data on Karolina
            progressText.textContent = 'Generating viz data on Karolina...';
            const vizJob = await this.karolinaRunner.generateRemoteViz(simName);
            const vizJobId = vizJob.job_id;

            statusEl.className = 'mesh-status';
            statusEl.textContent = `Viz generation job ${vizJobId} submitted`;
            statusEl.style.display = 'block';

            // Wait for viz generation to complete
            await new Promise((resolve, reject) => {
                this.karolinaRunner.startVizPolling(vizJobId, (status) => {
                    const pct = Math.round(status.progress * 0.2);
                    progressFill.style.width = `${pct}%`;
                    progressText.textContent = status.message || `Generating viz data (${status.status})...`;

                    if (status.status === 'COMPLETED') {
                        resolve();
                    } else if (['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'].includes(status.status)) {
                        reject(new Error(`Viz generation ${status.status}: ${status.message}`));
                    }
                });
            });

            progressFill.style.width = '20%';

            // Step 1b: Download viz data from Karolina
            progressText.textContent = 'Downloading viz data...';
            await this.karolinaRunner.downloadVizData(simName, (data) => {
                const pct = data.bytes_total > 0
                    ? 20 + Math.round(20 * data.bytes_done / data.bytes_total) : 20;
                progressFill.style.width = `${pct}%`;
                progressText.textContent = data.file || 'Downloading viz data...';
            });
        }

        progressFill.style.width = '40%';

        // Step 2: Load results into viewer
        progressText.textContent = 'Loading results into viewer...';
        this.selectedSimulation = simName;
        document.getElementById('simulation-selector').value = simName;
        await this.loadResults();
        progressFill.style.width = '50%';

        if (!this.resultsVizDir || !this.resultsTimeSteps || this.resultsTimeSteps.length === 0) {
            throw new Error('Failed to load results into viewer');
        }

        // Step 3: Capture frames using the Three.js renderer (same as local export)
        const fps = parseInt(document.getElementById('video-fps').value);
        progressText.textContent = 'Starting capture session...';

        const startResp = await fetch('/api/video/start-capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fps })
        });
        const { session_id } = await startResp.json();

        const numFrames = this.resultsTimeSteps.length;
        for (let i = 0; i < numFrames; i++) {
            await this.showResultsAtTime(i);
            this.viewer.renderer.render(this.viewer.scene, this.viewer.camera);

            const blob = await this._captureViewerFrame(i);
            await fetch(`/api/video/frame/${session_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'image/jpeg' },
                body: blob
            });

            const pct = 50 + Math.round(((i + 1) / numFrames) * 42);
            progressFill.style.width = `${pct}%`;
            progressText.textContent = `Capturing frame ${i + 1}/${numFrames}`;
        }

        // Step 4: Encode video
        progressText.textContent = 'Encoding video...';
        progressFill.style.width = '94%';

        const finishResp = await fetch(`/api/video/finish-capture/${session_id}`, {
            method: 'POST'
        });
        const result = await finishResp.json();

        if (result.error) throw new Error(result.error);

        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';

        statusEl.className = 'mesh-status converted';
        statusEl.textContent = 'Video exported successfully!';
        statusEl.style.display = 'block';

        downloadLink.href = `/api/video/download/${result.filename}`;
        downloadLink.textContent = `Download ${result.filename}`;
        downloadLink.style.display = 'block';

    } catch (error) {
        statusEl.className = 'mesh-status error';
        statusEl.textContent = `Failed: ${error.message}`;
        statusEl.style.display = 'block';
    } finally {
        remoteBtn.disabled = false;
        exportBtn.disabled = false;
        setTimeout(() => { progressBar.style.display = 'none'; }, 2000);
    }
};

App.prototype._captureViewerFrame = function(timeIndex) {
    return new Promise((resolve) => {
        const glCanvas = this.viewer.renderer.domElement;
        const w = glCanvas.width;
        const h = glCanvas.height;

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');

        // Draw 3D scene
        ctx.drawImage(glCanvas, 0, 0, w, h);

        // Draw time label (top-left)
        const time = this.resultsTimeSteps[timeIndex];
        ctx.font = 'bold 18px -apple-system, sans-serif';
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(10, 10, ctx.measureText(`t = ${time.toFixed(3)} ms`).width + 16, 30);
        ctx.fillStyle = '#fff';
        ctx.fillText(`t = ${time.toFixed(3)} ms`, 18, 32);

        // Draw colorbar (top-right)
        const cbW = 30, cbH = 150, cbPad = 20;
        const cbX = w - cbW - cbPad - 70, cbY = cbPad;

        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(cbX - 8, cbY - 8, cbW + 80, cbH + 16);

        const cm = this.viewer.colormaps[this.viewer.colormap];
        const grad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
        for (let i = 0; i < cm.colors.length; i++) {
            const [r, g, b] = cm.colors[i];
            const pos = 1 - cm.positions[i];
            grad.addColorStop(pos, `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`);
        }
        ctx.fillStyle = grad;
        ctx.fillRect(cbX, cbY, cbW, cbH);

        ctx.font = '12px -apple-system, sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'left';
        ctx.fillText(`${Math.round(this.viewer.vMax)} mV`, cbX + cbW + 6, cbY + 10);
        ctx.fillText(`${Math.round((this.viewer.vMax + this.viewer.vMin) / 2)} mV`, cbX + cbW + 6, cbY + cbH / 2 + 4);
        ctx.fillText(`${Math.round(this.viewer.vMin)} mV`, cbX + cbW + 6, cbY + cbH);

        // Draw voltage plot if a vertex is picked
        const plotPanel = document.getElementById('voltage-plot-panel');
        if (plotPanel && plotPanel.style.display !== 'none') {
            const plotCanvas = document.getElementById('voltage-plot-canvas');
            if (plotCanvas) {
                const plotW = plotCanvas.width;
                const plotH = plotCanvas.height;
                ctx.fillStyle = 'rgba(10, 10, 26, 0.92)';
                ctx.fillRect(15, h - plotH - 50, plotW + 20, plotH + 40);
                const coordsText = document.getElementById('voltage-plot-coords').textContent;
                ctx.font = '10px monospace';
                ctx.fillStyle = '#aaa';
                ctx.fillText(coordsText, 25, h - plotH - 18);
                ctx.drawImage(plotCanvas, 25, h - plotH - 10, plotW, plotH);
            }
        }

        canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92);
    });
};
