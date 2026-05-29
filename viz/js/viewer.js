// viewer.js - Three.js 3D visualization

class Viewer {
    constructor(containerId) {
        this.containerId = containerId;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.meshObject = null;
        this.ecsMeshObject = null;  // ECS (exterior) mesh
        this.cutMeshObject = null;  // Partition cut mesh (internal facets at partition boundaries)
        this.boundingBoxHelper = null;
        this.scarBoxHelper = null;
        this.scarBorderBoxHelper = null;
        this.meshData = null;
        this.showExcitedHighlight = true;
        this.scarZoneMask = null;  // Uint8Array: 0=healthy, 1=border, 2=dense

        // Voltage range for colormap
        this.vMin = -80;
        this.vMax = 0;

        // Rank coloring
        this.numRanks = 1;
        this.colorMode = 'voltage'; // 'voltage' or 'rank'

        // Cross-section mode: cells (closed surfaces, colored by φ_i) + ECS cap
        // (slice through ECS volume, colored by φ_e), each with its own colormap
        // and range. The ECS shell mesh is also clipped at the plane in this
        // mode so the volume below the plane stays visible.
        this.crossSectionMode = false;
        this.cellMeshes = new Map();          // tag -> THREE.Mesh
        this.capMeshObject = null;
        this._clippingPlane = null;           // THREE.Plane applied to ECS shell
        this._planeHelper = null;             // THREE.PlaneHelper preview wireframe
        this.vMinIntra = -80;
        this.vMaxIntra = 40;
        this.colormapIntra = 'coolwarm';
        this._intraLUT = null;
        this.vMinExtra = -5;
        this.vMaxExtra = 5;
        this.colormapExtra = 'viridis';
        this._extraLUT = null;

        // Explosion effect
        this.explosionFactor = 0;
        this.originalVertices = null;      // Original membrane vertex positions
        this.originalEcsVertices = null;   // Original ECS vertex positions
        this.originalCutVertices = null;   // Original cut vertex positions
        this.rankCentroids = null;
        this.globalCentroid = null;
        this.ranksData = null;
        this.ecsRanksData = null;
        this.cutRanksData = null;

        // Vertex picking
        this.onVertexPicked = null;  // Callback: (vertexIndex, worldPos) => void
        this.pickMarker = null;
        this._pickPointerMoved = false;

        // Current colormap
        this.colormap = 'coolwarm';

        // Available colormaps with their definitions
        this.colormaps = {
            coolwarm: {
                name: 'Cool to Warm',
                colors: [[0, 0, 1], [1, 1, 1], [1, 0, 0]],  // blue -> white -> red
                positions: [0, 0.5, 1]
            },
            viridis: {
                name: 'Viridis',
                colors: [
                    [0.267, 0.004, 0.329],
                    [0.282, 0.140, 0.458],
                    [0.254, 0.265, 0.530],
                    [0.207, 0.372, 0.553],
                    [0.164, 0.471, 0.558],
                    [0.128, 0.567, 0.551],
                    [0.135, 0.659, 0.518],
                    [0.267, 0.749, 0.441],
                    [0.478, 0.821, 0.318],
                    [0.741, 0.873, 0.150],
                    [0.993, 0.906, 0.144]
                ],
                positions: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            },
            plasma: {
                name: 'Plasma',
                colors: [
                    [0.050, 0.030, 0.528],
                    [0.294, 0.012, 0.615],
                    [0.492, 0.012, 0.658],
                    [0.658, 0.134, 0.588],
                    [0.798, 0.280, 0.470],
                    [0.899, 0.434, 0.358],
                    [0.963, 0.600, 0.246],
                    [0.984, 0.775, 0.154],
                    [0.940, 0.975, 0.131]
                ],
                positions: [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
            },
            inferno: {
                name: 'Inferno',
                colors: [
                    [0.001, 0.000, 0.014],
                    [0.133, 0.047, 0.298],
                    [0.341, 0.062, 0.429],
                    [0.550, 0.126, 0.405],
                    [0.735, 0.216, 0.330],
                    [0.878, 0.352, 0.218],
                    [0.963, 0.537, 0.114],
                    [0.988, 0.751, 0.145],
                    [0.988, 0.998, 0.645]
                ],
                positions: [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
            },
            jet: {
                name: 'Jet (Rainbow)',
                colors: [
                    [0, 0, 0.5],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                    [1, 0, 0],
                    [0.5, 0, 0]
                ],
                positions: [0, 0.125, 0.375, 0.625, 0.875, 1]
            },
            grayscale: {
                name: 'Grayscale',
                colors: [[0, 0, 0], [1, 1, 1]],
                positions: [0, 1]
            },
            turbo: {
                name: 'Turbo',
                colors: [
                    [0.190, 0.072, 0.232],
                    [0.254, 0.265, 0.600],
                    [0.137, 0.514, 0.855],
                    [0.059, 0.718, 0.675],
                    [0.318, 0.855, 0.400],
                    [0.651, 0.929, 0.255],
                    [0.929, 0.855, 0.200],
                    [0.996, 0.620, 0.161],
                    [0.957, 0.353, 0.161],
                    [0.796, 0.118, 0.173]
                ],
                positions: [0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1]
            }
        };

        // Interface color palette - distinct from rank colors
        this.interfaceColors = [
            { r: 1.00, g: 0.84, b: 0.00 },  // Gold
            { r: 0.00, g: 1.00, b: 0.50 },  // Spring Green
            { r: 1.00, g: 0.41, b: 0.71 },  // Hot Pink
            { r: 0.00, g: 0.80, b: 1.00 },  // Deep Sky Blue
            { r: 1.00, g: 0.55, b: 0.00 },  // Dark Orange
            { r: 0.58, g: 0.00, b: 0.83 },  // Dark Violet
            { r: 0.00, g: 1.00, b: 1.00 },  // Cyan
            { r: 1.00, g: 0.27, b: 0.00 },  // Orange Red
            { r: 0.50, g: 1.00, b: 0.00 },  // Chartreuse
            { r: 1.00, g: 0.08, b: 0.58 },  // Deep Pink
            { r: 0.25, g: 0.88, b: 0.82 },  // Turquoise
            { r: 1.00, g: 0.65, b: 0.00 },  // Orange
            { r: 0.80, g: 0.52, b: 0.25 },  // Peru
            { r: 0.60, g: 0.80, b: 0.20 },  // Yellow Green
            { r: 0.94, g: 0.50, b: 0.50 },  // Light Coral
            { r: 0.49, g: 0.99, b: 0.00 },  // Lawn Green
        ];

        // Categorical colormap for ranks (Tableau 20-like)
        this.rankColors = [
            { r: 0.12, g: 0.47, b: 0.71 },  // Blue
            { r: 1.00, g: 0.50, b: 0.05 },  // Orange
            { r: 0.17, g: 0.63, b: 0.17 },  // Green
            { r: 0.84, g: 0.15, b: 0.16 },  // Red
            { r: 0.58, g: 0.40, b: 0.74 },  // Purple
            { r: 0.55, g: 0.34, b: 0.29 },  // Brown
            { r: 0.89, g: 0.47, b: 0.76 },  // Pink
            { r: 0.50, g: 0.50, b: 0.50 },  // Gray
            { r: 0.74, g: 0.74, b: 0.13 },  // Olive
            { r: 0.09, g: 0.75, b: 0.81 },  // Cyan
            { r: 0.68, g: 0.78, b: 0.91 },  // Light Blue
            { r: 1.00, g: 0.73, b: 0.47 },  // Light Orange
            { r: 0.60, g: 0.87, b: 0.54 },  // Light Green
            { r: 1.00, g: 0.60, b: 0.59 },  // Light Red
            { r: 0.77, g: 0.69, b: 0.84 },  // Light Purple
            { r: 0.77, g: 0.61, b: 0.58 },  // Light Brown
            { r: 0.97, g: 0.71, b: 0.82 },  // Light Pink
            { r: 0.78, g: 0.78, b: 0.78 },  // Light Gray
            { r: 0.86, g: 0.86, b: 0.55 },  // Light Olive
            { r: 0.62, g: 0.85, b: 0.90 },  // Light Cyan
        ];
    }

    // Get color for a rank (categorical)
    rankToColor(rank) {
        const colorIndex = rank % this.rankColors.length;
        return this.rankColors[colorIndex];
    }

    // Set the active colormap
    setColormap(colormapName) {
        if (this.colormaps[colormapName]) {
            this.colormap = colormapName;
            this._colorLUT = null; // Invalidate cached LUT
        }
    }

    // Get the current colormap name
    getColormap() {
        return this.colormap;
    }

    // Get list of available colormaps
    getAvailableColormaps() {
        return Object.entries(this.colormaps).map(([key, val]) => ({
            id: key,
            name: val.name
        }));
    }

    // Get CSS gradient for the current colormap (for colorbar).
    // "to top" so the high-value end of the colormap is at the top while
    // stops stay in ascending order (out-of-order stops get snapped by the
    // browser, which produced solid-colour bars before this fix).
    getColormapGradient() {
        const cm = this.colormaps[this.colormap];
        const stops = cm.colors.map((color, i) => {
            const r = Math.round(color[0] * 255);
            const g = Math.round(color[1] * 255);
            const b = Math.round(color[2] * 255);
            const pos = cm.positions[i] * 100;
            return `rgb(${r}, ${g}, ${b}) ${pos}%`;
        });
        return `linear-gradient(to top, ${stops.join(', ')})`;
    }

    // Sample a named colormap at normalised t in [0, 1]
    sampleColormap(colormapName, t) {
        const cm = this.colormaps[colormapName] || this.colormaps[this.colormap];
        const colors = cm.colors;
        const positions = cm.positions;
        const clampedT = t <= 0 ? 0 : (t >= 1 ? 1 : t);
        let i = 0;
        while (i < positions.length - 1 && positions[i + 1] < clampedT) i++;
        if (i >= positions.length - 1) {
            const c = colors[colors.length - 1];
            return { r: c[0], g: c[1], b: c[2] };
        }
        const t0 = positions[i];
        const t1 = positions[i + 1];
        const localT = (clampedT - t0) / (t1 - t0);
        const c0 = colors[i];
        const c1 = colors[i + 1];
        return {
            r: c0[0] + (c1[0] - c0[0]) * localT,
            g: c0[1] + (c1[1] - c0[1]) * localT,
            b: c0[2] + (c1[2] - c0[2]) * localT,
        };
    }

    // Map value to color using the current colormap
    voltageToColor(v) {
        // Normalize voltage to 0-1 range
        const t = Math.max(0, Math.min(1, (v - this.vMin) / (this.vMax - this.vMin)));

        const cm = this.colormaps[this.colormap];
        const colors = cm.colors;
        const positions = cm.positions;

        // Find the two colors to interpolate between
        let i = 0;
        while (i < positions.length - 1 && positions[i + 1] < t) {
            i++;
        }

        // Handle edge cases
        if (i >= positions.length - 1) {
            const c = colors[colors.length - 1];
            return { r: c[0], g: c[1], b: c[2] };
        }

        // Interpolate between colors[i] and colors[i+1]
        const t0 = positions[i];
        const t1 = positions[i + 1];
        const localT = (t - t0) / (t1 - t0);

        const c0 = colors[i];
        const c1 = colors[i + 1];

        return {
            r: c0[0] + (c1[0] - c0[0]) * localT,
            g: c0[1] + (c1[1] - c0[1]) * localT,
            b: c0[2] + (c1[2] - c0[2]) * localT
        };
    }

    async init(meshData) {
        this.meshData = meshData;

        const container = document.getElementById(this.containerId);
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a1a);

        // Camera
        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        // Enable per-material clipping (used for cross-section ECS clipping)
        this.renderer.localClippingEnabled = true;
        container.appendChild(this.renderer.domElement);

        // Detect WebGL context loss (e.g. GPU memory exhaustion)
        this.renderer.domElement.addEventListener('webglcontextlost', (event) => {
            console.error('WebGL context lost!', event);
            const statusEl = document.getElementById('mesh-status');
            if (statusEl) {
                statusEl.className = 'mesh-status error';
                statusEl.textContent = 'WebGL context lost — mesh too large for GPU. Try closing other tabs.';
                statusEl.style.display = 'block';
            }
        });

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = false;  // Disable built-in zoom, we'll handle it

        // Raycaster for zoom-to-cursor and dynamic rotation target
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Custom zoom towards center of visible subdomains
        this.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();

            const zoomSpeed = 0.1;
            const targetPoint = this.getVisibleCentroid();

            // Direction from camera to target
            const direction = new THREE.Vector3().subVectors(targetPoint, this.camera.position);

            if (event.deltaY < 0) {
                // Zoom in - move camera towards target
                this.camera.position.addScaledVector(direction, zoomSpeed);
            } else {
                // Zoom out - move camera away from target
                this.camera.position.addScaledVector(direction, -zoomSpeed);
            }

            this.controls.update();
        }, { passive: false });

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(200, 200, 200);
        this.scene.add(directionalLight);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-100, -100, -100);
        this.scene.add(directionalLight2);

        // Create mesh geometry
        this.createMembraneMesh(meshData);

        // Create ECS mesh if available
        if (meshData.ecsVertices && meshData.ecsFacets) {
            this.createEcsMesh(meshData);
        }

        // Create partition cut mesh if available
        if (meshData.cutVertices && meshData.cutFacets) {
            this.createCutMesh(meshData);
        }

        // Cross-section per-cell surfaces (hidden until cross-section mode is on)
        if (meshData.cells) {
            this.createCellMeshes(meshData.cells);
        }

        // Position camera
        this.resetCamera();

        // Start render loop
        this.animate();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    /**
     * Split geometry into draw groups if index count exceeds Firefox's
     * webgl.max-vert-ids-per-draw limit (30M).
     */
    _applyDrawGroups(geometry) {
        const MAX_INDICES = 30_000_000;
        const indexCount = geometry.index.count;
        geometry.clearGroups();
        if (indexCount > MAX_INDICES) {
            const chunkSize = Math.floor(MAX_INDICES / 3) * 3;
            for (let start = 0; start < indexCount; start += chunkSize) {
                geometry.addGroup(start, Math.min(chunkSize, indexCount - start), 0);
            }
        }
    }

    createMembraneMesh(meshData) {
        const { vertices, facets, metadata } = meshData;
        const numVertices = vertices.length / 3;
        console.log(`createMembraneMesh: ${numVertices} vertices, ${facets.length / 3} facets`);

        // Defer originalVertices copy until explosion data is actually needed
        this.originalVertices = null;
        this._pendingVerticesForExplosion = vertices;

        // Create BufferGeometry
        const geometry = new THREE.BufferGeometry();

        // Set position attribute (vertices is flat array [x0,y0,z0, x1,y1,z1, ...])
        console.time('  setAttribute position');
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        console.timeEnd('  setAttribute position');

        // Set index attribute (facets is [v0,v1,v2, v0,v1,v2, ...])
        console.time('  setIndex');
        geometry.setIndex(new THREE.BufferAttribute(facets, 1));
        console.timeEnd('  setIndex');

        // Firefox limits drawElements to 30M indices per call
        // (webgl.max-vert-ids-per-draw). Split into draw groups to stay under the limit.
        this._applyDrawGroups(geometry);
        if (geometry.groups.length > 0) {
            console.log(`  Split into ${geometry.groups.length} draw groups`);
        }

        // Use flat shading — the GPU computes face normals automatically,
        // avoiding an expensive JS-side computeVertexNormals() pass.
        // For large meshes (>10M vertices) the JS normal computation can
        // freeze/crash the browser tab.
        const useFlatShading = numVertices > 2_000_000;

        if (!useFlatShading) {
            console.time('  computeVertexNormals');
            geometry.computeVertexNormals();
            console.timeEnd('  computeVertexNormals');
        } else {
            console.log('  Skipping computeVertexNormals (flat shading for large mesh)');
        }

        // Create material with vertex colors
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            flatShading: useFlatShading,
            transparent: false,
            shininess: 30
        });

        // Initialize vertex colors (default resting state - blue)
        console.time('  init colors');
        const colors = new Float32Array(vertices.length);
        const restingColor = this.voltageToColor(this.vMin);
        const rr = restingColor.r, rg = restingColor.g, rb = restingColor.b;
        for (let i = 0; i < vertices.length; i += 3) {
            colors[i] = rr;
            colors[i + 1] = rg;
            colors[i + 2] = rb;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        console.timeEnd('  init colors');

        // Create mesh and add to scene.
        // When draw groups are active, Three.js only uses them if the material
        // is an array (it indexes by group.materialIndex).
        console.time('  scene.add');
        const meshMaterial = geometry.groups.length > 0 ? [material] : material;
        this.meshObject = new THREE.Mesh(geometry, meshMaterial);
        this.scene.add(this.meshObject);
        console.timeEnd('  scene.add');

        // Check for WebGL errors after first render
        if (this.renderer) {
            const gl = this.renderer.getContext();
            const err = gl.getError();
            if (err !== gl.NO_ERROR) {
                console.error(`WebGL error after mesh creation: 0x${err.toString(16)}`);
            }
            console.log(`  WebGL MAX_ELEMENT_INDEX: ${gl.getParameter(gl.MAX_ELEMENT_INDEX)}`);
        }

        console.log('  Mesh added to scene');
    }

    createEcsMesh(meshData) {
        const { ecsVertices, ecsFacets } = meshData;

        // Store original ECS vertices for explosion effect
        this.originalEcsVertices = new Float32Array(ecsVertices);

        // Create BufferGeometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ecsVertices), 3));
        geometry.setIndex(new THREE.BufferAttribute(ecsFacets, 1));
        geometry.computeVertexNormals();

        // Create translucent material with vertex colors
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            flatShading: false,
            transparent: true,
            opacity: 0.15,
            shininess: 10,
            depthWrite: false  // Prevent z-fighting with membrane
        });

        // Initialize vertex colors (light gray for ECS)
        const colors = new Float32Array(ecsVertices.length);
        for (let i = 0; i < ecsVertices.length; i += 3) {
            colors[i] = 0.7;      // R
            colors[i + 1] = 0.7;  // G
            colors[i + 2] = 0.8;  // B - slightly blue tint
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create mesh (hidden by default)
        this.ecsMeshObject = new THREE.Mesh(geometry, material);
        this.ecsMeshObject.visible = false;
        this.ecsMeshObject.userData.userVisible = false;
        this.scene.add(this.ecsMeshObject);

        console.log(`ECS mesh created: ${ecsVertices.length / 3} vertices`);
    }

    setEcsVisible(visible) {
        if (this.ecsMeshObject) {
            this.ecsMeshObject.userData.userVisible = !!visible;
            // In cross-section mode the cap replaces the translucent ECS shell.
            this.ecsMeshObject.visible = !!visible && !this.crossSectionMode;
        }
        // Rebuild interface points when ECS visibility changes
        // (to add/remove ECS interface points while keeping membrane ones)
        if (this.highlightedInterfaceMap) {
            this.updateInterfacePoints();
        }
    }

    setEcsOpacity(opacity) {
        if (this.ecsMeshObject) {
            this.ecsMeshObject.material.opacity = opacity;
        }
    }

    createCutMesh(meshData) {
        const { cutVertices, cutFacets } = meshData;

        // Store original cut vertices for explosion effect
        this.originalCutVertices = new Float32Array(cutVertices);

        // Create BufferGeometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cutVertices), 3));
        geometry.setIndex(new THREE.BufferAttribute(cutFacets, 1));
        geometry.computeVertexNormals();

        // Create material with vertex colors (opaque, shows internal cuts)
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            flatShading: false,
            transparent: false,
            shininess: 30
        });

        // Initialize vertex colors (will be set by rank)
        const colors = new Float32Array(cutVertices.length);
        for (let i = 0; i < cutVertices.length; i += 3) {
            colors[i] = 0.5;
            colors[i + 1] = 0.5;
            colors[i + 2] = 0.5;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create mesh (hidden by default, only shown in partition mode)
        this.cutMeshObject = new THREE.Mesh(geometry, material);
        this.cutMeshObject.visible = false;
        this.scene.add(this.cutMeshObject);

        console.log(`Cut mesh created: ${cutVertices.length / 3} vertices`);
    }

    setCutVisible(visible) {
        if (this.cutMeshObject) {
            this.cutMeshObject.visible = visible;
        }
    }

    updateCutRankColors(cutRanks) {
        if (!this.cutMeshObject) return;

        const geometry = this.cutMeshObject.geometry;
        const colors = geometry.attributes.color.array;

        // Note: boundary facets are already duplicated in the mesh data,
        // with each copy having uniform rank assignment for all 3 vertices
        for (let i = 0; i < cutRanks.length; i++) {
            const color = this.rankToColor(cutRanks[i]);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    async reloadMesh(meshData) {
        // Remove old membrane mesh
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            const mat = this.meshObject.material;
            if (Array.isArray(mat)) { mat.forEach(m => m.dispose()); } else { mat.dispose(); }
        }

        // Remove old ECS mesh
        if (this.ecsMeshObject) {
            this.scene.remove(this.ecsMeshObject);
            this.ecsMeshObject.geometry.dispose();
            this.ecsMeshObject.material.dispose();
            this.ecsMeshObject = null;
        }

        // Remove old cut mesh
        if (this.cutMeshObject) {
            this.scene.remove(this.cutMeshObject);
            this.cutMeshObject.geometry.dispose();
            this.cutMeshObject.material.dispose();
            this.cutMeshObject = null;
        }

        // Update mesh data reference
        this.meshData = meshData;

        // Dispose old cell meshes + cap before recreating
        this.disposeCellMeshes();
        this.disposeCapMesh();

        // Create new membrane mesh
        this.createMembraneMesh(meshData);

        // Create new ECS mesh if available
        if (meshData.ecsVertices && meshData.ecsFacets) {
            this.createEcsMesh(meshData);
        }

        // Create new cut mesh if available
        if (meshData.cutVertices && meshData.cutFacets) {
            this.createCutMesh(meshData);
        }

        // Cross-section per-cell surfaces (hidden until cross-section mode is on)
        if (meshData.cells) {
            this.createCellMeshes(meshData.cells);
        }
        // Reapply cross-section visibility after recreating meshes
        this.setCrossSectionVisible(this.crossSectionMode);

        // Reset explosion — defer vertex copies until explosion data is set
        this.explosionFactor = 0;
        this.originalVertices = null;
        this._pendingVerticesForExplosion = meshData.vertices;
        if (meshData.ecsVertices) {
            this.originalEcsVertices = new Float32Array(meshData.ecsVertices);
        }
        if (meshData.cutVertices) {
            this.originalCutVertices = new Float32Array(meshData.cutVertices);
        }

        console.log(`Mesh reloaded: ${meshData.metadata.vertex_count} vertices, ${meshData.metadata.facet_count} facets`);
    }

    // Set explosion data (rank centroids for calculating offsets)
    setExplosionData(ranksData, ecsRanksData, cutRanksData, rankCentroids, globalCentroid) {
        this.ranksData = ranksData;
        this.ecsRanksData = ecsRanksData;
        this.cutRanksData = cutRanksData;
        this.rankCentroids = rankCentroids;
        this.globalCentroid = globalCentroid;

        // Lazily create originalVertices copy now that explosion is possible
        if (ranksData && !this.originalVertices && this._pendingVerticesForExplosion) {
            this.originalVertices = new Float32Array(this._pendingVerticesForExplosion);
        }
        this._pendingVerticesForExplosion = null;

        // Initialize visible ranks to all
        if (ranksData && this.numRanks) {
            this.visibleRanks = new Set();
            for (let i = 0; i < this.numRanks; i++) {
                this.visibleRanks.add(i);
            }
        }
    }

    // Apply explosion effect - moves each rank's vertices away from center
    setExplosionFactor(factor) {
        this.explosionFactor = factor;

        if (!this.ranksData || !this.rankCentroids || !this.globalCentroid) {
            return;
        }

        const gc = this.globalCentroid;

        // Update membrane mesh vertices
        if (this.meshObject && this.originalVertices) {
            const positions = this.meshObject.geometry.attributes.position.array;

            for (let i = 0; i < this.ranksData.length; i++) {
                const rank = this.ranksData[i];
                const centroid = this.rankCentroids[rank];

                // Direction from global centroid to rank centroid
                const dx = centroid[0] - gc[0];
                const dy = centroid[1] - gc[1];
                const dz = centroid[2] - gc[2];

                // Apply offset
                positions[i * 3] = this.originalVertices[i * 3] + dx * factor;
                positions[i * 3 + 1] = this.originalVertices[i * 3 + 1] + dy * factor;
                positions[i * 3 + 2] = this.originalVertices[i * 3 + 2] + dz * factor;
            }

            this.meshObject.geometry.attributes.position.needsUpdate = true;
            this.meshObject.geometry.computeVertexNormals();
        }

        // Update ECS mesh vertices
        if (this.ecsMeshObject && this.originalEcsVertices && this.ecsRanksData) {
            const positions = this.ecsMeshObject.geometry.attributes.position.array;

            for (let i = 0; i < this.ecsRanksData.length; i++) {
                const rank = this.ecsRanksData[i];
                const centroid = this.rankCentroids[rank];

                const dx = centroid[0] - gc[0];
                const dy = centroid[1] - gc[1];
                const dz = centroid[2] - gc[2];

                positions[i * 3] = this.originalEcsVertices[i * 3] + dx * factor;
                positions[i * 3 + 1] = this.originalEcsVertices[i * 3 + 1] + dy * factor;
                positions[i * 3 + 2] = this.originalEcsVertices[i * 3 + 2] + dz * factor;
            }

            this.ecsMeshObject.geometry.attributes.position.needsUpdate = true;
            this.ecsMeshObject.geometry.computeVertexNormals();
        }

        // Update cut mesh vertices
        if (this.cutMeshObject && this.originalCutVertices && this.cutRanksData) {
            const positions = this.cutMeshObject.geometry.attributes.position.array;

            for (let i = 0; i < this.cutRanksData.length; i++) {
                const rank = this.cutRanksData[i];
                const centroid = this.rankCentroids[rank];

                const dx = centroid[0] - gc[0];
                const dy = centroid[1] - gc[1];
                const dz = centroid[2] - gc[2];

                positions[i * 3] = this.originalCutVertices[i * 3] + dx * factor;
                positions[i * 3 + 1] = this.originalCutVertices[i * 3 + 1] + dy * factor;
                positions[i * 3 + 2] = this.originalCutVertices[i * 3 + 2] + dz * factor;
            }

            this.cutMeshObject.geometry.attributes.position.needsUpdate = true;
            this.cutMeshObject.geometry.computeVertexNormals();
        }

        // Update interface points positions if they exist
        this.updateInterfacePoints();
    }

    // Update ECS colors based on rank
    updateEcsRankColors(ecsRanks) {
        if (!this.ecsMeshObject) return;

        const geometry = this.ecsMeshObject.geometry;
        const colors = geometry.attributes.color.array;

        // Note: boundary facets are already duplicated in the mesh data,
        // with each copy having uniform rank assignment for all 3 vertices
        for (let i = 0; i < ecsRanks.length; i++) {
            const color = this.rankToColor(ecsRanks[i]);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    // Reset ECS to default gray color
    resetEcsColors() {
        if (!this.ecsMeshObject) return;

        const geometry = this.ecsMeshObject.geometry;
        const colors = geometry.attributes.color.array;

        for (let i = 0; i < colors.length; i += 3) {
            colors[i] = 0.7;
            colors[i + 1] = 0.7;
            colors[i + 2] = 0.8;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    updateBoundingBox(box) {
        // Remove existing box helper
        if (this.boundingBoxHelper) {
            this.scene.remove(this.boundingBoxHelper);
        }

        // Create box geometry
        const width = box.xMax - box.xMin;
        const height = box.yMax - box.yMin;
        const depth = box.zMax - box.zMin;

        // Only create if box has valid dimensions
        if (width > 0 && height > 0 && depth > 0) {
            const boxGeom = new THREE.BoxGeometry(width, height, depth);
            const edges = new THREE.EdgesGeometry(boxGeom);
            const lineMaterial = new THREE.LineBasicMaterial({
                color: 0x00ff00,
                linewidth: 2
            });

            this.boundingBoxHelper = new THREE.LineSegments(edges, lineMaterial);
            this.boundingBoxHelper.position.set(
                (box.xMin + box.xMax) / 2,
                (box.yMin + box.yMax) / 2,
                (box.zMin + box.zMax) / 2
            );

            this.scene.add(this.boundingBoxHelper);
        }

        // Update vertex colors based on bounding box
        if (this.showExcitedHighlight) {
            this.updateExcitedHighlight(box);
        }
    }

    updateExcitedHighlight(box, vExcited = this.vMax, vResting = this.vMin) {
        if (!this.meshObject || !this.meshData) return;

        const vertices = this.meshData.vertices;
        const geometry = this.meshObject.geometry;
        const colors = geometry.attributes.color.array;

        const excitedColor = this.voltageToColor(vExcited);
        const restingColor = this.voltageToColor(vResting);

        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];

            // Check if vertex is inside bounding box
            const inside = (
                x >= box.xMin && x <= box.xMax &&
                y >= box.yMin && y <= box.yMax &&
                z >= box.zMin && z <= box.zMax
            );

            const color = inside ? excitedColor : restingColor;

            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    updateScarBox(box, margin, enabled) {
        // Remove existing scar box helpers
        if (this.scarBoxHelper) {
            this.scene.remove(this.scarBoxHelper);
            this.scarBoxHelper = null;
        }
        if (this.scarBorderBoxHelper) {
            this.scene.remove(this.scarBorderBoxHelper);
            this.scarBorderBoxHelper = null;
        }

        if (!enabled) return;

        const createBox = (b, color) => {
            const w = b.xMax - b.xMin;
            const h = b.yMax - b.yMin;
            const d = b.zMax - b.zMin;
            if (w <= 0 || h <= 0 || d <= 0) return null;
            const geom = new THREE.BoxGeometry(w, h, d);
            const edges = new THREE.EdgesGeometry(geom);
            const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
            const helper = new THREE.LineSegments(edges, mat);
            helper.position.set(
                (b.xMin + b.xMax) / 2,
                (b.yMin + b.yMax) / 2,
                (b.zMin + b.zMax) / 2
            );
            return helper;
        };

        // Inner box (dense scar) - red
        this.scarBoxHelper = createBox(box, 0xff0000);
        if (this.scarBoxHelper) this.scene.add(this.scarBoxHelper);

        // Outer box (border zone) - orange
        const outerBox = {
            xMin: box.xMin - margin, xMax: box.xMax + margin,
            yMin: box.yMin - margin, yMax: box.yMax + margin,
            zMin: box.zMin - margin, zMax: box.zMax + margin,
        };
        this.scarBorderBoxHelper = createBox(outerBox, 0xff8800);
        if (this.scarBorderBoxHelper) this.scene.add(this.scarBorderBoxHelper);
    }

    setScarBoxVisible(visible) {
        if (this.scarBoxHelper) this.scarBoxHelper.visible = visible;
        if (this.scarBorderBoxHelper) this.scarBorderBoxHelper.visible = visible;
    }

    /**
     * Precompute per-vertex scar zone mask.
     * 0 = healthy, 1 = border zone, 2 = dense scar.
     * Called when scar config changes; used by updateVoltageColors for desaturation.
     */
    setScarZones(box, margin, enabled) {
        if (!enabled || !this.meshData) {
            this.scarZoneMask = null;
            return;
        }

        const vertices = this.meshData.vertices;
        const numVertices = vertices.length / 3;
        this.scarZoneMask = new Uint8Array(numVertices);

        const outerBox = {
            xMin: box.xMin - margin, xMax: box.xMax + margin,
            yMin: box.yMin - margin, yMax: box.yMax + margin,
            zMin: box.zMin - margin, zMax: box.zMax + margin,
        };

        for (let i = 0; i < numVertices; i++) {
            const vx = vertices[i * 3], vy = vertices[i * 3 + 1], vz = vertices[i * 3 + 2];

            const inInner = vx >= box.xMin && vx <= box.xMax &&
                            vy >= box.yMin && vy <= box.yMax &&
                            vz >= box.zMin && vz <= box.zMax;

            if (inInner) {
                this.scarZoneMask[i] = 2;
            } else {
                const inOuter = vx >= outerBox.xMin && vx <= outerBox.xMax &&
                                vy >= outerBox.yMin && vy <= outerBox.yMax &&
                                vz >= outerBox.zMin && vz <= outerBox.zMax;
                if (inOuter) {
                    this.scarZoneMask[i] = 1;
                }
            }
        }
    }

    highlightScarZones(box, margin) {
        // Legacy: update scar mask and refresh display
        this.setScarZones(box, margin, true);
        if (this.meshObject) {
            const geometry = this.meshObject.geometry;
            const colors = geometry.attributes.color.array;
            const numVertices = colors.length / 3;
            const restingColor = this.voltageToColor(this.vMin);
            const rr = restingColor.r, rg = restingColor.g, rb = restingColor.b;
            for (let i = 0; i < numVertices; i++) {
                colors[i * 3] = rr;
                colors[i * 3 + 1] = rg;
                colors[i * 3 + 2] = rb;
            }
            this._applyScarDesaturation(colors, numVertices);
            geometry.attributes.color.needsUpdate = true;
        }
    }

    /**
     * Apply desaturation to scar zone vertices in-place.
     * Dense scar: 50% desaturation. Border zone: 30% desaturation.
     */
    _applyScarDesaturation(colors, numVertices) {
        if (!this.scarZoneMask) return;

        for (let i = 0; i < numVertices; i++) {
            const zone = this.scarZoneMask[i];
            if (zone === 0) continue;

            const r = colors[i * 3], g = colors[i * 3 + 1], b = colors[i * 3 + 2];
            // Luminance
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            // Desaturation factor: 0 = full grayscale, 1 = original
            const factor = zone === 2 ? 0.5 : 0.7;
            colors[i * 3]     = r * factor + lum * (1 - factor);
            colors[i * 3 + 1] = g * factor + lum * (1 - factor);
            colors[i * 3 + 2] = b * factor + lum * (1 - factor);
        }
    }

    setVoltageRange(vMin, vMax) {
        this.vMin = vMin;
        this.vMax = vMax;
        this._colorLUT = null; // Invalidate cached LUT
    }

    /**
     * Build a color lookup table (256 entries) for the current colormap and
     * voltage range.  Avoids calling voltageToColor() per-vertex (37M+ calls).
     */
    _buildColorLUT() {
        const N = 256;
        // Flat Float32Array: [r0,g0,b0, r1,g1,b1, ...]
        const lut = new Float32Array(N * 3);
        for (let j = 0; j < N; j++) {
            const t = j / (N - 1);
            const v = this.vMin + t * (this.vMax - this.vMin);
            const c = this.voltageToColor(v);
            lut[j * 3]     = c.r;
            lut[j * 3 + 1] = c.g;
            lut[j * 3 + 2] = c.b;
        }
        this._colorLUT = lut;
        return lut;
    }

    updateVoltageColors(voltages) {
        if (!this.meshObject) return;

        const geometry = this.meshObject.geometry;
        const colors = geometry.attributes.color.array;

        // Use LUT for fast voltage -> color mapping
        const lut = this._colorLUT || this._buildColorLUT();
        const vMin = this.vMin;
        const vRange = this.vMax - this.vMin;
        const lutMax = 255;

        for (let i = 0; i < voltages.length; i++) {
            const t = (voltages[i] - vMin) / vRange;
            // Clamp to [0, 255] and index into LUT
            const idx = (t <= 0 ? 0 : t >= 1 ? lutMax : (t * lutMax) | 0) * 3;
            colors[i * 3]     = lut[idx];
            colors[i * 3 + 1] = lut[idx + 1];
            colors[i * 3 + 2] = lut[idx + 2];
        }

        // Desaturate scar zones
        this._applyScarDesaturation(colors, voltages.length);

        geometry.attributes.color.needsUpdate = true;
        this.colorMode = 'voltage';
    }

    updateRankColors(ranks) {
        if (!this.meshObject) return;

        const geometry = this.meshObject.geometry;
        const colors = geometry.attributes.color.array;

        // ranks is array of rank ID per vertex
        // Note: boundary facets are already duplicated in the mesh data,
        // with each copy having uniform rank assignment for all 3 vertices
        for (let i = 0; i < ranks.length; i++) {
            const color = this.rankToColor(ranks[i]);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geometry.attributes.color.needsUpdate = true;
        this.colorMode = 'rank';
    }

    // ---------------------- Cross-section mode ----------------------

    _buildLUT(colormapName, vMin, vMax) {
        const N = 256;
        const lut = new Float32Array(N * 3);
        const range = vMax - vMin || 1;
        for (let j = 0; j < N; j++) {
            const t = j / (N - 1);
            const v = vMin + t * range;
            const c = this.sampleColormap(colormapName, (v - vMin) / range);
            lut[j * 3]     = c.r;
            lut[j * 3 + 1] = c.g;
            lut[j * 3 + 2] = c.b;
        }
        return lut;
    }

    _getIntraLUT() {
        if (!this._intraLUT) {
            this._intraLUT = this._buildLUT(this.colormapIntra, this.vMinIntra, this.vMaxIntra);
        }
        return this._intraLUT;
    }

    _getExtraLUT() {
        if (!this._extraLUT) {
            this._extraLUT = this._buildLUT(this.colormapExtra, this.vMinExtra, this.vMaxExtra);
        }
        return this._extraLUT;
    }

    setIntraRange(vMin, vMax) {
        this.vMinIntra = vMin;
        this.vMaxIntra = vMax;
        this._intraLUT = null;
    }

    setExtraRange(vMin, vMax) {
        this.vMinExtra = vMin;
        this.vMaxExtra = vMax;
        this._extraLUT = null;
    }

    setIntraColormap(colormapName) {
        if (this.colormaps[colormapName]) {
            this.colormapIntra = colormapName;
            this._intraLUT = null;
        }
    }

    setExtraColormap(colormapName) {
        if (this.colormaps[colormapName]) {
            this.colormapExtra = colormapName;
            this._extraLUT = null;
        }
    }

    getColormapGradientFor(colormapName) {
        const cm = this.colormaps[colormapName] || this.colormaps[this.colormap];
        const stops = cm.colors.map((color, i) => {
            const r = Math.round(color[0] * 255);
            const g = Math.round(color[1] * 255);
            const b = Math.round(color[2] * 255);
            const pos = cm.positions[i] * 100;
            return `rgb(${r}, ${g}, ${b}) ${pos}%`;
        });
        return `linear-gradient(to top, ${stops.join(', ')})`;
    }

    createCellMeshes(cellsMap) {
        this.disposeCellMeshes();
        if (!cellsMap) return;
        const lut = this._getIntraLUT();
        for (const [tag, payload] of cellsMap.entries()) {
            const { vertices, facets } = payload;
            if (!vertices || !facets || vertices.length === 0) continue;
            const numVerts = vertices.length / 3;
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(facets, 1));
            this._applyDrawGroups(geometry);
            const useFlatShading = numVerts > 2_000_000;
            if (!useFlatShading) geometry.computeVertexNormals();

            const colors = new Float32Array(vertices.length);
            const r0 = lut[0], g0 = lut[1], b0 = lut[2];
            for (let i = 0; i < vertices.length; i += 3) {
                colors[i]     = r0;
                colors[i + 1] = g0;
                colors[i + 2] = b0;
            }
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const material = new THREE.MeshPhongMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                flatShading: useFlatShading,
                transparent: false,
                shininess: 30,
            });
            const meshMaterial = geometry.groups.length > 0 ? [material] : material;
            const mesh = new THREE.Mesh(geometry, meshMaterial);
            mesh.visible = false;
            mesh.userData.cellTag = tag;
            this.cellMeshes.set(tag, mesh);
            this.scene.add(mesh);
        }
    }

    disposeCellMeshes() {
        for (const mesh of this.cellMeshes.values()) {
            this.scene.remove(mesh);
            mesh.geometry.dispose();
            const mat = mesh.material;
            if (Array.isArray(mat)) mat.forEach(m => m.dispose());
            else mat.dispose();
        }
        this.cellMeshes.clear();
    }

    createCapMesh(capData) {
        this.disposeCapMesh();
        if (!capData || !capData.vertices || !capData.facets) return;
        const { vertices, facets } = capData;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(facets, 1));

        const lut = this._getExtraLUT();
        const colors = new Float32Array(vertices.length);
        const r0 = lut[0], g0 = lut[1], b0 = lut[2];
        for (let i = 0; i < vertices.length; i += 3) {
            colors[i]     = r0;
            colors[i + 1] = g0;
            colors[i + 2] = b0;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Unlit material: per-vertex colors are interpolated linearly across
        // each triangle (Gouraud), no lighting attenuation. Polygon offset
        // nudges the cap slightly toward the visible side of the clip plane
        // so it never z-fights with the clipped ECS shell.
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: false,
            polygonOffset: true,
            polygonOffsetFactor: -2,
            polygonOffsetUnits: -2,
        });
        this.capMeshObject = new THREE.Mesh(geometry, material);
        this.capMeshObject.visible = this.crossSectionMode;
        this.capMeshObject.renderOrder = 1;
        this.scene.add(this.capMeshObject);
    }

    disposeCapMesh() {
        if (this.capMeshObject) {
            this.scene.remove(this.capMeshObject);
            this.capMeshObject.geometry.dispose();
            this.capMeshObject.material.dispose();
            this.capMeshObject = null;
        }
    }

    setCrossSectionVisible(visible) {
        this.crossSectionMode = !!visible;
        for (const mesh of this.cellMeshes.values()) {
            mesh.visible = this.crossSectionMode;
        }
        if (this.capMeshObject) {
            this.capMeshObject.visible = this.crossSectionMode;
        }
        if (this.meshObject) {
            this.meshObject.visible = !this.crossSectionMode;
        }
        if (this.ecsMeshObject) {
            const mat = this.ecsMeshObject.material;
            if (this.crossSectionMode) {
                // Cross-section: render ECS shell opaque, clipped at the plane
                // so the volume below the plane stays visible. Cap closes it.
                // Hidden until the user has applied a plane (otherwise the
                // unclipped shell would obscure the cells).
                this.ecsMeshObject.visible = !!this._clippingPlane;
                mat.transparent = false;
                mat.opacity = 1.0;
                mat.depthWrite = true;
                mat.clippingPlanes = this._clippingPlane ? [this._clippingPlane] : null;
                mat.needsUpdate = true;
            } else {
                mat.transparent = true;
                mat.opacity = 0.15;
                mat.depthWrite = false;
                mat.clippingPlanes = null;
                mat.needsUpdate = true;
                this.ecsMeshObject.visible = !!this.ecsMeshObject.userData.userVisible;
            }
        }
        if (this._planeHelper) {
            this._planeHelper.visible = this.crossSectionMode;
        }
    }

    // Configure the clipping plane that crops the ECS shell.
    // Convention: `normal` and `offset` describe the user's cut plane n·x = offset.
    // The portion below (n·x < offset) stays visible.
    setClippingPlane(normal, offset) {
        const len = Math.hypot(normal[0], normal[1], normal[2]);
        if (len < 1e-12) return;
        const inv = 1 / len;
        const n = new THREE.Vector3(-normal[0] * inv, -normal[1] * inv, -normal[2] * inv);
        const constant = offset * inv;
        if (!this._clippingPlane) {
            this._clippingPlane = new THREE.Plane(n, constant);
        } else {
            this._clippingPlane.normal.copy(n);
            this._clippingPlane.constant = constant;
        }
        if (this.crossSectionMode && this.ecsMeshObject) {
            this.ecsMeshObject.material.clippingPlanes = [this._clippingPlane];
            this.ecsMeshObject.material.needsUpdate = true;
            this.ecsMeshObject.visible = true;
        }
    }

    clearClippingPlane() {
        this._clippingPlane = null;
        if (this.ecsMeshObject) {
            this.ecsMeshObject.material.clippingPlanes = null;
            this.ecsMeshObject.material.needsUpdate = true;
        }
    }

    // Wireframe preview rectangle showing the configured plane before applying.
    setPlanePreview(normal, offset, visible) {
        if (!visible) {
            if (this._planeHelper) this._planeHelper.visible = false;
            return;
        }
        const len = Math.hypot(normal[0], normal[1], normal[2]);
        if (len < 1e-12) return;
        const inv = 1 / len;
        const n = new THREE.Vector3(normal[0] * inv, normal[1] * inv, normal[2] * inv);
        // Three.js plane equation is n·p + constant = 0, so constant = -offset.
        const constant = -offset * inv;
        const plane = new THREE.Plane(n, constant);
        const size = this._planeHelperSize || 1;
        if (!this._planeHelper) {
            this._planeHelper = new THREE.PlaneHelper(plane, size, 0xe94560);
            this.scene.add(this._planeHelper);
        } else {
            this._planeHelper.plane.copy(plane);
            this._planeHelper.size = size;
            this._planeHelper.updateMatrixWorld(true);
        }
        this._planeHelper.visible = !!this.crossSectionMode;
    }

    setPlanePreviewSize(size) {
        this._planeHelperSize = size;
        if (this._planeHelper) this._planeHelper.size = size;
    }

    updatePhiEShellColors(phi) {
        if (!this.ecsMeshObject || !phi) return;
        const lut = this._getExtraLUT();
        const vMin = this.vMinExtra;
        const vRange = (this.vMaxExtra - this.vMinExtra) || 1;
        const lutMax = 255;
        const colors = this.ecsMeshObject.geometry.attributes.color.array;
        const limit = Math.min(phi.length, colors.length / 3);
        for (let i = 0; i < limit; i++) {
            const t = (phi[i] - vMin) / vRange;
            const idx = (t <= 0 ? 0 : t >= 1 ? lutMax : (t * lutMax) | 0) * 3;
            colors[i * 3]     = lut[idx];
            colors[i * 3 + 1] = lut[idx + 1];
            colors[i * 3 + 2] = lut[idx + 2];
        }
        this.ecsMeshObject.geometry.attributes.color.needsUpdate = true;
    }

    refreshShellColors(phi) {
        this.updatePhiEShellColors(phi);
    }

    // Update vertex colors on every cell mesh from per-tag φ_i arrays.
    // phiByTag: Map<tag, Float32Array> aligned with each cell's expanded vertex order
    updatePhiIColors(phiByTag) {
        if (!phiByTag) return;
        const lut = this._getIntraLUT();
        const vMin = this.vMinIntra;
        const vRange = (this.vMaxIntra - this.vMinIntra) || 1;
        const lutMax = 255;
        for (const [tag, mesh] of this.cellMeshes.entries()) {
            const phi = phiByTag.get(tag);
            if (!phi) continue;
            const colors = mesh.geometry.attributes.color.array;
            const limit = Math.min(phi.length, colors.length / 3);
            for (let i = 0; i < limit; i++) {
                const t = (phi[i] - vMin) / vRange;
                const idx = (t <= 0 ? 0 : t >= 1 ? lutMax : (t * lutMax) | 0) * 3;
                colors[i * 3]     = lut[idx];
                colors[i * 3 + 1] = lut[idx + 1];
                colors[i * 3 + 2] = lut[idx + 2];
            }
            mesh.geometry.attributes.color.needsUpdate = true;
        }
    }

    updatePhiECapColors(phi) {
        if (!this.capMeshObject || !phi) return;
        const lut = this._getExtraLUT();
        const vMin = this.vMinExtra;
        const vRange = (this.vMaxExtra - this.vMinExtra) || 1;
        const lutMax = 255;
        const colors = this.capMeshObject.geometry.attributes.color.array;
        const limit = Math.min(phi.length, colors.length / 3);
        for (let i = 0; i < limit; i++) {
            const t = (phi[i] - vMin) / vRange;
            const idx = (t <= 0 ? 0 : t >= 1 ? lutMax : (t * lutMax) | 0) * 3;
            colors[i * 3]     = lut[idx];
            colors[i * 3 + 1] = lut[idx + 1];
            colors[i * 3 + 2] = lut[idx + 2];
        }
        this.capMeshObject.geometry.attributes.color.needsUpdate = true;
    }

    // Re-apply current LUTs to all existing cell / cap geometry (when range or
    // colormap changes but data hasn't refetched). Caller passes the most
    // recent data so we don't need to keep an internal copy.
    refreshIntraColors(phiByTag) {
        this.updatePhiIColors(phiByTag);
    }

    refreshExtraColors(phi) {
        this.updatePhiECapColors(phi);
    }

    setNumRanks(numRanks) {
        this.numRanks = numRanks;
    }

    getColorMode() {
        return this.colorMode;
    }

    setBoundingBoxVisible(visible) {
        if (this.boundingBoxHelper) {
            this.boundingBoxHelper.visible = visible;
        }
    }

    setExcitedRegionHighlight(enabled) {
        this.showExcitedHighlight = enabled;
        if (this.meshObject) {
            if (!enabled) {
                // Reset to default color (neutral gray)
                const colors = this.meshObject.geometry.attributes.color.array;
                for (let i = 0; i < colors.length; i += 3) {
                    colors[i] = 0.5;
                    colors[i + 1] = 0.5;
                    colors[i + 2] = 0.5;
                }
                this.meshObject.geometry.attributes.color.needsUpdate = true;
            }
        }
    }

    clearAllMeshes() {
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            const mat = this.meshObject.material;
            if (Array.isArray(mat)) { mat.forEach(m => m.dispose()); } else { mat.dispose(); }
            this.meshObject = null;
        }
        if (this.ecsMeshObject) {
            this.scene.remove(this.ecsMeshObject);
            this.ecsMeshObject.geometry.dispose();
            this.ecsMeshObject.material.dispose();
            this.ecsMeshObject = null;
        }
        if (this.cutMeshObject) {
            this.scene.remove(this.cutMeshObject);
            this.cutMeshObject.geometry.dispose();
            this.cutMeshObject.material.dispose();
            this.cutMeshObject = null;
        }
        this.disposeCellMeshes();
        this.disposeCapMesh();
        this.meshData = null;
        this.originalVertices = null;
        this._pendingVerticesForExplosion = null;
        this.originalEcsVertices = null;
        this.originalCutVertices = null;
        this._colorLUT = null;
        this._intraLUT = null;
        this._extraLUT = null;
    }

    showBoundsOutline(bounds) {
        // Remove old outline if any
        if (this._boundsOutline) {
            this.scene.remove(this._boundsOutline);
            this._boundsOutline = null;
        }

        const sx = bounds.x[1] - bounds.x[0];
        const sy = bounds.y[1] - bounds.y[0];
        const sz = bounds.z[1] - bounds.z[0];
        const cx = (bounds.x[0] + bounds.x[1]) / 2;
        const cy = (bounds.y[0] + bounds.y[1]) / 2;
        const cz = (bounds.z[0] + bounds.z[1]) / 2;

        const geo = new THREE.BoxGeometry(sx, sy, sz);
        const edges = new THREE.EdgesGeometry(geo);
        const mat = new THREE.LineBasicMaterial({ color: 0x4ade80, linewidth: 2 });
        this._boundsOutline = new THREE.LineSegments(edges, mat);
        this._boundsOutline.position.set(cx, cy, cz);
        this.scene.add(this._boundsOutline);

        // Position camera to frame the bounds
        const maxSize = Math.max(sx, sy, sz);
        this.camera.position.set(
            cx + maxSize * 0.8,
            cy + maxSize * 0.5,
            cz + maxSize * 0.8
        );
        this.controls.target.set(cx, cy, cz);
        this.controls.update();
    }

    resetCamera() {
        if (!this.meshData) return;

        const bounds = this.meshData.metadata.bounds;
        const centerX = (bounds.x[0] + bounds.x[1]) / 2;
        const centerY = (bounds.y[0] + bounds.y[1]) / 2;
        const centerZ = (bounds.z[0] + bounds.z[1]) / 2;

        const sizeX = bounds.x[1] - bounds.x[0];
        const sizeY = bounds.y[1] - bounds.y[0];
        const sizeZ = bounds.z[1] - bounds.z[0];
        const maxSize = Math.max(sizeX, sizeY, sizeZ);

        this.camera.position.set(
            centerX + maxSize * 0.8,
            centerY + maxSize * 0.5,
            centerZ + maxSize * 0.8
        );

        this.controls.target.set(centerX, centerY, centerZ);
        this.controls.update();
    }

    onResize() {
        const container = document.getElementById(this.containerId);
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    getCameraState() {
        return {
            position: this.camera.position.toArray(),
            target: this.controls.target.toArray(),
            up: this.camera.up.toArray(),
            fov: this.camera.fov
        };
    }

    // Set which ranks are visible (for partition view filtering)
    setVisibleRanks(visibleRanks) {
        this.visibleRanks = new Set(visibleRanks);
        this.updateRankVisibility();
        this.updateControlsTarget();
    }

    // Compute centroid of visible subdomains
    getVisibleCentroid() {
        if (this.rankCentroids && this.visibleRanks && this.visibleRanks.size > 0) {
            let sumX = 0, sumY = 0, sumZ = 0;
            let count = 0;
            for (const rank of this.visibleRanks) {
                if (this.rankCentroids[rank]) {
                    sumX += this.rankCentroids[rank][0];
                    sumY += this.rankCentroids[rank][1];
                    sumZ += this.rankCentroids[rank][2];
                    count++;
                }
            }
            if (count > 0) {
                return new THREE.Vector3(sumX / count, sumY / count, sumZ / count);
            }
        }
        if (this.globalCentroid) {
            return new THREE.Vector3(
                this.globalCentroid[0],
                this.globalCentroid[1],
                this.globalCentroid[2]
            );
        }
        return this.controls.target.clone();
    }

    // Update controls target to centroid of visible subdomains
    updateControlsTarget() {
        const centroid = this.getVisibleCentroid();
        this.controls.target.copy(centroid);
        this.controls.update();
    }

    // Update mesh visibility based on selected ranks - actually hide geometry
    updateRankVisibility() {
        if (!this.meshObject || !this.ranksData || !this.visibleRanks) return;

        const geometry = this.meshObject.geometry;
        const colors = geometry.attributes.color.array;

        // Store original indices if not already stored
        if (!this.originalFacets) {
            this.originalFacets = new Uint32Array(geometry.index.array);
        }

        // Build filtered index array - only include triangles where vertices are in visible ranks
        const filteredIndices = [];
        for (let i = 0; i < this.originalFacets.length; i += 3) {
            const v0 = this.originalFacets[i];
            const v1 = this.originalFacets[i + 1];
            const v2 = this.originalFacets[i + 2];

            // Check if any vertex of this triangle belongs to a visible rank
            // (vertices on rank boundaries are duplicated, so checking first vertex is sufficient)
            const rank = this.ranksData[v0];
            if (this.visibleRanks.has(rank)) {
                filteredIndices.push(v0, v1, v2);
            }
        }

        // Update geometry index
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(filteredIndices), 1));
        this._applyDrawGroups(geometry);

        // Update colors for visible vertices (with interface highlighting)
        for (let i = 0; i < this.ranksData.length; i++) {
            const rank = this.ranksData[i];
            // Check if this vertex is an interface DOF that should be highlighted
            if (this.highlightedInterfaceMap && this.dofIndices) {
                const dofIndex = this.dofIndices[i];
                if (this.highlightedInterfaceMap.has(dofIndex)) {
                    // Get the interface-specific color
                    const interfaceIdx = this.highlightedInterfaceMap.get(dofIndex);
                    const color = this.interfaceToColor(interfaceIdx);
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                } else {
                    const color = this.rankToColor(rank);
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                }
            } else {
                const color = this.rankToColor(rank);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }
        }

        geometry.attributes.color.needsUpdate = true;
        geometry.computeVertexNormals();

        // Update ECS mesh - filter by rank
        if (this.ecsMeshObject && this.ecsRanksData) {
            const ecsGeometry = this.ecsMeshObject.geometry;
            if (!this.originalEcsFacets) {
                this.originalEcsFacets = new Uint32Array(ecsGeometry.index.array);
            }

            const ecsFilteredIndices = [];
            for (let i = 0; i < this.originalEcsFacets.length; i += 3) {
                const v0 = this.originalEcsFacets[i];
                const rank = this.ecsRanksData[v0];
                if (this.visibleRanks.has(rank)) {
                    ecsFilteredIndices.push(this.originalEcsFacets[i], this.originalEcsFacets[i + 1], this.originalEcsFacets[i + 2]);
                }
            }

            ecsGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(ecsFilteredIndices), 1));

            // Update ECS colors (with interface highlighting)
            const ecsColors = ecsGeometry.attributes.color.array;
            for (let i = 0; i < this.ecsRanksData.length; i++) {
                // Check if this ECS vertex is an interface DOF
                if (this.highlightedInterfaceMap && this.ecsDofIndices) {
                    const dofIndex = this.ecsDofIndices[i];
                    if (this.highlightedInterfaceMap.has(dofIndex)) {
                        // Get the interface-specific color
                        const interfaceIdx = this.highlightedInterfaceMap.get(dofIndex);
                        const color = this.interfaceToColor(interfaceIdx);
                        ecsColors[i * 3] = color.r;
                        ecsColors[i * 3 + 1] = color.g;
                        ecsColors[i * 3 + 2] = color.b;
                    } else {
                        const color = this.rankToColor(this.ecsRanksData[i]);
                        ecsColors[i * 3] = color.r;
                        ecsColors[i * 3 + 1] = color.g;
                        ecsColors[i * 3 + 2] = color.b;
                    }
                } else {
                    const color = this.rankToColor(this.ecsRanksData[i]);
                    ecsColors[i * 3] = color.r;
                    ecsColors[i * 3 + 1] = color.g;
                    ecsColors[i * 3 + 2] = color.b;
                }
            }
            ecsGeometry.attributes.color.needsUpdate = true;
            ecsGeometry.computeVertexNormals();
        }

        // Update cut mesh - filter by rank
        if (this.cutMeshObject && this.cutRanksData) {
            const cutGeometry = this.cutMeshObject.geometry;
            if (!this.originalCutFacets) {
                this.originalCutFacets = new Uint32Array(cutGeometry.index.array);
            }

            const cutFilteredIndices = [];
            for (let i = 0; i < this.originalCutFacets.length; i += 3) {
                const v0 = this.originalCutFacets[i];
                const rank = this.cutRanksData[v0];
                if (this.visibleRanks.has(rank)) {
                    cutFilteredIndices.push(this.originalCutFacets[i], this.originalCutFacets[i + 1], this.originalCutFacets[i + 2]);
                }
            }

            cutGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(cutFilteredIndices), 1));

            // Update cut colors
            const cutColors = cutGeometry.attributes.color.array;
            for (let i = 0; i < this.cutRanksData.length; i++) {
                const color = this.rankToColor(this.cutRanksData[i]);
                cutColors[i * 3] = color.r;
                cutColors[i * 3 + 1] = color.g;
                cutColors[i * 3 + 2] = color.b;
            }
            cutGeometry.attributes.color.needsUpdate = true;
            cutGeometry.computeVertexNormals();
        }

        // Update interface points to reflect visible ranks
        this.updateInterfacePoints();
    }

    // Restore full mesh (all ranks visible)
    restoreFullMesh() {
        if (this.meshObject && this.originalFacets) {
            const geometry = this.meshObject.geometry;
            geometry.setIndex(new THREE.BufferAttribute(this.originalFacets, 1));
            this._applyDrawGroups(geometry);
            geometry.computeVertexNormals();
        }
        if (this.ecsMeshObject && this.originalEcsFacets) {
            const ecsGeometry = this.ecsMeshObject.geometry;
            ecsGeometry.setIndex(new THREE.BufferAttribute(this.originalEcsFacets, 1));
            this._applyDrawGroups(ecsGeometry);
            ecsGeometry.computeVertexNormals();
        }
        if (this.cutMeshObject && this.originalCutFacets) {
            const cutGeometry = this.cutMeshObject.geometry;
            cutGeometry.setIndex(new THREE.BufferAttribute(this.originalCutFacets, 1));
            this._applyDrawGroups(cutGeometry);
            cutGeometry.computeVertexNormals();
        }

        // Update interface points (now all ranks are visible)
        this.updateInterfacePoints();
    }

    // Store DOF index mapping for interface highlighting
    setDofIndices(dofIndices) {
        this.dofIndices = dofIndices;
    }

    // Store ECS DOF index mapping for interface highlighting on ECS mesh
    setEcsDofIndices(ecsDofIndices) {
        this.ecsDofIndices = ecsDofIndices;
    }

    // Store DOF type mapping (vertex/edge/face) for interface visualization
    setInterfaceDofTypes(dofTypes) {
        this.interfaceDofTypes = dofTypes;
    }

    // Get color for an interface (by global interface index)
    interfaceToColor(interfaceIndex) {
        const colorIndex = interfaceIndex % this.interfaceColors.length;
        return this.interfaceColors[colorIndex];
    }

    // Set interface data with per-interface coloring
    // interfaceMap: Map from DOF index -> interface global index
    setHighlightedInterfaceDofs(interfaceMap) {
        this.highlightedInterfaceMap = interfaceMap;
        // Refresh visibility to apply highlighting
        if (this.visibleRanks) {
            this.updateRankVisibility();
        }
        // Update interface points overlay
        this.updateInterfacePoints();
    }

    // Clear interface highlighting
    clearInterfaceHighlight() {
        this.highlightedInterfaceMap = null;
        if (this.visibleRanks) {
            this.updateRankVisibility();
        }
        // Remove interface points overlay
        this.removeInterfacePoints();
    }

    // Create/update opaque point cloud for interface vertices
    updateInterfacePoints() {
        // Remove existing interface points
        this.removeInterfacePoints();

        if (!this.highlightedInterfaceMap || this.highlightedInterfaceMap.size === 0) {
            return;
        }

        // Separate arrays for vertex-type DOFs (bigger, light blue) and edge/face DOFs
        const vertexPositions = [];
        const vertexColors = [];
        const otherPositions = [];
        const otherColors = [];

        // Red color for vertex-type interface DOFs
        const vertexColor = { r: 1.0, g: 0.0, b: 0.0 };  // Red

        // Helper to add a point to the appropriate array
        const addPoint = (x, y, z, dofIndex) => {
            const dofType = this.interfaceDofTypes ? this.interfaceDofTypes[dofIndex] : null;
            if (dofType === 'vertex') {
                vertexPositions.push(x, y, z);
                vertexColors.push(vertexColor.r, vertexColor.g, vertexColor.b);
            } else {
                otherPositions.push(x, y, z);
                const interfaceIdx = this.highlightedInterfaceMap.get(dofIndex);
                const color = this.interfaceToColor(interfaceIdx);
                otherColors.push(color.r, color.g, color.b);
            }
        };

        // Collect interface vertex positions and colors from membrane mesh
        // Only include vertices belonging to visible ranks
        if (this.meshObject && this.dofIndices) {
            const membranePositions = this.meshObject.geometry.attributes.position.array;

            for (let i = 0; i < this.dofIndices.length; i++) {
                // Skip vertices not belonging to visible ranks
                if (this.ranksData && this.visibleRanks && !this.visibleRanks.has(this.ranksData[i])) {
                    continue;
                }

                const dofIndex = this.dofIndices[i];
                if (this.highlightedInterfaceMap.has(dofIndex)) {
                    addPoint(
                        membranePositions[i * 3],
                        membranePositions[i * 3 + 1],
                        membranePositions[i * 3 + 2],
                        dofIndex
                    );
                }
            }
        }

        // Collect interface vertex positions and colors from ECS mesh
        // Only include vertices belonging to visible ranks
        if (this.ecsMeshObject && this.ecsDofIndices && this.ecsMeshObject.visible) {
            const ecsPositions = this.ecsMeshObject.geometry.attributes.position.array;

            for (let i = 0; i < this.ecsDofIndices.length; i++) {
                // Skip vertices not belonging to visible ranks
                if (this.ecsRanksData && this.visibleRanks && !this.visibleRanks.has(this.ecsRanksData[i])) {
                    continue;
                }

                const dofIndex = this.ecsDofIndices[i];
                if (this.highlightedInterfaceMap.has(dofIndex)) {
                    addPoint(
                        ecsPositions[i * 3],
                        ecsPositions[i * 3 + 1],
                        ecsPositions[i * 3 + 2],
                        dofIndex
                    );
                }
            }
        }

        // Create instanced spheres for vertex-type DOFs (3D spheres, light blue)
        if (vertexPositions.length > 0) {
            const sphereRadius = 0.6;  // Adjust based on mesh scale
            const sphereGeometry = new THREE.SphereGeometry(sphereRadius, 16, 12);
            const sphereMaterial = new THREE.MeshPhongMaterial({
                color: new THREE.Color(vertexColor.r, vertexColor.g, vertexColor.b),
                shininess: 50
            });

            const numVertices = vertexPositions.length / 3;
            this.interfaceVertexPoints = new THREE.InstancedMesh(sphereGeometry, sphereMaterial, numVertices);

            const dummy = new THREE.Object3D();
            for (let i = 0; i < numVertices; i++) {
                dummy.position.set(
                    vertexPositions[i * 3],
                    vertexPositions[i * 3 + 1],
                    vertexPositions[i * 3 + 2]
                );
                dummy.updateMatrix();
                this.interfaceVertexPoints.setMatrixAt(i, dummy.matrix);
            }
            this.interfaceVertexPoints.instanceMatrix.needsUpdate = true;

            this.scene.add(this.interfaceVertexPoints);
        }

        // Create point cloud for edge/face DOFs (smaller dots, per-interface colors)
        if (otherPositions.length > 0) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(otherPositions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(otherColors, 3));

            const material = new THREE.PointsMaterial({
                size: 3,
                vertexColors: true,
                sizeAttenuation: false,
                depthTest: true,
                depthWrite: true
            });

            this.interfacePoints = new THREE.Points(geometry, material);
            this.scene.add(this.interfacePoints);
        }
    }

    // Vertex picking via raycasting
    setupPickingHandler() {
        const canvas = this.renderer.domElement;

        // Track pointer movement to distinguish click from drag
        canvas.addEventListener('pointerdown', () => { this._pickPointerMoved = false; });
        canvas.addEventListener('pointermove', () => { this._pickPointerMoved = true; });

        canvas.addEventListener('click', (event) => {
            if (this._pickPointerMoved) return;  // was a drag, not a click
            if (!this.meshObject || !this.onVertexPicked) return;

            const rect = canvas.getBoundingClientRect();
            this.mouse.x =  ((event.clientX - rect.left) / rect.width)  * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top)  / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObject(this.meshObject);
            if (intersects.length === 0) return;

            const hit = intersects[0];
            const face = hit.face;
            const pos = this.meshObject.geometry.attributes.position.array;

            // Pick the face vertex closest to the actual intersection point
            let closestIdx = face.a;
            let minDist = Infinity;
            for (const vi of [face.a, face.b, face.c]) {
                const dx = pos[vi * 3]     - hit.point.x;
                const dy = pos[vi * 3 + 1] - hit.point.y;
                const dz = pos[vi * 3 + 2] - hit.point.z;
                const d = dx*dx + dy*dy + dz*dz;
                if (d < minDist) { minDist = d; closestIdx = vi; }
            }

            this.setPickMarker(hit.point);
            this.onVertexPicked(closestIdx, hit.point);
        });
    }

    setPickMarker(position) {
        if (!this.pickMarker) {
            const geo = new THREE.SphereGeometry(1.5, 10, 8);
            const mat = new THREE.MeshBasicMaterial({ color: 0xffff00, depthTest: false });
            this.pickMarker = new THREE.Mesh(geo, mat);
            this.pickMarker.renderOrder = 999;
            this.scene.add(this.pickMarker);
        }
        this.pickMarker.position.copy(position);
        this.pickMarker.visible = true;
    }

    clearPickMarker() {
        if (this.pickMarker) this.pickMarker.visible = false;
    }

    // Remove interface points overlay
    removeInterfacePoints() {
        if (this.interfacePoints) {
            this.scene.remove(this.interfacePoints);
            this.interfacePoints.geometry.dispose();
            this.interfacePoints.material.dispose();
            this.interfacePoints = null;
        }
        if (this.interfaceVertexPoints) {
            this.scene.remove(this.interfaceVertexPoints);
            this.interfaceVertexPoints.geometry.dispose();
            this.interfaceVertexPoints.material.dispose();
            this.interfaceVertexPoints = null;
        }
    }
}
