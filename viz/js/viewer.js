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
        this.meshData = null;
        this.showExcitedHighlight = true;

        // Voltage range for colormap
        this.vMin = -80;
        this.vMax = 0;

        // Rank coloring
        this.numRanks = 1;
        this.colorMode = 'voltage'; // 'voltage' or 'rank'

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

    // Get CSS gradient for the current colormap (for colorbar)
    getColormapGradient() {
        const cm = this.colormaps[this.colormap];
        const stops = cm.colors.map((color, i) => {
            const r = Math.round(color[0] * 255);
            const g = Math.round(color[1] * 255);
            const b = Math.round(color[2] * 255);
            const pos = (1 - cm.positions[i]) * 100;  // Invert for top-to-bottom
            return `rgb(${r}, ${g}, ${b}) ${pos}%`;
        });
        return `linear-gradient(to bottom, ${stops.join(', ')})`;
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
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

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

        // Position camera
        this.resetCamera();

        // Start render loop
        this.animate();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    createMembraneMesh(meshData) {
        const { vertices, facets, metadata } = meshData;

        // Store original vertices for explosion effect
        this.originalVertices = new Float32Array(vertices);

        // Create BufferGeometry
        const geometry = new THREE.BufferGeometry();

        // Set position attribute (vertices is flat array [x0,y0,z0, x1,y1,z1, ...])
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Set index attribute (facets is [v0,v1,v2, v0,v1,v2, ...])
        geometry.setIndex(new THREE.BufferAttribute(facets, 1));

        // Compute normals for lighting
        geometry.computeVertexNormals();

        // Create material with vertex colors
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            flatShading: false,
            transparent: false,
            shininess: 30
        });

        // Initialize vertex colors (default resting state - blue)
        const colors = new Float32Array(vertices.length);
        const restingColor = this.voltageToColor(this.vMin);
        for (let i = 0; i < vertices.length; i += 3) {
            colors[i] = restingColor.r;
            colors[i + 1] = restingColor.g;
            colors[i + 2] = restingColor.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create mesh
        this.meshObject = new THREE.Mesh(geometry, material);
        this.scene.add(this.meshObject);
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
        this.scene.add(this.ecsMeshObject);

        console.log(`ECS mesh created: ${ecsVertices.length / 3} vertices`);
    }

    setEcsVisible(visible) {
        if (this.ecsMeshObject) {
            this.ecsMeshObject.visible = visible;
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
            this.meshObject.material.dispose();
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

        // Reset explosion
        this.explosionFactor = 0;
        this.originalVertices = new Float32Array(meshData.vertices);
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

        // Initialize visible ranks to all
        if (ranksData) {
            const maxRank = Math.max(...ranksData);
            this.visibleRanks = new Set();
            for (let i = 0; i <= maxRank; i++) {
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

            // Get voltage and convert to color
            const voltage = inside ? vExcited : vResting;
            const color = this.voltageToColor(voltage);

            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    setVoltageRange(vMin, vMax) {
        this.vMin = vMin;
        this.vMax = vMax;
    }

    updateVoltageColors(voltages) {
        if (!this.meshObject) return;

        const geometry = this.meshObject.geometry;
        const colors = geometry.attributes.color.array;

        // voltages is array of voltage per vertex
        for (let i = 0; i < voltages.length; i++) {
            const color = this.voltageToColor(voltages[i]);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

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
            geometry.computeVertexNormals();
        }
        if (this.ecsMeshObject && this.originalEcsFacets) {
            const ecsGeometry = this.ecsMeshObject.geometry;
            ecsGeometry.setIndex(new THREE.BufferAttribute(this.originalEcsFacets, 1));
            ecsGeometry.computeVertexNormals();
        }
        if (this.cutMeshObject && this.originalCutFacets) {
            const cutGeometry = this.cutMeshObject.geometry;
            cutGeometry.setIndex(new THREE.BufferAttribute(this.originalCutFacets, 1));
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

        const pointPositions = [];
        const pointColors = [];

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
                    // Add position
                    pointPositions.push(
                        membranePositions[i * 3],
                        membranePositions[i * 3 + 1],
                        membranePositions[i * 3 + 2]
                    );
                    // Add color based on interface index
                    const interfaceIdx = this.highlightedInterfaceMap.get(dofIndex);
                    const color = this.interfaceToColor(interfaceIdx);
                    pointColors.push(color.r, color.g, color.b);
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
                    // Add position
                    pointPositions.push(
                        ecsPositions[i * 3],
                        ecsPositions[i * 3 + 1],
                        ecsPositions[i * 3 + 2]
                    );
                    // Add color based on interface index
                    const interfaceIdx = this.highlightedInterfaceMap.get(dofIndex);
                    const color = this.interfaceToColor(interfaceIdx);
                    pointColors.push(color.r, color.g, color.b);
                }
            }
        }

        // Create point cloud if we have any interface points
        if (pointPositions.length > 0) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(pointPositions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(pointColors, 3));

            const material = new THREE.PointsMaterial({
                size: 3,
                vertexColors: true,
                sizeAttenuation: false,  // Constant size regardless of distance
                depthTest: true,
                depthWrite: true
            });

            this.interfacePoints = new THREE.Points(geometry, material);
            this.scene.add(this.interfacePoints);
        }
    }

    // Remove interface points overlay
    removeInterfacePoints() {
        if (this.interfacePoints) {
            this.scene.remove(this.interfacePoints);
            this.interfacePoints.geometry.dispose();
            this.interfacePoints.material.dispose();
            this.interfacePoints = null;
        }
    }
}
