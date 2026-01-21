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

    // Blue to Red colormap (like ParaView's "Cool to Warm")
    voltageToColor(v) {
        // Normalize voltage to 0-1 range
        const t = Math.max(0, Math.min(1, (v - this.vMin) / (this.vMax - this.vMin)));

        // Blue (cold) -> White (mid) -> Red (hot)
        let r, g, b;
        if (t < 0.5) {
            // Blue to White
            const s = t * 2;
            r = s;
            g = s;
            b = 1;
        } else {
            // White to Red
            const s = (t - 0.5) * 2;
            r = 1;
            g = 1 - s;
            b = 1 - s;
        }

        return { r, g, b };
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
}
