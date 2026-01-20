// viewer.js - Three.js 3D visualization

class Viewer {
    constructor(containerId) {
        this.containerId = containerId;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.meshObject = null;
        this.boundingBoxHelper = null;
        this.meshData = null;
        this.showExcitedHighlight = true;

        // Voltage range for colormap
        this.vMin = -80;
        this.vMax = 0;
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

        // Position camera
        this.resetCamera();

        // Start render loop
        this.animate();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    createMembraneMesh(meshData) {
        const { vertices, facets, metadata } = meshData;

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

    async reloadMesh(meshData) {
        // Remove old mesh
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
        }

        // Update mesh data reference
        this.meshData = meshData;

        // Create new mesh
        this.createMembraneMesh(meshData);

        console.log(`Mesh reloaded: ${meshData.metadata.vertex_count} vertices, ${meshData.metadata.facet_count} facets`);
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
