// mesh-loader.js - Load and parse binary mesh data

class MeshLoader {
    constructor(basePath = 'data') {
        this.basePath = basePath;
        this.currentMesh = null;
    }

    setMesh(meshName) {
        this.currentMesh = meshName;
    }

    getMeshPath() {
        if (this.currentMesh) {
            return `${this.basePath}/${this.currentMesh}`;
        }
        return this.basePath;
    }

    async load() {
        const meshPath = this.getMeshPath();
        console.log(`Loading mesh data from ${meshPath}...`);

        // Load metadata first
        const metadataResponse = await fetch(`${meshPath}/mesh_metadata.json`);
        if (!metadataResponse.ok) {
            throw new Error('Failed to load mesh metadata');
        }
        const metadata = await metadataResponse.json();
        console.log('  Metadata loaded:', metadata);

        // Load binary vertex data
        const verticesResponse = await fetch(`${meshPath}/mesh_vertices.bin`);
        if (!verticesResponse.ok) {
            throw new Error('Failed to load vertex data');
        }
        const verticesBuffer = await verticesResponse.arrayBuffer();
        const vertices = new Float32Array(verticesBuffer);
        console.log(`  Vertices loaded: ${vertices.length / 3} points`);

        // Load binary facet data
        const facetsResponse = await fetch(`${meshPath}/membrane_facets.bin`);
        if (!facetsResponse.ok) {
            throw new Error('Failed to load facet data');
        }
        const facetsBuffer = await facetsResponse.arrayBuffer();
        const facets = new Uint32Array(facetsBuffer);
        console.log(`  Facets loaded: ${facets.length / 3} triangles`);

        // Load facet tags (for coloring by membrane type)
        let tags = null;
        try {
            const tagsResponse = await fetch(`${meshPath}/membrane_tags.bin`);
            if (tagsResponse.ok) {
                const tagsBuffer = await tagsResponse.arrayBuffer();
                tags = new Int32Array(tagsBuffer);
                console.log(`  Tags loaded: ${tags.length} values`);
            }
        } catch (e) {
            console.log('  Tags not available');
        }

        return {
            vertices,
            facets,
            tags,
            metadata
        };
    }
}
