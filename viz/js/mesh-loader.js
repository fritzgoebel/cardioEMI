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

        // Try to load ECS (exterior) mesh data
        let ecsVertices = null;
        let ecsFacets = null;
        try {
            const ecsVerticesResponse = await fetch(`${meshPath}/ecs_vertices.bin`);
            if (ecsVerticesResponse.ok) {
                const ecsVerticesBuffer = await ecsVerticesResponse.arrayBuffer();
                ecsVertices = new Float32Array(ecsVerticesBuffer);
                console.log(`  ECS vertices loaded: ${ecsVertices.length / 3} points`);

                const ecsFacetsResponse = await fetch(`${meshPath}/ecs_facets.bin`);
                if (ecsFacetsResponse.ok) {
                    const ecsFacetsBuffer = await ecsFacetsResponse.arrayBuffer();
                    ecsFacets = new Uint32Array(ecsFacetsBuffer);
                    console.log(`  ECS facets loaded: ${ecsFacets.length / 3} triangles`);
                }
            }
        } catch (e) {
            console.log('  ECS mesh not available');
        }

        // Try to load partition cut facets (internal facets at partition boundaries)
        let cutVertices = null;
        let cutFacets = null;
        try {
            const cutVerticesResponse = await fetch(`${meshPath}/cut_vertices.bin`);
            if (cutVerticesResponse.ok) {
                const cutVerticesBuffer = await cutVerticesResponse.arrayBuffer();
                cutVertices = new Float32Array(cutVerticesBuffer);
                console.log(`  Cut vertices loaded: ${cutVertices.length / 3} points`);

                const cutFacetsResponse = await fetch(`${meshPath}/cut_facets.bin`);
                if (cutFacetsResponse.ok) {
                    const cutFacetsBuffer = await cutFacetsResponse.arrayBuffer();
                    cutFacets = new Uint32Array(cutFacetsBuffer);
                    console.log(`  Cut facets loaded: ${cutFacets.length / 3} triangles`);
                }
            }
        } catch (e) {
            console.log('  Cut mesh not available');
        }

        return {
            vertices,
            facets,
            tags,
            metadata,
            ecsVertices,
            ecsFacets,
            cutVertices,
            cutFacets
        };
    }
}
