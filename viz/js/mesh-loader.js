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

    async load(statusCallback) {
        const meshPath = this.getMeshPath();
        const report = statusCallback || (() => {});
        console.log(`Loading mesh data from ${meshPath}...`);

        // Load metadata first
        report('Loading metadata...');
        const metadataResponse = await fetch(`${meshPath}/mesh_metadata.json`);
        if (!metadataResponse.ok) {
            throw new Error('Failed to load mesh metadata');
        }
        const metadata = await metadataResponse.json();
        console.log('  Metadata loaded:', metadata);

        // Load binary vertex data (can be very large)
        const vertexMB = ((metadata.vertex_count * 3 * 4) / 1e6).toFixed(0);
        report(`Loading vertices (${vertexMB} MB)...`);
        console.time('  Vertex fetch');
        const verticesResponse = await fetch(`${meshPath}/mesh_vertices.bin`);
        if (!verticesResponse.ok) {
            throw new Error('Failed to load vertex data');
        }
        const verticesBuffer = await verticesResponse.arrayBuffer();
        const vertices = new Float32Array(verticesBuffer);
        console.timeEnd('  Vertex fetch');
        console.log(`  Vertices loaded: ${vertices.length / 3} points`);

        // Load binary facet data
        const facetMB = ((metadata.facet_count * 3 * 4) / 1e6).toFixed(0);
        report(`Loading facets (${facetMB} MB)...`);
        console.time('  Facet fetch');
        const facetsResponse = await fetch(`${meshPath}/membrane_facets.bin`);
        if (!facetsResponse.ok) {
            throw new Error('Failed to load facet data');
        }
        const facetsBuffer = await facetsResponse.arrayBuffer();
        const facets = new Uint32Array(facetsBuffer);
        console.timeEnd('  Facet fetch');
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

        // Optional cross-section payload: per-cell closed surfaces (intracellular).
        // Cap geometry is loaded on demand once a plane has been applied.
        let cells = null;
        if (metadata.cross_section && Array.isArray(metadata.cross_section.cell_tags)) {
            cells = new Map();
            const tagsList = metadata.cross_section.cell_tags;
            const totalTagCount = tagsList.length;
            for (let i = 0; i < totalTagCount; i++) {
                const tag = tagsList[i];
                try {
                    report(`Loading cell surface ${i + 1}/${totalTagCount} (tag ${tag})...`);
                    const vResp = await fetch(`${meshPath}/cells/${tag}_vertices.bin`);
                    const fResp = await fetch(`${meshPath}/cells/${tag}_facets.bin`);
                    if (!vResp.ok || !fResp.ok) continue;
                    const cellVerts = new Float32Array(await vResp.arrayBuffer());
                    const cellFacets = new Uint32Array(await fResp.arrayBuffer());
                    cells.set(tag, { vertices: cellVerts, facets: cellFacets });
                } catch (e) {
                    console.log(`  Cell surface for tag ${tag} unavailable:`, e);
                }
            }
            if (cells.size === 0) cells = null;
        }

        return {
            vertices,
            facets,
            tags,
            metadata,
            ecsVertices,
            ecsFacets,
            cutVertices,
            cutFacets,
            cells
        };
    }

    async loadCap(statusCallback) {
        const meshPath = this.getMeshPath();
        const report = statusCallback || (() => {});
        report('Loading cap geometry...');
        const vResp = await fetch(`${meshPath}/cap/vertices.bin?ts=${Date.now()}`);
        const fResp = await fetch(`${meshPath}/cap/facets.bin?ts=${Date.now()}`);
        if (!vResp.ok || !fResp.ok) {
            throw new Error('Cap geometry not available (configure a plane first)');
        }
        return {
            vertices: new Float32Array(await vResp.arrayBuffer()),
            facets: new Uint32Array(await fResp.arrayBuffer()),
        };
    }
}
