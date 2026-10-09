# cardioEMI Architecture

The repository started as the EMI solver shown in [solver.md](solver.md). Everything outside the highlighted "Original EMI solver" group below was added around that core: mesh preparation tools, a Flask + Three.js viz layer, a GPU solver backend, and remote-execution glue.

```mermaid
%%{init: {"flowchart": {"rankSpacing": 45, "nodeSpacing": 45}, "themeVariables": {"lineColor": "#555", "textColor": "#1a1a1a", "clusterBkg": "#fafafa", "clusterBorder": "#888", "edgeLabelBackground": "#eef0f4"}, "themeCSS": ".edgeLabel, .edgeLabel * { color: #1a1a1a !important; fill: #1a1a1a !important; } .cluster-label, .cluster-label * { color: #1a1a1a !important; fill: #1a1a1a !important; }"}}%%
graph TB
    MESH[("Mesh & config<br/>(XDMF / pickle / YAML)")]

    subgraph ORIGINAL["Original EMI solver"]
        direction LR
        SOLVER["EMI solver"]
        FENICSX["FEniCSx + PETSc"]
        SOLVER --> FENICSX
    end

    GINKGO["Ginkgo backend"]
    VIZ["Visualization Webapp<br/>(Flask server + Three.js UI)"]
    KAROLINA["Karolina HPC<br/>(SLURM + Apptainer)"]

    MESH --> SOLVER
    SOLVER -.->|if available| GINKGO
    SOLVER -->|results| VIZ
    VIZ -.->|launch: subprocess| SOLVER
    VIZ -->|ssh / sbatch| KAROLINA
    KAROLINA -->|runs remotely| SOLVER

    classDef core fill:#ede0f5,color:#3d2645,stroke:#9b6dc7,stroke-width:1.5px,font-size:16px
    classDef backend fill:#dfe8f5,color:#1e3a5f,stroke:#5a8ec7,stroke-width:1.5px,font-size:16px

    class SOLVER core
    class FENICSX,MESH,GINKGO,VIZ,KAROLINA backend

    style ORIGINAL fill:#f5eef9,stroke:#9b6dc7,stroke-width:2px
```

## Layer summary

| Layer | Path | Role |
|---|---|---|
| **Original solver** | `main.py`, `utils.py`, `ionic_model.py`, `*_assembly.py`, `mesh_partition.py` | FEniCSx + multiphenicsx EMI assembly and time stepping (zoom in: [solver.md](solver.md)) |
| Mesh prep | `geometry/` | XDMF + tags + pickle generation, coloring, component partitioning |
| Ginkgo backend | `dolfinx-ginkgo/` | Optional GPU linear solver (CUDA / HIP / SYCL / OMP) |
| Flask bridge | `viz/server.py` | REST API: meshes, config, run simulation (subprocess + SSE), results, cross-sections, interfaces |
| Browser UI | `viz/index.html`, `viz/js/` | Three.js mesh viewer, simulation/solver/Karolina/results UI |
| Viz scripts | `viz/scripts/` | Post-process solver output → JSON for browser, export MP4 |
| Remote | `viz/karolina.py`, Apptainer SIF | SLURM job submission and monitoring on Karolina HPC (zoom in: [remote_execution.md](remote_execution.md)) |
