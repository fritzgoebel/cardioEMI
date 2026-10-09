# dolfinx-ginkgo — C++ core layers

Zoom-in on the header-only C++ core (`dolfinx-ginkgo/cpp/dolfinx_ginkgo/`) from
the dolfinx-ginkgo section of [building_blocks.md](building_blocks.md). It turns
the DOLFINx/PETSc system into Ginkgo's distributed objects and runs the solve on
the chosen device, organised into three layers.

![C++ core layers](ginkgo_cpp_core.png)

```mermaid
%%{init: {"flowchart": {"rankSpacing": 55, "nodeSpacing": 45, "wrappingWidth": 400}, "themeVariables": {"lineColor": "#555", "textColor": "#1a1a1a", "clusterBkg": "#fafafa", "clusterBorder": "#888", "edgeLabelBackground": "#eef0f4"}, "themeCSS": ".edgeLabel, .edgeLabel * { color: #1a1a1a !important; fill: #1a1a1a !important; }"}}%%
graph TB
    CONFIG["<b>Configuration layer</b><br/>• Create Ginkgo executor<br/>• Wrap MPI communicator<br/>• Read solver configuration"]
    CONVERT["<b>Conversion layer</b><br/>• Matrix assembly<br/>• Map DOLFINx partition<br/>• Move vectors in / out"]
    SOLVE["<b>Solver layer</b><br/>• Generate solver + preconditioner<br/>• Apply solver each time step"]

    CONFIG -->|executor + communicator| CONVERT
    CONFIG -->|executor + config| SOLVE
    CONVERT -->|matrix + RHS| SOLVE
    SOLVE -->|solution| CONVERT

    classDef core fill:#ede0f5,color:#3d2645,stroke:#9b6dc7,stroke-width:1.5px,font-size:16px
    classDef backend fill:#dfe8f5,color:#1e3a5f,stroke:#5a8ec7,stroke-width:1.5px,font-size:16px

    class SOLVE core
    class CONFIG,CONVERT backend
```

## How they interact

The **Configuration** layer hands its executor and communicator to the
**Conversion** layer (so matrices and vectors are built on the right device) and
its executor + config to the **Solver** layer. The **Conversion** layer feeds the
assembled operator and RHS into the **Solver**, which runs the iteration and
passes the solution back out through Conversion to PETSc. The Python
`GinkgoSolver` sits above all three, selecting options and shuttling the system
in and the answer out.
