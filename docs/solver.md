# EMI Solver Internals

Zoom-in on the **EMI solver** and **FEniCSx + PETSc** blocks from [architecture.md](architecture.md).

```mermaid
%%{init: {"flowchart": {"subGraphTitleMargin": {"top": 0, "bottom": 0}, "rankSpacing": 40, "nodeSpacing": 30}, "themeVariables": {"lineColor": "#555", "textColor": "#1a1a1a", "clusterBkg": "#fafafa", "clusterBorder": "#888", "edgeLabelBackground": "#eef0f4"}, "themeCSS": ".edgeLabel, .edgeLabel * { color: #1a1a1a !important; fill: #1a1a1a !important; }"}}%%
graph TB
    subgraph REPO[" "]
        direction TB
        REPO_LABEL["EMI solver"]
        subgraph REPO_ROW[" "]
            direction LR
            DRIVER["main.py"]
            INPUT["utils.py"]
            IONIC["ionic_model.py"]
        end
    end

    subgraph DEPS[" "]
        direction TB
        subgraph DEPS_ROW[" "]
            direction LR
            MPX["multiphenicsx"]
            DOLFINX["dolfinx"]
            PETSC["PETSc"]
            UFL["ufl / ffcx"]
        end
        DEPS_LABEL["Dependencies"]
    end

    REPO_LABEL ~~~ DRIVER

    PETSC ~~~ DEPS_LABEL

    DRIVER --> INPUT
    DRIVER --> IONIC
    DRIVER -->|matrix assembly| MPX
    DRIVER -->|FEM| DOLFINX
    DRIVER -->|time step solve| PETSC

    INPUT -->|form compilation| UFL
    IONIC -->|form compilation| UFL

    classDef core fill:#ede0f5,color:#3d2645,stroke:#9b6dc7,stroke-width:1.5px,font-size:16px
    classDef backend fill:#dfe8f5,color:#1e3a5f,stroke:#5a8ec7,stroke-width:1.5px,font-size:16px
    classDef boxLabel fill:none,stroke:none,color:#222,font-size:20px

    class DRIVER,INPUT,IONIC core
    class MPX,DOLFINX,UFL,PETSC backend
    class REPO_LABEL,DEPS_LABEL boxLabel

    style REPO_ROW fill:none,stroke:none
    style DEPS_ROW fill:none,stroke:none
```
