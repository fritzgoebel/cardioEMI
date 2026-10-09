# Remote execution on Karolina

Zoom-in on the **Karolina HPC** block from [architecture.md](architecture.md).
The webapp submits runs to the cluster, watches them in the SLURM queue and
pulls the results back — all of it plain `ssh` / `scp` calls to the `karolina`
host alias.

![Remote execution on Karolina](remote_execution.png)

```mermaid
%%{init: {"flowchart": {"rankSpacing": 50, "nodeSpacing": 40}, "themeVariables": {"lineColor": "#555", "textColor": "#1a1a1a", "clusterBkg": "#fafafa", "clusterBorder": "#888", "edgeLabelBackground": "#eef0f4"}, "themeCSS": ".edgeLabel, .edgeLabel * { color: #1a1a1a !important; fill: #1a1a1a !important; } .cluster-label, .cluster-label * { color: #1a1a1a !important; fill: #1a1a1a !important; }"}}%%
graph TB
    UI["Webapp"]
    SUBMIT["submit job"]
    POLL["poll jobs"]
    DL["download results"]

    subgraph REMOTE["Karolina"]
        QUEUE[("SLURM queue")]
        J1["job 1"]
        J2["job 2"]
        DOTS["…"]
        J3["job N"]
        APPT["Apptainer container"]
        RES[("results")]
    end

    UI --> SUBMIT
    UI --> POLL
    UI --> DL

    SUBMIT --> QUEUE
    POLL -.->|"status of all jobs at once"| QUEUE
    DL -.->|"per simulation"| RES

    QUEUE --> J1
    QUEUE --> J2
    QUEUE ~~~ DOTS
    QUEUE --> J3
    J1 --> APPT
    J2 --> APPT
    J3 --> APPT
    APPT --> RES

    classDef core fill:#ede0f5,color:#3d2645,stroke:#9b6dc7,stroke-width:1.5px,font-size:16px
    classDef backend fill:#dfe8f5,color:#1e3a5f,stroke:#5a8ec7,stroke-width:1.5px,font-size:16px

    class SUBMIT,POLL,DL core
    class UI,QUEUE,J1,J2,J3,APPT,RES backend

    style DOTS fill:none,stroke:none
```

**Submitting.** Each run gets its own remote directory holding its config and a
generated SLURM script, which is then `sbatch`ed. The script runs cardioEMI under
Apptainer via `srun`. A comma-separated node count (`1,2,4`) submits one job per
value in a single batch — the usual way to launch a scaling sweep.

**Polling.** A background thread checks every tracked job every 5 s with one
batched `squeue`/`sacct` call, so cost stays flat no matter how many jobs are
queued. The UI reads that cached state, and jobs drop out once they finish.

**Downloading.** Results come back as a single streamed `tar.gz` with byte
progress, or — for just the convergence plots — as the pickles alone.
