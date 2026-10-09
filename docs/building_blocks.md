# cardioEMI — Technical Report on the Supporting Building Blocks

This report documents the four subsystems that were built around the original
EMI solver: **mesh conversion & preprocessing**, the **dolfinx-ginkgo** GPU
solver backend, the **visualization web application**, and **remote job
execution** on the Karolina supercomputer. A short recap of the solver core is
included first, because the other three subsystems are shaped by its data model.

---

## 0. System context

cardioEMI solves the cell-by-cell (EMI — Extracellular–Membrane–Intracellular)
electro-diffusion model on cardiac micro-geometries using FEniCSx (DOLFINx) +
multiphenicsx. The defining structural invariant of the whole codebase is:

> **Each volume tag maps to its own finite-element space.** Membrane facets — the
> triangular faces shared by two cells with *different* tags — carry the
> coupling and ionic-model jump conditions between those spaces.

Everything downstream follows from this: the mesh tools exist to produce
correctly-tagged meshes plus a dictionary describing which membranes each tag
touches; the solver assembles one block per tag pair; the assembly backends and
BDDC preconditioner exploit the block/domain-decomposition structure; and the
viz layer reconstructs per-membrane voltages for rendering.

### End-to-end pipeline

```
 raw mesh (.pts/.elem or .vtu)
      │  geometry/  (convert, color, tag facets)
      ▼
 (mesh.xdmf + mesh.h5)  +  tags.pickle  +  input.yml
      │  main.py  (FEniCSx assembly + time stepping)
      │     ├─ PETSc backend (default)         ── KSP/PC, incl. PCBDDC via MATIS
      │     └─ Ginkgo backend (optional, GPU)  ── native COO / DdMatrix
      ▼
 output dir:  solution.xdmf, v.h5, v_i_j.h5, tags.xdmf,
              iterations.pickle, residuals.pickle, conditions.json, IF_*.txt
      │  viz/scripts  (post-process → web binaries)
      ▼
 viz/data/<sim>/  (mesh_vertices.bin, membrane_facets.bin, voltages/*.bin, …)
      │  viz/server.py (Flask) + Three.js UI
      ▼
 browser playback / MP4 export
```

Execution of `main.py` happens either **locally in Docker** (driven by the viz
server) or **remotely under SLURM + Apptainer** on Karolina (driven by
`viz/karolina.py`).

### The solver core in brief (so the rest makes sense)

`main.py` builds, for every tag `i`, a clone `V_i` of a global Lagrange space
restricted (via `multiphenicsx.fem.DofMapRestriction`) to the cells carrying tag
`i`. The bilinear form is a block system over tag pairs `(i, j)`:

- **Diagonal block** (`i == j`), with `τ = dt / C_M` and `σ = σ_e` for the ECS
  tag / `σ_i` otherwise:
  `a_ii = τ·(σ ∇u_i, ∇v_i)_Ω_i + (u_i⁻, v_i⁻)_Γ_ij`
- **Off-diagonal block** (`i ≠ j` sharing a membrane `Γ_ij`):
  `a_ij = −(u_j⁺, v_i⁻)_Γ_ij`

The membrane terms are interior-facet integrals (`dS`) using DG-style `'+'/'-'`
restrictions, which is *why* interior-facet integrals demand
`GhostMode.shared_facet` throughout. The right-hand side per timestep is built
from `fg = v_ij − τ·I_ion(v_ij)` integrated over each membrane, plus an optional
time-switched stimulus current. `I_ion` comes from the per-membrane ionic model
(`ionic_model.py`). This is a backward-Euler-in-time scheme with the ionic ODEs
integrated explicitly (Rush–Larsen) inside each step — a classic operator split.

After each solve the sub-components `u_i` are extracted with
`BlockVecSubVectorWrapper`, and the membrane potentials are recovered as
`v_ij = u_i − u_j`. With no Dirichlet BC the system is pure-Neumann (singular),
so a constant nullspace is created and removed from the RHS every step.

Key modules: `utils.py` (YAML parsing, `read_input_field`, progress/status),
`ionic_model.py` (ionic current models), `native_assembly.py` /
`matis_assembly.py` (alternative assembly targets), `mesh_partition.py`
(component partitioner).

---

## 1. Mesh conversion and preprocessing (`geometry/`, `mesh_partition.py`)

### 1.1 Purpose and inputs

The preprocessing tools convert raw tagged tetrahedral meshes into the exact
artifacts the solver requires:

1. an **XDMF/HDF5 mesh** carrying `cell_tags` (volume subdomain per tet) and
   `facet_tags` (membrane tag per triangular face), and
2. a **pickle dictionary** `membrane_tags: volume_tag → {membrane_tags}` giving,
   for each subdomain, the set of membrane facet tags on its boundary.

Two source formats are supported:

| Entry point | Source format | Notes |
|---|---|---|
| `geometry/convert_pts_elem.py` | Carp-style `.pts` (points) + `.elem` (tagged tets) | Primary path; supports graph coloring |
| `geometry/conversion.py` | synthetic-generator `.vtu` via `meshio` | Legacy path; equivalent output |

Supporting scripts: `tag_facets.py` / `geometry.py` (`get_facet_tags_and_dictionary`
for an already-loaded DOLFINx mesh), `reduce_mesh_tags.py`, `refine_mesh.py`,
`create_square_mesh.py` (test geometries), `compute_component_partitions.py`
(feeds the METIS partitioner). `scripts/rebuild_mesh_companions.py` regenerates
the `.pickle`/`.xdmf` companions for an existing `.h5`.

`generate_weak_scaling_mesh.py` is a third, synthetic source: an Nx×Ny×Nz
lattice of identical boxes of `ax·L × L × L`, one cell per box, so growing the
lattice grows the problem without changing anything per rank. See §1.6.

### 1.2 Tag remapping and graph coloring — the central design decision

A physiological mesh may contain hundreds of cell tags (one ECS chunk + one
intracellular tag per cell). Two problems arise from using all of them
directly:

- **One FEM space per tag** ⇒ hundreds of spaces, hundreds of matrix blocks —
  assembly becomes prohibitively expensive.
- **MPI communicator exhaustion** — DOLFINx allocates communicators per space;
  MPI caps out near the 2048-communicator limit.

`convert_pts_elem.py --color-intracellular` solves both by collapsing the tag
set to ~4 tags while *preserving the invariant that neighbouring cells never
share a tag*:

1. All **even** tags → `0` (extracellular space, ECS).
2. **Odd** (intracellular) tags → greedy **graph coloring**: build the
   subdomain adjacency graph from shared tet faces
   (`build_subdomain_adjacency`), restrict to intracellular–intracellular
   edges, and assign each cell the smallest color not used by a neighbour
   (`graph_coloring`). Colors start at 1 (0 is reserved for ECS).

Typically ~3 colors suffice, so a mesh with 44 tags collapses to 4. A
`<prefix>_tag_mapping.pickle` records the original→colored map so the true
component identity can be recovered later (needed by component partitioning,
§1.5).

### 1.3 Facet extraction and membrane tag encoding

`extract_facets_and_membrane_dict` hashes each tetrahedron's four triangular
faces (`face → list of (cell_id, cell_tag)`):

- A face touched by **one** cell (or two cells with the **same** tag) is an
  interior/boundary facet ⇒ default tag `−5` (ignored by the solver).
- A face shared by **two differently-tagged** cells is a **membrane facet**, tagged
  with an encoding `min(t1,t2)·(N_TAGS+1) + max(t1,t2)`, and both tags record
  this membrane tag in `membrane_dict`.

The encoding makes the membrane tag a reversible function of the unordered tag
pair, which the solver inverts (`facet_tag_to_pair` in `main.py`) to know which
two spaces a membrane couples.

### 1.4 Output format

`write_xdmf_h5` writes an HDF5 file with `Mesh/mesh/{geometry,topology}`,
`Mesh/facet_tags/{topology,Values}`, `Mesh/cell_tags/{topology,Values}`, and an
XDMF wrapper with three grids (`mesh`, `facet_tags`, `cell_tags`) that share
geometry via `xi:include`. The membrane dictionary is pickled separately. The
solver reads `ECS_TAG = min(volume tags)` unless overridden.

### 1.5 Component-based partitioning (`mesh_partition.py`)

Graph coloring destroys the correspondence between a cell and *its* surrounding
ECS chunk (both may end up on different ranks under default partitioning),
causing heavy ghosting across the membrane where the physics is tightest.
`partition_mode: component` restores locality:

1. **Recover components from the original mesh.** The uncolored mesh's pickle
   gives per-tag interface sets. Components are consecutive tag pairs
   `(2c, 2c+1)` = (ECS chunk, cell). `compute_component_partitions` builds a
   component adjacency graph from shared interfaces and partitions it with
   **pymetis** (`part_graph`), falling back to round-robin if pymetis is
   unavailable.
2. **Map colored cells to components.** Because coloring changed the tags, a
   **KD-tree** (`scipy.spatial.cKDTree`) matches each colored-mesh cell centroid
   to the nearest original-mesh centroid, inheriting its component → partition
   assignment.
3. **Custom DOLFINx partitioner.** A multi-destination `AdjacencyList_int32` is
   built where, for each cell, the **first destination is the owner** and the
   **remaining destinations are ghost ranks**, derived from cell→facet→cell
   connectivity (a cell ghosts to any rank owning a facet-neighbour). This
   satisfies the `shared_facet` requirement. The partitioner is invoked with
   local cells only (rank 0 holds all cells initially).
4. **Reconstruct facet tags after redistribution.** Vertex global indices are
   *not* reliable for ghost vertices after redistribution, so tags are
   recovered via **cell-pair matching**: a `(sorted original_cell_index pair) →
   membrane_tag` map is built before redistribution and looked up afterward
   using `topology.original_cell_index` (which DOLFINx maintains reliably for
   local *and* ghost cells) plus facet→cell connectivity.

This reduces ghosting dramatically (~10k vs ~600k ghosted cells versus
round-robin on the test meshes).

### 1.6 Synthetic weak-scaling lattices (`geometry/generate_weak_scaling_mesh.py`, `geometry/cell_box_gmsh.py`)

Weak scaling needs a mesh family where the *per-rank* problem is literally
identical as the domain grows. The generator produces one: an Nx×Ny×Nz tiling of
boxes of `ax·L × L × L`, one cell per box. Exactly **one box is meshed**, then
stamped over the lattice and welded, so cost is dominated by the tet/facet
bookkeeping rather than the geometry. Cells are 2-colored on the 3D
checkerboard, giving three volume tags: ECS → 0, ICS → 1 or 2. `--n` is elements
per L for both shapes.

Two `--shape`s:

- **`plus`** (`--ax 1`) — the original geometry on a structured voxel grid:
  three orthogonal `L/2 × L/2` bars through the box centre, each voxel
  Kuhn/Freudenthal-split into 6 tets. `3w²L − 2w³ = L³/2` has the clean root
  `w = L/2`, so ICS and ECS volumes are *exactly* equal.
- **`cell`** (default, `--ax 4`) — **unstructured tetrahedra built with gmsh +
  OpenCASCADE** (`cell_box_gmsh.mesh_box`): a **body** cylinder along the x axis
  spanning the whole box, so both x-face cross-sections are full-radius discs
  and cells butt together end to end the way myocytes meet at an intercalated
  disc; plus two **connectors** that cross the whole box, one from `y=0` to
  `y=L` and one from `z=0` to `z=L`, each an exact circular disc extruded along
  an *oblique* vector. Extruding a disc rather than sweeping a cylinder along a
  tilted axis is what keeps every face-parallel cross-section a true circle — the
  interface stays round while the connector leans in x (the lean is reported in
  degrees off the face normal).

#### The lattice is shifted, not mirrored

Every cell must be a pure **translate** of every other one, or the connectors
alternate direction and the tissue has no consistent fibre direction. (An earlier
variant mirrored odd-parity boxes in x. It tiled perfectly and the checkerboard
parity gave the mirror for free — but it flipped the lean on every second cell,
which is exactly wrong for a fibre-aligned tissue.)

Pure translates force the lattice to be **shifted**: a connector entering the
`y = 0` face at `lat_x` leaves the `y = L` face at `lat_x + s`, so the neighbour
above has to sit `s` further along x.

Taken literally that gives a diagonal lattice, `a2 = (s, L, 0)`, whose block runs
off by `(ny + nz − 2)·s` in x — a lot of empty wedge. So the offset is taken
**modulo the box**:

```
box (I,J,K)  at  ( I*Lx + ((J+K) mod slabs)*s,  J*L,  K*L )
```

Rows then **zig-zag** between `slabs` x-positions and the block overhangs by only
`(slabs − 1)*s`. A cell that would have walked past the end of the block folds
back and sits in the padding instead. For a 4×4×4 block at the defaults that is
50 µm of overhang rather than 300 µm — 11% of the block's length instead of 43%.

`--slabs` also sets where a connector's two interfaces sit, since they are `s`
apart: with `slabs = 2`, `s = Lx/2` puts them at `Lx/4` and `3Lx/4`, which is as
far towards the cell's two ends as a single straight connector can reach (the
entry must clear the near face by its own radius, and `s` cannot exceed `Lx/2`).
Larger `slabs` shortens the shift and pulls both interfaces back towards the
small-x end.

#### The wrap forces a different 2-colouring

Wrapping changes who a connector reaches. When `(J+K) mod slabs` rolls over, the
y connector of box `(I,J,K)` lands in box `(I+1,J+1,K)` rather than `(I,J+1,K)` —
and those two have the **same** `(I+J+K)` parity, so the plain checkerboard would
hand two touching cells the same tag and violate the one-space-per-tag invariant.

Colouring by the *unwrapped* index fixes it:

```
colour = (I - (J+K)//slabs + J + K) % 2
```

that being the index the cell would have carried on the diagonal lattice, where
the checkerboard is correct. The direct test is connected components of the ICS
tetrahedra: there must be exactly `nx*ny*nz` of them. Fewer means two touching
cells fused because they shared a tag; more means a cell was split.

#### How the shifted faces still mate

The rectangular box remains a fundamental domain of that lattice, but its `+y`
face is the `−y` face translated by `(s, L, 0)`, which runs off the end in x — no
single affine map takes one onto the other. Splitting the box into
`m = Lx / s` equal slabs along x (`--slabs`, default `ax`, giving `s = L` and a
45° lean) makes the correspondence exact slab by slab: slab `k` of the `−y` face
maps to slab `k+1` of the `+y` face, and the last slab wraps to the first under
an extra `−a1`. gmsh therefore gets `m` **pure translations** per face pair
instead of one impossible constraint:

```
+x <- -x   translate (Lx, 0, 0)
+y <- -y   translate (s, L, 0), less (Lx,0,0) for the last slab
+z <- -z   translate (s, 0, L), likewise
```

Face pieces are paired by centre of mass **and area** — on an x face the membrane
disc and the ECS annulus around it share a centre of mass, so position alone is
ambiguous. A connector disc has to sit inside a single slab, or its face piece
straddles a slab boundary and has no image one slab further along; `check_shape`
rejects that with the offending distance.

#### Verifying the tiling

Tiling only has to weld coincident nodes, and `_verify_conforming` checks the
welded boundary area against `free_surface()`. Getting that expectation right is
the whole value of the check: because of the shift, a box's `+y` face is shared
with **two** boxes above, and whatever they leave uncovered stays exposed — those
leftovers are exactly the zig-zag steps. `free_surface` therefore sums box by box
from the actual x overlaps rather than from a cuboid formula, which under-predicts
and would mask a genuine welding failure. (It fired on the first shifted run; the
discrepancy turned out to be the staircase, not a bad weld.) Measured against the
analytic prediction the resulting gap-junction area lands within ~1%, the
chord-vs-arc error of the surface mesh.

Two OpenCASCADE traps are worth knowing, both of which fail *silently* or
obscurely:

1. A multi-tool `occ.fuse` whose tools overlap **each other** returns just the
   object — the connectors vanish with no error and the ICS comes out exactly the
   body's volume. Fuse one tool at a time.
2. When two solids have to be exact images of one another for a periodic
   constraint, build the second as an OCC transform of the first rather than from
   transformed coordinates: the constraint matches the circles' *seam* points
   too, and an independently built disc puts its seam on the wrong side.

#### Volume fraction and padding

The defaults land near 50% ICS. That matters beyond aesthetics: under
`partition_mode: cube` each cell is one rank and the ECS left in its box is
another, so **the ICS volume fraction is that load balance** — the generator
prints it, computed analytically from `cell_contains` before any meshing. The
other printed numbers to watch are the ECS channel width at the box face centres
and the connector width, both in elements; the generator warns when either falls
below two.

`--pad` wraps the block in pure ECS so no membrane sits on the raw domain
boundary. It counts **element layers of `L/n`** for both shapes; for `plus` those
are voxels, and for `cell` a genuine **boundary layer**: every boundary node is
offset along its area-weighted outward normal, and the shell between the original
and offset triangulations is filled with prisms, three tets each. Two properties
make that the right construction:

* it shares the block's boundary triangulation exactly, so it is conforming by
  construction — no welding and no coincident-node bookkeeping;
* it is built from the *surface* rather than from six planar slabs, so there are
  no gaps or overlaps at edges and corners. That matters because the zig-zag
  block is not convex: extruding face patches along their own axis would run the
  x pad of one row straight into the row above it.

The cost scales with surface area, not volume — about +37% elements for two
layers on a 2×2×2 block, against 27× for the whole-box padding this replaces.

One trap worth recording. The obvious offset for an axis-aligned surface is a
*square mitre*: move a node by `step` along each distinct face normal it sees, so
face nodes move `step·e`, convex-edge nodes `step·(e1+e2)`, corner nodes
`step·(e1+e2+e3)`. It grows the bounding box exactly and looks perfect — and it
collapses. A convex-corner node moves `step` *tangentially*, which is precisely
the node spacing, so it lands exactly on the offset of its neighbour one element
along the edge and every prism between them has zero volume. Area-weighted
averaged normals of fixed length `step` keep the direction field smooth across
the edge, so consecutive nodes cannot cross; the layer is thinner at edges by a
cosine, which is only cosmetic, and the bounding box still grows by exactly
`pad·L/n` because the extremes are on the flat faces. `pad_boundary_layers`
checks every resulting tet has positive volume and says what to do if not.

`--dry-run` prints the shape statistics and ASCII mid-plane slices of the
analytic solid without meshing or writing.

Mesh names are canonical and machine-parsed by both `mesh_partition.py` and
`viz/server.py`:
`<plus|cell>_<nx>x<ny>x<nz>_n<n>_L<L>[_p<pad>][_a<ax>][_s<slabs>]` (`.` → `p` in `L`;
optional suffixes omitted at their defaults, so legacy `plus_…` names still
parse). `partition_mode: cube` recovers `(shape, nx, ny, nz, n, L, pad, ax)` from
it and bins every tet centroid into its box. `shape` tells it whether `pad` is
boxes or voxels; for the shifted `cell` lattice it bins y and z first and
subtracts `(J+K)·s` before binning x. Verify with
`mpirun -n <2·boxes> python3 test_cube_partition.py <mesh.xdmf>`.

---

## 2. dolfinx-ginkgo — GPU/accelerated solver backend (`dolfinx-ginkgo/`)

### 2.1 Motivation and integration model

`dolfinx-ginkgo` is an **optional** alternative to the default PETSc linear
solve, wrapping the [Ginkgo](https://ginkgo-project.github.io/) library to
provide distributed, GPU-capable Krylov solvers and preconditioners (CUDA / HIP
/ SYCL / OpenMP / reference backends). It is selected per-run via
`solver_backend: ginkgo` in the YAML.

`main.py` imports it opportunistically through several candidate paths (dev
Docker `build/_cpp*.so`, then the Apptainer system install
`/usr/local/dolfinx_ginkgo/`). **If the import fails, the run silently falls
back to PETSc** — a deliberate robustness choice, but a known footgun: a broken
Ginkgo build produces PETSc numbers without warning, so new kernels must be
verified as *loadable* (not merely compiled) before trusting results.

### 2.2 Layered architecture

```
dolfinx-ginkgo/
├── cpp/dolfinx_ginkgo/            (header-only C++ core)
│   ├── ginkgo.h            Backend enum, executor/communicator factories,
│   │                       SolverConfig / AMGConfig / BDDCConfig structs
│   ├── convert.h           PETSc MPIAIJ → CSR extraction (per rank)
│   ├── Partition.h         DOLFINx IndexMap → Ginkgo distributed partition
│   ├── DistributedMatrix.h PETSc Mat / local-COO → gko distributed::Matrix / DdMatrix
│   ├── DistributedVector.h PETSc Vec ↔ gko distributed::Vector
│   └── Solver.h            DistributedSolver: Krylov solvers + preconditioners
├── python/dolfinx_ginkgo/
│   ├── _cpp.cpp            nanobind bindings  →  _cpp.*.so
│   └── solver.py           GinkgoSolver high-level wrapper
├── tests/                  C++ + Python unit/integration tests
├── examples/poisson.cpp
├── Dockerfile / Dockerfile.bddc / apptainer-bddc.def   (build recipes)
```

### 2.3 Matrix ingestion — three paths tied to three assembly routes

The interesting part of the integration is *how the assembled operator reaches
Ginkgo*, and this is where `native_assembly.py` and `matis_assembly.py` come in.

| Path | Assembly source | Ginkgo target | Used for |
|---|---|---|---|
| `set_operator_from_petsc` | `multiphenicsx.fem.petsc.assemble_matrix_block` | `distributed::Matrix` from a PETSc `Mat` | Ginkgo without native assembly |
| `set_operator_from_local_coo` | `native_assembly.assemble_block_to_coo` | `distributed::Matrix` summed from per-rank COO | `ginkgo.native_assembly: true` |
| `set_operator_dd_from_local_coo` | same COO | `DdMatrix` (unassembled, R/P operators) | BDDC (`ginkgo.dd_matrix: true`) |

**`native_assembly.py`** assembles the DOLFINx block form directly to COO
triples in *global* indices, deliberately bypassing PETSc. It uses
`dfx.fem.assemble_matrix` which returns a `MatrixCSR` with **un-accumulated
ghost values** — precisely what Ginkgo's `assembly_mode::communicate` expects
(Ginkgo sums cross-rank contributions to the same `(row,col)`). It handles the
restricted DOF numbering (`unrestricted_to_restricted` maps), reconstructs the
rank-major global block layout PETSc would have produced, and applies Dirichlet
BCs **symmetrically** (zeroing both the row and the column, unit diagonal) to
preserve symmetry for CG/BDDC. It also emits a `matrix_row → mesh_vertex` map
and the set of vertices this rank actually contributes to, used later for
partition visualization.

**`matis_assembly.py`** reuses those exact COO triples to build a PETSc `MATIS`
matrix for the *PETSc* `PCBDDC` preconditioner (the non-Ginkgo BDDC path). Each
rank's local `SeqAIJ` contains its owned rows plus only the ghost rows it
actually contributes to (via an `ISLocalToGlobalMapping`), matching the Ginkgo
`DdMatrix` scope and avoiding the oversized Neumann sub-problems that
multiphenicsx's restriction index map would otherwise induce.

### 2.4 Solver and preconditioner menu

Declared in `ginkgo.h` and surfaced through `GinkgoSolver`:

- **Krylov solvers**: CG, FCG, GMRES (configurable `krylov_dim`), BiCGSTAB, CGS.
- **Preconditioners**: none, point Jacobi, block Jacobi (`jacobi_block_size`),
  ILU, IC, ISAI, **AMG** (Ginkgo PGM coarsening; V/W/F cycles; Jacobi/GS/ILU
  smoothers; direct/CG/GMRES coarse solve; optional mixed precision), and
  **BDDC** (requires the `DdMatrix` path).
- Convergence via `rtol`/`atol`/`max_iterations`; optional per-iteration true
  residual capture (`track_iter_residuals`, via `gko::log::Record`); pure-Neumann
  constant-nullspace handling.

### 2.5 BDDC — the deep configuration surface

BDDC (Balancing Domain Decomposition by Constraints) is the focus of much of the
recent work and has by far the richest config (`BDDCConfig`), mirrored between
`main.py`, `solver.py`, and the viz server:

- **Primal constraints**: `vertices` / `edges` / `faces`.
- **Interface scaling**: `stiffness` (PCIS partition-of-unity) or `deluxe`
  (sub-Schur complements).
- **Local subdomain solver** (`A_LL`): direct / direct_lu / ILU / IC / AMG /
  Hypre BoomerAMG, with a full BoomerAMG parameter block
  (`local_hypre`: cycle/coarsening/strength-threshold/smoother/sweeps/…).
- **Inner (interior `A_II`) solver**: independently tunable
  (`inner_solver` + `inner_amg` / `inner_hypre`); when unset it reuses the local
  solver. The distinction matters — the Ginkgo-vs-PETSc BDDC iteration gap
  documented in the project memory was traced to how the **inexact interior
  solve** is handled.
- **Coarse solver**: CG / GMRES / nested BDDC / additive Schwarz (MUMPS local),
  with `repartition_coarse` for load balance.
- **Nullspace**: `constant_nullspace` per rank (set true where a rank has no
  Dirichlet BC) and `coarse_constant_nullspace`; `main.py` computes these from
  the actual per-rank Dirichlet situation (`rank_has_dirichlet`, reduced with
  `MPI.MIN`).
- Optional fill-reducing `reordering: amd` on the local matrices.

A representative `ginkgo` YAML block (native DdMatrix + BDDC + Schwarz coarse +
HMIS local AMG) is shown in `exp_p8_gko_dL_hI_cg.yml`.

For comparison, the **PETSc PCBDDC** path (`pc_type: bddc`, no Ginkgo) is
configured directly through PETSc options in `main.py`
(`pc_bddc_use_vertices/edges/faces`, deluxe vs stiffness scaling, approximate
Dirichlet/Neumann sub-PCs with the inexact-solver nullspace correction, redundant
coarse solve), fed by the same MATIS matrix from `matis_assembly.py`.

### 2.6 Debugging aids

`GinkgoSolver` exposes `apply_preconditioner_local`, `apply_operator_local`, and
`solve_local` (numpy in/out) for inspecting operator/preconditioner spectra and
symmetry. `main.py` supports `CARDIOEMI_DUMP_COO` / `CARDIOEMI_DUMP_RHS`
environment variables to dump each rank's exact COO operator and first-timestep
RHS for offline BDDC bisection.

---

## 3. Visualization web application (`viz/`)

A Flask + Three.js application that is the human-facing control panel for the
whole pipeline: browse meshes, configure a run, launch it (local or remote),
then play back and export results.

### 3.1 Backend — `viz/server.py` (Flask REST + SSE)

Route groups:

- **Config** (`/api/config…`): reads/writes the YAML input files. Simple field
  updates are done **line-by-line to preserve formatting**; nested blocks
  (`ginkgo`, `petsc_bddc`, scar) are round-tripped through PyYAML. A dedicated
  builder (`_build_scar_expressions`) generates **scar-tissue conductivity**
  fields as UFL expression strings — nested `ufl.conditional(ufl.And(ufl.ge(...),
  …))` chains defining dense-core / border-ring / healthy conductivities per
  bounding box. (It deliberately emits `ufl.And/ge/le` rather than Python
  comparison products, because the strings are compiled by ffcx into variational
  forms.)
- **Meshes** (`/api/meshes…`): list `data/*.h5`, convert to the web binary
  format with SSE progress, select/track the current mesh, auto-derive a config.
- **Simulation** (`/api/simulation/run`): launches the solver in a **Docker
  container** — image chosen by `solver_backend` (`ghcr.io/fenics/dolfinx:v0.9.0`
  or `dolfinx-ginkgo:bddc`, the latter (re)building the bindings if stale) — via
  `mpirun -n <ranks>`, streaming stdout over Server-Sent Events. Sentinel lines
  emitted by the solver (`PROGRESS:pct:msg`, `ITERATIONS:step:count`,
  `RESIDUAL:step:abs:rel`) are parsed into structured SSE events for live charts.
  Post-run, per-solve `IF_*.txt` BDDC interface files are moved into the sim dir.
- **Results / simulations** (`/api/results`, `/api/simulations…`): enumerate
  `*_sim*` output dirs, read `v.h5` or per-membrane `v_i_j.h5`, map voltages onto
  membrane facets, cache float32 voltage binaries per timestep, and serve
  iterations/residuals/rank-partition metadata. Also bulk-delete (local + viz
  cache + remote).
- **Viz generation** (`/api/generate-viz`): runs the post-processing script in a
  worker thread with SSE progress.

### 3.2 Frontend — `viz/index.html` + `viz/js/`

An `App` prototype whose methods are split across mixin files
(`app-mesh`, `app-ui`, `app-scar`, `app-solver`, `app-karolina`,
`app-karolina-jobs`, `app-simulation`, `app-partition`, `app-results`,
`app-charts`, `app-video`) plus helper classes (`MeshLoader`, `Viewer`,
`ConfigManager`, `SimulationRunner`, `KarolinaRunner`). **Three.js** (r128)
renders the mesh/membrane surface and animates voltage playback; **Chart.js**
draws iteration/residual/voltage-time plots. The UI lets the user set the
stimulus bounding box, excited/resting initial voltages, scar regions, solver
options (including the full BDDC surface), MPI ranks, and the local-vs-Karolina
run target.

### 3.3 Post-processing pipeline (`viz/scripts/`)

- `convert_hdf5.py` — mesh `.h5` → web binaries (`mesh_vertices.bin`,
  `membrane_facets.bin`, `membrane_tags.bin`, `mesh_metadata.json`).
- `generate_viz_from_output.py` — the key post-processor. It rebuilds each
  cell's **closed boundary surface** from the tets (faces appearing exactly once
  are boundary faces) as an *expanded* mesh (every triangle gets its own three
  vertices) plus an orig-vertex map, so per-vertex `φ_i` from `v_i_j.h5` can be
  looked up correctly — this is what makes cell–cell junction disks (where
  multiple membranes meet) render properly. Emits per-timestep voltage binaries.
- `video_exporter.py` / `remote_video_pipeline.py` — PyVista + imageio MP4
  export (used both locally and on Karolina).

Checked-in `viz/data/<mesh>/` caches and `viz/videos/*.mp4` are the outputs of
these stages.

---

## 4. Remote job execution (`viz/karolina.py`, `scripts/`)

The glue for running simulations on the **Karolina** supercomputer under SLURM +
Apptainer, driven from the same web UI. There is no dedicated API — everything
goes through **SSH/SCP against the `karolina` ssh-config alias**.

### 4.1 Transport and containers

`_run_ssh` / `_apptainer_exec` (with `_shell_quote` for injection safety) wrap
SSH; `check_ssh` / `check_containers` verify reachability and the two SIF images
(`dolfinx-v0.9.0.sif`, `dolfinx-ginkgo-bddc.sif`) under `containers/`. All paths
are rooted at `REMOTE_PATH = /scratch/project/eu-26-11/fritz/cardioEMI`.

### 4.2 Remote mesh operations

- `list_remote_meshes` — one SSH call finds `.pts` families and already-converted
  `.h5` files, grouping by family and flagging converted / converted_colored,
  including h5-only meshes with no `.pts` source.
- `fetch_mesh_metadata` — runs a tiny h5py script **inside the container** to
  read bounds / conversion factor / vertex & facet counts / unique tags from the
  H5 header *without downloading the mesh*.
- `convert_remote_mesh` — streams `geometry/convert_pts_elem.py` inside the
  DOLFINx container (installing `lxml`/`h5py` to a writable target since the
  container FS is read-only).
- `download_mesh_data` — SCPs the converted `.h5`/`.xdmf`/`.pickle` back.

### 4.3 SLURM script generation and NUMA-aware pinning

`generate_slurm_script` builds the batch script with the hardware-specific
tuning that makes cardioEMI memory-bandwidth-efficient on Karolina's dual AMD
EPYC 7H12 nodes:

- `FI_PROVIDER=tcp` (the container's libfabric lacks Karolina's native OFI
  provider), `OMP_NUM_THREADS=1`.
- `srun --cpu-bind=cores --distribution=block:block` with `cpus_per_task =
  128 / ntasks_per_node`, pinning ranks across the 8 NUMA domains / node
  (optimal memory-bound config ≈ 16 ranks/node = 1 rank per memory channel).
- Runs `main.py` inside `apptainer exec --bind … --pwd /home/fenics <sif>`,
  tee-ing output to a per-job log; post-run it moves `IF_*.txt` into the sim dir.

The container image is selected by `solver_backend` (DOLFINx vs Ginkgo SIF).

### 4.4 Job submission and monitoring

- `submit_job` — creates a per-job remote directory, rewrites the config's
  `out_name` to a timestamped value, uploads config + script + `conditions.json`,
  `sbatch`es it, parses the job id.
- `submit_jobs` — a **scaling sweep**: prepares one job dir per node count in a
  local temp dir, uploads them all in a single `scp -r`, and submits them all in
  a single batched SSH `sbatch` call (with `@@JOB name@@` markers to correlate ids).
- **Background poller** — a daemon thread (`_poll_active_jobs`, 5 s) batches
  `squeue`/`sacct` status and log tails for **all** active jobs into single SSH
  calls, caching `(status, log)` per job so HTTP endpoints stay cheap; polling
  stops at terminal states. `cancel_job`, `delete_job_data` (with `out_name`
  whitelisted before any remote `rm`), and `tail_remote_log` round it out.

### 4.5 Results and remote post-processing

- `download_results_streaming` / `download_viz_data_streaming` — create a remote
  `tar.gz` and stream+extract it with progress, far faster than per-file SCP.
- `download_iterations` — pulls just `iterations.pickle` / `residuals.pickle` /
  `conditions.json` for cheap convergence comparison across runs.
- `generate_remote_viz` / `generate_remote_video` — submit single-node SLURM jobs
  to build viz data / MP4s on the cluster, with `check_viz_job` / `check_video_job`
  parsing `PROGRESS:` / `VIDEO_FILE:` / `ERROR:` log sentinels;
  `download_video` retrieves the result.
- `list_remote_simulations` — a remote Python one-liner walks `*_sim*` dirs and
  emits a single JSON array (with a bare-listing fallback).

### 4.6 Shell helpers and documented build pitfalls

`scripts/setup_karolina.sh` (one-time remote setup) and
`scripts/sync_to_karolina.sh` (rsync project files). The Apptainer build has
several documented gotchas (unset `SINGULARITY_BINDPATH`/`LD_PRELOAD` before
apptainer; `--contain` to avoid host `/tmp` overlay; `TMPDIR` on a large fs for
Ginkgo `-j`; unset host `CC/CXX/BOOST_ROOT` inside the container; manual
`libGKlib.so` symlink; ParMETIS built from source against the container's
`mpicc`) captured in `CLAUDE.md`, the project memory, and `docs/`.

---

## 5. Cross-cutting artifacts and conventions

| Artifact | Produced by | Consumed by |
|---|---|---|
| `mesh.xdmf` + `mesh.h5` | `geometry/` | `main.py`, viz mesh conversion |
| `tags.pickle` (`membrane_tags`) | `geometry/` | `main.py` (spaces, blocks, facet↔pair map) |
| `<mesh>_tag_mapping.pickle` | coloring | `mesh_partition.py` |
| `input.yml` | user / viz server | `main.py` (`utils.read_input_file`) |
| COO triples | `native_assembly.py` | Ginkgo (`distributed`/`Dd`), `matis_assembly.py` |
| `v.h5`, `v_i_j.h5`, `solution.xdmf`, `tags.xdmf` | `main.py` | viz post-processing |
| `iterations.pickle`, `residuals.pickle` | `main.py` | viz charts, convergence comparison |
| `conditions.json` | viz server / karolina | run labelling, dedup hashing |
| `dof_ranks.pickle`, `matrix_to_vertex.pickle`, `IF_*.txt` | `main.py` | partition visualization, BDDC analysis |
| web binaries (`*.bin`, `mesh_metadata.json`) | `viz/scripts/` | Three.js frontend |

Two recurring conventions worth noting: the **`PROGRESS:`/`ITERATIONS:`/
`RESIDUAL:`/`VIDEO_FILE:` stdout sentinel protocol** is the single mechanism by
which long-running solver/video processes report structured progress to the SSE
layer (local *and* remote); and **`conditions.json` + its hash** is how the viz
layer labels, deduplicates, and compares runs.
```
