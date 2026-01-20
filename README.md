# cardioEMI
Solving cell-by-cell (a.k.a. EMI) models for cardiac geometries.

### Dependencies

* FEniCSx (www.fenicsproject.org)
* multiphenicsx (https://github.com/multiphenics/multiphenicsx)


### Download

```
git clone https://github.com/pietrobe/cardioEMI.git
cd cardioEMI
```

### Installation via Docker

To install FEniCSx:

```
docker run -t -v $(pwd):/home/fenics -i ghcr.io/fenics/dolfinx/dolfinx:v0.9.0
cd /home/fenics
```

To install dependencies:

```
pip install --no-build-isolation -r requirements.txt
```

### Running a Simulation

```
mpirun -n 8 python3 -u main.py input.yml
```
Modify `input.yml` for different input data. Parallel execution with `mpirun -n X`.

### Web Visualization Tool

A web-based interface for configuring and running simulations with 3D visualization.

**Start the server:**
```bash
cd viz
pip install flask h5py numpy scipy pyvista "imageio[ffmpeg]" Pillow
python server.py
```

**Open in browser:** http://localhost:8000

**Features:**
- 3D membrane mesh visualization (Three.js)
- Bounding box selection for excitation region
- Configurable simulation parameters (dt, time_steps, voltage)
- Live simulation output with progress bar
- Results visualization with time slider
- Video export of simulation results
- Mesh file browser and converter

### Mesh Conversion

Convert pts/elem format meshes to XDMF with optional graph coloring optimization:

```bash
python geometry/convert_pts_elem.py input.pts input.elem output.xdmf --colored
```

Graph coloring reduces the number of volume tags by grouping non-adjacent cells, significantly improving assembly performance (up to 38x speedup observed).

### Geometry and tagging
In the input .yml file two input files have to be provided:
- path to an XDMF mesh with volume and facets tags 
- path to a dictionary file containing the connectivity map between cells and facets

Each volume tag correspond to a FEM space, thus it makes sense to choose the minimum number of volume tags, so that there are no neighbour cells with the same tag. The ECS_TAG can be provided in the input .yml file, otherwise the minimum between all the volume tags will be used. 

The *geometry* directory contains scripts to generate tagged meshes and connectivity dictionaries. For example, the script `geometry/tag_facets.py` produces the needed input files given a volume-tagged cell.

An square input mesh can be created via

```
cd geometry
python3 create_square_mesh.py
```
in *create_square_mesh.py* geometric settings (#elements and #cells) can be modified.

### Experimental: Ginkgo Backend

An experimental GPU-accelerated solver backend using [Ginkgo](https://ginkgo-project.github.io/) is available in `dolfinx-ginkgo/`.

**Features:**
- Distributed matrix/vector conversion from PETSc to Ginkgo
- Krylov solvers: CG, FCG, GMRES, BiCGSTAB, CGS
- Preconditioners: Jacobi, Block Jacobi, ILU, IC, ISAI, AMG
- GPU backends: CUDA, HIP, SYCL, OpenMP

**Build the Docker image:**
```bash
cd dolfinx-ginkgo
docker build -t dolfinx-ginkgo:latest .
```

**Run tests:**
```bash
docker run --rm -v "$(pwd):/home/fenics/dolfinx-ginkgo" \
  -w /home/fenics/dolfinx-ginkgo/build dolfinx-ginkgo:latest \
  mpirun -n 2 ./tests/test_solver
```

See `docs/GINKGO_INTEGRATION_DESIGN.md` for the full design document.

###  Visualize output in Paraview
In Paraview `File > Load State...` of `output/bulk_state.pvsm`, selecting the correct path in *Load State Options*, to visualise the bulk potential evolution.

Similarly with `output/membrane_state.pvsm` to visualise ECS-ICS membrane potential jump (relying on ECS_TAG = 0).

### Contributors

* Pietro Benedusi
* Edoardo Centofanti
* Joshua Steyer
* Fritz Goebel

### Cite
```
@article{benedusi2024scalable,
  title={Scalable approximation and solvers for ionic electrodiffusion in cellular geometries},
  author={Benedusi, Pietro and Ellingsrud, Ada Johanne and Herlyng, Halvor and Rognes, Marie E},
  journal={SIAM Journal on Scientific Computing},
  volume={46},
  number={5},
  pages={B725--B751},
  year={2024},
  publisher={SIAM}
}

@article{benedusi2024modeling,
  title={Modeling excitable cells with the EMI equations: spectral analysis and iterative solution strategy},
  author={Benedusi, Pietro and Ferrari, Paola and Rognes, Marie E and Serra-Capizzano, Stefano},
  journal={Journal of Scientific Computing},
  volume={98},
  number={3},
  pages={58},
  year={2024},
  publisher={Springer}
}
```


