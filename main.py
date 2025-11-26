import ufl
import time
import pickle
import multiphenicsx.fem
import multiphenicsx.fem.petsc
import dolfinx  as dfx
import numpy as np
#import matplotlib.pyplot as plt
from ufl      import inner, grad
from sys      import argv, stdout
from mpi4py   import MPI
from pathlib  import Path
from petsc4py import PETSc
from utils             import *
from ionic_model       import *

# Options for the fenicsx form compiler optimization
cache_dir       = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast","-march=native"]
jit_parameters  = {"cffi_extra_compile_args"  : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

#----------------------------------------#
#     PARAMETERS AND SOLVER SETTINGS     #
#----------------------------------------#

# Timers
solve_time    = 0
assemble_time = 0
ODEs_time     = 0

start_time = time.perf_counter()

# MPI communicator
comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("\n#-----------SETUP----------#")
    print("Processing input file:", argv[1])

# Read input file
params = read_input_file(argv[1])

# aliases
mesh_file = params["mesh_file"]
ECS_TAG   = params["ECS_TAG"]
dt        = params["dt"]

#-----------------------#
#          MESH         #
#-----------------------#

if comm.rank == 0: print("Input mesh file:", mesh_file)

with open(params["tags_dictionary_file"], "rb") as f:
    membrane_tags = pickle.load(f)

# set tags info
TAGS   = sorted(membrane_tags.keys())
N_TAGS = len(TAGS)

# Read mesh
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
    # Read mesh and cell tags
    mesh       = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.shared_facet)
    subdomains = xdmf.read_meshtags(mesh, name="cell_tags")

    # Create facet-to-cell connectivity
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    # Also the identity is needed
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    boundaries = xdmf.read_meshtags(mesh, name="facet_tags")
    
# Scale mesh
mesh.geometry.x[:] *= params["mesh_conversion_factor"]

# timers
if comm.rank == 0: print(f"Reading input time:     {time.perf_counter() - start_time:.2f} seconds")
t1 = time.perf_counter()

# Define integral measures
dx = ufl.Measure("dx", subdomain_data=subdomains) # Cell integrals
dS = ufl.Measure("dS", subdomain_data=boundaries) # Facet integrals

# Read physical constants
sigma_i = read_input_field(params['sigma_i'], mesh=mesh)
sigma_e = read_input_field(params['sigma_e'], mesh=mesh)
tau     = dt/params["C_M"]

# add electrode conductivity if present
using_electrode = ("sigma_electrode" in params)

if using_electrode:    
    sigma_electrode = read_input_field(params['sigma_electrode'], mesh=mesh)
    ELECTRODE_TAG   = params["ELECTRODE_TAG"]

#------------------------------------------#
#     FUNCTION SPACES AND RESTRICTIONS     #
#------------------------------------------#
V = dfx.fem.functionspace(mesh, ("Lagrange", params["P"])) # Space for functions defined on the entire mesh

# vector for space, one for each tag
V_dict = dict()

# trial and test functions
u_dict = dict()
v_dict = dict()

# list for storing the solutions and forcing factors
uh_dict  = dict()
vij_dict = dict()
fg_dict  = dict()

# to store membrane potential
v = dfx.fem.Function(V)
v.name = "v"

for i in TAGS:

    V_i = V.clone()

    V_dict[i]  = V_i
    u_dict[i]  = ufl.TrialFunction(V_i)
    v_dict[i]  =  ufl.TestFunction(V_i)
    uh_dict[i] =  dfx.fem.Function(V_i)
    # v_ij con i < j to avoid repetions
    for j in TAGS:
        if i < j:
            # Membrane potential and forcing term function
            vij_dict[(i,j)] = dfx.fem.Function(V)
            fg_dict[(i,j)]  = dfx.fem.Function(V)

# get expression of initial membrane potential
v_init_expr = read_input_field(params['v_init'], V=V)

# turn expression into a Function with actual DOF values
v_init = dfx.fem.Function(V)
v_init.interpolate(v_init_expr)
        
# init vij using initial membrane potential
for i in TAGS:

    # interpolate v_init in intra_extra, intra_intra is 0 by default
    if i < ECS_TAG:
        vij_dict[(i,ECS_TAG)].interpolate(v_init)
        # v.x.array[:] += vij_dict[(i,ECS_TAG)].x.array[:]

    elif i > ECS_TAG:
        vij_dict[(ECS_TAG,i)].interpolate(v_init)

# save membrane potential for visualization (valid only for extra-intra)
v.x.array[:] = vij_dict[(TAGS[0],TAGS[1])].x.array[:]

##### Restrictions #####
restriction = []

for i in TAGS:

    V_i = V_dict[i]

    # Get indices of the cells of the intra- and extracellular subdomains
    cells_Omega_i = subdomains.indices[subdomains.values == i]

    if i == ECS_TAG and using_electrode:        
        cells_Omega_electrode = subdomains.indices[subdomains.values == ELECTRODE_TAG]      
        cells_Omega_i = np.concatenate([cells_Omega_i, cells_Omega_electrode])                          

    # Get dofs of the intra- and extracellular subdomains
    dofs_Vi_Omega_i = dfx.fem.locate_dofs_topological(V_i, subdomains.dim, cells_Omega_i)

    # Define the restrictions of the subdomains
    restriction_Vi_Omega_i = multiphenicsx.fem.DofMapRestriction(V_i.dofmap, dofs_Vi_Omega_i)

    restriction.append(restriction_Vi_Omega_i)

# timers
if comm.rank == 0: print(f"Creating FEM spaces:    {time.perf_counter() - t1:.2f} seconds")
t1 = time.perf_counter()
setup_time = t1 - start_time

# set ionic models
ionic_models = dict()

for i in TAGS:
    for j in TAGS:

        if i < j:
            if i == ECS_TAG or j == ECS_TAG:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=False)
            else:
                ionic_models[(i,j)] = ionic_model_factory(params, intra_intra=True, V=V)


####### BC #######
number_of_Dirichlet_points = params['Dirichlet_points']
Dirichletbc = (number_of_Dirichlet_points > 0) 

bcs = []

if Dirichletbc:

    # Apply zero Dirichlet condition
    zero = dfx.fem.Constant(mesh, 0.0)        

    # identify local boundary DOFs + coords
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    local_bdofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
    coords = V.tabulate_dof_coordinates()
    local_coords = coords[local_bdofs]      # shape (n_loc, gdim)

    # local to global
    imap = V.dofmap.index_map
    first_global = imap.local_range[0]       # first global index on this rank
    local_global_bdofs = first_global + local_bdofs

    # gather everyone’s cands to rank 0
    all_globals = comm.gather(local_global_bdofs, root=0)
    all_coords  = comm.gather(local_coords,      root=0)

    # on rank 0 pick the 10 “corner‐nearest” by taxi‐distance
    if comm.rank == 0:
        G  = np.concatenate(all_globals)
        C  = np.vstack(all_coords)
        scores = C.sum(axis=1)
        chosen_global = G[np.argsort(scores)[:number_of_Dirichlet_points]]
    else:
        chosen_global = None

    # broadcast the final 10 GLOBAL DOFs to everyone
    chosen_global = comm.bcast(chosen_global, root=0)

    # each rank picks from its local globals, maps back to local indices
    mask = np.isin(local_global_bdofs, chosen_global)
    local_chosen = local_bdofs[mask].astype(np.int32)

    # impose BCs only on these local_chosen
    for i in TAGS:
        bc_i = dfx.fem.dirichletbc(zero, local_chosen, V_dict[i])
        bcs.append(bc_i)

##############
#------------------------------------#
#        VARIATIONAL PROBLEM         #
#------------------------------------#

# bilinear form
a = []

# assemble block form
for i in TAGS:

    a_i = []

    # membranes tags for cell tag i
    membrane_i = membrane_tags[i]

    if i == ECS_TAG:
        sigma = sigma_e # extra-cellular

    else:
        sigma = sigma_i # intra-cellular

    v_i = v_dict[i]
    
    for j in TAGS:
        
        u_j  = u_dict[j]

        membrane_ij = tuple(common_elements(membrane_i,membrane_tags[j]))   

        # if cells i and j have a membrane in common
        if len(membrane_ij) > 0:

            if i == j:                                                      
                
                a_ij = tau * inner(sigma * grad(u_j), grad(v_i)) * dx(i) + inner(u_j('-'), v_i('-')) * dS(membrane_ij)   

                if i == ECS_TAG and using_electrode:
                    a_ij +=  tau * inner(sigma_electrode * grad(u_j), grad(v_i)) * dx(ELECTRODE_TAG)                       

            else:                                
                a_ij = - inner(u_j('+'), v_i('-')) * dS(membrane_ij)                                      
        else:
            a_ij = None

        a_i.append(a_ij)   

    a.append(a_i)

# Convert form to dolfinx form
a = dfx.fem.form(a, jit_options=jit_parameters)

# timers
if comm.rank == 0: print(f"Creating bilinear form: {time.perf_counter() - t1:.2f} seconds")
t1 = time.perf_counter() 

# #---------------------------#
# #      MATRIX ASSEMBLY      #
# #---------------------------#

# Assemble the block linear system matrix
A = multiphenicsx.fem.petsc.assemble_matrix_block(a, bcs=bcs, restriction=(restriction, restriction))
A.assemble()
assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time

if comm.rank == 0: print(f"Assembling matrix A:    {time.perf_counter() - t1:.2f} seconds")

# # Save A
# save_petsc_matrix_to_matlab(A, 'output/A.mat','A')
# # Plot sparsity pattern 
# save_sparsity_pattern(A, 'output/sparsity.png')
# exit()

#---------------------------------#
#        CREATE NULLSPACE         #
#---------------------------------#

if not Dirichletbc:
    # Create the PETSc nullspace vector and check that it is a valid nullspace of A
    nullspace = PETSc.NullSpace().create(constant=True,comm=comm)
    assert nullspace.test(A)
    # For convenience, we explicitly inform PETSc that A is symmetric, so that it automatically
    # sets the nullspace of A^T too (see the documentation of MatSetNullSpace).
    # Symmetry checked also by direct inspection through the plot_sparsity_pattern() function
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
    # Set the nullspace
    A.setNullSpace(nullspace)
    if params["ksp_type"] == "cg":        
        A.setNearNullSpace(nullspace)
    
#---------------------------------#
#      CONFIGURE SOLVER           #
#---------------------------------#

# Configure solver
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)

# Set solver
ksp.setType(params["ksp_type"])
ksp.getPC().setType(params["pc_type"])
if params['pc_type'] == "lu":
    ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()

if params["verbose"]:
    opts.setValue('ksp_view', None)
    opts.setValue('ksp_monitor_true_residual', None)

# for iterative solvers set tolerance
if params['pc_type'] != "lu" and params['ksp_type'] != "preonly":
    opts.setValue('ksp_rtol', params["ksp_rtol"])
    opts.setValue('ksp_converged_reason', None)

if params['pc_type'] == "hypre" and mesh.geometry.dim == 3:
    opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.7)

ksp.setFromOptions()

# intial time
t = 0.0

# Create output files
if params["save_output"]:

    # rename solutions
    for i in TAGS:        
        uh_dict[i].name  = "u_" + str(i)
    
    out_name = params.get("out_name", "").strip().lstrip("_")

    # potentials xdmf
    out_sol = dfx.io.XDMFFile(comm, out_name + "/solution.xdmf", "w")
    out_sol.write_mesh(mesh)            
        
    # memebrane potential xdmf
    out_v = dfx.io.XDMFFile(comm, out_name + "/v.xdmf" , "w")
    out_v.write_mesh(mesh)
    out_v.write_function(v, t)

    # save subdomain data, needed for parallel visualizaiton
    with dfx.io.XDMFFile(comm, out_name + "/tags.xdmf", "w") as out_tags:                     
        out_tags.write_mesh(mesh)            
        out_tags.write_meshtags(subdomains, mesh.geometry)
        out_tags.write_meshtags(boundaries, mesh.geometry)        
        out_tags.close()


#---------------------------------#
#        STIMULUS SETUP           #
#---------------------------------#

# user parameters
stim_expr  = params.get("I_stim", "100.0 * (x[0] < 0.03)")
stim_start = params.get("stim_start", 0.0)  # ms
stim_end   = params.get("stim_end", 1.0)    # ms

# Build a stimulus Function per tag/space (so spaces match v_i and uh_dict[i])
stim_fun = {}
for i in TAGS:
    Vi = V_dict[i]
    coords = Vi.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
    xlist = [coords[:, 0], coords[:, 1], coords[:, 2]]
    vals  = eval(stim_expr, {"x": xlist, "np": np})
    f = dfx.fem.Function(Vi)
    f.x.array[:] = np.asarray(vals, dtype=float)
    stim_fun[i] = f

# Time-dependent amplitude as a Constant (UFL-safe)
stim_amp = dfx.fem.Constant(mesh, PETSc.ScalarType(0.0))

ksp_iterations = []
I_ion = {}

#---------------------------------#
#        SOLUTION TIMELOOP        #
#---------------------------------#

# init auxiliary data structures
ksp_iterations = []
#I_ion = dict()

if comm.rank == 0: print("\n#-----------SOLVE----------#")

for time_step in range(params["time_steps"]):

    if comm.rank == 0: update_status(f'Time stepping: {int(100*time_step/params["time_steps"])}%')

    # physical time at current step (before advancing)
    t_n = float(time_step) * float(dt)

    # update stimulus amplitude based on current time
    if (stim_start <= t_n) and (t_n < stim_end):
        stim_amp.value = 1.0
    else:
        stim_amp.value = 0.0

    # init data structure for linear form
    L_list = []

    # Update and assemble vector that is the RHS of the linear system
    t1 = time.perf_counter() # Timestamp for assembly time-lapse      
    
    for i in TAGS:

        membrane_i = membrane_tags[i]
        
        v_i = v_dict[i]

        L_i = 0    

        for j in TAGS:                        
            
            if i != j:
            
                membrane_ij = tuple(common_elements(membrane_i,membrane_tags[j]))   
                
                if i < j:
                    ij_tuple = (i,j)                                        
                    L_coeff  = 1
                    with vij_dict[ij_tuple].x.petsc_vec.localForm() as v_local:

                        t_ODE = time.perf_counter()
                        
                        I_ion[ij_tuple] = ionic_models[ij_tuple]._eval(v_local[:])          

                        ODEs_time += time.perf_counter() - t_ODE 
                else:
                    ij_tuple = (j,i)
                    L_coeff  = -1                    
                    
                with fg_dict[ij_tuple].x.petsc_vec.localForm() as fg_local, vij_dict[ij_tuple].x.petsc_vec.localForm() as v_local:

                    fg_local[:] = v_local[:] - tau * I_ion[ij_tuple]

                L_i += L_coeff * inner(fg_dict[ij_tuple], v_i('+')) * dS(membrane_ij)

                # external stimulus (time-switched by Constant)
                if ECS_TAG in (i, j):
                    L_i += L_coeff * tau * stim_amp * inner(stim_fun[i], v_i('+')) * dS(membrane_ij)

                                
        L_list.append(L_i)

    # Increment time
    t += float(dt)

    t_test = time.perf_counter()
    
    # create some data structures
    if time_step == 0:

        # Convert form to dolfinx form                    
        L = dfx.fem.form(L_list, jit_options=jit_parameters) 

        # Create right-hand side and solution vectors        
        b       = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)
        sol_vec = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)                

    
    # Clear RHS vector to avoid accumulation and assemble RHS
    b.array[:] = 0
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    multiphenicsx.fem.petsc.assemble_vector_block(b, L, a, bcs=bcs, restriction=restriction) # Assemble RHS vector        
        
    # dump(b, 'output/bvec')
        
    # Neumann BC
    if time_step == 0:
        # Create solution vector
        sol_vec = multiphenicsx.fem.petsc.create_vector_block(L, restriction=restriction)

    if not Dirichletbc:
        # if the timestep is not zero, b changes anyway and the nullspace must be removed
        nullspace.remove(b)
    
    assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time
    
    # Solve the system
    t1 = time.perf_counter() # Timestamp for solver time-lapse
    ksp.solve(b, sol_vec)

    # store iterisons 
    ksp_iterations.append(ksp.getIterationNumber())

    # Update ghost values
    sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Extract sub-components of solution
    dofmap_list = (N_TAGS) * [V.dofmap]
    with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(sol_vec, dofmap_list, restriction) as uij_wrapper:
        for ui_ue_wrapper_local, component in zip(uij_wrapper, tuple(uh_dict.values())): 
            with component.x.petsc_vec.localForm() as component_local:
                component_local[:] = ui_ue_wrapper_local

    for i in TAGS:
        for j in TAGS:
            if i < j:                
                vij_dict[(i,j)].x.array[:] = uh_dict[i].x.array - uh_dict[j].x.array # TODO test other order?
                
    
    # fill v for visualization
    v.x.array[:] = uh_dict[ECS_TAG].x.array

    for i in TAGS:
        if i != ECS_TAG:
            v.x.array[:] -= uh_dict[i].x.array


    solve_time += time.perf_counter() - t1 # Add time lapsed to total solver time

    # save xdmf output
    if params["save_output"] and time_step % params["save_interval"] == 0:               
        for i in TAGS:
            out_sol.write_function(uh_dict[i], t)        

        out_v.write_function(v, t)


if comm.rank == 0: update_status(f'Time stepping: 100%')        

#------------------------------#
#         POST PROCESS         #
#------------------------------#
# Sum local assembly and solve times to get global values
max_local_assemble_time = comm.allreduce(assemble_time, op=MPI.MAX) # Global assembly time
max_local_solve_time    = comm.allreduce(solve_time   , op=MPI.MAX) # Global solve time
max_local_ODE_time      = comm.allreduce(ODEs_time    , op=MPI.MAX) # Global ODEs time
max_local_setup_time    = comm.allreduce(setup_time   , op=MPI.MAX) # Global setup time
total_time = max_local_assemble_time + max_local_solve_time + max_local_ODE_time + max_local_setup_time

# Print stuff
if comm.rank == 0: 
    print("\n\n#-----------INFO-----------#")
    print("MPI size     =", comm.size)        
    print("N_TAGS       =", N_TAGS   )
    print("dt           =", dt       )
    print("time steps   =", params["time_steps"])
    print("T            =", dt * params["time_steps"])
    print("P (FE order) =", params["P"])
    print("ksp_type     =", params["ksp_type"])
    print("pc_type      =", params["pc_type"] )
    print("Global #DoFs =", b.getSize())
    print("Average KSP iterations =", sum(ksp_iterations)/len(ksp_iterations))
    
    if isinstance(params["ionic_model"], dict):
        print("Ionic models:")
        for key, value in params["ionic_model"].items():
            print(f"  {key}: {value}")
    else:
        print("Ionic model:", params['ionic_model'])        


    print("\n#-------TIME ELAPSED-------#")
    print(f"Setup time:       {max_local_setup_time:.3f} seconds")
    print(f"Assembly time:    {max_local_assemble_time:.3f} seconds")
    print(f"Solve time:       {max_local_solve_time:.3f} seconds")
    print(f"Ionic model time: {max_local_ODE_time:.3f} seconds")
    print(f"Total time:       {total_time:.3f} seconds")    
    

if params["save_output"]:    

    out_sol.close()
    out_v.close()

    if comm.rank == 0: 
        print("\nSolution saved in output folder")    
        print(f"Total script time (with output): {time.perf_counter() - start_time:.3f} seconds\n")

