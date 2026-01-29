from mpi4py   import MPI
from typing   import Union
from petsc4py import PETSc
import scipy.sparse as sparse
import scipy.io as sio
import numpy        as np
import numpy.typing as npt
#import matplotlib.pyplot as plt
import sys
import ufl
import os
import yaml
import pickle 
from dolfinx.fem import Expression
from dolfinx.fem import FunctionSpace

# Assign intial membrane potential
class Read_input_field:
    def __init__(self, expression: Union[str, float, int]):
        self.expression = expression

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        # If expression is a number, return it directly as an array of the same shape as `x`
        if isinstance(self.expression, (int, float)):
            return self.expression + 0 * x[0] # hack
        # If expression is a string, evaluate it
        elif isinstance(self.expression, str):
            return eval(self.expression, {"np": np, "ufl": ufl, "x": x})
        else:
            raise ValueError("Expression must be a string, int, or float.")                     


# Create input field based on type and value
def read_input_field(expr: Union[str, float, int], V: FunctionSpace = None, mesh=None):
    
    if isinstance(expr, (float, int)):
        if V is None:
            return float(expr)
        return Expression(ufl.as_ufl(expr), V.element.interpolation_points(), MPI.COMM_WORLD)

    elif isinstance(expr, str):
        # resolve mesh
        m = V.mesh if V else mesh
        if m is None:
            raise ValueError("Need FunctionSpace or mesh to evaluate symbolic expression.")
        x = ufl.SpatialCoordinate(m)
        ufl_expr = eval(expr, {"ufl": ufl, "x": x, "np": np})
        return Expression(ufl_expr, V.element.interpolation_points(), MPI.COMM_WORLD) if V else ufl_expr

    raise TypeError(f"Unsupported expression type: {type(expr)}")


def parse_nonneg_int(s):
    try:
        i = int(s)
    except ValueError:
        raise ValueError(f"Invalid input “{s}”: not an integer.")
    if i < 0:
        raise ValueError(f"Invalid input “{s}”: must be ≥ 0.")

    if i != s:
        raise ValueError(f"Invalid input “{s}”: must be integer")
    

# read yml file
def read_input_file(input_yml_file):
        
        # read input yml file
        with open(input_yml_file, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)        
            
        input_parameters = dict()

        ######### geometry #########
        if 'mesh_file' in config:              
            check_if_file_exists(config['mesh_file'])
            input_parameters['mesh_file'] = config['mesh_file']                                        
        else:
            print('INPUT ERROR: provide mesh_file field in input .yml file')
            return

        if 'tags_dictionary_file' in config:      
            check_if_file_exists(config['tags_dictionary_file'])
            input_parameters['tags_dictionary_file'] = config['tags_dictionary_file']                                        
        else:
            print('INPUT ERROR: provide tags_dictionary_file field in input .yml file')
            return

        # get ECC tag if specified, otherwise use the minimum
        if 'ECS_TAG' in config:                  
            input_parameters['ECS_TAG'] = config['ECS_TAG']                                        
        else:
            
            with open(config["tags_dictionary_file"], "rb") as f:
                membrane_tags = pickle.load(f)

            input_parameters['ECS_TAG'] = min(membrane_tags.keys())          
            
            # Read input file 
            if MPI.COMM_WORLD .rank == 0: print("ECS tag not specified, using minimum one:", input_parameters['ECS_TAG'])  
            
                
        ######### problem #########
        if 'dt' in config:
            input_parameters['dt'] = config['dt']
        else:
            print('INPUT ERROR: provide dt in input .yml file')
            return
        
        if 'time_steps' in config: 
            input_parameters['time_steps'] = config['time_steps']            
        elif 'T' in config:            
            input_parameters['time_steps'] = int(config['T']/config['dt'])        
        else:
            raise SyntaxError(f'INPUT ERROR: provide final time T or time_steps in input .yml file.')
            
        if 'mesh_conversion_factor' in config: 
            input_parameters['mesh_conversion_factor'] = config['mesh_conversion_factor']
        else:
            input_parameters['mesh_conversion_factor'] = 1.0
        
        # Membrane capacitance, (dafult 1) 
        if 'C_M' in config: 
            input_parameters['C_M'] = config['C_M']
        else:
            input_parameters['C_M'] = 1.0

        # conductivities 
        if 'sigma_i' in config: 
            input_parameters['sigma_i'] = config['sigma_i']
        else:
            input_parameters['sigma_i'] = 1.0

        if 'sigma_e' in config: 
            input_parameters['sigma_e'] = config['sigma_e']
        else:
            input_parameters['sigma_e'] = 1.0
        
        if 'ELECTRODE_TAG' in config: 
            input_parameters['ELECTRODE_TAG'] = config['ELECTRODE_TAG']

            if 'sigma_electrode' in config: 
                input_parameters['sigma_electrode'] = config['sigma_electrode']                       
            else:
                print(f"WARNING: ELECTRODE_TAG with no sigma_electrode in input file!")            

        # Resistance 
        if 'R_g' in config: 
            input_parameters['R_g'] = config['R_g']
        else:
            input_parameters['R_g'] = 1.0
                            
        # finite element polynomial order (dafult 1) 
        if 'fem_order' in config: 
            input_parameters['P'] = config['fem_order']
        else:
            input_parameters['P'] = 1
        
        # initial membrane potential (dafult 1)
        if 'v_init' in config: 
            input_parameters['v_init'] = config['v_init']
        else:
            raise KeyError(f"Set v_init in input file!")

        # stimulus parameters
        if 'I_stim' in config:
            input_parameters['I_stim'] = config['I_stim']
        else:
            input_parameters['I_stim'] = "100.0 * (x[0] < 0.03)"

        if 'stim_start' in config:
            input_parameters['stim_start'] = config['stim_start']
        else:
            input_parameters['stim_start'] = 0.0

        if 'stim_end' in config:
            input_parameters['stim_end'] = config['stim_end']
        else:
            input_parameters['stim_end'] = 1.0

        # boundary conditions (default = 0 -> zero Neumann)
        if 'Dirichlet_points' in config:
            input_parameters['Dirichlet_points'] = config['Dirichlet_points']
        else:
            input_parameters['Dirichlet_points'] = 0

        # bc_type: "none", "one_corner", "all_corners", "all_boundary"
        if 'bc_type' in config:
            input_parameters['bc_type'] = config['bc_type']

        # ionic model 
        if 'ionic_model' in config: 
            input_parameters['ionic_model'] = config['ionic_model']

            if isinstance(input_parameters['ionic_model'], dict):
                if input_parameters['ionic_model'].keys() != {"intra_intra", "intra_extra"}:
                    raise KeyError(f"Use intra_intra and intra_extra input entries!")

        else:
            print('WARNING: setting default passive ionic model')
            input_parameters['ionic_model'] = "Passive"        
            
        ############### solver parameters ###############

        if 'ksp_type' in config: 
            input_parameters['ksp_type'] = config['ksp_type']
        else:
            input_parameters['ksp_type'] = 'cg'        

        if 'pc_type' in config: 
            input_parameters['pc_type'] = config['pc_type']
        else:
            input_parameters['pc_type'] = 'hypre'        

        if 'ksp_rtol' in config: 
            input_parameters['ksp_rtol'] = config['ksp_rtol']
        else:
            input_parameters['ksp_rtol'] = 1e-8
        
        if 'save_output' in config: 
            input_parameters['save_output'] = config['save_output']
        else:
            input_parameters['save_output'] = False

        if 'save_interval' in config: 
            input_parameters['save_interval'] = config['save_interval']
        else:
            input_parameters['save_interval'] = 1            
        
        if 'verbose' in config: 
            input_parameters['verbose'] = config['verbose']
        else:
            input_parameters['verbose'] = False

        if 'out_name' in config:
            input_parameters['out_name'] = "_" + config['out_name']
            print('WARNING: Paraview states only work with defult output names.')
        else:
            input_parameters['out_name'] = ''
            # raise SyntaxError(f'INPUT ERROR: provide name of output in input .yml file.')

        ######### Solver Backend #########
        if 'solver_backend' in config:
            input_parameters['solver_backend'] = config['solver_backend']
        else:
            input_parameters['solver_backend'] = 'petsc'

        # Ginkgo-specific configuration (optional)
        if 'ginkgo' in config:
            input_parameters['ginkgo'] = config['ginkgo']
        else:
            input_parameters['ginkgo'] = {}

        # sanuty checks
        parse_nonneg_int(input_parameters['P'])
        parse_nonneg_int(input_parameters['time_steps'])
        parse_nonneg_int(input_parameters['Dirichlet_points'])
                
        return input_parameters
     

def update_status(progress, total=None, width=40):
    """Display a progress bar in the terminal.

    Args:
        progress: Current progress value (0-100 if total is None, otherwise current step)
        total: Total number of steps (optional, if provided progress is treated as current step)
        width: Width of the progress bar in characters
    """
    if total is not None:
        percent = int(100 * progress / total)
    else:
        percent = int(progress)

    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    # Output with newline so it gets flushed, use special prefix for frontend detection
    sys.stdout.write(f'PROGRESS:{percent}:[{bar}] {percent:3d}%\n')
    sys.stdout.flush()


def check_if_file_exists(file_path):
    if not os.path.exists(file_path):        
        print(f"The file '{file_path}' does not exist.")
        exit()


def norm_2(vec):
    return sqrt(dot(vec,vec))


def dump(thing, path):
    name = path.split("/")[-1]
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data]), sio.savemat(path, {name: m})


def common_elements(set1, set2):    
    return set1.intersection(set2)

def plot_sparsity_pattern(A):
    ai, aj, av = A.getValuesCSR()
    rows, cols = A.getSize()
    sparse_matrix = sparse.csr_matrix((av, aj, ai), shape=(rows, cols))
    plt.figure(figsize=(8, 8))
    plt.spy(sparse_matrix, markersize=1)
    plt.title("Sparsity Pattern of the Matrix")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

