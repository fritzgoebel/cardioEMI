#!/usr/bin/env python3
"""
Flask server for the Cardiac EMI Visualization Tool.
Handles:
- Serving static files
- Updating YAML config
- Running Docker simulation with streaming output
"""

import os
import json
import subprocess
import re
import h5py
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response

app = Flask(__name__, static_folder='.')
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Simulation state
simulation_state = {
    'running': False,
    'process': None
}

# Mesh state
mesh_state = {
    'current': 'pepe36_colored',
    'currentConfig': 'input_pepe36_colored.yml',
    'converting': False
}

# --------------------- Static Files ---------------------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# --------------------- Config API ---------------------

@app.route('/api/config', methods=['GET'])
def get_config():
    """Read YAML config file and return as JSON."""
    config_file = request.args.get('file', 'input_pepe36_colored.yml')
    config_path = PROJECT_ROOT / config_file

    try:
        # Simple YAML parsing (good enough for our config)
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
        return jsonify(config)
    except FileNotFoundError:
        return jsonify({'error': f'Config file not found: {config_file}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update specific fields in YAML config file."""
    data = request.json
    config_file = data.get('file', 'input_pepe36_colored.yml')
    updates = data.get('updates', {})

    config_path = PROJECT_ROOT / config_file

    try:
        # Read existing file
        with open(config_path, 'r') as f:
            lines = f.readlines()

        # Update lines with new values
        for key, value in updates.items():
            # Match key with optional whitespace before colon
            pattern = rf'^(\s*)({re.escape(key)})\s*:'
            found = False
            for i, line in enumerate(lines):
                match = re.match(pattern, line)
                if match:
                    # Preserve original formatting (indentation and key spacing)
                    indent = match.group(1)
                    key_indent_len = len(indent)
                    # Keep simple format for updated values
                    lines[i] = f'{indent}{key}: {value}\n'
                    # Remove any continuation lines (lines that are more indented or start with whitespace after key)
                    # This handles multi-line YAML values that may have been set previously
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        # Check if this is a continuation line (starts with more whitespace and no key)
                        if next_line.strip() and not next_line.lstrip().startswith('#'):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            # If it's indented more than the key, or starts with special YAML chars, it's a continuation
                            if next_indent > key_indent_len and ':' not in next_line.split('#')[0]:
                                lines[j] = ''  # Mark for removal
                                j += 1
                                continue
                        break
                    found = True
                    break
            if not found:
                # Add new key at end
                lines.append(f'{key}: {value}\n')

        # Remove empty lines that were marked for deletion
        lines = [l for l in lines if l != '']

        # Write back
        with open(config_path, 'w') as f:
            f.writelines(lines)

        return jsonify({'success': True, 'message': 'Config updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/ginkgo', methods=['POST'])
def update_ginkgo_config():
    """Update the nested ginkgo configuration in YAML config file."""
    import yaml

    data = request.json
    config_file = data.get('file', 'input_pepe36_colored.yml')
    ginkgo_config = data.get('ginkgo', {})

    config_path = PROJECT_ROOT / config_file

    try:
        # Read the full YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Build the ginkgo config dictionary
        ginkgo_dict = {
            'backend': ginkgo_config.get('backend', 'omp'),
            'solver': ginkgo_config.get('solver', 'cg'),
            'preconditioner': ginkgo_config.get('preconditioner', 'jacobi'),
            'rtol': float(ginkgo_config.get('rtol', 1e-8)),
            'atol': float(ginkgo_config.get('atol', 1e-12)),
            'max_iterations': int(ginkgo_config.get('maxIterations', 1000))
        }

        # Add AMG config if present
        amg_config = ginkgo_config.get('amg', {})
        if amg_config:
            ginkgo_dict['amg'] = {
                'max_levels': int(amg_config.get('maxLevels', 10)),
                'cycle': amg_config.get('cycle', 'v'),
                'smoother': amg_config.get('smoother', 'jacobi'),
                'relaxation_factor': float(amg_config.get('relaxationFactor', 0.9))
            }

        config['ginkgo'] = ginkgo_dict

        # Write back with YAML formatting
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return jsonify({'success': True, 'message': 'Ginkgo config updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------- Mesh API ---------------------

def find_config_for_mesh(mesh_name):
    """Find a matching config file for a mesh."""
    # Try exact match first: input_{mesh_name}.yml
    config_path = PROJECT_ROOT / f'input_{mesh_name}.yml'
    if config_path.exists():
        return f'input_{mesh_name}.yml'

    # Try base name (e.g., robin-24335 -> robin)
    base_name = mesh_name.split('-')[0]
    config_path = PROJECT_ROOT / f'input_{base_name}.yml'
    if config_path.exists():
        return f'input_{base_name}.yml'

    # Try without underscores (e.g., pepe36_colored -> pepe36)
    base_name = mesh_name.split('_')[0]
    config_path = PROJECT_ROOT / f'input_{base_name}.yml'
    if config_path.exists():
        return f'input_{base_name}.yml'

    return None

@app.route('/api/meshes')
def list_meshes():
    """List available mesh files from data/ directory."""
    data_dir = PROJECT_ROOT / 'data'
    viz_data_dir = Path(__file__).parent / 'data'

    meshes = []
    for h5_file in sorted(data_dir.glob('*.h5')):
        name = h5_file.stem
        converted_dir = viz_data_dir / name
        config_file = find_config_for_mesh(name)
        meshes.append({
            'name': name,
            'file': h5_file.name,
            'size': h5_file.stat().st_size,
            'converted': (converted_dir / 'mesh_metadata.json').exists(),
            'configFile': config_file
        })

    # Also list available config files
    config_files = [f.name for f in sorted(PROJECT_ROOT.glob('input*.yml'))]

    return jsonify({
        'meshes': meshes,
        'configFiles': config_files,
        'current': mesh_state['current'],
        'currentConfig': mesh_state.get('currentConfig', 'input_pepe36_colored.yml'),
        'converting': mesh_state['converting']
    })

@app.route('/api/meshes/convert', methods=['POST'])
def convert_mesh_endpoint():
    """Convert an HDF5 mesh to visualization format with SSE progress."""
    data = request.json
    mesh_name = data.get('mesh')

    if not mesh_name:
        return jsonify({'error': 'No mesh specified'}), 400

    def generate():
        if mesh_state['converting']:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Conversion already in progress'})}\n\n"
            return

        mesh_state['converting'] = True

        try:
            h5_path = PROJECT_ROOT / 'data' / f'{mesh_name}.h5'
            if not h5_path.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': f'HDF5 file not found: {mesh_name}.h5'})}\n\n"
                return

            output_dir = Path(__file__).parent / 'data' / mesh_name

            # Import and run conversion
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
            from convert_hdf5 import convert_mesh

            def progress_callback(percent, message):
                pass  # Will be handled by yielding

            yield f"data: {json.dumps({'type': 'progress', 'percent': 0, 'message': 'Starting conversion...'})}\n\n"

            # Run conversion (this is synchronous, but we report start/end)
            metadata = convert_mesh(h5_path, output_dir)

            yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Conversion complete!'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'metadata': metadata})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

        finally:
            mesh_state['converting'] = False

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/meshes/select', methods=['POST'])
def select_mesh():
    """Select a converted mesh for use."""
    data = request.json
    mesh_name = data.get('mesh')
    config_file = data.get('configFile')  # Optional: explicitly set config

    if not mesh_name:
        return jsonify({'error': 'No mesh specified'}), 400

    viz_data_dir = Path(__file__).parent / 'data' / mesh_name
    metadata_path = viz_data_dir / 'mesh_metadata.json'

    if not metadata_path.exists():
        return jsonify({'error': f'Mesh not converted: {mesh_name}'}), 400

    mesh_state['current'] = mesh_name

    # Set config file - use provided one, or find matching one, or keep current
    if config_file:
        mesh_state['currentConfig'] = config_file
    else:
        found_config = find_config_for_mesh(mesh_name)
        if found_config:
            mesh_state['currentConfig'] = found_config

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return jsonify({
        'success': True,
        'message': f'Selected mesh: {mesh_name}',
        'metadata': metadata,
        'configFile': mesh_state['currentConfig']
    })

@app.route('/api/meshes/current')
def get_current_mesh():
    """Get currently selected mesh and its metadata."""
    mesh_name = mesh_state['current']
    viz_data_dir = Path(__file__).parent / 'data' / mesh_name
    metadata_path = viz_data_dir / 'mesh_metadata.json'

    if not metadata_path.exists():
        return jsonify({'error': f'Current mesh not found: {mesh_name}'}), 404

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return jsonify({
        'name': mesh_name,
        'metadata': metadata
    })

# --------------------- Simulation API ---------------------

@app.route('/api/simulation/run')
def run_simulation():
    """Run Docker simulation with Server-Sent Events for streaming output."""
    import yaml

    # Get MPI ranks from query parameter, default to 8, clamp to 1-32
    ranks = request.args.get('ranks', 8, type=int)
    ranks = max(1, min(32, ranks))

    config_file = request.args.get('config', 'input_pepe36_colored.yml')

    # Check config file for solver backend to determine Docker image
    config_path = PROJECT_ROOT / config_file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        solver_backend = config.get('solver_backend', 'petsc').lower()
    except:
        solver_backend = 'petsc'

    # Select Docker image based on solver backend
    if solver_backend == 'ginkgo':
        docker_image = 'dolfinx-ginkgo:latest'
        # For Ginkgo, we need to build the Python bindings first if not already done
        setup_cmd = 'cd dolfinx-ginkgo && mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/usr/local/dolfinx-real -DDOLFINX_GINKGO_BUILD_PYTHON=ON && make -j2 && cd /home/fenics && '
    else:
        docker_image = 'ghcr.io/fenics/dolfinx/dolfinx:v0.9.0'
        setup_cmd = ''

    def generate():
        if simulation_state['running']:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Simulation already running'})}\n\n"
            return

        simulation_state['running'] = True

        docker_cmd = [
            'docker', 'run', '-t',
            '-v', f'{PROJECT_ROOT}:/home/fenics',
            '-w', '/home/fenics',
            docker_image,
            'bash', '-c',
            f'{setup_cmd}pip install --no-build-isolation -q -r requirements.txt && mpirun -n {ranks} python3 -u main.py {config_file}'
        ]

        try:
            backend_msg = f"Using {solver_backend.upper()} solver backend ({docker_image})"
            yield f"data: {json.dumps({'type': 'output', 'text': backend_msg + '\\n'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'text': 'Starting simulation...\\n'})}\n\n"

            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            simulation_state['process'] = process

            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Check for progress bar output (format: PROGRESS:percent:message)
                    if line.startswith('PROGRESS:'):
                        parts = line.strip().split(':', 2)
                        if len(parts) >= 3:
                            percent = int(parts[1])
                            message = parts[2]
                            yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'message': message})}\n\n"
                    # Check for iterations output (format: ITERATIONS:step:count)
                    elif line.startswith('ITERATIONS:'):
                        parts = line.strip().split(':')
                        if len(parts) >= 3:
                            step = int(parts[1])
                            count = int(parts[2])
                            yield f"data: {json.dumps({'type': 'iterations', 'step': step, 'count': count})}\n\n"
                    # Check for residual output (format: RESIDUAL:step:abs:rel)
                    elif line.startswith('RESIDUAL:'):
                        parts = line.strip().split(':')
                        if len(parts) >= 4:
                            step = int(parts[1])
                            res_abs = float(parts[2])
                            res_rel = float(parts[3])
                            yield f"data: {json.dumps({'type': 'residual', 'step': step, 'abs': res_abs, 'rel': res_rel})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            # Wait for completion
            process.wait()

            success = process.returncode == 0
            yield f"data: {json.dumps({'type': 'complete', 'success': success, 'returncode': process.returncode})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        finally:
            simulation_state['running'] = False
            simulation_state['process'] = None

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/simulation/status')
def simulation_status():
    """Get current simulation status."""
    return jsonify({
        'running': simulation_state['running']
    })

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop running simulation."""
    if simulation_state['process']:
        simulation_state['process'].terminate()
        return jsonify({'success': True, 'message': 'Simulation stopped'})
    return jsonify({'success': False, 'message': 'No simulation running'})

@app.route('/api/system/info')
def system_info():
    """Return system information for UI defaults."""
    import os
    cpu_count = os.cpu_count() or 8
    return jsonify({
        'cpu_count': cpu_count,
        'recommended_ranks': min(8, cpu_count),
        'max_ranks': min(32, cpu_count * 2)
    })

# --------------------- Results API ---------------------

@app.route('/api/simulations')
def list_simulations():
    """List available simulation output directories."""
    simulations = []

    # Look for directories ending in _sim that contain v.h5
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name.endswith('_sim'):
            v_h5 = item / 'v.h5'
            if v_h5.exists():
                # Check if viz data exists
                viz_data_dir = Path(__file__).parent / 'data' / item.name
                has_viz = (viz_data_dir / 'mesh_metadata.json').exists()

                simulations.append({
                    'name': item.name,
                    'path': str(item),
                    'has_viz_data': has_viz,
                    'size': sum(f.stat().st_size for f in item.glob('*.h5'))
                })

    return jsonify({
        'simulations': sorted(simulations, key=lambda x: x['name'])
    })

@app.route('/api/results')
def get_results():
    """Load simulation results from HDF5 file with per-facet voltage mapping."""
    output_dir = request.args.get('dir', 'pepe36_colored_sim')
    regenerate = request.args.get('regenerate', 'false').lower() == 'true'

    sim_output_dir = PROJECT_ROOT / output_dir
    if not sim_output_dir.exists():
        return jsonify({'error': f'Simulation output not found: {output_dir}'}), 404

    # Use simulation output directory name as viz data source
    sim_name = Path(output_dir).name
    mesh_data_dir = Path(__file__).parent / 'data' / sim_name
    viz_mesh_path = mesh_data_dir / 'mesh_vertices.bin'
    metadata_path = mesh_data_dir / 'mesh_metadata.json'

    # Auto-generate viz data from simulation output if not present or if regenerate requested
    if not viz_mesh_path.exists() or regenerate:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
            from generate_viz_from_output import generate_viz_data
            generate_viz_data(sim_output_dir, mesh_data_dir)
        except Exception as e:
            import traceback
            return jsonify({
                'error': f'Failed to generate visualization data: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500

    if not viz_mesh_path.exists():
        return jsonify({'error': f'Visualization data not found for: {sim_name}'}), 404

    try:
        # Load viz metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load facet-to-original-vertex mapping
        facet_orig_vertices_path = mesh_data_dir / 'facet_orig_vertices.bin'
        facet_pair_indices_path = mesh_data_dir / 'facet_pair_indices.bin'

        if facet_orig_vertices_path.exists() and facet_pair_indices_path.exists():
            facet_orig_vertices = np.fromfile(facet_orig_vertices_path, dtype=np.uint32).reshape(-1, 3)
            facet_pair_indices = np.fromfile(facet_pair_indices_path, dtype=np.int32)
            unique_pairs = [tuple(p) for p in metadata.get('unique_pairs', [])]
        else:
            # Fallback: no per-facet mapping, use old method
            facet_orig_vertices = None
            facet_pair_indices = None
            unique_pairs = []

        # Find available vij files
        vij_files = {}
        for vij_path in sim_output_dir.glob('v_*_*.h5'):
            parts = vij_path.stem.split('_')
            if len(parts) == 3:
                try:
                    i, j = int(parts[1]), int(parts[2])
                    vij_files[(i, j)] = vij_path
                except ValueError:
                    pass

        # Determine which voltage source to use
        use_per_facet = len(vij_files) > 0 and facet_orig_vertices is not None

        def parse_time(key):
            return float(key.replace('_', '.'))

        if use_per_facet:
            # Use per-membrane voltage files for correct visualization
            # Pick the first vij file to get timesteps
            first_vij_path = list(vij_files.values())[0]
            with h5py.File(first_vij_path, 'r') as f:
                func_name = list(f['Function'].keys())[0]
                v_group = f['Function'][func_name]
                timestep_keys = sorted(v_group.keys(), key=parse_time)

            # Sample timesteps
            max_timesteps = 50
            if len(timestep_keys) > max_timesteps:
                step = len(timestep_keys) // max_timesteps
                timestep_keys = timestep_keys[::step]

            # Load vij data for all pairs and timesteps
            vij_data = {}
            for pair, vij_path in vij_files.items():
                with h5py.File(vij_path, 'r') as f:
                    func_name = list(f['Function'].keys())[0]
                    v_group = f['Function'][func_name]
                    vij_data[pair] = {}
                    for key in timestep_keys:
                        if key in v_group:
                            vij_data[pair][key] = v_group[key][:].flatten()

            # Build per-vertex voltages for expanded mesh
            num_facets = len(facet_orig_vertices)
            voltages = []
            times = []

            for key in timestep_keys:
                # For each facet, get voltage from the correct vij
                expanded_voltages = np.zeros(num_facets * 3, dtype=np.float32)

                for facet_idx in range(num_facets):
                    pair_idx = facet_pair_indices[facet_idx]
                    orig_verts = facet_orig_vertices[facet_idx]

                    if pair_idx < len(unique_pairs):
                        pair = unique_pairs[pair_idx]
                        if pair in vij_data and key in vij_data[pair]:
                            v_data = vij_data[pair][key]
                            for local_v, orig_v in enumerate(orig_verts):
                                if orig_v < len(v_data):
                                    expanded_voltages[facet_idx * 3 + local_v] = v_data[orig_v]

                voltages.append(expanded_voltages.tolist())
                times.append(parse_time(key))

        else:
            # Fallback: use summed v.h5 (old method)
            h5_path = sim_output_dir / 'v.h5'
            if not h5_path.exists():
                return jsonify({'error': f'v.h5 not found in {output_dir}'}), 404

            with h5py.File(h5_path, 'r') as f:
                v_group = f['Function']['v']
                timestep_keys = sorted(v_group.keys(), key=parse_time)

                max_timesteps = 50
                if len(timestep_keys) > max_timesteps:
                    step = len(timestep_keys) // max_timesteps
                    timestep_keys = timestep_keys[::step]

                voltages = []
                times = []
                for key in timestep_keys:
                    v_data = v_group[key][:].flatten()
                    voltages.append(v_data.tolist())
                    times.append(parse_time(key))

        # Compute voltage range
        all_v = np.concatenate([np.array(v) for v in voltages])
        v_min = float(np.min(all_v))
        v_max = float(np.max(all_v))

        # Load iterations if available
        iterations_path = sim_output_dir / 'iterations.pickle'
        iterations = None
        if iterations_path.exists():
            import pickle
            with open(iterations_path, 'rb') as f:
                iterations = pickle.load(f)

        # Load residuals if available
        residuals_path = sim_output_dir / 'residuals.pickle'
        residuals = None
        if residuals_path.exists():
            import pickle
            with open(residuals_path, 'rb') as f:
                residuals = pickle.load(f)

        # Load DOF rank data if available
        dof_ranks_path = mesh_data_dir / 'dof_ranks.bin'
        rank_metadata_path = mesh_data_dir / 'rank_metadata.json'
        ranks_data = None
        num_ranks = None
        rank_centroids = None
        global_centroid = None

        if dof_ranks_path.exists():
            ranks_data = np.fromfile(dof_ranks_path, dtype=np.int32).tolist()
            if rank_metadata_path.exists():
                with open(rank_metadata_path, 'r') as f:
                    rank_meta = json.load(f)
                    num_ranks = rank_meta.get('num_ranks')
                    rank_centroids = rank_meta.get('rank_centroids')
                    global_centroid = rank_meta.get('global_centroid')

        # Load ECS rank data if available
        ecs_ranks_data = None
        ecs_ranks_path = mesh_data_dir / 'ecs_ranks.bin'
        if ecs_ranks_path.exists():
            ecs_ranks_data = np.fromfile(ecs_ranks_path, dtype=np.int32).tolist()

        # Load partition cut rank data if available
        cut_ranks_data = None
        cut_ranks_path = mesh_data_dir / 'cut_ranks.bin'
        if cut_ranks_path.exists():
            cut_ranks_data = np.fromfile(cut_ranks_path, dtype=np.int32).tolist()

        return jsonify({
            'voltages': voltages,
            'times': times,
            'vMin': v_min,
            'vMax': v_max,
            'numTimesteps': len(timestep_keys),
            'vizDataDir': sim_name,
            'perFacet': use_per_facet,
            'iterations': iterations,
            'residuals': residuals,
            'ranks': ranks_data,
            'numRanks': num_ranks,
            'rankCentroids': rank_centroids,
            'globalCentroid': global_centroid,
            'ecsRanks': ecs_ranks_data,
            'cutRanks': cut_ranks_data
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# --------------------- Video Export API ---------------------

video_state = {
    'exporting': False,
    'progress': 0,
    'filename': None
}

@app.route('/api/video/export', methods=['POST'])
def export_video():
    """Export simulation animation as video with SSE progress.

    Runs video export in a subprocess to avoid macOS threading issues with VTK.
    """
    data = request.json or {}

    def generate():
        if video_state['exporting']:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Video export already in progress'})}\n\n"
            return

        video_state['exporting'] = True
        video_state['progress'] = 0

        try:
            # Get parameters
            output_dir = data.get('output_dir', 'pepe36_colored_sim')
            camera_config = data.get('camera')
            width = data.get('width', 1920)
            height = data.get('height', 1080)
            fps = data.get('fps', 30)

            # Paths - use simulation-specific viz data directory
            sim_name = Path(output_dir).name
            viz_data_dir = Path(__file__).parent / 'data' / sim_name
            sim_output_dir = PROJECT_ROOT / output_dir
            video_output_dir = Path(__file__).parent / 'videos'

            if not sim_output_dir.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Simulation output not found: {sim_output_dir}'})}\n\n"
                return

            # Auto-generate viz data if not present
            if not (viz_data_dir / 'mesh_vertices.bin').exists():
                yield f"data: {json.dumps({'type': 'progress', 'percent': 0, 'message': 'Generating visualization data...'})}\n\n"
                import sys
                sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
                from generate_viz_from_output import generate_viz_data
                generate_viz_data(sim_output_dir, viz_data_dir)

            yield f"data: {json.dumps({'type': 'progress', 'percent': 5, 'message': 'Starting video export subprocess...'})}\n\n"

            # Create videos directory
            video_output_dir.mkdir(parents=True, exist_ok=True)

            # Build command to run video exporter as subprocess
            # This avoids macOS threading issues with VTK (NSWindow must be on main thread)
            script_path = Path(__file__).parent / 'scripts' / 'video_exporter.py'

            cmd = [
                'python', str(script_path),
                '--viz-data', str(viz_data_dir),
                '--sim-output', str(sim_output_dir),
                '--video-output', str(video_output_dir),
                '--width', str(width),
                '--height', str(height),
                '--fps', str(fps),
            ]

            if camera_config:
                cmd.extend(['--camera', json.dumps(camera_config)])

            # Run subprocess and stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            video_filename = None

            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    # Parse progress output from video_exporter
                    if line.startswith('PROGRESS:'):
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            percent = int(parts[1])
                            message = parts[2]
                            video_state['progress'] = percent
                            yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'message': message})}\n\n"
                    elif line.startswith('VIDEO_FILE:'):
                        video_filename = line.split(':', 1)[1].strip()
                    elif line.startswith('ERROR:'):
                        error_msg = line.split(':', 1)[1].strip()
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    else:
                        # Regular output
                        yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            process.wait()

            if process.returncode == 0 and video_filename:
                video_state['filename'] = video_filename
                yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Video export complete!'})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'filename': video_filename})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Video export failed with code {process.returncode}'})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

        finally:
            video_state['exporting'] = False

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/video/status')
def video_status():
    """Get video export status."""
    return jsonify({
        'exporting': video_state['exporting'],
        'progress': video_state['progress'],
        'filename': video_state['filename']
    })

@app.route('/api/video/download/<filename>')
def download_video(filename):
    """Download a generated video file."""
    videos_dir = Path(__file__).parent / 'videos'
    return send_from_directory(videos_dir, filename, as_attachment=True)

# --------------------- Main ---------------------

if __name__ == '__main__':
    print("=" * 50)
    print("Cardiac EMI Visualization Server")
    print("=" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Open http://localhost:8000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
