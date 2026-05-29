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
import threading
import h5py
import numpy as np
from pathlib import Path
from flask import Flask, request, send_from_directory, Response
import math


class NaNSafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null for valid JSON output."""
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return super().default(obj)

    def encode(self, obj):
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


def jsonify(obj):
    """Custom jsonify that handles NaN/Inf values."""
    return Response(
        json.dumps(obj, cls=NaNSafeJSONEncoder),
        mimetype='application/json'
    )


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

# Viz generation state
viz_gen_state = {
    'generating': False,
    'sim_name': None,
    'progress': 0,
    'message': '',
    'done': False,
    'error': None,
    'lock': threading.Lock(),
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
    import yaml
    config_file = request.args.get('file', 'input_pepe36_colored.yml')
    config_path = PROJECT_ROOT / config_file

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
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

        # Ensure essential keys exist (for remote-only meshes where config was created without them)
        mesh_name = config_file.replace('input_', '').replace('.yml', '')
        essential_defaults = [
            ('mesh_file', f'data/{mesh_name}.xdmf'),
            ('tags_dictionary_file', f'data/{mesh_name}.pickle'),
            ('mesh_conversion_factor', '0.0001'),
            ('save_output', 'true'),
            ('save_interval', '10'),
        ]
        insert_lines = []
        for key, default_value in essential_defaults:
            pattern = rf'^(\s*){re.escape(key)}\s*:'
            if not any(re.match(pattern, line) for line in lines):
                insert_lines.append(f'{key}: {default_value}\n')
        # Add ionic_model as block-style YAML if missing
        if not any(re.match(r'^(\s*)ionic_model\s*:', line) for line in lines):
            insert_lines.append('ionic_model:\n')
            insert_lines.append('  intra_intra: Passive\n')
            insert_lines.append('  intra_extra: AP\n')
        for line in reversed(insert_lines):
            lines.insert(0, line)

        # Write back
        with open(config_path, 'w') as f:
            f.writelines(lines)

        return jsonify({'success': True, 'message': 'Config updated'})
    except FileNotFoundError:
        # Config file doesn't exist - create from template for remote mesh
        try:
            import yaml
            mesh_name = config_file.replace('input_', '').replace('.yml', '')
            template = {
                'mesh_file': f'data/{mesh_name}.xdmf',
                'tags_dictionary_file': f'data/{mesh_name}.pickle',
                'mesh_conversion_factor': 0.0001,
                'fem_order': 1,
                'dt': 0.001,
                'time_steps': 1000,
                'C_M': 1,
                'sigma_i': 4,
                'sigma_e': 20,
                'R_g': 0.003,
                'v_init': '(-80.0) + (80.0) * ((x[0] >= -0.0062) * (x[0] <= 0.0015) * (x[1] >= -0.0019) * (x[1] <= 0.0068) * (x[2] >= -0.002) * (x[2] <= 0.0118))',
                'Dirichlet_points': 1,
                'ionic_model': {'intra_intra': 'Passive', 'intra_extra': 'AP'},
                'solver_backend': 'petsc',
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'ksp_rtol': '1e-4',
                'ksp_atol': '1e-8',
            }
            # Apply updates
            for key, value in updates.items():
                template[key] = value
            with open(config_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            return jsonify({'success': True, 'message': 'Config created from template'})
        except Exception as e2:
            return jsonify({'error': str(e2)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/scar', methods=['POST'])
def update_scar_config():
    """Generate scar tissue conductivity expressions and write to YAML config."""
    import yaml

    data = request.json
    config_file = data.get('file', 'input_pepe36_colored.yml')
    regions = data.get('regions', [])  # [{box: {xMin,...}, margin, dense: {si,se}, border: {si,se}}]
    healthy = data.get('healthy', {'sigma_i': 4.0, 'sigma_e': 20.0})
    cf = data.get('conversionFactor', 0.0001)

    config_path = PROJECT_ROOT / config_file

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        if not regions:
            config['sigma_i'] = healthy['sigma_i']
            config['sigma_e'] = healthy['sigma_e']
            config.pop('scar_config', None)
        else:
            sigma_i_expr, sigma_e_expr = _build_scar_expressions(regions, cf, healthy)
            config['sigma_i'] = sigma_i_expr
            config['sigma_e'] = sigma_e_expr
            # Save scar geometry in micrometers for visualization playback
            config['scar_config'] = {
                'regions': regions,
                'healthy': healthy,
            }

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return jsonify({
            'success': True,
            'sigma_i': str(config['sigma_i']),
            'sigma_e': str(config['sigma_e']),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def _build_scar_expressions(regions, cf, healthy):
    """Build UFL expression strings for sigma_i and sigma_e with scar regions.

    Uses ufl.conditional/ufl.ge/ufl.le/ufl.And instead of comparison operators
    (* products of >= / <=), because sigma expressions are used in variational
    forms compiled by ffcx, which cannot handle Python comparison products.

    Each region defines:
      - box (inner): dense scar zone
      - box + margin (outer): border zone (the ring between inner and outer)
      - outside outer: healthy tissue
    """
    def fmt(v):
        return f'{v:.8g}'

    def box_condition(prefix, box_coords):
        """Build a ufl.And chain for a 3D box condition."""
        xmin, xmax, ymin, ymax, zmin, zmax = box_coords
        return (
            f'ufl.And(ufl.ge({prefix}[0], {fmt(xmin)}), '
            f'ufl.And(ufl.le({prefix}[0], {fmt(xmax)}), '
            f'ufl.And(ufl.ge({prefix}[1], {fmt(ymin)}), '
            f'ufl.And(ufl.le({prefix}[1], {fmt(ymax)}), '
            f'ufl.And(ufl.ge({prefix}[2], {fmt(zmin)}), '
            f'ufl.le({prefix}[2], {fmt(zmax)}))))))'
        )

    # Build nested conditionals: for each region, check inner first, then outer
    # Result: conditional(inner, dense, conditional(outer, border, healthy))
    si_healthy = fmt(healthy['sigma_i'])
    se_healthy = fmt(healthy['sigma_e'])

    # Start from the outermost fallback (healthy) and wrap inward
    si_expr = si_healthy
    se_expr = se_healthy

    for region in regions:
        box = region['box']
        margin = region.get('margin', 10)
        dense = region.get('dense', {'sigma_i': 0.2, 'sigma_e': 1.0})
        border = region.get('border', {'sigma_i': 2.0, 'sigma_e': 10.0})

        inner_coords = (
            box['xMin'] * cf, box['xMax'] * cf,
            box['yMin'] * cf, box['yMax'] * cf,
            box['zMin'] * cf, box['zMax'] * cf,
        )
        outer_coords = (
            (box['xMin'] - margin) * cf, (box['xMax'] + margin) * cf,
            (box['yMin'] - margin) * cf, (box['yMax'] + margin) * cf,
            (box['zMin'] - margin) * cf, (box['zMax'] + margin) * cf,
        )

        inner_cond = box_condition('x', inner_coords)
        outer_cond = box_condition('x', outer_coords)

        # Wrap: conditional(outer, conditional(inner, dense, border), previous)
        si_expr = (
            f'ufl.conditional({outer_cond}, '
            f'ufl.conditional({inner_cond}, {fmt(dense["sigma_i"])}, {fmt(border["sigma_i"])}), '
            f'{si_expr})'
        )
        se_expr = (
            f'ufl.conditional({outer_cond}, '
            f'ufl.conditional({inner_cond}, {fmt(dense["sigma_e"])}, {fmt(border["sigma_e"])}), '
            f'{se_expr})'
        )

    return si_expr, se_expr


@app.route('/api/config/conditions', methods=['POST'])
def save_conditions():
    """Save conditions.json to a simulation output directory."""
    import hashlib as _hashlib
    data = request.json
    out_name = data.get('out_name')
    conditions = data.get('conditions')

    if not out_name or not conditions:
        return jsonify({'error': 'out_name and conditions required'}), 400

    conditions_hash = _hashlib.sha256(
        json.dumps(conditions, sort_keys=True).encode()
    ).hexdigest()[:12]
    conditions['_hash'] = conditions_hash

    out_dir = PROJECT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'conditions.json', 'w') as f:
        json.dump(conditions, f, indent=2)

    return jsonify({'success': True, 'hash': conditions_hash})

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
            'native_assembly': ginkgo_config.get('nativeAssembly', True),  # Default to native
            'dd_matrix': ginkgo_config.get('ddMatrix', False),  # Domain decomposition matrix
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

        # Add BDDC config if present
        bddc_config = ginkgo_config.get('bddc', {})
        if bddc_config:
            bddc_dict = {
                'local_solver': bddc_config.get('localSolver', 'direct'),
                'local_max_iterations': int(bddc_config.get('localMaxIterations', 100)),
                'local_tolerance': float(bddc_config.get('localTolerance', 1e-12)),
                'coarse_solver': bddc_config.get('coarseSolver', 'cg'),
                'coarse_max_iterations': int(bddc_config.get('coarseMaxIterations', 100)),
                'coarse_bddc_local_solver': bddc_config.get('coarseBddcLocalSolver', 'direct'),
                'vertices': bddc_config.get('vertices', True),
                'edges': bddc_config.get('edges', True),
                'faces': bddc_config.get('faces', True),
                'repartition_coarse': bddc_config.get('repartitionCoarse', True)
            }

            # Add local AMG config if present
            local_amg_config = bddc_config.get('localAmg', {})
            if local_amg_config:
                bddc_dict['local_amg'] = {
                    'coarsening': local_amg_config.get('coarsening', 'pgm'),
                    'strength_threshold': float(local_amg_config.get('strengthThreshold', 0.25)),
                    'cycle': local_amg_config.get('cycle', 'v'),
                    'smoother': local_amg_config.get('smoother', 'jacobi'),
                    'smooth_steps': int(local_amg_config.get('smoothSteps', 1)),
                    'max_levels': int(local_amg_config.get('maxLevels', 10)),
                    'coarse_solver': local_amg_config.get('coarseSolver', 'direct'),
                    'relaxation_factor': float(local_amg_config.get('relaxationFactor', 0.9))
                }

            # Add local Hypre BoomerAMG config if present
            local_hypre_config = bddc_config.get('localHypre', {})
            if local_hypre_config:
                bddc_dict['local_hypre'] = {
                    'cycle_type': int(local_hypre_config.get('cycleType', 1)),
                    'coarsening_type': int(local_hypre_config.get('coarseningType', 10)),
                    'strength_threshold': float(local_hypre_config.get('strengthThreshold', 0.25)),
                    'smoother_type': int(local_hypre_config.get('smootherType', 6)),
                    'num_sweeps': int(local_hypre_config.get('numSweeps', 1)),
                    'interpolation_type': int(local_hypre_config.get('interpolationType', 0)),
                    'max_levels': int(local_hypre_config.get('maxLevels', 25))
                }

            ginkgo_dict['bddc'] = bddc_dict

        config['ginkgo'] = ginkgo_dict

        # Write back with YAML formatting
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return jsonify({'success': True, 'message': 'Ginkgo config updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/petsc_bddc', methods=['POST'])
def update_petsc_bddc_config():
    """Update the nested petsc_bddc configuration in YAML config file."""
    import yaml

    data = request.json
    config_file = data.get('file', 'input_pepe36_colored.yml')
    bddc_config = data.get('petsc_bddc', {})

    config_path = PROJECT_ROOT / config_file

    try:
        # Read the full YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Build the petsc_bddc config dictionary
        petsc_bddc_dict = {
            'scaling': bddc_config.get('scaling', 'stiffness'),
            'local_solver': bddc_config.get('localSolver', 'mumps'),
            'coarse_solver': bddc_config.get('coarseSolver', 'mumps'),
            'coarse_pc_type': bddc_config.get('coarsePcType', 'lu'),
            'use_vertices': bddc_config.get('useVertices', True),
            'use_edges': bddc_config.get('useEdges', True),
            'use_faces': bddc_config.get('useFaces', False)
        }

        # Forward sub-PC options when the local solver is hypre BoomerAMG.
        # main.py reads dirichlet_options/neumann_options and prepends the
        # -pc_bddc_<side>_ prefix, so keys here are the inner option names.
        if petsc_bddc_dict['local_solver'] == 'hypre':
            h = bddc_config.get('localHypre', {})
            hypre_options = {
                'pc_hypre_type': 'boomeramg',
                'pc_hypre_boomeramg_cycle_type': h.get('cycleType', 'V'),
                'pc_hypre_boomeramg_coarsen_type': h.get('coarsenType', 'HMIS'),
                'pc_hypre_boomeramg_strong_threshold': float(h.get('strongThreshold', 0.7)),
                'pc_hypre_boomeramg_relax_type_all': h.get('relaxType', 'symmetric-SOR/Jacobi'),
                'pc_hypre_boomeramg_grid_sweeps_all': int(h.get('numSweeps', 1)),
                'pc_hypre_boomeramg_interp_type': h.get('interpType', 'classical'),
                'pc_hypre_boomeramg_max_levels': int(h.get('maxLevels', 25)),
            }
            petsc_bddc_dict['dirichlet_options'] = hypre_options
            petsc_bddc_dict['neumann_options'] = dict(hypre_options)

        config['petsc_bddc'] = petsc_bddc_dict

        # Write back with YAML formatting
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return jsonify({'success': True, 'message': 'PETSc BDDC config updated'})
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


def create_config_for_mesh(mesh_name, base_config=None):
    """Create a config file for a mesh by copying an existing one and updating mesh paths.

    If base_config is None, uses the current config as template.
    Returns the new config filename.
    """
    import yaml

    new_config_name = f'input_{mesh_name}.yml'
    new_config_path = PROJECT_ROOT / new_config_name

    # Use base config or current config as template
    if base_config:
        template_path = PROJECT_ROOT / base_config
    else:
        template_path = PROJECT_ROOT / mesh_state.get('currentConfig', 'input_pepe36_colored.yml')

    if not template_path.exists():
        return None

    with open(template_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Update mesh-specific fields
    config['mesh_file'] = f'data/{mesh_name}.xdmf'
    config['tags_dictionary_file'] = f'data/{mesh_name}.pickle'
    config['out_name'] = f'{mesh_name}_sim'

    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return new_config_name

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

    # Set config file - use provided one, find matching one, or auto-create
    if config_file:
        mesh_state['currentConfig'] = config_file
    else:
        found_config = find_config_for_mesh(mesh_name)
        if found_config:
            mesh_state['currentConfig'] = found_config
        else:
            # Auto-create config from current template
            new_config = create_config_for_mesh(mesh_name)
            if new_config:
                mesh_state['currentConfig'] = new_config

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
        docker_image = 'dolfinx-ginkgo:bddc'
        # For Ginkgo, build Python bindings only if not already built or if source changed
        setup_cmd = 'cd dolfinx-ginkgo && if [ ! -f build/_cpp*.so ] || [ python/dolfinx_ginkgo/_cpp.cpp -nt build/_cpp*.so ]; then rm -rf build && mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/usr/local/dolfinx-real -DDOLFINX_GINKGO_BUILD_PYTHON=ON && make -j2; else echo "Ginkgo bindings up to date"; fi && cd /home/fenics && '
    else:
        docker_image = 'ghcr.io/fenics/dolfinx/dolfinx:v0.9.0'
        setup_cmd = ''

    def generate():
        if simulation_state['running']:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Simulation already running'})}\n\n"
            return

        simulation_state['running'] = True

        # Clean up old interface files to prevent stale data from previous runs
        for old_if in PROJECT_ROOT.glob('IF_*.txt'):
            old_if.unlink()

        docker_cmd = [
            'docker', 'run', '--rm', '-t',
            '-v', f'{PROJECT_ROOT}:/home/fenics',
            '-w', '/home/fenics',
            docker_image,
            'bash', '-c',
            f'{setup_cmd}pip install -q pymetis && pip install --no-build-isolation -q -r requirements.txt && mpirun -n {ranks} python3 -B -u main.py {config_file}'
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

    # Look for directories containing _sim that have v.h5
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and '_sim' in item.name:
            v_h5 = item / 'v.h5'
            if v_h5.exists():
                # Check if viz data exists
                viz_data_dir = Path(__file__).parent / 'data' / item.name
                has_viz = (viz_data_dir / 'mesh_metadata.json').exists()

                # Pull solver/mesh info from conditions.json for compact labels
                solver_info = {}
                cond_file = item / 'conditions.json'
                if cond_file.exists():
                    try:
                        with open(cond_file, 'r') as f:
                            cond = json.load(f)
                        solver_info = {
                            'mesh': cond.get('mesh'),
                            'solver': cond.get('solver'),
                            'preconditioner': cond.get('preconditioner'),
                            'localSolver': cond.get('localSolver'),
                            'nRanks': cond.get('nRanks'),
                        }
                    except Exception:
                        pass

                simulations.append({
                    'name': item.name,
                    'path': str(item),
                    'has_viz_data': has_viz,
                    'size': sum(f.stat().st_size for f in item.glob('*.h5')),
                    **solver_info,
                })

    return jsonify({
        'simulations': sorted(simulations, key=lambda x: x['name'])
    })

@app.route('/api/simulations/with-iterations')
def list_simulations_with_iterations():
    """List simulation directories that have iterations.pickle (for comparison)."""
    simulations = []
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and '_sim' in item.name:
            iters_file = item / 'iterations.pickle'
            if iters_file.exists():
                cond_file = item / 'conditions.json'
                conditions_hash = None
                physical_hash = None
                solver_info = {}
                if cond_file.exists():
                    try:
                        with open(cond_file, 'r') as f:
                            cond = json.load(f)
                        conditions_hash = cond.get('_hash')
                        solver_info = {
                            'solver': cond.get('solver'),
                            'preconditioner': cond.get('preconditioner'),
                            'localSolver': cond.get('localSolver'),
                            'nRanks': cond.get('nRanks'),
                        }
                        # Physical conditions hash (same keys used in JS warning)
                        phys = {k: cond.get(k) for k in (
                            'mesh', 'boundingBox', 'vExcited', 'vResting',
                            'scarEnabled', 'scarBox', 'scarMargin', 'scarConductivities'
                        ) if cond.get(k) is not None}
                        physical_hash = json.dumps(phys, sort_keys=True)
                    except Exception:
                        pass
                simulations.append({
                    'name': item.name,
                    'conditions_hash': conditions_hash,
                    'physical_hash': physical_hash,
                    **solver_info,
                })
    return jsonify({
        'simulations': sorted(simulations, key=lambda x: x['name'])
    })

# --------------------- Viz Generation API ---------------------

@app.route('/api/generate-viz', methods=['POST'])
def generate_viz_endpoint():
    """Generate visualization data from simulation output asynchronously with SSE progress."""
    data = request.json
    sim_name = data.get('dir')
    if not sim_name:
        return jsonify({'error': 'No simulation directory specified'}), 400

    sim_output_dir = PROJECT_ROOT / sim_name
    if not sim_output_dir.exists():
        return jsonify({'error': f'Simulation output not found: {sim_name}'}), 404

    mesh_data_dir = Path(__file__).parent / 'data' / Path(sim_name).name

    def generate():
        with viz_gen_state['lock']:
            if viz_gen_state['generating']:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Viz generation already in progress'})}\n\n"
                return
            viz_gen_state['generating'] = True
            viz_gen_state['sim_name'] = sim_name
            viz_gen_state['progress'] = 0
            viz_gen_state['message'] = 'Starting...'
            viz_gen_state['done'] = False
            viz_gen_state['error'] = None

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
            from generate_viz_from_output import generate_viz_data

            # Use a list to collect progress events from the worker thread
            progress_queue = []
            progress_event = threading.Event()

            def progress_callback(percent, message):
                viz_gen_state['progress'] = percent
                viz_gen_state['message'] = message
                progress_queue.append({'type': 'progress', 'percent': percent, 'message': message})
                progress_event.set()

            # Run generation in a background thread
            result = {'error': None}
            def worker():
                try:
                    generate_viz_data(sim_output_dir, mesh_data_dir, progress_callback=progress_callback)
                except Exception as e:
                    import traceback
                    result['error'] = str(e)
                    result['traceback'] = traceback.format_exc()
                finally:
                    progress_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # Stream progress events to client
            while thread.is_alive():
                progress_event.wait(timeout=2)
                progress_event.clear()
                while progress_queue:
                    event = progress_queue.pop(0)
                    yield f"data: {json.dumps(event)}\n\n"

            # Drain remaining events
            while progress_queue:
                event = progress_queue.pop(0)
                yield f"data: {json.dumps(event)}\n\n"

            if result['error']:
                yield f"data: {json.dumps({'type': 'error', 'message': result['error'], 'traceback': result.get('traceback', '')})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'complete', 'success': True})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

        finally:
            with viz_gen_state['lock']:
                viz_gen_state['generating'] = False
                viz_gen_state['done'] = True

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/generate-viz/status')
def generate_viz_status():
    """Check current viz generation status."""
    return jsonify({
        'generating': viz_gen_state['generating'],
        'sim_name': viz_gen_state['sim_name'],
        'progress': viz_gen_state['progress'],
        'message': viz_gen_state['message'],
    })

# --------------------- Results API ---------------------

@app.route('/api/results')
def get_results():
    """Load simulation results from HDF5 file with per-facet voltage mapping."""
    output_dir = request.args.get('dir', 'pepe36_colored_sim')

    sim_output_dir = PROJECT_ROOT / output_dir

    # Use simulation output directory name as viz data source
    sim_name = Path(output_dir).name
    mesh_data_dir = Path(__file__).parent / 'data' / sim_name
    viz_mesh_path = mesh_data_dir / 'mesh_vertices.bin'
    metadata_path = mesh_data_dir / 'mesh_metadata.json'

    # Allow loading from viz data even if sim output dir doesn't exist locally
    # (e.g. viz data downloaded from Karolina without full results)
    if not viz_mesh_path.exists():
        if not sim_output_dir.exists():
            return jsonify({'error': f'Simulation output not found: {output_dir}'}), 404
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

        use_per_facet = len(vij_files) > 0 and facet_orig_vertices is not None

        def parse_time(key):
            return float(key.replace('_', '.'))

        # Build voltage data and save as binary files for efficient serving
        voltages_dir = mesh_data_dir / 'voltages'
        voltages_dir.mkdir(parents=True, exist_ok=True)

        # If voltage binaries already exist (e.g. downloaded from Karolina), use them directly
        existing_bins = sorted(voltages_dir.glob('*.bin'))
        if existing_bins:
            # Use times/vMin/vMax from metadata if available (saved by generate_viz_from_output)
            times = metadata.get('times')
            v_min = metadata.get('vMin')
            v_max = metadata.get('vMax')

            if times is None or v_min is None:
                # Fallback: scan binaries
                times = []
                v_min = float('inf')
                v_max = float('-inf')
                for ti, bin_path in enumerate(sorted(existing_bins)):
                    v_data = np.fromfile(bin_path, dtype=np.float32)
                    v_min = min(v_min, float(np.min(v_data)))
                    v_max = max(v_max, float(np.max(v_data)))
                    dt = metadata.get('dt', 0.001)
                    times.append(ti * dt)
                if v_min == float('inf'):
                    v_min, v_max = 0.0, 0.0

            # Load iterations/residuals if available
            iterations_path = sim_output_dir / 'iterations.pickle'
            iterations = None
            if iterations_path.exists():
                import pickle
                with open(iterations_path, 'rb') as f:
                    iterations = pickle.load(f)
            residuals_path = sim_output_dir / 'residuals.pickle'
            residuals = None
            if residuals_path.exists():
                import pickle
                with open(residuals_path, 'rb') as f:
                    residuals = pickle.load(f)

            # Load scar config
            scar_config = None
            conditions_path = sim_output_dir / 'conditions.json'
            if conditions_path.exists():
                with open(conditions_path) as f:
                    cond = json.load(f)
                    scar_config = cond.get('scar')

            # Rank metadata for partition view (same as full path below)
            rank_metadata_path = mesh_data_dir / 'rank_metadata.json'
            num_ranks = None
            rank_centroids = None
            global_centroid = None
            has_rank_data = (mesh_data_dir / 'dof_ranks.bin').exists()
            if has_rank_data and rank_metadata_path.exists():
                with open(rank_metadata_path, 'r') as f:
                    rank_meta = json.load(f)
                    num_ranks = rank_meta.get('num_ranks')
                    rank_centroids = rank_meta.get('rank_centroids')
                    global_centroid = rank_meta.get('global_centroid')

            return jsonify({
                'times': times,
                'vMin': v_min,
                'vMax': v_max,
                'vizDataDir': sim_name,
                'iterations': iterations or [],
                'residuals': residuals,
                'scarConfig': scar_config,
                'hasRankData': has_rank_data,
                'numRanks': num_ranks,
                'rankCentroids': rank_centroids,
                'globalCentroid': global_centroid,
                'hasEcsRanks': (mesh_data_dir / 'ecs_ranks.bin').exists(),
                'hasCutRanks': (mesh_data_dir / 'cut_ranks.bin').exists(),
                'hasDofIndices': (mesh_data_dir / 'facet_orig_vertices.bin').exists(),
                'hasEcsDofIndices': (mesh_data_dir / 'ecs_orig_vertices.bin').exists(),
            })

        if use_per_facet:
            first_vij_path = list(vij_files.values())[0]
            with h5py.File(first_vij_path, 'r') as f:
                func_name = list(f['Function'].keys())[0]
                v_group = f['Function'][func_name]
                timestep_keys = sorted(v_group.keys(), key=parse_time)

            max_timesteps = 100
            if len(timestep_keys) > max_timesteps:
                step = len(timestep_keys) // max_timesteps
                timestep_keys = timestep_keys[::step][:max_timesteps]

            # Load vij data for all pairs and timesteps
            vij_data = {}
            for pair, vij_path in vij_files.items():
                with h5py.File(vij_path, 'r') as f:
                    func_name = list(f['Function'].keys())[0]
                    v_group = f['Function'][func_name]
                    vij_data[pair] = {}
                    for key in timestep_keys:
                        if key in v_group:
                            v_arr = v_group[key][:].flatten()
                            vij_data[pair][key] = np.nan_to_num(v_arr, nan=0.0, posinf=0.0, neginf=0.0)

            num_facets = len(facet_orig_vertices)
            times = []
            v_min = float('inf')
            v_max = float('-inf')

            for ti, key in enumerate(timestep_keys):
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

                expanded_voltages.tofile(voltages_dir / f'{ti}.bin')
                v_min = min(v_min, float(np.min(expanded_voltages)))
                v_max = max(v_max, float(np.max(expanded_voltages)))
                times.append(parse_time(key))

        else:
            h5_path = sim_output_dir / 'v.h5'
            if not h5_path.exists():
                return jsonify({'error': f'v.h5 not found in {output_dir}'}), 404

            with h5py.File(h5_path, 'r') as f:
                v_group = f['Function']['v']
                timestep_keys = sorted(v_group.keys(), key=parse_time)

                max_timesteps = 100
                if len(timestep_keys) > max_timesteps:
                    step = len(timestep_keys) // max_timesteps
                    timestep_keys = timestep_keys[::step][:max_timesteps]

                times = []
                v_min = float('inf')
                v_max = float('-inf')
                for ti, key in enumerate(timestep_keys):
                    v_data = v_group[key][:].flatten()
                    v_data = np.nan_to_num(v_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    v_data.tofile(voltages_dir / f'{ti}.bin')
                    v_min = min(v_min, float(np.min(v_data)))
                    v_max = max(v_max, float(np.max(v_data)))
                    times.append(parse_time(key))

        if v_min == float('inf'):
            v_min = 0.0
            v_max = 0.0

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

        # Load rank metadata (small JSON only - binary data served separately)
        rank_metadata_path = mesh_data_dir / 'rank_metadata.json'
        num_ranks = None
        rank_centroids = None
        global_centroid = None
        has_rank_data = (mesh_data_dir / 'dof_ranks.bin').exists()

        if has_rank_data and rank_metadata_path.exists():
            with open(rank_metadata_path, 'r') as f:
                rank_meta = json.load(f)
                num_ranks = rank_meta.get('num_ranks')
                rank_centroids = rank_meta.get('rank_centroids')
                global_centroid = rank_meta.get('global_centroid')

        # Find scar config from the YAML config that produced this simulation
        scar_config = None
        import yaml as _yaml
        for yml_path in PROJECT_ROOT.glob('input_*.yml'):
            try:
                with open(yml_path, 'r') as f:
                    cfg = _yaml.safe_load(f) or {}
                if cfg.get('out_name') == sim_name and 'scar_config' in cfg:
                    scar_config = cfg['scar_config']
                    break
            except Exception:
                pass

        return jsonify({
            'times': times,
            'vMin': v_min,
            'vMax': v_max,
            'numTimesteps': len(timestep_keys),
            'vizDataDir': sim_name,
            'perFacet': use_per_facet,
            'iterations': iterations,
            'residuals': residuals,
            'hasRankData': has_rank_data,
            'numRanks': num_ranks,
            'rankCentroids': rank_centroids,
            'globalCentroid': global_centroid,
            'hasEcsRanks': (mesh_data_dir / 'ecs_ranks.bin').exists(),
            'hasCutRanks': (mesh_data_dir / 'cut_ranks.bin').exists(),
            'hasDofIndices': facet_orig_vertices is not None,
            'hasEcsDofIndices': (mesh_data_dir / 'ecs_orig_vertices.bin').exists(),
            'scarConfig': scar_config,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/results/iterations/<sim_name>')
def get_iterations_data(sim_name):
    """Fetch iterations and conditions data for a simulation (for comparison plots)."""
    import pickle
    sim_dir = PROJECT_ROOT / sim_name
    if not sim_dir.is_dir():
        return jsonify({'error': f'Simulation not found: {sim_name}'}), 404

    result = {'sim_name': sim_name}

    iterations_path = sim_dir / 'iterations.pickle'
    if iterations_path.exists():
        with open(iterations_path, 'rb') as f:
            result['iterations'] = pickle.load(f)

    residuals_path = sim_dir / 'residuals.pickle'
    if residuals_path.exists():
        with open(residuals_path, 'rb') as f:
            result['residuals'] = pickle.load(f)

    conditions_path = sim_dir / 'conditions.json'
    if conditions_path.exists():
        with open(conditions_path, 'r') as f:
            result['conditions'] = json.load(f)

    return jsonify(result)

@app.route('/api/results/binary/<sim_name>/<filename>')
def get_results_binary(sim_name, filename):
    """Serve binary data files (voltages, ranks, dof indices) for a simulation."""
    # Validate filename to prevent path traversal
    allowed_files = {
        'dof_ranks.bin', 'ecs_ranks.bin', 'cut_ranks.bin',
        'facet_orig_vertices.bin', 'ecs_orig_vertices.bin',
    }
    # Also allow voltage timestep files: 0.bin, 1.bin, ...
    is_voltage = filename.endswith('.bin') and filename[:-4].isdigit()

    if filename not in allowed_files and not is_voltage:
        return jsonify({'error': 'Invalid file'}), 400

    if is_voltage:
        file_path = Path(__file__).parent / 'data' / sim_name / 'voltages' / filename
    else:
        file_path = Path(__file__).parent / 'data' / sim_name / filename

    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(str(file_path.parent), file_path.name,
                               mimetype='application/octet-stream')


# --------------------- Cross-section API ---------------------

def _is_timestep_filename(name):
    return name.endswith('.bin') and name[:-4].isdigit()


def _resolve_cross_section_path(sim_name, filepath):
    """Validate and resolve a cross-section binary path.

    Allows: cells/<tag>_(vertices|facets|orig_verts).bin,
            phi_i/<tag>/<ti>.bin,
            phi_e/<ti>.bin,
            ecs_volume/(vertices|tets|orig_verts).bin,
            cap/(vertices|facets).bin,
            cap/phi_e/<ti>.bin.
    Returns the absolute Path or None.
    """
    parts = filepath.split('/')
    base = Path(__file__).parent / 'data' / sim_name

    if any(p in ('', '.', '..') for p in parts):
        return None

    if len(parts) == 2 and parts[0] == 'cells':
        fname = parts[1]
        if not fname.endswith('.bin'):
            return None
        stem = fname[:-4]
        if '_' not in stem:
            return None
        tag_part, suffix = stem.split('_', 1)
        if not tag_part.lstrip('-').isdigit():
            return None
        if suffix not in ('vertices', 'facets', 'orig_verts'):
            return None
        return base / 'cells' / fname

    if len(parts) == 3 and parts[0] == 'phi_i':
        tag_part = parts[1]
        if not tag_part.lstrip('-').isdigit():
            return None
        if not _is_timestep_filename(parts[2]):
            return None
        return base / 'phi_i' / tag_part / parts[2]

    if len(parts) == 2 and parts[0] == 'phi_e':
        if not _is_timestep_filename(parts[1]):
            return None
        return base / 'phi_e' / parts[1]

    if len(parts) == 2 and parts[0] == 'phi_e_shell':
        if not _is_timestep_filename(parts[1]):
            return None
        return base / 'phi_e_shell' / parts[1]

    if len(parts) == 2 and parts[0] == 'ecs_volume':
        if parts[1] not in ('vertices.bin', 'tets.bin', 'orig_verts.bin'):
            return None
        return base / 'ecs_volume' / parts[1]

    if len(parts) == 2 and parts[0] == 'cap':
        if parts[1] not in ('vertices.bin', 'facets.bin'):
            return None
        return base / 'cap' / parts[1]

    if len(parts) == 3 and parts[0] == 'cap' and parts[1] == 'phi_e':
        if not _is_timestep_filename(parts[2]):
            return None
        return base / 'cap' / 'phi_e' / parts[2]

    return None


@app.route('/api/cross-section/binary/<sim_name>/<path:filepath>')
def get_cross_section_binary(sim_name, filepath):
    """Serve cross-section binary files (cells/, phi_i/, phi_e/, ecs_volume/, cap/)."""
    file_path = _resolve_cross_section_path(sim_name, filepath)
    if file_path is None:
        return jsonify({'error': 'Invalid path'}), 400
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(str(file_path.parent), file_path.name,
                               mimetype='application/octet-stream')


def _slice_ecs_volume(ecs_vertices, ecs_tets, normal, offset):
    """Compute the cap triangulation of the plane n·x = offset cutting the ECS tets.

    Returns dict with cap geometry plus per-cap-vertex interpolation weights
    (edge endpoints in ECS volume vertex indices + t in [0,1]) so that
    cap value = (1-t) * f(a) + t * f(b).
    Returns None if the plane misses every tet.
    """
    normal = np.asarray(normal, dtype=np.float64)
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-12:
        return None
    normal = normal / norm_mag
    offset = float(offset)

    d = (ecs_vertices.astype(np.float64) @ normal) - offset
    tet_dists = d[ecs_tets]
    pos_mask = tet_dists > 0.0
    n_pos = pos_mask.sum(axis=1)

    cap_edges = []   # list of (a, b, t), a < b
    cap_facets = []
    edge_cache = {}

    def edge_index(va, vb, da, db):
        if va == vb:
            return None
        a, b = (va, vb) if va < vb else (vb, va)
        key = (a, b)
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        # t such that (1-t)*pos_a + t*pos_b is on the plane, with t referring to b
        # Solve (1-t)*da + t*db = 0 → t = da / (da - db)
        denom = (da - db) if a == va else (db - da)
        da_eff = da if a == va else db
        if denom == 0.0:
            t = 0.0
        else:
            t = da_eff / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        idx = len(cap_edges)
        cap_edges.append((int(a), int(b), float(t)))
        edge_cache[key] = idx
        return idx

    n_tets = ecs_tets.shape[0]
    for ti in range(n_tets):
        np_pos = n_pos[ti]
        if np_pos == 0 or np_pos == 4:
            continue
        verts = ecs_tets[ti]
        dists = tet_dists[ti]
        positives = [i for i in range(4) if dists[i] > 0.0]
        negatives = [i for i in range(4) if dists[i] <= 0.0]

        if np_pos == 1:
            p = positives[0]
            n0, n1, n2 = negatives
            i0 = edge_index(int(verts[p]), int(verts[n0]), dists[p], dists[n0])
            i1 = edge_index(int(verts[p]), int(verts[n1]), dists[p], dists[n1])
            i2 = edge_index(int(verts[p]), int(verts[n2]), dists[p], dists[n2])
            cap_facets.append((i0, i1, i2))
        elif np_pos == 3:
            n = negatives[0]
            p0, p1, p2 = positives
            i0 = edge_index(int(verts[n]), int(verts[p0]), dists[n], dists[p0])
            i1 = edge_index(int(verts[n]), int(verts[p1]), dists[n], dists[p1])
            i2 = edge_index(int(verts[n]), int(verts[p2]), dists[n], dists[p2])
            cap_facets.append((i0, i1, i2))
        else:  # np_pos == 2 → quad → two triangles
            p0, p1 = positives
            n0, n1 = negatives
            i00 = edge_index(int(verts[p0]), int(verts[n0]), dists[p0], dists[n0])
            i01 = edge_index(int(verts[p0]), int(verts[n1]), dists[p0], dists[n1])
            i11 = edge_index(int(verts[p1]), int(verts[n1]), dists[p1], dists[n1])
            i10 = edge_index(int(verts[p1]), int(verts[n0]), dists[p1], dists[n0])
            cap_facets.append((i00, i01, i11))
            cap_facets.append((i00, i11, i10))

    if not cap_edges:
        return None

    edges = np.array(cap_edges, dtype=np.float64)
    a_idx = edges[:, 0].astype(np.int64)
    b_idx = edges[:, 1].astype(np.int64)
    t = edges[:, 2].astype(np.float32)
    cap_vertices = (
        ecs_vertices[a_idx].astype(np.float32) * (1.0 - t[:, None])
        + ecs_vertices[b_idx].astype(np.float32) * t[:, None]
    ).astype(np.float32)
    cap_facets_arr = np.array(cap_facets, dtype=np.uint32)
    return {
        'vertices': cap_vertices,
        'facets': cap_facets_arr,
        'weights_a': a_idx.astype(np.int64),
        'weights_b': b_idx.astype(np.int64),
        'weights_t': t,
    }


@app.route('/api/cross-section/slice', methods=['POST'])
def slice_cross_section():
    """Compute the ECS cap for a user-configured plane and pre-render per-timestep
    φ_e interpolated onto the cap. Plane is n · x = offset (normal+offset in the
    same coordinate frame as the visualization mesh)."""
    data = request.get_json() or {}
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'sim_name is required'}), 400
    normal = data.get('normal')
    offset = data.get('offset')
    if normal is None or offset is None:
        return jsonify({'error': 'normal and offset are required'}), 400
    try:
        normal_arr = [float(x) for x in normal]
        if len(normal_arr) != 3:
            raise ValueError
        offset_val = float(offset)
    except (TypeError, ValueError):
        return jsonify({'error': 'normal must be 3 floats, offset must be float'}), 400

    sim_dir = Path(__file__).parent / 'data' / sim_name
    ecs_vert_file = sim_dir / 'ecs_volume' / 'vertices.bin'
    ecs_tets_file = sim_dir / 'ecs_volume' / 'tets.bin'
    if not ecs_vert_file.exists() or not ecs_tets_file.exists():
        return jsonify({'error': 'ECS volume data not available for this simulation'}), 404

    ecs_vertices = np.fromfile(ecs_vert_file, dtype=np.float32).reshape(-1, 3)
    ecs_tets = np.fromfile(ecs_tets_file, dtype=np.uint32).reshape(-1, 4)

    cap = _slice_ecs_volume(ecs_vertices, ecs_tets, normal_arr, offset_val)
    if cap is None:
        return jsonify({'error': 'Plane does not intersect ECS volume'}), 400

    cap_dir = sim_dir / 'cap'
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / 'phi_e').mkdir(parents=True, exist_ok=True)
    cap['vertices'].tofile(cap_dir / 'vertices.bin')
    cap['facets'].tofile(cap_dir / 'facets.bin')

    phi_e_dir = sim_dir / 'phi_e'
    timesteps_done = 0
    phi_e_min = float('inf')
    phi_e_max = float('-inf')
    skip_reason = None
    if not phi_e_dir.exists():
        skip_reason = f"phi_e dir not found: {phi_e_dir}"
    else:
        weights_a = cap['weights_a']
        weights_b = cap['weights_b']
        weights_t = cap['weights_t']
        max_idx = max(int(weights_a.max()), int(weights_b.max())) if len(weights_a) else 0
        ti = 0
        while True:
            src = phi_e_dir / f'{ti}.bin'
            if not src.exists():
                if ti == 0:
                    skip_reason = f"phi_e/0.bin missing in {phi_e_dir}"
                break
            phi_e_vol = np.fromfile(src, dtype=np.float32)
            if max_idx >= len(phi_e_vol):
                skip_reason = (
                    f"index mismatch at ti={ti}: cap weight max_idx={max_idx} "
                    f">= len(phi_e/{ti}.bin)={len(phi_e_vol)} "
                    f"(re-run generate_viz_from_output.py to regenerate ECS volume + φ_e in sync)"
                )
                break
            cap_phi = (
                phi_e_vol[weights_a] * (1.0 - weights_t)
                + phi_e_vol[weights_b] * weights_t
            ).astype(np.float32)
            cap_phi.tofile(cap_dir / 'phi_e' / f'{ti}.bin')
            if cap_phi.size:
                phi_e_min = min(phi_e_min, float(cap_phi.min()))
                phi_e_max = max(phi_e_max, float(cap_phi.max()))
            ti += 1
            timesteps_done += 1

    if phi_e_min == float('inf'):
        phi_e_min, phi_e_max = 0.0, 0.0

    return jsonify({
        'cap_vertex_count': int(len(cap['vertices'])),
        'cap_facet_count': int(len(cap['facets'])),
        'num_timesteps': int(timesteps_done),
        'phi_e_range': [phi_e_min, phi_e_max],
        'normal': normal_arr,
        'offset': offset_val,
        'warning': skip_reason,
    })

# --------------------- Interface Data API ---------------------

@app.route('/api/interfaces')
def get_interfaces():
    """Load BDDC interface data from IF_*.txt files and map to mesh vertices."""
    import pickle

    interfaces = {}

    # Find all IF_*.txt files in project root
    for if_file in PROJECT_ROOT.glob('IF_*.txt'):
        try:
            rank = int(if_file.stem.split('_')[1])
            with open(if_file, 'r') as f:
                content = f.read().strip()
                if content:
                    # Each line is one interface (space-separated DOF indices)
                    rank_interfaces = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            indices = [int(x) for x in line.split()]
                            if indices:
                                rank_interfaces.append(indices)
                    interfaces[rank] = rank_interfaces
        except (ValueError, IOError) as e:
            print(f"Warning: Could not parse {if_file}: {e}")
            continue

    if not interfaces:
        return jsonify({'interfaces': {}, 'numRanks': 0, 'message': 'No interface files found'})

    num_ranks = max(interfaces.keys()) + 1

    # Load matrix-to-vertex mapping from the most recent simulation
    # The mapping depends on MPI rank count (rank-major DOF ordering), so it must
    # come from the same simulation run that produced the IF_*.txt files.
    # Use the most recently modified *_sim directory to match the current IF files.
    matrix_to_vertex = None
    sim_dirs = sorted(
        [d for d in PROJECT_ROOT.glob('*_sim*') if d.is_dir() and (d / 'matrix_to_vertex.pickle').exists()],
        key=lambda d: (d / 'matrix_to_vertex.pickle').stat().st_mtime,
        reverse=True
    )
    if sim_dirs:
        mapping_file = sim_dirs[0] / 'matrix_to_vertex.pickle'
        try:
            with open(mapping_file, 'rb') as f:
                matrix_to_vertex = pickle.load(f)
            print(f"Loaded matrix-to-vertex mapping from {mapping_file} ({len(matrix_to_vertex)} entries)")
        except Exception as e:
            print(f"Warning: Could not load {mapping_file}: {e}")

    # Convert interface DOF indices to mesh vertex indices
    interface_vertices = {}
    all_interface_vertices = set()

    # Track DOF classifications for BDDC interface types:
    # - vertex: single-DOF interfaces (size 1)
    # - face: DOFs shared between exactly 2 partitions
    # - edge: DOFs in interfaces of size > 1 shared between 3+ partitions
    dof_to_ranks = {}  # DOF -> set of ranks that have this DOF
    dofs_in_size1_interfaces = set()  # DOFs that appear in any size-1 interface

    if matrix_to_vertex:
        # First pass: collect DOF sharing info and identify size-1 interfaces
        for rank, rank_interfaces in interfaces.items():
            for interface in rank_interfaces:
                interface_size = len(interface)
                for dof in interface:
                    if dof in matrix_to_vertex:
                        vertex = matrix_to_vertex[dof]
                        if vertex not in dof_to_ranks:
                            dof_to_ranks[vertex] = set()
                        dof_to_ranks[vertex].add(rank)
                        if interface_size == 1:
                            dofs_in_size1_interfaces.add(vertex)

        # Classify each DOF based on number of sharing partitions:
        # - face: shared by exactly 2 partitions (regardless of interface size)
        # - vertex: shared by 3+ partitions AND appears in a size-1 interface
        # - edge: shared by 3+ partitions AND only in multi-DOF interfaces
        dof_types = {}  # vertex_index -> 'vertex' | 'edge' | 'face'
        for vertex, ranks in dof_to_ranks.items():
            num_ranks = len(ranks)
            if num_ranks == 2:
                dof_types[vertex] = 'face'
            elif vertex in dofs_in_size1_interfaces:
                dof_types[vertex] = 'vertex'
            else:
                dof_types[vertex] = 'edge'

        # Second pass: build interface_vertices structure
        for rank, rank_interfaces in interfaces.items():
            rank_vertex_interfaces = []
            for interface in rank_interfaces:
                vertex_indices = []
                for dof in interface:
                    if dof in matrix_to_vertex:
                        vertex_indices.append(matrix_to_vertex[dof])
                if vertex_indices:
                    rank_vertex_interfaces.append(vertex_indices)
                    all_interface_vertices.update(vertex_indices)
            interface_vertices[rank] = rank_vertex_interfaces

        # Count DOFs by type
        type_counts = {'vertex': 0, 'edge': 0, 'face': 0}
        for dof_type in dof_types.values():
            type_counts[dof_type] += 1
        print(f"Interface DOF types: {type_counts['vertex']} vertices, {type_counts['edge']} edges, {type_counts['face']} faces")
    else:
        # No mapping available - return empty
        print("Warning: No matrix_to_vertex.pickle found - interface visualization won't work")
        dof_types = {}

    return jsonify({
        'interfaces': interface_vertices,  # Now contains mesh vertex indices
        'numRanks': num_ranks,
        'allInterfaceVertices': sorted(list(all_interface_vertices)),
        'totalInterfaces': sum(len(v) for v in interface_vertices.values()),
        'hasMappingFile': matrix_to_vertex is not None,
        'dofTypes': dof_types  # vertex_index -> 'vertex' | 'edge' | 'face'
    })

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

# ----- Client-side frame capture video export -----

import uuid
import shutil
import tempfile

_capture_sessions = {}

@app.route('/api/video/start-capture', methods=['POST'])
def start_capture():
    """Start a new frame capture session. Returns a session_id."""
    data = request.json or {}
    session_id = uuid.uuid4().hex[:12]
    frames_dir = Path(tempfile.mkdtemp(prefix=f'video_{session_id}_'))
    _capture_sessions[session_id] = {
        'frames_dir': frames_dir,
        'fps': data.get('fps', 30),
        'frame_count': 0
    }
    return jsonify({'session_id': session_id})


@app.route('/api/video/frame/<session_id>', methods=['POST'])
def receive_frame(session_id):
    """Receive a single JPEG frame for a capture session."""
    session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Invalid session'}), 404

    frame_num = session['frame_count']
    frame_path = session['frames_dir'] / f'frame_{frame_num:06d}.jpg'
    frame_path.write_bytes(request.data)
    session['frame_count'] = frame_num + 1
    return jsonify({'ok': True, 'frame': frame_num})


@app.route('/api/video/finish-capture/<session_id>', methods=['POST'])
def finish_capture(session_id):
    """Encode captured frames into MP4 using ffmpeg."""
    session = _capture_sessions.pop(session_id, None)
    if not session:
        return jsonify({'error': 'Invalid session'}), 404

    frames_dir = session['frames_dir']
    fps = session['fps']

    try:
        videos_dir = Path(__file__).parent / 'videos'
        videos_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'simulation_{timestamp}.mp4'
        video_path = videos_dir / filename

        # Use ffmpeg to encode frames
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(frames_dir / 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'medium',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            return jsonify({'error': f'ffmpeg failed: {result.stderr[-500:]}'}), 500

        return jsonify({'filename': filename})
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


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

# --------------------- Karolina API ---------------------

try:
    from karolina import (
        karolina_state, karolina_jobs, mesh_convert_state,
        check_ssh, check_containers, upload_config, submit_job, submit_jobs,
        check_job_status, cancel_job, delete_job_data, tail_remote_log,
        get_cached_status,
        download_results_streaming, download_iterations,
        list_remote_simulations,
        list_remote_meshes, fetch_mesh_metadata,
        convert_remote_mesh, finish_conversion,
        download_mesh_data,
        generate_remote_video, check_video_job, download_video,
        generate_remote_viz, check_viz_job, download_viz_data_streaming
    )
except ImportError:
    from viz.karolina import (
        karolina_state, karolina_jobs, mesh_convert_state,
        check_ssh, check_containers, upload_config, submit_job, submit_jobs,
        check_job_status, cancel_job, delete_job_data, tail_remote_log,
        get_cached_status,
        download_results_streaming, download_iterations,
        list_remote_simulations,
        list_remote_meshes, fetch_mesh_metadata,
        convert_remote_mesh, finish_conversion,
        download_mesh_data,
        generate_remote_video, check_video_job, download_video,
        generate_remote_viz, check_viz_job, download_viz_data_streaming
    )

@app.route('/api/karolina/check')
def karolina_check():
    """Test SSH connectivity to Karolina and check container availability."""
    available = check_ssh()
    containers = check_containers() if available else {}
    return jsonify({'available': available, 'containers': containers})

@app.route('/api/karolina/submit', methods=['POST'])
def karolina_submit():
    """Upload config and submit SLURM job(s) to Karolina.

    Accepts 'nodes' as a single int or comma-separated string (e.g. "1,2,4")
    to submit multiple jobs with different node counts in one go.
    """
    data = request.json
    config_file = data.get('config', 'input_pepe36_colored.yml')
    nodes_raw = data.get('nodes', 1)
    ntasks_per_node = data.get('ntasks_per_node', 128)
    walltime = data.get('walltime', '01:00:00')
    partition = data.get('partition', 'qcpu_exp')
    account = data.get('account', 'eu-26-11')
    solver_backend = data.get('solver_backend', 'petsc')
    conditions = data.get('conditions')

    # Parse node counts: single int or comma-separated string
    if isinstance(nodes_raw, str):
        node_counts = [int(n.strip()) for n in nodes_raw.split(',') if n.strip()]
    elif isinstance(nodes_raw, list):
        node_counts = [int(n) for n in nodes_raw]
    else:
        node_counts = [int(nodes_raw)]

    try:
        config_path = PROJECT_ROOT / config_file
        if not config_path.exists():
            return jsonify({'error': f'Config file not found: {config_file}'}), 404

        if len(node_counts) == 1:
            # Single job: use original path
            job_info = submit_job(
                config_file, node_counts[0], ntasks_per_node,
                walltime, partition, account,
                solver_backend=solver_backend,
                conditions=conditions,
                local_config_path=str(config_path)
            )
            return jsonify({
                'success': True,
                'jobs': [job_info],
                **job_info,
                'message': f'Job {job_info["job_id"]} submitted to Karolina'
            })
        else:
            # Multiple node counts: batch submit
            job_infos = submit_jobs(
                config_file, node_counts, ntasks_per_node,
                walltime, partition, account,
                solver_backend=solver_backend,
                conditions=conditions,
                local_config_path=str(config_path)
            )
            job_ids = [j['job_id'] for j in job_infos]
            return jsonify({
                'success': True,
                'jobs': job_infos,
                # Legacy compat: expose first job at top level
                **(job_infos[0] if job_infos else {}),
                'message': f'{len(job_infos)} jobs submitted: {", ".join(job_ids)}'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/jobs')
def karolina_list_jobs():
    """List all tracked Karolina jobs."""
    return jsonify({'jobs': list(karolina_jobs.values())})

@app.route('/api/karolina/status')
@app.route('/api/karolina/status/<job_id>')
def karolina_status(job_id=None):
    """Poll SLURM job status and tail log output."""
    if job_id is None:
        job_id = request.args.get('job_id') or karolina_state.get('job_id')
    if not job_id:
        return jsonify({
            'status': None,
            'job_id': None,
            'log': '',
            'message': 'No job submitted'
        })

    job = karolina_jobs.get(job_id, {})

    # Use cached status from background poller (non-blocking)
    cached_status, cached_log = get_cached_status(job_id)
    if cached_status is not None:
        return jsonify({
            'job_id': job_id,
            'status': cached_status,
            'out_name': job.get('out_name', karolina_state.get('out_name', '')),
            'conditions_hash': job.get('conditions_hash'),
            'log': cached_log or ''
        })

    # Fallback for jobs not yet in cache (e.g. from before poller started)
    try:
        status = check_job_status(job_id)
        log = tail_remote_log(job_id)

        return jsonify({
            'job_id': job_id,
            'status': status,
            'out_name': job.get('out_name', karolina_state.get('out_name', '')),
            'conditions_hash': job.get('conditions_hash'),
            'log': log
        })
    except Exception as e:
        return jsonify({
            'job_id': job_id,
            'status': job.get('status', karolina_state.get('status', 'UNKNOWN')),
            'log': '',
            'error': str(e)
        })

@app.route('/api/karolina/cancel', methods=['POST'])
def karolina_cancel():
    """Cancel a SLURM job."""
    data = request.json or {}
    job_id = data.get('job_id') or karolina_state.get('job_id')
    if not job_id:
        return jsonify({'error': 'No job to cancel'}), 400

    try:
        cancel_job(job_id)
        return jsonify({'success': True, 'message': f'Job {job_id} cancelled'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/delete', methods=['POST'])
def karolina_delete():
    """Cancel a job (if running), then remove its remote dir and local download."""
    data = request.json or {}
    job_id = data.get('job_id')
    out_name = data.get('out_name')
    if not job_id and not out_name:
        return jsonify({'error': 'job_id or out_name required'}), 400

    try:
        errors = delete_job_data(job_id, out_name, str(PROJECT_ROOT))
        return jsonify({'success': True, 'errors': errors})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/download-iterations', methods=['POST'])
def karolina_download_iterations():
    """Download just iterations.pickle + conditions.json from a remote simulation."""
    data = request.json
    remote_dir = data.get('remote_dir')
    if not remote_dir:
        return jsonify({'error': 'No remote directory specified'}), 400

    local_dest = PROJECT_ROOT / remote_dir
    try:
        download_iterations(remote_dir, local_dest)
        return jsonify({'success': True, 'out_name': remote_dir})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/download', methods=['POST'])
def karolina_download():
    """Download simulation results from Karolina via SCP (SSE stream with byte progress)."""
    data = request.json
    remote_dir = data.get('remote_dir') or karolina_state.get('out_name')

    if not remote_dir:
        return jsonify({'error': 'No remote directory specified'}), 400

    local_dest = PROJECT_ROOT / remote_dir

    def generate():
        for status in download_results_streaming(remote_dir, local_dest):
            yield f"data: {json.dumps(status)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/karolina/remote-simulations')
def karolina_remote_simulations():
    """List simulation output directories on Karolina."""
    try:
        dirs = list_remote_simulations()
        return jsonify({'simulations': dirs})
    except Exception as e:
        return jsonify({'error': str(e), 'simulations': []}), 500

@app.route('/api/karolina/meshes')
def karolina_meshes():
    """List mesh families available on Karolina."""
    try:
        families = list_remote_meshes()
        return jsonify({'families': families})
    except Exception as e:
        return jsonify({'error': str(e), 'families': []}), 500

@app.route('/api/karolina/meshes/metadata/<mesh_name>')
def karolina_mesh_metadata(mesh_name):
    """Fetch mesh bounding box and metadata from Karolina without downloading."""
    try:
        metadata = fetch_mesh_metadata(mesh_name)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/meshes/convert', methods=['POST'])
def karolina_convert_mesh():
    """Convert a remote mesh on Karolina inside the DOLFINx container (SSE stream)."""
    data = request.json
    family = data.get('family')
    pts_file = data.get('pts')
    elem_file = data.get('elem')
    output_prefix = data.get('output_prefix')
    color = data.get('color', False)

    if not all([family, pts_file, elem_file, output_prefix]):
        return jsonify({'error': 'Missing required fields'}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'output', 'text': f'Starting conversion of {output_prefix} on Karolina...\\n'})}\n\n"
            if color:
                yield f"data: {json.dumps({'type': 'output', 'text': 'Graph coloring enabled (--color-intracellular)\\n'})}\n\n"

            process = convert_remote_mesh(family, pts_file, elem_file, output_prefix, color)

            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            process.wait()
            finish_conversion()

            if process.returncode == 0:
                yield f"data: {json.dumps({'type': 'complete', 'success': True})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Conversion failed with exit code {process.returncode}'})}\n\n"

        except Exception as e:
            finish_conversion()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/karolina/meshes/download', methods=['POST'])
def karolina_download_mesh():
    """Download converted mesh data from Karolina to local data/ directory."""
    data = request.json
    mesh_name = data.get('mesh_name')

    if not mesh_name:
        return jsonify({'error': 'No mesh name specified'}), 400

    try:
        local_data_dir = PROJECT_ROOT / 'data'
        download_mesh_data(mesh_name, local_data_dir)
        return jsonify({
            'success': True,
            'message': f'Downloaded {mesh_name} mesh data to data/'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------- Remote Video API ---------------------

# In-memory tracking for remote video jobs
remote_video_jobs = {}

@app.route('/api/karolina/video/generate', methods=['POST'])
def karolina_generate_video():
    """Submit a video generation job on Karolina."""
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400

    try:
        result = generate_remote_video(
            sim_name,
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            fps=data.get('fps', 30),
            camera_config=data.get('camera'),
            colormap=data.get('colormap', 'coolwarm'),
            partition=data.get('partition', 'qcpu_exp'),
            account=data.get('account', 'eu-26-11'),
        )
        remote_video_jobs[result['job_id']] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/video/status/<job_id>')
def karolina_video_status(job_id):
    """Check status of a remote video generation job."""
    job = remote_video_jobs.get(job_id, {})
    log_file = job.get('log_file')
    try:
        result = check_video_job(job_id, log_file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/video/download/<job_id>', methods=['POST'])
def karolina_download_video(job_id):
    """Download a generated video from Karolina."""
    job = remote_video_jobs.get(job_id, {})
    log_file = job.get('log_file')

    # Get video filename from job log
    try:
        status = check_video_job(job_id, log_file)
        video_filename = status.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'Video file not found in job output'}), 404

        local_dir = PROJECT_ROOT / 'viz' / 'videos'
        local_path = download_video(video_filename, local_dir)
        return jsonify({
            'success': True,
            'filename': video_filename,
            'download_url': f'/api/video/download/{video_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------- Remote Viz Data API ---------------------

remote_viz_jobs = {}

@app.route('/api/karolina/viz/generate', methods=['POST'])
def karolina_generate_viz():
    """Submit a viz data generation job on Karolina."""
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400
    try:
        result = generate_remote_viz(sim_name)
        remote_viz_jobs[result['job_id']] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/viz/status/<job_id>')
def karolina_viz_status(job_id):
    """Check status of a remote viz generation job."""
    job = remote_viz_jobs.get(job_id, {})
    try:
        result = check_viz_job(job_id, job.get('log_file'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/karolina/viz/download', methods=['POST'])
def karolina_download_viz():
    """Download viz data from Karolina as streaming SSE."""
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400

    local_dest = PROJECT_ROOT / 'viz' / 'data' / sim_name

    def stream():
        for event in download_viz_data_streaming(sim_name, local_dest):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(stream(), mimetype='text/event-stream')

# --------------------- Main ---------------------

if __name__ == '__main__':
    print("=" * 50)
    print("Cardiac EMI Visualization Server")
    print("=" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Open http://localhost:8000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
