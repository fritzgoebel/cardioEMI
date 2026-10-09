#!/usr/bin/env python3
"""
Flask server for the Cardiac EMI Visualization Tool.
Handles:
- Serving static files
- Updating YAML config
- Running Docker simulation with streaming output
"""

import os
import collections
import json
import pickle
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
                # Same as essential_defaults above: utils.py treats a missing
                # save_output as False, so without it the first run from a
                # freshly viewed cluster mesh wrote no results at all.
                'save_output': True,
                'save_interval': 10,
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
                'reordering': bddc_config.get('reordering', 'none'),
                'local_max_iterations': int(bddc_config.get('localMaxIterations', 100)),
                'local_tolerance': float(bddc_config.get('localTolerance', 1e-12)),
                'coarse_solver': bddc_config.get('coarseSolver', 'cg'),
                'coarse_max_iterations': int(bddc_config.get('coarseMaxIterations', 100)),
                'coarse_bddc_local_solver': bddc_config.get('coarseBddcLocalSolver', 'direct'),
                'vertices': bddc_config.get('vertices', True),
                'edges': bddc_config.get('edges', True),
                'faces': bddc_config.get('faces', True),
                'repartition_coarse': True,  # always on, see main.py
                'distributed_coarse': bddc_config.get('distributedCoarse', False),
                'write_interfaces': bddc_config.get('writeInterfaces', True),
                'unanimous_connectivity': bddc_config.get('unanimousConnectivity', True)
            }
            # Inner (interior A_II) solver: only emit when explicitly chosen;
            # omitting it makes the inner solve reuse local_solver (Ginkgo default).
            inner_solver = bddc_config.get('innerSolver')
            if inner_solver:
                bddc_dict['inner_solver'] = inner_solver
                bddc_dict['inner_max_iterations'] = int(bddc_config.get('innerMaxIterations', 100))
                bddc_dict['inner_tolerance'] = float(bddc_config.get('innerTolerance', 1e-12))
                inner_amg_config = bddc_config.get('innerAmg', {})
                if inner_amg_config:
                    bddc_dict['inner_amg'] = {
                        'coarsening': inner_amg_config.get('coarsening', 'pgm'),
                        'strength_threshold': float(inner_amg_config.get('strengthThreshold', 0.25)),
                        'cycle': inner_amg_config.get('cycle', 'v'),
                        'smoother': inner_amg_config.get('smoother', 'jacobi'),
                        'smooth_steps': int(inner_amg_config.get('smoothSteps', 1)),
                        'max_levels': int(inner_amg_config.get('maxLevels', 10)),
                        'coarse_solver': inner_amg_config.get('coarseSolver', 'direct'),
                        'relaxation_factor': float(inner_amg_config.get('relaxationFactor', 0.9))
                    }
                inner_hypre_config = bddc_config.get('innerHypre', {})
                if inner_hypre_config:
                    bddc_dict['inner_hypre'] = {
                        'cycle_type': int(inner_hypre_config.get('cycleType', 1)),
                        'coarsening_type': int(inner_hypre_config.get('coarseningType', 10)),
                        'strength_threshold': float(inner_hypre_config.get('strengthThreshold', 0.25)),
                        'smoother_type': int(inner_hypre_config.get('smootherType', 6)),
                        'num_sweeps': int(inner_hypre_config.get('numSweeps', 1)),
                        'interpolation_type': int(inner_hypre_config.get('interpolationType', 0)),
                        'max_levels': int(inner_hypre_config.get('maxLevels', 25)),
                        'coarse_smoother_type': int(inner_hypre_config.get('coarseSmootherType', 9)),
                        'relax_order': int(inner_hypre_config.get('relaxOrder', 1)),
                        'max_coarse_size': int(inner_hypre_config.get('maxCoarseSize', 64)),
                        'print_level': int(inner_hypre_config.get('printLevel', 1))
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
                    'max_levels': int(local_hypre_config.get('maxLevels', 25)),
                    'coarse_smoother_type': int(local_hypre_config.get('coarseSmootherType', 9)),
                    'relax_order': int(local_hypre_config.get('relaxOrder', 1)),
                    'max_coarse_size': int(local_hypre_config.get('maxCoarseSize', 64)),
                    'print_level': int(local_hypre_config.get('printLevel', 1))
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

def get_mesh_tag_counts(mesh_name):
    """Tag/rank-target counts for 'Tag based' (component) partitioning.

    `numTags` is read straight from the mesh's own tags_dictionary_file. A
    `_colored` mesh's own file only has the ~4 coloring tags, so the numbers
    that matter for partitioning instead come from the original uncolored
    mesh's tag pickle (tags come in even/odd ECS+cell pairs - see
    mesh_partition.py):
      - `numOriginalTags`: one rank per individual tag (component_granularity
        "tag" - e.g. to match a BDDC convergence theory stated per volume tag)
      - `numComponents`: one rank per ECS+cell pair (component_granularity
        "component", the default - numOriginalTags // 2)

    Both are left None for a non-colored mesh: main.py's "component"
    partition_mode derives `original_mesh_file` by stripping "_colored" off
    `mesh_file`, so it only works out of the box on a colored mesh - there is
    nothing to point rank-matching at otherwise.
    """
    data_dir = PROJECT_ROOT / 'data'

    def _tag_count(name):
        p = data_dir / f'{name}.pickle'
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            return len(pickle.load(f))

    num_tags = _tag_count(mesh_name)
    num_original_tags = None
    num_components = None
    if mesh_name.endswith('_colored'):
        num_original_tags = _tag_count(mesh_name[:-len('_colored')])
        num_components = num_original_tags // 2 if num_original_tags is not None else None

    return {
        'numTags': num_tags,
        'numOriginalTags': num_original_tags,
        'numComponents': num_components,
    }


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
            'configFile': config_file,
            **get_mesh_tag_counts(name)
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
        'configFile': mesh_state['currentConfig'],
        **get_mesh_tag_counts(mesh_name)
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
        'metadata': metadata,
        **get_mesh_tag_counts(mesh_name)
    })

# --------------------- Weak-scaling meshes ---------------------

# Canonical name: <plus|cell>_<nx>x<ny>x<nz>_n<n>_L<L>[_p<pad>][_a<ax>]
# (L: '.' -> 'p'; the optional suffixes are omitted at their defaults, so legacy
# unpadded cubic 'plus' names still parse). 'cell' is the myocyte-like shape,
# whose boxes are stretched to ax*L in x; 'plus' is the original cubic geometry.
WS_NAME_RE = re.compile(
    r'^(plus|cell)_(\d+)x(\d+)x(\d+)_n(\d+)_L([0-9p]+?)(?:_p(\d+))?'
    r'(?:_a(\d+))?(?:_s(\d+)t(\d+)(?:u(\d+))?)?'
    r'(?:_d(\d+)(?:z(\d+))?)?(?:_r(\d+))?(?:_g(\d+))?$')


def _generator_reason(lines):
    """Pull the actionable line out of the mesh generator's output.

    Its failures are raised as ValueError/RuntimeError with a message naming the
    parameter to change, so the last exception line is what the user needs; a
    bare traceback frame is not.
    """
    text = [l.strip() for l in lines if l.strip()]
    for line in reversed(text):
        for marker in ('ValueError:', 'RuntimeError:', 'Error:'):
            if marker in line:
                return line.split(marker, 1)[1].strip() or line
    # No exception line (killed, or failed without raising): the last line of
    # output is still more use than the exit code.
    return text[-1] if text else None


def resolve_ws_shift(ax, d_y, d_z, max_slabs=24):
    """Common denominator and numerators for the two lateral shifts.

    Mirrors CellShape.resolve: each direction's shift is Lx - 2d, and both have
    to be expressed over one denominator because a box's offset is
    (J*step_y + K*step_z)/slabs of the box length.
    """
    want = [(ax - 2 * d) / ax for d in (d_y, d_z)]
    best, best_err = (2, 1, 1), float('inf')
    for q in range(2, max_slabs + 1):
        ps = [min(q - 1, max(1, int(round(w * q)))) for w in want]
        err = max(abs(w - p / q) for w, p in zip(want, ps))
        if err < best_err - 1e-12:
            best, best_err = (q, ps[0], ps[1]), err
    return best


def weak_scaling_name(nx, ny, nz, n, L, pad=0, shape='cell', ax=1, slabs=0,
                      step_y=1, step_z=1, d_y=0.0, d_z=0.0, lat_r=0.0, lean=0.0):
    Ls = ('%g' % L).replace('.', 'p')
    name = f'{shape}_{nx}x{ny}x{nz}_n{n}_L{Ls}'
    if pad:
        name += f'_p{pad}'
    if shape != 'plus' and ax != 1:
        name += f'_a{ax}'
    if shape != 'plus' and slabs:
        name += f'_s{slabs}t{step_y}'
        if step_z != step_y:
            name += f'u{step_z}'
    if shape != 'plus' and (d_y or d_z):
        name += f'_d{int(round((d_y or d_z) * 100))}'
        if d_z and d_z != d_y:
            name += f'z{int(round(d_z * 100))}'
    if shape != 'plus' and lat_r:
        name += f'_r{int(round(lat_r * 100))}'
    if shape != 'plus' and lean:
        name += f'_g{int(round(lean))}'
    return name


def parse_weak_scaling_name(name):
    m = WS_NAME_RE.match(name)
    if not m:
        return None
    def num(i, scale=1.0, dflt=0.0):
        return int(m.group(i)) / scale if m.group(i) else dflt
    return {
        'name': name, 'shape': m.group(1),
        'nx': int(m.group(2)), 'ny': int(m.group(3)), 'nz': int(m.group(4)),
        'n': int(m.group(5)), 'L': float(m.group(6).replace('p', '.')),
        'pad': int(num(7, 1.0, 0)), 'ax': int(num(8, 1.0, 1)),
        'slabs': int(num(9, 1.0, 0)), 'step_y': int(num(10, 1.0, 1)),
        'step_z': int(num(11, 1.0, 0)) or int(num(10, 1.0, 1)),
        'd_y': num(12, 100.0), 'd_z': num(13, 100.0),
        'lat_r': num(14, 100.0), 'lean': num(15, 1.0),
    }


def validate_weak_scaling(nx, ny, nz, n, L, pad, shape, ax, slabs=0):
    """Return an error string, or None if the parameters are generatable."""
    if shape not in ('cell', 'plus'):
        return "shape must be 'cell' or 'plus'"
    if min(nx, ny, nz) < 1:
        return 'Require nx, ny, nz >= 1'
    if L <= 0 or pad < 0:
        return 'Require L > 0 and pad >= 0'
    if shape == 'plus':
        if n < 4 or n % 4 != 0:
            return 'The plus shape needs n a multiple of 4 (>= 4)'
    else:
        if n < 4:
            return 'The cell shape needs n >= 4 elements per L'
        if ax < 2:
            # Both the near-face and far-face connectors must fit along x.
            return 'The cell shape needs an x aspect of at least 2'
        if slabs and slabs < 2:
            return 'Slabs per box must be >= 2 (the lattice shift is Lx/slabs)'
    return None


@app.route('/api/weak-scaling/list')
def weak_scaling_list():
    """List already-generated weak-scaling (3D-plus) meshes."""
    data_dir = PROJECT_ROOT / 'data'
    viz_data_dir = Path(__file__).parent / 'data'

    items = []
    seen = set()
    for pattern in ('plus_*_n*_L*.h5', 'cell_*_n*_L*.h5'):
        for h5_file in sorted(data_dir.glob(pattern)):
            info = parse_weak_scaling_name(h5_file.stem)
            if not info or h5_file.stem in seen:
                continue
            seen.add(h5_file.stem)
            info['converted'] = (viz_data_dir / h5_file.stem / 'mesh_metadata.json').exists()
            info['size'] = h5_file.stat().st_size
            items.append(info)

    return jsonify({
        'meshes': sorted(items, key=lambda i: i['name']),
        'defaults': {'n': 12, 'L': 25.0, 'pad': 0, 'shape': 'cell', 'ax': 4,
                     'slabs': 2},
    })


@app.route('/api/weak-scaling/generate', methods=['POST'])
def weak_scaling_generate():
    """Generate (or reuse) a weak-scaling mesh, then convert it for the viewer.

    Streams SSE progress. On completion the frontend selects the mesh.
    """
    data = request.json or {}
    try:
        nx = int(data['nx']); ny = int(data['ny']); nz = int(data['nz'])
        n = int(data.get('n', 12)); L = float(data.get('L', 25.0))
        pad = int(data.get('pad', 0))
        shape = str(data.get('shape', 'cell'))
        ax = int(data.get('ax', 4)) if shape != 'plus' else 1
        cell = shape != 'plus'
        d_y = float(data.get('d_y', 0.5) or 0.5) if cell else 0.0
        d_z = float(data.get('d_z', 0) or 0) if cell else 0.0
        lean = float(data.get('lean', 55) or 55) if cell else 0.0
        lat_r = float(data.get('lat_r', 0.26) or 0.26) if cell else 0.0
        slabs = step_y = step_z = 0
        if cell:
            slabs, step_y, step_z = resolve_ws_shift(ax, d_y, d_z or d_y)
            d_y = 0.5 * (ax - ax * step_y / slabs)
            d_z = 0.5 * (ax - ax * step_z / slabs)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    err = validate_weak_scaling(nx, ny, nz, n, L, pad, shape, ax, slabs)
    if err:
        return jsonify({'error': err}), 400

    name = weak_scaling_name(nx, ny, nz, n, L, pad, shape, ax, slabs,
                             step_y, step_z, d_y, d_z, lat_r, lean)

    def generate():
        if mesh_state['converting']:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Another mesh operation is in progress'})}\n\n"
            return

        mesh_state['converting'] = True
        try:
            import sys
            h5_path = PROJECT_ROOT / 'data' / f'{name}.h5'
            prefix = PROJECT_ROOT / 'data' / name

            # --- 1. Generate mesh if it does not already exist ---
            if h5_path.exists():
                yield f"data: {json.dumps({'type': 'progress', 'percent': 55, 'message': f'Reusing existing mesh {name}'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'progress', 'percent': 5, 'message': f'Generating {name} ({nx}x{ny}x{nz} cubes)...'})}\n\n"
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / 'geometry' / 'generate_weak_scaling_mesh.py'),
                    '--nx', str(nx), '--ny', str(ny), '--nz', str(nz),
                    '--n', str(n), '--L', str(L), '--pad', str(pad),
                    '--shape', shape, '--ax', str(ax), '--no-preview',
                    '--dist', str(d_y), '--dist-z', str(d_z),
                    '--lean', str(lean), '--lat-r', str(lat_r),
                    '--slabs', str(slabs), '--prefix', str(prefix),
                ]
                proc = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                tail = collections.deque(maxlen=40)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        tail.append(line)
                        yield f"data: {json.dumps({'type': 'progress', 'percent': 40, 'message': line})}\n\n"
                proc.wait()
                if proc.returncode != 0 or not h5_path.exists():
                    # "exit 1" on its own tells the user nothing, and the
                    # generator's own messages say exactly which knob to move --
                    # so hand the reason back rather than the return code.
                    why = _generator_reason(tail)
                    yield f"data: {json.dumps({'type': 'error', 'message': why or f'Mesh generation failed (exit {proc.returncode})'})}\n\n"
                    return

            # --- 2. Convert for the viewer (skip if already done) ---
            output_dir = Path(__file__).parent / 'data' / name
            if (output_dir / 'mesh_metadata.json').exists():
                yield f"data: {json.dumps({'type': 'progress', 'percent': 90, 'message': 'Visualization already converted'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'progress', 'percent': 70, 'message': 'Converting for visualization...'})}\n\n"
                sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
                from convert_hdf5 import convert_mesh
                convert_mesh(h5_path, output_dir)

            # --- 3. Ensure a matching config exists ---
            cfg = find_config_for_mesh(name)
            if not cfg:
                base = f'input_{shape}_weak_scaling.yml'
                base = base if (PROJECT_ROOT / base).exists() else None
                cfg = create_config_for_mesh(name, base_config=base)

            yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Ready!'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'name': name, 'configFile': cfg})}\n\n"

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
            'X-Accel-Buffering': 'no',
        },
    )

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
    sim_out_name = None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        solver_backend = config.get('solver_backend', 'petsc').lower()
        sim_out_name = (config.get('out_name') or '').strip().lstrip('_') or None
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

        # Clean up stray IF files at the project root (older runs left them there).
        # Per-sim IF files now live inside each simulation's output dir.
        for old_if in PROJECT_ROOT.glob('IF_*.txt'):
            try:
                old_if.unlink()
            except OSError:
                pass

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

            # Move any IF_*.txt files produced by the BDDC solver into the
            # simulation output dir so each run keeps its own partition data.
            if sim_out_name:
                target_dir = PROJECT_ROOT / sim_out_name
                if target_dir.is_dir():
                    for if_file in PROJECT_ROOT.glob('IF_*.txt'):
                        try:
                            if_file.replace(target_dir / if_file.name)
                        except OSError as exc:
                            print(f"Warning: could not move {if_file} into {target_dir}: {exc}")

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

# --------------------- Run / category names (Compare runs) ---------------------

try:
    import run_index
except ImportError:
    from viz import run_index

# User-assigned run names, virtual folders and heading names (viz/run_labels.json);
# see run_index.py for the format.
_load_run_labels = run_index.load_labels
_save_run_labels = run_index.save_labels
_forget_run_labels = run_index.forget_runs


@app.route('/api/simulations/labels', methods=['GET'])
def get_run_labels():
    """Every user-assigned run and category name."""
    return jsonify(_load_run_labels())


@app.route('/api/simulations/labels', methods=['POST'])
def set_run_labels():
    """Merge in renamed runs / categories; an empty value restores the default.

    Body: {"runs":       {"<sim>": {"label": str|null}},
           "categories": {"<kind>:<key>": str|null}}
    Folders are changed through /api/runs/move and /api/runs/folder.
    """
    data = request.json or {}
    labels = _load_run_labels()

    for name, patch in (data.get('runs') or {}).items():
        if not isinstance(patch, dict):
            continue
        entry = dict(labels['runs'].get(name) or {})
        for field in ('label',):
            if field not in patch:
                continue
            value = (patch[field] or '').strip()
            if value:
                entry[field] = value
            else:
                entry.pop(field, None)
        if entry:
            labels['runs'][name] = entry
        else:
            labels['runs'].pop(name, None)

    for key, value in (data.get('categories') or {}).items():
        value = (value or '').strip()
        if value:
            labels['categories'][key] = value
        else:
            labels['categories'].pop(key, None)

    _save_run_labels(labels)
    return jsonify(labels)


# --------------------- Runs browser (virtual folders) ---------------------

@app.route('/api/runs')
def list_runs():
    """Every run, local and on every cluster, with its folder and label.

    ?refresh=1 re-lists every reachable cluster (one ssh each, in parallel);
    without it the clusters' last listings come from the on-disk cache, so the
    tree renders instantly; a cluster whose last refresh failed shows as stale.
    """
    errors = run_index.refresh_remote(cluster_registry) if request.args.get('refresh') else {}
    return jsonify(run_index.build_index(PROJECT_ROOT, cluster_registry, errors,
                                         parse_ws_name=parse_weak_scaling_name))


@app.route('/api/runs/move', methods=['POST'])
def move_runs():
    """File runs into a folder. Body: {"names": [...], "folder": "a/b"} ('' = unfiled)."""
    data = request.json or {}
    names = [n for n in data.get('names') or [] if run_index.is_run_name(n)]
    try:
        run_index.file_runs(names, data.get('folder') or '')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'success': True})


@app.route('/api/runs/folder', methods=['POST'])
def edit_folder():
    """Body: {"op": "create", "path"} or {"op": "move", "path", "new_path"}.
    "move" covers renaming and dragging a folder into another one."""
    data = request.json or {}
    try:
        if data.get('op') == 'create':
            run_index.create_folder(data.get('path'))
        elif data.get('op') == 'move':
            run_index.rename_folder(data.get('path'), data.get('new_path'))
        else:
            return jsonify({'error': 'op must be create or move'}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'success': True})


@app.route('/api/runs/delete', methods=['POST'])
def delete_runs():
    """Delete runs everywhere: local folder, viz cache, every cluster copy, label.

    Body: {"names": [...], "folders": [...], "only_if_empty": bool,
           "copies": [{"cluster": id, "name": run}]}
    - folders: every run filed in them (recursively) is deleted, then the
      folders themselves.
    - only_if_empty: leave anything holding iterations or results - checked
      where the data lives, so a stale index can't cause a wrong delete.
    - copies: empty cluster-side copies of runs that have data elsewhere (old
      sync artifacts); only that copy goes, and only if it is still empty.
    Runs with a queued or running SLURM job are skipped.
    """
    data = request.json or {}
    only_if_empty = bool(data.get('only_if_empty'))
    folders = data.get('folders') or []
    index = run_index.build_index(PROJECT_ROOT, cluster_registry)
    by_name = {r['name']: r for r in index['runs']}

    names = {n for n in data.get('names') or [] if run_index.is_run_name(n)}
    names |= set(run_index.runs_in_folders(folders, list(by_name)))
    skipped = sorted(n for n in names if by_name.get(n, {}).get('active'))
    names -= set(skipped)
    touched = {by_name[n]['folder'] for n in names if n in by_name and by_name[n]['folder']}

    removed = set()
    errors = []
    local_removed, local_errors = run_index.delete_local(
        PROJECT_ROOT, [n for n in names if n in by_name and by_name[n]['local']], only_if_empty)
    removed |= set(local_removed)
    errors += local_errors

    targets = {}  # cluster id -> [(name, only_if_empty)]
    for n in names:
        for cid in (by_name.get(n) or {}).get('remote', {}):
            targets.setdefault(cid, []).append((n, only_if_empty))
    for copy in data.get('copies') or []:
        n, cid = copy.get('name'), copy.get('cluster')
        if run_index.is_run_name(n) and cid and n not in skipped:
            targets.setdefault(cid, []).append((n, True))

    copies_removed = []
    for cid, items in targets.items():
        cl = _get_cluster(cid)
        if cl is None:
            continue
        done = []
        for guarded in (False, True):
            batch = [n for n, g in items if g == guarded]
            if not batch:
                continue
            try:
                got, err = cl.delete_runs(batch, only_if_empty=guarded)
            except Exception as e:
                got, err = [], str(e)
            done += got
            if err:
                errors.append(f'{cl.label}: {err}')
        run_index.forget_remote(cid, done)
        removed |= {n for n in done if n in names}
        copies_removed += [f'{cid}:{n}' for n in done if n not in names]
        for job_id in [j for j, info in cl.jobs.items() if info.get('out_name') in done]:
            cl.jobs.pop(job_id, None)

    # A name only disappears from the labels once no copy of it is left.
    after = {r['name'] for r in run_index.build_index(PROJECT_ROOT, cluster_registry)['runs']}
    run_index.forget_runs([n for n in removed if n not in after])
    if folders:
        run_index.drop_folders(folders)
    # A folder whose last run was just deleted goes too (with emptied parents).
    pruned = run_index.prune_emptied_folders(touched, after)
    return jsonify({'removed': sorted(removed), 'copies_removed': copies_removed,
                    'skipped': skipped, 'errors': errors, 'folders_removed': pruned})


@app.route('/api/simulations/with-iterations')
def list_simulations_with_iterations():
    """List simulation directories that have iterations.pickle (for comparison)."""
    simulations = []
    run_labels = _load_run_labels()['runs']
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
                            'mesh': cond.get('mesh'),
                        }
                        # Physical conditions hash (same keys used in JS warning)
                        phys = {k: cond.get(k) for k in (
                            'mesh', 'boundingBox', 'vExcited', 'vResting',
                            'scarEnabled', 'scarBox', 'scarMargin', 'scarConductivities'
                        ) if cond.get(k) is not None}
                        physical_hash = json.dumps(phys, sort_keys=True)
                    except Exception:
                        pass
                named = run_labels.get(item.name) or {}

                # Weak-scaling geometry, for the "avg iterations vs. subdomains / H/h"
                # scaling plot: number of DD subdomains (2 per cube - cell + ECS
                # remainder) and H/h = voxels per cube edge (cube side / element
                # size), recovered from the plus_<nx>x<ny>x<nz>_n<n>_L<L> mesh name.
                mesh_name = solver_info.get('mesh')
                if not mesh_name:
                    m = re.match(r'^(.+?)_sim', item.name)
                    mesh_name = m.group(1) if m else None
                ws_geom = parse_weak_scaling_name(mesh_name) if mesh_name else None
                if ws_geom:
                    solver_info['nSubdomains'] = 2 * ws_geom['nx'] * ws_geom['ny'] * ws_geom['nz']
                    solver_info['hRatio'] = ws_geom['n']

                simulations.append({
                    'name': item.name,
                    'conditions_hash': conditions_hash,
                    'physical_hash': physical_hash,
                    'label': named.get('label'),
                    'folder': named.get('folder'),
                    **solver_info,
                })
    return jsonify({
        'simulations': sorted(simulations, key=lambda x: x['name'])
    })


@app.route('/api/simulations/delete_by_mesh', methods=['POST'])
def delete_simulations_by_mesh():
    """Delete every local *_sim* output for the given mesh, plus its viz/data caches."""
    import shutil
    data = request.json or {}
    mesh = (data.get('mesh') or '').strip()
    if not mesh or not re.fullmatch(r'[A-Za-z0-9_.\-]+', mesh):
        return jsonify({'error': 'invalid or missing mesh name'}), 400

    viz_data_root = Path(__file__).parent / 'data'
    removed = []
    errors = []
    for item in PROJECT_ROOT.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith(f'{mesh}_sim'):
            continue
        try:
            shutil.rmtree(item)
            removed.append(item.name)
        except OSError as exc:
            errors.append(f'{item.name}: {exc}')
        viz_cache = viz_data_root / item.name
        if viz_cache.exists():
            try:
                shutil.rmtree(viz_cache)
            except OSError as exc:
                errors.append(f'viz/data/{item.name}: {exc}')

    _forget_run_labels(removed)
    return jsonify({'removed': removed, 'errors': errors})


@app.route('/api/simulations/delete_by_names', methods=['POST'])
def delete_simulations_by_names():
    """Delete exactly the named local *_sim* outputs, plus their viz/data caches.

    Used by a compare-section group's "Delete all": a group can be a user-named
    collection rather than a whole mesh, so deleting by mesh would take runs the
    group does not show.
    """
    import shutil
    data = request.json or {}
    names = data.get('names') or []
    if not isinstance(names, list) or not names:
        return jsonify({'error': 'no simulation names given'}), 400

    viz_data_root = Path(__file__).parent / 'data'
    project_root = PROJECT_ROOT.resolve()
    removed = []
    errors = []
    for name in names:
        name = (name or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_.\-]+', name) or '_sim' not in name:
            errors.append(f'{name}: invalid simulation name')
            continue
        item = (PROJECT_ROOT / name).resolve()
        if item.parent != project_root or not item.is_dir():
            errors.append(f'{name}: not a simulation directory')
            continue
        try:
            shutil.rmtree(item)
            removed.append(name)
        except OSError as exc:
            errors.append(f'{name}: {exc}')
        viz_cache = viz_data_root / name
        if viz_cache.exists():
            try:
                shutil.rmtree(viz_cache)
            except OSError as exc:
                errors.append(f'viz/data/{name}: {exc}')

    _forget_run_labels(removed)
    return jsonify({'removed': removed, 'errors': errors})


@app.route('/api/simulations/delete_all', methods=['POST'])
def delete_all_simulations():
    """Wipe every local sim dir, the viz/data cache, and every remote sim dir."""
    import shutil
    removed_local = []
    errors = []

    # 1. Local sim dirs
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and '_sim' in item.name:
            try:
                shutil.rmtree(item)
                removed_local.append(item.name)
            except OSError as exc:
                errors.append(f'local {item.name}: {exc}')

    # 2. viz/data/* (regenerable cache)
    viz_data_root = Path(__file__).parent / 'data'
    viz_cleared = []
    if viz_data_root.exists():
        for child in viz_data_root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                viz_cleared.append(child.name)
            except OSError as exc:
                errors.append(f'viz/data/{child.name}: {exc}')

    # 3. Remote sim dirs on every configured cluster (best-effort)
    remote_cleared = False
    try:
        try:
            from cluster import registry as _registry
        except ImportError:
            from viz.cluster import registry as _registry
        for _cl in _registry.clusters():
            try:
                # Glob is built server-side; no user input flows into the SSH command.
                _, ssh_err, rc = _cl._run_ssh(
                    f'rm -rf -- {_cl.remote_path}/*_sim*',
                    timeout=60,
                )
                if rc == 0:
                    remote_cleared = True
                else:
                    errors.append(f'remote rm ({_cl.id}): {ssh_err}')
            except Exception as exc:
                errors.append(f'remote rm ({_cl.id}): {exc}')
    except Exception as exc:
        errors.append(f'remote rm: {exc}')

    _forget_run_labels(removed_local)
    if remote_cleared and run_index.CACHE_FILE.exists():
        run_index.CACHE_FILE.unlink()  # the cached listings name runs that are gone
    return jsonify({
        'removed_local': removed_local,
        'viz_cleared': viz_cleared,
        'remote_cleared': remote_cleared,
        'errors': errors,
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
    """Load BDDC interface data from IF_*.txt files and map to mesh vertices.

    Optional ?sim=<name> query parameter selects a specific simulation directory.
    When provided, IF_*.txt and matrix_to_vertex.pickle are read from there. When
    absent, falls back to the most recently modified _sim directory (and project
    root for stray IF files from older runs).
    """
    import pickle

    sim_name = request.args.get('sim') or None
    if sim_name and not re.fullmatch(r'[A-Za-z0-9_.\-]+', sim_name):
        return jsonify({'error': 'invalid sim name'}), 400

    # Resolve where to look for IF_*.txt files
    if sim_name:
        sim_dir = PROJECT_ROOT / sim_name
        if_search_paths = [sim_dir]
    else:
        # Backwards-compat: fall back to project root (older runs left files there)
        # plus the most recent sim dir.
        if_search_paths = [PROJECT_ROOT]
        sim_dir = None
        recent = sorted(
            [d for d in PROJECT_ROOT.glob('*_sim*') if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if recent:
            sim_dir = recent[0]
            if_search_paths.insert(0, sim_dir)

    interfaces = {}
    for base in if_search_paths:
        if not base.exists():
            continue
        for if_file in base.glob('IF_*.txt'):
            try:
                rank = int(if_file.stem.split('_')[1])
                if rank in interfaces:
                    continue  # already loaded from a higher-priority path
                with open(if_file, 'r') as f:
                    content = f.read().strip()
                    if content:
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
        if interfaces:
            break  # stop at first path with IF files

    if not interfaces:
        return jsonify({'interfaces': {}, 'numRanks': 0, 'message': 'No interface files found'})

    num_ranks = max(interfaces.keys()) + 1

    # Load matrix-to-vertex mapping from the selected (or most recent) sim dir.
    matrix_to_vertex = None
    if sim_dir is None:
        # Fallback discovery (e.g. when ?sim not given and we found IF at root)
        sim_dirs = sorted(
            [d for d in PROJECT_ROOT.glob('*_sim*') if d.is_dir() and (d / 'matrix_to_vertex.pickle').exists()],
            key=lambda d: (d / 'matrix_to_vertex.pickle').stat().st_mtime,
            reverse=True
        )
        if sim_dirs:
            sim_dir = sim_dirs[0]
    if sim_dir and (sim_dir / 'matrix_to_vertex.pickle').exists():
        mapping_file = sim_dir / 'matrix_to_vertex.pickle'
        try:
            with open(mapping_file, 'rb') as f:
                matrix_to_vertex = pickle.load(f)
            print(f"Loaded matrix-to-vertex mapping from {mapping_file} ({len(matrix_to_vertex)} entries)")
        except Exception as e:
            print(f"Warning: Could not load {mapping_file}: {e}")

    # Convert interface DOF indices to mesh vertex indices
    interface_vertices = {}
    interface_info = {}  # rank -> [{'ranks': [...], 'type': ...}, ...]
    all_interface_vertices = set()

    # Track BDDC interface classification keyed on the global MATRIX DOF (not the
    # mesh vertex): in EMI a single mesh vertex can host several matrix DOFs
    # (different function spaces), so keying on the vertex merges distinct primal
    # DOFs and mis-classifies them. IF_*.txt stores global matrix indices, so the
    # same index across ranks is the same DOF.
    # - face:   DOF shared by exactly 2 subdomains
    # - vertex: DOF shared by 3+ subdomains AND appearing as a size-1 (corner) interface
    # - edge:   DOF shared by 3+ subdomains only inside multi-DOF interfaces
    dof_to_ranks = {}  # matrix DOF -> set of ranks that have this DOF
    dofs_in_size1_interfaces = set()  # matrix DOFs that appear in any size-1 interface
    dof_types_by_vertex = {}  # mesh vertex -> 'vertex'|'edge'|'face' (for point styling)

    if matrix_to_vertex:
        # First pass: per-DOF sharing info + size-1 membership (over all DOFs,
        # whether or not they have a vertex mapping).
        for rank, rank_interfaces in interfaces.items():
            for interface in rank_interfaces:
                size1 = (len(interface) == 1)
                for dof in interface:
                    dof_to_ranks.setdefault(dof, set()).add(rank)
                    if size1:
                        dofs_in_size1_interfaces.add(dof)

        # Classify each matrix DOF.
        def _classify(dof):
            if len(dof_to_ranks[dof]) == 2:
                return 'face'
            return 'vertex' if dof in dofs_in_size1_interfaces else 'edge'
        dof_types = {dof: _classify(dof) for dof in dof_to_ranks}

        # Second pass: build interface_vertices (DOF->vertex, for 3D rendering)
        # and per-interface metadata. interface_info[rank][i] = {'ranks', 'type'}
        # aligned 1:1 with interface_vertices[rank][i]; the sharing rank-set and
        # type are the mode over the line's matrix DOFs (a clean BDDC equivalence
        # class is uniform; the mode tolerates stray DOFs).
        from collections import Counter
        for rank, rank_interfaces in interfaces.items():
            rank_vertex_interfaces = []
            rank_interface_info = []
            for interface in rank_interfaces:
                vertex_indices = [matrix_to_vertex[d] for d in interface if d in matrix_to_vertex]
                if not vertex_indices:
                    continue
                rank_vertex_interfaces.append(vertex_indices)
                all_interface_vertices.update(vertex_indices)
                shared_ranks = sorted(
                    Counter(frozenset(dof_to_ranks[d]) for d in interface).most_common(1)[0][0])
                itype = Counter(dof_types[d] for d in interface).most_common(1)[0][0]
                rank_interface_info.append({'ranks': shared_ranks, 'type': itype})
            interface_vertices[rank] = rank_vertex_interfaces
            interface_info[rank] = rank_interface_info

        # Project per-DOF types onto mesh vertices for the viewer's point styling
        # (a vertex is drawn as a corner if ANY of its DOFs is a vertex DOF):
        # precedence vertex > edge > face.
        _prec = {'vertex': 3, 'edge': 2, 'face': 1}
        for dof, t in dof_types.items():
            if dof in matrix_to_vertex:
                v = matrix_to_vertex[dof]
                if v not in dof_types_by_vertex or _prec[t] > _prec[dof_types_by_vertex[v]]:
                    dof_types_by_vertex[v] = t

        type_counts = {'vertex': 0, 'edge': 0, 'face': 0}
        for t in dof_types.values():
            type_counts[t] += 1
        print(f"Interface DOF types (per matrix DOF): {type_counts['vertex']} vertices, {type_counts['edge']} edges, {type_counts['face']} faces")
    else:
        # No mapping available - return empty
        print("Warning: No matrix_to_vertex.pickle found - interface visualization won't work")

    return jsonify({
        'interfaces': interface_vertices,  # Now contains mesh vertex indices
        'interfaceInfo': interface_info,   # per-interface {ranks, type}, aligned with interfaces
        'numRanks': num_ranks,
        'allInterfaceVertices': sorted(list(all_interface_vertices)),
        'totalInterfaces': sum(len(v) for v in interface_vertices.values()),
        'hasMappingFile': matrix_to_vertex is not None,
        'dofTypes': dof_types_by_vertex  # vertex_index -> 'vertex' | 'edge' | 'face'
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

# --------------------- Cluster API ---------------------
#
# Every remote-cluster operation is addressed as /api/cluster/<cid>/... where
# <cid> is an id from viz/clusters.yml. /api/clusters manages the registry.

try:
    from cluster import registry as cluster_registry, TERMINAL_STATES
except ImportError:
    from viz.cluster import registry as cluster_registry, TERMINAL_STATES


def _get_cluster(cluster_id):
    try:
        return cluster_registry.get(cluster_id)
    except KeyError:
        return None


@app.route('/api/clusters')
def clusters_list():
    """List configured clusters (with cheap local master-connection status)."""
    out = []
    for cid in cluster_registry.ids():
        try:
            cl = cluster_registry.get(cid)
            out.append(cl.to_dict(connected=cl.master_alive()))
        except Exception as e:
            out.append({'id': cid, 'error': str(e)})
    return jsonify({'clusters': out})


@app.route('/api/clusters', methods=['POST'])
def clusters_save():
    """Create or update a cluster entry in clusters.yml."""
    data = request.json or {}
    cluster_id = (data.get('id') or '').strip().lower()
    cfg = data.get('cfg') or {}
    try:
        cl = cluster_registry.save(cluster_id, cfg)
        return jsonify({'success': True, 'cluster': cl.to_dict()})
    except (ValueError, KeyError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clusters/<cluster_id>', methods=['DELETE'])
def clusters_delete(cluster_id):
    try:
        cluster_registry.delete(cluster_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/check')
def cluster_check(cluster_id):
    """Test SSH connectivity and check container availability."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    connected = cl.master_alive()
    available = connected or cl.check_ssh()
    containers = cl.check_containers() if available else {}
    return jsonify({
        'available': available,
        'connected': connected or available,
        'needs_otp': cl.needs_otp,
        'containers': containers,
        'label': cl.label,
    })


# --------------------- Interactive connect (OTP/2FA) ---------------------

@app.route('/api/cluster/<cluster_id>/connect', methods=['POST'])
def cluster_connect(cluster_id):
    """Start establishing the persistent (ControlMaster) connection."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    state = cl.connect_start()
    return jsonify(state)


@app.route('/api/cluster/<cluster_id>/connect/state')
def cluster_connect_state(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    return jsonify(cl.connect_state)


@app.route('/api/cluster/<cluster_id>/connect/cancel', methods=['POST'])
def cluster_connect_cancel(cluster_id):
    """Abandon a login in progress (the OTP window was closed)."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    return jsonify(cl.connect_cancel())


@app.route('/api/cluster/<cluster_id>/connect/input', methods=['POST'])
def cluster_connect_input(cluster_id):
    """Deliver the user's OTP/password answer to the waiting ssh."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json or {}
    state = cl.connect_send(data.get('text', ''))
    return jsonify(state)


@app.route('/api/cluster/<cluster_id>/disconnect', methods=['POST'])
def cluster_disconnect(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    cl.disconnect()
    return jsonify({'success': True})


@app.route('/api/cluster/<cluster_id>/install', methods=['POST'])
def cluster_install(cluster_id):
    """Set up the cluster (dirs, code sync, container SIFs) as an SSE stream."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404

    def generate():
        try:
            for event in cl.install_stream(PROJECT_ROOT, cluster_registry):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'Connection': 'keep-alive',
                             'X-Accel-Buffering': 'no'})


# --------------------- Jobs ---------------------

@app.route('/api/cluster/<cluster_id>/submit-batch', methods=['POST'])
def cluster_submit_batch(cluster_id):
    """Submit one SLURM job per (mesh, rank count) from one template config.

    Body: {"config": "<template yml>",
           "meshes": [{"mesh": str, "ranks": [int, ...], "walltime": str|null,
                       "config_overrides": {...}, "conditions_overrides": {...}}],
           "max_tasks_per_node", "partition", "account", "solver_backend",
           "conditions", "folder"}

    A single mesh with a single rank count is just a batch of one. With a
    folder path ('a/b'), every submitted run is filed there in the Runs
    browser (virtual folders in run_labels.json, created as needed).
    """
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404

    data = request.json or {}
    config_file = data.get('config')
    config_path = PROJECT_ROOT / (config_file or '')
    if not config_file or not config_path.is_file():
        return jsonify({'error': f'Template config not found: {config_file}'}), 404

    jobs = []
    for entry in data.get('meshes') or []:
        mesh = entry.get('mesh')
        if not mesh or not re.fullmatch(r'[A-Za-z0-9_.\-]+', mesh):
            return jsonify({'error': f'Invalid mesh name: {mesh!r}'}), 400
        try:
            ranks = sorted({int(r) for r in entry.get('ranks') or []})
        except (TypeError, ValueError):
            return jsonify({'error': f'{mesh}: rank counts must be integers'}), 400
        if not ranks or ranks[0] < 1:
            return jsonify({'error': f'{mesh}: needs at least one positive rank count'}), 400
        for r in ranks:
            jobs.append({
                'mesh': mesh, 'ranks': r,
                'walltime': entry.get('walltime') or data.get('walltime') or cl.default_walltime,
                'config_overrides': entry.get('config_overrides') or {},
                'conditions_overrides': entry.get('conditions_overrides') or {},
            })
    if not jobs:
        return jsonify({'error': 'No meshes selected'}), 400

    try:
        submitted, failed, err = cl.submit_batch(
            config_path, jobs,
            max_tasks_per_node=data.get('max_tasks_per_node') or cl.default_ntasks_per_node,
            partition=data.get('partition') or cl.default_partition,
            account=data.get('account') or cl.default_account,
            solver_backend=data.get('solver_backend', 'petsc'),
            conditions=data.get('conditions'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    msg = f'{len(submitted)} job(s) submitted to {cl.label}'
    folder = (data.get('folder') or '').strip()
    if folder and submitted:
        try:
            run_index.file_runs([job['out_name'] for job in submitted], folder)
        except ValueError as e:
            msg += f' (not filed into a folder: {e})'
    if failed:
        msg += f'; {len(failed)} failed: {", ".join(failed)}' + (f' ({err})' if err else '')
    return jsonify({'success': True, 'jobs': submitted, 'failed': failed, 'message': msg})


@app.route('/api/cluster/<cluster_id>/jobs')
def cluster_list_jobs(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    return jsonify({'jobs': list(cl.jobs.values())})


@app.route('/api/cluster/<cluster_id>/status')
@app.route('/api/cluster/<cluster_id>/status/<job_id>')
def cluster_status(cluster_id, job_id=None):
    """Poll SLURM job status and tail log output."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404

    if job_id is None:
        job_id = request.args.get('job_id') or cl.legacy_state.get('job_id')
    if not job_id:
        return jsonify({'status': None, 'job_id': None, 'log': '',
                        'message': 'No job submitted'})

    job = cl.jobs.get(job_id, {})
    # The caller names the run: after a server restart this server no longer
    # knows the job, and guessing (the last submitted run) mislabels it.
    out_name = request.args.get('out_name') or job.get('out_name') or ''

    cached_status, cached_log = cl.get_cached_status(job_id)
    if cached_status is not None and (cached_log or cached_status not in TERMINAL_STATES):
        return jsonify({
            'job_id': job_id,
            'status': cached_status,
            'out_name': out_name,
            'conditions_hash': job.get('conditions_hash'),
            'log': cached_log or ''
        })

    try:
        status = cl.check_job_status(job_id)
        log = cl.tail_remote_log(job_id, out_name=out_name or None)
        return jsonify({
            'job_id': job_id,
            'status': status,
            'out_name': out_name,
            'conditions_hash': job.get('conditions_hash'),
            'log': log
        })
    except Exception as e:
        return jsonify({
            'job_id': job_id,
            'status': job.get('status', cl.legacy_state.get('status', 'UNKNOWN')),
            'log': '',
            'error': str(e)
        })


@app.route('/api/cluster/<cluster_id>/cancel', methods=['POST'])
def cluster_cancel(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json or {}
    job_id = data.get('job_id') or cl.legacy_state.get('job_id')
    if not job_id:
        return jsonify({'error': 'No job to cancel'}), 400
    try:
        cl.cancel_job(job_id)
        return jsonify({'success': True, 'message': f'Job {job_id} cancelled'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/download-iterations', methods=['POST'])
def cluster_download_iterations(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    remote_dir = data.get('remote_dir')
    if not remote_dir:
        return jsonify({'error': 'No remote directory specified'}), 400
    local_dest = PROJECT_ROOT / remote_dir
    try:
        cl.download_iterations(remote_dir, local_dest)
        return jsonify({'success': True, 'out_name': remote_dir})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/download', methods=['POST'])
def cluster_download(cluster_id):
    """Download simulation results (SSE stream with byte progress)."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    remote_dir = data.get('remote_dir') or cl.legacy_state.get('out_name')
    if not remote_dir:
        return jsonify({'error': 'No remote directory specified'}), 400

    local_dest = PROJECT_ROOT / remote_dir

    def generate():
        for status in cl.download_results_streaming(remote_dir, local_dest):
            yield f"data: {json.dumps(status)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/cluster/<cluster_id>/meshes')
def cluster_meshes(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    try:
        families = cl.list_remote_meshes()
        return jsonify({'families': families})
    except Exception as e:
        return jsonify({'error': str(e), 'families': []}), 500


@app.route('/api/cluster/<cluster_id>/meshes/batch-info', methods=['POST'])
def cluster_mesh_batch_info(cluster_id):
    """Bounds + partition-unit counts for several remote meshes at once.

    Body: {"meshes": [name, ...]} -> {"meshes": {name: {bounds,
    mesh_conversion_factor, num_tags, num_original_tags, num_components,
    cube_subdomains} | {error}}}. cube_subdomains (2*nx*ny*nz) comes from the
    weak-scaling filename, so it is known even without the ssh round trip.
    """
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    names = [n for n in (request.json or {}).get('meshes') or []
             if isinstance(n, str) and re.fullmatch(r'[A-Za-z0-9_.\-]+', n)]
    if not names:
        return jsonify({'meshes': {}})
    try:
        info = cl.fetch_batch_mesh_info(names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    for name in names:
        entry = info.setdefault(name, {'error': 'no data returned'})
        ws = parse_weak_scaling_name(name.removesuffix('_colored'))
        entry['cube_subdomains'] = 2 * ws['nx'] * ws['ny'] * ws['nz'] if ws else None
    return jsonify({'meshes': info})


_preview_locks = {}
_preview_locks_guard = threading.Lock()


@app.route('/api/cluster/<cluster_id>/meshes/preview', methods=['POST'])
def cluster_mesh_preview(cluster_id):
    """Membrane-only preview of a mesh that only exists on the cluster.

    Body: {"mesh": name, "max_facets": int (default 1e6), "force": bool}.
    Built next to the mesh (Cluster.build_mesh_preview), cached on the cluster
    and in viz/data/_preview/<mesh>/ - apart from real conversions in
    viz/data/<mesh>/, since its vertices are a decimated subset. Returns the
    path the viewer loads it from.
    """
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json or {}
    mesh = data.get('mesh') or ''
    if not re.fullmatch(r'[A-Za-z0-9_.\-]+', mesh):
        return jsonify({'error': f'invalid mesh name: {mesh!r}'}), 400
    max_facets = int(data.get('max_facets') or 1_000_000)
    local_dir = Path(__file__).parent / 'data' / '_preview' / mesh
    path = f'data/_preview/{mesh}'

    def local_meta():
        try:
            with open(local_dir / 'mesh_metadata.json') as f:
                meta = json.load(f)
            return meta if meta.get('preview', {}).get('max_facets') == max_facets else None
        except (OSError, ValueError):
            return None

    with _preview_locks_guard:
        lock = _preview_locks.setdefault((cluster_id, mesh), threading.Lock())
    with lock:  # a second request for the same mesh waits and then hits the cache
        meta = None if data.get('force') else local_meta()
        if meta:
            return jsonify({'path': path, 'metadata': meta, 'cached': 'local'})
        try:
            meta, on_cluster = cl.build_mesh_preview(mesh, local_dir, max_facets)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'path': path, 'metadata': meta, 'cached': 'cluster' if on_cluster else None})


@app.route('/api/cluster/<cluster_id>/meshes/metadata/<mesh_name>')
def cluster_mesh_metadata(cluster_id, mesh_name):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    try:
        metadata = cl.fetch_mesh_metadata(mesh_name)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/meshes/convert', methods=['POST'])
def cluster_convert_mesh(cluster_id):
    """Convert a remote mesh inside the DOLFINx container (SSE stream)."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
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
            yield f"data: {json.dumps({'type': 'output', 'text': f'Starting conversion of {output_prefix} on {cl.label}...\\n'})}\n\n"
            if color:
                yield f"data: {json.dumps({'type': 'output', 'text': 'Graph coloring enabled (--color-intracellular)\\n'})}\n\n"

            process = cl.convert_remote_mesh(family, pts_file, elem_file, output_prefix, color)

            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            process.wait()
            cl.finish_conversion()

            if process.returncode == 0:
                yield f"data: {json.dumps({'type': 'complete', 'success': True})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Conversion failed with exit code {process.returncode}'})}\n\n"
        except Exception as e:
            cl.finish_conversion()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'Connection': 'keep-alive',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/cluster/<cluster_id>/meshes/download', methods=['POST'])
def cluster_download_mesh(cluster_id):
    """Download converted mesh data to the local data/ directory."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    mesh_name = data.get('mesh_name')
    if not mesh_name:
        return jsonify({'error': 'No mesh name specified'}), 400
    try:
        cl.download_mesh_data(mesh_name, PROJECT_ROOT / 'data')
        return jsonify({'success': True,
                        'message': f'Downloaded {mesh_name} mesh data to data/'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/weak-scaling/generate', methods=['POST'])
def cluster_weak_scaling_generate(cluster_id):
    """Generate (or reuse) a weak-scaling mesh on the cluster (SSE stream)."""
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json or {}
    try:
        nx = int(data['nx']); ny = int(data['ny']); nz = int(data['nz'])
        n = int(data.get('n', 12)); L = float(data.get('L', 25.0))
        pad = int(data.get('pad', 0))
        shape = str(data.get('shape', 'cell'))
        ax = int(data.get('ax', 4)) if shape != 'plus' else 1
        cell = shape != 'plus'
        d_y = float(data.get('d_y', 0.5) or 0.5) if cell else 0.0
        d_z = float(data.get('d_z', 0) or 0) if cell else 0.0
        lean = float(data.get('lean', 55) or 55) if cell else 0.0
        lat_r = float(data.get('lat_r', 0.26) or 0.26) if cell else 0.0
        slabs = step_y = step_z = 0
        if cell:
            slabs, step_y, step_z = resolve_ws_shift(ax, d_y, d_z or d_y)
            d_y = 0.5 * (ax - ax * step_y / slabs)
            d_z = 0.5 * (ax - ax * step_z / slabs)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    err = validate_weak_scaling(nx, ny, nz, n, L, pad, shape, ax, slabs)
    if err:
        return jsonify({'error': err}), 400

    name = weak_scaling_name(nx, ny, nz, n, L, pad, shape, ax, slabs,
                             step_y, step_z, d_y, d_z, lat_r, lean)

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'output', 'text': f'Generating {name} on {cl.label} ({nx}x{ny}x{nz} cubes, pad {pad})...\\n'})}\n\n"

            process = cl.generate_remote_weak_scaling_mesh(nx, ny, nz, n, L, pad, name,
                                                           shape=shape, ax=ax,
                                                           slabs=slabs, d_y=d_y,
                                                           d_z=d_z, lean=lean,
                                                           lat_r=lat_r)

            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            process.wait()
            cl.finish_conversion()

            if process.returncode == 0:
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'name': name})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Generation failed with exit code {process.returncode}'})}\n\n"
        except Exception as e:
            cl.finish_conversion()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'Connection': 'keep-alive',
                             'X-Accel-Buffering': 'no'})


# --------------------- Remote Video API ---------------------

# In-memory tracking for remote video jobs (job_id -> info incl. cluster)
remote_video_jobs = {}


@app.route('/api/cluster/<cluster_id>/video/generate', methods=['POST'])
def cluster_generate_video(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400
    try:
        result = cl.generate_remote_video(
            sim_name,
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            fps=data.get('fps', 30),
            camera_config=data.get('camera'),
            colormap=data.get('colormap', 'coolwarm'),
            partition=data.get('partition'),
            account=data.get('account'))
        remote_video_jobs[result['job_id']] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/video/status/<job_id>')
def cluster_video_status(cluster_id, job_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    job = remote_video_jobs.get(job_id, {})
    try:
        result = cl.check_video_job(job_id, job.get('log_file'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/video/download/<job_id>', methods=['POST'])
def cluster_download_video(cluster_id, job_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    job = remote_video_jobs.get(job_id, {})
    try:
        status = cl.check_video_job(job_id, job.get('log_file'))
        video_filename = status.get('video_filename')
        if not video_filename:
            return jsonify({'error': 'Video file not found in job output'}), 404
        cl.download_video(video_filename, PROJECT_ROOT / 'viz' / 'videos')
        return jsonify({
            'success': True,
            'filename': video_filename,
            'download_url': f'/api/video/download/{video_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --------------------- Remote Viz Data API ---------------------

remote_viz_jobs = {}


@app.route('/api/cluster/<cluster_id>/viz/generate', methods=['POST'])
def cluster_generate_viz(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400
    try:
        result = cl.generate_remote_viz(sim_name)
        remote_viz_jobs[result['job_id']] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/viz/status/<job_id>')
def cluster_viz_status(cluster_id, job_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    job = remote_viz_jobs.get(job_id, {})
    try:
        result = cl.check_viz_job(job_id, job.get('log_file'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<cluster_id>/viz/download', methods=['POST'])
def cluster_download_viz(cluster_id):
    cl = _get_cluster(cluster_id)
    if cl is None:
        return jsonify({'error': f'unknown cluster {cluster_id}'}), 404
    data = request.json
    sim_name = data.get('sim_name')
    if not sim_name:
        return jsonify({'error': 'No simulation name specified'}), 400

    local_dest = PROJECT_ROOT / 'viz' / 'data' / sim_name

    def stream():
        for event in cl.download_viz_data_streaming(sim_name, local_dest):
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
