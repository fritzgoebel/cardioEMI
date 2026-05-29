"""
Karolina supercomputer SSH/SLURM backend.

Wraps subprocess calls to ssh/scp using ~/.ssh/config host alias 'karolina'.
Runs simulation and mesh conversion inside Apptainer (Singularity) containers.
"""

import subprocess
import re
import json
import os
import hashlib
import yaml
import threading
from datetime import datetime
from pathlib import Path

REMOTE_HOST = 'karolina'
REMOTE_PATH = '/scratch/project/eu-26-11/fritz/cardioEMI'
CONTAINERS_PATH = f'{REMOTE_PATH}/containers'
MESHES_PATH = f'{REMOTE_PATH}/meshes'
DATA_PATH = f'{REMOTE_PATH}/data'

# Container image names
DOLFINX_SIF = f'{CONTAINERS_PATH}/dolfinx-v0.9.0.sif'
GINKGO_SIF = f'{CONTAINERS_PATH}/dolfinx-ginkgo-bddc.sif'

# Multi-job state: keyed by job_id
karolina_jobs = {}

# Legacy single-job alias (for backward compat in status/download)
karolina_state = {
    'job_id': None,
    'status': None,
    'config_file': None,
    'out_name': None,
    'num_ranks': None,
    'submitting': False,
}

# Mesh conversion state
mesh_convert_state = {
    'converting': False,
    'process': None,  # subprocess.Popen for streaming output
}

# Background poller state: cached status/log per job, updated by a background thread
_poll_cache = {}        # job_id -> {'status': str, 'log': str}
_poll_lock = threading.Lock()
_poll_thread = None
_poll_stop = threading.Event()
POLL_INTERVAL = 5  # seconds between background poll cycles

TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'}


def _poll_active_jobs():
    """Background thread: periodically check all active jobs via batched SSH."""
    while not _poll_stop.is_set():
        # Collect job IDs that still need polling
        with _poll_lock:
            active = {jid: karolina_jobs[jid]
                      for jid in list(karolina_jobs)
                      if _poll_cache.get(jid, {}).get('status') not in TERMINAL_STATES}

        if active:
            job_ids = list(active.keys())
            job_ids_str = ','.join(job_ids)

            # --- Batch status check: single SSH call for all jobs ---
            statuses = {}
            try:
                stdout, _, rc = _run_ssh(
                    f'squeue -j {job_ids_str} --noheader -o "%i %T" 2>/dev/null; '
                    f'sacct -j {job_ids_str} --noheader -o JobID,State -P 2>/dev/null',
                    timeout=20
                )
                if rc == 0 and stdout:
                    for line in stdout.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('|') if '|' in line else line.split()
                        if len(parts) >= 2:
                            jid = parts[0].strip().split('.')[0]  # strip sacct sub-steps
                            st = parts[1].strip()
                            if jid in active and st:
                                statuses[jid] = st
            except (subprocess.TimeoutExpired, Exception):
                pass

            # --- Batch log tail: single SSH call for all jobs ---
            logs = {}
            # Build a combined tail command
            tail_parts = []
            for jid in job_ids:
                job = active[jid]
                out_name = job.get('out_name', '')
                # Try the tee'd log first (more useful), fall back to slurm log
                if out_name:
                    tail_parts.append(
                        f'echo "@@JOB {jid}@@"; '
                        f'tail -n 50 {REMOTE_PATH}/{out_name}_slurm.log 2>/dev/null || '
                        f'tail -n 50 {REMOTE_PATH}/slurm_{jid}.out 2>/dev/null || true'
                    )
                else:
                    tail_parts.append(
                        f'echo "@@JOB {jid}@@"; '
                        f'tail -n 50 {REMOTE_PATH}/slurm_{jid}.out 2>/dev/null || true'
                    )
            try:
                combined = '; '.join(tail_parts)
                stdout, _, rc = _run_ssh(combined, timeout=20)
                if rc == 0 and stdout:
                    current_jid = None
                    current_lines = []
                    for line in stdout.split('\n'):
                        if line.startswith('@@JOB ') and line.endswith('@@'):
                            if current_jid:
                                logs[current_jid] = '\n'.join(current_lines)
                            current_jid = line[6:-2]
                            current_lines = []
                        else:
                            current_lines.append(line)
                    if current_jid:
                        logs[current_jid] = '\n'.join(current_lines)
            except (subprocess.TimeoutExpired, Exception):
                pass

            # --- Update cache ---
            with _poll_lock:
                for jid in job_ids:
                    entry = _poll_cache.setdefault(jid, {'status': 'PENDING', 'log': ''})
                    if jid in statuses:
                        entry['status'] = statuses[jid]
                        # Also update karolina_jobs so other code sees it
                        if jid in karolina_jobs:
                            karolina_jobs[jid]['status'] = statuses[jid]
                    if jid in logs:
                        entry['log'] = logs[jid]

        _poll_stop.wait(POLL_INTERVAL)


def start_background_poller():
    """Start the background polling thread (idempotent)."""
    global _poll_thread
    if _poll_thread and _poll_thread.is_alive():
        return
    _poll_stop.clear()
    _poll_thread = threading.Thread(target=_poll_active_jobs, daemon=True)
    _poll_thread.start()


def stop_background_poller():
    """Stop the background polling thread."""
    _poll_stop.set()
    if _poll_thread:
        _poll_thread.join(timeout=5)


def get_cached_status(job_id):
    """Return cached (status, log) for a job, or None if not cached."""
    with _poll_lock:
        entry = _poll_cache.get(job_id)
        if entry:
            return entry['status'], entry['log']
    return None, None


def _run_ssh(cmd, timeout=30):
    """Run a command on Karolina via SSH. Returns (stdout, stderr, returncode)."""
    full_cmd = ['ssh', REMOTE_HOST, cmd]
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _apptainer_exec(sif, cmd, workdir=REMOTE_PATH):
    """Build an apptainer exec command string for running inside a container."""
    return (
        f'apptainer exec --bind {REMOTE_PATH}:/home/fenics '
        f'--pwd /home/fenics {sif} '
        f'bash -c {_shell_quote(cmd)}'
    )


def _shell_quote(s):
    """Quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"


def check_ssh():
    """Test SSH connectivity to Karolina. Returns True if reachable."""
    try:
        stdout, stderr, rc = _run_ssh('echo ok', timeout=10)
        return rc == 0 and 'ok' in stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def check_containers():
    """Check which container images are available on Karolina."""
    result = {}
    try:
        stdout, stderr, rc = _run_ssh(
            f'ls -la {CONTAINERS_PATH}/*.sif 2>/dev/null || true',
            timeout=15
        )
        if rc == 0 and stdout:
            result['dolfinx'] = 'dolfinx-v0.9.0.sif' in stdout
            result['ginkgo'] = 'dolfinx-ginkgo-bddc.sif' in stdout
        else:
            result['dolfinx'] = False
            result['ginkgo'] = False
    except (subprocess.TimeoutExpired, OSError):
        result['dolfinx'] = False
        result['ginkgo'] = False
    return result


# --------------------- Remote Mesh Operations ---------------------

def fetch_mesh_metadata(mesh_name):
    """Fetch mesh bounding box and metadata from Karolina without downloading the mesh.

    Runs a tiny Python script inside the container that reads the H5 file header.
    Returns dict with bounds, mesh_conversion_factor, vertex_count, etc.
    """
    pylibs = '/home/fenics/.pylibs'
    script = (
        'import h5py, json, numpy as np; '
        f'f = h5py.File("data/{mesh_name}.h5", "r"); '
        'v = f["/Mesh/mesh/geometry"][:]; '
        'tags = f["/Mesh/facet_tags/Values"][:]; '
        'mt = tags[tags > 0]; '
        'ext = max(v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min()); '
        'cf = 0.0001 if ext > 10 else 1.0; '
        'print(json.dumps({"bounds": {'
        '"x": [float(v[:,0].min()), float(v[:,0].max())], '
        '"y": [float(v[:,1].min()), float(v[:,1].max())], '
        '"z": [float(v[:,2].min()), float(v[:,2].max())]}, '
        '"mesh_conversion_factor": cf, '
        '"vertex_count": len(v), '
        '"facet_count": int((tags > 0).sum()), '
        '"unique_tags": sorted(set(int(t) for t in mt))}))'
    )
    container_cmd = (
        f'pip install --target={pylibs} -q h5py 2>/dev/null; '
        f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
        f'python3 -c {_shell_quote(script)}'
    )
    apptainer_cmd = _apptainer_exec(DOLFINX_SIF, container_cmd)
    # Unset host env vars that interfere with apptainer before running
    full_cmd = (
        f'unset SINGULARITY_BINDPATH SINGULARITYENV_LD_PRELOAD LD_PRELOAD; '
        f'{apptainer_cmd}'
    )
    stdout, stderr, rc = _run_ssh(full_cmd, timeout=120)

    # Parse JSON from stdout (may have module warnings and other noise)
    combined = stdout + '\n' + stderr
    for line in combined.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    if rc != 0:
        raise RuntimeError(f'Failed to fetch mesh metadata: {stderr or stdout}')
    raise RuntimeError(f'No JSON output from metadata script. stdout: {stdout[:500]}')


def list_remote_meshes():
    """List mesh families under meshes/ on Karolina.

    Returns a list of dicts with structure:
    [
        {
            'family': 'rizzo',
            'meshes': [
                {'name': 'rizzo36', 'pts': 'rizzo36-sep-domi.pts', 'elem': 'rizzo36-sep-domi.elem'},
                {'name': 'rizzo37', ...},
            ]
        },
        ...
    ]

    Uses a single SSH call to gather all information.
    """
    try:
        # Single SSH command: list all .pts files and all converted .h5 files
        cmd = (
            f'find {MESHES_PATH} -name "*.pts" -type f 2>/dev/null; '
            f'echo "---SEPARATOR---"; '
            f'ls {DATA_PATH}/*.h5 2>/dev/null || true'
        )
        stdout, stderr, rc = _run_ssh(cmd, timeout=30)
        if rc != 0 or not stdout:
            return []

        parts = stdout.split('---SEPARATOR---')
        pts_section = parts[0].strip() if len(parts) > 0 else ''
        h5_section = parts[1].strip() if len(parts) > 1 else ''

        # Build set of converted mesh names from .h5 files
        converted_names = set()
        if h5_section:
            for line in h5_section.split('\n'):
                h5_name = Path(line.strip()).stem  # e.g. "rizzo36" or "rizzo36_colored"
                if h5_name:
                    converted_names.add(h5_name)

        # Group .pts files by family directory
        family_map = {}  # family_name -> list of pts paths
        if pts_section:
            for line in pts_section.split('\n'):
                pts_path = line.strip()
                if not pts_path or not pts_path.endswith('.pts'):
                    continue
                p = Path(pts_path)
                family_name = p.parent.name
                family_map.setdefault(family_name, []).append(p)

        # Track mesh names that come from .pts files
        pts_mesh_names = set()

        families = []
        for family_name in sorted(family_map):
            meshes = []
            for pts_path in family_map[family_name]:
                pts_name = pts_path.name
                mesh_name = pts_name.split('-')[0]
                elem_name = pts_name.replace('.pts', '.elem')
                pts_mesh_names.add(mesh_name)

                meshes.append({
                    'name': mesh_name,
                    'pts': pts_name,
                    'elem': elem_name,
                    'converted': mesh_name in converted_names,
                    'converted_colored': f'{mesh_name}_colored' in converted_names,
                })

            if meshes:
                families.append({
                    'family': family_name,
                    'meshes': meshes,
                })

        # Discover h5-only meshes (no .pts source files)
        h5_only = {}  # base_name -> {'converted': bool, 'converted_colored': bool}
        for name in converted_names:
            base = name.removesuffix('_colored')
            if base in pts_mesh_names:
                continue
            h5_only.setdefault(base, {'converted': False, 'converted_colored': False})
            if name.endswith('_colored'):
                h5_only[base]['converted_colored'] = True
            else:
                h5_only[base]['converted'] = True

        if h5_only:
            # Group h5-only meshes by family prefix (letters before first digit)
            h5_family_map = {}
            for base_name, flags in sorted(h5_only.items()):
                # Derive family from name: strip trailing digits/hyphens
                family = re.match(r'^([a-zA-Z]+)', base_name)
                family_name = family.group(1) if family else base_name
                h5_family_map.setdefault(family_name, []).append({
                    'name': base_name,
                    'pts': None,
                    'elem': None,
                    'converted': flags['converted'],
                    'converted_colored': flags['converted_colored'],
                })

            for family_name in sorted(h5_family_map):
                # Merge into existing family if present
                existing = next((f for f in families if f['family'] == family_name), None)
                if existing:
                    existing_names = {m['name'] for m in existing['meshes']}
                    for mesh in h5_family_map[family_name]:
                        if mesh['name'] not in existing_names:
                            existing['meshes'].append(mesh)
                else:
                    families.append({
                        'family': family_name,
                        'meshes': h5_family_map[family_name],
                    })

            families.sort(key=lambda f: f['family'])

        return families

    except (subprocess.TimeoutExpired, OSError):
        return []


def convert_remote_mesh(family, pts_file, elem_file, output_prefix, color=False):
    """Start mesh conversion on Karolina inside the DOLFINx container.

    Returns a subprocess.Popen that streams output.
    """
    if mesh_convert_state['converting']:
        raise RuntimeError('A conversion is already in progress')

    pts_path = f'meshes/{family}/{pts_file}'
    elem_path = f'meshes/{family}/{elem_file}'
    out_path = f'data/{output_prefix}'

    # Ensure data directory exists
    _run_ssh(f'mkdir -p {DATA_PATH}', timeout=10)

    # Build conversion command inside container
    # lxml must be installed to a writable target (container FS is read-only)
    pylibs = '/home/fenics/.pylibs'
    convert_cmd = (
        f'pip install --target={pylibs} -q lxml h5py 2>/dev/null; '
        f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
        f'python3 geometry/convert_pts_elem.py '
        f'{pts_path} {elem_path} {out_path}'
    )
    if color:
        convert_cmd += ' --color-intracellular'

    remote_cmd = _apptainer_exec(DOLFINX_SIF, convert_cmd)

    # Start SSH process with streaming output
    full_cmd = ['ssh', REMOTE_HOST, remote_cmd]
    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    mesh_convert_state['converting'] = True
    mesh_convert_state['process'] = process

    return process


def finish_conversion():
    """Clean up after conversion completes."""
    mesh_convert_state['converting'] = False
    mesh_convert_state['process'] = None


def download_mesh_data(mesh_name, local_data_dir):
    """Download converted mesh files (h5, xdmf, pickle) from Karolina."""
    local_data_dir = Path(local_data_dir)
    local_data_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        f'{mesh_name}.h5',
        f'{mesh_name}.xdmf',
        f'{mesh_name}.pickle',
    ]

    for fname in files_to_download:
        remote_file = f'{REMOTE_HOST}:{DATA_PATH}/{fname}'
        result = subprocess.run(
            ['scp', remote_file, str(local_data_dir / fname)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f'Failed to download {fname}: {result.stderr}')

    return True


# --------------------- Config Upload ---------------------

def upload_config(local_path):
    """SCP a config YAML file to the remote cardioEMI directory."""
    remote_dest = f'{REMOTE_HOST}:{REMOTE_PATH}/'
    result = subprocess.run(
        ['scp', str(local_path), remote_dest],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f'SCP upload failed: {result.stderr}')
    return True


# --------------------- SLURM Job Management ---------------------

def generate_slurm_script(config_file, nodes=1, ntasks_per_node=128,
                          walltime='01:00:00', partition='qcpu_exp',
                          account='eu-26-11', solver_backend='petsc',
                          include_ranks_in_name=False):
    """Generate a SLURM batch script string for cardioEMI."""
    base = Path(config_file).stem.replace('input_', '') + '_sim'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ranks_suffix = f'_{nodes * ntasks_per_node}r' if include_ranks_in_name else ''
    out_name = f'{base}_{timestamp}{ranks_suffix}'

    # Select container image
    if solver_backend == 'ginkgo':
        sif = GINKGO_SIF
    else:
        sif = DOLFINX_SIF

    # Compute total ranks and cpus-per-task for memory-bound pinning
    total_ranks = nodes * ntasks_per_node
    cpus_per_task = 128 // ntasks_per_node

    script = f"""#!/bin/bash
#SBATCH --job-name=cardioEMI
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --exclusive
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd {REMOTE_PATH}

# Use tcp provider - container's libfabric lacks Karolina's native OFI provider
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=1

# Karolina CPU nodes: 2x AMD EPYC 7H12 (64 cores/socket, 8 memory channels/socket)
# NPS4 = 8 NUMA domains/node. Optimal memory-bound config: 16 ranks/node = 1 rank
# per memory channel (8 cpus-per-task). --cpu-bind=cores pins ranks to cores,
# --distribution=block:block assigns consecutive ranks to consecutive NUMA domains.
srun -n {total_ranks} --cpu-bind=cores --distribution=block:block apptainer exec \\
    --bind {REMOTE_PATH}:/home/fenics \\
    --pwd /home/fenics \\
    {sif} \\
    bash -c 'unset CC CXX && export PYTHONPATH=/home/fenics/.pylibs:$PYTHONPATH && python3 -B -u main.py {out_name}/{config_file}' 2>&1 | tee {out_name}_slurm.log
"""
    return script, out_name


def submit_job(config_file, nodes=1, ntasks_per_node=128,
               walltime='01:00:00', partition='qcpu_exp',
               account='eu-26-11', solver_backend='petsc',
               conditions=None, local_config_path=None):
    """Upload config and SLURM script, then submit via sbatch. Returns job info dict."""
    # Generate SLURM script with timestamped out_name
    script_content, out_name = generate_slurm_script(
        config_file, nodes, ntasks_per_node, walltime, partition, account,
        solver_backend
    )

    # Create remote job directory and upload all job files into it
    job_dir = f'{REMOTE_PATH}/{out_name}'
    _run_ssh(f'mkdir -p {job_dir}', timeout=10)

    # Update the config YAML's out_name to the timestamped version
    if local_config_path and Path(local_config_path).exists():
        with open(local_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        config['out_name'] = out_name
        with open(local_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Upload config file into job directory
    if local_config_path:
        remote_dest = f'{REMOTE_HOST}:{job_dir}/{config_file}'
        result = subprocess.run(
            ['scp', str(local_config_path), remote_dest],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f'SCP upload failed: {result.stderr}')

    # Write script to temp file and upload into job directory
    script_name = 'run_cardioemi.sh'
    local_script = Path('/tmp') / script_name
    local_script.write_text(script_content)

    remote_dest = f'{REMOTE_HOST}:{job_dir}/{script_name}'
    result = subprocess.run(
        ['scp', str(local_script), remote_dest],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f'Failed to upload SLURM script: {result.stderr}')

    # Upload conditions.json to the job directory
    if conditions:
        conditions_hash = hashlib.sha256(
            json.dumps(conditions, sort_keys=True).encode()
        ).hexdigest()[:12]
        conditions['_hash'] = conditions_hash

        local_cond = Path('/tmp') / 'conditions.json'
        local_cond.write_text(json.dumps(conditions, indent=2))
        subprocess.run(
            ['scp', str(local_cond), f'{REMOTE_HOST}:{job_dir}/conditions.json'],
            capture_output=True, text=True, timeout=30
        )

    # Submit job from its own directory
    stdout, stderr, rc = _run_ssh(
        f'cd {job_dir} && sbatch {script_name}',
        timeout=30
    )
    if rc != 0:
        raise RuntimeError(f'sbatch failed: {stderr}')

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', stdout)
    if not match:
        raise RuntimeError(f'Could not parse job ID from: {stdout}')

    job_id = match.group(1)
    num_ranks = nodes * ntasks_per_node

    # Store in multi-job dict
    job_info = {
        'job_id': job_id,
        'status': 'PENDING',
        'config_file': config_file,
        'out_name': out_name,
        'num_ranks': num_ranks,
        'conditions_hash': conditions.get('_hash') if conditions else None,
    }
    karolina_jobs[job_id] = job_info

    # Register in poll cache and ensure background poller is running
    with _poll_lock:
        _poll_cache[job_id] = {'status': 'PENDING', 'log': ''}
    start_background_poller()

    # Update legacy single-job state
    karolina_state['job_id'] = job_id
    karolina_state['status'] = 'PENDING'
    karolina_state['config_file'] = config_file
    karolina_state['out_name'] = out_name
    karolina_state['num_ranks'] = num_ranks

    return job_info


def submit_jobs(config_file, node_counts, ntasks_per_node=128,
                walltime='01:00:00', partition='qcpu_exp',
                account='eu-26-11', solver_backend='petsc',
                conditions=None, local_config_path=None):
    """Submit multiple jobs (one per node count) using minimal SSH/SCP calls.

    Returns a list of job info dicts.
    """
    import tempfile

    if conditions:
        conditions_hash = hashlib.sha256(
            json.dumps(conditions, sort_keys=True).encode()
        ).hexdigest()[:12]
        conditions['_hash'] = conditions_hash

    # --- Phase 1: Prepare all local files in a temp directory ---
    jobs_prep = []  # list of (out_name, nodes, script_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        scp_sources = []

        for nodes in node_counts:
            script_content, out_name = generate_slurm_script(
                config_file, nodes, ntasks_per_node, walltime, partition, account,
                solver_backend, include_ranks_in_name=True
            )

            # Create local job directory
            job_local = tmpdir / out_name
            job_local.mkdir()

            # Write config with this job's out_name
            if local_config_path and Path(local_config_path).exists():
                with open(local_config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                config['out_name'] = out_name
                local_cfg = job_local / config_file
                with open(local_cfg, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Write SLURM script
            script_name = 'run_cardioemi.sh'
            (job_local / script_name).write_text(script_content)

            # Write conditions
            if conditions:
                job_conditions = dict(conditions)
                job_conditions['nRanks'] = nodes * ntasks_per_node
                (job_local / 'conditions.json').write_text(
                    json.dumps(job_conditions, indent=2)
                )

            scp_sources.append(str(job_local))
            jobs_prep.append((out_name, nodes, script_name))

        # --- Phase 2: Single scp to upload all job directories ---
        result = subprocess.run(
            ['scp', '-r'] + scp_sources + [f'{REMOTE_HOST}:{REMOTE_PATH}/'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f'SCP upload failed: {result.stderr}')

    # --- Phase 3: Single SSH to sbatch all jobs ---
    sbatch_cmds = '; '.join(
        f'echo "@@JOB {out_name}@@"; cd {REMOTE_PATH}/{out_name} && sbatch {script_name}'
        for out_name, nodes, script_name in jobs_prep
    )
    stdout, stderr, rc = _run_ssh(sbatch_cmds, timeout=30)
    if rc != 0:
        raise RuntimeError(f'sbatch failed: {stderr}')

    # --- Phase 4: Parse job IDs and register ---
    job_infos = []
    current_out_name = None
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('@@JOB ') and line.endswith('@@'):
            current_out_name = line[6:-2]
            continue
        match = re.search(r'Submitted batch job (\d+)', line)
        if match and current_out_name:
            job_id = match.group(1)
            # Find the matching prep entry
            for out_name, nodes, _ in jobs_prep:
                if out_name == current_out_name:
                    num_ranks = nodes * ntasks_per_node
                    job_info = {
                        'job_id': job_id,
                        'status': 'PENDING',
                        'config_file': config_file,
                        'out_name': out_name,
                        'num_ranks': num_ranks,
                        'conditions_hash': conditions.get('_hash') if conditions else None,
                    }
                    karolina_jobs[job_id] = job_info
                    with _poll_lock:
                        _poll_cache[job_id] = {'status': 'PENDING', 'log': ''}
                    # Update legacy state to last job
                    karolina_state['job_id'] = job_id
                    karolina_state['status'] = 'PENDING'
                    karolina_state['config_file'] = config_file
                    karolina_state['out_name'] = out_name
                    karolina_state['num_ranks'] = num_ranks
                    job_infos.append(job_info)
                    break
            current_out_name = None

    if job_infos:
        start_background_poller()

    # Update local config's out_name to the last job (for consistency)
    if local_config_path and jobs_prep:
        last_out_name = jobs_prep[-1][0]
        with open(local_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        config['out_name'] = last_out_name
        with open(local_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return job_infos


def check_job_status(job_id):
    """Check SLURM job status via squeue/sacct. Returns status string."""
    job = karolina_jobs.get(job_id, {})

    # First try squeue (for queued/running jobs)
    try:
        stdout, stderr, rc = _run_ssh(
            f'squeue -j {job_id} --noheader -o "%T"',
            timeout=15
        )
        if rc == 0 and stdout:
            status = stdout.strip().split('\n')[0].strip()
            if status:
                if job:
                    job['status'] = status
                karolina_state['status'] = status
                return status
    except subprocess.TimeoutExpired:
        pass

    # Job not in queue - check sacct for completed/failed
    try:
        stdout, stderr, rc = _run_ssh(
            f'sacct -j {job_id} --noheader -o State -P',
            timeout=15
        )
        if rc == 0 and stdout:
            status = stdout.strip().split('\n')[0].strip()
            if status:
                if job:
                    job['status'] = status
                karolina_state['status'] = status
                return status
    except subprocess.TimeoutExpired:
        pass

    return job.get('status', karolina_state.get('status', 'UNKNOWN'))


def cancel_job(job_id):
    """Cancel a SLURM job via scancel."""
    stdout, stderr, rc = _run_ssh(f'scancel {job_id}', timeout=15)
    if rc != 0:
        raise RuntimeError(f'scancel failed: {stderr}')
    if job_id in karolina_jobs:
        karolina_jobs[job_id]['status'] = 'CANCELLED'
    with _poll_lock:
        if job_id in _poll_cache:
            _poll_cache[job_id]['status'] = 'CANCELLED'
    karolina_state['status'] = 'CANCELLED'
    return True


def delete_job_data(job_id, out_name, local_root):
    """Cancel the job (if non-terminal), then remove its remote dir and local download.

    Returns a list of error messages for partial failures. Empty list = full success.
    The caller should treat any non-empty list as a warning, not a hard failure —
    e.g. the remote dir may already be gone, but we still want to drop the entry.
    """
    import shutil

    errors = []

    # Cancel if still active. cancel_job raises if scancel fails — but a job that's
    # already done will get a benign "Invalid job id" from scancel; just swallow.
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'}
    cached_status, _ = get_cached_status(job_id) if job_id else (None, None)
    if job_id and cached_status not in terminal:
        try:
            _run_ssh(f'scancel {job_id}', timeout=15)
        except Exception as e:
            errors.append(f'scancel: {e}')

    if out_name:
        # Whitelist out_name to avoid any chance of arg injection. Slurm out_names
        # are derived from config stem + timestamp (+ optional ranks suffix), so
        # alphanumerics + underscore + dot + hyphen is the full alphabet.
        if not re.fullmatch(r'[A-Za-z0-9_.\-]+', out_name):
            errors.append(f'invalid out_name (refusing remote rm): {out_name!r}')
        else:
            remote_dir = f'{REMOTE_PATH}/{out_name}'
            try:
                _, stderr, rc = _run_ssh(f'rm -rf -- {remote_dir}', timeout=30)
                if rc != 0:
                    errors.append(f'remote rm: {stderr}')
            except Exception as e:
                errors.append(f'remote rm: {e}')

            local_dir = Path(local_root) / out_name
            if local_dir.exists():
                try:
                    shutil.rmtree(local_dir)
                except Exception as e:
                    errors.append(f'local rm: {e}')

    # Drop in-memory tracking regardless.
    if job_id:
        karolina_jobs.pop(job_id, None)
        with _poll_lock:
            _poll_cache.pop(job_id, None)
        if karolina_state.get('job_id') == job_id:
            karolina_state['job_id'] = None
            karolina_state['status'] = None

    return errors


def tail_remote_log(job_id, num_lines=50):
    """Tail the SLURM output log for the given job."""
    # Try the slurm output file first
    try:
        stdout, stderr, rc = _run_ssh(
            f'tail -n {num_lines} {REMOTE_PATH}/slurm_{job_id}.out 2>/dev/null',
            timeout=15
        )
        if rc == 0 and stdout:
            return stdout
    except subprocess.TimeoutExpired:
        pass

    # Also try the tee'd log file
    job = karolina_jobs.get(job_id, {})
    out_name = job.get('out_name', karolina_state.get('out_name', ''))
    if out_name:
        try:
            stdout, stderr, rc = _run_ssh(
                f'tail -n {num_lines} {REMOTE_PATH}/{out_name}_slurm.log 2>/dev/null',
                timeout=15
            )
            if rc == 0 and stdout:
                return stdout
        except subprocess.TimeoutExpired:
            pass

    return ''


# --------------------- Results Download ---------------------

def download_results_streaming(remote_out_name, local_dest):
    """Download simulation results as a single tar.gz archive with progress.

    Creates a tar.gz on the remote side, streams it via SSH, and extracts locally.
    Much faster than per-file SCP due to fewer SSH handshakes and compression.

    Yields dicts: {'type': 'progress', 'bytes_done': int, 'bytes_total': int, 'file': str}
                  {'type': 'complete', 'message': str}
                  {'type': 'error', 'message': str}
    """
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    # Build list of paths to include in the archive
    remote_dir = f'{REMOTE_PATH}/{remote_out_name}'
    tar_paths = [remote_out_name]

    # Also include IF_*.txt files if available
    num_ranks = karolina_state.get('num_ranks')
    if_pattern = ''
    if num_ranks:
        if_pattern = ' '.join(f'IF_{i}.txt' for i in range(num_ranks))

    # Get total uncompressed size for progress estimation
    size_cmd = f'du -sb {remote_dir} 2>/dev/null | cut -f1'
    stdout, stderr, rc = _run_ssh(size_cmd, timeout=30)
    if rc != 0 or not stdout.strip():
        yield {'type': 'error', 'message': f'Failed to get remote directory size: {stderr}'}
        return

    bytes_total_uncompressed = int(stdout.strip())
    # Estimate compressed size (HDF5 files compress ~40-60%)
    bytes_total_estimate = int(bytes_total_uncompressed * 0.5)

    yield {'type': 'progress', 'bytes_done': 0, 'bytes_total': bytes_total_estimate,
           'file': 'Creating archive...'}

    # Build remote tar command: tar the sim directory + IF files, gzip, stream to stdout
    tar_cmd = f'cd {REMOTE_PATH} && tar czf - {remote_out_name}'
    if if_pattern:
        # Add IF files if they exist (ignore missing ones)
        tar_cmd += f' {if_pattern} 2>/dev/null'

    # Stream tar.gz via SSH and extract locally
    archive_path = local_dest.parent / f'.{remote_out_name}.tar.gz'

    try:
        # Download archive with progress tracking
        process = subprocess.Popen(
            ['ssh', REMOTE_HOST, tar_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        bytes_done = 0
        chunk_size = 256 * 1024  # 256KB chunks
        with open(archive_path, 'wb') as f:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_done += len(chunk)
                # Yield progress every chunk
                yield {'type': 'progress', 'bytes_done': bytes_done,
                       'bytes_total': bytes_total_estimate,
                       'file': f'Downloading archive ({bytes_done // (1024*1024)} MB)...'}

        process.wait()
        if process.returncode != 0:
            stderr_out = process.stderr.read().decode()
            yield {'type': 'error', 'message': f'SSH tar failed: {stderr_out}'}
            return

        compressed_size = bytes_done
        yield {'type': 'progress', 'bytes_done': compressed_size,
               'bytes_total': compressed_size, 'file': 'Extracting archive...'}

        # Extract to local destination's parent (archive contains remote_out_name/ dir)
        result = subprocess.run(
            ['tar', 'xzf', str(archive_path), '-C', str(local_dest.parent)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            yield {'type': 'error', 'message': f'Extraction failed: {result.stderr}'}
            return

        ratio = (1 - compressed_size / bytes_total_uncompressed) * 100 if bytes_total_uncompressed > 0 else 0
        yield {'type': 'complete',
               'message': f'Downloaded {compressed_size // (1024*1024)} MB '
                          f'(compressed {ratio:.0f}% from {bytes_total_uncompressed // (1024*1024)} MB)'}

    finally:
        # Clean up temporary archive
        if archive_path.exists():
            archive_path.unlink()


def download_iterations(remote_out_name, local_dest):
    """Download just iterations.pickle and conditions.json from a remote simulation."""
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    remote_dir = f'{REMOTE_PATH}/{remote_out_name}'
    files = ['iterations.pickle', 'residuals.pickle', 'conditions.json']

    for fname in files:
        remote_file = f'{REMOTE_HOST}:{remote_dir}/{fname}'
        result = subprocess.run(
            ['scp', remote_file, str(local_dest / fname)],
            capture_output=True, text=True, timeout=30
        )
        # Don't fail if a file doesn't exist — it's optional
        if result.returncode != 0:
            continue

    return True


def list_remote_simulations():
    """List simulation output directories on the remote machine."""
    try:
        stdout, stderr, rc = _run_ssh(
            f'ls -d {REMOTE_PATH}/*_sim* 2>/dev/null || true',
            timeout=15
        )
        if rc == 0 and stdout:
            dirs = []
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    dirs.append(Path(line).name)
            return dirs
    except subprocess.TimeoutExpired:
        pass

    return []


# --------------------- Remote Video Generation ---------------------

def generate_remote_video(sim_name, width=1920, height=1080, fps=30,
                          camera_config=None, colormap='coolwarm',
                          partition='qcpu_exp', account='eu-26-11'):
    """Submit a SLURM job to generate a video on Karolina.

    Returns job info dict with job_id and log_file.
    """
    pylibs = '/home/fenics/.pylibs'
    video_dir = f'{REMOTE_PATH}/viz/videos'

    # Ensure video output dir exists
    _run_ssh(f'mkdir -p {video_dir}', timeout=10)

    # Build the pipeline command
    pipeline_cmd = (
        f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
        f'pip install --target={pylibs} -q pyvista matplotlib "imageio[ffmpeg]" 2>/dev/null; '
        f'python3 -u viz/scripts/remote_video_pipeline.py {sim_name} '
        f'--width {width} --height {height} --fps {fps} '
        f'--pylibs {pylibs} --project-root /home/fenics'
    )
    if camera_config:
        cam_json = json.dumps(camera_config).replace('"', '\\"')
        pipeline_cmd += f' --camera "{cam_json}"'
    if colormap:
        pipeline_cmd += f' --colormap {colormap}'

    # Use the DOLFINx container (has h5py, numpy)
    container_cmd = (
        f'apptainer exec --bind {REMOTE_PATH}:/home/fenics '
        f'--pwd /home/fenics {DOLFINX_SIF} '
        f'bash -c {_shell_quote(pipeline_cmd)}'
    )

    # Create SLURM script for single-node video generation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_content = f"""#!/bin/bash
#SBATCH --job-name=video_{sim_name[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output={REMOTE_PATH}/video_{timestamp}_%j.log
#SBATCH --error={REMOTE_PATH}/video_{timestamp}_%j.log

cd {REMOTE_PATH}
{container_cmd}
"""

    # Upload SLURM script
    script_name = f'video_{timestamp}.sh'
    local_script = Path('/tmp') / script_name
    local_script.write_text(script_content)

    remote_dest = f'{REMOTE_HOST}:{REMOTE_PATH}/{script_name}'
    result = subprocess.run(
        ['scp', str(local_script), remote_dest],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f'Failed to upload video script: {result.stderr}')

    # Submit
    stdout, stderr, rc = _run_ssh(
        f'cd {REMOTE_PATH} && sbatch {script_name}',
        timeout=30
    )
    if rc != 0:
        raise RuntimeError(f'sbatch failed: {stderr}')

    match = re.search(r'Submitted batch job (\d+)', stdout)
    if not match:
        raise RuntimeError(f'Could not parse job ID from: {stdout}')

    job_id = match.group(1)
    return {
        'job_id': job_id,
        'log_file': f'video_{timestamp}_{job_id}.log',
        'script': script_name,
        'sim_name': sim_name,
    }


def check_video_job(job_id, log_file=None):
    """Check video generation job status and parse progress from log."""
    status = check_job_status(job_id)

    progress = 0
    message = ''
    video_filename = None

    if log_file:
        try:
            stdout, stderr, rc = _run_ssh(
                f'tail -20 {REMOTE_PATH}/{log_file} 2>/dev/null',
                timeout=15
            )
            if rc == 0 and stdout:
                for line in stdout.strip().split('\n'):
                    if line.startswith('PROGRESS:'):
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            progress = int(parts[1])
                            message = parts[2]
                    elif line.startswith('VIDEO_FILE:'):
                        video_filename = line.split(':', 1)[1].strip()
                    elif line.startswith('ERROR:'):
                        message = line.split(':', 1)[1].strip()
        except subprocess.TimeoutExpired:
            pass

    return {
        'status': status,
        'progress': progress,
        'message': message,
        'video_filename': video_filename,
    }


def download_video(video_filename, local_dest):
    """Download a generated video from Karolina."""
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    remote_file = f'{REMOTE_HOST}:{REMOTE_PATH}/viz/videos/{video_filename}'
    local_file = local_dest / video_filename

    result = subprocess.run(
        ['scp', remote_file, str(local_file)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f'Failed to download video: {result.stderr}')

    return str(local_file)


# --------------------- Remote Viz Data Generation ---------------------

def generate_remote_viz(sim_name, partition='qcpu', account='eu-26-11', membrane_only=True):
    """Submit a SLURM job to generate viz data on Karolina (no video rendering).

    Returns job info dict with job_id and log_file.
    """
    viz_dir = f'viz/data/{sim_name}'
    pylibs = '/home/fenics/.pylibs'

    membrane_flag = ' --membrane-only' if membrane_only else ''

    # Build the command: install h5py if needed, then run generate script
    gen_cmd = (
        f'pip install --target={pylibs} -q h5py 2>/dev/null; '
        f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
        f'python3 -u viz/scripts/generate_viz_from_output.py'
        f'{membrane_flag} '
        f'{sim_name} {viz_dir}'
    )

    container_cmd = (
        f'apptainer exec --bind {REMOTE_PATH}:/home/fenics '
        f'--pwd /home/fenics {DOLFINX_SIF} '
        f'bash -c {_shell_quote(gen_cmd)}'
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_content = f"""#!/bin/bash
#SBATCH --job-name=viz_{sim_name[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output={REMOTE_PATH}/vizgen_{timestamp}_%j.log
#SBATCH --error={REMOTE_PATH}/vizgen_{timestamp}_%j.log

cd {REMOTE_PATH}
unset SINGULARITY_BINDPATH SINGULARITYENV_LD_PRELOAD LD_PRELOAD
{container_cmd}
"""

    script_name = f'vizgen_{timestamp}.sh'
    local_script = Path('/tmp') / script_name
    local_script.write_text(script_content)

    remote_dest = f'{REMOTE_HOST}:{REMOTE_PATH}/{script_name}'
    result = subprocess.run(
        ['scp', str(local_script), remote_dest],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f'Failed to upload viz script: {result.stderr}')

    stdout, stderr, rc = _run_ssh(
        f'cd {REMOTE_PATH} && sbatch {script_name}',
        timeout=30
    )
    if rc != 0:
        raise RuntimeError(f'sbatch failed: {stderr}')

    match = re.search(r'Submitted batch job (\d+)', stdout)
    if not match:
        raise RuntimeError(f'Could not parse job ID from: {stdout}')

    job_id = match.group(1)
    return {
        'job_id': job_id,
        'log_file': f'vizgen_{timestamp}_{job_id}.log',
        'sim_name': sim_name,
    }


def check_viz_job(job_id, log_file=None):
    """Check viz generation job status and parse progress from log."""
    status = check_job_status(job_id)

    progress = 0
    message = ''

    if log_file:
        try:
            stdout, stderr, rc = _run_ssh(
                f'tail -20 {REMOTE_PATH}/{log_file} 2>/dev/null',
                timeout=15
            )
            if rc == 0 and stdout:
                for line in stdout.strip().split('\n'):
                    if line.startswith('PROGRESS:'):
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            progress = int(parts[1])
                            message = parts[2]
                    elif line.startswith('ERROR:'):
                        message = line.split(':', 1)[1].strip()
        except subprocess.TimeoutExpired:
            pass

    return {
        'status': status,
        'progress': progress,
        'message': message,
    }


def download_viz_data_streaming(sim_name, local_dest):
    """Download viz data directory from Karolina via streaming tar.

    Yields progress dicts similar to download_results_streaming.
    """
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    remote_viz_dir = f'viz/data/{sim_name}'

    # Get total size
    size_cmd = f'du -sb {REMOTE_PATH}/{remote_viz_dir} 2>/dev/null | cut -f1'
    stdout, stderr, rc = _run_ssh(size_cmd, timeout=30)
    if rc != 0 or not stdout.strip():
        yield {'type': 'error', 'message': f'Viz data not found on Karolina for {sim_name}'}
        return

    bytes_total_uncompressed = int(stdout.strip())
    bytes_total_estimate = int(bytes_total_uncompressed * 0.5)

    yield {'type': 'progress', 'bytes_done': 0, 'bytes_total': bytes_total_estimate,
           'file': 'Creating archive...'}

    tar_cmd = f'cd {REMOTE_PATH} && tar czf - {remote_viz_dir}'
    archive_path = local_dest / f'.{sim_name}_viz.tar.gz'

    try:
        process = subprocess.Popen(
            ['ssh', REMOTE_HOST, tar_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        bytes_done = 0
        chunk_size = 256 * 1024
        with open(archive_path, 'wb') as f:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_done += len(chunk)
                yield {'type': 'progress', 'bytes_done': bytes_done,
                       'bytes_total': bytes_total_estimate,
                       'file': f'Downloading viz data ({bytes_done // (1024*1024)} MB)...'}

        process.wait()
        if process.returncode != 0:
            stderr_out = process.stderr.read().decode()
            yield {'type': 'error', 'message': f'SSH tar failed: {stderr_out}'}
            return

        compressed_size = bytes_done
        yield {'type': 'progress', 'bytes_done': compressed_size,
               'bytes_total': compressed_size, 'file': 'Extracting...'}

        # Extract — the archive contains viz/data/<sim_name>/ so extract to project root
        result = subprocess.run(
            ['tar', 'xzf', str(archive_path), '-C', str(local_dest.parent.parent.parent)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            yield {'type': 'error', 'message': f'Extraction failed: {result.stderr}'}
            return

        yield {'type': 'complete',
               'message': f'Downloaded viz data ({compressed_size // (1024*1024)} MB)'}

    finally:
        if archive_path.exists():
            archive_path.unlink()
