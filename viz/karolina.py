"""
Karolina supercomputer SSH/SLURM backend.

Wraps subprocess calls to ssh/scp using ~/.ssh/config host alias 'karolina'.
Runs simulation and mesh conversion inside Apptainer (Singularity) containers.
"""

import subprocess
import re
import json
import os
from pathlib import Path

REMOTE_HOST = 'karolina'
REMOTE_PATH = '/scratch/project/eu-26-11/fritz/cardioEMI'
CONTAINERS_PATH = f'{REMOTE_PATH}/containers'
MESHES_PATH = f'{REMOTE_PATH}/meshes'
DATA_PATH = f'{REMOTE_PATH}/data'

# Container image names
DOLFINX_SIF = f'{CONTAINERS_PATH}/dolfinx-v0.9.0.sif'
GINKGO_SIF = f'{CONTAINERS_PATH}/dolfinx-ginkgo-bddc.sif'

# Karolina job state (mirrors simulation_state pattern)
karolina_state = {
    'job_id': None,
    'status': None,       # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT
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

        families = []
        for family_name in sorted(family_map):
            meshes = []
            for pts_path in family_map[family_name]:
                pts_name = pts_path.name
                mesh_name = pts_name.split('-')[0]
                elem_name = pts_name.replace('.pts', '.elem')

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
                          account='eu-26-11', solver_backend='petsc'):
    """Generate a SLURM batch script string for cardioEMI."""
    out_name = Path(config_file).stem.replace('input_', '') + '_sim'

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
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd {REMOTE_PATH}

# Use tcp provider - container's libfabric lacks Karolina's native OFI provider
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=1

# Run inside Apptainer container
# --cpu-bind=cores and --cpus-per-task spread ranks across NUMA domains
# for optimal memory bandwidth (16 ranks/node = 8 cpus/task = 1 per memory channel)
srun -n {total_ranks} --cpu-bind=cores --cpus-per-task={cpus_per_task} apptainer exec \\
    --bind {REMOTE_PATH}:/home/fenics \\
    --pwd /home/fenics \\
    {sif} \\
    bash -c 'unset CC CXX && export PYTHONPATH=/home/fenics/.pylibs:$PYTHONPATH && python3 -B -u main.py {config_file}' 2>&1 | tee {out_name}_slurm.log
"""
    return script, out_name


def submit_job(config_file, nodes=1, ntasks_per_node=128,
               walltime='01:00:00', partition='qcpu_exp',
               account='eu-26-11', solver_backend='petsc'):
    """Upload config and SLURM script, then submit via sbatch. Returns job ID."""
    if karolina_state['submitting']:
        raise RuntimeError('A submission is already in progress')

    karolina_state['submitting'] = True
    try:
        # Generate SLURM script
        script_content, out_name = generate_slurm_script(
            config_file, nodes, ntasks_per_node, walltime, partition, account,
            solver_backend
        )

        # Write script to temp file and upload
        script_name = 'run_cardioemi.sh'
        local_script = Path('/tmp') / script_name
        local_script.write_text(script_content)

        remote_dest = f'{REMOTE_HOST}:{REMOTE_PATH}/{script_name}'
        result = subprocess.run(
            ['scp', str(local_script), remote_dest],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f'Failed to upload SLURM script: {result.stderr}')

        # Submit job
        stdout, stderr, rc = _run_ssh(
            f'cd {REMOTE_PATH} && sbatch {script_name}',
            timeout=30
        )
        if rc != 0:
            raise RuntimeError(f'sbatch failed: {stderr}')

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r'Submitted batch job (\d+)', stdout)
        if not match:
            raise RuntimeError(f'Could not parse job ID from: {stdout}')

        job_id = match.group(1)

        # Update state
        karolina_state['job_id'] = job_id
        karolina_state['status'] = 'PENDING'
        karolina_state['config_file'] = config_file
        karolina_state['out_name'] = out_name
        karolina_state['num_ranks'] = nodes * ntasks_per_node

        return job_id

    finally:
        karolina_state['submitting'] = False


def check_job_status(job_id):
    """Check SLURM job status via squeue/sacct. Returns status string."""
    # First try squeue (for queued/running jobs)
    try:
        stdout, stderr, rc = _run_ssh(
            f'squeue -j {job_id} --noheader -o "%T"',
            timeout=15
        )
        if rc == 0 and stdout:
            status = stdout.strip().split('\n')[0].strip()
            if status:
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
            # sacct may return multiple lines (job + job steps); take the first
            status = stdout.strip().split('\n')[0].strip()
            if status:
                karolina_state['status'] = status
                return status
    except subprocess.TimeoutExpired:
        pass

    return karolina_state.get('status', 'UNKNOWN')


def cancel_job(job_id):
    """Cancel a SLURM job via scancel."""
    stdout, stderr, rc = _run_ssh(f'scancel {job_id}', timeout=15)
    if rc != 0:
        raise RuntimeError(f'scancel failed: {stderr}')
    karolina_state['status'] = 'CANCELLED'
    return True


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
    out_name = karolina_state.get('out_name', '')
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
    """Download simulation results with byte-level progress, yielding status dicts.

    Yields dicts: {'type': 'progress', 'bytes_done': int, 'bytes_total': int, 'file': str}
                  {'type': 'complete', 'message': str}
                  {'type': 'error', 'message': str}
    """
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    # List files with sizes in the remote results directory
    stdout, stderr, rc = _run_ssh(
        f'find {REMOTE_PATH}/{remote_out_name} -type f -exec stat --format="%s %n" {{}} +',
        timeout=30
    )
    if rc != 0:
        yield {'type': 'error', 'message': f'Failed to list remote files: {stderr}'}
        return

    # Parse file list: [(size, remote_path), ...]
    files = []
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        size_str, fpath = line.split(' ', 1)
        files.append((int(size_str), fpath))

    # Also collect IF_*.txt files with sizes
    num_ranks = karolina_state.get('num_ranks')
    if num_ranks:
        candidates = [f'{REMOTE_PATH}/IF_{i}.txt' for i in range(num_ranks)]
        file_list = ' '.join(candidates)
        stdout, _, rc = _run_ssh(
            f'stat --format="%s %n" {file_list} 2>/dev/null', timeout=10
        )
        if rc == 0 and stdout.strip():
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                size_str, fpath = line.split(' ', 1)
                files.append((int(size_str), fpath))

    if not files:
        yield {'type': 'error', 'message': 'No files found to download'}
        return

    bytes_total = sum(size for size, _ in files)
    bytes_done = 0
    base_remote = f'{REMOTE_PATH}/{remote_out_name}'

    yield {'type': 'progress', 'bytes_done': 0, 'bytes_total': bytes_total, 'file': ''}

    for size, remote_file in files:
        # Compute local path preserving directory structure
        if remote_file.startswith(base_remote):
            rel = remote_file[len(base_remote):].lstrip('/')
            local_file = local_dest / rel
        else:
            # IF files go to parent dir
            local_file = local_dest.parent / Path(remote_file).name

        local_file.parent.mkdir(parents=True, exist_ok=True)
        fname = Path(remote_file).name

        yield {'type': 'progress', 'bytes_done': bytes_done, 'bytes_total': bytes_total, 'file': fname}

        result = subprocess.run(
            ['scp', f'{REMOTE_HOST}:{remote_file}', str(local_file)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            yield {'type': 'error', 'message': f'Failed to download {fname}: {result.stderr}'}
            return

        bytes_done += size

    yield {'type': 'complete', 'message': f'Downloaded {len(files)} files ({bytes_total} bytes)'}


def list_remote_simulations():
    """List simulation output directories on the remote machine."""
    try:
        stdout, stderr, rc = _run_ssh(
            f'ls -d {REMOTE_PATH}/*_sim 2>/dev/null || true',
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
