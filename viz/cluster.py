"""
Generic remote HPC cluster backend (SSH/SLURM/Apptainer).

Each cluster is described by an entry in viz/clusters.yml (see that file for
the field reference) and wrapped in a `Cluster` instance. All ssh/scp/rsync
traffic runs through OpenSSH connection multiplexing (ControlMaster): the
first authenticated connection becomes a persistent background master and
every later call reuses it without re-authenticating. That is what makes
2FA/OTP clusters (Vega) workable — the OTP is entered exactly once, through
`connect_start()`/`connect_send()`, which drive an interactive ssh under a
pty (pexpect) and forward the prompt to the webapp.

Non-interactive calls always run with BatchMode=yes so they fail fast instead
of hanging on a prompt when no master is up.
"""

import subprocess
import re
import json
import os
import hashlib
import shutil
import yaml
import threading
import time
from datetime import datetime
from pathlib import Path

# Directory for ControlMaster sockets (0700, like ~/.ssh)
CM_DIR = Path.home() / '.ssh' / 'cardioemi-cm'

POLL_INTERVAL = 5  # seconds between background poll cycles

TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'}

# Container image filenames (same names on every cluster, under <remote_path>/containers/)
DOLFINX_SIF_NAME = 'dolfinx-v0.9.0.sif'
GINKGO_SIF_NAME = 'dolfinx-ginkgo-bddc.sif'
DOLFINX_DOCKER_REF = 'docker://ghcr.io/fenics/dolfinx/dolfinx:v0.9.0'

# Python packages the SIFs lack. Jobs bind the project at /home/fenics and
# prepend .pylibs to PYTHONPATH, so they live there (same set as Karolina's
# restore-pylibs.sh). multiphenicsx goes in --no-deps so pip never replaces
# the image's dolfinx.
PYLIBS_PATH = '/home/fenics/.pylibs'
PYLIBS_PACKAGES = ['PyYAML', 'scipy', 'matplotlib', 'h5py', 'pyvista',
                   'imageio[ffmpeg]', 'Pillow', 'lxml', 'pymetis']
MULTIPHENICSX_REQ = ('multiphenicsx @ '
                     'git+https://github.com/multiphenics/multiphenicsx.git@v0.3.9')

# rsync exclude list for syncing project code to a cluster
SYNC_EXCLUDES = [
    '*_sim*/', 'data/', 'viz/data/', 'viz/videos/', '__pycache__/', '.git/',
    '.venv/', '*.pyc', '*.h5', '*.xdmf', '*.pickle', 'IF_*.txt',
    'SESSION_SUMMARY.md', 'test_*.py', 'containers/', '*.sif', '.DS_Store',
]


def _shell_quote(s):
    """Quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"


class Cluster:
    """One remote HPC cluster: ssh transport + SLURM job management +
    remote mesh/video/viz operations, all inside Apptainer containers."""

    def __init__(self, cluster_id, cfg):
        self.id = cluster_id
        self.cfg = cfg
        self.label = cfg.get('label', cluster_id)
        self.host = cfg['host']
        self.user = cfg.get('user')
        self.identity_file = cfg.get('identity_file')
        self.needs_otp = bool(cfg.get('needs_otp', False))
        self.remote_path = cfg['remote_path'].rstrip('/')
        self.env_unset = list(cfg.get('env_unset') or [])
        self.job_env = dict(cfg.get('job_env') or {})
        self.srun_flags = cfg.get('srun_flags', '')

        slurm = cfg.get('slurm') or {}
        self.default_account = slurm.get('account', '')
        self.default_partition = slurm.get('partition', '')
        self.partitions = list(slurm.get('partitions') or [])
        self.default_ntasks_per_node = int(slurm.get('ntasks_per_node', 128))
        self.cores_per_node = int(slurm.get('cores_per_node', 128))
        self.default_walltime = str(slurm.get('walltime', '01:00:00'))

        self.containers_path = f'{self.remote_path}/containers'
        self.meshes_path = f'{self.remote_path}/meshes'
        self.data_path = f'{self.remote_path}/data'
        self.dolfinx_sif = f'{self.containers_path}/{DOLFINX_SIF_NAME}'
        self.ginkgo_sif = f'{self.containers_path}/{GINKGO_SIF_NAME}'

        # Multi-job state: keyed by SLURM job_id
        self.jobs = {}
        # Legacy single-job alias (backward compat in status/download routes)
        self.legacy_state = {
            'job_id': None, 'status': None, 'config_file': None,
            'out_name': None, 'num_ranks': None, 'submitting': False,
        }
        self.mesh_convert_state = {'converting': False, 'process': None}

        # Background poller
        self._poll_cache = {}
        self._poll_lock = threading.Lock()
        self._poll_thread = None
        self._poll_stop = threading.Event()

        # Interactive connect (OTP) state
        self._connect_lock = threading.Lock()
        self._connect_answer = None
        self._connect_answer_event = threading.Event()
        self._connect_token = None   # identifies the current login attempt
        self._connect_child = None   # its pexpect child, so a cancel can kill it
        self._connect_since = 0.0    # when its phase last changed
        self.connect_state = {'phase': 'idle', 'prompt': '', 'error': '', 'log': ''}

        # Install state (background thread appends lines; route streams them)
        self.install_state = {'running': False, 'log': [], 'error': None, 'done': False}

    # --------------------- Public description ---------------------

    def to_dict(self, connected=None):
        d = {
            'id': self.id,
            'label': self.label,
            'host': self.host,
            'user': self.user,
            'identity_file': self.identity_file,
            'needs_otp': self.needs_otp,
            'remote_path': self.remote_path,
            'slurm': {
                'account': self.default_account,
                'partition': self.default_partition,
                'partitions': self.partitions,
                'ntasks_per_node': self.default_ntasks_per_node,
                'cores_per_node': self.cores_per_node,
                'walltime': self.default_walltime,
            },
        }
        if connected is not None:
            d['connected'] = connected
        return d

    # --------------------- SSH transport ---------------------

    def _control_opts(self):
        CM_DIR.mkdir(mode=0o700, exist_ok=True)
        return [
            '-o', 'ControlMaster=auto',
            '-o', f'ControlPath={CM_DIR}/%C',
            '-o', 'ControlPersist=yes',
        ]

    def _ssh_opts(self, batch=True):
        opts = self._control_opts()
        opts += ['-o', 'ServerAliveInterval=30', '-o', 'StrictHostKeyChecking=accept-new']
        if batch:
            opts += ['-o', 'BatchMode=yes']
        if self.identity_file:
            opts += ['-i', os.path.expanduser(self.identity_file), '-o', 'IdentitiesOnly=yes']
        return opts

    def ssh_dest(self):
        return f'{self.user}@{self.host}' if self.user else self.host

    NOT_CONNECTED = 'not connected - use Connect (this cluster needs an OTP)'

    def _offline(self):
        """True for an OTP cluster without its master connection. Such a
        cluster only accepts keyboard-interactive logins, so a BatchMode ssh can
        never succeed - and every attempt is a failed login that counts against
        sshd's MaxStartups / intrusion limits (Vega then stops answering, even
        for the real OTP connect). So don't try: fail locally instead."""
        return self.needs_otp and not self.master_alive()

    def _run_ssh(self, cmd, timeout=30):
        """Run a command on the cluster via SSH. Returns (stdout, stderr, returncode)."""
        if self._offline():
            return '', self.NOT_CONNECTED, 255
        full_cmd = ['ssh'] + self._ssh_opts() + [self.ssh_dest(), cmd]
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    def _popen_ssh(self, cmd, **kwargs):
        """Popen an SSH command (for streaming output / piping)."""
        if self._offline():
            raise RuntimeError(self.NOT_CONNECTED)
        full_cmd = ['ssh'] + self._ssh_opts() + [self.ssh_dest(), cmd]
        return subprocess.Popen(full_cmd, **kwargs)

    def _offline_result(self, cmd):
        return subprocess.CompletedProcess(cmd, 255, stdout='', stderr=self.NOT_CONNECTED)

    def _scp(self, sources, remote_dest, timeout=60, recursive=False):
        """scp local file(s) to a remote path (or remote->local when
        remote_src=True style paths are passed explicitly by the caller)."""
        if isinstance(sources, (str, Path)):
            sources = [str(sources)]
        cmd = ['scp'] + self._ssh_opts()
        if recursive:
            cmd.append('-r')
        cmd += [str(s) for s in sources] + [remote_dest]
        if self._offline():
            return self._offline_result(cmd)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def scp_upload(self, sources, remote_path, timeout=60, recursive=False):
        return self._scp(sources, f'{self.ssh_dest()}:{remote_path}',
                         timeout=timeout, recursive=recursive)

    def scp_download(self, remote_path, local_path, timeout=120):
        cmd = ['scp'] + self._ssh_opts() + [f'{self.ssh_dest()}:{remote_path}', str(local_path)]
        if self._offline():
            return self._offline_result(cmd)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _apptainer_exec(self, sif, cmd):
        """Build an apptainer exec command string for running inside a container."""
        return (
            f'apptainer exec --bind {self.remote_path}:/home/fenics '
            f'--pwd /home/fenics {sif} '
            f'bash -c {_shell_quote(cmd)}'
        )

    def _env_unset_prefix(self):
        """Shell prefix that unsets host env vars which break apptainer."""
        if not self.env_unset:
            return ''
        return 'unset ' + ' '.join(self.env_unset) + '; '

    # --------------------- Connectivity / containers ---------------------

    def check_ssh(self):
        """Test SSH connectivity. Returns True if reachable without a prompt."""
        try:
            stdout, stderr, rc = self._run_ssh('echo ok', timeout=10)
            return rc == 0 and 'ok' in stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def master_alive(self):
        """True when a ControlMaster connection for this cluster is up."""
        cmd = ['ssh'] + self._control_opts() + ['-O', 'check', self.ssh_dest()]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def disconnect(self):
        """Tear down the ControlMaster connection (next connect needs OTP again)."""
        cmd = ['ssh'] + self._control_opts() + ['-O', 'exit', self.ssh_dest()]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass
        self.connect_state = {'phase': 'idle', 'prompt': '', 'error': '', 'log': ''}
        return True

    def check_containers(self):
        """Check which container images are available on the cluster."""
        result = {}
        try:
            stdout, stderr, rc = self._run_ssh(
                f'ls {self.containers_path}/*.sif 2>/dev/null || true', timeout=15)
            result['dolfinx'] = DOLFINX_SIF_NAME in (stdout or '')
            result['ginkgo'] = GINKGO_SIF_NAME in (stdout or '')
        except (subprocess.TimeoutExpired, OSError):
            result['dolfinx'] = False
            result['ginkgo'] = False
        return result

    # --------------------- Interactive connect (OTP) ---------------------

    def connect_start(self):
        """Open the persistent master connection interactively.

        Runs ssh under a pty in a background thread. Any password/OTP prompt is
        surfaced in connect_state['prompt']; the webapp answers via
        connect_send(). On success the master persists (ControlPersist) and all
        subsequent calls are prompt-free.
        """
        with self._connect_lock:
            # Reuse a login in progress - unless it is stale: an OTP prompt the
            # user walked away from has long been dropped by the server, and
            # answering it only burns an OTP.
            age = time.time() - getattr(self, '_connect_since', 0)
            busy = self.connect_state['phase'] in ('connecting', 'prompt')
            if busy and age < (60 if self.connect_state['phase'] == 'prompt' else 120):
                return self.connect_state
            if busy:
                self._cancel_connect_locked()
            if self.master_alive():
                self.connect_state = {'phase': 'connected', 'prompt': '', 'error': '', 'log': ''}
                return self.connect_state
            self.connect_state = {'phase': 'connecting', 'prompt': '', 'error': '', 'log': ''}
            self._connect_since = time.time()
            self._connect_answer = None
            self._connect_answer_event.clear()
            # A fresh token per login: a cancelled worker sees it changed and
            # stops instead of touching the new login's state.
            self._connect_token = token = object()
            t = threading.Thread(target=self._connect_worker, args=(token,), daemon=True)
            t.start()
        return self.connect_state

    def connect_cancel(self):
        """Abandon a login in progress (the user closed the OTP window)."""
        with self._connect_lock:
            self._cancel_connect_locked()
            self.connect_state = {'phase': 'idle', 'prompt': '', 'error': '', 'log': ''}
        return self.connect_state

    def _cancel_connect_locked(self):
        self._connect_token = None
        child = getattr(self, '_connect_child', None)
        self._connect_child = None
        self._connect_answer_event.set()  # wake a worker waiting for the OTP
        if child is not None:
            try:
                child.close(force=True)
            except Exception:
                pass

    def connect_send(self, text):
        """Deliver the user's answer (OTP, password, 'yes') to the waiting ssh."""
        self._connect_answer = text
        self._connect_answer_event.set()
        return self.connect_state

    def _connect_worker(self, token):
        def live():
            return self._connect_token is token

        def set_state(**kw):
            if live():
                if kw.get('phase') in ('prompt', 'connecting'):
                    self._connect_since = time.time()
                self.connect_state.update(**kw)

        try:
            import pexpect
        except ImportError:
            set_state(
                phase='failed',
                error="pexpect is not installed on the machine running the viz "
                      "server — run: pip install pexpect")
            return

        argv = (['ssh'] + self._ssh_opts(batch=False) +
                ['-tt', self.ssh_dest(), 'echo CARDIOEMI_CONNECT_OK'])
        try:
            child = pexpect.spawn(argv[0], argv[1:], encoding='utf-8',
                                  timeout=60, maxread=4096)
        except Exception as e:
            set_state(phase='failed', error=f'ssh spawn failed: {e}')
            return
        self._connect_child = child
        if not live():  # cancelled while spawning
            child.close(force=True)
            return

        patterns = [
            'CARDIOEMI_CONNECT_OK',                                       # 0 success
            re.compile(r'(?i)(one[ -]?time|verification|passcode|otp|token|2fa)[^\r\n]*:\s*'),  # 1 OTP
            re.compile(r'(?i)passphrase[^\r\n]*:\s*'),                    # 2 key passphrase
            re.compile(r'(?i)password[^\r\n]*:\s*'),                      # 3 password
            re.compile(r'(?i)continue connecting[^\r\n]*\?\s*'),          # 4 host key
            pexpect.EOF,                                                  # 5
            pexpect.TIMEOUT,                                              # 6
        ]

        try:
            while True:
                idx = child.expect(patterns, timeout=90)
                if idx == 0:
                    # Drain until the client exits; the background master persists.
                    try:
                        child.expect(pexpect.EOF, timeout=30)
                    except Exception:
                        pass
                    child.close()
                    if self.master_alive():
                        set_state(phase='connected', prompt='')
                    else:
                        set_state(
                            phase='failed',
                            error='ssh succeeded but no master connection persisted')
                    return
                if idx in (1, 2, 3, 4):
                    raw = ((child.before or '') + (child.after or '')).replace('\r', '')
                    lines = [l for l in raw.split('\n') if l.strip()]
                    prompt_text = '\n'.join(lines[-6:]).strip()
                    self._connect_answer_event.clear()
                    set_state(phase='prompt', prompt=prompt_text)
                    if not self._connect_answer_event.wait(timeout=600) or not live():
                        if not live():
                            return  # cancelled: the canceller closed the child
                        set_state(phase='failed', error='Timed out waiting for input')
                        child.close(force=True)
                        return
                    answer = self._connect_answer or ''
                    self._connect_answer = None
                    set_state(phase='connecting', prompt='')
                    child.sendline(answer)
                    continue
                if idx == 5:
                    tail = ((child.before or '')[-500:]).strip()
                    child.close()
                    set_state(
                        phase='failed',
                        error=f'ssh exited before authenticating: {tail}')
                    return
                if idx == 6:
                    child.close(force=True)
                    set_state(phase='failed', error='ssh timed out')
                    return
        except Exception as e:
            try:
                child.close(force=True)
            except Exception:
                pass
            set_state(phase='failed', error=str(e))  # no-op once cancelled

    # --------------------- Install (dirs + code + containers) ---------------------

    def rsync_code(self, project_root, on_line=None, delete=False):
        """rsync the project source tree to the cluster (same exclude list as
        scripts/sync_to_karolina.sh). Returns (ok, tail_of_output)."""
        if self._offline():
            return False, self.NOT_CONNECTED
        cmd = ['rsync', '-az', '--out-format=%n']
        for ex in SYNC_EXCLUDES:
            cmd += ['--exclude', ex]
        ssh_cmd = 'ssh ' + ' '.join(self._ssh_opts())
        cmd += ['-e', ssh_cmd, f'{str(project_root).rstrip("/")}/',
                f'{self.ssh_dest()}:{self.remote_path}/']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip('\n')
            if line:
                tail.append(line)
                if on_line:
                    on_line(line)
        proc.wait()
        return proc.returncode == 0, '\n'.join(tail[-20:])

    def remote_file_size(self, path):
        """Size in bytes of a remote file, or None."""
        try:
            stdout, _, rc = self._run_ssh(
                f'stat -c %s {_shell_quote(path)} 2>/dev/null || '
                f'stat -f %z {_shell_quote(path)} 2>/dev/null', timeout=15)
            return int(stdout.strip()) if rc == 0 and stdout.strip().isdigit() else None
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return None

    def install_stream(self, project_root, registry):
        """Generator: set up this cluster end to end, yielding SSE-able dicts.

        Steps: directory layout -> code rsync -> container SIFs -> final check.
        SIF sources, in order of preference:
          1. a local copy under <project_root>/containers/
          2. another registered cluster that already has the file (streamed
             through this machine, nothing kept locally)
          3. for the plain DOLFINx image only: `apptainer pull docker://...`
             run directly on the target cluster
        """
        def out(text):
            return {'type': 'output', 'text': text + '\n'}

        # 1. Directory layout
        yield out(f'[{self.label}] Creating directory layout under {self.remote_path} ...')
        _, stderr, rc = self._run_ssh(
            f'mkdir -p {self.remote_path} {self.containers_path} {self.meshes_path} '
            f'{self.data_path} {self.remote_path}/viz/videos {self.remote_path}/viz/data',
            timeout=20)
        if rc != 0:
            yield {'type': 'error', 'message': f'mkdir failed: {stderr}'}
            return

        # 2. Code sync
        yield out('Syncing project code (rsync) ...')
        ok, tail = self.rsync_code(project_root)
        if not ok:
            yield {'type': 'error', 'message': f'rsync failed:\n{tail}'}
            return
        yield out('Code sync complete.')

        # 3. Containers
        for kind, sif_name, remote_sif in [
            ('dolfinx', DOLFINX_SIF_NAME, self.dolfinx_sif),
            ('ginkgo', GINKGO_SIF_NAME, self.ginkgo_sif),
        ]:
            if self.remote_file_size(remote_sif):
                yield out(f'{sif_name}: already present, skipping.')
                continue

            local_sif = Path(project_root) / 'containers' / sif_name
            if local_sif.exists():
                yield out(f'{sif_name}: uploading local copy '
                          f'({local_sif.stat().st_size // (1024*1024)} MB) ...')
                result = self.scp_upload(local_sif, f'{remote_sif}.part', timeout=3600)
                if result.returncode != 0:
                    yield {'type': 'error', 'message': f'upload failed: {result.stderr}'}
                    return
                self._run_ssh(f'mv {remote_sif}.part {remote_sif}', timeout=15)
                yield out(f'{sif_name}: uploaded.')
                continue

            # Try streaming from another cluster that has it
            source = None
            for other in registry.clusters():
                if other.id == self.id:
                    continue
                try:
                    if other.check_ssh() and other.remote_file_size(
                            f'{other.containers_path}/{sif_name}'):
                        source = other
                        break
                except Exception:
                    continue

            if source is not None:
                src_path = f'{source.containers_path}/{sif_name}'
                total = source.remote_file_size(src_path) or 0
                yield out(f'{sif_name}: streaming from {source.label} '
                          f'({total // (1024*1024)} MB, via this machine, no local copy) ...')
                yield from self._relay_file(source, src_path, remote_sif, total, sif_name)
                continue

            if kind == 'dolfinx':
                yield out(f'{sif_name}: no local or cluster copy found — pulling '
                          f'{DOLFINX_DOCKER_REF} on {self.label} (this can take a while) ...')
                pull_cmd = (
                    f'{self._env_unset_prefix()}cd {self.containers_path} && '
                    f'apptainer pull {sif_name} {DOLFINX_DOCKER_REF}'
                )
                proc = self._popen_ssh(pull_cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in iter(proc.stdout.readline, ''):
                    if line.strip():
                        yield out('  ' + line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    yield {'type': 'error', 'message': 'apptainer pull failed'}
                    return
                yield out(f'{sif_name}: pulled.')
            else:
                yield out(f'{sif_name}: NOT INSTALLED — no local copy under containers/ '
                          f'and no other cluster has it. The Ginkgo backend will be '
                          f'unavailable on {self.label} until it is provided.')

        # 4. Python packages (.pylibs)
        yield from self._install_pylibs(out)

        # 5. Final check
        containers = self.check_containers()
        yield {'type': 'complete', 'success': True, 'containers': containers}

    def _pylibs_cmd(self, py):
        """Shell command running `py` in the DOLFINx SIF the way a job does.
        PYTHONPATH must survive into the container: the image's own entry is
        what makes dolfinx importable (host PYTHONPATH is dropped first)."""
        inner = (f'unset CC CXX FC BOOST_ROOT; export FI_PROVIDER=tcp '
                 f'PYTHONPATH={PYLIBS_PATH}:$PYTHONPATH; {py}')
        return (self._env_unset_prefix() + 'unset PYTHONPATH; '
                + self._apptainer_exec(self.dolfinx_sif, inner))

    def _install_pylibs(self, out):
        """Install main.py's Python dependencies into .pylibs unless
        multiphenicsx already imports there. Yields SSE dicts."""
        probe = self._pylibs_cmd(
            'python3 -B -c "import multiphenicsx.fem.petsc, yaml, scipy; print(\'PYLIBS_OK\')"')
        stdout, _, _ = self._run_ssh(probe, timeout=180)
        if 'PYLIBS_OK' in stdout:
            yield out('.pylibs: multiphenicsx and dependencies present, skipping.')
            return

        yield out('.pylibs: installing Python dependencies + multiphenicsx v0.3.9 '
                  '(builds C++ - a few minutes) ...')
        pip = (f'python3 -m pip install --disable-pip-version-check -q '
               f'--target={PYLIBS_PATH}')
        pkgs = ' '.join(_shell_quote(p) for p in PYLIBS_PACKAGES)
        install = self._pylibs_cmd(
            f'{pip} {pkgs} && '
            f'{pip} --no-deps --no-build-isolation {_shell_quote(MULTIPHENICSX_REQ)} && '
            f'python3 -B -c "import multiphenicsx.fem.petsc; print(\'PYLIBS_OK\')"')
        proc = self._popen_ssh(install, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
        ok = False
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip()
            ok = ok or 'PYLIBS_OK' in line
            if line and 'lmod' not in line and 'PYLIBS_OK' not in line:
                yield out('  ' + line)
        proc.wait()
        if not ok:
            yield out('.pylibs: INSTALL FAILED - jobs will die with '
                      '"No module named multiphenicsx" until this is fixed.')
            return
        yield out('.pylibs: installed, multiphenicsx imports.')

    def _relay_file(self, source, src_path, dst_path, total, name):
        """Stream a file from another cluster to this one through this machine."""
        p_src = source._popen_ssh(f'cat {_shell_quote(src_path)}', stdout=subprocess.PIPE)
        p_dst = self._popen_ssh(
            f"cat > {_shell_quote(dst_path + '.part')} && "
            f"mv {_shell_quote(dst_path + '.part')} {_shell_quote(dst_path)}",
            stdin=subprocess.PIPE)
        done = 0
        last_reported = 0
        try:
            while True:
                chunk = p_src.stdout.read(1024 * 1024)
                if not chunk:
                    break
                p_dst.stdin.write(chunk)
                done += len(chunk)
                if done - last_reported >= 64 * 1024 * 1024:
                    last_reported = done
                    pct = f' ({100 * done // total}%)' if total else ''
                    yield {'type': 'output',
                           'text': f'  {name}: {done // (1024*1024)} MB{pct}\n'}
            p_dst.stdin.close()
            p_src.wait()
            p_dst.wait()
            if p_src.returncode != 0 or p_dst.returncode != 0:
                yield {'type': 'error', 'message': f'streaming {name} failed '
                       f'(src rc={p_src.returncode}, dst rc={p_dst.returncode})'}
                return
            yield {'type': 'output', 'text': f'  {name}: transfer complete '
                   f'({done // (1024*1024)} MB).\n'}
        finally:
            for p in (p_src, p_dst):
                if p.poll() is None:
                    p.kill()

    # --------------------- Background job poller ---------------------

    def get_cached_status(self, job_id):
        with self._poll_lock:
            entry = self._poll_cache.get(job_id)
            if entry:
                return entry['status'], entry['log']
        return None, None

    def start_background_poller(self):
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_active_jobs, daemon=True)
        self._poll_thread.start()

    def _poll_active_jobs(self):
        """Background thread: periodically check all active jobs via batched SSH."""
        while not self._poll_stop.is_set():
            with self._poll_lock:
                active = {jid: self.jobs[jid] for jid in list(self.jobs)
                          if self._poll_cache.get(jid, {}).get('status') not in TERMINAL_STATES}

            if active:
                job_ids = list(active.keys())
                job_ids_str = ','.join(job_ids)

                statuses = {}
                try:
                    stdout, _, rc = self._run_ssh(
                        f'squeue -j {job_ids_str} --noheader -o "%i %T" 2>/dev/null; '
                        f'sacct -j {job_ids_str} --noheader -o JobID,State -P 2>/dev/null',
                        timeout=20)
                    if rc == 0 and stdout:
                        for line in stdout.strip().split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split('|') if '|' in line else line.split()
                            if len(parts) >= 2:
                                jid = parts[0].strip().split('.')[0]
                                st = parts[1].strip()
                                if jid in active and st:
                                    statuses[jid] = st
                except (subprocess.TimeoutExpired, Exception):
                    pass

                logs = {}
                tail_parts = []
                for jid in job_ids:
                    job = active[jid]
                    out_name = job.get('out_name', '')
                    if out_name:
                        tail_parts.append(
                            f'echo "@@JOB {jid}@@"; '
                            f'tail -n 50 {self.remote_path}/{out_name}_slurm.log 2>/dev/null || '
                            f'tail -n 50 {self.remote_path}/slurm_{jid}.out 2>/dev/null || true')
                    else:
                        tail_parts.append(
                            f'echo "@@JOB {jid}@@"; '
                            f'tail -n 50 {self.remote_path}/slurm_{jid}.out 2>/dev/null || true')
                try:
                    combined = '; '.join(tail_parts)
                    stdout, _, rc = self._run_ssh(combined, timeout=20)
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

                with self._poll_lock:
                    for jid in job_ids:
                        entry = self._poll_cache.setdefault(jid, {'status': 'PENDING', 'log': ''})
                        if jid in statuses:
                            entry['status'] = statuses[jid]
                            if jid in self.jobs:
                                self.jobs[jid]['status'] = statuses[jid]
                        if jid in logs:
                            entry['log'] = logs[jid]

            self._poll_stop.wait(POLL_INTERVAL)

    # --------------------- Remote mesh operations ---------------------

    def fetch_mesh_metadata(self, mesh_name):
        """Fetch mesh bounding box and metadata without downloading the mesh."""
        pylibs = '/home/fenics/.pylibs'
        original_name = mesh_name[:-len('_colored')] if mesh_name.endswith('_colored') else None
        script = (
            f'mesh_name = {mesh_name!r}\n'
            f'original_name = {original_name!r}\n'
            'import h5py, json, numpy as np, pickle, os\n'
            'def _tag_count(name):\n'
            '    p = f"data/{name}.pickle"\n'
            '    return len(pickle.load(open(p, "rb"))) if os.path.exists(p) else None\n'
            'num_tags = _tag_count(mesh_name)\n'
            'num_original_tags = _tag_count(original_name) if original_name else None\n'
            'num_components = num_original_tags // 2 if num_original_tags is not None else None\n'
            'f = h5py.File(f"data/{mesh_name}.h5", "r")\n'
            'v = f["/Mesh/mesh/geometry"][:]\n'
            'tags = f["/Mesh/facet_tags/Values"][:]\n'
            'mt = tags[tags > 0]\n'
            'ext = max(v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min())\n'
            'cf = 0.0001 if ext > 10 else 1.0\n'
            'out = {"bounds": {"x": [float(v[:,0].min()), float(v[:,0].max())], '
            '"y": [float(v[:,1].min()), float(v[:,1].max())], '
            '"z": [float(v[:,2].min()), float(v[:,2].max())]}, '
            '"mesh_conversion_factor": cf, "vertex_count": len(v), '
            '"facet_count": int((tags > 0).sum()), '
            '"unique_tags": sorted(set(int(t) for t in mt)), '
            '"num_tags": num_tags, "num_original_tags": num_original_tags, '
            '"num_components": num_components}\n'
            'print(json.dumps(out))\n'
        )
        container_cmd = (
            f'pip install --target={pylibs} -q h5py 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 -c {_shell_quote(script)}'
        )
        full_cmd = self._env_unset_prefix() + self._apptainer_exec(self.dolfinx_sif, container_cmd)
        stdout, stderr, rc = self._run_ssh(full_cmd, timeout=120)

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

    def fetch_batch_mesh_info(self, mesh_names):
        """Bounds, conversion factor and tag counts for several meshes in one
        container launch (one SSH call). Same fields as fetch_mesh_metadata
        minus the facet scan, which batch submission doesn't need.

        Returns {mesh_name: {...}}; a mesh that can't be read maps to
        {'error': str} instead of failing the whole call.
        """
        pylibs = '/home/fenics/.pylibs'
        script = (
            f'names = {list(mesh_names)!r}\n'
            'import h5py, json, os, pickle\n'
            'import numpy as np\n'
            'def _tag_count(name):\n'
            '    p = f"data/{name}.pickle"\n'
            '    return len(pickle.load(open(p, "rb"))) if os.path.exists(p) else None\n'
            'out = {}\n'
            'for name in names:\n'
            '    try:\n'
            '        with h5py.File(f"data/{name}.h5", "r") as f:\n'
            '            g = f["/Mesh/mesh/geometry"]\n'
            '            lo = hi = None\n'
            '            for i in range(0, g.shape[0], 1 << 22):\n'  # bounded memory on GB meshes
            '                v = g[i:i + (1 << 22)]\n'
            '                lo = v.min(0) if lo is None else np.minimum(lo, v.min(0))\n'
            '                hi = v.max(0) if hi is None else np.maximum(hi, v.max(0))\n'
            '        ext = float((hi - lo).max())\n'
            '        orig = name[:-len("_colored")] if name.endswith("_colored") else None\n'
            '        n_orig = _tag_count(orig) if orig else None\n'
            '        out[name] = {\n'
            '            "bounds": {a: [float(lo[i]), float(hi[i])] for i, a in enumerate("xyz")},\n'
            '            "mesh_conversion_factor": 0.0001 if ext > 10 else 1.0,\n'
            '            "num_tags": _tag_count(name),\n'
            '            "num_original_tags": n_orig,\n'
            '            "num_components": n_orig // 2 if n_orig is not None else None}\n'
            '    except Exception as e:\n'
            '        out[name] = {"error": f"{type(e).__name__}: {e}"}\n'
            'print("@@BATCHINFO " + json.dumps(out))\n'
        )
        container_cmd = (
            f'pip install --target={pylibs} -q h5py 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 -c {_shell_quote(script)}'
        )
        full_cmd = self._env_unset_prefix() + self._apptainer_exec(self.dolfinx_sif, container_cmd)
        stdout, stderr, rc = self._run_ssh(full_cmd, timeout=300)

        for line in (stdout + '\n' + stderr).split('\n'):
            line = line.strip()
            if line.startswith('@@BATCHINFO '):
                return json.loads(line[len('@@BATCHINFO '):])
        raise RuntimeError(f'Failed to fetch mesh info: {(stderr or stdout)[-500:]}')

    def build_mesh_preview(self, mesh_name, local_dir, max_facets):
        """Membrane-only preview of a cluster mesh, built next to the mesh and
        streamed into local_dir (see viz/scripts/mesh_preview.py, sent inline so
        it doesn't depend on the cluster's code copy being synced). The cluster
        keeps its copy in viz/previews/<mesh>/ and reuses it while it is newer
        than the mesh and was built for the same triangle budget.
        Returns (metadata dict, cached_on_cluster bool)."""
        if not re.fullmatch(r'[A-Za-z0-9_.\-]+', mesh_name):
            raise ValueError(f'invalid mesh name: {mesh_name!r}')
        script = (Path(__file__).parent / 'scripts' / 'mesh_preview.py').read_text()
        pylibs = '/home/fenics/.pylibs'
        rel = f'viz/previews/{mesh_name}'
        container_cmd = (
            f'pip install --target={pylibs} -q h5py 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 -c {_shell_quote(script)} data/{mesh_name}.h5 {rel} '
            f'--max-facets {int(max_facets)}'
        )
        cmd = (
            f'cd {_shell_quote(self.remote_path)} && '
            f'[ -f data/{mesh_name}.h5 ] || {{ echo "@@ERR no data/{mesh_name}.h5"; exit 3; }}; '
            f'if [ {rel}/mesh_metadata.json -nt data/{mesh_name}.h5 ] && '
            f'grep -q \'"max_facets": {int(max_facets)},\' {rel}/mesh_metadata.json; '
            f'then echo @@CACHED; else mkdir -p {rel} && '
            + self._env_unset_prefix() + self._apptainer_exec(self.dolfinx_sif, container_cmd)
            + '; fi'
        )
        stdout, stderr, rc = self._run_ssh(cmd, timeout=900)
        if '@@ERR' in stdout or (rc != 0 and '@@PREVIEW' not in stdout):
            msg = next((l for l in stdout.split('\n') if '@@ERR' in l), '') or stderr or stdout
            raise RuntimeError(f'preview failed: {msg.strip()[-400:]}')
        cached = '@@CACHED' in stdout

        # Stream the (small) preview back as a gzipped tar, into a temp dir
        # first so a broken transfer never leaves a half preview behind.
        local_dir = Path(local_dir)
        tmp = local_dir.with_name(local_dir.name + '.part')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        src = self._popen_ssh(f'tar czf - -C {_shell_quote(self.remote_path + "/" + rel)} .',
                              stdout=subprocess.PIPE)
        untar = subprocess.run(['tar', 'xzf', '-', '-C', str(tmp)], stdin=src.stdout,
                               capture_output=True, timeout=600)
        src.wait(timeout=60)
        if src.returncode != 0 or untar.returncode != 0 or not (tmp / 'mesh_metadata.json').exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f'preview download failed: {untar.stderr.decode()[-300:]}')
        shutil.rmtree(local_dir, ignore_errors=True)
        tmp.rename(local_dir)
        with open(local_dir / 'mesh_metadata.json') as f:
            return json.load(f), cached

    def list_remote_meshes(self):
        """List mesh families under meshes/ on the cluster (single SSH call)."""
        try:
            cmd = (
                f'find {self.meshes_path} -name "*.pts" -type f 2>/dev/null; '
                f'echo "---SEPARATOR---"; '
                f'ls {self.data_path}/*.h5 2>/dev/null || true'
            )
            stdout, stderr, rc = self._run_ssh(cmd, timeout=30)
            if rc != 0 or not stdout:
                return []

            parts = stdout.split('---SEPARATOR---')
            pts_section = parts[0].strip() if len(parts) > 0 else ''
            h5_section = parts[1].strip() if len(parts) > 1 else ''

            converted_names = set()
            if h5_section:
                for line in h5_section.split('\n'):
                    h5_name = Path(line.strip()).stem
                    if h5_name:
                        converted_names.add(h5_name)

            family_map = {}
            if pts_section:
                for line in pts_section.split('\n'):
                    pts_path = line.strip()
                    if not pts_path or not pts_path.endswith('.pts'):
                        continue
                    p = Path(pts_path)
                    family_map.setdefault(p.parent.name, []).append(p)

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
                    families.append({'family': family_name, 'meshes': meshes})

            h5_only = {}
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
                h5_family_map = {}
                for base_name, flags in sorted(h5_only.items()):
                    family = re.match(r'^([a-zA-Z]+)', base_name)
                    family_name = family.group(1) if family else base_name
                    h5_family_map.setdefault(family_name, []).append({
                        'name': base_name, 'pts': None, 'elem': None,
                        'converted': flags['converted'],
                        'converted_colored': flags['converted_colored'],
                    })
                for family_name in sorted(h5_family_map):
                    existing = next((f for f in families if f['family'] == family_name), None)
                    if existing:
                        existing_names = {m['name'] for m in existing['meshes']}
                        for mesh in h5_family_map[family_name]:
                            if mesh['name'] not in existing_names:
                                existing['meshes'].append(mesh)
                    else:
                        families.append({'family': family_name,
                                         'meshes': h5_family_map[family_name]})
                families.sort(key=lambda f: f['family'])

            return families
        except (subprocess.TimeoutExpired, OSError):
            return []

    def convert_remote_mesh(self, family, pts_file, elem_file, output_prefix, color=False):
        """Start mesh conversion on the cluster inside the DOLFINx container.
        Returns a subprocess.Popen that streams output."""
        if self.mesh_convert_state['converting']:
            raise RuntimeError('A conversion is already in progress')

        pts_path = f'meshes/{family}/{pts_file}'
        elem_path = f'meshes/{family}/{elem_file}'
        out_path = f'data/{output_prefix}'

        self._run_ssh(f'mkdir -p {self.data_path}', timeout=10)

        pylibs = '/home/fenics/.pylibs'
        convert_cmd = (
            f'pip install --target={pylibs} -q lxml h5py 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 geometry/convert_pts_elem.py '
            f'{pts_path} {elem_path} {out_path}'
        )
        if color:
            convert_cmd += ' --color-intracellular'

        remote_cmd = self._env_unset_prefix() + self._apptainer_exec(self.dolfinx_sif, convert_cmd)
        process = self._popen_ssh(remote_cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.mesh_convert_state['converting'] = True
        self.mesh_convert_state['process'] = process
        return process

    def generate_remote_weak_scaling_mesh(self, nx, ny, nz, n, L, pad, output_prefix,
                                          shape='cell', ax=1, slabs=0, d_y=0.5,
                                          d_z=0.5, lean=55.0, lat_r=0.26):
        """Generate a weak-scaling mesh on the cluster inside the DOLFINx container."""
        if self.mesh_convert_state['converting']:
            raise RuntimeError('A conversion is already in progress')

        self._run_ssh(f'mkdir -p {self.data_path}', timeout=10)

        pylibs = '/home/fenics/.pylibs'
        gen_cmd = (
            f'if [ -f data/{output_prefix}.h5 ]; then '
            f'echo "Reusing existing data/{output_prefix}.h5"; '
            f'else '
            f'pip install --target={pylibs} -q lxml h5py scipy 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 geometry/generate_weak_scaling_mesh.py '
            f'--nx {int(nx)} --ny {int(ny)} --nz {int(nz)} '
            f'--n {int(n)} --L {float(L)} --pad {int(pad)} '
            f'--shape {shape} --ax {int(ax)} --slabs {int(slabs)} '
            f'--dist {float(d_y)} --dist-z {float(d_z)} --lean {float(lean)} '
            f'--lat-r {float(lat_r)} --no-preview '
            f'--prefix data/{output_prefix}; '
            f'fi'
        )

        remote_cmd = self._env_unset_prefix() + self._apptainer_exec(self.dolfinx_sif, gen_cmd)
        process = self._popen_ssh(remote_cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.mesh_convert_state['converting'] = True
        self.mesh_convert_state['process'] = process
        return process

    def finish_conversion(self):
        self.mesh_convert_state['converting'] = False
        self.mesh_convert_state['process'] = None

    def download_mesh_data(self, mesh_name, local_data_dir):
        """Download converted mesh files (h5, xdmf, pickle) from the cluster."""
        local_data_dir = Path(local_data_dir)
        local_data_dir.mkdir(parents=True, exist_ok=True)

        for fname in [f'{mesh_name}.h5', f'{mesh_name}.xdmf', f'{mesh_name}.pickle']:
            result = self.scp_download(f'{self.data_path}/{fname}',
                                       local_data_dir / fname, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f'Failed to download {fname}: {result.stderr}')
        return True

    # --------------------- Config upload ---------------------

    def upload_config(self, local_path):
        """SCP a config YAML file to the remote cardioEMI directory."""
        result = self.scp_upload(local_path, f'{self.remote_path}/', timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'SCP upload failed: {result.stderr}')
        return True

    # --------------------- SLURM job management ---------------------

    def generate_slurm_script(self, config_file, nodes=1, ntasks_per_node=None,
                              walltime=None, partition=None, account=None,
                              solver_backend='petsc', include_ranks_in_name=False,
                              active_ranks=None):
        """Generate a SLURM batch script string for cardioEMI.

        active_ranks: exact number of MPI ranks to launch via `srun -n`, when it
            can't be hit as nodes*ntasks_per_node exactly (see the Karolina
            component_granularity notes in the original module).
        """
        ntasks_per_node = ntasks_per_node or self.default_ntasks_per_node
        walltime = walltime or self.default_walltime
        partition = partition or self.default_partition
        account = account or self.default_account

        base = Path(config_file).stem.replace('input_', '') + '_sim'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        total_ranks = nodes * ntasks_per_node
        launch_ranks = total_ranks if active_ranks is None else active_ranks
        if launch_ranks > total_ranks:
            raise ValueError(
                f"active_ranks ({launch_ranks}) exceeds the allocation "
                f"({nodes} nodes x {ntasks_per_node} tasks/node = {total_ranks})")
        cpus_per_task = max(1, self.cores_per_node // ntasks_per_node)

        ranks_suffix = f'_{launch_ranks}r' if include_ranks_in_name else ''
        out_name = f'{base}_{timestamp}{ranks_suffix}'

        sif = self.ginkgo_sif if solver_backend == 'ginkgo' else self.dolfinx_sif

        env_lines = '\n'.join(f'export {k}={v}' for k, v in self.job_env.items())
        unset_line = ('unset ' + ' '.join(self.env_unset)) if self.env_unset else ''
        srun_flags = f' {self.srun_flags}' if self.srun_flags else ''

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

cd {self.remote_path}

{env_lines}
{unset_line}

# The solver runs inside its own run folder: Ginkgo's BDDC writes IF_<rank>.txt
# to the working directory, so a shared one let concurrent jobs overwrite each
# other's interface files. The job YAML uses absolute /home/fenics paths, and
# dolfinx-ginkgo/python is on PYTHONPATH so the Ginkgo import never depends
# on the working directory (an import failure silently falls back to PETSc).
srun -n {launch_ranks}{srun_flags} apptainer exec \\
    --bind {self.remote_path}:/home/fenics \\
    --pwd /home/fenics/{out_name} \\
    {sif} \\
    bash -c 'unset CC CXX && export PYTHONPATH=/home/fenics/.pylibs:/home/fenics/dolfinx-ginkgo/python:$PYTHONPATH && python3 -B -u /home/fenics/main.py {config_file}' 2>&1 | tee {out_name}_slurm.log
# srun's status, not tee's - otherwise a crashed run is reported COMPLETED.
exit ${{PIPESTATUS[0]}}
"""
        return script, out_name

    @staticmethod
    def pack_ranks(ranks, max_tasks_per_node):
        """(nodes, ntasks_per_node) for launching exactly `ranks` ranks with at
        most `max_tasks_per_node` per node. ntasks_per_node is spread evenly, so
        nodes*ntasks_per_node may overshoot `ranks` (e.g. a prime); the job
        still launches exactly `ranks` via `srun -n`."""
        ranks, cap = int(ranks), max(1, int(max_tasks_per_node))
        nodes = max(1, -(-ranks // cap))
        return nodes, -(-ranks // nodes)

    def submit_batch(self, template_config_path, jobs, max_tasks_per_node=None,
                     partition=None, account=None, solver_backend='petsc',
                     conditions=None):
        """Submit one SLURM job per entry of `jobs`, all from one template config,
        using a single scp and a single ssh for the sbatch calls.

        jobs: [{'mesh', 'ranks', 'walltime', 'config_overrides', 'conditions_overrides'}]
            Each job's YAML is the template with mesh_file/tags_dictionary_file/
            out_name pointed at its mesh, cube_partition dropped (so main.py reads
            the lattice from the mesh filename) and config_overrides applied.
        """
        import copy
        import tempfile

        cap = max_tasks_per_node or self.default_ntasks_per_node
        with open(template_config_path, 'r') as f:
            template = yaml.safe_load(f) or {}

        prepared = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for job in jobs:
                mesh, ranks = job['mesh'], int(job['ranks'])
                nodes, ntasks = self.pack_ranks(ranks, cap)
                config_file = f'input_{mesh}.yml'
                script, out_name = self.generate_slurm_script(
                    config_file, nodes, ntasks, job.get('walltime'), partition,
                    account, solver_backend, include_ranks_in_name=True,
                    active_ranks=ranks)

                config = copy.deepcopy(template)
                # Both describe the template's mesh; main.py re-derives them
                # from mesh_file when absent.
                config.pop('cube_partition', None)
                config.pop('original_mesh_file', None)
                config.update(job.get('config_overrides') or {})
                # Absolute container paths: the solver's cwd is the run folder.
                config['mesh_file'] = f'/home/fenics/data/{mesh}.xdmf'
                config['tags_dictionary_file'] = f'/home/fenics/data/{mesh}.pickle'
                config['out_name'] = f'/home/fenics/{out_name}'

                job_local = tmpdir / out_name
                job_local.mkdir()
                with open(job_local / config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                (job_local / 'run_cardioemi.sh').write_text(script)

                conditions_hash = None
                if conditions:
                    job_conditions = {**conditions, **(job.get('conditions_overrides') or {}),
                                      'mesh': mesh, 'nRanks': ranks}
                    job_conditions.pop('_hash', None)
                    conditions_hash = hashlib.sha256(
                        json.dumps(job_conditions, sort_keys=True).encode()).hexdigest()[:12]
                    job_conditions['_hash'] = conditions_hash
                    (job_local / 'conditions.json').write_text(
                        json.dumps(job_conditions, indent=2))

                prepared.append({
                    'out_name': out_name,
                    'info': {
                        'cluster': self.id,
                        'status': 'PENDING',
                        'config_file': config_file,
                        'out_name': out_name,
                        'mesh': mesh,
                        'num_ranks': ranks,
                        'nodes': nodes,
                        'ntasks_per_node': ntasks,
                        'conditions_hash': conditions_hash,
                    },
                })

            result = self.scp_upload([str(tmpdir / p['out_name']) for p in prepared],
                                     f'{self.remote_path}/', timeout=120, recursive=True)
            if result.returncode != 0:
                raise RuntimeError(f'SCP upload failed: {result.stderr}')

        sbatch_cmds = '; '.join(
            f'echo "@@JOB {p["out_name"]}@@"; '
            f'cd {self.remote_path}/{p["out_name"]} && sbatch run_cardioemi.sh'
            for p in prepared)
        stdout, stderr, rc = self._run_ssh(sbatch_cmds, timeout=60)

        by_name = {p['out_name']: p['info'] for p in prepared}
        submitted, current = [], None
        for line in stdout.split('\n'):
            line = line.strip()
            if line.startswith('@@JOB ') and line.endswith('@@'):
                current = line[6:-2]
                continue
            match = re.search(r'Submitted batch job (\d+)', line)
            if match and current in by_name:
                job_info = {'job_id': match.group(1), **by_name[current]}
                self.jobs[job_info['job_id']] = job_info
                with self._poll_lock:
                    self._poll_cache[job_info['job_id']] = {'status': 'PENDING', 'log': ''}
                self.legacy_state.update(job_id=job_info['job_id'], status='PENDING',
                                         config_file=job_info['config_file'],
                                         out_name=current, num_ranks=job_info['num_ranks'])
                submitted.append(job_info)
                current = None

        if submitted:
            self.start_background_poller()
        failed = [n for n in by_name if n not in {j['out_name'] for j in submitted}]
        if failed and not submitted:
            raise RuntimeError(f'sbatch failed: {stderr or stdout}')
        return submitted, failed, (stderr if failed else '')

    def check_job_status(self, job_id):
        """Check SLURM job status via squeue/sacct. Returns status string."""
        job = self.jobs.get(job_id, {})
        try:
            stdout, stderr, rc = self._run_ssh(
                f'squeue -j {job_id} --noheader -o "%T"', timeout=15)
            if rc == 0 and stdout:
                status = stdout.strip().split('\n')[0].strip()
                if status:
                    if job:
                        job['status'] = status
                    self.legacy_state['status'] = status
                    return status
        except subprocess.TimeoutExpired:
            pass

        try:
            stdout, stderr, rc = self._run_ssh(
                f'sacct -j {job_id} --noheader -o State -P', timeout=15)
            if rc == 0 and stdout:
                status = stdout.strip().split('\n')[0].strip()
                if status:
                    if job:
                        job['status'] = status
                    self.legacy_state['status'] = status
                    return status
        except subprocess.TimeoutExpired:
            pass

        return job.get('status', self.legacy_state.get('status', 'UNKNOWN'))

    def cancel_job(self, job_id):
        """Cancel a SLURM job via scancel."""
        stdout, stderr, rc = self._run_ssh(f'scancel {job_id}', timeout=15)
        if rc != 0:
            raise RuntimeError(f'scancel failed: {stderr}')
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'CANCELLED'
        with self._poll_lock:
            if job_id in self._poll_cache:
                self._poll_cache[job_id]['status'] = 'CANCELLED'
        self.legacy_state['status'] = 'CANCELLED'
        return True

    def tail_remote_log(self, job_id, num_lines=50, out_name=None):
        """Tail a job's output, wherever it is: the <run>_slurm.log the job
        script tees next to the run folder, the job's own slurm_<id>.out inside
        it, or (jobs from before per-job folders) slurm_<id>.out at the root."""
        out_name = out_name or self.jobs.get(job_id, {}).get('out_name')
        if out_name and not re.fullmatch(r'[A-Za-z0-9_.\-]+', out_name):
            out_name = None
        jid = re.sub(r'[^0-9_]', '', str(job_id))
        root = _shell_quote(self.remote_path)
        candidates = []
        if out_name:
            candidates += [f'{root}/{out_name}_slurm.log', f'{root}/{out_name}/slurm_{jid}.out']
        candidates.append(f'{root}/slurm_{jid}.out')
        cmd = ' || '.join(f'tail -n {int(num_lines)} {c} 2>/dev/null' for c in candidates)
        try:
            stdout, _, rc = self._run_ssh(cmd, timeout=15)
            return stdout if rc == 0 else ''
        except subprocess.TimeoutExpired:
            return ''

    # --------------------- Results download ---------------------

    def download_results_streaming(self, remote_out_name, local_dest):
        """Download simulation results as a streamed tar.gz with progress."""
        local_dest = Path(local_dest)
        local_dest.mkdir(parents=True, exist_ok=True)

        remote_dir = f'{self.remote_path}/{remote_out_name}'

        size_cmd = f'du -sb {remote_dir} 2>/dev/null | cut -f1'
        stdout, stderr, rc = self._run_ssh(size_cmd, timeout=30)
        if rc != 0 or not stdout.strip():
            yield {'type': 'error', 'message': f'Failed to get remote directory size: {stderr}'}
            return

        bytes_total_uncompressed = int(stdout.strip())
        bytes_total_estimate = int(bytes_total_uncompressed * 0.5)

        yield {'type': 'progress', 'bytes_done': 0, 'bytes_total': bytes_total_estimate,
               'file': 'Creating archive...'}

        tar_cmd = f'cd {self.remote_path} && tar czf - {remote_out_name}'
        archive_path = local_dest.parent / f'.{remote_out_name}.tar.gz'

        try:
            process = self._popen_ssh(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
                           'file': f'Downloading archive ({bytes_done // (1024*1024)} MB)...'}

            process.wait()
            if process.returncode != 0:
                stderr_out = process.stderr.read().decode()
                yield {'type': 'error', 'message': f'SSH tar failed: {stderr_out}'}
                return

            compressed_size = bytes_done
            yield {'type': 'progress', 'bytes_done': compressed_size,
                   'bytes_total': compressed_size, 'file': 'Extracting archive...'}

            result = subprocess.run(
                ['tar', 'xzf', str(archive_path), '-C', str(local_dest.parent)],
                capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                yield {'type': 'error', 'message': f'Extraction failed: {result.stderr}'}
                return

            ratio = (1 - compressed_size / bytes_total_uncompressed) * 100 \
                if bytes_total_uncompressed > 0 else 0
            yield {'type': 'complete',
                   'message': f'Downloaded {compressed_size // (1024*1024)} MB '
                              f'(compressed {ratio:.0f}% from '
                              f'{bytes_total_uncompressed // (1024*1024)} MB)'}
        finally:
            if archive_path.exists():
                archive_path.unlink()

    def download_iterations(self, remote_out_name, local_dest):
        """Download just iterations/residuals/conditions from a remote simulation."""
        local_dest = Path(local_dest)
        local_dest.mkdir(parents=True, exist_ok=True)

        remote_dir = f'{self.remote_path}/{remote_out_name}'
        for fname in ['iterations.pickle', 'residuals.pickle', 'conditions.json']:
            result = self.scp_download(f'{remote_dir}/{fname}', local_dest / fname, timeout=30)
            if result.returncode != 0:
                continue  # optional files
        return True

    def list_remote_runs(self):
        """Every *_sim* run folder on the cluster, for the Runs browser.

        One ssh call running a small script on the login node: per folder, whether
        it holds iterations / full results, its size, the SLURM job ids of its logs,
        and - from squeue's working directory column - whether a job for it is still
        queued or running. A pending job has no slurm_<id>.out yet, so squeue is the
        only thing that keeps its empty folder from looking failed; `active_ok`
        says whether that check worked. Raises if the cluster can't be listed.
        """
        py = (
            'import json, os, glob, subprocess\n'
            f'root = {self.remote_path!r}\n'
            'active, active_ok = {}, False\n'
            'try:\n'
            '    p = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),\n'
            '                        "-o", "%i|%T|%Z"], stdout=subprocess.PIPE,\n'
            '                       stderr=subprocess.PIPE, universal_newlines=True, timeout=30)\n'
            '    active_ok = p.returncode == 0\n'
            '    for line in p.stdout.splitlines():\n'
            '        parts = line.split("|", 2)\n'
            '        if len(parts) == 3:\n'
            '            active[os.path.basename(parts[2].rstrip("/"))] = [parts[0], parts[1]]\n'
            'except Exception:\n'
            '    pass\n'
            'runs = []\n'
            'for d in sorted(glob.glob(root + "/*_sim*")):\n'
            '    if not os.path.isdir(d):\n'
            '        continue\n'
            '    name = os.path.basename(d)\n'
            '    files, size = set(), 0\n'
            '    for e in os.scandir(d):\n'
            '        files.add(e.name)\n'
            '        try:\n'
            '            if e.is_file():\n'
            '                size += e.stat().st_size\n'
            '        except OSError:\n'
            '            pass\n'
            '    cond = {}\n'
            '    if "conditions.json" in files:\n'
            '        try:\n'
            '            with open(os.path.join(d, "conditions.json")) as f:\n'
            '                c = json.load(f)\n'
            '            cond = {k: c.get(k) for k in ("mesh", "solver", "preconditioner",\n'
            '                    "localSolver", "nRanks") if c.get(k) is not None}\n'
            '        except Exception:\n'
            '            pass\n'
            '    runs.append({"name": name, "size": size, "mtime": os.stat(d).st_mtime,\n'
            '                 "iterations": "iterations.pickle" in files,\n'
            '                 "results": "v.h5" in files or "solution.h5" in files,\n'
            '                 "jobs": sorted(f[6:-4] for f in files\n'
            '                                if f.startswith("slurm_") and f.endswith(".out")),\n'
            '                 "active": active.get(name), "conditions": cond})\n'
            'print("@@RUNS " + json.dumps({"runs": runs, "active_ok": active_ok}))\n'
        )
        stdout, stderr, rc = self._run_ssh(f'python3 -c {_shell_quote(py)}', timeout=60)
        for line in stdout.split('\n'):
            if line.startswith('@@RUNS '):
                return json.loads(line[len('@@RUNS '):])
        raise RuntimeError((stderr or stdout or f'ssh exit {rc}').strip()[-300:])

    def delete_runs(self, names, only_if_empty=False):
        """Remove run folders (and their top-level <name>_slurm.log) on the
        cluster. With only_if_empty, folders holding iterations or results are
        left alone - the guard runs on the cluster, so it holds even if the
        caller's view is stale. Returns (removed names, error str or None)."""
        names = [n for n in names if re.fullmatch(r'[A-Za-z0-9_.\-]+', n) and '_sim' in n]
        if not names:
            return [], None
        keep = ('[ -e "$n/iterations.pickle" ] || [ -e "$n/v.h5" ] || '
                '[ -e "$n/solution.h5" ] || ') if only_if_empty else ''
        cmd = (f'cd {_shell_quote(self.remote_path)} && for n in {" ".join(names)}; do '
               f'[ -d "$n" ] || continue; {keep}'
               f'{{ rm -rf -- "$n" "${{n}}_slurm.log" && echo "@@DEL $n"; }}; done')
        stdout, stderr, rc = self._run_ssh(cmd, timeout=120)
        removed = [l[len('@@DEL '):].strip() for l in stdout.split('\n') if l.startswith('@@DEL ')]
        error = None if rc == 0 else (stderr.strip()[-300:] or f'ssh exit {rc}')
        return removed, error

    # --------------------- Remote video generation ---------------------

    def generate_remote_video(self, sim_name, width=1920, height=1080, fps=30,
                              camera_config=None, colormap='coolwarm',
                              partition=None, account=None):
        """Submit a SLURM job to generate a video on the cluster."""
        partition = partition or self.default_partition
        account = account or self.default_account
        pylibs = '/home/fenics/.pylibs'
        video_dir = f'{self.remote_path}/viz/videos'

        self._run_ssh(f'mkdir -p {video_dir}', timeout=10)

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

        container_cmd = self._apptainer_exec(self.dolfinx_sif, pipeline_cmd)
        unset_line = ('unset ' + ' '.join(self.env_unset)) if self.env_unset else ''

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_content = f"""#!/bin/bash
#SBATCH --job-name=video_{sim_name[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output={self.remote_path}/video_{timestamp}_%j.log
#SBATCH --error={self.remote_path}/video_{timestamp}_%j.log

cd {self.remote_path}
{unset_line}
{container_cmd}
"""

        script_name = f'video_{timestamp}.sh'
        local_script = Path('/tmp') / script_name
        local_script.write_text(script_content)
        result = self.scp_upload(local_script, f'{self.remote_path}/{script_name}', timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'Failed to upload video script: {result.stderr}')

        stdout, stderr, rc = self._run_ssh(
            f'cd {self.remote_path} && sbatch {script_name}', timeout=30)
        if rc != 0:
            raise RuntimeError(f'sbatch failed: {stderr}')

        match = re.search(r'Submitted batch job (\d+)', stdout)
        if not match:
            raise RuntimeError(f'Could not parse job ID from: {stdout}')

        job_id = match.group(1)
        return {
            'job_id': job_id,
            'cluster': self.id,
            'log_file': f'video_{timestamp}_{job_id}.log',
            'script': script_name,
            'sim_name': sim_name,
        }

    def check_video_job(self, job_id, log_file=None):
        """Check video generation job status and parse progress from log."""
        status = self.check_job_status(job_id)

        progress = 0
        message = ''
        video_filename = None

        if log_file:
            try:
                stdout, stderr, rc = self._run_ssh(
                    f'tail -20 {self.remote_path}/{log_file} 2>/dev/null', timeout=15)
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

        return {'status': status, 'progress': progress, 'message': message,
                'video_filename': video_filename}

    def download_video(self, video_filename, local_dest):
        """Download a generated video from the cluster."""
        local_dest = Path(local_dest)
        local_dest.mkdir(parents=True, exist_ok=True)
        local_file = local_dest / video_filename

        result = self.scp_download(f'{self.remote_path}/viz/videos/{video_filename}',
                                   local_file, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f'Failed to download video: {result.stderr}')
        return str(local_file)

    # --------------------- Remote viz data generation ---------------------

    def generate_remote_viz(self, sim_name, partition=None, account=None, membrane_only=True):
        """Submit a SLURM job to generate viz data on the cluster."""
        partition = partition or (self.partitions[1] if len(self.partitions) > 1
                                  else self.default_partition)
        account = account or self.default_account
        viz_dir = f'viz/data/{sim_name}'
        pylibs = '/home/fenics/.pylibs'

        membrane_flag = ' --membrane-only' if membrane_only else ''

        gen_cmd = (
            f'pip install --target={pylibs} -q h5py 2>/dev/null; '
            f'export PYTHONPATH={pylibs}:$PYTHONPATH && '
            f'python3 -u viz/scripts/generate_viz_from_output.py'
            f'{membrane_flag} '
            f'{sim_name} {viz_dir}'
        )

        container_cmd = self._apptainer_exec(self.dolfinx_sif, gen_cmd)
        unset_line = ('unset ' + ' '.join(self.env_unset)) if self.env_unset else ''

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_content = f"""#!/bin/bash
#SBATCH --job-name=viz_{sim_name[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={self.cores_per_node}
#SBATCH --time=02:00:00
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output={self.remote_path}/vizgen_{timestamp}_%j.log
#SBATCH --error={self.remote_path}/vizgen_{timestamp}_%j.log

cd {self.remote_path}
{unset_line}
{container_cmd}
"""

        script_name = f'vizgen_{timestamp}.sh'
        local_script = Path('/tmp') / script_name
        local_script.write_text(script_content)
        result = self.scp_upload(local_script, f'{self.remote_path}/{script_name}', timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'Failed to upload viz script: {result.stderr}')

        stdout, stderr, rc = self._run_ssh(
            f'cd {self.remote_path} && sbatch {script_name}', timeout=30)
        if rc != 0:
            raise RuntimeError(f'sbatch failed: {stderr}')

        match = re.search(r'Submitted batch job (\d+)', stdout)
        if not match:
            raise RuntimeError(f'Could not parse job ID from: {stdout}')

        job_id = match.group(1)
        return {
            'job_id': job_id,
            'cluster': self.id,
            'log_file': f'vizgen_{timestamp}_{job_id}.log',
            'sim_name': sim_name,
        }

    def check_viz_job(self, job_id, log_file=None):
        """Check viz generation job status and parse progress from log."""
        status = self.check_job_status(job_id)

        progress = 0
        message = ''

        if log_file:
            try:
                stdout, stderr, rc = self._run_ssh(
                    f'tail -20 {self.remote_path}/{log_file} 2>/dev/null', timeout=15)
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

        return {'status': status, 'progress': progress, 'message': message}

    def download_viz_data_streaming(self, sim_name, local_dest):
        """Download viz data directory via streaming tar (progress dicts)."""
        local_dest = Path(local_dest)
        local_dest.mkdir(parents=True, exist_ok=True)

        remote_viz_dir = f'viz/data/{sim_name}'

        size_cmd = f'du -sb {self.remote_path}/{remote_viz_dir} 2>/dev/null | cut -f1'
        stdout, stderr, rc = self._run_ssh(size_cmd, timeout=30)
        if rc != 0 or not stdout.strip():
            yield {'type': 'error',
                   'message': f'Viz data not found on {self.label} for {sim_name}'}
            return

        bytes_total_uncompressed = int(stdout.strip())
        bytes_total_estimate = int(bytes_total_uncompressed * 0.5)

        yield {'type': 'progress', 'bytes_done': 0, 'bytes_total': bytes_total_estimate,
               'file': 'Creating archive...'}

        tar_cmd = f'cd {self.remote_path} && tar czf - {remote_viz_dir}'
        archive_path = local_dest / f'.{sim_name}_viz.tar.gz'

        try:
            process = self._popen_ssh(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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

            # Archive contains viz/data/<sim_name>/ — extract to project root
            result = subprocess.run(
                ['tar', 'xzf', str(archive_path), '-C', str(local_dest.parent.parent.parent)],
                capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                yield {'type': 'error', 'message': f'Extraction failed: {result.stderr}'}
                return

            yield {'type': 'complete',
                   'message': f'Downloaded viz data ({compressed_size // (1024*1024)} MB)'}
        finally:
            if archive_path.exists():
                archive_path.unlink()


# --------------------- Registry ---------------------

class ClusterRegistry:
    """Loads viz/clusters.yml and hands out (cached) Cluster instances."""

    def __init__(self, config_path=None):
        self.config_path = Path(config_path or Path(__file__).parent / 'clusters.yml')
        self._instances = {}
        self._lock = threading.Lock()

    def _load_raw(self):
        if not self.config_path.exists():
            return {'clusters': {}}
        with open(self.config_path) as f:
            data = yaml.safe_load(f) or {}
        data.setdefault('clusters', {})
        return data

    def ids(self):
        return list(self._load_raw()['clusters'].keys())

    def clusters(self):
        return [self.get(cid) for cid in self.ids()]

    def get(self, cluster_id):
        raw = self._load_raw()['clusters']
        if cluster_id not in raw:
            raise KeyError(f'Unknown cluster: {cluster_id}')
        with self._lock:
            inst = self._instances.get(cluster_id)
            cfg = raw[cluster_id]
            if inst is None or inst.cfg != cfg:
                # Config changed (or first use): build a fresh instance but
                # carry over live job state so edits don't orphan running jobs.
                new = Cluster(cluster_id, cfg)
                if inst is not None:
                    new.jobs = inst.jobs
                    new.legacy_state = inst.legacy_state
                    new.mesh_convert_state = inst.mesh_convert_state
                    new._poll_cache = inst._poll_cache
                    new._poll_lock = inst._poll_lock
                    new.connect_state = inst.connect_state
                    # The login in progress too, so its OTP answer / cancel
                    # still reaches the worker that owns it.
                    for attr in ('_connect_lock', '_connect_answer_event', '_connect_token',
                                 '_connect_child', '_connect_since', '_connect_answer'):
                        if hasattr(inst, attr):
                            setattr(new, attr, getattr(inst, attr))
                    inst._poll_stop.set()  # old poller dies; new starts on demand
                    if new.jobs:
                        new.start_background_poller()
                self._instances[cluster_id] = new
            return self._instances[cluster_id]

    def save(self, cluster_id, cfg):
        """Create or update a cluster entry and persist to clusters.yml.

        Keys the settings form doesn't cover (env_unset, job_env, srun_flags,
        and anything else hand-edited into the YAML) are carried over from the
        existing entry when the incoming cfg omits them, so a webapp save
        never silently strips them.
        """
        if not re.fullmatch(r'[a-z0-9_\-]+', cluster_id):
            raise ValueError('cluster id must be lowercase alphanumeric/-/_')
        for key in ('host', 'remote_path'):
            if not cfg.get(key):
                raise ValueError(f'missing required field: {key}')
        data = self._load_raw()
        existing = data['clusters'].get(cluster_id, {})
        merged = dict(existing)
        merged.update(cfg)
        data['clusters'][cluster_id] = merged
        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return self.get(cluster_id)

    def delete(self, cluster_id):
        data = self._load_raw()
        if cluster_id in data['clusters']:
            del data['clusters'][cluster_id]
            with open(self.config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        with self._lock:
            inst = self._instances.pop(cluster_id, None)
            if inst:
                inst._poll_stop.set()


registry = ClusterRegistry()
