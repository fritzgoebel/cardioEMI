"""Run index for the webapp's Runs browser.

One record per simulation run, merged from three places that each know only
part of the story: local `*_sim*` output folders under the project root, the
`*_sim*` folders on every registered cluster, and the regenerable viz cache in
viz/data/. On top of that sits the user's organisation of those runs, kept in
viz/run_labels.json:

    {'runs':       {'<sim dir>': {'label': str, 'folder': 'a/b/c'}},
     'folders':    ['a', 'a/b', 'a/b/c', ...],
     'categories': {'<kind>:<key>': str}}      e.g. 'mesh:plus_4x4x4_n16_L8'

Folders are virtual: a run's folder is a '/'-separated path stored next to its
label, so nothing on disk or on a cluster moves when runs are filed, renamed or
moved, and every lookup by run name (viz cache, downloads, remote listing)
keeps working. `folders` lists every folder explicitly so empty ones persist.

Remote listings need ssh (and an OTP connection for Vega), so each cluster's
last successful listing is cached in viz/run_index_cache.json; a cluster that
can't be reached right now is shown from that cache, flagged stale.
"""

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIZ_DIR = Path(__file__).parent
RUN_LABELS_FILE = VIZ_DIR / 'run_labels.json'
CACHE_FILE = VIZ_DIR / 'run_index_cache.json'
VIZ_DATA_DIR = VIZ_DIR / 'data'

RUN_NAME_RE = re.compile(r'[A-Za-z0-9_.\-]+')
TIMESTAMP_RE = re.compile(r'_sim_(\d{8})_(\d{6})')
ACTIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'REQUEUED',
                 'RESIZING', 'SUSPENDED'}

_cache_lock = threading.Lock()


def is_run_name(name):
    return bool(name) and '_sim' in name and bool(RUN_NAME_RE.fullmatch(name))


def normalize_folder(path):
    """'/a//b/ ' -> 'a/b'; '' means unfiled. Segments may not be '.' or '..'."""
    parts = [p.strip() for p in (path or '').split('/')]
    parts = [p for p in parts if p]
    if any(p in ('.', '..') for p in parts):
        raise ValueError(f'invalid folder path: {path!r}')
    return '/'.join(parts)


def _ancestors(path):
    """'a/b/c' -> ['a', 'a/b', 'a/b/c']."""
    parts = path.split('/') if path else []
    return ['/'.join(parts[:i + 1]) for i in range(len(parts))]


def _under(path, root):
    return path == root or path.startswith(root + '/')


# --------------------- labels + folders store ---------------------

def load_labels():
    """The user's names and folders. Migrates the earlier one-level
    `collection` field into a top-level folder on first load."""
    data = {}
    if RUN_LABELS_FILE.exists():
        try:
            with open(RUN_LABELS_FILE, 'r') as f:
                data = json.load(f) or {}
        except (OSError, ValueError) as exc:
            print(f"Warning: could not read {RUN_LABELS_FILE.name}: {exc}")
    labels = {
        'runs': data.get('runs') or {},
        'folders': list(data.get('folders') or []),
        'categories': data.get('categories') or {},
    }

    migrated = False
    for entry in labels['runs'].values():
        if 'collection' in entry:
            collection = entry.pop('collection')
            if collection and not entry.get('folder'):
                entry['folder'] = normalize_folder(collection.replace('/', ' - '))
            migrated = True
    folders = set(labels['folders'])
    for entry in labels['runs'].values():
        folders.update(_ancestors(entry.get('folder') or ''))
    if migrated or folders != set(labels['folders']):
        labels['folders'] = sorted(folders)
        if migrated:
            save_labels(labels)
    else:
        labels['folders'] = sorted(folders)
    return labels


def save_labels(labels):
    """Write atomically - the file is the only record of these names."""
    folders = set(labels.get('folders') or [])
    for entry in labels['runs'].values():
        folders.update(_ancestors(entry.get('folder') or ''))
    labels['folders'] = sorted(folders)
    tmp = RUN_LABELS_FILE.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    tmp.replace(RUN_LABELS_FILE)


def forget_runs(names):
    """Drop names of deleted runs so the file does not collect orphans."""
    labels = load_labels()
    if any(n in labels['runs'] for n in names):
        for n in names:
            labels['runs'].pop(n, None)
        save_labels(labels)


def file_runs(names, folder):
    """Put runs into `folder` ('' = unfiled), creating it and its parents."""
    folder = normalize_folder(folder)
    labels = load_labels()
    for name in names:
        if not is_run_name(name):
            continue
        entry = dict(labels['runs'].get(name) or {})
        if folder:
            entry['folder'] = folder
        else:
            entry.pop('folder', None)
        if entry:
            labels['runs'][name] = entry
        else:
            labels['runs'].pop(name, None)
    if folder:
        labels['folders'] = sorted(set(labels['folders']) | set(_ancestors(folder)))
    save_labels(labels)
    return labels


def create_folder(path):
    path = normalize_folder(path)
    if not path:
        raise ValueError('folder name is empty')
    labels = load_labels()
    labels['folders'] = sorted(set(labels['folders']) | set(_ancestors(path)))
    save_labels(labels)
    return labels


def rename_folder(path, new_path):
    """Move a folder (and everything under it) to new_path - covers both a
    rename and dragging a folder into another one."""
    path, new_path = normalize_folder(path), normalize_folder(new_path)
    if not path or not new_path:
        raise ValueError('folder path is empty')
    if _under(new_path, path) and new_path != path:
        raise ValueError('cannot move a folder into itself')
    labels = load_labels()

    def repath(p):
        return new_path + p[len(path):] if _under(p, path) else p

    labels['folders'] = sorted({repath(p) for p in labels['folders']}
                               | set(_ancestors(new_path)))
    for entry in labels['runs'].values():
        if entry.get('folder') and _under(entry['folder'], path):
            entry['folder'] = repath(entry['folder'])
    save_labels(labels)
    return labels


def drop_folders(paths):
    """Remove folders (and their subfolders) from the store; the caller
    deletes the runs in them first."""
    paths = [normalize_folder(p) for p in paths if normalize_folder(p)]
    if not paths:
        return
    labels = load_labels()
    labels['folders'] = [f for f in labels['folders']
                         if not any(_under(f, p) for p in paths)]
    for entry in labels['runs'].values():
        if entry.get('folder') and any(_under(entry['folder'], p) for p in paths):
            entry.pop('folder')
    labels['runs'] = {k: v for k, v in labels['runs'].items() if v}
    save_labels(labels)


def prune_emptied_folders(paths, existing):
    """Drop folders a delete left without runs: each of `paths` (the folders
    the deleted runs were in) and its ancestors, if no run in `existing` (the
    names still present anywhere) is filed in or below it. Folders the delete
    didn't touch stay, so a freshly created empty folder survives."""
    labels = load_labels()
    filed = [(e.get('folder') or '') for n, e in labels['runs'].items() if n in existing]
    candidates = set()
    for p in paths:
        candidates.update(_ancestors(normalize_folder(p)))
    empty = {c for c in candidates if not any(_under(f, c) for f in filed if f)}
    if not empty:
        return []
    labels['folders'] = [f for f in labels['folders'] if not any(_under(f, e) for e in empty)]
    for entry in labels['runs'].values():
        if entry.get('folder') and any(_under(entry['folder'], e) for e in empty):
            entry.pop('folder')  # stale labels of runs that no longer exist
    labels['runs'] = {k: v for k, v in labels['runs'].items() if v}
    save_labels(labels)
    return sorted(empty)


def runs_in_folders(paths, names):
    """Names among `names` filed in any of `paths` or below."""
    paths = [normalize_folder(p) for p in paths if normalize_folder(p)]
    runs = load_labels()['runs']
    return [n for n in names
            if any(_under((runs.get(n) or {}).get('folder') or '', p) for p in paths)]


# --------------------- local scan ---------------------

def _flat_size(path):
    """Bytes in a run folder (run folders are flat)."""
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def scan_local(project_root):
    """{name: {iterations, results, viz, size, mtime, conditions}} for every
    local *_sim* folder."""
    out = {}
    for item in Path(project_root).iterdir():
        if not item.is_dir() or not is_run_name(item.name):
            continue
        cond = {}
        cond_file = item / 'conditions.json'
        if cond_file.exists():
            try:
                with open(cond_file) as f:
                    cond = json.load(f) or {}
            except (OSError, ValueError):
                cond = {}
        try:
            mtime = item.stat().st_mtime
        except OSError:
            mtime = 0
        out[item.name] = {
            'iterations': (item / 'iterations.pickle').exists(),
            'results': (item / 'v.h5').exists(),
            'viz': (VIZ_DATA_DIR / item.name / 'mesh_metadata.json').exists(),
            'size': _flat_size(item) + _flat_size(VIZ_DATA_DIR / item.name),
            'mtime': mtime,
            'conditions': cond,
        }
    return out


def delete_local(project_root, names, only_if_empty=False):
    """Remove local run folders and their viz caches. Returns (removed, errors)."""
    root = Path(project_root).resolve()
    removed, errors = [], []
    for name in names:
        if not is_run_name(name):
            errors.append(f'{name}: invalid run name')
            continue
        item = (root / name).resolve()
        if item.parent != root:
            errors.append(f'{name}: outside the project')
            continue
        if only_if_empty and item.is_dir() and (
                (item / 'iterations.pickle').exists() or (item / 'v.h5').exists()):
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            cache = VIZ_DATA_DIR / name
            if cache.is_dir():
                shutil.rmtree(cache)
            removed.append(name)
        except OSError as exc:
            errors.append(f'{name}: {exc}')
    return removed, errors


# --------------------- remote listings (cached) ---------------------

def _load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f) or {}
        except (OSError, ValueError):
            pass
    return {}


def _save_cache(cache):
    tmp = CACHE_FILE.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(cache, f)
    tmp.replace(CACHE_FILE)


def refresh_remote(registry):
    """List every cluster in parallel; update the cache for those that answer.
    Returns {cluster_id: error str} for the ones that didn't. Each listing is
    bounded by its own ssh timeout, so a dead cluster only goes stale."""
    clusters = []
    for cid in registry.ids():
        try:
            clusters.append(registry.get(cid))
        except Exception:
            pass

    def list_one(cl):
        try:
            return cl.id, cl.list_remote_runs(), None
        except Exception as exc:
            return cl.id, None, str(exc)

    errors = {}
    with ThreadPoolExecutor(max_workers=max(1, len(clusters))) as pool:
        results = list(pool.map(list_one, clusters))
    with _cache_lock:
        cache = _load_cache()
        for cid, listing, err in results:
            if listing is None:
                # Keep the last good listing; remember why it couldn't be renewed.
                errors[cid] = err or 'unreachable'
                cache.setdefault(cid, {'runs': [], 'updated': None})['error'] = errors[cid]
            else:
                cache[cid] = {'updated': time.time(), **listing}
        _save_cache(cache)
    return errors


def forget_remote(cluster_id, names):
    """Drop deleted runs from a cluster's cached listing."""
    names = set(names)
    with _cache_lock:
        cache = _load_cache()
        entry = cache.get(cluster_id)
        if entry:
            entry['runs'] = [r for r in entry.get('runs', []) if r['name'] not in names]
            _save_cache(cache)


def cached_remote():
    with _cache_lock:
        return _load_cache()


# --------------------- merged index ---------------------

def _timestamp(name):
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    d, t = m.groups()
    return f'{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}'


def build_index(project_root, registry, refresh_errors=None, parse_ws_name=None):
    """Merged run records + folders + per-cluster listing status. A cluster is
    stale when it has never been listed or its last refresh failed (its runs
    then come from the last good listing)."""
    refresh_errors = refresh_errors or {}
    labels = load_labels()
    local = scan_local(project_root)
    remote = cached_remote()
    cluster_labels = {}
    for cid in registry.ids():
        try:
            cluster_labels[cid] = registry.get(cid).label
        except Exception:
            cluster_labels[cid] = cid

    runs = {}

    def record(name):
        if name not in runs:
            runs[name] = {'name': name, 'local': None, 'remote': {}, 'conditions': {}}
        return runs[name]

    for name, info in local.items():
        rec = record(name)
        rec['conditions'] = info.pop('conditions')
        rec['local'] = info

    clusters = {}
    for cid, label in cluster_labels.items():
        entry = remote.get(cid)
        error = refresh_errors.get(cid) or (entry or {}).get('error')
        clusters[cid] = {
            'label': label,
            'updated': entry.get('updated') if entry else None,
            'stale': entry is None or bool(error),
            'error': error,
            'active_ok': bool(entry and entry.get('active_ok') and not error),
        }
        if not entry:
            continue
        for r in entry.get('runs', []):
            rec = record(r['name'])
            rec['remote'][cid] = {k: r.get(k) for k in
                                  ('iterations', 'results', 'size', 'mtime', 'active', 'jobs')}
            if not rec['conditions'] and r.get('conditions'):
                rec['conditions'] = r['conditions']

    out = []
    for name, rec in runs.items():
        cond = rec.pop('conditions') or {}
        mesh = cond.get('mesh')
        if not mesh:
            m = re.match(r'^(.+?)_sim', name)
            mesh = m.group(1) if m else 'other'
        named = labels['runs'].get(name) or {}
        rec.update({
            'mesh': mesh,
            'solver': cond.get('solver'),
            'preconditioner': cond.get('preconditioner'),
            'localSolver': cond.get('localSolver'),
            'nRanks': cond.get('nRanks'),
            'timestamp': _timestamp(name),
            'label': named.get('label'),
            'folder': named.get('folder') or '',
        })
        if parse_ws_name:
            ws = parse_ws_name(mesh)
            if ws:
                rec['nSubdomains'] = 2 * ws['nx'] * ws['ny'] * ws['nz']
                rec['hRatio'] = ws['n']
        loc = rec['local'] or {}
        rec['hasData'] = bool(loc.get('iterations') or loc.get('results') or any(
            r.get('iterations') or r.get('results') for r in rec['remote'].values()))
        rec['active'] = any(r.get('active') for r in rec['remote'].values())
        out.append(rec)

    out.sort(key=lambda r: r['name'])
    return {'runs': out, 'folders': labels['folders'],
            'labels': labels['runs'], 'categories': labels['categories'],
            'clusters': clusters}
