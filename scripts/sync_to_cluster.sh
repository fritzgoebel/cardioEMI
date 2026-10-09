#!/bin/bash
# sync_to_cluster.sh - Sync cardioEMI source code to any cluster from viz/clusters.yml
#
# Usage:
#   bash scripts/sync_to_cluster.sh <cluster-id>     # e.g. karolina, vega
#
# Reads host/user/identity/remote_path from viz/clusters.yml and rsyncs over
# the same persistent ControlMaster connection the webapp uses, so an already
# connected OTP cluster (Vega) needs no new authentication.

set -euo pipefail

CLUSTER_ID="${1:?usage: sync_to_cluster.sh <cluster-id>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Pull connection details out of viz/clusters.yml
eval "$(python3 - "$CLUSTER_ID" "$PROJECT_ROOT/viz/clusters.yml" << 'EOF'
import sys, yaml
cid, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f)['clusters']
if cid not in cfg:
    print(f'echo "Unknown cluster: {cid}"; exit 1')
    sys.exit(0)
c = cfg[cid]
dest = f"{c['user']}@{c['host']}" if c.get('user') else c['host']
ident = c.get('identity_file') or ''
print(f'DEST={dest}')
print(f'REMOTE_PATH={c["remote_path"]}')
print(f'IDENTITY={ident}')
EOF
)"

SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cardioemi-cm/%C -o ControlPersist=yes -o StrictHostKeyChecking=accept-new"
if [ -n "${IDENTITY}" ]; then
    SSH_OPTS="$SSH_OPTS -i ${IDENTITY/#\~/$HOME} -o IdentitiesOnly=yes"
fi
mkdir -p -m 700 "$HOME/.ssh/cardioemi-cm"

echo "Syncing ${PROJECT_ROOT} -> ${DEST}:${REMOTE_PATH}"

rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    --exclude='*_sim*/' \
    --exclude='data/' \
    --exclude='viz/data/' \
    --exclude='viz/videos/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='*.h5' \
    --exclude='*.xdmf' \
    --exclude='*.pickle' \
    --exclude='IF_*.txt' \
    --exclude='SESSION_SUMMARY.md' \
    --exclude='test_*.py' \
    --exclude='containers/' \
    --exclude='*.sif' \
    --exclude='.DS_Store' \
    "${PROJECT_ROOT}/" "${DEST}:${REMOTE_PATH}/"

echo ""
echo "Sync complete."
