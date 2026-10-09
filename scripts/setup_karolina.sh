#!/bin/bash
# setup_karolina.sh - One-time environment setup for Karolina supercomputer
#
# This script sets up the Python environment and builds DOLFINx + Ginkgo
# on the Karolina supercomputer. Run once after first login.
#
# Usage:
#   ssh karolina
#   cd /scratch/project/eu-26-11/fritz/cardioEMI
#   bash scripts/setup_karolina.sh
#
# Prerequisites:
#   - SSH access to Karolina configured in ~/.ssh/config as host 'karolina'
#   - Project directory synced via scripts/sync_to_karolina.sh

set -euo pipefail

echo "=== Karolina Environment Setup ==="
echo "Working directory: $(pwd)"
echo ""

# Placeholder: Load required modules
# These need to be verified by checking 'module avail' on Karolina
echo "Loading modules..."
# module load Python/3.11
# module load CMake
# module load GCC
# module load OpenMPI
# module load HDF5

# Create Python virtual environment
echo "Setting up Python venv..."
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pymetis

# Placeholder: Build DOLFINx and Ginkgo
# This section needs to be customized based on available modules
echo ""
echo "=== Setup complete ==="
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "NOTE: DOLFINx and Ginkgo builds need to be configured"
echo "based on available modules. Check 'module avail' and update"
echo "this script accordingly."
