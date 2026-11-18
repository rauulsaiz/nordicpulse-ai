#!/bin/bash
# Create the Python virtual environment for PDF parsing on the DTU HPC.
#
# Run ONCE on a login node, from the repository root:
#     bash scripts/hpc/setup_env.sh
#
# It loads a Python module, creates a venv, installs this package in editable mode
# with the parsing extras (docling, pymupdf + base deps pyyaml, requests, pypdf) and
# pre-downloads the Docling models so the batch job does not depend on network access.

set -eo pipefail  # no -u: environment-module shell functions may reference unset variables

# --------------------------------------------------------------------------------------
# Python module to load (Python >= 3.11 is required by pyproject.toml).
# By default the newest "python3/3.1x" module (3.11 or later) reported by
# `module avail python3` is picked automatically. To pin a specific one, list the
# available modules on a login node with:
#     module avail python3
# and set it here or on the command line, e.g.:
#     PYTHON_MODULE=python3/<version> bash scripts/hpc/setup_env.sh
PYTHON_MODULE="${PYTHON_MODULE:-}"
# --------------------------------------------------------------------------------------

# Location of the virtual environment (override with: VENV_DIR=/path bash setup_env.sh).
VENV_DIR="${VENV_DIR:-$PWD/.venv}"

if [[ -z "$PYTHON_MODULE" ]]; then
    # Terse listing (one module per line, printed on stderr); drop "(default)"-style tags.
    PYTHON_MODULE="$(module -t avail python3 2>&1 \
        | sed 's/(.*)//' | grep -E '^python3/3\.(1[1-9]|[2-9][0-9])(\.|$)' \
        | sort -V | tail -n 1)"
    if [[ -z "$PYTHON_MODULE" ]]; then
        echo "ERROR: no python3 >= 3.11 module found. Run 'module avail python3' and" >&2
        echo "       set PYTHON_MODULE at the top of $0." >&2
        exit 1
    fi
    echo "Auto-selected Python module: $PYTHON_MODULE"
fi
if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: run this script from the repository root." >&2
    exit 1
fi

module load "$PYTHON_MODULE"
echo "Using $(command -v python3): $(python3 --version)"

python3 -m venv "$VENV_DIR"
# Record the module so the LSF job loads exactly the same one (the venv's python
# binary links against the module's shared libraries).
echo "$PYTHON_MODULE" > "$VENV_DIR/hpc_python_module"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# Base deps (requests, pyyaml, pypdf[crypto]) come from [project.dependencies];
# docling and pymupdf come from the "parsing" extra.
python -m pip install -e ".[parsing]"

echo
echo "Installed versions:"
python - <<'EOF'
import importlib.metadata as md
for pkg in ("docling", "pymupdf", "torch", "pypdf", "pyyaml", "requests"):
    try:
        print(f"  {pkg:10s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:10s} NOT INSTALLED")
import torch
print(f"  torch CUDA build: {torch.version.cuda} "
      f"(cuda available on this node: {torch.cuda.is_available()}; False is expected on a login node)")
EOF

echo
echo "Pre-downloading Docling models (layout + TableFormer) ..."
if ! docling-tools models download; then
    echo "WARNING: model pre-download failed; Docling will try to download them at job start." >&2
fi

echo
echo "Done. Virtual environment: $VENV_DIR (module: $PYTHON_MODULE)"
