#!/bin/bash
# Create (or re-create) the Python virtual environment for PDF parsing on the DTU HPC.
#
# Run on a login node, from the repository root:
#     bash scripts/hpc/setup_env.sh
#
# Steps: load a Python 3.12/3.11 module, (re)create the venv, install PyTorch from the
# official PyTorch index with a CUDA variant that supports the cluster GPUs, install
# this package with the [parsing] extra (docling, pymupdf, ...) WITHOUT letting pip
# replace that torch build, and pre-download the Docling models.
#
# Everything large (venv, pip cache, Hugging Face / torch model caches) goes to the DTU
# scratch area $BLACKHOLE instead of the size-limited home directory. $BLACKHOLE is
# TEMPORARY: it is wiped in maintenance windows (and at the end of January 2027), after
# which this script simply has to be run again.

set -eo pipefail  # no -u: environment-module shell functions may reference unset variables

if [[ -z "$BLACKHOLE" || ! -d "$BLACKHOLE" ]]; then
    echo "ERROR: \$BLACKHOLE is not set or does not exist (see /dtu/blackhole/readme.txt)." >&2
    exit 1
fi
if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: run this script from the repository root." >&2
    exit 1
fi
# Keep these cache paths identical to the ones exported in parse_reports.lsf, so the
# job finds the models pre-downloaded here.
export PIP_CACHE_DIR="$BLACKHOLE/.cache/pip"
export HF_HOME="$BLACKHOLE/.cache/huggingface"
export TORCH_HOME="$BLACKHOLE/.cache/torch"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

# --------------------------------------------------------------------------------------
# Python module. PyTorch/Docling wheels lag behind new Python releases (a 3.15 venv gave
# a torch build that could not run on the V100s), so only 3.12 (preferred) or 3.11 are
# accepted. By default the newest python3/3.12.x module is used, else python3/3.11.x,
# as listed by `module avail python3`. To pin one from that list:
#     PYTHON_MODULE=python3/<version> bash scripts/hpc/setup_env.sh
PYTHON_MODULE="${PYTHON_MODULE:-}"

# PyTorch CUDA variant, from the official index https://download.pytorch.org/whl/.
# cu126 (CUDA 12.6) is the last variant whose wheels include Volta kernels (sm_70,
# V100) and it also covers A100 (sm_80); PyTorch 2.14 is the last release built for it.
# Newer variants (cu128/cu13x) drop sm_70. The driver must support CUDA 12.6: check the
# "CUDA Version" printed by nvidia-smi (scripts/hpc/check_gpu.sh).
TORCH_CUDA="${TORCH_CUDA:-cu126}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
# --------------------------------------------------------------------------------------

VENV_DIR="${VENV_DIR:-$BLACKHOLE/nordicpulse/.venv}"
mkdir -p "$(dirname "$VENV_DIR")"

if [[ -z "$PYTHON_MODULE" ]]; then
    # Terse listing (one module per line, printed on stderr); drop "(default)"-style tags.
    available="$(module -t avail python3 2>&1 | sed 's/(.*)//')"
    for minor in 12 11; do
        PYTHON_MODULE="$(grep -E "^python3/3\.${minor}(\.|$)" <<<"$available" | sort -V | tail -n 1)"
        [[ -n "$PYTHON_MODULE" ]] && break
    done
    if [[ -z "$PYTHON_MODULE" ]]; then
        echo "ERROR: no python3/3.12.x or python3/3.11.x module found. Available:" >&2
        grep '^python3/' <<<"$available" >&2 || true
        exit 1
    fi
    echo "Auto-selected Python module: $PYTHON_MODULE"
fi

module load "$PYTHON_MODULE"
echo "Using $(command -v python3): $(python3 --version)"
case "$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" in
    3.11|3.12) ;;
    *) echo "ERROR: $PYTHON_MODULE is not Python 3.11/3.12 (see comment above)." >&2; exit 1 ;;
esac

# --clear wipes an existing venv (e.g. the old Python 3.15 one) before re-creating it.
echo "(Re)creating venv $VENV_DIR ..."
python3 -m venv --clear "$VENV_DIR"
# Record the module so the LSF jobs load exactly the same one (the venv's python
# binary links against the module's shared libraries).
echo "$PYTHON_MODULE" > "$VENV_DIR/hpc_python_module"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# 1) PyTorch + torchvision (Docling needs both) from the official CUDA-specific index.
echo "Installing torch + torchvision from $TORCH_INDEX_URL ..."
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

# 2) The package and its [parsing] extra from PyPI, with torch/torchvision pinned to
#    the builds above (otherwise pip may "upgrade" them to PyPI wheels without sm_70).
constraints="$VENV_DIR/torch-constraints.txt"
python -m pip freeze | grep -E '^(torch|torchvision)==' > "$constraints"
echo "Pinned: $(tr '\n' ' ' < "$constraints")"
python -m pip install -e ".[parsing]" --constraint "$constraints"

echo
echo "Installed versions:"
python - <<'EOF'
import importlib.metadata as md

import torch

for pkg in ("docling", "pymupdf", "torch", "torchvision", "pypdf", "pyyaml", "requests"):
    try:
        print(f"  {pkg:12s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:12s} NOT INSTALLED")
print(f"  torch CUDA build : {torch.version.cuda}")
print(f"  compiled for     : {torch.cuda.get_arch_list() or '(listed on GPU nodes only)'}")
print("  (cuda is not available on a login node: verify on a GPU node with check_gpu.sh)")
EOF

echo
echo "Pre-downloading Docling models (layout + TableFormer) ..."
if ! docling-tools models download; then
    echo "WARNING: model pre-download failed; Docling will try to download them at job start." >&2
fi

echo
echo "Done. Virtual environment: $VENV_DIR (module: $PYTHON_MODULE, torch: $TORCH_CUDA)"
echo "      Model caches: $HF_HOME, $TORCH_HOME"
echo "Next: verify the GPU with 'bsub < scripts/hpc/check_gpu.lsf' (see docs/hpc.md)."
