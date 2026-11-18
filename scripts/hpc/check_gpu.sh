#!/bin/bash
# Check that PyTorch in the project venv can really run on this node's GPU.
#
# Prints nvidia-smi, the torch build (version, CUDA version, compiled GPU architectures)
# and the GPU's compute capability, then runs two tiny CUDA operations (a reduction
# and a cuDNN convolution, like Docling's layout model). Exits 1 if anything fails.
#
# Run it ON A GPU NODE, from the repository root:
#   - interactively, after `voltash` / `sxm2sh` (V100) or `a100sh` (A100):
#         bash scripts/hpc/check_gpu.sh
#   - or as a short batch job in a given queue:
#         bsub < scripts/hpc/check_gpu.lsf               # queue set in the file
#         bsub -q gpuv100 < scripts/hpc/check_gpu.lsf    # -q on the command line wins
#
# A typical failure is "no kernel image is available for execution on the device":
# the installed torch was not compiled for this GPU (e.g. no sm_70 for a V100).

set -eo pipefail  # no -u: environment-module shell functions may reference unset variables

VENV_DIR="${VENV_DIR:-$BLACKHOLE/nordicpulse/.venv}"

echo "=== check_gpu on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== nvidia-smi"
if ! nvidia-smi; then
    echo "FAIL: nvidia-smi failed: no GPU visible on this node/job." >&2
    exit 1
fi

if [[ ! -f "$VENV_DIR/hpc_python_module" ]]; then
    echo "FAIL: venv $VENV_DIR not set up. Run: bash scripts/hpc/setup_env.sh" >&2
    exit 1
fi
module load "$(cat "$VENV_DIR/hpc_python_module")"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "=== PyTorch"
python - <<'EOF'
import sys

import torch

print(f"python            : {sys.version.split()[0]} ({sys.executable})")
print(f"torch.__version__ : {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False (CPU-only torch, or no GPU in this job)")

major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
archs = torch.cuda.get_arch_list()
print(f"device            : {torch.cuda.get_device_name(0)}")
print(f"capability        : {(major, minor)} -> {arch}")
print(f"get_arch_list()   : {archs}")
if arch in archs:
    print(f"OK  : this torch build has kernels for {arch}")
else:
    print(f"WARN: {arch} is NOT in get_arch_list(); CUDA ops will most likely fail")

try:
    total = torch.ones(10, device="cuda").sum().item()
    assert total == 10, total
    x = torch.randn(1, 3, 64, 64, device="cuda")
    w = torch.randn(8, 3, 3, 3, device="cuda")
    torch.nn.functional.conv2d(x, w).sum().item()
    torch.cuda.synchronize()
except Exception as exc:  # e.g. torch.AcceleratorError: no kernel image is available
    sys.exit(f"FAIL: CUDA operation failed on {arch}: {type(exc).__name__}: {exc}")
print("OK  : CUDA reduction and cuDNN convolution ran on the GPU")
EOF
echo "=== check_gpu: PASSED"
