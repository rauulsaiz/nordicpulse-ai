# Parsing the reports on the DTU HPC (LSF)

This guide runs `python -m nordicpulse.ingestion.parse` (PyMuPDF + Docling) as a GPU
batch job on the DTU HPC cluster, which uses the LSF scheduler (`bsub`, `bjobs`, ...).
Run every command on a **login node** unless stated otherwise; the heavy work only
happens inside the batch job.

Outputs (per report id `{company}_{year}`, e.g. `vestas_2025`):

| File | Content |
|---|---|
| `data/processed/pymupdf/{id}.jsonl` | one line per page: `{"page": n, "text": ...}`, pages from 1 |
| `data/processed/docling/{id}.md` | Docling Markdown export |
| `data/processed/docling/{id}.json` | full Docling document (provenance, page numbers, tables) |
| `data/processed/processing_summary.csv` | id, n_pages, device, docling_seconds, pymupdf_seconds, n_tables_docling, status |

## Storage: everything lives in `$BLACKHOLE` (temporary!)

The repository, the virtual environment and the model caches all go to the DTU scratch
area, whose path is in the environment variable `$BLACKHOLE`. This keeps torch, Docling
and the Hugging Face models (several GB) out of the size-limited home directory.

| What | Where |
|---|---|
| Repository (PDFs + results) | `$BLACKHOLE/nordicpulse-ai` |
| Virtual environment | `$BLACKHOLE/nordicpulse/.venv` |
| pip cache | `$BLACKHOLE/.cache/pip` |
| Docling / Hugging Face models | `$BLACKHOLE/.cache/huggingface` (`HF_HOME`) |
| torch cache | `$BLACKHOLE/.cache/torch` (`TORCH_HOME`) |

> **`$BLACKHOLE` is temporary.** It is wiped during maintenance windows and at the end of
> January 2027, and it is not backed up. Nothing there is safe: **copy the results to your
> laptop as soon as the job finishes (step 8).** After a wipe, just repeat steps 2-5;
> the environment and the models are recreated by `setup_env.sh`.

## 1. Log in and read the blackhole rules

From the DTU network (campus or DTU VPN):

```bash
ssh <dtu-username>@login1.hpc.dtu.dk
```

From outside the DTU network you also need a registered SSH key; see
<https://www.hpc.dtu.dk/?page_id=4317>.

Before doing anything else, read the current rules of the scratch area and check that
the variable is set:

```bash
cat /dtu/blackhole/readme.txt
echo $BLACKHOLE
```

## 2. Clone the repository into `$BLACKHOLE`

```bash
cd $BLACKHOLE
git clone https://github.com/rauulsaiz/nordicpulse-ai.git
cd $BLACKHOLE/nordicpulse-ai
```

All later commands on the HPC are run from `$BLACKHOLE/nordicpulse-ai`.

## 3. Create the environment (once per blackhole wipe)

```bash
bash scripts/hpc/setup_env.sh
```

The script does the following:

1. Loads a **Python 3.12 module, or 3.11 if there is no 3.12**, from those listed by
   `module avail python3`. Newer Pythons (3.13+) are refused; see Troubleshooting.
2. (Re)creates the venv in `$BLACKHOLE/nordicpulse/.venv`, wiping any previous one.
3. Installs **torch + torchvision from the official PyTorch index
   `https://download.pytorch.org/whl/cu126`**. This CUDA 12.6 build includes kernels for
   both V100 (sm_70) and A100 (sm_80).
4. Installs the package in editable mode with the `parsing` extra (docling, pymupdf,
   plus pyyaml, requests, pypdf), keeping torch pinned to that build. The pip cache is
   in `$BLACKHOLE/.cache/pip`.
5. Pre-downloads the Docling models into `$BLACKHOLE/.cache/huggingface`, which is
   where the job looks for them.

To pin a specific module from `module avail python3`:

```bash
PYTHON_MODULE=python3/<version> bash scripts/hpc/setup_env.sh
```

The login node has no GPU, so the GPU itself is checked in the next step.

## 3b. Check that PyTorch runs on the GPU (after every setup)

`scripts/hpc/check_gpu.sh` prints `nvidia-smi`, `torch.__version__`,
`torch.version.cuda`, `torch.cuda.get_arch_list()` and the GPU's compute capability. It
then runs a small CUDA reduction and a cuDNN convolution, and exits with code 1 if
anything fails. Run it on a GPU node, with either of these:

```bash
# a) short batch job (10 min) in the queue you will use; -q overrides the file's queue
mkdir -p logs
bsub < scripts/hpc/check_gpu.lsf                 # gpua100
bsub -q gpuv100 < scripts/hpc/check_gpu.lsf      # or the V100 queue
# when it finishes: cat logs/check_gpu_<jobid>.out   -> must end with "check_gpu: PASSED"

# b) interactive GPU node (DTU: voltash / sxm2sh = V100, a100sh = A100), then:
cd $BLACKHOLE/nordicpulse-ai && bash scripts/hpc/check_gpu.sh
exit
```

The job script `parse_reports.lsf` also runs this check first and aborts if it fails.

## 4. Download the PDFs (on the login node)

Compute nodes may not have internet access, so download the PDFs before submitting:

```bash
source $BLACKHOLE/nordicpulse/.venv/bin/activate
python -m nordicpulse.ingestion.download
deactivate
```

This should leave six PDFs in `data/raw/`: `novo_nordisk_2024`, `novo_nordisk_2025`,
`orsted_2024`, `orsted_2025`, `vestas_2024`, `vestas_2025`.

## 5. Check the queue and submit the job

`scripts/hpc/parse_reports.lsf` uses the `gpua100` queue (A100). The PyTorch build also
supports the `gpuv100` queue (V100). Both queues are documented by DTU. Check that they
are available to you and how busy they are:

```bash
bqueues
nodestat -G gpua100
nodestat -G gpuv100
```

The job requests 1 GPU, 8 cores on one host, 4 GB per core (32 GB total) and 4 h of
walltime. It activates the venv from `$BLACKHOLE`, sets `HF_HOME` and `TORCH_HOME` to
the caches there and runs `check_gpu.sh` before parsing. Submit it from the repository
root:

```bash
cd $BLACKHOLE/nordicpulse-ai
mkdir -p logs                          # LSF does not create the log directory
bsub < scripts/hpc/parse_reports.lsf                 # gpua100
bsub -q gpuv100 < scripts/hpc/parse_reports.lsf      # or: V100 queue
```

To process only some reports, or to redo reports that are already done:

```bash
PARSE_ARGS="--ids vestas_2025 orsted_2025" bsub < scripts/hpc/parse_reports.lsf
PARSE_ARGS="--force" bsub < scripts/hpc/parse_reports.lsf
```

## 6. Follow the job

```bash
bjobs                              # PEND / RUN; disappears from the list when finished
tail -f logs/parse_<jobid>.out     # progress log (Ctrl+C to stop following)
tail -f logs/parse_<jobid>.err     # tracebacks and library warnings
```

In `logs/parse_<jobid>.out` check that:

- the `Venv` / `HF_HOME` / `TORCH_HOME` line shows paths under `$BLACKHOLE`;
- the GPU check shows `nvidia-smi`, the GPU in `get_arch_list()` and `check_gpu: PASSED`;
- the parser logs `GPU smoke test passed`, `device: cuda` and
  `Docling pipeline: ... device=cuda`;
- there is one `Docling done, N pages, T tables in S s` line per report, followed by a
  final summary.

The job ends in state `EXIT` (exit code 1) if any report failed or its PDF was missing.
An ordinary failure (e.g. a broken PDF) does not stop the other reports. A **CUDA error
stops the whole run** (`ABORTED, GPU unusable`), because if the GPU does not work no
report can succeed.

## 7. Resume after an interruption

If the job is killed or reaches the walltime, submit it again (step 5). It skips the
reports whose three outputs already exist and continues with the rest.

## 8. ⚠️ MANDATORY: copy the results to your laptop

**Do this as soon as the job has finished.** `$BLACKHOLE` can be wiped at any
maintenance window, and anything not copied is lost.

On your laptop, from the repository root, replacing the placeholders. Use the full
path: run `echo $BLACKHOLE` on the HPC to see it.

```bash
rsync -avz <dtu-username>@<hpc-login-host>:<blackhole-path>/nordicpulse-ai/data/processed/ data/processed/
```

On Windows, where `rsync` is not available by default, use `scp` from PowerShell
instead:

```powershell
scp -r <dtu-username>@<hpc-login-host>:<blackhole-path>/nordicpulse-ai/data/processed/* data/processed/
```

Optionally also copy the logs (same command with `logs/` instead of `data/processed/`).
Then open `data/processed/processing_summary.csv`. Every row should have status `ok`
(or `skipped`, if an earlier run already processed it) and device `cuda`.

## Troubleshooting

### "CUDA error: no kernel image is available for execution on the device"

**Symptom (job 30104585, November 2026).** On the `gpuv100` queue, Docling's layout model
failed with `torch.AcceleratorError: CUDA error: no kernel image is available for
execution on the device`. Docling records page errors and moves on, so the job kept
failing page after page instead of stopping.

**Cause.** The installed PyTorch had no compiled kernels for the GPU's architecture.
The V100 is compute capability 7.0 (`sm_70`, Volta). The first `setup_env.sh` picked the
newest Python module (3.15) and installed torch from PyPI. Recent PyTorch CUDA builds
(CUDA 12.8+ in newer releases, and all CUDA 13 builds) no longer include `sm_70`.
PyTorch's last builds that do are the **CUDA 12.6 (`cu126`)** wheels, up to PyTorch 2.14.

**Fix.**

1. `setup_env.sh` now uses Python 3.12 (or 3.11) and installs torch + torchvision from
   `https://download.pytorch.org/whl/cu126`, pinned so that pip does not replace them.
   This build includes both `sm_70` (V100) and `sm_80` (A100).
2. `check_gpu.sh` verifies the build on a GPU node: the GPU's `sm_XY` must appear in
   `get_arch_list()` and the CUDA operations must pass.
3. `parse_reports.lsf` runs `check_gpu.sh` first, and the parser runs a GPU smoke test
   before loading Docling. Any CUDA error in a report stops the run (marked `failed`
   in `processing_summary.csv`) instead of failing every page.
4. The default queue is now `gpua100` (A100); `gpuv100` works too with this build.

**To apply it** (the old venv is wiped automatically):

```bash
cd $BLACKHOLE/nordicpulse-ai && git pull
bash scripts/hpc/setup_env.sh
bsub < scripts/hpc/check_gpu.lsf          # wait, then: cat logs/check_gpu_<jobid>.out
bsub < scripts/hpc/parse_reports.lsf
```

**Related checks.**

- The CUDA version in the top-right corner of `nvidia-smi` (the newest the driver
  supports) must be ≥ `torch.version.cuda` (12.6).
- PyTorch 2.15+ will not publish `cu126` wheels, so V100 support stays on 2.14. If a
  future setup needs a newer torch, use the A100 queue.

Sources: [PyTorch notice on dropping cu126 / Volta from 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432),
[DTU GPU nodes](https://www.hpc.dtu.dk/?page_id=2129),
[DTU: using GPUs under LSF10](https://www.hpc.dtu.dk/?page_id=2759).
