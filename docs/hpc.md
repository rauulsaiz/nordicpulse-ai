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

The script loads a Python module and creates the venv in
`$BLACKHOLE/nordicpulse/.venv`. It installs the package in editable mode with the
`parsing` extra (docling, pymupdf, plus pyyaml, requests, pypdf), using the pip cache in
`$BLACKHOLE/.cache/pip`. Finally it pre-downloads the Docling models into
`$BLACKHOLE/.cache/huggingface`, which is where the job looks for them. By default it
uses the newest `python3` ≥ 3.11 module listed by `module avail python3`. To use a
specific module instead:

```bash
module avail python3
PYTHON_MODULE=python3/<version> bash scripts/hpc/setup_env.sh
```

Seeing `cuda available on this node: False` here is expected, because login nodes have
no GPU.

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

`scripts/hpc/parse_reports.lsf` uses the `gpuv100` queue (DTU also documents `gpua100`).
Check which one is available and less busy:

```bash
bqueues
nodestat -G gpuv100
nodestat -G gpua100
```

If needed, edit this line in `scripts/hpc/parse_reports.lsf`:

```bash
#BSUB -q gpuv100
```

The job requests 1 GPU, 8 cores on one host, 4 GB per core (32 GB total) and 4 h of
walltime. It activates the venv from `$BLACKHOLE` and sets `HF_HOME` and `TORCH_HOME`
to the caches there. Submit it from the repository root:

```bash
cd $BLACKHOLE/nordicpulse-ai
mkdir -p logs                          # LSF does not create the log directory
bsub < scripts/hpc/parse_reports.lsf
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

- the `nvidia-smi` table at the top shows a GPU;
- the `Venv` / `HF_HOME` / `TORCH_HOME` line shows paths under `$BLACKHOLE`;
- the log says `cuda available: True` and `Docling pipeline: ... device=cuda`;
- there is one `Docling done, N pages, T tables in S s` line per report, followed by a
  final summary.

If the log says `CUDA NOT available`, Docling ran on CPU: check the queue, and compare
`torch.version.cuda` with the driver version shown by `nvidia-smi`. The job ends in
state `EXIT` (exit code 1) if any report failed or its PDF was missing; the other reports
are still processed.

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
