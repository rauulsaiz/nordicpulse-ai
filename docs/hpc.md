# Parsing the reports on the DTU HPC (LSF)

This guide runs `python -m nordicpulse.ingestion.parse` (PyMuPDF + Docling) as a GPU
batch job on the DTU HPC cluster, which uses the LSF scheduler (`bsub`, `bjobs`, ...).
Run every command on a **login node**, from the repository root; the heavy work only
happens inside the batch job.

Outputs (per report id `{company}_{year}`, e.g. `vestas_2025`):

| File | Content |
|---|---|
| `data/processed/pymupdf/{id}.jsonl` | one line per page: `{"page": n, "text": ...}`, pages from 1 |
| `data/processed/docling/{id}.md` | Docling Markdown export |
| `data/processed/docling/{id}.json` | full Docling document (provenance, page numbers, tables) |
| `data/processed/processing_summary.csv` | id, n_pages, device, docling_seconds, pymupdf_seconds, n_tables_docling, status |

## 1. Clone the repository

```bash
git clone https://github.com/rauulsaiz/nordicpulse-ai.git
cd nordicpulse-ai
```

## 2. Create the environment (once)

```bash
bash scripts/hpc/setup_env.sh
```

The script loads a Python module, creates `.venv/`, installs the package in editable
mode with the `parsing` extra (docling, pymupdf, plus pyyaml, requests, pypdf) and
pre-downloads the Docling models. By default it picks the newest `python3` ≥ 3.11
module listed by `module avail python3`. To use a specific one instead:

```bash
module avail python3
PYTHON_MODULE=python3/<version> bash scripts/hpc/setup_env.sh
```

The module name is saved in `.venv/hpc_python_module`, and the batch job loads the same
module. Seeing `cuda available on this node: False` here is expected, because login nodes
have no GPU. The install takes several GB (torch + models). If you run out of home quota,
re-run with `VENV_DIR` and `HF_HOME` pointing to a larger scratch area, and export the
same variables before submitting the job.

## 3. Download the PDFs (on the login node)

Compute nodes may not have internet access, so download the PDFs before submitting:

```bash
source .venv/bin/activate
python -m nordicpulse.ingestion.download
deactivate
```

This should leave six PDFs in `data/raw/`: `novo_nordisk_2024`, `novo_nordisk_2025`,
`orsted_2024`, `orsted_2025`, `vestas_2024`, `vestas_2025`.

## 4. Check the queue and submit the job

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
walltime. Submit it:

```bash
mkdir -p logs                          # LSF does not create the log directory
bsub < scripts/hpc/parse_reports.lsf
```

To process only some reports, or to redo reports that are already done:

```bash
PARSE_ARGS="--ids vestas_2025 orsted_2025" bsub < scripts/hpc/parse_reports.lsf
PARSE_ARGS="--force" bsub < scripts/hpc/parse_reports.lsf
```

## 5. Follow the job

```bash
bjobs                              # PEND / RUN; disappears from the list when finished
tail -f logs/parse_<jobid>.out     # progress log (Ctrl+C to stop following)
tail -f logs/parse_<jobid>.err     # tracebacks and library warnings
```

In `logs/parse_<jobid>.out` check that:

- the `nvidia-smi` table at the top shows a GPU;
- the log says `cuda available: True` and `Docling pipeline: ... device=cuda`;
- there is one `Docling done, N pages, T tables in S s` line per report, followed by a
  final summary.

If the log says `CUDA NOT available`, Docling ran on CPU: check the queue, and compare
`torch.version.cuda` with the driver version shown by `nvidia-smi`. The job ends in
state `EXIT` (exit code 1) if any report failed or its PDF was missing; the other reports
are still processed. If the job is killed or reaches the walltime, submit it again. It
skips the reports whose three outputs already exist and continues with the rest.

## 6. Copy the results to your laptop

Run this on your laptop, from the repository root, replacing the placeholders:

```bash
rsync -avz <dtu-username>@<hpc-login-host>:~/nordicpulse-ai/data/processed/ data/processed/
```

Then open `data/processed/processing_summary.csv`. Every row should have status `ok`
(or `skipped`, if an earlier run already processed it) and device `cuda`.
