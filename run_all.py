"""One-command entry point for all experiments of the paper.

    CUDA_VISIBLE_DEVICES=0,1 nohup python run_all.py > run.log 2>&1 &

Steps: (1) download the datasets if missing  (2) per-dataset preprocessing + RQ-VAE tokenizer
       (3) all training / evaluation jobs, one job per GPU, in dependency order
       (4) results/summary.md and results/main_table.csv (python -m opensid.aggregate)
Re-running the same command resumes: a job whose done-file results/<dataset>/<exp>/done exists is skipped.
Monitoring: logs/_status.json (running / pending / done / failed), logs/<job>.log, run.log.
Options:
    --quick            main methods only, one seed (sanity run)
    --datasets a,b     subset of datasets
    --config path      alternative YAML config
    --skip_download
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from opensid import runtime  # noqa: E402  (standard library only)

runtime.setup_env()

REQS = [("yaml", "pyyaml"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("transformers", "transformers>=4.51"),
        ("sentence_transformers", "sentence-transformers>=3.0"),
        ("sentencepiece", "sentencepiece"),
        ("google.protobuf", "protobuf"),
        ("tqdm", "tqdm"),
        ("accelerate", "accelerate")]


def check_env():
    """installs only MISSING packages (an installed torch is never upgraded)."""
    runtime.ensure_packages(REQS)
    import torch
    print(f"[run_all] python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    runtime.report_gpus(lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip_download", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    check_env()
    from opensid import download
    from opensid.experiments import build_jobs
    from opensid.scheduler import run_jobs

    if not a.skip_download:
        download.main()
    datasets = a.datasets.split(",") if a.datasets else None
    jobs = build_jobs(a.config, datasets, a.quick)
    summary = run_jobs(jobs, ROOT / "logs", ROOT)
    subprocess.call([sys.executable, "-m", "opensid.aggregate", *(["--config", a.config] if a.config else [])], cwd=ROOT)
    print(f"[run_all] finished in {(time.time() - t0) / 3600:.1f} h; failed={summary['failed']} "
          f"skipped={summary['skipped']}. Summary: results/summary.md", flush=True)


if __name__ == "__main__":
    main()
