"""Environment helpers used by run_all.py and the job scheduler (standard library only).

* setup_env()        sensible defaults for tokenizers / CUDA device order / UTF-8 output
* gpu_memory()       {nvidia-smi index: (used MiB, total MiB)}
* report_gpus()      prints the visible GPUs and their memory use
* ensure_packages()  installs only the packages that are missing (never upgrades an installed torch)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_DEFAULTS = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",   # CUDA ids == nvidia-smi ids
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUTF8": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def setup_env():
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------------------------- GPUs
def gpu_memory():
    """{nvidia-smi index: (used_MiB, total_MiB)}; {} if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            try:
                res[p[0]] = (float(p[1]), float(p[2]))
            except ValueError:
                pass
    return res


def visible_physical_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    return sorted(gpu_memory(), key=int)


def report_gpus(log=print):
    mem = gpu_memory()
    vis = visible_physical_gpus()
    log(f"[run_all] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} -> GPUs {vis}")
    for g in vis:
        if g in mem:
            used, tot = mem[g]
            log(f"[run_all]   gpu{g}: {used / 1024:.1f}/{tot / 1024:.1f} GiB used")


# ---------------------------------------------------------------------------------------------- packages
def _importable(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def ensure_packages(reqs, log=print):
    """reqs: list of (import_name, pip_spec). Installs ONLY the missing ones (no --upgrade) and pins the installed
    torch version, so an existing CUDA build of torch is never replaced. Returns the modules that are still missing."""
    missing = [(m, s) for m, s in reqs if not _importable(m)]
    if not missing:
        return []
    if os.environ.get("OPENSID_NO_PIP"):
        log(f"[run_all] missing {[m for m, _ in missing]} and OPENSID_NO_PIP is set -> not installing")
        return [m for m, _ in missing]
    specs = [s for _, s in missing]
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off"]
    cons = None
    try:
        from importlib.metadata import version
        cons = Path(tempfile.gettempdir()) / f"opensid_constraints_{os.getpid()}.txt"
        cons.write_text(f"torch=={version('torch')}\n")
        cmd += ["-c", str(cons)]
    except Exception:
        cons = None
    log(f"[run_all] pip installing missing packages: {specs}")
    subprocess.call(cmd + specs)
    if cons is not None:
        cons.unlink(missing_ok=True)
    import importlib
    importlib.invalidate_caches()
    still = [m for m, _ in missing if not _importable(m)]
    if still:
        log(f"[run_all] WARNING: could not install {still}; jobs that need them will fail (see pip output above)")
    return still
