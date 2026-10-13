"""Dependency-aware multi-GPU job scheduler, resumable through done-files.

* Job(n_gpus=k) occupies k GPUs (CUDA_VISIBLE_DEVICES is set to exactly those); n_gpus=0 (or gpu=False) is a CPU job.
* A job whose done-file exists is skipped, so re-running the same command resumes an interrupted run.
* A failed job is retried once (OPENSID_JOB_RETRIES). If it still fails, its dependents are skipped and everything
  else continues.
* logs/_status.json is rewritten on every poll (running / pending / done / failed / skipped); every job writes its
  own log to logs/<job>.log.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import runtime


class Job:
    def __init__(self, name, cmd, done_file, deps=(), n_gpus=1, env=None, gpu=None):
        if gpu is not None:
            n_gpus = 1 if gpu else 0
        self.name, self.cmd, self.done_file = name, [str(c) for c in cmd], Path(done_file)
        self.deps, self.n_gpus, self.env = list(deps), n_gpus, env or {}
        self.status = "done" if self.done_file.exists() else "pending"
        self.proc = None
        self.slots = []
        self.attempts = 0
        self.t0 = None


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _write_status(log_dir, jobs):
    st = {s: [j.name for j in jobs if j.status == s] for s in ("pending", "done", "failed", "skipped")}
    st["running"] = [f"{j.name} gpus={j.slots} {(time.time() - j.t0) / 60:.0f}min" for j in jobs if j.status == "running"]
    st["time"] = time.ctime()
    try:
        (Path(log_dir) / "_status.json").write_text(json.dumps(st, indent=1))
    except OSError:
        pass


def run_jobs(jobs, log_dir, cwd, gpus=None, max_cpu_jobs=2, poll=10):
    runtime.setup_env()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = list(gpus) if gpus is not None else visible_gpus()
    free = list(gpus)
    max_retries = int(os.environ.get("OPENSID_JOB_RETRIES", "1"))
    by_name = {j.name: j for j in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("duplicate job names")
    for j in jobs:
        for d in j.deps:
            if d not in by_name:
                raise KeyError(f"{j.name} depends on unknown job {d}")
        if j.n_gpus > len(gpus):
            print(f"[scheduler] {j.name} wants {j.n_gpus} GPUs but only {len(gpus)} are visible -> "
                  f"{'CPU' if not gpus else len(gpus)}", flush=True)
            j.n_gpus = len(gpus)
    print(f"[scheduler] {len(jobs)} jobs ({sum(j.status == 'done' for j in jobs)} already done); gpus={gpus}",
          flush=True)
    cpu_running = 0

    def on_term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, AttributeError):
        pass

    try:
        while True:
            for j in jobs:
                if j.status == "pending" and any(by_name[d].status in ("failed", "skipped") for d in j.deps):
                    j.status = "skipped"
                    print(f"[scheduler] SKIP  {j.name} (dependency failed)", flush=True)
            for j in jobs:
                if j.status != "pending" or not all(by_name[d].status == "done" for d in j.deps):
                    continue
                if j.n_gpus == 0:
                    if cpu_running >= max_cpu_jobs:
                        continue
                    slots = []
                    cpu_running += 1
                else:
                    if len(free) < j.n_gpus:
                        continue
                    slots = free[:j.n_gpus]
                    del free[:j.n_gpus]
                j.slots = slots
                j.attempts += 1
                env = dict(os.environ, **j.env)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(slots)
                env["PYTHONUNBUFFERED"] = "1"
                f = open(log_dir / f"{j.name}.log", "a", encoding="utf-8")
                f.write(f"\n===== {time.ctime()} | attempt {j.attempts} | gpus={slots} | {' '.join(j.cmd)}\n")
                f.flush()
                j.proc = subprocess.Popen(j.cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
                j._f, j.t0, j.status = f, time.time(), "running"
                print(f"[scheduler] START {j.name} gpus={slots or 'cpu'}"
                      f"{'' if j.attempts == 1 else f' (retry {j.attempts - 1})'}", flush=True)
            for j in [x for x in jobs if x.status == "running"]:
                rc = j.proc.poll()
                if rc is None:
                    continue
                j._f.close()
                free.extend(j.slots)
                if j.n_gpus == 0:
                    cpu_running -= 1
                mins = (time.time() - j.t0) / 60
                if rc == 0 and j.done_file.exists():
                    j.status = "done"
                    print(f"[scheduler] DONE  {j.name} ({mins:.1f} min)", flush=True)
                else:
                    why = f"rc={rc}" if rc != 0 else f"rc=0 but {j.done_file} missing"
                    j.status = "pending" if j.attempts <= max_retries else "failed"
                    tag = "RETRY" if j.status == "pending" else "FAIL "
                    print(f"[scheduler] {tag} {j.name} {why} ({mins:.1f} min) -> {log_dir / (j.name + '.log')}",
                          flush=True)
            _write_status(log_dir, jobs)
            if not any(j.status in ("pending", "running") for j in jobs):
                break
            if not any(j.status == "running" for j in jobs):
                launchable = [j for j in jobs if j.status == "pending" and all(by_name[d].status == "done" for d in j.deps)]
                if not launchable:
                    for j in jobs:
                        if j.status == "pending":
                            j.status = "skipped"
                            failed = any(by_name[d].status in ("failed", "skipped") for d in j.deps)
                            why = "dependency failed" if failed else "unsatisfiable dependencies"
                            print(f"[scheduler] SKIP  {j.name} ({why})", flush=True)
                    break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("[scheduler] interrupted -> terminating running jobs", flush=True)
        for j in jobs:
            if j.status == "running" and j.proc is not None:
                j.proc.terminate()
        for j in jobs:
            if j.status == "running" and j.proc is not None:
                try:
                    j.proc.wait(60)
                except subprocess.TimeoutExpired:
                    j.proc.kill()
                j.status = "failed"
        _write_status(log_dir, jobs)
        raise
    summary = {s: [j.name for j in jobs if j.status == s] for s in ("done", "failed", "skipped")}
    print(f"[scheduler] finished: done={len(summary['done'])} failed={summary['failed']} skipped={summary['skipped']}",
          flush=True)
    return summary


def py(module, *args):
    return [sys.executable, "-m", module, *[str(a) for a in args]]
