"""Dataset downloader (idempotent, resumable). Honors HF_ENDPOINT (e.g. https://hf-mirror.com)."""
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

AMAZON_CATEGORIES = ["Industrial_and_Scientific", "Musical_Instruments", "Video_Games"]
AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
MIND_REPO = "huyva/MIND-small"  # mirror of official MINDsmall (train/dev behaviors.tsv + news.tsv)


def _hf_url(repo, path):
    return f"{HF}/datasets/{repo}/resolve/main/{path}"


def fetch(url, dst: Path, retries=5):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not Path(str(dst) + ".part").exists():
        return dst
    part = Path(str(dst) + ".part")
    for attempt in range(retries):
        try:
            pos = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={pos}-"} if pos else {})})
            tok = os.environ.get("HF_TOKEN")
            if tok and "huggingface" in url:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=120) as r:
                if pos and r.status != 206:
                    pos = 0
                    part.unlink(missing_ok=True)
                total = int(r.headers.get("Content-Length", 0)) + pos
                with open(part, "ab") as f:
                    done, t0 = pos, time.time()
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t0 > 30:
                            print(f"  {dst.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                            t0 = time.time()
            part.rename(dst)
            print(f"[download] ok {dst}", flush=True)
            return dst
        except Exception as e:  # noqa
            print(f"[download] retry {attempt + 1}/{retries} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def download_amazon(categories=AMAZON_CATEGORIES):
    for c in categories:
        fetch(_hf_url(AMAZON_REPO, f"benchmark/5core/rating_only/{c}.csv"), RAW / "amazon" / f"{c}.csv")
        fetch(_hf_url(AMAZON_REPO, f"raw/meta_categories/meta_{c}.jsonl"), RAW / "amazon" / f"meta_{c}.jsonl")


def download_mind():
    for split in ["train", "dev"]:
        for f in ["behaviors.tsv", "news.tsv"]:
            fetch(_hf_url(MIND_REPO, f"{split}/{f}"), RAW / "mind" / split / f)


def main():
    # a dead mirror must not abort run_all.py: the affected dataset's prepare job then fails on its own (and only
    # its dependents are skipped)
    for fn in (download_amazon, download_mind):
        try:
            fn()
        except Exception as e:  # noqa
            print(f"[download] {fn.__name__} FAILED ({type(e).__name__}: {e}); continuing", flush=True)


if __name__ == "__main__":
    main()
