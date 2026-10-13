"""preprocess -> text embedding -> RQ-VAE tokenizer for one dataset (each step is skipped if its output exists)."""
import argparse

from . import embed, preprocess, rqvae
from .utils import load_config, processed_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--fit", default=None, help="all_items = refit the tokenizer on every item incl. new ones")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not (processed_dir(a.dataset) / "data.pkl").exists():
        preprocess.build(a.dataset, cfg)
    embed.build(a.dataset, cfg)
    rqvae.build(a.dataset, cfg, a.K, a.L, a.fit)


if __name__ == "__main__":
    main()
