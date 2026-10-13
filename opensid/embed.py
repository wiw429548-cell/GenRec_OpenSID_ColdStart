"""Encode item text with a frozen sentence encoder -> data/processed/<ds>/item_emb.npy (L2-normalized)."""
import argparse

import numpy as np
import torch

from .utils import get_logger, load_config, load_pickle, processed_dir

log = get_logger("embed")


def build(ds, cfg):
    out = processed_dir(ds) / "item_emb.npy"
    if out.exists():
        return np.load(out)
    from sentence_transformers import SentenceTransformer

    data = load_pickle(processed_dir(ds) / "data.pkl")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg["embedding"]["model"], device=dev)
    emb = model.encode(
        data["item_text"], batch_size=cfg["embedding"]["batch_size"], show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.save(out, emb)
    log.info(f"[{ds}] item embeddings {emb.shape}")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    build(a.dataset, load_config(a.config))


if __name__ == "__main__":
    main()
