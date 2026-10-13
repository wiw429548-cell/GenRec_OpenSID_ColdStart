"""Non-parametric reference points with the same protocol and metrics.

pop          : most popular items in the train period (cannot recommend new items by construction)
content_knn  : rank ALL catalogue items by cosine(profile of the history, item text embedding) - a pure content
               cold-start baseline that needs no training
recent_pop   : popularity in the most recent 10% of the train period (captures trends; still no new items)
"""
import argparse
import time

import numpy as np
import torch

from .data import SplitArrays
from .evaluate import compute_metrics
from .sid import SIDCatalog
from .utils import ROOT, load_config, load_pickle, processed_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    pdir = processed_dir(a.dataset)
    data = load_pickle(pdir / "data.pkl")
    emb = np.load(pdir / "item_emb.npy")
    K, L = cfg["tokenizer"]["codebook_size"], cfg["tokenizer"]["levels"]
    cat = SIDCatalog(load_pickle(pdir / f"sid_K{K}_L{L}" / "tokenizer.pkl")["codes"], K, data["warm"])
    H = cfg["preprocess"]["max_hist_len"]
    tr = SplitArrays(data, "train", H)
    te = SplitArrays(data, "test", H, emb)
    n = len(data["item_ids"])
    cnt = np.bincount(tr.tgt, minlength=n).astype(float)
    recent = tr.tgt[tr.ts >= np.quantile(tr.ts, 0.9)] if len(tr.ts) else tr.tgt
    rcnt = np.bincount(recent, minlength=n).astype(float)
    out = ROOT / cfg["out_dir"] / a.dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.tensor(emb, device=dev)
    FT = torch.tensor(data["first_ts"], device=dev)
    C, RC = torch.tensor(cnt, device=dev), torch.tensor(rcnt, device=dev)

    def rank_split(sp):
        knn, pop, rpop = [], [], []
        t0 = time.time()
        for s in range(0, sp.N, 1024):
            q = torch.tensor(sp.profile[s:s + 1024], device=dev)
            sc = q @ E.t()
            hb = torch.tensor(sp.hist[s:s + 1024], device=dev)
            r_, c_ = (hb >= 0).nonzero(as_tuple=True)
            sc[r_, hb[r_, c_]] = -1e9  # do not recommend the history items themselves
            pc, rc = C[None, :].repeat(len(q), 1), RC[None, :].repeat(len(q), 1)
            if cfg.get("time_aware_catalog", True):
                # same time-aware catalogue as every other method (matters for LOO, where train interactions of other
                # users can be later than this test interaction)
                fut = FT[None, :] > torch.tensor(sp.ts[s:s + 1024], device=dev)[:, None]
                sc, pc, rc = sc.masked_fill(fut, -1e9), pc.masked_fill(fut, -1e9), rc.masked_fill(fut, -1e9)
            knn.append(torch.topk(sc, 100, dim=1).indices.cpu().numpy())
            pop.append(torch.topk(pc, 100, dim=1).indices.cpu().numpy())
            rpop.append(torch.topk(rc, 100, dim=1).indices.cpu().numpy())
        ranks = {"pop": np.concatenate(pop), "recent_pop": np.concatenate(rpop), "content_knn": np.concatenate(knn)}
        return ranks, time.time() - t0

    test_ranks, t_test = rank_split(te)
    # validation rankings: weight selection of the generative + content-kNN fusion (hybrid.py)
    valid_ranks, _ = rank_split(SplitArrays(data, "valid", H, emb))
    for name, r in test_ranks.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        np.save(out / name / "rank_test_plain.npy", r)
        np.save(out / name / "rank_valid_plain.npy", valid_ranks[name])
        # infer time = the three rankers together (an upper bound for each; content-kNN dominates)
        save_json({"args": {"model": name}, "test_n": int(te.N), "n_params": 0, "train_time_sec": 0.0,
                   "infer_time_plain_sec": t_test,
                   "test_plain": compute_metrics(r, te.tgt, cat, cfg["metrics"]["ks"], te.ts)},
                  out / name / "metrics.json")


if __name__ == "__main__":
    main()
