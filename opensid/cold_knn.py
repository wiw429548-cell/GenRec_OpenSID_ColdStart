"""Training-free new-item controls: content-kNN restricted to, or biased towards, new items.

Question: is CCD's gain on new items a property of the generator, or would any content retriever that is told
"prefer items released after the training cut-off" get the same?  Two training-free controls, same protocol
(time-aware catalogue, history items excluded) as content_knn in simple_baselines.py:

knn_new      : rank only new (non-warm) items by cosine(history profile, item embedding)
knn_newprior : cosine + lam * 1[item is new]; lam selected on the validation period by overall NDCG@10, the
               criterion of OpenSID's headline CCD (lam = 0 is plain content-kNN, lam >= 2 equals knn_new)
"""
import argparse
import time

import numpy as np
import torch

from .data import SplitArrays
from .evaluate import compute_metrics
from .sid import SIDCatalog
from .utils import ROOT, load_config, load_pickle, processed_dir, save_json

LAMS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2.0]


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
    ks = cfg["metrics"]["ks"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.tensor(emb, device=dev)
    FT = torch.tensor(data["first_ts"], device=dev)
    NEW = torch.tensor(~data["warm"], device=dev).float()

    def rank(sp, lam):
        out = []
        for s in range(0, sp.N, 1024):
            q = torch.tensor(sp.profile[s:s + 1024], device=dev)
            sc = q @ E.t() + lam * NEW[None, :]
            hb = torch.tensor(sp.hist[s:s + 1024], device=dev)
            r_, c_ = (hb >= 0).nonzero(as_tuple=True)
            sc[r_, hb[r_, c_]] = -1e9
            fut = FT[None, :] > torch.tensor(sp.ts[s:s + 1024], device=dev)[:, None]
            sc = sc.masked_fill(fut, -1e9)
            out.append(torch.topk(sc, 100, dim=1).indices.cpu().numpy())
        return np.concatenate(out)

    va = SplitArrays(data, "valid", H, emb)
    te = SplitArrays(data, "test", H, emb)
    vgrid, tgrid = {}, {}
    for lam in LAMS:
        vgrid[f"lam={lam}"] = compute_metrics(rank(va, lam), va.tgt, cat, ks)
        tgrid[f"lam={lam}"] = compute_metrics(rank(te, lam), te.tgt, cat, ks, te.ts)
    best = max(vgrid, key=lambda k: vgrid[k]["all"]["NDCG@10"])
    out = ROOT / cfg["out_dir"] / a.dataset
    for name, key in [("knn_new", f"lam={LAMS[-1]}"), ("knn_newprior", best)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = rank(te, float(key.split("=")[1]))
        np.save(d / "rank_test_plain.npy", r)
        np.save(d / "rank_valid_plain.npy", rank(va, float(key.split("=")[1])))  # for hybrid.py weight selection
        save_json({"args": {"model": name, "dataset": a.dataset}, "selected": key, "test_n": int(te.N),
                   "n_params": 0, "train_time_sec": 0.0, "infer_time_plain_sec": time.time() - t0,
                   "valid_grid": vgrid, "test_grid": tgrid,
                   "test_plain": compute_metrics(r, te.tgt, cat, ks, te.ts)}, d / "metrics.json")
        (d / "done").write_text("ok\n")
        m = tgrid[key]
        print(f"{a.dataset:22s} {name:13s} {key:9s} all N@10={m['all']['NDCG@10']:.4f} "
              f"warm={m['warm']['NDCG@10']:.4f} cold N@10={m['cold']['NDCG@10']:.4f} "
              f"cold R@10={m['cold']['Recall@10']:.4f}")


if __name__ == "__main__":
    main()
