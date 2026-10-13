"""Hybrid systems: weighted reciprocal-rank fusion (RRF) of a generative recommender and a content channel - the
common industrial fix of adding a separate content-based cold-start retriever.

* tiger_hybrid          : TIGER (plain) + TextSASRec                         (trained content channel)
* hybrid_knn_tiger      : TIGER (plain) + Content-kNN                        (training-free content channel)
* hybrid_knn_opensid    : OpenSID (CCD, headline operating point) + Content-kNN  (is OpenSID complementary to kNN?)

score(i) = w / (k + rank_gen(i)) + (1 - w) / (k + rank_content(i)), k = 60. The weight w is selected on the
VALIDATION period by overall NDCG@10 (the criterion of OpenSID's headline CCD). Validation and test metrics are stored
for every w (warm/cold Pareto curves) together with the cold-share-aware selection rules of evaluate.select_rules.
"""
import argparse

import numpy as np

from .data import SplitArrays
from .evaluate import cold_share, compute_metrics, select_rules
from .sid import SIDCatalog
from .utils import ROOT, get_logger, load_config, load_pickle, processed_dir, save_json

WEIGHTS = [round(0.1 * i, 1) for i in range(11)]   # w = 1: generative only, w = 0: content channel only
log = get_logger("hybrid")


def rrf(a, b, w=0.5, k=60, topn=100):
    """vectorised weighted RRF of two rankings a, b [N, n] (item ids, -1 = padding; no duplicates within a row)."""
    x = np.concatenate([a, b], 1)
    pos = np.concatenate([np.arange(a.shape[1]), np.arange(b.shape[1])])
    wt = np.concatenate([np.full(a.shape[1], w), np.full(b.shape[1], 1.0 - w)])
    sc = np.where(x >= 0, (wt / (k + pos + 1))[None, :], -np.inf)
    order = np.argsort(x, axis=1, kind="stable")
    xs, ss = np.take_along_axis(x, order, 1), np.take_along_axis(sc, order, 1)
    dup = (xs[:, 1:] == xs[:, :-1]) & (xs[:, 1:] >= 0)        # an item in both lists -> adjacent pair after sorting
    ss[:, :-1] += np.where(dup, ss[:, 1:], 0.0)
    ss[:, 1:][dup] = -np.inf
    top = np.argsort(-ss, axis=1, kind="stable")[:, :topn]
    out = np.take_along_axis(xs, top, 1)
    out[~np.isfinite(np.take_along_axis(ss, top, 1))] = -1
    if out.shape[1] < topn:
        out = np.concatenate([out, np.full((len(out), topn - out.shape[1]), -1, dtype=out.dtype)], 1)
    return out.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gen_exp", default="tiger")
    ap.add_argument("--gen_rank", default="plain", choices=["plain", "ccd_overall"],
                    help="which ranking of the generative model is fused (files rank_{valid,test}_<gen_rank>.npy)")
    ap.add_argument("--content_exp", default="textsasrec")
    ap.add_argument("--exp", default="tiger_hybrid")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    pdir = processed_dir(a.dataset)
    data = load_pickle(pdir / "data.pkl")
    K, L = cfg["tokenizer"]["codebook_size"], cfg["tokenizer"]["levels"]
    cat = SIDCatalog(load_pickle(pdir / f"sid_K{K}_L{L}" / "tokenizer.pkl")["codes"], K, data["warm"])
    H = cfg["preprocess"]["max_hist_len"]
    va, te = SplitArrays(data, "valid", H), SplitArrays(data, "test", H)
    rd = ROOT / cfg["out_dir"] / a.dataset
    ks = cfg["metrics"]["ks"]
    f = {(sp, ch): rd / exp / f"rank_{sp}_{rk}.npy" for sp in ("valid", "test")
         for ch, exp, rk in (("gen", a.gen_exp, a.gen_rank), ("content", a.content_exp, "plain"))}
    missing = [str(p) for p in f.values() if not p.exists()]
    if missing:  # never fall back to an untuned weight silently
        raise FileNotFoundError(f"missing rankings: {missing}")
    gv, cv = np.load(f["valid", "gen"]), np.load(f["valid", "content"])
    gt, ct = np.load(f["test", "gen"]), np.load(f["test", "content"])
    grid_valid, grid_test, ranks = {}, {}, {}
    for w in WEIGHTS:
        key = f"w={w}"
        grid_valid[key] = compute_metrics(rrf(gv, cv, w), va.tgt, cat, ks, va.ts)
        ranks[key] = rrf(gt, ct, w)
        grid_test[key] = compute_metrics(ranks[key], te.tgt, cat, ks, te.ts)
    rules, fc = select_rules(grid_valid, f"w={WEIGHTS[-1]}", cold_share(grid_test[f"w={WEIGHTS[-1]}"]))
    sel = rules["overall"]
    log.info(f"[{a.dataset}/{a.exp}] selected {sel} (rules {rules}); test {grid_test[sel]['all']}")
    out = rd / a.exp
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rank_test_plain.npy", ranks[sel])
    save_json({"args": vars(a), "selected_weight": float(sel.split("=")[1]), "selected_rules": rules,
               "cold_share_forecast": fc, "valid_grid": grid_valid, "test_grid": grid_test,
               "test_rules": {r: grid_test[k] for r, k in rules.items()}, "test_plain": grid_test[sel]},
              out / "metrics.json")
    (out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
