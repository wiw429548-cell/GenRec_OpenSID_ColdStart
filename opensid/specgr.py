"""SpecGR-lite baseline (Ding et al., 2024, "Inductive Generative Recommendation via Retrieval-based Speculation"):
a content drafter (TextSASRec) proposes candidates INCLUDING new items, the generative model (TIGER) verifies them by
the likelihood of their Semantic IDs; final score = log p_TIGER(SID) + gamma * drafter rank prior.
gamma is selected on the validation period by overall NDCG@10, the criterion of OpenSID's headline CCD."""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from .decode import item_logprob
from .evaluate import compute_metrics
from .train_gen import Trainer
from .utils import ROOT, save_json

GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def candidates(gen_rank, draft_rank, n_gen=50, n_draft=50):
    N = len(gen_rank)
    out = np.full((N, n_gen + n_draft), -1, dtype=np.int64)
    prior = np.full((N, n_gen + n_draft), -np.log(n_draft + 1), dtype=np.float32)
    for n in range(N):
        seen, k = set(), 0
        dpos = {it: p for p, it in enumerate(draft_rank[n, :n_draft]) if it >= 0}
        for it in list(draft_rank[n, :n_draft]) + list(gen_rank[n, :n_gen]):
            if it >= 0 and it not in seen:
                seen.add(it)
                out[n, k] = it
                prior[n, k] = -np.log(dpos.get(it, n_draft) + 1)
                k += 1
    return out, prior


def rank_with(cands, lp, prior, gamma, topn=100):
    sc = lp + gamma * prior
    order = np.argsort(-sc, axis=1, kind="stable")
    r = np.take_along_axis(cands, order, 1)
    r = np.where(np.isfinite(np.take_along_axis(sc, order, 1)), r, -1)
    return r[:, :topn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    args = SimpleNamespace(dataset=a.dataset, exp="specgr", config=a.config, emb="standard", cpm=0, backbone="t5",
                           K=None, L=None, rho=None, stage="A", init_from=None, trainable="all", ccd_from=None,
                           max_steps=0, seed=None, fit="train_items", no_delta_mask=False, cpm_keep_ext=False, liger=False)
    tr = Trainer(args)
    rd = ROOT / tr.cfg["out_dir"] / a.dataset
    tr.model.load_state_dict(torch.load(rd / "tiger" / "best.pt", map_location=tr.dev))
    tr.model.eval()
    ks = tr.cfg["metrics"]["ks"]
    res = {"args": vars(a)}
    lps = {}
    for split, name in [(tr.valid, "valid"), (tr.test, "test")]:
        gen = np.load(rd / "tiger" / f"rank_{name}_plain.npy")
        drf = np.load(rd / "textsasrec" / f"rank_{name}_plain.npy")
        cands, prior = candidates(gen, drf)
        lp = []
        for s in range(0, split.N, 64):
            hi = torch.tensor(split.hist[s:s + 64], device=tr.dev)
            lp.append(item_logprob(tr.model, tr.cat, hi, torch.tensor(cands[s:s + 64], device=tr.dev)).cpu().numpy())
        lps[name] = (cands, prior, np.concatenate(lp))
    grid = {g: compute_metrics(rank_with(*lps["valid"], g, topn=max(ks)), tr.valid.tgt, tr.cat, ks) for g in GAMMAS}
    g_best = max(grid, key=lambda g: grid[g]["all"]["NDCG@10"])
    r = rank_with(*lps["test"], g_best)
    out = rd / "specgr"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rank_test_plain.npy", r)
    res.update(selected_gamma=g_best, valid_grid={str(k): v for k, v in grid.items()},
               test_plain=compute_metrics(r, tr.test.tgt, tr.cat, ks, tr.test.ts))
    save_json(res, out / "metrics.json")
    (out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
