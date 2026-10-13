"""Train/evaluate SASRec and TextSASRec baselines with the same temporal protocol and metrics."""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import SplitArrays
from .evaluate import compute_metrics
from .models.sasrec import SASRec, TextSASRec
from .sid import SIDCatalog
from .utils import ROOT, get_logger, load_config, load_pickle, processed_dir, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=["sasrec", "textsasrec"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(a.seed if a.seed is not None else cfg["seed"])
    out = ROOT / cfg["out_dir"] / a.dataset / a.exp
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"{a.dataset}/{a.exp}", out / "train.log")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = processed_dir(a.dataset)
    data = load_pickle(pdir / "data.pkl")
    emb = np.load(pdir / "item_emb.npy")
    K, L = cfg["tokenizer"]["codebook_size"], cfg["tokenizer"]["levels"]
    tok = load_pickle(pdir / f"sid_K{K}_L{L}" / "tokenizer.pkl")  # only for bucket definitions
    cat = SIDCatalog(tok["codes"], K, data["warm"])
    H = cfg["preprocess"]["max_hist_len"]
    tr, va, te = (SplitArrays(data, s, H) for s in ["train", "valid", "test"])
    sc = cfg["seq_baseline"]
    n_items = len(data["item_ids"])
    if a.model == "sasrec":
        model = SASRec(n_items, sc["d_model"], sc["num_layers"], sc["num_heads"], sc["dropout"], H).to(dev)
    else:
        model = TextSASRec(emb, sc["d_model"], sc["num_layers"], sc["num_heads"], sc["dropout"], H).to(dev)
    warm_idx = torch.tensor(np.where(data["warm"])[0], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=0.01)
    total = sc["max_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * s / total)))
    hist_all, tgt_all = torch.tensor(tr.hist, device=dev), torch.tensor(tr.tgt, device=dev)

    first_ts = torch.tensor(data["first_ts"], device=dev) if cfg.get("time_aware_catalog", True) else None

    @torch.no_grad()
    def rank(split, topn=100):
        model.eval()
        im = model.item_matrix()
        res = []
        for s in range(0, split.N, 512):
            hb = torch.tensor(split.hist[s:s + 512], device=dev)
            q = model(hb) if a.model == "sasrec" else model(hb, im)
            sc = q @ im.t()
            if first_ts is not None:  # time-aware catalogue: items released after the interaction are excluded
                tsb = torch.tensor(split.ts[s:s + 512], device=dev)
                sc = sc.masked_fill(first_ts[None, :] > tsb[:, None], -float("inf"))
            res.append(torch.topk(sc, topn, dim=1).indices.cpu().numpy())
        model.train()
        return np.concatenate(res)

    vsub = va.subset(np.random.RandomState(0).choice(va.N, min(3000, va.N), replace=False))
    best, bad, t0 = -1, 0, time.time()
    for step in range(total):
        idx = torch.randint(0, tr.N, (sc["batch_size"],), device=dev)
        hb, tg = hist_all[idx], tgt_all[idx]
        if a.model == "sasrec":
            q = model(hb)
            loss = F.cross_entropy(q @ model.item_matrix().t(), tg)
        else:  # sampled softmax over warm items (cold items are never seen during training)
            neg = warm_idx[torch.randint(0, len(warm_idx), (4096,), device=dev)]
            cand, inv = torch.unique(torch.cat([tg, neg]), return_inverse=True)
            im = F.normalize(model.adaptor(model.text[cand]), dim=-1)
            mat = torch.zeros(n_items, im.shape[1], device=dev)
            hist_items = torch.unique(hb[hb >= 0])
            mat[hist_items] = F.normalize(model.adaptor(model.text[hist_items]), dim=-1)
            mat[cand] = im
            q = model(hb, mat)
            loss = F.cross_entropy(q @ im.t(), inv[:len(tg)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log.info(f"step {step} loss {loss.item():.4f}")
        if (step + 1) % sc["eval_every"] == 0 or step == total - 1:
            m = compute_metrics(rank(vsub, 20), vsub.tgt, cat)["all"]["NDCG@10"]
            log.info(f"[valid@{step + 1}] NDCG@10 {m:.4f}")
            if m > best:
                best, bad = m, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                bad += 1
                if bad >= sc["patience"]:
                    break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    t1 = time.time()
    rt = rank(te)
    infer_time = time.time() - t1
    np.save(out / "rank_test_plain.npy", rt)
    np.save(out / "rank_valid_plain.npy", rank(va))
    res = {"args": vars(a), "train_time_sec": train_time, "infer_time_plain_sec": infer_time, "test_n": int(len(te.tgt)),
           "n_params": int(sum(p.numel() for p in model.parameters())),
           "test_plain": compute_metrics(rt, te.tgt, cat, cfg["metrics"]["ks"], te.ts)}
    save_json(res, out / "metrics.json")
    (out / "done").write_text("ok\n")
    log.info(f"[test] all {res['test_plain']['all']} | cold {res['test_plain']['cold']}")


if __name__ == "__main__":
    main()
