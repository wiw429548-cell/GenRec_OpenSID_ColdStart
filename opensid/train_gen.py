"""Train + evaluate a Semantic-ID generative recommender (TIGER baseline or OpenSID variants).

Examples
  python -m opensid.train_gen --dataset amazon_industrial --exp tiger      --emb standard --cpm 0
  python -m opensid.train_gen --dataset amazon_industrial --exp opensid    --emb cate     --cpm 1
  python -m opensid.train_gen --dataset amazon_industrial --exp inc_opensid --stage inc --init_from results/.../opensid/best.pt ...

Every run writes results/<dataset>/<exp>/{metrics.json, best.pt, rank_*.npy, gt_token_ranks.npy, ...} and finally
the done-file `done`. CCD (module C) is inference-only, so every trained model is evaluated without CCD
(beta=0, quota=0) and over the whole CCD grid; (beta, quota) is selected on the validation period.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import SplitArrays, hist_tokens
from .decode import Ranker, beam_search, gt_token_ranks
from .evaluate import balanced_score, cold_share, compute_metrics, select_rules
from .models.generative import GenRec
from .sid import SIDCatalog
from .utils import ROOT, deep_update, get_logger, load_config, load_pickle, processed_dir, save_json, set_seed


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--emb", default="standard", choices=["standard", "cate"])
    ap.add_argument("--cpm", type=int, default=0)
    ap.add_argument("--backbone", default="t5", choices=["t5", "gpt", "qwen"])
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--rho", type=float, default=None, help="override cpm_rho_max")
    ap.add_argument("--stage", default="A", choices=["A", "inc"],
                    help="A: train on train-period; inc: continue/retrain with train+valid periods, eval on test")
    ap.add_argument("--init_from", default=None, help="checkpoint for stage inc (omit = retrain from scratch)")
    ap.add_argument("--trainable", default="all", choices=["all", "inc"],
                    help="inc = OpenSID-inc: only CATE projections W_l and Delta are updated")
    ap.add_argument("--ccd_from", default=None,
                    help="metrics.json whose selected CCD params are reused instead of tuning on valid "
                         "(stage inc; also the expensive LLM backbones, which reuse the T5 selection)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fit", default="train_items", choices=["train_items", "all_items"],
                    help="tokenizer variant: all_items = RQ-VAE refit on all items (new ones included)")
    ap.add_argument("--no_delta_mask", action="store_true", help="ablation: CATE without masking Delta of unseen tokens")
    ap.add_argument("--cpm_keep_ext", action="store_true", help="ablation: CPM without dropping the extension-step loss")
    ap.add_argument("--liger", action="store_true", help="baseline: LIGER-lite (generative + dense retrieval with cold-item supplement)")
    return ap.parse_args()


class Trainer:
    def __init__(self, a):
        self.a = a
        cfg = load_config(a.config)
        if a.backbone != "t5":
            cfg["gen_model"] = deep_update(cfg["gen_model"], cfg["backbones"][a.backbone])
        if a.max_steps is not None:
            cfg["gen_model"]["max_steps"] = a.max_steps
        if a.rho is not None:
            cfg["opensid"]["cpm_rho_max"] = a.rho
        self.cfg = cfg
        self.seed = a.seed if a.seed is not None else cfg["seed"]
        set_seed(self.seed)
        self.out = ROOT / cfg["out_dir"] / a.dataset / a.exp
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"{a.dataset}/{a.exp}", self.out / "train.log")
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pdir = processed_dir(a.dataset)
        self.data = load_pickle(pdir / "data.pkl")
        self.item_emb = np.load(pdir / "item_emb.npy")
        K = a.K or cfg["tokenizer"]["codebook_size"]
        L = a.L or cfg["tokenizer"]["levels"]
        from .rqvae import sid_dir
        self.tok = load_pickle(sid_dir(a.dataset, K, L, a.fit) / "tokenizer.pkl")
        warm = self.data["warm"].copy()
        if a.stage == "inc":  # after the valid period has been logged, its items become warm
            seqs = self.data["user_seqs"]
            for items, ts in seqs:
                warm[items[ts < self.data["t_test"]]] = True
        self.cat = SIDCatalog(self.tok["codes"], K, warm,
                              self.data["first_ts"] if cfg.get("time_aware_catalog", True) else None).to_torch(self.dev)
        H = cfg["preprocess"]["max_hist_len"]
        self.train = SplitArrays(self.data, "train", H)
        if a.stage == "inc":
            va = SplitArrays(self.data, "valid", H)
            for f in ["hist", "tgt", "ts"]:
                setattr(self.train, f, np.concatenate([getattr(self.train, f), getattr(va, f)]))
            self.train.N = len(self.train.tgt)
        self.valid = SplitArrays(self.data, "valid", H, self.item_emb)
        self.test = SplitArrays(self.data, "test", H, self.item_emb)
        self.ranker = Ranker(self.cat, self.item_emb, self.dev,
                             first_ts=self.data["first_ts"] if cfg.get("time_aware_catalog", True) else None)
        self.mcfg = cfg["gen_model"]
        self.compositional = bool(a.cpm) and a.emb == "cate"
        seen = np.ones_like(self.cat.seen_token) if a.no_delta_mask else self.cat.seen_token
        self.model = GenRec(self.mcfg, L, K, a.emb, self.tok["codebooks"], seen).to(self.dev)
        if a.liger:
            self.model.enable_dense(self.item_emb)
            self.model.to(self.dev)
        self.model._codes_t = torch.tensor(self.tok["codes"], device=self.dev)
        self.warm_idx = torch.tensor(np.where(warm)[0], device=self.dev)
        if a.init_from:
            sd = torch.load(a.init_from, map_location=self.dev)
            sd = {k: v for k, v in sd.items() if not k.endswith("emb.seen")}  # seen-mask follows the new warm set
            self.model.load_state_dict(sd, strict=False)
        # bf16 autocast only where the GPU supports it (e.g. not on V100); fp32 otherwise
        self.amp = self.mcfg.get("bf16", False) and self.dev.type == "cuda" and torch.cuda.is_bf16_supported()
        self.log.info(f"args={vars(a)} | train={self.train.N} valid={self.valid.N} test={self.test.N} "
                      f"| params={sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M | buckets(test)="
                      f"{dict(zip(*np.unique(self.cat.bucket[self.test.tgt], return_counts=True)))}")

    # ------------------------------------------------------------------ CPM masked prefixes
    def _sample_masked_prefixes(self, rho):
        """for each prefix length l in 1..L-1 mask a rho-fraction of the warm prefixes."""
        out = []
        for l in range(1, self.cat.L):
            keys = self.cat.seen_prefix[l]
            n = int(round(rho * len(keys)))
            sel = np.sort(np.random.choice(keys, n, replace=False)) if n > 0 else np.zeros(0, dtype=np.int64)
            out.append(torch.tensor(sel, device=self.dev, dtype=torch.long))
        return out

    def _anchor_from(self, tgt_codes, masked):
        B, L, K = tgt_codes.shape[0], self.cat.L, self.cat.K
        af = torch.full((B,), L, dtype=torch.long, device=self.dev)
        key = torch.zeros(B, dtype=torch.long, device=self.dev)
        for l in range(1, L):
            key = key * K + tgt_codes[:, l - 1]
            mk = masked[l - 1]
            if len(mk) == 0:
                continue
            pos = torch.searchsorted(mk, key).clamp(max=len(mk) - 1)
            hit = mk[pos] == key
            af = torch.where(hit & (af == L), torch.full_like(af, l), af)
        return af

    # ------------------------------------------------------------------ training
    def train_loop(self):
        a, m, ocfg = self.a, self.mcfg, self.cfg["opensid"]
        if a.trainable == "inc":
            for n, p in self.model.named_parameters():
                p.requires_grad = n.startswith("emb.proj") or n.startswith("emb.delta")
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=m["lr"], weight_decay=m["weight_decay"])
        total = max(0, m["max_steps"])
        warm = min(m["warmup_steps"], total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
        bs = m["batch_size"]
        steps_per_epoch = max(1, self.train.N // bs)
        do_eval = a.stage == "A"
        vsub = self.valid.subset(np.random.RandomState(0).choice(self.valid.N, min(3000, self.valid.N), replace=False))
        best, bad, best_step = -1.0, 0, 0
        masked = None
        t0 = time.time()
        hist_all = torch.tensor(self.train.hist, device=self.dev)
        tgt_all = torch.tensor(self.train.tgt, device=self.dev)
        self.model.train()
        for step in range(total):
            if self.compositional and step % steps_per_epoch == 0:
                rho = ocfg["cpm_rho_max"] * min(1.0, step / max(1, ocfg["cpm_warmup_frac"] * total))
                masked = self._sample_masked_prefixes(rho)
            idx = torch.randint(0, self.train.N, (bs,), device=self.dev)
            hi, tg = hist_all[idx], tgt_all[idx]
            ids, mask = hist_tokens(hi, self.model._codes_t, self.cat.L, self.cat.K)
            tc = self.model._codes_t[tg]
            af = self._anchor_from(tc, masked) if self.compositional else None
            drop = (af < self.cat.L) if (af is not None and not a.cpm_keep_ext) else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                dn = self.warm_idx[torch.randint(0, len(self.warm_idx), (2048,), device=self.dev)] if a.liger else None
                loss = self.model.loss(ids, mask, tc, af, drop, dense_tgt=tg if a.liger else None, dense_neg=dn)
                loss = loss + ocfg["cate_delta_l2"] * self.model.emb.reg_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 200 == 0:
                self.log.info(f"step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                              f"{(time.time() - t0) / 60:.1f}min")
            if do_eval and ((step + 1) % m["eval_every"] == 0 or step == total - 1):
                met = self.evaluate(vsub, beta=0.0, quota=0)["metrics"]
                score = met["all"]["NDCG@10"]
                self.log.info(f"[valid@{step + 1}] NDCG@10 {score:.4f} cold-R@10 {met['cold'].get('Recall@10', 0):.4f}")
                if score > best:
                    best, bad, best_step = score, 0, step + 1
                    torch.save(self.model.state_dict(), self.out / "best.pt")
                else:
                    bad += 1
                    if bad >= m["patience"]:
                        self.log.info(f"early stop at {step + 1} (best {best:.4f} @ {best_step})")
                        break
                self.model.train()
        if not do_eval:
            torch.save(self.model.state_dict(), self.out / "best.pt")
        self.train_time = time.time() - t0
        self.model.load_state_dict(torch.load(self.out / "best.pt", map_location=self.dev))
        self.model.eval()

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def beams(self, split, quota):
        m = self.mcfg
        bsz = m["eval_batch_size"] if self.model.kind == "t5" else max(4, m["eval_batch_size"] // 8)
        codes, lps = [], []
        self.model.eval()
        for s in range(0, split.N, bsz):
            hi = torch.tensor(split.hist[s:s + bsz], device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                c, lp = beam_search(self.model, self.cat, hi, m["beam_size"], quota, self.compositional,
                                    ts=split.ts[s:s + bsz])
            if c.shape[1] < m["beam_size"]:
                pad = m["beam_size"] - c.shape[1]
                c = torch.cat([c, c.new_zeros(c.shape[0], pad, c.shape[2])], 1)
                lp = torch.cat([lp, lp.new_full((lp.shape[0], pad), -float("inf"))], 1)
            codes.append(c.cpu().numpy())
            lps.append(lp.float().cpu().numpy())
        return np.concatenate(codes), np.concatenate(lps)

    @torch.no_grad()
    def liger_rank(self, split, ranked_gen, k_gen=20, n_cold=20, topn=100):
        """LIGER inference: K generative candidates + N cold-start items from dense retrieval, re-ranked densely"""
        R = self.model.dense_items()
        cold = torch.tensor(~self.cat.warm, device=self.dev)
        first_ts = torch.tensor(self.data["first_ts"], device=self.dev)
        out = np.full((split.N, topn), -1, dtype=np.int64)
        for s in range(0, split.N, 256):
            hi = torch.tensor(split.hist[s:s + 256], device=self.dev)
            ids, mask = hist_tokens(hi, self.model._codes_t, self.cat.L, self.cat.K)
            q = self.model.dense_query(self.model.encode(ids, mask))
            sc = q @ R.t()
            if self.cfg.get("time_aware_catalog", True):
                tsb = torch.tensor(split.ts[s:s + 256], device=self.dev)
                sc = sc.masked_fill(first_ts[None, :] > tsb[:, None], -float("inf"))
            cold_top = torch.topk(sc.masked_fill(~cold[None, :], -float("inf")), n_cold, dim=1).indices
            dense_top = torch.topk(sc, topn, dim=1).indices.cpu().numpy()
            gen = torch.tensor(ranked_gen[s:s + 256, :k_gen], device=self.dev)
            for b in range(len(hi)):
                cand = torch.unique(torch.cat([gen[b][gen[b] >= 0], cold_top[b]]))
                cand = cand[torch.isfinite(sc[b, cand])]
                order = cand[torch.argsort(-sc[b, cand])].cpu().numpy().tolist()
                rest = [x for x in dense_top[b] if x not in set(order)]
                lst = (order + rest)[:topn]
                out[s + b, :len(lst)] = lst
        return out

    def evaluate(self, split, beta, quota, beams=None, return_rank=False):
        codes, lps = beams if beams is not None else self.beams(split, quota)
        topn = 100 if return_rank else max(self.cfg["metrics"]["ks"])
        ranked, bkeys, bvalid = self.ranker.rank(codes, lps, split.profile, beta=beta, topn=max(topn, 20), ts=split.ts)
        if self.a.liger:
            ranked = self.liger_rank(split, ranked, topn=topn)
        met = compute_metrics(ranked, split.tgt, self.cat, self.cfg["metrics"]["ks"], split.ts, (bkeys, bvalid))
        return {"metrics": met, "ranked": ranked if return_rank else None}

    def run(self):
        a, ocfg = self.a, self.cfg["opensid"]
        self.train_loop()
        res = {"args": _portable_args(a), "train_time_sec": getattr(self, "train_time", 0.0),
               "n_params": int(sum(p.numel() for p in self.model.parameters())),
               "test_n": int(self.test.N),
               "collision_rate": self.tok["collision_rate"]}
        t0 = time.time()
        test_plain = self.evaluate(self.test, 0.0, 0, return_rank=True)
        res["infer_time_plain_sec"] = time.time() - t0
        res["test_plain"] = test_plain["metrics"]
        np.save(self.out / "rank_test_plain.npy", test_plain["ranked"])
        # validation ranking without CCD: used by the rank-fusion jobs (hybrid.py) to select the fusion weight
        np.save(self.out / "rank_valid_plain.npy", self.evaluate(self.valid, 0.0, 0, return_rank=True)["ranked"])
        self.log.info(f"[test plain] all {res['test_plain']['all']} | cold {res['test_plain']['cold']}")

        # ----- CCD: tune (beta, quota) on valid, apply to test (test beams cached per quota)
        tbeams, tbeam_time = {}, {}

        def test_beams(q):
            if q not in tbeams:
                t0 = time.time()
                tbeams[q] = self.beams(self.test, q)
                tbeam_time[q] = time.time() - t0
            return tbeams[q]

        grid_valid, grid_test, vbeams = {}, {}, {}
        if a.stage == "A" and not a.liger and not a.ccd_from:  # LIGER has its own dense re-ranking
            for q in ocfg["ccd_quota_grid"]:
                vb = vbeams[int(q)] = self.beams(self.valid, q)
                tb = test_beams(q)
                for b in ocfg["ccd_beta_grid"]:
                    key = f"beta={float(b)},quota={int(q)}"
                    grid_valid[key] = self.evaluate(self.valid, b, q, vb)["metrics"]
                    grid_test[key] = self.evaluate(self.test, b, q, tb)["metrics"]
            sel_overall = max(grid_valid, key=lambda k: grid_valid[k]["all"]["NDCG@10"])
            sel_bal = max(grid_valid, key=lambda k: balanced_score(grid_valid[k]))
        else:
            prev = json.load(open(a.ccd_from)) if (a.ccd_from and not a.liger) else {}
            sel_overall = prev.get("ccd_selected_overall", "beta=0.0,quota=0")
            sel_bal = prev.get("ccd_selected_balanced", sel_overall)
            for key in {sel_overall, sel_bal}:
                b, q = [float(x.split("=")[1]) for x in key.split(",")]
                grid_test[key] = self.evaluate(self.test, b, int(q), test_beams(int(q)))["metrics"]
        res["ccd_selected_overall"], res["ccd_selected_balanced"] = sel_overall, sel_bal
        res["test_ccd_overall"] = grid_test[sel_overall]
        res["test_ccd_balanced"] = grid_test[sel_bal]
        res["ccd_grid_valid"], res["ccd_grid_test"] = grid_valid, grid_test
        if grid_valid:  # cold-share-aware operating points (validation-only rules + labelled test-share oracle)
            rules, fc = select_rules(grid_valid, "beta=0.0,quota=0", cold_share(res["test_plain"]))
            res["ccd_selected_rules"], res["cold_share_forecast"] = rules, fc
            res["test_ccd_rules"] = {r: grid_test[k] for r, k in rules.items()}
            self.log.info(f"cold share forecast {fc} -> selected {rules}")
        b, q = [float(x.split("=")[1]) for x in sel_bal.split(",")]
        t0 = time.time()
        rk = self.evaluate(self.test, b, int(q), test_beams(int(q)), return_rank=True)["ranked"]
        res["infer_time_ccd_sec"] = time.time() - t0 + tbeam_time.get(int(q), 0.0)
        np.save(self.out / "rank_test_ccd.npy", rk)
        bo, qo = [float(x.split("=")[1]) for x in sel_overall.split(",")]  # headline operating point
        rko = rk if sel_overall == sel_bal else self.evaluate(self.test, bo, int(qo), test_beams(int(qo)), return_rank=True)["ranked"]
        np.save(self.out / "rank_test_ccd_overall.npy", rko)
        if int(qo) in vbeams:  # validation ranking at the headline point (weight selection of the kNN fusion)
            np.save(self.out / "rank_valid_ccd_overall.npy",
                    self.evaluate(self.valid, bo, int(qo), vbeams[int(qo)], return_rank=True)["ranked"])
        self.log.info(f"[test CCD {sel_bal}] all {res['test_ccd_balanced']['all']} | cold {res['test_ccd_balanced']['cold']}")

        # ----- reachability analysis: teacher-forced GT token ranks per level (0 = top-1), per bucket
        ranks = []
        for s in range(0, self.test.N, 512):
            hi = torch.tensor(self.test.hist[s:s + 512], device=self.dev)
            tg = torch.tensor(self.test.tgt[s:s + 512], device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                ranks.append(gt_token_ranks(self.model, self.cat, hi, tg, self.compositional).cpu().numpy())
        ranks = np.concatenate(ranks)
        np.save(self.out / "gt_token_ranks.npy", ranks)
        bucket = self.cat.bucket[self.test.tgt]
        res["gt_token_rank"] = {
            bk: {f"level{l + 1}": {"median": float(np.median(ranks[bucket == bk, l])),
                                   "top10_frac": float((ranks[bucket == bk, l] < 10).mean())}
                 for l in range(self.cat.L)} for bk in np.unique(bucket)}
        np.save(self.out / "test_bucket.npy", bucket.astype(str))
        save_json(res, self.out / "metrics.json")
        (self.out / "done").write_text("ok\n")
        self.log.info("done")


def _portable_args(a):
    """command-line arguments with file paths made relative to the repository root (no machine-specific paths)."""
    out = vars(a).copy()
    for k in ("init_from", "ccd_from", "config"):
        if out.get(k):
            try:
                out[k] = str(Path(out[k]).resolve().relative_to(ROOT)).replace("\\", "/")
            except ValueError:
                out[k] = Path(out[k]).name
    return out


def main():
    a = parse()
    Trainer(a).run()


if __name__ == "__main__":
    main()
