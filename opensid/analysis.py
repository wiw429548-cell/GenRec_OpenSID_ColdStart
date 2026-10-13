"""Paper analyses computed from the saved rankings (no retraining):

* significance : paired bootstrap (10k resamples) and paired t-test of per-sample NDCG@10 (all / cold targets),
                 OpenSID (full, seed 2026) vs every baseline; OpenSID+kNN fusion vs TIGER+kNN fusion and Content-kNN
* cold-sim     : cold-target Recall@10 by quintile of the target's max cosine similarity to any warm item
                 (are we only winning on new items that look like old ones?)
* case studies : users whose new (C2/C3) target is hit by OpenSID but missed by TIGER
* efficiency   : parameters, training time, inference time per 1k test users
"""
import json

import numpy as np

from .data import SplitArrays
from .sid import SIDCatalog
from .utils import ROOT, load_config, load_pickle, processed_dir

RES = ROOT / "results"
OURS = ("opensid", "rank_test_ccd_overall.npy")  # headline operating point (CCD tuned for overall NDCG)
OTHERS = [("TIGER", "tiger", "rank_test_plain.npy"), ("TIGER + Hybrid (RRF)", "tiger_hybrid", "rank_test_plain.npy"),
          ("TextSASRec", "textsasrec", "rank_test_plain.npy"), ("Content-kNN", "content_knn", "rank_test_plain.npy"),
          ("LIGER-lite", "liger", "rank_test_plain.npy"), ("SpecGR-lite", "specgr", "rank_test_plain.npy"),
          ("SASRec", "sasrec", "rank_test_plain.npy"), ("TIGER fit-all", "tiger_fitall", "rank_test_plain.npy"),
          ("TIGER + kNN (RRF)", "hybrid_knn_tiger", "rank_test_plain.npy")]
# is OpenSID complementary to a content retriever? (both generators fused with Content-kNN by the same tuned RRF)
FUSION = ("hybrid_knn_opensid", "rank_test_plain.npy")
FUSION_VS = [("TIGER + kNN (RRF)", "hybrid_knn_tiger"), ("Content-kNN", "content_knn")]


def ndcg_vec(ranked, tgt, k=10):
    r = ranked[:, :k]
    eq = r == tgt[:, None]
    pos = np.where(eq.any(1), eq.argmax(1), -1)
    return np.where(pos >= 0, 1.0 / np.log2(np.maximum(pos, 0) + 2), 0.0)


def paired_tests(a, b, n_boot=10000, seed=0):
    from scipy.stats import ttest_rel
    d = a - b
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n_boot, len(d))) if len(d) * n_boot <= 3e8 else None
    if idx is None:  # memory-friendly loop
        boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)])
    else:
        boots = d[idx].mean(1)
    p_boot = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    t = ttest_rel(a, b)
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_bootstrap": p_boot, "p_ttest": float(t.pvalue) if t.pvalue == t.pvalue else None}


def load_ds(ds, cfg):
    pdir = processed_dir(ds)
    data = load_pickle(pdir / "data.pkl")
    K, L = cfg["tokenizer"]["codebook_size"], cfg["tokenizer"]["levels"]
    cat = SIDCatalog(load_pickle(pdir / f"sid_K{K}_L{L}" / "tokenizer.pkl")["codes"], K, data["warm"])
    te = SplitArrays(data, "test", cfg["preprocess"]["max_hist_len"])
    return data, cat, te


def run(lines, datasets, res=None, config=None, load_fn=None):
    """appends Markdown to `lines`; returns {"sig": [...], "eff": [...]} rows for the LaTeX tables.
    load_fn(ds, exp) -> metrics or None is used to skip missing experiments."""
    global RES
    RES = res or RES
    cfg = load_config(config)
    ok = (lambda ds, exp: load_fn(ds, exp) is not None) if load_fn else (lambda ds, exp: True)
    sig_rows, eff_rows = [], []
    lines += ["", "## Statistical significance (OpenSID full, seed 2026, vs baselines; per-sample NDCG@10)", "",
              "| dataset | baseline | subset | Δ NDCG@10 | 95% CI | p (bootstrap) | p (paired t) |", "|---|---|---|---|---|---|---|"]
    sim_lines, case_lines, eff_lines = [], [], []

    def add_sig(ds, label, tex_label, nd_a, nd_b, cold):
        for sub, m in [("all", np.ones(len(nd_a), bool)), ("cold", cold)]:
            if m.sum() < 2:
                continue
            r = paired_tests(nd_a[m], nd_b[m])
            pt_ = "-" if r["p_ttest"] is None else f"{r['p_ttest']:.2e}"
            sig_rows.append(dict(ds=ds, label=tex_label, subset=sub, **r))
            lines.append(f"| {ds} | {label} | {sub} | {r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
                         f"{r['p_bootstrap']:.4f} | {pt_} |")

    for ds in datasets:
        pa = RES / ds / OURS[0] / OURS[1]
        if not pa.exists():
            pa = RES / ds / OURS[0] / "rank_test_ccd.npy"
        if not pa.exists() or not ok(ds, OURS[0]) or not (processed_dir(ds) / "data.pkl").exists():
            continue
        data, cat, te = load_ds(ds, cfg)
        ours = np.load(pa)
        nd_o = ndcg_vec(ours, te.tgt)
        cold = cat.bucket[te.tgt] != "warm"
        for label, exp, f in OTHERS:
            pb = RES / ds / exp / f
            if not pb.exists() or not ok(ds, exp):
                continue
            add_sig(ds, label, label, nd_o, ndcg_vec(np.load(pb), te.tgt), cold)
        pf = RES / ds / FUSION[0] / FUSION[1]
        if pf.exists() and ok(ds, FUSION[0]):
            nd_f = ndcg_vec(np.load(pf), te.tgt)
            for label, exp in FUSION_VS:
                pb = RES / ds / exp / "rank_test_plain.npy"
                if pb.exists() and ok(ds, exp):
                    add_sig(ds, f"OpenSID+kNN vs {label}", f"\\method{{}}+kNN vs {label}", nd_f,
                            ndcg_vec(np.load(pb), te.tgt), cold)
        # ---- cold-target similarity quintiles
        emb = np.load(processed_dir(ds) / "item_emb.npy")
        warm_idx = np.where(data["warm"])[0]
        ci = np.where(cold)[0]
        if len(ci) >= 10:
            T = emb[te.tgt[ci]]
            W = emb[warm_idx]
            sim = np.concatenate([(T[s:s + 2048] @ W.T).max(1) for s in range(0, len(T), 2048)])
            qs = np.quantile(sim, np.linspace(0, 1, 6))
            hdr = f"| {ds} | " + " | ".join(f"Q{i + 1} [{qs[i]:.2f},{qs[i + 1]:.2f}]" for i in range(5)) + " |"
            sim_lines += [hdr, "|---|" + "---|" * 5]
            for label, exp, f in [("OpenSID", OURS[0], pa.name)] + OTHERS[:4]:
                p = RES / ds / exp / f
                if not p.exists() or not ok(ds, exp):
                    continue
                r = np.load(p)[ci, :10]
                hit = (r == te.tgt[ci][:, None]).any(1)
                vals = []
                for i in range(5):
                    m = (sim >= qs[i]) & ((sim <= qs[i + 1]) if i == 4 else (sim < qs[i + 1]))
                    vals.append(f"{hit[m].mean():.4f}" if m.any() else "-")
                sim_lines.append(f"| {label} | " + " | ".join(vals) + " |")
            sim_lines.append("")
        # ---- case studies
        pt = RES / ds / "tiger" / "rank_test_plain.npy"
        if pt.exists() and ok(ds, "tiger"):
            tig = np.load(pt)
            hit_o = (ours[:, :10] == te.tgt[:, None]).any(1)
            hit_t = (tig[:, :10] == te.tgt[:, None]).any(1)
            b = cat.bucket[te.tgt]
            idx = np.where(hit_o & ~hit_t & np.isin(b, ["C2", "C3"]))[0][:3]
            txt = lambda i: data["item_text"][i][:90].replace("|", "/")
            for n in idx:
                case_lines += [f"**{ds} / test sample {n}** (target bucket {b[n]})", "",
                               "- history (most recent last): " + " → ".join(txt(i) for i in te.hist[n][te.hist[n] >= 0][-4:]),
                               f"- target (new item): {txt(te.tgt[n])}",
                               "- OpenSID top-3: " + " ; ".join(txt(i) for i in ours[n, :3] if i >= 0),
                               "- TIGER top-3: " + " ; ".join(txt(i) for i in tig[n, :3] if i >= 0), ""]
        # ---- efficiency
        for exp, label in [("tiger", "TIGER"), ("opensid", "OpenSID"), ("textsasrec", "TextSASRec"), ("sasrec", "SASRec"),
                           ("content_knn", "Content-kNN")]:
            p = RES / ds / exp / "metrics.json"
            if p.exists() and ok(ds, exp):
                m = json.load(open(p))
                n = m.get("test_n") or 1
                ip, ic = m.get("infer_time_plain_sec"), m.get("infer_time_ccd_sec")
                eff_rows.append(dict(ds=ds, label=label, params=f"{m.get('n_params', 0) / 1e6:.2f}M",
                                     train_min=f"{m.get('train_time_sec', 0) / 60:.0f}",
                                     plain=f"{1000 * ip / n:.2f}" if ip else "--",
                                     ccd=f"{1000 * ic / n:.2f}" if ic else "--"))
                eff_lines.append(f"| {ds} | {label} | {m.get('n_params', '-')} | {m.get('train_time_sec', 0) / 60:.1f} | "
                                 f"{1000 * (ip or float('nan')) / n:.3f} | {1000 * (ic or float('nan')) / n:.3f} |")
    lines += ["", "## Are gains limited to new items that resemble old ones? (cold-target Recall@10 by similarity quintile)", ""]
    lines += sim_lines
    lines += ["## Efficiency", "", "| dataset | model | #params | train (min) | infer s / 1k users (plain) | infer s / 1k users (CCD) |",
              "|---|---|---|---|---|---|"] + eff_lines
    lines += ["", "## Case studies (new item hit by OpenSID, missed by TIGER)", ""] + case_lines
    return {"sig": sig_rows, "eff": eff_rows}
