"""Collect results/<dataset>/<exp>/metrics.json into a readable summary.

    python -m opensid.aggregate              # -> results/summary.md, results/main_table.csv

Columns are `bucket/metric` on the test period; `±` is the standard deviation over seeds 2026-2028. Figures are made
by scripts/make_figures.py from the same files. If the saved rankings (rank_*.npy) are present, the summary also
contains paired significance tests, similarity quintiles, efficiency and case studies (opensid/analysis.py).
"""
import argparse
import json

import numpy as np

from .evaluate import SELECTION_RULES, WARM_GUARD, cold_share, select_rules
from .experiments import BACKBONE_DATASETS, INC_DATASETS, LOO_DATASET, MAIN_DATASETS, SEEDS, SENS_DATASET
from .utils import ROOT, load_config

RES = ROOT / "results"


def _overall(m):
    return m.get("all", {}).get("NDCG@10", 0.0)


def load(ds, exp):
    """metrics.json of one run, extended with the operating point of every selection rule (None if missing)."""
    p = RES / ds / exp / "metrics.json"
    if not p.exists():
        return None
    m = json.load(open(p, encoding="utf-8"))
    gv, gt = m.get("ccd_grid_valid") or {}, m.get("ccd_grid_test") or {}
    # single-component CCD ablations, selected on valid with the same (overall NDCG@10) criterion as the headline
    for name, cond in [("ccd_beta_only", lambda k: k.endswith("quota=0")),
                       ("ccd_quota_only", lambda k: k.startswith("beta=0.0,"))]:
        keys = [k for k in gv if cond(k)]
        if keys:
            m[name] = gt[max(keys, key=lambda k: _overall(gv[k]))]
    # operating points of every selection rule (evaluate.select_rules), recomputed from the stored grids:
    # CCD (beta, quota) grid of generative models, fusion-weight grid of the hybrids
    ref = "beta=0.0,quota=0"
    if not (gv and gt) and m.get("valid_grid") and m.get("test_grid"):
        gv, gt = m["valid_grid"], m["test_grid"]
        ref = max(gv, key=lambda k: float(k.split("=")[1]))  # w = 1: generative channel only
    if gv and gt and ref in gt:
        rules, fc = select_rules(gv, ref, cold_share(gt[ref]))
        m["op_selected"], m["op_forecast"] = rules, fc
        for r, k in rules.items():
            if k in gt:
                m[f"op_{r}"] = gt[k]
        m["op_grid_test"], m["op_ref"] = gt, ref
    return m


def g(m, *path, default=np.nan):
    for k in path:
        if m is None or not isinstance(m, dict) or k not in m:
            return default
        m = m[k]
    return m


def gb(m, key, bucket, metric):
    """metric of a bucket; nan if the bucket has no test targets (e.g. C3 with K=256)."""
    if g(m, key, bucket, "n", default=0) in (0, np.nan):
        return np.nan
    return g(m, key, bucket, metric)


# (label, experiment, metrics key); None = separator.
# OpenSID's headline operating point: CCD (beta, quota) selected on valid by overall NDCG@10 ("test_ccd_overall");
# "balanced" (mean of warm & cold NDCG@10 on valid) is the cold-favouring operating point.
MAIN_ROWS = [
    ("Pop", "pop", "test_plain"),
    ("Recent-Pop", "recent_pop", "test_plain"),
    ("Content-kNN", "content_knn", "test_plain"),
    ("Content-kNN, new items only", "knn_new", "test_plain"),
    ("Content-kNN + new-item prior", "knn_newprior", "test_plain"),
    ("SASRec", "sasrec", "test_plain"),
    ("TextSASRec", "textsasrec", "test_plain"),
    None,
    ("TIGER", "tiger", "test_plain"),
    ("TIGER + TextSASRec (RRF)", "tiger_hybrid", "test_plain"),
    ("TIGER + Content-kNN (RRF)", "hybrid_knn_tiger", "test_plain"),
    ("TIGER + kNN-prior (RRF)", "hybrid_knnprior_tiger", "test_plain"),
    ("LIGER-lite", "liger", "test_plain"),
    ("SpecGR-lite", "specgr", "test_plain"),
    None,
    ("TIGER + CCD", "tiger", "test_ccd_overall"),
    ("TIGER + CATE (no CCD)", "opensid_nocpm", "test_plain"),
    ("TIGER + CATE + CCD", "opensid_nocpm", "test_ccd_overall"),
    ("TIGER + CATE + CPM (no CCD)", "opensid", "test_plain"),
    ("**OpenSID (CATE + CPM + CCD)**", "opensid", "test_ccd_overall"),
    ("OpenSID, cold-favouring CCD (balanced)", "opensid", "test_ccd_balanced"),
    ("OpenSID + Content-kNN (RRF)", "hybrid_knn_opensid", "test_plain"),
    ("OpenSID + kNN-prior (RRF)", "hybrid_knnprior_opensid", "test_plain"),
]
ABL_ROWS = [  # design alternatives (seed 2026)
    ("TIGER, tokenizer refit on all items", "tiger_fitall", "test_plain"),
    ("TIGER, refit + CCD", "tiger_fitall", "test_ccd_overall"),
    ("OpenSID, refit on all items", "opensid_fitall", "test_ccd_overall"),
    ("OpenSID w/o Delta masking of unseen tokens", "opensid_nomask", "test_ccd_overall"),
    ("OpenSID, CPM keeps extension-step loss", "opensid_keepext", "test_ccd_overall"),
    ("OpenSID, CCD beta only (quota=0)", "opensid", "ccd_beta_only"),
    ("OpenSID, CCD quota only (beta=0)", "opensid", "ccd_quota_only"),
    ("**OpenSID (full)**", "opensid", "test_ccd_overall"),
]
COLS = [("all", "Recall@10"), ("all", "NDCG@10"), ("warm", "NDCG@10"), ("cold", "Recall@10"), ("cold", "NDCG@10"),
        ("C1", "Recall@10"), ("C2", "Recall@10"), ("C3", "Recall@10"), ("cold", "Reach@Beam")]
MULTI_SEED = ("tiger", "opensid", "opensid_nocpm", "sasrec", "textsasrec", "liger", "hybrid_knn_tiger",
              "hybrid_knn_opensid", "hybrid_knnprior_tiger", "hybrid_knnprior_opensid")


def seeds_of(ds, exp):
    """metrics of every seed of an experiment (seed 2026 = no suffix, other seeds = _s<seed>)."""
    ms = [load(ds, exp)]
    if exp in MULTI_SEED:
        ms += [load(ds, f"{exp}_s{s}") for s in SEEDS[1:]]
    return [m for m in ms if m is not None]


def stats(ms, key, a, b):
    vals = [gb(m, key, a, b) if a in ("C1", "C2", "C3") else g(m, key, a, b) for m in ms]
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else np.nan), len(vals)


def fmt(mean, std=np.nan):
    if mean != mean:
        return "-"
    return f"{mean:.4f}" if std != std else f"{mean:.4f}±{std:.4f}"


# ------------------------------------------------------------------ Markdown tables
def main_table(lines, csv, datasets=MAIN_DATASETS, rows=MAIN_ROWS):
    for ds in datasets:
        lines.append(f"\n### {ds}\n")
        m0 = load(ds, "tiger") or load(ds, "content_knn")
        if m0:
            cnt = {b: g(m0, "test_plain", b, "n", default=0) for b in ["warm", "C1", "C2", "C3"]}
            lines.append(f"test targets per bucket: {cnt}\n")
        hdr = ["Method"] + [f"{a}/{b}" for a, b in COLS] + ["ColdHit@10", "#seeds"]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for r in rows:
            if r is None:
                continue
            label, exp, key = r
            ms = seeds_of(ds, exp)
            st = [stats(ms, key, a, b) for a, b in COLS]
            ch = [g(m, key, "ColdHit@10") for m in ms]
            ch = [x for x in ch if x == x]
            row = [label] + [fmt(mu, sd) for mu, sd, _ in st] + [fmt(np.mean(ch)) if ch else "-", str(len(ms))]
            lines.append("| " + " | ".join(row) + " |")
            if ms:
                csv.append([ds, label.strip("*")] + [mu for mu, _, _ in st])


def simple_table(lines, title, rows, cols=COLS):
    lines.append(f"\n## {title}\n")
    hdr = ["Dataset", "Setting"] + [f"{a}/{b}" for a, b in cols] + ["train_time(min)", "CCD selected"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for ds, label, exp, key in rows:
        m = load(ds, exp)
        if m is None:
            lines.append("| " + " | ".join([ds, label] + ["-"] * (len(cols) + 2)) + " |")
            continue
        vals = [fmt(stats([m], key, a, b)[0]) for a, b in cols]
        sel = m.get("ccd_selected_overall" if key == "test_ccd_overall" else "ccd_selected_balanced", "") if "ccd" in key else ""
        lines.append("| " + " | ".join([ds, label] + vals + [f"{g(m, 'train_time_sec', default=0) / 60:.1f}", sel]) + " |")


# ------------------------------------------------------------------ operating points and warm/cold frontier
RULE_MD = {"overall": "overall valid N@10 (headline)", "prior_last": "prior: cold share of last valid window",
           "prior_trend": "prior: forecast (trend) cold share", "warmguard": f"warm-guarded (valid warm >= {WARM_GUARD:g} x plain)",
           "balanced": "balanced (mean of warm & cold)", "oracle": "oracle: true test cold share (reference only)"}
OP_METHODS = [("TIGER + CCD", "tiger"), ("OpenSID", "opensid"), ("TIGER + kNN", "hybrid_knn_tiger"),
              ("OpenSID + kNN", "hybrid_knn_opensid")]
GRID_METHODS = [("TIGER + CCD", "tiger"), ("+ CATE + CCD", "opensid_nocpm"), ("OpenSID (CCD grid)", "opensid"),
                ("TIGER + TextSASRec (RRF)", "tiger_hybrid"), ("TIGER + kNN (RRF)", "hybrid_knn_tiger"),
                ("OpenSID + kNN (RRF)", "hybrid_knn_opensid")]
POINT_METHODS = [("Content-kNN", "content_knn"), ("TextSASRec", "textsasrec"), ("LIGER-lite", "liger"),
                 ("SpecGR-lite", "specgr")]
WARM_LEVELS = (0.9, 0.7)


def _test_points(m):
    """all test (warm N@10, cold N@10) points of a model: its whole grid if it has one, else its single result."""
    grid = (m or {}).get("op_grid_test") or ({"x": m["test_plain"]} if m and "test_plain" in m else {})
    return [(g(v, "warm", "NDCG@10"), g(v, "cold", "NDCG@10")) for v in grid.values()]


def cold_at_warm(ms, level, ref_warm):
    """mean over seeds of the best TEST cold N@10 among points with test warm N@10 >= level * ref_warm
    (frontier summary, not a deployable selection); nan = no result, -1 = no point reaches the warm level."""
    vals = []
    for m in ms:
        ok = [c for w, c in _test_points(m) if w >= level * ref_warm and c == c]
        vals.append(max(ok) if ok else -1.0)
    if not vals:
        return np.nan
    return -1.0 if min(vals) < 0 else float(np.mean(vals))


def tiger_warm(ds):
    w = [g(m, "test_plain", "warm", "NDCG@10") for m in seeds_of(ds, "tiger")]
    w = [x for x in w if x == x]
    return float(np.mean(w)) if w else np.nan


def forecast_of(ds):
    for exp in ("opensid", "tiger", "opensid_nocpm", "hybrid_knn_tiger"):
        m = load(ds, exp)
        if m and m.get("op_forecast"):
            return m["op_forecast"]
    return None


def operating_points(lines, datasets):
    lines.append("\n## Cold-share-aware operating points (CCD / fusion weight selected on VALID by each rule)\n")
    lines.append("Overall NDCG@10 = (1-pi) warm + pi cold, pi = share of cold targets. Selecting on overall valid N@10 "
                 "assumes the valid cold share, which is far below the test one. The prior rules plug in a forecast "
                 "pi_hat computed from the validation period only; the oracle uses the true test share. Seed 2026.\n")
    lines.append("| dataset | valid pi | last-window pi_hat | trend pi_hat | test pi |")
    lines.append("|---|---|---|---|---|")
    for ds in datasets:
        fc = forecast_of(ds)
        lines.append(f"| {ds} | " + " | ".join(f"{fc[k]:.3f}" if fc and k in fc else "-"
                                                for k in ("valid", "last", "trend", "test")) + " |")
    lines.append("")
    hdr = ["dataset", "method", "rule", "selected", "all N@10", "warm N@10", "cold N@10", "cold R@10"]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for ds in datasets:
        for lab, exp in OP_METHODS:
            m = load(ds, exp)
            if not m or "op_selected" not in m:
                continue
            for r in SELECTION_RULES:
                v = m.get(f"op_{r}")
                if v is None:
                    continue
                lines.append(f"| {ds} | {lab} | {RULE_MD[r]} | {m['op_selected'][r]} | " + " | ".join(
                    fmt(g(v, a, b)) for a, b in [("all", "NDCG@10"), ("warm", "NDCG@10"), ("cold", "NDCG@10"),
                                                 ("cold", "Recall@10")]) + " |")
    lines.append("\n## Warm/cold frontier on TEST (best cold N@10 at a given warm level)\n")
    lines.append("Warm level = fraction of plain TIGER's test warm N@10. `x` = no grid point keeps that warm level.\n")
    hdr = ["dataset", "TIGER warm N@10"] + [f"{lab} @{lv:g}" for lab, _ in GRID_METHODS + POINT_METHODS for lv in WARM_LEVELS]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for ds in datasets:
        tw = tiger_warm(ds)
        cells = []
        for _, exp in GRID_METHODS + POINT_METHODS:
            for lv in WARM_LEVELS:
                v = cold_at_warm(seeds_of(ds, exp), lv, tw) if tw == tw else np.nan
                cells.append("x" if v == -1.0 else fmt(v))
        lines.append(f"| {ds} | {fmt(tw)} | " + " | ".join(cells) + " |")


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg_path = ap.parse_args().config
    cfg = load_config(cfg_path)
    RES = ROOT / cfg["out_dir"]
    RES.mkdir(parents=True, exist_ok=True)
    lines = ["# OpenSID: experiment summary (generated by `python -m opensid.aggregate`)", "",
             "Columns are `bucket/metric` on the TEST period. `±` = std over 3 seeds. `-` = not available. "
             "CCD (beta, quota) is selected on the validation period: headline = overall NDCG@10 (`ccd_overall`), "
             "cold-favouring = mean of warm & cold NDCG@10 (`ccd_balanced`).", "",
             "## Main results and module ablation"]
    csv = []
    main_table(lines, csv)
    lines.append("\n## Design alternatives (seed 2026)\n")
    main_table(lines, [], MAIN_DATASETS, ABL_ROWS)
    lines.append("\n## Leave-one-out protocol of prior work (few new items)\n")
    main_table(lines, [], [LOO_DATASET], MAIN_ROWS)
    rows = []
    for ds in BACKBONE_DATASETS:
        for bb in ["", "_gpt", "_qwen"]:
            name = {"": "T5 (TIGER)", "_gpt": "GPT (from scratch)", "_qwen": "Qwen2.5-0.5B"}[bb]
            rows += [(ds, f"{name} baseline", f"tiger{bb}", "test_plain"),
                     (ds, f"{name} + OpenSID", f"opensid{bb}", "test_ccd_overall")]
    simple_table(lines, "Backbone generality (seed 2026)", rows)
    lines.append("\n## Performance over the three test time windows (NDCG@10 / cold fraction)\n")
    for ds in MAIN_DATASETS:
        for exp, key in [("tiger", "test_plain"), ("textsasrec", "test_plain"), ("opensid", "test_ccd_overall")]:
            w = g(load(ds, exp), key, "time_windows", default={})
            if w:
                lines.append(f"- {ds} / {exp}: " + ", ".join(
                    f"{k}: {v['NDCG@10']:.4f} (cold {v['cold_frac']:.2f})" for k, v in w.items()))
    rows = []
    for ds in INC_DATASETS:
        for exp, lab in [("inc_tiger_noupdate", "TIGER no update"), ("inc_tiger_finetune", "TIGER full fine-tune"),
                         ("inc_tiger_retrain", "TIGER retrain"), ("inc_opensid_noupdate", "OpenSID no update"),
                         ("inc_opensid_inc", "OpenSID-inc (W, Delta only)"),
                         ("inc_opensid_finetune", "OpenSID full fine-tune"), ("inc_opensid_retrain", "OpenSID retrain")]:
            rows.append((ds, lab, exp, "test_ccd_overall"))
    simple_table(lines, "Incremental update (train + valid periods, evaluated on test; seed 2026)", rows)
    rows = [(SENS_DATASET, f"rho_max={r}", e, "test_ccd_overall") for r, e in
            [(0.1, "opensid_rho0.1"), (0.2, "opensid"), (0.4, "opensid_rho0.4")]]
    rows += [(SENS_DATASET, f"K={K} {n}", f"{e}{s}", key) for K, s in [(128, "_K128"), (256, ""), (512, "_K512")]
             for n, e, key in [("TIGER", "tiger", "test_plain"), ("OpenSID", "opensid", "test_ccd_overall")]]
    simple_table(lines, "Sensitivity: CPM rho_max and codebook size K (seed 2026)", rows)
    operating_points(lines, MAIN_DATASETS + [LOO_DATASET])
    lines.append("\n## Reachability: teacher-forced rank of the ground-truth token (0 = top-1; median / top-10 share)\n")
    for ds in MAIN_DATASETS:
        for exp in ["tiger", "opensid"]:
            m = load(ds, exp)
            if m and "gt_token_rank" in m:
                lines.append(f"- {ds}/{exp}: " + "; ".join(
                    f"{b}: " + ", ".join(f"{lv} med={v['median']:.0f} top10={v['top10_frac']:.2f}" for lv, v in d.items())
                    for b, d in m["gt_token_rank"].items()))
    try:  # needs the saved rankings (rank_*.npy) and data/processed; skipped silently if they are missing
        from . import analysis
        analysis.run(lines, MAIN_DATASETS + [LOO_DATASET], RES, cfg_path, load)
    except Exception as e:  # analyses must never block the summary
        lines.append(f"\n(ranking-based analyses skipped: {type(e).__name__}: {e})")
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with open(RES / "main_table.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("dataset,method," + ",".join(f"{a}/{b}" for a, b in COLS) + "\n")
        for r in csv:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {RES / 'summary.md'} and {RES / 'main_table.csv'}")


if __name__ == "__main__":
    main()
