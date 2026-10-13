"""Figures and result tables of the README, computed from results/<dataset>/<exp>/metrics.json.

    python scripts/make_figures.py          # -> assets/*.png, assets/*.pdf, assets/results.md

Every number comes from the stored metrics (loaded through opensid.aggregate, mean over the seeds that exist).
The token-rank figure additionally reads results/<dataset>/<exp>/gt_token_ranks.npy and test_bucket.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opensid import aggregate as A  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)
DS = ["amazon_industrial", "amazon_musical", "amazon_videogames", "mind"]
NAME = {"amazon_industrial": "Industrial", "amazon_musical": "Musical", "amazon_videogames": "Games", "mind": "MIND"}

# categorical colours in a fixed order (checked for colour-vision deficiency); text stays in neutral ink
C = {"tiger": "#8C8C8C", "tigerccd": "#9DC3E6", "cateccd": "#3D85C6", "opensid": "#C8553D", "knn": "#2A9D8F",
     "textsasrec": "#8E7CC3", "liger": "#E9A03B", "warm": "#C9D3DE", "C1": "#F2C14E", "C2": "#E07A5F", "C3": "#81171B"}
INK, MUTED, FAINT = "#1F1F1F", "#5F5F5F", "#9A9A9A"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.edgecolor": "#6B6B6B",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.08, "figure.facecolor": "white",
    "lines.solid_capstyle": "round",
})


def header(fig, title, subtitle=None, y=1.0):
    """bold figure title and a muted one-line subtitle, left-aligned above the axes."""
    fig.text(0.0, y + 0.075, title, ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.0, y + 0.02, subtitle, ha="left", va="bottom", fontsize=8.8, color=MUTED)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def jload(ds, exp):
    p = RES / ds / exp / "metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def mean(ds, exp, key, bucket, met):
    return A.stats(A.seeds_of(ds, exp), key, bucket, met)[0]


def cold_frac(ds):
    return json.load(open(ROOT / "data" / "processed" / ds / "stats.json"))["test_cold_frac"]


# ---------------------------------------------------------------------------------------------------- figures
def fig_teaser():
    """(a) test-target composition; (b) cold Recall@10 of TIGER, OpenSID and the new-item kNN control."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.08})
    names = [NAME[d] for d in DS]
    y = np.arange(len(DS))[::-1]

    ax = axes[0]
    comp = np.array([[A.load(d, "tiger")["test_plain"][k]["n"] / A.load(d, "tiger")["test_plain"]["all"]["n"]
                      for k in ["warm", "C1", "C2", "C3"]] for d in DS])
    left = np.zeros(len(DS))
    labels = {"warm": "warm (seen in training)", "C1": "C1: seen path", "C2": "C2: known tokens, new path",
              "C3": "C3: unseen token"}
    for j, k in enumerate(["warm", "C1", "C2", "C3"]):
        ax.barh(y, comp[:, j], left=left, color=C[k], height=0.6, label=labels[k], edgecolor="white", lw=1.2)
        left += comp[:, j]
    for yy, row in zip(y, comp):
        ax.text(0.985, yy, f"{100 * (1 - row[0]):.0f}% new", va="center", ha="right", fontsize=9,
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("(a) Composition of the test targets")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)

    ax = axes[1]
    sysm = [("TIGER", "tiger", "test_plain", C["tiger"], "X", 70),
            ("OpenSID (cold-favouring CCD)", "opensid", "test_ccd_balanced", C["opensid"], "D", 52),
            ("Content-kNN over new items", "knn_new", "test_plain", C["knn"], "o", 64)]
    floor = 0.006
    vals = {e: np.array([mean(d, e, k, "cold", "Recall@10") for d in DS]) * 100 for _, e, k, *_ in sysm}
    for yy, i in zip(y, range(len(DS))):
        lo = max(min(vals[e][i] for _, e, *_ in sysm), floor)
        hi = max(vals[e][i] for _, e, *_ in sysm)
        ax.plot([lo, hi], [yy, yy], color="#DDDDDD", lw=3.2, zorder=1)
    for lab, e, k, col, mk, s in sysm:
        v = vals[e]
        ax.scatter(np.maximum(v, floor), y, s=s, color=col, marker=mk, zorder=3, edgecolor="white", lw=0.8, label=lab)
        for yy, vv in zip(y, v):
            txt = "0" if vv < 5e-3 else (f"{vv:.2f}" if vv < 1 else f"{vv:.1f}")
            off = -0.24 if e == "opensid" else 0.24
            ax.text(max(vv, floor), yy + off, txt, ha="center", va="center", fontsize=7.2, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(4e-3, 30)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%"])
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(DS) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Cold-item Recall@10 (log scale; 0 drawn at the left edge)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1)
    header(fig, "New items dominate the test period, and a SID generator barely reaches them",
           "Global temporal split; TIGER-style generator vs. the repaired decoder and a training-free content control")
    save(fig, "teaser")


def fig_token_ranks():
    """teacher-forced rank of the ground-truth token, levels 1 and 2, for C2 targets (seed 2026)."""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), gridspec_kw={"wspace": 0.22})
    for ax, (d, lvl) in zip(axes, [("amazon_industrial", 0), ("amazon_industrial", 1), ("mind", 0), ("mind", 1)]):
        med = {}
        for lab, e, col, ls, lw, al in [("TIGER", "tiger", C["tiger"], "-", 3.2, 0.45),
                                        ("+ CATE", "opensid_nocpm", C["cateccd"], "--", 1.5, 1.0),
                                        ("+ CATE + CPM (OpenSID)", "opensid", C["opensid"], "-", 1.5, 1.0)]:
            r = np.load(RES / d / e / "gt_token_ranks.npy")[:, lvl] + 1          # 1 = top-1
            b = np.load(RES / d / e / "test_bucket.npy", allow_pickle=True)
            x = np.sort(r[b == "C2"])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=lw, ls=ls, alpha=al,
                    label=f"{lab}, new-path (C2) targets")
            if e == "tiger":
                w = np.sort(r[b == "warm"])
                ax.plot(w, np.arange(1, len(w) + 1) / len(w), color="#3A3A3A", lw=1.3, ls=":",
                        label="TIGER, warm targets")
                med = {"warm": np.median(r[b == "warm"]), "new": np.median(r[b == "C2"])}
        ax.axhline(0.5, color="#CFCFCF", lw=0.8, zorder=0)
        ax.text(1.15, 0.94, f"median rank\nwarm {med['warm']:.0f}  ·  new {med['new']:.0f}", ha="left", va="top",
                fontsize=7.8, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", lw=0.6))
        ax.set_xscale("log")
        ax.set_xlim(1, 256)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title(f"{NAME[d]} · level {lvl + 1}")
        ax.set_xlabel("rank of the ground-truth token")
    axes[0].set_ylabel("share of targets (CDF)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    header(fig, "New items are lost at the second code level",
           "Teacher-forced rank among the codes the trie allows (seed 2026). CATE and CPM leave the new-path curves "
           "where they are.", y=0.98)
    save(fig, "token_ranks")


def _grid(m):
    out = []
    for k, v in (m.get("ccd_grid_test") or {}).items():
        b, q = [float(x.split("=")[1]) for x in k.split(",")]
        out.append((b, q, v["warm"]["NDCG@10"], v["cold"]["NDCG@10"]))
    return np.array(out)


def _frontier(pts):
    pts = sorted(pts, key=lambda p: (-p[0], -p[1]))
    out, best = [], -1
    for x, y in pts:
        if y > best:
            out.append((x, y))
            best = y
    return np.array(out)


def fig_tradeoff():
    """warm vs cold NDCG@10 of every CCD setting (seed 2026) against the kNN-prior lambda grid."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.25), gridspec_kw={"wspace": 0.26})
    for ax, d in zip(axes, DS):
        for lab, e, col, mk in [("TIGER + CCD", "tiger", C["tigerccd"], "o"),
                                ("TIGER + CATE + CCD", "opensid_nocpm", C["cateccd"], "s"),
                                ("OpenSID", "opensid", C["opensid"], "D")]:
            m = A.load(d, e)
            g = _grid(m)
            ax.scatter(g[:, 2] * 100, g[:, 3] * 100, s=10, color=col, alpha=0.35, lw=0, marker=mk)
            fr = _frontier(list(zip(g[:, 2] * 100, g[:, 3] * 100)))
            ax.plot(fr[:, 0], fr[:, 1], color=col, lw=1.8, label=f"{lab}: CCD frontier")
            s = m["ccd_grid_test"][m["ccd_selected_overall"]]
            ax.scatter([s["warm"]["NDCG@10"] * 100], [s["cold"]["NDCG@10"] * 100], s=62, facecolor="white",
                       edgecolor=col, lw=1.8, zorder=5, marker=mk)
        k = jload(d, "knn_newprior")
        pts = np.array([(v["warm"]["NDCG@10"] * 100, v["cold"]["NDCG@10"] * 100) for v in k["test_grid"].values()])
        fr = _frontier(list(map(tuple, pts)))
        ax.plot(fr[:, 0], fr[:, 1], color=C["knn"], lw=1.8, ls=(0, (4, 2)), label=r"Content-kNN + prior: $\lambda$ frontier")
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=C["knn"], lw=0, alpha=0.75)
        for lab, e, col, mk in [("TIGER", "tiger", C["tiger"], "X"), ("TextSASRec", "textsasrec", C["textsasrec"], "^"),
                                ("LIGER-lite", "liger", C["liger"], "v")]:
            ax.scatter([mean(d, e, "test_plain", "warm", "NDCG@10") * 100],
                       [mean(d, e, "test_plain", "cold", "NDCG@10") * 100],
                       s=58, color=col, marker=mk, zorder=6, edgecolor="white", lw=0.7, label=lab)
        ax.set_title(f"{NAME[d]}  ·  {100 * cold_frac(d):.0f}% new")
        ax.set_xlabel("warm NDCG@10 (×100)")
        x1, y1 = ax.get_xlim()[1], ax.get_ylim()[1]
        ax.set_xlim(-0.04 * x1, x1)
        ax.set_ylim(-0.04 * y1, y1)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel("cold NDCG@10 (×100)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#333", markersize=7,
                    markeredgewidth=1.5))
    l.append("setting selected on validation")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    header(fig, "Product catalogs favour content retrieval, news favours the repaired generator",
           "Every CCD (β, q) setting (seed 2026) against the λ grid of a content-kNN with a new-item prior; "
           "baselines are means over three seeds", y=0.98)
    save(fig, "tradeoff")


def fig_beam_ceiling():
    """share of new items still in the beam (Reach@Beam) vs in the top-20 of content-kNN over new items."""
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(DS))
    w = 0.2
    series = [
        ("TIGER, beam without quota", [A.g(A.load(d, "tiger"), "ccd_grid_test", "beta=0.0,quota=0", "cold", "Reach@Beam") for d in DS], C["tiger"]),
        ("TIGER, beam with quota q = 20", [A.g(A.load(d, "tiger"), "ccd_grid_test", "beta=64.0,quota=20", "cold", "Reach@Beam") for d in DS], C["tigerccd"]),
        ("OpenSID, beam with quota q = 20", [A.g(A.load(d, "opensid"), "ccd_grid_test", "beta=64.0,quota=20", "cold", "Reach@Beam") for d in DS], C["opensid"]),
        ("Content-kNN over new items, top-20", [jload(d, "knn_new")["test_plain"]["cold"]["Recall@20"] for d in DS], C["knn"]),
    ]
    for i, (lab, v, col) in enumerate(series):
        v = np.array(v) * 100
        xx = x + (i - 1.5) * w
        ax.bar(xx, v, width=w * 0.9, color=col, label=lab, edgecolor="white", lw=0.6)
        for a, b in zip(xx, v):
            ax.text(a, b + 0.18, f"{b:.1f}", ha="center", va="bottom", fontsize=7.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[d] for d in DS], fontsize=9.5, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("% of new-item targets")
    ax.set_ylim(0, 12.6)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    header(fig, "The beam is CCD's ceiling",
           "Share of new-item targets whose SID is still among the 50 finished beams, vs. the top-20 of a content "
           "retriever (seed 2026)", y=0.93)
    save(fig, "beam_ceiling")


def fig_ccd_grid():
    """warm / cold test NDCG@10 of OpenSID over the CCD (beta, quota) grid (seed 2026); selected setting outlined."""
    betas, quotas = [0, 2, 4, 6, 8, 10, 12, 16, 32, 64], [0, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.1), gridspec_kw={"wspace": 0.3})
    for ax, (d, grp) in zip(axes, [("amazon_industrial", "warm"), ("amazon_industrial", "cold"),
                                   ("mind", "warm"), ("mind", "cold")]):
        m = A.load(d, "opensid")
        gt = m["ccd_grid_test"]
        M = np.array([[gt[f"beta={float(b)},quota={q}"][grp]["NDCG@10"] * 100 for b in betas] for q in quotas])
        cmap = "Blues" if grp == "warm" else "Oranges"
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=0)
        vmax = M.max() if M.max() > 0 else 1
        for qi in range(len(quotas)):
            for bi in range(len(betas)):
                v = M[qi, bi]
                lab = f"{v:.1f}" if v >= 1 else f"{v:.2f}".lstrip("0")
                ax.text(bi, qi, lab, ha="center", va="center", fontsize=6.4,
                        color="white" if v > 0.55 * vmax else INK)
        sb, sq = [float(x.split("=")[1]) for x in m["ccd_selected_overall"].split(",")]
        ax.add_patch(Rectangle((betas.index(int(sb)) - 0.5, quotas.index(int(sq)) - 0.5), 1, 1, fill=False,
                               ec=INK, lw=1.8, zorder=5))
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()
        cb.outline.set_visible(False)
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels(betas, fontsize=7.5)
        ax.set_yticks(range(len(quotas)))
        ax.set_yticklabels(quotas if ax is axes[0] else [])
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(r"rescoring weight $\beta$")
        ax.set_title(f"{NAME[d]} · {grp} NDCG@10 (×100)", fontsize=9.5)
    axes[0].set_ylabel("beam quota $q$")
    header(fig, "Neither part of CCD works alone",
           "OpenSID over the CCD grid (seed 2026). Bottom row: rescoring only; left columns: quota only. "
           "Outlined: setting selected on validation.", y=0.98)
    save(fig, "ccd_grid")


def fig_cold_share():
    """cold share of the validation period, its last third and the three test windows; trend forecast."""
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cols = {"amazon_industrial": "#3D85C6", "amazon_musical": "#E9A03B", "amazon_videogames": "#8E7CC3", "mind": "#2A9D8F"}
    mk = {"amazon_industrial": "o", "amazon_musical": "s", "amazon_videogames": "D", "mind": "^"}
    for d in DS:
        tw = A.load(d, "tiger")["test_plain"]["time_windows"]
        fc = A.load(d, "opensid")["cold_share_forecast"]
        ys = [fc["valid"], fc["last"]] + [tw[f"w{i}"]["cold_frac"] for i in (1, 2, 3)]
        ax.plot(range(5), ys, marker=mk[d], ms=5.5, lw=2.0, color=cols[d], label=NAME[d],
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([1, 2.5], [fc["last"], fc["trend"]], ls=":", lw=1.3, color=cols[d])
        ax.scatter([2.5], [fc["trend"]], marker="x", s=34, color=cols[d], lw=1.6)
        ax.text(4.12, ys[-1], NAME[d], va="center", ha="left", fontsize=8.5, color=INK)
    ax.axvspan(-0.3, 1.5, color="#F4F4F4", zorder=0)
    ax.text(0.6, 0.97, "validation", ha="center", va="top", color=MUTED, transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.97, "test", ha="center", va="top", color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(5))
    ax.set_xticklabels(["whole\nperiod", "last\nthird", "window 1", "window 2", "window 3"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.3, 4.75)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of new-item targets")
    ax.grid(axis="x", visible=False)
    header(fig, "The share of new items grows faster than validation suggests",
           "Dotted lines and crosses: trend forecast for the test period, computed from validation only", y=0.93)
    save(fig, "cold_share")


# ---------------------------------------------------------------------------------------------------- tables
MAIN = [("Content-kNN", "content_knn", "test_plain"), ("Content-kNN + new-item prior", "knn_newprior", "test_plain"),
        ("SASRec", "sasrec", "test_plain"), ("TextSASRec", "textsasrec", "test_plain"),
        ("TIGER", "tiger", "test_plain"), ("LIGER-lite", "liger", "test_plain"),
        ("TIGER + CCD", "tiger", "test_ccd_overall"), ("TIGER + CATE + CCD", "opensid_nocpm", "test_ccd_overall"),
        ("OpenSID (CATE + CPM + CCD)", "opensid", "test_ccd_overall")]


def results_md():
    """README result table: overall NDCG@10 and cold Recall@10 (x100), best per column in bold."""
    cols = [(d, b, m) for d in DS for b, m in [("all", "NDCG@10"), ("cold", "Recall@10")]]
    vals = {(lab, c): mean(c[0], e, k, c[1], c[2]) * 100 for lab, e, k in MAIN for c in cols}
    best = {c: max(round(vals[(lab, c)], 2) for lab, _, _ in MAIN) for c in cols}
    lines = ["| Method | " + " | ".join(f"{NAME[d]} N@10 | {NAME[d]} cold R@10" for d in DS) + " |",
             "|---|" + "---:|" * len(cols)]
    for lab, _, _ in MAIN:
        cells = []
        for c in cols:
            v = round(vals[(lab, c)], 2)
            cells.append(f"**{v:.2f}**" if v == best[c] else f"{v:.2f}")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT / "results.md")


if __name__ == "__main__":
    fig_teaser()
    fig_token_ranks()
    fig_tradeoff()
    fig_beam_ceiling()
    fig_ccd_grid()
    fig_cold_share()
    results_md()
