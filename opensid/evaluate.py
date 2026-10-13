"""Ranking metrics broken down by target bucket (warm / C1 / C2 / C3 / cold) and by test time window."""
import numpy as np

from .sid import BUCKETS


def compute_metrics(ranked, tgt, cat, ks=(5, 10, 20), ts=None, beam=None, n_windows=3):
    """ranked [N, >=max(ks)] item ids (-1 pad); tgt [N]; beam = (beam_keys [N,B], valid [N,B]) for Reach@Beam."""
    N = len(tgt)
    kmax = max(ks)
    r = ranked[:, :kmax]
    hit_pos = np.full(N, -1)
    eq = r == tgt[:, None]
    has = eq.any(1)
    hit_pos[has] = eq[has].argmax(1)
    bucket = cat.bucket[tgt]
    cold_item = ~cat.warm
    groups = {"all": np.ones(N, bool), "cold": bucket != "warm"}
    reach = None
    if beam is not None:
        bkeys, bvalid = beam
        gt = cat.prefix_keys[cat.L][tgt]
        reach = ((bkeys == gt[:, None]) & bvalid).any(1)
    for b in BUCKETS:
        groups[b] = bucket == b
    res = {}
    for g, m in groups.items():
        d = {"n": int(m.sum())}
        if m.sum() > 0:
            hp = hit_pos[m]
            for k in ks:
                hit = (hp >= 0) & (hp < k)
                d[f"Recall@{k}"] = float(hit.mean())
                d[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(np.maximum(hp, 0) + 2), 0.0).mean())
            if reach is not None:
                d["Reach@Beam"] = float(reach[m].mean())
        res[g] = d
    valid = r[:, :10] >= 0
    res["ColdHit@10"] = float((cold_item[np.where(valid, r[:, :10], 0)] & valid).sum(1).mean() / 10)
    if ts is not None and n_windows > 1:
        qs = np.quantile(ts, np.linspace(0, 1, n_windows + 1))
        win = {}
        for w in range(n_windows):
            m = (ts >= qs[w]) & ((ts < qs[w + 1]) if w < n_windows - 1 else (ts <= qs[w + 1]))
            if m.sum() == 0:
                continue
            hp = hit_pos[m]
            hit = (hp >= 0) & (hp < 10)
            win[f"w{w + 1}"] = {"n": int(m.sum()), "Recall@10": float(hit.mean()),
                                "NDCG@10": float(np.where(hit, 1.0 / np.log2(np.maximum(hp, 0) + 2), 0.0).mean()),
                                "cold_frac": float((bucket[m] != "warm").mean())}
        res["time_windows"] = win
    return res


def balanced_score(m):
    """selection criterion that weighs warm and cold targets equally (used for the 'balanced' CCD setting)."""
    w = m.get("warm", {}).get("NDCG@10", 0.0)
    c = m.get("cold", {}).get("NDCG@10", 0.0)
    return 0.5 * (w + c)


# ------------------------------------------------------------------ cold-share-aware operating-point selection
# Overall NDCG@10 is exactly (1-pi) * warm NDCG@10 + pi * cold NDCG@10 with pi = share of cold targets. Selecting on
# overall VALIDATION NDCG therefore implicitly assumes the deployment period has the validation cold share, which is
# much lower than in the (later) test period. The rules below only use validation data, except "oracle" (upper bound).
SELECTION_RULES = ["overall", "prior_last", "prior_trend", "warmguard", "balanced", "oracle"]
WARM_GUARD = 0.9


def _n10(m, grp):
    return (m.get(grp) or {}).get("NDCG@10", 0.0)


def cold_share(m):
    return (m.get("cold") or {}).get("n", 0) / max((m.get("all") or {}).get("n", 0), 1)


def prior_score(m, pi):
    return (1.0 - pi) * _n10(m, "warm") + pi * _n10(m, "cold")


def cold_share_forecast(m_valid):
    """forecast of the cold-target share of the next period from the validation metrics alone:
    valid = whole validation period; last = its most recent time window; trend = least-squares line through the
    per-window cold shares (window centres (w+0.5)/n), extrapolated to the centre of an equally long next period."""
    v = cold_share(m_valid)
    win = m_valid.get("time_windows") or {}
    fr = [win[k]["cold_frac"] for k in sorted(win, key=lambda s: int(s[1:]))]
    if len(fr) < 2:
        return {"valid": v, "last": v, "trend": v}
    x = (np.arange(len(fr)) + 0.5) / len(fr)
    slope, icpt = np.polyfit(x, np.asarray(fr, dtype=float), 1)
    return {"valid": v, "last": float(fr[-1]), "trend": float(np.clip(icpt + 1.5 * slope, 0.0, 0.95))}


def select_rules(grid_valid, ref_key, test_cold_share=None, guard=WARM_GUARD):
    """operating point (grid key) chosen by every rule in SELECTION_RULES, using VALIDATION metrics only.
    ref_key = the 'no cold-start intervention' point (plain decoding / generative-only), the warm reference of the
    warm-guarded rule (max valid cold NDCG@10 s.t. valid warm NDCG@10 >= guard * reference). Ties -> first key."""
    keys = list(grid_valid)
    if not keys:
        return {}, {}
    gv = grid_valid
    fc = cold_share_forecast(gv[keys[0]])
    sel = {"overall": max(keys, key=lambda k: _n10(gv[k], "all")),
           "prior_last": max(keys, key=lambda k: prior_score(gv[k], fc["last"])),
           "prior_trend": max(keys, key=lambda k: prior_score(gv[k], fc["trend"])),
           "balanced": max(keys, key=lambda k: balanced_score(gv[k]))}
    ref = _n10(gv[ref_key], "warm") if ref_key in gv else 0.0
    ok = [k for k in keys if _n10(gv[k], "warm") >= guard * ref] or ([ref_key] if ref_key in gv else keys)
    sel["warmguard"] = max(ok, key=lambda k: (_n10(gv[k], "cold"), _n10(gv[k], "all")))
    if test_cold_share is not None:
        fc["test"] = float(test_cold_share)
        sel["oracle"] = max(keys, key=lambda k: prior_score(gv[k], test_cold_share))
    return sel, fc
