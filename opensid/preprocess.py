"""Build unified temporal-split datasets.

Output: data/processed/<ds>/data.pkl with
  item_ids, item_text, user_seqs (list of (items[int32], ts[int64])),
  t_valid, t_test, warm (bool[n_items]), first_ts (int64[n_items]),
  splits: {train|valid|test: np.ndarray[N, 2] of (user, pos)}  -- target = seq[pos], history = seq[:pos]

Temporal protocol (global time cut, NOT leave-one-out):
  train-period: ts <  t_valid ; valid-period: t_valid <= ts < t_test ; test-period: ts >= t_test
  An item is *warm* iff it occurs in any train-period interaction, otherwise *cold* (new item).
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from .download import RAW
from .utils import get_logger, load_config, processed_dir, save_pickle

log = get_logger("preprocess")


def _clean(s):
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(str(x) for x in s if x)
    return " ".join(str(s).split())


def amazon_item_text(m, max_chars):
    parts = []
    if m.get("title"):
        parts.append(f"Title: {_clean(m['title'])}")
    if m.get("store"):
        parts.append(f"Brand: {_clean(m['store'])}")
    if m.get("categories"):
        parts.append(f"Categories: {_clean(m['categories'])}")
    if m.get("price") not in (None, "None", ""):
        parts.append(f"Price: {m['price']}")
    if m.get("features"):
        parts.append(f"Features: {_clean(m['features'])}")
    if m.get("description"):
        parts.append(f"Description: {_clean(m['description'])}")
    return "; ".join(parts)[:max_chars]


def load_amazon(cat, max_chars):
    rows = []
    with open(RAW / "amazon" / f"{cat}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["user_id"], r["parent_asin"], int(float(r["timestamp"]))))
    need = {r[1] for r in rows}
    text = {}
    with open(RAW / "amazon" / f"meta_{cat}.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            pa = m.get("parent_asin")
            if pa in need:
                t = amazon_item_text(m, max_chars)
                if t:
                    text[pa] = t
    before = len(rows)
    rows = [r for r in rows if r[1] in text]
    log.info(f"[{cat}] {before} interactions, {len(rows)} with item text, {len(text)} items")
    by_user = defaultdict(list)
    for u, i, t in rows:
        by_user[u].append((t, i))
    return by_user, text


def load_mind(max_chars):
    text = {}
    for split in ["train", "dev"]:
        with open(RAW / "mind" / split / "news.tsv", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                nid, cat, sub, title, abst = p[:5]
                text[nid] = f"Category: {cat} / {sub}; Title: {_clean(title)}; Abstract: {_clean(abst)}"[:max_chars]
    hist = {}
    clicks = defaultdict(list)
    for split in ["train", "dev"]:
        with open(RAW / "mind" / split / "behaviors.tsv", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                _, uid, tstr, h, imps = p[:5]
                ts = int(datetime.strptime(tstr, "%m/%d/%Y %I:%M:%S %p").timestamp())
                hl = [x for x in h.split() if x in text]
                if uid not in hist or len(hl) > len(hist[uid]):
                    hist[uid] = hl
                for it in imps.split():
                    nid, _, lab = it.rpartition("-")
                    if lab == "1" and nid in text:
                        clicks[uid].append((ts, nid))
    by_user = {}
    for uid in set(hist) | set(clicks):
        # pre-period history has no timestamps -> ts=0 (strictly before every click)
        seq = [(0, n) for n in hist.get(uid, [])] + sorted(clicks.get(uid, []))
        by_user[uid] = seq
    log.info(f"[mind] users={len(by_user)} news={len(text)} clicks={sum(len(v) for v in clicks.values())}")
    return by_user, text


def build(ds, cfg):
    dcfg = cfg["datasets"][ds]
    pcfg = cfg["preprocess"]
    rng = np.random.RandomState(cfg["seed"])
    if dcfg["source"] == "amazon":
        by_user, text = load_amazon(dcfg["category"], pcfg["max_text_chars"])
    else:
        by_user, text = load_mind(pcfg["max_text_chars"])

    # stable chronological sort per user, drop users with < 2 interactions
    seqs = []
    for u in sorted(by_user):
        ev = sorted(by_user[u], key=lambda x: x[0])
        if len(ev) >= 2:
            seqs.append(ev)
    all_ts = np.array([t for ev in seqs for t, _ in ev if t > 0], dtype=np.int64)
    t_valid = int(np.quantile(all_ts, dcfg["valid_quantile"]))
    t_test = int(np.quantile(all_ts, dcfg["test_quantile"]))

    item_ids = sorted({i for ev in seqs for _, i in ev})
    iid = {x: k for k, x in enumerate(item_ids)}
    n_items = len(item_ids)
    first_ts = np.full(n_items, np.iinfo(np.int64).max, dtype=np.int64)
    warm = np.zeros(n_items, dtype=bool)
    user_seqs = []
    splits = {"train": [], "valid": [], "test": []}
    loo = dcfg.get("split") == "loo"  # leave-one-out protocol of prior work (TIGER, LC-Rec), for comparability
    for u, ev in enumerate(seqs):
        items = np.array([iid[i] for _, i in ev], dtype=np.int32)
        ts = np.array([t for t, _ in ev], dtype=np.int64)
        user_seqs.append((items, ts))
        np.minimum.at(first_ts, items, ts)
        if loo:
            if len(items) < 3:
                continue
            warm[items[:len(items) - 2]] = True
            for pos in range(1, len(items) - 2):
                splits["train"].append((u, pos))
            splits["valid"].append((u, len(items) - 2))
            splits["test"].append((u, len(items) - 1))
            continue
        warm[items[ts < t_valid]] = True
        for pos in range(1, len(items)):
            t = ts[pos]
            sp = "train" if t < t_valid else ("valid" if t < t_test else "test")
            splits[sp].append((u, pos))
    splits = {k: np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in splits.items()}

    caps = {"train": dcfg.get("max_train_samples"), "valid": pcfg["max_valid_samples"], "test": pcfg["max_test_samples"]}
    for k, cap in caps.items():
        if cap and len(splits[k]) > cap:
            sel = np.sort(rng.choice(len(splits[k]), cap, replace=False))
            splits[k] = splits[k][sel]

    data = dict(
        name=ds, item_ids=item_ids, item_text=[text[i] for i in item_ids], user_seqs=user_seqs,
        t_valid=t_valid, t_test=t_test, warm=warm, first_ts=first_ts, splits=splits,
    )
    stats = {k: int(len(v)) for k, v in splits.items()}
    for k in ["valid", "test"]:
        tgt = np.array([user_seqs[u][0][p] for u, p in splits[k]])
        stats[f"{k}_cold_frac"] = float((~warm[tgt]).mean()) if len(tgt) else 0.0
    stats.update(users=len(user_seqs), items=n_items, warm_items=int(warm.sum()))
    data["stats"] = stats
    log.info(f"[{ds}] {stats}")
    out = processed_dir(ds)
    save_pickle(data, out / "data.pkl")
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    build(a.dataset, load_config(a.config))


if __name__ == "__main__":
    main()
