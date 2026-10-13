"""Constrained beam search over the catalog trie + Catalog-aware Calibrated Decoding (CCD, OpenSID module C).

CCD = (i) beam quota: `quota` beam slots per step are reserved for prefixes NOT supported by train-period SIDs
      (ii) calibrated rescoring of finished beams:
           score(s) = log p(s|H) + beta * (1 - support(s)) * cos(profile(H), e_s)
Compositional inference (used when the model was trained with CPM): a decoder input token c_j is fed with its
content-anchored embedding iff the prefix (c_1..c_j) is unseen in train-period SIDs.
"""
import numpy as np
import torch

from .data import hist_tokens
from .sid import BOS, N_SPECIAL


def _select(scores, unsupported, B, quota):
    """scores [N, M] (−inf = invalid), unsupported [N, M] bool -> indices [N, B]"""
    quota = min(quota, B)
    k_main = min(B - quota, scores.shape[1])
    top_main = scores.topk(k_main, dim=1).indices if k_main > 0 else scores.new_zeros((scores.shape[0], 0), dtype=torch.long)
    if quota <= 0:
        return top_main
    # reserved slots: best unsupported candidates first, then (if too few exist) the next best overall
    rest = scores.clone()
    rest.scatter_(1, top_main, -float("inf"))
    prio = torch.where(torch.isfinite(rest) & unsupported, rest + 1e6, rest)
    k_q = min(quota, scores.shape[1] - k_main)
    top_q = prio.topk(k_q, dim=1).indices
    return torch.cat([top_main, top_q], 1)


@torch.no_grad()
def beam_search(model, cat, hist_items, beam, quota=0, compositional=False, ts=None):
    """hist_items LongTensor [N, H] -> codes [N, beam, L] (int64), logp [N, beam] (−inf for empty beams)"""
    L, K = cat.L, cat.K
    dev = hist_items.device
    N = hist_items.shape[0]
    ids, mask = hist_tokens(hist_items, model._codes_t, L, K)
    ctx = model.encode(ids, mask)
    # level 0
    dec = torch.full((N, 1), BOS, dtype=torch.long, device=dev)
    h = model.decode(ctx, dec)[:, -1]
    lg = model.level_logits(h, 0)
    allowed = cat.allowed_mask(0, torch.zeros(N, dtype=torch.long, device=dev))
    tsv = None if ts is None else torch.as_tensor(ts, device=dev)
    if tsv is not None:  # prune prefixes whose items are all released after the interaction
        allowed = allowed & cat.released(1, torch.arange(K, device=dev)[None].expand(N, K), tsv[:, None])
    lp = torch.log_softmax(lg.masked_fill(~allowed, -1e9), -1).masked_fill(~allowed, -float("inf"))
    keys = torch.arange(K, device=dev)[None].expand(N, K)
    unsup = ~cat.is_seen_prefix(1, keys.reshape(-1)).reshape(N, K)
    sel = _select(lp, unsup, beam, quota)
    Bc = sel.shape[1]
    codes = sel[..., None]  # [N, Bc, 1]
    score = lp.gather(1, sel)
    ctx_r = model.ctx_repeat(ctx, Bc)
    for l in range(1, L):
        flat = codes.reshape(N * Bc, l)
        tok = N_SPECIAL + torch.arange(l, device=dev) * K + flat
        dec = torch.cat([torch.full((N * Bc, 1), BOS, dtype=torch.long, device=dev), tok], 1)
        anchored = None
        if compositional:
            anchored = torch.zeros_like(dec, dtype=torch.bool)
            key = torch.zeros(N * Bc, dtype=torch.long, device=dev)
            for j in range(l):
                key = key * K + flat[:, j]
                anchored[:, j + 1] = ~cat.is_seen_prefix(j + 1, key)
        h = model.decode(ctx_r, dec, anchored)[:, -1]
        lg = model.level_logits(h, l)
        pkey = torch.zeros(N * Bc, dtype=torch.long, device=dev)
        for j in range(l):
            pkey = pkey * K + flat[:, j]
        allowed = cat.allowed_mask(l, pkey)
        if tsv is not None:
            ck = pkey[:, None] * K + torch.arange(K, device=dev)[None]
            allowed = allowed & cat.released(l + 1, ck, tsv.repeat_interleave(Bc)[:, None])
        lp = torch.log_softmax(lg.masked_fill(~allowed, -1e9), -1).masked_fill(~allowed, -float("inf"))
        cand = (score.reshape(N * Bc, 1) + lp).reshape(N, Bc * K)
        ckey = (pkey[:, None] * K + torch.arange(K, device=dev)[None]).reshape(N, Bc * K)
        unsup = ~cat.is_seen_prefix(l + 1, ckey.reshape(-1)).reshape(N, Bc * K)
        sel = _select(cand, unsup, beam, quota)
        parent, code = sel // K, sel % K
        codes = torch.cat([codes.gather(1, parent[..., None].expand(-1, -1, l)), code[..., None]], 2)
        score = cand.gather(1, sel)
        if sel.shape[1] != Bc:
            Bc = sel.shape[1]
            ctx_r = model.ctx_repeat(ctx, Bc)
    return codes, score


class Ranker:
    """Turns finished beams into ranked item lists (vectorised on GPU, with optional CCD rescoring).

    Ordering: beams by (calibrated) score; items inside a colliding SID group by cosine to the history profile.
    Groups are capped at `max_group` items (larger collision groups are truncated; collision rate is logged).
    """

    def __init__(self, cat, item_emb, device=None, max_group=8, first_ts=None):
        """first_ts: availability time of every item -> items that appear only AFTER a test interaction are never
        recommended for it (time-aware catalogue; prevents future-item leakage)"""
        self.cat = cat
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = torch.tensor(item_emb, device=self.dev)
        full = cat.prefix_keys[cat.L]
        ukeys, inv = np.unique(full, return_inverse=True)
        G = min(max_group, int(np.bincount(inv).max()))
        table = np.full((len(ukeys), G), -1, dtype=np.int64)
        fill = np.zeros(len(ukeys), dtype=np.int64)
        for item in np.argsort(inv, kind="stable"):
            g = inv[item]
            if fill[g] < G:
                table[g, fill[g]] = item
                fill[g] += 1
        self.ukeys = torch.tensor(ukeys, device=self.dev)
        self.first_ts = None if first_ts is None else torch.tensor(np.asarray(first_ts, dtype=np.int64), device=self.dev)
        self.table = torch.tensor(table, device=self.dev)
        self.G = G

    @torch.no_grad()
    def rank(self, codes, logp, profile, beta=0.0, topn=100, chunk=512, ts=None):
        """codes [N,B,L] np, logp [N,B] np, profile [N,d] np -> (ranked [N,topn] np (-1 pad), beam_keys [N,B] np, valid [N,B])"""
        N, B, L = codes.shape
        keys = np.zeros((N, B), dtype=np.int64)
        for j in range(L):
            keys = keys * self.cat.K + codes[:, :, j]
        sup = self.cat.support(codes.reshape(N * B, L)).reshape(N, B)
        out = np.full((N, topn), -1, dtype=np.int64)
        for s in range(0, N, chunk):
            k = torch.tensor(keys[s:s + chunk], device=self.dev)
            lp = torch.tensor(logp[s:s + chunk], device=self.dev, dtype=torch.float32)
            sp = torch.tensor(sup[s:s + chunk], device=self.dev)
            pr = torch.tensor(profile[s:s + chunk], device=self.dev)
            n = k.shape[0]
            gid = torch.searchsorted(self.ukeys, k).clamp(max=len(self.ukeys) - 1)
            ok = (self.ukeys[gid] == k) & torch.isfinite(lp)
            items = self.table[gid]  # [n,B,G]
            items = torch.where(ok[..., None], items, torch.full_like(items, -1))
            if self.first_ts is not None and ts is not None:  # not yet released at the time of the interaction
                tsb = torch.tensor(np.asarray(ts[s:s + chunk], dtype=np.int64), device=self.dev)
                future = self.first_ts[items.clamp(min=0)] > tsb[:, None, None]
                items = torch.where(future, torch.full_like(items, -1), items)
            cos = torch.einsum("nbgd,nd->nbg", self.emb[items.clamp(min=0)], pr)
            cos = cos.masked_fill(items < 0, -float("inf"))
            score = lp.masked_fill(~ok, -float("inf"))
            if beta != 0.0:
                best = cos.max(-1).values.masked_fill(~ok, 0.0)
                score = score + beta * (1.0 - sp) * best
            beam_rank = torch.argsort(torch.argsort(-score, dim=1, stable=True), dim=1)  # rank of each beam
            in_rank = torch.argsort(torch.argsort(-cos, dim=2, stable=True), dim=2)
            order_key = (beam_rank[..., None] * self.G + in_rank).float()
            order_key = order_key.masked_fill(items < 0, float("inf"))
            flat_items = items.reshape(n, -1)
            top = torch.topk(-order_key.reshape(n, -1), min(topn, flat_items.shape[1]), dim=1)
            sel = flat_items.gather(1, top.indices)
            sel = torch.where(torch.isfinite(top.values), sel, torch.full_like(sel, -1))
            out[s:s + n, :sel.shape[1]] = sel.cpu().numpy()
        valid = np.isfinite(logp)
        return out, keys, valid


@torch.no_grad()
def gt_token_ranks(model, cat, hist_items, tgt_items, compositional=False):
    """Teacher-forced rank of the ground-truth code at each level (trie-constrained). -> [N, L] (0 = top-1)."""
    L, K = cat.L, cat.K
    dev = hist_items.device
    N = hist_items.shape[0]
    ids, mask = hist_tokens(hist_items, model._codes_t, L, K)
    ctx = model.encode(ids, mask)
    tc = model._codes_t[tgt_items]  # [N, L]
    tok = N_SPECIAL + torch.arange(L, device=dev) * K + tc
    dec = torch.cat([torch.full((N, 1), BOS, dtype=torch.long, device=dev), tok[:, :-1]], 1)
    anchored = None
    if compositional:
        anchored = torch.zeros_like(dec, dtype=torch.bool)
        key = torch.zeros(N, dtype=torch.long, device=dev)
        for j in range(L - 1):
            key = key * K + tc[:, j]
            anchored[:, j + 1] = ~cat.is_seen_prefix(j + 1, key)
    h = model.decode(ctx, dec, anchored)
    ranks = []
    pkey = torch.zeros(N, dtype=torch.long, device=dev)
    for l in range(L):
        lg = model.level_logits(h[:, l], l).masked_fill(~cat.allowed_mask(l, pkey), -float("inf"))
        gt = lg.gather(1, tc[:, l:l + 1])
        ranks.append((lg > gt).sum(1))
        pkey = pkey * K + tc[:, l]
    return torch.stack(ranks, 1)


@torch.no_grad()
def item_logprob(model, cat, hist_items, items, chunk_items=25):
    """teacher-forced, trie-constrained log p(SID(item) | history) for a candidate matrix items [N, M] (-1 = pad)
    -> [N, M] (-inf for pads). Used by the SpecGR-style verifier."""
    L, K = cat.L, cat.K
    dev = hist_items.device
    N, M = items.shape
    ids, mask = hist_tokens(hist_items, model._codes_t, L, K)
    ctx = model.encode(ids, mask)
    out = torch.full((N, M), -float("inf"), device=dev)
    for s in range(0, M, chunk_items):
        it = items[:, s:s + chunk_items]
        m = it.shape[1]
        flat = it.reshape(-1)
        valid = flat >= 0
        tc = model._codes_t[flat.clamp(min=0)]  # [N*m, L]
        tok = N_SPECIAL + torch.arange(L, device=dev) * K + tc
        dec = torch.cat([torch.full((N * m, 1), BOS, dtype=torch.long, device=dev), tok[:, :-1]], 1)
        h = model.decode(model.ctx_repeat(ctx, m), dec)
        lp = torch.zeros(N * m, device=dev)
        pkey = torch.zeros(N * m, dtype=torch.long, device=dev)
        for l in range(L):
            lg = model.level_logits(h[:, l], l)
            allowed = cat.allowed_mask(l, pkey)
            lsm = torch.log_softmax(lg.masked_fill(~allowed, -1e9), -1)
            lp = lp + lsm.gather(1, tc[:, l:l + 1]).squeeze(1)
            pkey = pkey * K + tc[:, l]
        out[:, s:s + m] = torch.where(valid, lp, torch.full_like(lp, -float("inf"))).reshape(N, m)
    return out
