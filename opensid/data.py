"""Tensorised samples: history item matrix, target, timestamp, user content profile."""
import numpy as np
import torch

from .sid import N_SPECIAL, PAD


class SplitArrays:
    def __init__(self, data, split, max_hist, item_emb=None):
        pairs = data["splits"][split]
        seqs = data["user_seqs"]
        N = len(pairs)
        hist = np.full((N, max_hist), -1, dtype=np.int64)
        tgt = np.zeros(N, dtype=np.int64)
        ts = np.zeros(N, dtype=np.int64)
        for n, (u, p) in enumerate(pairs):
            items, t = seqs[u]
            h = items[max(0, p - max_hist):p]
            hist[n, max_hist - len(h):] = h  # left padding
            tgt[n] = items[p]
            ts[n] = t[p]
        self.hist, self.tgt, self.ts, self.N = hist, tgt, ts, N
        self.split = split
        if item_emb is not None:  # content profile of the history (only needed for eval splits)
            prof = np.zeros((N, item_emb.shape[1]), dtype=np.float32)
            for s in range(0, N, 2048):
                hb = hist[s:s + 2048]
                m = hb >= 0
                e = item_emb[np.where(m, hb, 0)] * m[..., None]
                prof[s:s + 2048] = e.sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
            prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8
            self.profile = prof
        else:
            self.profile = None

    def subset(self, idx):
        o = object.__new__(SplitArrays)
        o.hist, o.tgt, o.ts = self.hist[idx], self.tgt[idx], self.ts[idx]
        o.profile = None if self.profile is None else self.profile[idx]
        o.N, o.split = len(idx), self.split
        return o


def hist_tokens(hist_items, codes_t, L, K):
    """hist_items LongTensor [B, H] (-1 pad) -> token ids [B, H*L], mask [B, H*L]"""
    B, H = hist_items.shape
    valid = hist_items >= 0
    c = codes_t[hist_items.clamp(min=0)]  # [B, H, L]
    tok = N_SPECIAL + torch.arange(L, device=c.device) * K + c
    tok = torch.where(valid[..., None], tok, torch.full_like(tok, PAD))
    mask = valid[..., None].expand(B, H, L)
    tok, mask = tok.reshape(B, H * L), mask.reshape(B, H * L).long()
    # guarantee at least one attended position (all-pad rows would produce NaNs)
    empty = mask.sum(1) == 0
    if empty.any():
        mask[empty, -1] = 1
    return tok, mask
