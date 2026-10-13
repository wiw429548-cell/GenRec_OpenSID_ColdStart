"""Semantic-ID catalog: token vocabulary, prefix trie (vectorised), training support and cold buckets.

Bucket of an item w.r.t. the set of warm (train-period) items:
  warm : item itself occurs in train-period interactions
  C1   : cold item whose full SID path equals the path of some warm item (seen path)
  C2   : cold item, every (level, code) token seen in warm SIDs, but the full path is unseen
  C3   : cold item with at least one unseen atomic token
"""
import numpy as np
import torch

PAD, BOS = 0, 1
N_SPECIAL = 2
BUCKETS = ["warm", "C1", "C2", "C3"]


class SIDCatalog:
    def __init__(self, codes, K, warm_mask, first_ts=None):
        self.codes = np.asarray(codes, dtype=np.int64)  # [n_items, L]
        self.n_items, self.L = self.codes.shape
        self.K = K
        self.n_tokens = N_SPECIAL + self.L * K
        self.warm = np.asarray(warm_mask, dtype=bool)

        # prefix keys: key(prefix of length l) = sum_j c_j * K^(l-1-j)  (row-major)
        self.prefix_keys = [self._keys(self.codes[:, :l]) for l in range(self.L + 1)]
        # trie masks: for level l, rows = unique prefixes of length l, cols = allowed codes
        self.trie_keys, self.trie_masks = [], []
        for l in range(self.L):
            keys = self.prefix_keys[l]
            uk, inv = np.unique(keys, return_inverse=True)
            m = np.zeros((len(uk), K), dtype=bool)
            m[inv, self.codes[:, l]] = True
            self.trie_keys.append(uk)
            self.trie_masks.append(m)
        # support from warm items
        wc = self.codes[self.warm]
        self.seen_token = np.zeros((self.L, K), dtype=bool)
        for l in range(self.L):
            self.seen_token[l, wc[:, l]] = True
        self.seen_prefix = [np.unique(self._keys(wc[:, :l])) for l in range(self.L + 1)]
        # item lookup by full path
        full = self.prefix_keys[self.L]
        self.item_order = np.argsort(full, kind="stable")
        self.sorted_full = full[self.item_order]
        self.bucket = self._buckets()
        # earliest release time of any item under each prefix (time-aware constrained decoding)
        self.prefix_min_ts = None
        if first_ts is not None:
            ft = np.asarray(first_ts, dtype=np.int64)
            self.prefix_min_ts = []
            for l in range(1, self.L + 1):
                keys = self.prefix_keys[l]
                uk, inv = np.unique(keys, return_inverse=True)
                mt = np.full(len(uk), np.iinfo(np.int64).max, dtype=np.int64)
                np.minimum.at(mt, inv, ft)
                self.prefix_min_ts.append((uk, mt))

    def _keys(self, prefix):
        prefix = np.asarray(prefix, dtype=np.int64)
        key = np.zeros(prefix.shape[0], dtype=np.int64)
        for j in range(prefix.shape[1]):
            key = key * self.K + prefix[:, j]
        return key

    def _buckets(self):
        b = np.empty(self.n_items, dtype=object)
        full = self.prefix_keys[self.L]
        seen_full = np.isin(full, self.seen_prefix[self.L])
        all_tok = np.all(self.seen_token[np.arange(self.L)[None, :], self.codes], axis=1)
        b[:] = "C3"
        b[all_tok] = "C2"
        b[seen_full] = "C1"
        b[self.warm] = "warm"
        return b

    # ---- token ids ----
    def token_ids(self, items):
        """items: int array [...] -> token ids [..., L]"""
        c = self.codes[np.asarray(items)]
        return N_SPECIAL + np.arange(self.L) * self.K + c

    def items_of_key(self, key):
        lo = np.searchsorted(self.sorted_full, key, "left")
        hi = np.searchsorted(self.sorted_full, key, "right")
        return self.item_order[lo:hi]

    def support(self, prefix_codes):
        """longest seen prefix length / L, for full paths [N, L]"""
        n = prefix_codes.shape[0]
        sup = np.zeros(n, dtype=np.float32)
        for l in range(1, self.L + 1):
            ok = np.isin(self._keys(prefix_codes[:, :l]), self.seen_prefix[l])
            sup = np.where(ok, l / self.L, sup)
            # a prefix of length l can only be seen if length l-1 was seen; isin handles that implicitly
        return sup

    def to_torch(self, device):
        """GPU tensors for vectorised constrained decoding."""
        self.t_trie_keys = [torch.tensor(k, device=device) for k in self.trie_keys]
        self.t_trie_masks = [torch.tensor(m, device=device) for m in self.trie_masks]
        self.t_seen_prefix = [torch.tensor(k, device=device) for k in self.seen_prefix]
        self.t_prefix_min_ts = None if self.prefix_min_ts is None else             [(torch.tensor(k, device=device), torch.tensor(t, device=device)) for k, t in self.prefix_min_ts]
        return self

    def released(self, length, prefix_key, ts):
        """bool: some item under this prefix (of `length` codes) is already released at time ts (broadcastable)"""
        if self.t_prefix_min_ts is None or ts is None:
            return torch.ones_like(prefix_key, dtype=torch.bool)
        keys, mt = self.t_prefix_min_ts[length - 1]
        pos = torch.searchsorted(keys, prefix_key).clamp(max=len(keys) - 1)
        return (keys[pos] == prefix_key) & (mt[pos] <= ts)

    def allowed_mask(self, level, prefix_key):
        """prefix_key: LongTensor [N] (keys of length `level`) -> bool mask [N, K]"""
        keys = self.t_trie_keys[level]
        pos = torch.searchsorted(keys, prefix_key).clamp(max=len(keys) - 1)
        ok = keys[pos] == prefix_key
        m = self.t_trie_masks[level][pos]
        return m & ok[:, None]

    def is_seen_prefix(self, length, prefix_key):
        keys = self.t_seen_prefix[length]
        if len(keys) == 0:
            return torch.zeros_like(prefix_key, dtype=torch.bool)
        pos = torch.searchsorted(keys, prefix_key).clamp(max=len(keys) - 1)
        return keys[pos] == prefix_key
