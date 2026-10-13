"""SID token embeddings.

standard : free embedding per (level, code) token (TIGER / LC-Rec style)
cate     : Codebook-Anchored Token Embedding (OpenSID module A)
           E[l, c] = W_l * norm(z_{l,c}) + b_l + mask_{l,c} * Delta_{l,c}
           z_{l,c}  : RQ-VAE codeword (content anchor, frozen)
           mask     : 1 for tokens seen in train-period SIDs, 0 otherwise  -> unseen tokens are purely content-defined
           anchored : Delta switched off (used by CPM / compositional decoding)
Output logits are tied to the same matrix, so unseen tokens also get content-based logits.
"""
import torch
import torch.nn as nn

from ..sid import N_SPECIAL


class SIDEmbedding(nn.Module):
    def __init__(self, L, K, d, mode="standard", codebooks=None, seen_token=None, init_std=0.02):
        super().__init__()
        self.L, self.K, self.d, self.mode = L, K, d, mode
        self.special = nn.Embedding(N_SPECIAL, d)
        self.level = nn.Embedding(L + 1, d)  # level index of an input position (L = BOS/special)
        nn.init.normal_(self.special.weight, std=init_std)
        nn.init.normal_(self.level.weight, std=init_std)
        if mode == "standard":
            self.tok = nn.Parameter(torch.randn(L * K, d) * init_std)
        elif mode == "cate":
            z = torch.tensor(codebooks, dtype=torch.float32)  # [L, K, latent]
            z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
            self.register_buffer("z", z)
            self.proj = nn.ModuleList([nn.Linear(z.shape[-1], d) for _ in range(L)])
            for p in self.proj:
                nn.init.normal_(p.weight, std=init_std)
                nn.init.zeros_(p.bias)
            self.delta = nn.Parameter(torch.zeros(L * K, d))
            self.register_buffer("seen", torch.tensor(seen_token, dtype=torch.float32).reshape(L * K, 1))
        else:
            raise ValueError(mode)

    def table(self, anchored=False):
        """[L*K, d] token matrix."""
        if self.mode == "standard":
            return self.tok
        base = torch.cat([self.proj[l](self.z[l]) for l in range(self.L)], 0)
        if anchored:
            return base
        return base + self.delta * self.seen

    def level_matrix(self, level, anchored=False):
        t = self.table(anchored)
        return t[level * self.K:(level + 1) * self.K]

    def forward(self, ids, levels, anchored=None):
        """ids [B,T] (0/1 special, else SID token), levels [B,T] in [0..L], anchored [B,T] bool or None."""
        is_sid = ids >= N_SPECIAL
        sid_idx = (ids - N_SPECIAL).clamp(min=0)
        tab = self.table(False)
        out = tab[sid_idx]
        if self.mode == "cate" and anchored is not None and anchored.any():
            out = torch.where(anchored[..., None], self.table(True)[sid_idx], out)
        out = torch.where(is_sid[..., None], out, self.special(ids.clamp(max=N_SPECIAL - 1)))
        return out + self.level(levels)

    def reg_loss(self):
        if self.mode != "cate":
            return torch.zeros((), device=self.level.weight.device)
        return (self.delta * self.seen).pow(2).sum(-1).mean()
