"""SASRec (ID-based) and TextSASRec (UniSRec/VQ-Rec-style text-transferable sequential recommender).

TextSASRec represents every item purely by an MLP adaptor over its frozen text embedding, so it can score cold
items -> the strongest non-generative cold-start baseline.
"""
import torch
import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(self, d, layers, heads, dropout, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d)
        self.heads = heads
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        """x [B,T,d] left-padded, mask [B,T] bool (True=valid) -> last position representation [B,d]

        A single [B*heads, T, T] mask (causal + padded keys) in which every position may attend to itself. The
        previous causal + src_key_padding_mask combination left the padded (left) query rows fully masked; in eval
        mode PyTorch's fast path turned them into NaNs that propagated to the last position of every user with a
        history shorter than max_len (=> near-random test rankings)."""
        B, T = mask.shape
        x = self.drop(x + self.pos(torch.arange(T, device=x.device))[None])
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        blocked = causal[None] | ~mask[:, None, :]
        blocked = blocked & ~torch.eye(T, dtype=torch.bool, device=x.device)[None]
        h = self.enc(x, mask=blocked.repeat_interleave(self.heads, 0))
        return self.norm(h[:, -1])


class SASRec(nn.Module):
    def __init__(self, n_items, d, layers, heads, dropout, max_len):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d)  # index n_items = padding
        nn.init.normal_(self.item.weight, std=0.02)
        self.seq = SeqEncoder(d, layers, heads, dropout, max_len)
        self.n_items = n_items

    def item_matrix(self):
        return self.item.weight[:-1]

    def forward(self, hist):
        mask = hist >= 0
        x = self.item(torch.where(mask, hist, torch.full_like(hist, self.n_items)))
        return self.seq(x, mask)


class TextSASRec(nn.Module):
    def __init__(self, item_emb, d, layers, heads, dropout, max_len):
        super().__init__()
        self.register_buffer("text", torch.tensor(item_emb, dtype=torch.float32))
        e = self.text.shape[1]
        self.adaptor = nn.Sequential(nn.Linear(e, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.seq = SeqEncoder(d, layers, heads, dropout, max_len)
        self.pad = nn.Parameter(torch.zeros(d))
        self.temp = nn.Parameter(torch.tensor(0.07).log())

    def item_matrix(self):
        return nn.functional.normalize(self.adaptor(self.text), dim=-1)

    def forward(self, hist, item_mat=None):
        mask = hist >= 0
        im = item_mat if item_mat is not None else self.item_matrix()
        x = torch.where(mask[..., None], im[hist.clamp(min=0)], self.pad)
        return nn.functional.normalize(self.seq(x, mask), dim=-1) / self.temp.exp()
