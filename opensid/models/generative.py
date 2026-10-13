"""Generative recommender over Semantic IDs with pluggable backbones.

backbone = t5  : TIGER-style encoder-decoder trained from scratch
backbone = gpt : decoder-only GPT-2 trained from scratch
backbone = hf  : pretrained decoder-only LM (e.g. Qwen2.5-0.5B) with SID tokens fed via inputs_embeds (LC-Rec-lite)

Decoder input for target SID (c_1..c_L): [BOS, c_1, ..., c_{L-1}] ; position j predicts c_{j+1} with a
level-restricted softmax over the K codes of level j (tied to the token embedding matrix).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sid import BOS, N_SPECIAL
from .embeddings import SIDEmbedding


class GenRec(nn.Module):
    def __init__(self, mcfg, L, K, emb_mode="standard", codebooks=None, seen_token=None):
        super().__init__()
        self.L, self.K = L, K
        self.kind = mcfg["backbone"]
        if self.kind == "t5":
            from transformers import T5Config, T5Model
            d = mcfg["d_model"]
            conf = T5Config(vocab_size=8, d_model=d, d_ff=mcfg["d_ff"], num_layers=mcfg["num_layers"],
                            num_decoder_layers=mcfg["num_layers"], num_heads=mcfg["num_heads"], d_kv=mcfg["d_kv"],
                            dropout_rate=mcfg["dropout"], feed_forward_proj="relu", pad_token_id=0,
                            decoder_start_token_id=0)
            self.backbone = T5Model(conf)
        elif self.kind == "gpt":
            from transformers import GPT2Config, GPT2Model
            d = mcfg["d_model"]
            conf = GPT2Config(vocab_size=8, n_embd=d, n_layer=mcfg["num_layers"], n_head=mcfg["num_heads"],
                              n_positions=512, resid_pdrop=mcfg.get("dropout", 0.1),
                              embd_pdrop=mcfg.get("dropout", 0.1), attn_pdrop=mcfg.get("dropout", 0.1))
            self.backbone = GPT2Model(conf)
        elif self.kind == "hf":
            from transformers import AutoModel
            # fp32 master weights; bf16 autocast is applied by the trainer
            self.backbone = AutoModel.from_pretrained(mcfg["hf_name"], torch_dtype=torch.float32)
            d = self.backbone.config.hidden_size
        else:
            raise ValueError(self.kind)
        self.d = d
        self.emb = SIDEmbedding(L, K, d, emb_mode, codebooks, seen_token)
        self.logit_scale = d ** -0.5 if self.kind != "t5" else 1.0
        self.out_norm = nn.LayerNorm(d) if self.kind == "hf" else nn.Identity()

    # ------------------------------------------------------------------ helpers
    def _bdtype(self):
        return next(self.backbone.parameters()).dtype

    def _hist_levels(self, hist_ids):
        lv = (hist_ids - N_SPECIAL).clamp(min=0) // self.K
        return torch.where(hist_ids >= N_SPECIAL, lv, torch.full_like(lv, self.L))

    def _dec_levels(self, t, B, device):
        # position 0 = BOS (level L), position j>=1 holds c_j (level j-1)
        lv = torch.arange(t, device=device) - 1
        lv[0] = self.L
        return lv[None].expand(B, t)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, hist_ids, hist_mask):
        h = self.emb(hist_ids, self._hist_levels(hist_ids)).to(self._bdtype())
        if self.kind == "t5":
            enc = self.backbone.encoder(inputs_embeds=h, attention_mask=hist_mask).last_hidden_state
            return {"enc": enc, "mask": hist_mask}
        return {"hist": h, "mask": hist_mask}

    def decode(self, ctx, dec_ids, dec_anchored=None):
        """returns hidden states [B, t, d] for every decoder position."""
        B, t = dec_ids.shape
        x = self.emb(dec_ids, self._dec_levels(t, B, dec_ids.device), dec_anchored).to(self._bdtype())
        if self.kind == "t5":
            out = self.backbone.decoder(inputs_embeds=x, encoder_hidden_states=ctx["enc"],
                                        encoder_attention_mask=ctx["mask"]).last_hidden_state
        else:
            seq = torch.cat([ctx["hist"], x], 1)
            mask = torch.cat([ctx["mask"], torch.ones(B, t, dtype=ctx["mask"].dtype, device=x.device)], 1)
            pos = (mask.long().cumsum(1) - 1).clamp(min=0)
            out = self.backbone(inputs_embeds=seq, attention_mask=mask, position_ids=pos).last_hidden_state[:, -t:]
        return self.out_norm(out.float())

    def level_logits(self, h, level):
        """h [N, d] -> logits [N, K] over the codes of `level` (tied weights)."""
        return (h @ self.emb.level_matrix(level).t()) * self.logit_scale

    def ctx_repeat(self, ctx, n):
        return {k: v.repeat_interleave(n, 0) for k, v in ctx.items()}

    def ctx_select(self, ctx, idx):
        return {k: v[idx] for k, v in ctx.items()}

    # ------------------------------------------------------------------ training loss
    # ------------------------------------------------------------------ LIGER-style dense head (baseline)
    def enable_dense(self, item_text_emb, d_e=128, temp=0.07):
        """LIGER (Yang et al. 2024): generative retrieval + sequential dense retrieval over item content."""
        self.register_buffer("item_text", torch.tensor(item_text_emb, dtype=torch.float32))
        self.dense_q = nn.Linear(self.d, d_e)
        self.dense_r = nn.Sequential(nn.Linear(self.item_text.shape[1], 2 * d_e), nn.GELU(), nn.Linear(2 * d_e, d_e))
        self.dense_temp = temp

    def dense_query(self, ctx):
        B = ctx["mask"].shape[0]
        dev = ctx["mask"].device
        h0 = self.decode(ctx, torch.full((B, 1), BOS, dtype=torch.long, device=dev))[:, 0]
        return F.normalize(self.dense_q(h0), dim=-1)

    def dense_items(self, idx=None):
        x = self.item_text if idx is None else self.item_text[idx]
        return F.normalize(self.dense_r(x), dim=-1)

    def loss(self, hist_ids, hist_mask, tgt_codes, anchor_from=None, drop_ext=None, dense_tgt=None, dense_neg=None):
        """tgt_codes [B, L] codes; anchor_from [B] int (L means no anchoring): decoder inputs c_j with j>=anchor_from
        use content-anchored embeddings (CPM). drop_ext [B] bool: skip the loss term of the masked extension step."""
        B = tgt_codes.shape[0]
        dev = tgt_codes.device
        tok = N_SPECIAL + torch.arange(self.L, device=dev) * self.K + tgt_codes  # [B, L]
        dec_ids = torch.cat([torch.full((B, 1), BOS, device=dev, dtype=torch.long), tok[:, :-1]], 1)
        anchored = None
        if anchor_from is not None:
            j = torch.arange(self.L, device=dev)[None]  # decoder position j holds c_j (j>=1)
            anchored = (j >= anchor_from[:, None]) & (j >= 1)
        ctx = self.encode(hist_ids, hist_mask)
        h = self.decode(ctx, dec_ids, anchored)  # [B, L, d]
        total = 0.0
        weight = torch.ones(B, self.L, device=dev)
        if drop_ext is not None and anchor_from is not None:
            # the step that *creates* the masked prefix (predicts c_{anchor_from}) is dropped
            j = torch.arange(self.L, device=dev)[None]
            weight = torch.where(drop_ext[:, None] & (j == (anchor_from[:, None] - 1)), torch.zeros_like(weight), weight)
        for l in range(self.L):
            lg = self.level_logits(h[:, l], l)
            ce = F.cross_entropy(lg, tgt_codes[:, l], reduction="none")
            total = total + (ce * weight[:, l]).sum() / weight[:, l].sum().clamp(min=1)
        total = total / self.L
        if dense_tgt is not None and hasattr(self, "dense_q"):  # sampled-softmax dense retrieval loss (LIGER)
            q = F.normalize(self.dense_q(h[:, 0]), dim=-1)
            cand, inv = torch.unique(torch.cat([dense_tgt, dense_neg]), return_inverse=True)
            logits = q @ self.dense_items(cand).t() / self.dense_temp
            total = total + F.cross_entropy(logits, inv[:len(dense_tgt)])
        return total
