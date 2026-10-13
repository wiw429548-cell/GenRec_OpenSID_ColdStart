"""Residual-Quantized VAE item tokenizer (TIGER-style) -> Semantic IDs.

The tokenizer is fit ONLY on train-period (warm) items (config tokenizer.fit_on=train_items); new items are
quantized with the frozen encoder/codebooks, exactly as a deployed system would do.
Saves data/processed/<ds>/sid_K{K}_L{L}/tokenizer.pkl with
  codes  int64[n_items, L]
  codebooks float32[L, K, latent]   (used by CATE as content anchors z_c)
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger, load_config, load_pickle, processed_dir, save_pickle, set_seed

log = get_logger("rqvae")


def mlp(dims, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def kmeans(x, k, iters=50):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device)[:k]
    if n < k:  # pad with jittered copies
        idx = torch.cat([idx, torch.randint(0, n, (k - n,), device=x.device)])
    c = x[idx].clone() + 1e-4 * torch.randn(k, x.shape[1], device=x.device)
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).float()
        nz = cnt > 0
        c[nz] = sums[nz] / cnt[nz, None]
    return c


class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden, latent, levels, K, beta):
        super().__init__()
        self.enc = mlp([in_dim] + hidden + [latent])
        self.dec = mlp([latent] + hidden[::-1] + [in_dim])
        self.codebooks = nn.Parameter(torch.randn(levels, K, latent) * 0.1)
        self.levels, self.K, self.beta = levels, K, beta

    def quantize(self, z):
        r = z
        codes, qs, loss = [], [], 0.0
        for l in range(self.levels):
            cb = self.codebooks[l]
            idx = torch.cdist(r, cb).argmin(1)
            q = cb[idx]
            loss = loss + F.mse_loss(q, r.detach()) + self.beta * F.mse_loss(r, q.detach())
            codes.append(idx)
            qs.append(q)
            r = r - q.detach()
        zq = torch.stack(qs, 0).sum(0)
        return torch.stack(codes, 1), zq, loss, r

    def forward(self, x):
        z = self.enc(x)
        codes, zq, qloss, _ = self.quantize(z)
        zst = z + (zq - z).detach()
        xr = self.dec(zst)
        return xr, codes, qloss

    @torch.no_grad()
    def init_codebooks(self, x):
        r = self.enc(x)
        for l in range(self.levels):
            self.codebooks.data[l] = kmeans(r, self.K)
            idx = torch.cdist(r, self.codebooks[l]).argmin(1)
            r = r - self.codebooks[l][idx]


def sid_dir(ds, K, L, fit="train_items"):
    return processed_dir(ds) / (f"sid_K{K}_L{L}" + ("" if fit == "train_items" else "_fitall"))


def build(ds, cfg, K=None, L=None, fit=None):
    tcfg = dict(cfg["tokenizer"])
    K = K or tcfg["codebook_size"]
    L = L or tcfg["levels"]
    if fit:
        tcfg["fit_on"] = fit
    out = sid_dir(ds, K, L, tcfg["fit_on"]) / "tokenizer.pkl"
    if out.exists():
        return load_pickle(out)
    set_seed(cfg["seed"])
    data = load_pickle(processed_dir(ds) / "data.pkl")
    emb = np.load(processed_dir(ds) / "item_emb.npy")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fit_idx = np.where(data["warm"])[0] if tcfg["fit_on"] == "train_items" else np.arange(len(emb))
    x_all = torch.tensor(emb, device=dev)
    x_fit = x_all[torch.tensor(fit_idx, device=dev)]
    model = RQVAE(emb.shape[1], tcfg["hidden_dims"], tcfg["latent_dim"], L, K, tcfg["beta"]).to(dev)
    model.init_codebooks(x_fit[torch.randperm(len(x_fit), device=dev)[: min(len(x_fit), 50000)]])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    bs = tcfg["batch_size"]
    for ep in range(tcfg["epochs"]):
        perm = torch.randperm(len(x_fit), device=dev)
        tot, usage = 0.0, torch.zeros(L, K, device=dev)
        for i in range(0, len(perm), bs):
            xb = x_fit[perm[i:i + bs]]
            xr, codes, qloss = model(xb)
            loss = F.mse_loss(xr, xb) + qloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            for l in range(L):
                usage[l] += torch.bincount(codes[:, l], minlength=K)
        # dead-code restart (keeps codebook utilisation high)
        with torch.no_grad():
            z = model.enc(x_fit[torch.randperm(len(x_fit), device=dev)[:4096]])
            r = z
            for l in range(L):
                dead = (usage[l] == 0).nonzero().flatten()
                if len(dead):
                    model.codebooks.data[l, dead] = r[torch.randint(0, len(r), (len(dead),), device=dev)]
                idx = torch.cdist(r, model.codebooks[l]).argmin(1)
                r = r - model.codebooks[l][idx]
        if ep % 50 == 0 or ep == tcfg["epochs"] - 1:
            log.info(f"[{ds}] K={K} L={L} ep {ep} loss {tot / len(x_fit):.5f} dead/level "
                     f"{[(usage[l] == 0).sum().item() for l in range(L)]}")
    model.eval()
    with torch.no_grad():
        codes = []
        for i in range(0, len(x_all), 8192):
            c, _, _, _ = model.quantize(model.enc(x_all[i:i + 8192]))
            codes.append(c.cpu())
        codes = torch.cat(codes).numpy().astype(np.int64)
    sids = [tuple(c) for c in codes]
    n_unique = len(set(sids))
    tok = dict(codes=codes, codebooks=model.codebooks.detach().cpu().numpy().astype(np.float32), K=K, L=L,
               collision_rate=1 - n_unique / len(sids))
    log.info(f"[{ds}] SIDs assigned: {len(sids)} items, {n_unique} unique, collision {tok['collision_rate']:.4f}")
    save_pickle(tok, out)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--fit", default=None, choices=[None, "train_items", "all_items"])
    a = ap.parse_args()
    build(a.dataset, load_config(a.config), a.K, a.L, a.fit)


if __name__ == "__main__":
    main()
