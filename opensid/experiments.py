"""Registry of every experiment in the paper (RQ1-RQ5 + analyses). run_all.py turns it into scheduler jobs."""
from .scheduler import Job, py
from .utils import ROOT, load_config, processed_dir

MAIN_DATASETS = ["amazon_industrial", "amazon_musical", "amazon_videogames", "mind"]
SEEDS = [2026, 2027, 2028]           # main generative models are run with 3 seeds (mean ± std)
BACKBONE_DATASETS = ["amazon_industrial", "amazon_musical"]   # RQ3
INC_DATASETS = ["amazon_industrial", "mind"]                  # RQ5
SENS_DATASET = "amazon_industrial"                            # rho / codebook-size sensitivity
LOO_DATASET = "amazon_industrial_loo"                         # leave-one-out protocol of prior work
INC_STEPS = 3000


_CFG = {"out_dir": "results", "K": 256, "L": 3}


def _res(ds, exp):
    return ROOT / _CFG["out_dir"] / ds / exp / "metrics.json"


def _done(ds, exp):
    """done-file written by a job after its metrics.json (a job whose done-file exists is skipped on re-run)."""
    return ROOT / _CFG["out_dir"] / ds / exp / "done"


def _tok(ds, K=None, L=None, fit=""):
    return processed_dir(ds) / f"sid_K{K or _CFG['K']}_L{L or _CFG['L']}{fit}" / "tokenizer.pkl"


def build_jobs(config_path=None, datasets=None, quick=False):
    cfgarg = ["--config", config_path] if config_path else []
    cfg = load_config(config_path)
    _CFG.update(out_dir=cfg["out_dir"], K=cfg["tokenizer"]["codebook_size"], L=cfg["tokenizer"]["levels"])
    datasets = datasets or (MAIN_DATASETS + ([] if quick else [LOO_DATASET]))
    jobs = []

    def gen(ds, exp, deps, *args):
        jobs.append(Job(f"{ds}.{exp}", py("opensid.train_gen", "--dataset", ds, "--exp", exp, *args, *cfgarg),
                        _done(ds, exp), deps))

    for ds in datasets:
        prep = f"{ds}.prepare"
        jobs.append(Job(prep, py("opensid.prepare", "--dataset", ds, *cfgarg), _tok(ds)))
        # ---------------- RQ1/RQ2: main table + ablations (every trained model: 3 seeds)
        seeds = SEEDS[:1] if quick else SEEDS
        for model in ["sasrec", "textsasrec"]:
            for i, seed in enumerate(seeds):
                sfx = "" if i == 0 else f"_s{seed}"
                jobs.append(Job(f"{ds}.{model}{sfx}", py("opensid.train_seq", "--dataset", ds, "--exp", model + sfx,
                                                           "--model", model, "--seed", seed, *cfgarg),
                                _done(ds, model + sfx), [prep]))
        for i, seed in enumerate(seeds):
            sfx = "" if i == 0 else f"_s{seed}"
            gen(ds, f"tiger{sfx}", [prep], "--emb", "standard", "--cpm", 0, "--seed", seed)          # TIGER (+CCD = w/o CATE&CPM)
            gen(ds, f"opensid_nocpm{sfx}", [prep], "--emb", "cate", "--cpm", 0, "--seed", seed)      # CATE (+CCD)
            gen(ds, f"opensid{sfx}", [prep], "--emb", "cate", "--cpm", 1, "--seed", seed)            # CATE+CPM (+CCD = full)
            gen(ds, f"liger{sfx}", [prep], "--emb", "standard", "--cpm", 0, "--liger", "--seed", seed)  # LIGER-lite baseline
        jobs.append(Job(f"{ds}.tiger_hybrid", py("opensid.hybrid", "--dataset", ds, *cfgarg),
                        _done(ds, "tiger_hybrid"), [f"{ds}.tiger", f"{ds}.textsasrec"], gpu=False))
        # generative + training-free content-kNN fusion (CPU, seconds): is OpenSID complementary to a content retriever?
        for i, seed in enumerate(seeds):
            sfx = "" if i == 0 else f"_s{seed}"
            for base, rk in [("tiger", "plain"), ("opensid", "ccd_overall")]:
                exp = f"hybrid_knn_{base}{sfx}"
                jobs.append(Job(f"{ds}.{exp}", py("opensid.hybrid", "--dataset", ds, "--gen_exp", base + sfx,
                                                  "--gen_rank", rk, "--content_exp", "content_knn", "--exp", exp, *cfgarg),
                                _done(ds, exp), [f"{ds}.{base}{sfx}", f"{ds}.simple"], gpu=False))
        jobs.append(Job(f"{ds}.specgr", py("opensid.specgr", "--dataset", ds, *cfgarg), _done(ds, "specgr"),
                        [f"{ds}.tiger", f"{ds}.textsasrec"]))
        jobs.append(Job(f"{ds}.simple", py("opensid.simple_baselines", "--dataset", ds, *cfgarg),
                        # rank_valid_plain.npy is the last file written; the kNN fusion jobs need it
                        ROOT / _CFG["out_dir"] / ds / "content_knn" / "rank_valid_plain.npy", [prep]))
        # training-free new-item controls (knn_new, knn_newprior) and their fusion with the generators
        jobs.append(Job(f"{ds}.cold_knn", py("opensid.cold_knn", "--dataset", ds, *cfgarg), _done(ds, "knn_newprior"),
                        [prep]))
        for i, seed in enumerate(seeds):
            sfx = "" if i == 0 else f"_s{seed}"
            for base, rk in [("tiger", "plain"), ("opensid", "ccd_overall")]:
                exp = f"hybrid_knnprior_{base}{sfx}"
                jobs.append(Job(f"{ds}.{exp}", py("opensid.hybrid", "--dataset", ds, "--gen_exp", base + sfx,
                                                  "--gen_rank", rk, "--content_exp", "knn_newprior", "--exp", exp,
                                                  *cfgarg),
                                _done(ds, exp), [f"{ds}.{base}{sfx}", f"{ds}.cold_knn"], gpu=False))
        if quick or ds == LOO_DATASET:
            continue
        # ---------------- fine-grained component ablations
        gen(ds, "opensid_nomask", [prep], "--emb", "cate", "--cpm", 1, "--no_delta_mask")    # Delta also on unseen tokens
        gen(ds, "opensid_keepext", [prep], "--emb", "cate", "--cpm", 1, "--cpm_keep_ext")    # CPM keeps extension loss
        # ---------------- "just refit the tokenizer on all items (incl. new ones)" baseline
        pf = f"{ds}.prepare_fitall"
        jobs.append(Job(pf, py("opensid.prepare", "--dataset", ds, "--fit", "all_items", *cfgarg), _tok(ds, fit="_fitall"), [prep]))
        gen(ds, "tiger_fitall", [pf], "--emb", "standard", "--cpm", 0, "--fit", "all_items")
        gen(ds, "opensid_fitall", [pf], "--emb", "cate", "--cpm", 1, "--fit", "all_items")
        # ---------------- RQ3: backbones
        if ds in BACKBONE_DATASETS:
            for bb in ["gpt", "qwen"]:
                # Qwen beam search is ~10x slower than T5 (a full CCD grid costs ~5.5 GPU-h per model), so the
                # Qwen models reuse the CCD (beta, quota) selected on valid for the T5 model of the same variant
                for base, extra in [("tiger", ["--emb", "standard", "--cpm", 0]), ("opensid", ["--emb", "cate", "--cpm", 1])]:
                    transfer = ["--ccd_from", _res(ds, base)] if bb == "qwen" else []
                    deps = [prep] + ([f"{ds}.{base}"] if bb == "qwen" else [])
                    gen(ds, f"{base}_{bb}", deps, *extra, "--backbone", bb, *transfer)
        # ---------------- RQ5: incremental update after the valid period is logged
        if ds in INC_DATASETS:
            t, o = f"{ds}.tiger", f"{ds}.opensid"
            rdir = ROOT / _CFG["out_dir"] / ds
            tck, ock = rdir / "tiger" / "best.pt", rdir / "opensid" / "best.pt"
            tm, om = _res(ds, "tiger"), _res(ds, "opensid")
            gen(ds, "inc_tiger_noupdate", [t], "--stage", "inc", "--init_from", tck, "--max_steps", 0, "--ccd_from", tm)
            gen(ds, "inc_tiger_finetune", [t], "--stage", "inc", "--init_from", tck, "--trainable", "all",
                "--max_steps", INC_STEPS, "--ccd_from", tm)
            gen(ds, "inc_tiger_retrain", [t], "--stage", "inc", "--ccd_from", tm)
            gen(ds, "inc_opensid_noupdate", [o], "--stage", "inc", "--emb", "cate", "--cpm", 1, "--init_from", ock,
                "--max_steps", 0, "--ccd_from", om)
            gen(ds, "inc_opensid_inc", [o], "--stage", "inc", "--emb", "cate", "--cpm", 1, "--init_from", ock,
                "--trainable", "inc", "--max_steps", INC_STEPS, "--ccd_from", om)
            gen(ds, "inc_opensid_finetune", [o], "--stage", "inc", "--emb", "cate", "--cpm", 1, "--init_from", ock,
                "--trainable", "all", "--max_steps", INC_STEPS, "--ccd_from", om)
            gen(ds, "inc_opensid_retrain", [o], "--stage", "inc", "--emb", "cate", "--cpm", 1, "--ccd_from", om)
        # ---------------- sensitivity: rho_max and codebook size K
        if ds == SENS_DATASET:
            for rho in [0.1, 0.4]:
                gen(ds, f"opensid_rho{rho}", [prep], "--emb", "cate", "--cpm", 1, "--rho", rho)
            for K in [128, 512]:
                pk = f"{ds}.prepare_K{K}"
                jobs.append(Job(pk, py("opensid.prepare", "--dataset", ds, "--K", K, *cfgarg), _tok(ds, K), [prep]))
                gen(ds, f"tiger_K{K}", [pk], "--emb", "standard", "--cpm", 0, "--K", K)
                gen(ds, f"opensid_K{K}", [pk], "--emb", "cate", "--cpm", 1, "--K", K)
    return jobs
