"""Main comparison, ablations and regularisation study on Elliptic.

Example:
    python experiments/run_elliptic.py --models rtxgnn,gat --seeds 0-9 --tag main
Variants of RTXGNN are given as rtxgnn:<variant> (see VARIANTS).
"""
import argparse
import os
import time

import numpy as np
import torch

from common import RESULTS, TABULAR, elliptic, parse_seeds, append_jsonl, done_keys, n_params
from rtxgnn.graph import split_regions
from rtxgnn.train import build, train_model, evaluate, per_step, set_seed, metrics, best_threshold
from rtxgnn.tabular import fit_tabular, predict

# name -> (model kwargs, objective overrides, optimiser overrides)
VARIANTS = {
    "full": ({}, {}, {}),
    "no_hrape": ({"temporal": "none"}, {}, {}),
    "time2vec": ({"temporal": "time2vec"}, {}, {}),
    "hrape_abs": ({"temporal": "hrape_abs"}, {}, {}),
    "time2vec_abs": ({"temporal": "time2vec_abs"}, {}, {}),
    "no_seal": ({"use_masks": False}, {}, {}),
    "no_sparsity": ({}, {"lam_sp": 0.0}, {}),
    "no_suf": ({}, {"lam_suf": 0.0}, {}),
    "no_nec": ({}, {"lam_nec": 0.0}, {}),
    "no_fidelity": ({}, {"lam_suf": 0.0, "lam_nec": 0.0}, {}),
    "no_curriculum": ({}, {"curriculum": False}, {}),
    "focal_loss": ({}, {"loss": "focal"}, {}),
    # regularisation study: masks without sparsity versus conventional regularisers
    "no_sparsity_drop0": ({"drop": 0.0}, {"lam_sp": 0.0}, {}),
    "no_sparsity_drop5": ({"drop": 0.5}, {"lam_sp": 0.0}, {}),
    "no_sparsity_wd1e3": ({}, {"lam_sp": 0.0}, {"wd": 1e-3}),
    "sparsity_x4": ({}, {"lam_sp": 0.2}, {}),
    "dim32": ({"dim": 32}, {}, {}),
    "dim128": ({"dim": 128}, {}, {}),
    "layers1": ({"layers": 1}, {}, {}),
}


def run_one(name, seed, d, dtr, dev, out_path, save_pred=True, epochs=300, patience=40):
    set_seed(seed)
    t0 = time.time()
    test_steps = list(range(35, 50))
    if name in TABULAR:
        m = fit_tabular(name, dtr.x_raw[dtr.train_mask].numpy(), dtr.y[dtr.train_mask].numpy(), seed)
        p = predict(m, dev.x_raw.numpy())
        thr = best_threshold(p[dev.val_mask.numpy()], dev.y[dev.val_mask].numpy())
        info = dict(epochs=None, params=None)
    else:
        base, _, var = name.partition(":")
        mkw, ocfg, okw = VARIANTS[var or "full"] if base == "rtxgnn" else ({}, {}, {})
        mkw = dict(mkw)
        model = build(base, d.x.size(1), **mkw)
        p, thr, hist = train_model(model, dtr, dev, epochs=epochs, patience=patience, rtx_cfg=ocfg,
                                   eval_every=2 if base == "rtxgnn" else 1, **okw)
        info = dict(epochs=len(hist), params=n_params(model), best_val_ap=max(h["val_ap"] for h in hist))
        if base in ("rtxgnn", "sefraud", "gat") and seed < 5 and not getattr(run_one, "no_ckpt", False):
            os.makedirs(os.path.join(RESULTS, "ckpt"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(RESULTS, "ckpt", f"{name.replace(':', '_')}_s{seed}.pt"))
    row = dict(model=name, seed=seed, thr=thr, time=time.time() - t0, **info,
               test=evaluate(p, dev, thr), test_at_0_5=evaluate(p, dev, 0.5),
               val=evaluate(p, dev, thr, dev.val_mask),
               steps=per_step(p, dev, thr, test_steps))
    append_jsonl(out_path, row)
    if save_pred and not getattr(run_one, "no_ckpt", False):
        os.makedirs(os.path.join(RESULTS, "preds"), exist_ok=True)
        np.save(os.path.join(RESULTS, "preds", f"{name.replace(':', '_')}_s{seed}.npy"), p.astype(np.float32))
    print(f"{name} seed {seed}: F1 {row['test']['f1']:.4f} AUC {row['test']['auc']:.4f} "
          f"AP {row['test']['ap']:.4f} ({row['time']:.0f}s)", flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--raw", action="store_true", help="neural models use the raw (not quantile-normalised) features")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = os.path.join(RESULTS, f"elliptic_{a.tag}.jsonl")
    d = elliptic()
    if a.raw:
        d.x = d.x_raw.clone()
    dtr, dev = split_regions(d)
    np.save(os.path.join(RESULTS, "elliptic_eval_ids.npy"), dev.orig_id.numpy())
    done = done_keys(out)
    run_one.no_ckpt = a.raw
    for seed in parse_seeds(a.seeds):
        for name in a.models.split(","):
            if (name, seed) in done:
                continue
            run_one(name, seed, d, dtr, dev, out)
