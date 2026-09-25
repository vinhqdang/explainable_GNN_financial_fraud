"""Hyperparameter tuning on Elliptic with an equal budget for every tuned model.

Each model gets the same number of configurations drawn at random from its
search space (configuration 0 is the default used in the main comparison).
Every configuration is trained with seeds 0-2, and configurations are ranked by
their mean *validation* AP (steps 31-34). Test metrics are recorded but never
used for selection. ``--final`` re-runs the selected configuration with seeds
0-9 and writes results/elliptic_tuned.jsonl.

Example:
    python experiments/run_tuning.py --models rtxgnn,xgb,gas --n 20
    python experiments/run_tuning.py --models rtxgnn,xgb,gas --final
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import RESULTS, TABULAR, elliptic, append_jsonl, parse_seeds
from rtxgnn.graph import split_regions
from rtxgnn.device import to_dev
from rtxgnn.train import build, train_model, evaluate, per_step, set_seed, best_threshold
from rtxgnn.tabular import fit_tabular, predict

# search spaces: name -> {"model": {...}, "opt": {...}, "obj": {...}} (tabular: "params")
GNN = {"model": {"dim": [32, 64, 128], "drop": [0.0, 0.2, 0.3, 0.5]},
       "opt": {"lr": [1e-3, 2e-3, 5e-3, 1e-2], "wd": [0.0, 1e-5, 1e-4, 1e-3]}}
SPACES = {
    "rtxgnn": {"model": {"dim": [64, 128], "drop": [0.0, 0.2, 0.3, 0.5], "layers": [2, 3]},
               "opt": {"lr": [1e-3, 2e-3, 5e-3, 1e-2], "wd": [0.0, 1e-5, 1e-4, 1e-3]},
               "obj": {"lam_suf": [0.25, 0.5, 1.0], "lam_nec": [0.25, 0.5, 1.0], "lam_sp": [0.01, 0.05],
                       "k_feat": [10, 20]}},
    "sage": {"model": dict(GNN["model"], layers=[2, 3]), "opt": GNN["opt"]},
    "gcn": {"model": dict(GNN["model"], layers=[2, 3]), "opt": GNN["opt"]},
    "gat": {"model": dict(GNN["model"], layers=[2, 3]), "opt": GNN["opt"]},
    "gas": GNN, "fraudre": GNN, "tgn": GNN, "tgat": GNN, "sefraud": GNN,
    "xgb": {"params": {"n_estimators": [300, 600, 1000], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
                       "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
                       "min_child_weight": [1, 5, 10]}},
    "rf": {"params": {"n_estimators": [100, 300, 500], "max_features": [20, 50, 100, "sqrt"],
                      "min_samples_leaf": [1, 2, 5], "class_weight": [None, "balanced_subsample"]}},
}


def sample(name, cid):
    """Configuration ``cid`` of model ``name``; configuration 0 is the default."""
    cfg = {g: {} for g in SPACES[name]}
    if cid == 0:
        return cfg
    rng = np.random.RandomState(1000 * (1 + list(SPACES).index(name)) + cid)
    for g, space in SPACES[name].items():
        for k, vals in space.items():
            cfg[g][k] = vals[rng.randint(len(vals))]
    return cfg


def run(name, cfg, seed, d, dtr, dev):
    set_seed(seed)
    t0 = time.time()
    if name in TABULAR:
        m = fit_tabular(name, dtr.x_raw[dtr.train_mask].cpu().numpy(), dtr.y[dtr.train_mask].cpu().numpy(), seed,
                        **cfg.get("params", {}))
        p = predict(m, dev.x_raw.cpu().numpy())
        thr = best_threshold(p[dev.val_mask.cpu().numpy()], dev.y[dev.val_mask].cpu().numpy())
        epochs = None
    else:
        model = build(name, d.x.size(1), **cfg.get("model", {}))
        p, thr, hist = train_model(model, dtr, dev, rtx_cfg=cfg.get("obj") or None,
                                   eval_every=2 if name == "rtxgnn" else 1, **cfg.get("opt", {}))
        epochs = len(hist)
    return dict(thr=thr, time=time.time() - t0, epochs=epochs, val=evaluate(p, dev, thr, dev.val_mask),
                test=evaluate(p, dev, thr), test_at_0_5=evaluate(p, dev, 0.5),
                steps=per_step(p, dev, thr, list(range(35, 50))))


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def select(name):
    """Configuration with the highest mean validation AP over its tuning seeds."""
    rows = load(os.path.join(RESULTS, f"tuning_{name}.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(r["cid"], []).append(r)
    full = {c: v for c, v in by.items() if len(v) >= 3}
    if not full:
        return None
    c = max(full, key=lambda c: (np.mean([r["val"]["ap"] for r in full[c]]), np.mean([r["val"]["f1"] for r in full[c]])))
    return c, full[c][0]["config"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=20, help="configurations per model (including the default)")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = elliptic()
    dtr, dev = split_regions(d)
    dtr, dev = to_dev(dtr), to_dev(dev)
    for name in a.models.split(","):
        if a.final:
            out = os.path.join(RESULTS, "elliptic_tuned.jsonl")
            sel = select(name)
            if sel is None:
                print("no complete tuning results for", name)
                continue
            cid, cfg = sel
            done = {(r["model"], r["seed"]) for r in load(out)}
            for seed in range(10):
                if (name, seed) in done:
                    continue
                row = dict(model=name, seed=seed, cid=cid, config=cfg, **run(name, cfg, seed, d, dtr, dev))
                append_jsonl(out, row)
                print(f"final {name} cid {cid} seed {seed}: val AP {row['val']['ap']:.4f}", flush=True)
            continue
        out = os.path.join(RESULTS, f"tuning_{name}.jsonl")
        done = {(r["cid"], r["seed"]) for r in load(out)}
        for cid in range(a.n):
            cfg = sample(name, cid)
            for seed in parse_seeds(a.seeds):
                if (cid, seed) in done:
                    continue
                try:
                    row = dict(model=name, cid=cid, seed=seed, config=cfg, **run(name, cfg, seed, d, dtr, dev))
                except (ValueError, RuntimeError) as e:  # e.g. out of memory for a large configuration
                    print(f"{name} cid {cid} seed {seed} failed: {e!r}", flush=True)
                    continue
                append_jsonl(out, row)
                print(f"{name} cid {cid} seed {seed}: val AP {row['val']['ap']:.4f} ({row['time']:.0f}s)", flush=True)
