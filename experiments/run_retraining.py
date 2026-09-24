"""Concept drift and periodic retraining on Elliptic.

Static: the model trained on steps 1-30 scores every test step (main results).
Retrain: before scoring test step t, the model is re-fitted on all labelled
steps 1..t-5, early-stopped and thresholded on steps t-4..t-1, i.e. labels
become available with a one-step delay. Neural models are warm-started from
the model of the previous step (at most 60 epochs, patience 10); tree
ensembles are refitted from scratch.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

from common import slot, RESULTS, TABULAR, elliptic, parse_seeds, append_jsonl, done_keys
from rtxgnn.graph import split_regions
from rtxgnn.train import build, train_model, metrics, set_seed, best_threshold
from rtxgnn.tabular import fit_tabular, predict


def masks_for(d, t):
    lab = d.y >= 0
    tr = lab & (d.t <= t - 5)
    va = lab & (d.t >= t - 4) & (d.t <= t - 1)
    te = lab & (d.t == t)
    return tr, va, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,rtxgnn")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = os.path.join(RESULTS, "retraining.jsonl")
    done = done_keys(out, ("model", "seed", "step"))
    d = elliptic()
    orig = (d.train_mask.clone(), d.val_mask.clone(), d.test_mask.clone())
    for seed in parse_seeds(a.seeds):
        for name in a.models.split(","):
            prev = None
            for t in range(35, 50):
                if (name, seed, t) in done:
                    continue
                set_seed(seed * 100 + t)
                d.train_mask, d.val_mask, d.test_mask = masks_for(d, t)
                dtr, dev = split_regions(d)
                t0 = time.time()
                if name in TABULAR:
                    m = fit_tabular(name, dtr.x_raw[dtr.train_mask].numpy(), dtr.y[dtr.train_mask].numpy(), seed)
                    p = predict(m, dev.x_raw.numpy())
                    thr = best_threshold(p[dev.val_mask.numpy()], dev.y[dev.val_mask].numpy())
                else:
                    model = build(name, d.x.size(1))
                    lock = slot() if name == "rtxgnn" else open(os.devnull)
                    lock.__enter__()
                    if prev is not None:
                        model.load_state_dict(prev)
                        p, thr, _ = train_model(model, dtr, dev, epochs=60, patience=10)
                    else:
                        p, thr, _ = train_model(model, dtr, dev, epochs=300, patience=40,
                                                eval_every=2 if name == "rtxgnn" else 1)
                    prev = copy.deepcopy(model.state_dict())
                    lock.__exit__(None, None, None)
                te = dev.test_mask.numpy()
                row = dict(model=name, seed=seed, step=t, time=time.time() - t0, thr=thr,
                           n=int(te.sum()), illicit=int(dev.y[dev.test_mask].sum()),
                           test=metrics(p[te], dev.y[dev.test_mask].numpy(), thr))
                append_jsonl(out, row)
                print(name, seed, t, {k: round(v, 4) for k, v in row["test"].items()}, flush=True)
            d.train_mask, d.val_mask, d.test_mask = orig
