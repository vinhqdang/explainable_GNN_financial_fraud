"""Label efficiency on Elliptic.

For a fraction f of the labelled training transactions (steps 1-30) we draw a
subset stratified by (time step, class): in every time step, round(f * n) licit
and round(f * n) illicit transactions are kept (at least one illicit
transaction per step when the step has any). Validation (31-34) and test
(35-49) sets are never subsampled. Each fraction is repeated with 5 subsets;
subset r is used with model seed r.
"""
import argparse
import os
import time

import numpy as np
import torch

from common import RESULTS, TABULAR, elliptic, parse_seeds, append_jsonl, done_keys, n_params
from rtxgnn.graph import split_regions
from rtxgnn.train import build, train_model, evaluate, set_seed, best_threshold
from rtxgnn.tabular import fit_tabular, predict


def stratified_subset(d, frac, seed):
    rng = np.random.default_rng(seed)
    keep = torch.zeros_like(d.train_mask)
    for t in range(1, 31):
        for c in (0, 1):
            idx = torch.nonzero(d.train_mask & (d.t == t) & (d.y == c)).squeeze(-1).numpy()
            if len(idx) == 0:
                continue
            k = max(1 if c == 1 else 0, int(round(frac * len(idx))))
            keep[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,mlp,gat,rtxgnn")
    ap.add_argument("--fracs", default="0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = os.path.join(RESULTS, "label_efficiency.jsonl")
    done = done_keys(out, ("model", "frac", "seed"))
    d = elliptic()
    for seed in parse_seeds(a.seeds):
        for frac in [float(f) for f in a.fracs.split(",")]:
            sub = stratified_subset(d, frac, seed) if frac < 1 else d.train_mask.clone()
            dtr, dev = split_regions(d, train_mask=sub)
            for name in a.models.split(","):
                if (name, frac, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.time()
                if name in TABULAR:
                    m = fit_tabular(name, dtr.x_raw[dtr.train_mask].numpy(), dtr.y[dtr.train_mask].numpy(), seed)
                    p = predict(m, dev.x_raw.numpy())
                    thr = best_threshold(p[dev.val_mask.numpy()], dev.y[dev.val_mask].numpy())
                else:
                    model = build(name, d.x.size(1))
                    p, thr, _ = train_model(model, dtr, dev, epochs=300, patience=40,
                                            eval_every=2 if name == "rtxgnn" else 1)
                row = dict(model=name, frac=frac, seed=seed, n_train=int(sub.sum()),
                           n_train_illicit=int((sub & (d.y == 1)).sum()), time=time.time() - t0,
                           test=evaluate(p, dev, thr), val=evaluate(p, dev, thr, dev.val_mask))
                append_jsonl(out, row)
                print(name, frac, seed, row["n_train"], {k: round(v, 4) for k, v in row["test"].items()}, flush=True)
