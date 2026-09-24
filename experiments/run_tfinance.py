"""Second real-world financial dataset: T-Finance (GADBench splits 0-4,
40/20/40 train/validation/test). T-Finance has no timestamps, so temporal
baselines are not applicable and RTXGNN runs without HRAPE (SEAL + objective
only). Graph models see at most 25 sampled neighbours per node."""
import argparse
import os
import time

import torch

from common import slot, RESULTS, TABULAR, append_jsonl, done_keys, parse_seeds, n_params
from rtxgnn.data import load_tfinance, quantile_normalize
from rtxgnn.graph import prepare
from rtxgnn.train import build, train_model, evaluate, set_seed, best_threshold
from rtxgnn.tabular import fit_tabular, predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,rf,xgb,mlp,gcn,sage,gat,caregnn,pcgnn,fraudre,sefraud,rtxgnn")
    ap.add_argument("--splits", default="0-4")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = os.path.join(RESULTS, "tfinance.jsonl")
    done = done_keys(out, ("model", "seed"))
    for split in parse_seeds(a.splits):
        d = prepare(load_tfinance(split=split, seed=split))
        d.all_steps = [0.0]
        d.x_raw = d.x.clone()
        d.x = quantile_normalize(d.x, d.train_mask)
        for name in a.models.split(","):
            if (name, split) in done:
                continue
            set_seed(split)
            t0 = time.time()
            if name in TABULAR:
                m = fit_tabular(name, d.x_raw[d.train_mask].numpy(), d.y[d.train_mask].numpy(), split)
                p = predict(m, d.x_raw.numpy())
                thr = best_threshold(p[d.val_mask.numpy()], d.y[d.val_mask].numpy())
            else:
                kw = {"temporal": "none"} if name == "rtxgnn" else {}
                model = build(name, d.x.size(1), **kw)
                with slot() if name in ("rtxgnn", "sefraud") else open(os.devnull):
                    p, thr, _ = train_model(model, d, d, epochs=300, patience=40,
                                            eval_every=2 if name == "rtxgnn" else 1)
            row = dict(model=name, seed=split, thr=thr, time=time.time() - t0, test=evaluate(p, d, thr))
            append_jsonl(out, row)
            print(name, split, {k: round(v, 4) for k, v in row["test"].items()}, f"{row['time']:.0f}s", flush=True)
