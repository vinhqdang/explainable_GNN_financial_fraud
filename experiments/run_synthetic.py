"""Controlled synthetic benchmark (see rtxgnn/synthetic.py for the generator).

Settings (all other generator parameters at their defaults):
  default      : 50 timed rings, 50 slow decoy cycles, no feature shift, no label noise
  no_decoy     : as default without decoy cycles (structure alone is informative)
  label_noise  : as default with 5% of labels flipped (evaluation uses the noisy labels)
  wide_burst   : rings completed within 10 days instead of 2 (weaker timing signal)
  shift        : as default with a +0.5 shift of the noise features of ring accounts
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import RESULTS, TABULAR, append_jsonl, done_keys, parse_seeds
from rtxgnn.synthetic import generate, prepare_synthetic
from rtxgnn.train import build, train_model, evaluate, set_seed, best_threshold
from rtxgnn.tabular import fit_tabular, predict
from rtxgnn.explain import attr_gnnexplainer, disjoint_batch

SETTINGS = {
    "default": {},
    "no_decoy": {"n_decoys": 0},
    "label_noise": {"label_noise": 0.05},
    "wide_burst": {"burst": 10.0},
    "shift": {"shift": 0.5},
}
PERIODS = (1.0, 7.0, 30.0, 90.0)
VARIANTS = {"rtxgnn": {}, "rtxgnn:no_hrape": {"temporal": "none"}, "rtxgnn:time2vec": {"temporal": "time2vec"},
            "rtxgnn:no_seal": {"use_masks": False}}


def edge_auc(model, d, targets):
    """AUC of the edge importances of the last SEAL layer (edge mask x source
    mask x attention) against the ground-truth ring transactions, computed over
    the incoming edges of each correctly detected ring account."""
    with torch.no_grad():
        out = model(d)
    last = out["layers"][-1]
    s, t = d.ei_dir
    imp = (last["edge"] * last["node"][s] * last["att"]).numpy()
    att = last["att"].numpy()
    res = {"seal": [], "attention": [], "random": []}
    rng = np.random.default_rng(0)
    for v in targets.tolist():
        inc = np.nonzero(t.numpy() == v)[0]
        gt = d.gt_edge[inc].numpy()
        if gt.min() == gt.max():
            continue
        res["seal"].append(roc_auc_score(gt, imp[inc]))
        res["attention"].append(roc_auc_score(gt, att[inc]))
        res["random"].append(roc_auc_score(gt, rng.random(len(inc))))
    # GNNExplainer on the same model and targets
    b, pos = disjoint_batch(d, targets)
    cls = torch.ones(len(pos), dtype=torch.long)
    _, em = attr_gnnexplainer(model, b, pos, cls)
    gtb = d.gt_edge[b.edge_ids].numpy()
    em = em.numpy()
    bt = b.ei_dir[1].numpy()
    ge = []
    for p in pos.tolist():
        inc = np.nonzero(bt == p)[0]
        g = gtb[inc]
        if g.min() != g.max():
            ge.append(roc_auc_score(g, em[inc]))
    res["gnnexplainer"] = ge
    return {k: float(np.mean(v)) for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,rf,xgb,gcn,sage,gat,tgat,tgn,sefraud,rtxgnn,rtxgnn:no_hrape,rtxgnn:time2vec,rtxgnn:no_seal")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = os.path.join(RESULTS, "synthetic.jsonl")
    done = done_keys(out, ("setting", "model", "seed"))
    for st in a.settings.split(","):
        for seed in parse_seeds(a.seeds):
            d = prepare_synthetic(generate(seed=seed, **SETTINGS[st]))
            stats = dict(nodes=d.num_nodes, edges=int(d.edge_index.size(1)), pos_rate=float(d.y.float().mean()))
            for name in a.models.split(","):
                if (st, name, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.time()
                extra = {}
                if name in TABULAR:
                    m = fit_tabular(name, d.x[d.train_mask].numpy(), d.y[d.train_mask].numpy(), seed)
                    p = predict(m, d.x.numpy())
                    thr = best_threshold(p[d.val_mask.numpy()], d.y[d.val_mask].numpy())
                else:
                    base = name.split(":")[0]
                    kw = dict(VARIANTS.get(name, {}))
                    if base == "rtxgnn":
                        kw["periods"] = PERIODS
                    model = build(base, d.x.size(1), **kw)
                    p, thr, hist = train_model(model, d, d, epochs=300, patience=40)
                    if base == "rtxgnn" and model.use_masks and seed < 3 and st in ("default", "wide_burst"):
                        det = torch.nonzero(d.test_mask & (d.y_true == 1) & torch.tensor(p >= thr)).squeeze(-1)
                        if len(det):
                            extra["edge_auc"] = edge_auc(model, d, det[:100])
                        if st == "default" and seed == 0 and name == "rtxgnn":
                            os.makedirs(os.path.join(RESULTS, "ckpt"), exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(RESULTS, "ckpt", "synthetic_rtxgnn_s0.pt"))
                row = dict(setting=st, model=name, seed=seed, thr=thr, time=time.time() - t0, **stats,
                           test=evaluate(p, d, thr), **extra)
                append_jsonl(out, row)
                print(st, name, seed, {k: round(v, 4) for k, v in row["test"].items()}, extra, flush=True)
