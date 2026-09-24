"""Faithfulness, sparsity-fidelity trade-off, stability and cost of feature
explanations on Elliptic.

Targets: 300 illicit and 300 licit test transactions (fixed random draw).
Explainers of the trained RTXGNN (checkpoints of seeds 0-4): SEAL feature mask,
saliency, integrated gradients, GNNExplainer and a random ranking; SEFraud is
evaluated with its own feature mask on its own model.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import RESULTS, elliptic
from rtxgnn.graph import split_regions
from rtxgnn.train import build
from rtxgnn.explain import (disjoint_batch, attr_mask, attr_random, attr_saliency, attr_ig,
                            attr_gnnexplainer, fidelity, jaccard_topk)

KS = (5, 10, 20, 40)


VARIANT_KW = {"rtxgnn_no_fidelity": {}, "rtxgnn_no_sparsity": {}}


def load(name, seed, in_dim):
    m = build(name.split("_")[0], in_dim, **VARIANT_KW.get(name, {}))
    m.load_state_dict(torch.load(os.path.join(RESULTS, "ckpt", f"{name}_s{seed}.pt")))
    m.eval()
    return m


def explainers(model, b, pos, cls, is_rtx):
    out = {}
    tm = {}
    if hasattr(model, "g_feat"):
        t = time.perf_counter(); out["mask"] = attr_mask(model, b, pos); tm["mask"] = time.perf_counter() - t
    if is_rtx:
        t = time.perf_counter(); out["saliency"] = attr_saliency(model, b, pos, cls); tm["saliency"] = time.perf_counter() - t
        t = time.perf_counter(); out["ig"] = attr_ig(model, b, pos, cls); tm["ig"] = time.perf_counter() - t
        t = time.perf_counter(); out["gnnexplainer"] = attr_gnnexplainer(model, b, pos, cls)[0]; tm["gnnexplainer"] = time.perf_counter() - t
        out["random"] = attr_random(model, b, pos); tm["random"] = 0.0
    return out, tm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = elliptic()
    _, dev = split_regions(d)
    g = torch.Generator().manual_seed(0)
    ill = torch.nonzero(dev.test_mask & (dev.y == 1)).squeeze(-1)
    lic = torch.nonzero(dev.test_mask & (dev.y == 0)).squeeze(-1)
    targets = torch.cat([ill[torch.randperm(len(ill), generator=g)[:a.n]], lic[torch.randperm(len(lic), generator=g)[:a.n]]])
    ytrue = dev.y[targets].numpy()
    b, pos = disjoint_batch(dev, targets)
    out_path = os.path.join(RESULTS, "explanations.jsonl")
    lo, hi = [int(s) for s in a.seeds.split("-")]
    masks_by_seed = {}
    for seed in range(lo, hi + 1):
        for name in ("rtxgnn", "sefraud", "rtxgnn_no_fidelity", "rtxgnn_no_sparsity"):
            ck = os.path.join(RESULTS, "ckpt", f"{name}_s{seed}.pt")
            if not os.path.exists(ck):
                continue
            model = load(name, seed, d.x.size(1))
            with torch.no_grad():
                p = torch.sigmoid(model(b)["logit"][pos])
            cls = (p > 0.5).long()
            attrs, tm = explainers(model, b, pos, cls, name == "rtxgnn")
            # stability under a small input perturbation of the target's features
            b2 = b.clone()
            gen = torch.Generator().manual_seed(seed)
            b2.x = b.x.clone()
            b2.x[pos] = b.x[pos] + 0.05 * torch.randn(len(pos), b.x.size(1), generator=gen)
            attrs2, _ = explainers(model, b2, pos, cls, name == "rtxgnn")
            for ex, at in attrs.items():
                fid = fidelity(model, b, pos, at, KS, cls)
                row = dict(model=name, explainer=ex, seed=seed, time_per_target_ms=1000 * tm[ex] / len(pos),
                           stability_jaccard10=float(jaccard_topk(at, attrs2[ex]).mean()) if ex != "random" else None)
                for k, (fp, fm) in fid.items():
                    for lab, sel in (("all", np.ones(len(pos), bool)), ("illicit", ytrue == 1), ("licit", ytrue == 0)):
                        row[f"fid+@{k}_{lab}"] = float(fp[sel].mean())
                        row[f"fid-@{k}_{lab}"] = float(fm[sel].mean())
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(name, ex, seed, {k: round(v, 4) for k, v in row.items() if isinstance(v, float) and ("@10_all" in k or "stab" in k or "time" in k)}, flush=True)
            if name == "rtxgnn":
                masks_by_seed[seed] = attrs["mask"]
    # agreement of SEAL feature masks across independently trained models
    seeds = sorted(masks_by_seed)
    agree = [float(jaccard_topk(masks_by_seed[i], masks_by_seed[j]).mean()) for i in seeds for j in seeds if i < j]
    rnd = [float(jaccard_topk(torch.rand(len(pos), b.x.size(1)), torch.rand(len(pos), b.x.size(1))).mean()) for _ in range(10)]
    with open(os.path.join(RESULTS, "explanations_cross_seed.json"), "w") as f:
        json.dump(dict(seal_mean=float(np.mean(agree)), seal_std=float(np.std(agree)), random_mean=float(np.mean(rnd))), f)
    np.save(os.path.join(RESULTS, "explanation_targets.npy"), dev.orig_id[targets].numpy())
