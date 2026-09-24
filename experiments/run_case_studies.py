"""Case studies: explanations of individual Elliptic test transactions.

Elliptic's features are anonymised; we therefore report only the feature
index and its documented group (local: our indices 0-92, i.e. original
features 2-94; aggregated: our indices 93-164, original features 95-166) and
never assign a semantic meaning to a feature.
"""
import json
import os

import numpy as np
import torch

from common import RESULTS, ROOT, elliptic
from rtxgnn.graph import split_regions
from rtxgnn.train import build
from rtxgnn.serving import CSR, sample_batch, explain_targets

OUT = os.path.join(ROOT, "manuscript", "revision1", "tables")


def group(i):
    return "local" if i < 93 else "aggregated"


if __name__ == "__main__":
    torch.set_num_threads(2)
    d = elliptic()
    _, dev = split_regions(d)
    model = build("rtxgnn", d.x.size(1))
    model.load_state_dict(torch.load(os.path.join(RESULTS, "ckpt", "rtxgnn_s0.pt"), map_location="cpu"))
    model.eval()
    thr = [json.loads(l) for l in open(os.path.join(RESULTS, "elliptic_main.jsonl"))
           if json.loads(l)["model"] == "rtxgnn" and json.loads(l)["seed"] == 0][0]["thr"]
    with torch.no_grad():
        p = torch.sigmoid(model(dev)["logit"]).cpu().numpy()
    te = dev.test_mask.cpu().numpy(); y = dev.y.cpu().numpy()
    deg = torch.bincount(dev.ei_dir[1], minlength=dev.num_nodes).cpu().numpy()
    rng = np.random.default_rng(0)
    cand_tp = np.nonzero(te & (y == 1) & (p >= thr) & (deg >= 2) & (dev.t.cpu().numpy() <= 42))[0]
    cand_tn = np.nonzero(te & (y == 0) & (p < thr) & (deg >= 2))[0]
    cand_fn = np.nonzero(te & (y == 1) & (p < thr) & (dev.t.cpu().numpy() >= 43))[0]
    picks = [("true positive", int(rng.choice(cand_tp))), ("true positive", int(rng.choice(cand_tp))),
             ("true negative", int(rng.choice(cand_tn))), ("false negative (after step 43)", int(rng.choice(cand_fn)))]
    csr = CSR(dev.ei_dir, dev.num_nodes)
    cases = []
    for kind, v in picks:
        sub, pos = sample_batch(dev, csr, torch.tensor([v]), fanout=10 ** 9)
        with torch.no_grad():
            out = model(sub)
        rec = explain_targets(out, sub, pos, k_feat=5, k_edge=3)[0]
        # effect of removing the top-5 features on the score
        x = sub.x.clone(); idx = [f for f, _ in rec["features"]]
        x[pos[0], idx] = 0
        with torch.no_grad():
            p_rm = float(torch.sigmoid(model(sub, x=x)["logit"][pos[0]]))
        lab = {1: "illicit", 0: "licit", -1: "unlabelled"}
        nb = []
        for u, et, w in rec["edges"]:  # u is an index into dev (sub.orig_id holds dev indices)
            nb.append(dict(tx=int(dev.orig_id[u]), direction="incoming" if et == 0 else "outgoing", importance=w,
                           label=lab[int(dev.y[u])], score=float(p[u])))
        fm = out["feat"][pos[0]]
        sat = float((fm > 0.99).float().mean())
        with torch.no_grad():
            logit_f = (model.g_feat(sub.x[pos[0]:pos[0] + 1]) + 2.0)[0]
        open_ = torch.nonzero(fm > 0.5).squeeze(-1)
        order = open_[torch.argsort(logit_f[open_], descending=True)]
        rec["features"] = [(int(f), float(fm[f])) for f in order[:5]]
        n_open, n_open_local = int(len(open_)), int((open_ < 93).sum())
        cases.append(dict(kind=kind, tx=int(dev.orig_id[v]), mask_saturated_frac=sat, n_open=n_open, n_open_local=n_open_local, step=int(dev.t[v]), label=lab[int(y[v])], score=rec["score"],
                          threshold=thr, features=[(f, group(f), m) for f, m in rec["features"]],
                          score_without_top5=p_rm, neighbours=nb, degree=int(deg[v])))
    json.dump(cases, open(os.path.join(RESULTS, "case_studies.json"), "w"), indent=1)
    lines = ["\\begin{tabular}{p{2.2cm}p{1.3cm}p{1.5cm}p{3.6cm}p{4.0cm}}", "\\toprule",
             "Case & score (w/o top-5) & open features (local) & first five open features: index (group) & top neighbours: direction, label, importance\\\\",
             "\\midrule"]
    for c in cases:
        feats = ", ".join(f"{f} ({'L' if g == 'local' else 'A'})" for f, g, m in c["features"])
        nbs = "; ".join(f"{n['direction'][:3]}., {n['label']}, {n['importance']:.2f}" for n in c["neighbours"]) or "--"
        lines.append(f"{c['kind']}, step {c['step']} & {c['score']:.2f} ({c['score_without_top5']:.2f}) & {c['n_open']} ({c['n_open_local']}) & {feats} & {nbs}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "case_studies.tex"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(cases, indent=1)[:3000])
