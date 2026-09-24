"""Figures of the revised manuscript (written to manuscript/revision1/figures)."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import RESULTS, ROOT

FIG = os.path.join(ROOT, "manuscript", "revision1", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
COL = {"rtxgnn": "#c0392b", "rf": "#2c7fb8", "xgb": "#41b6c4", "sage": "#7f7f7f", "sefraud": "#e69f00", "gat": "#999999"}
LAB = {"rtxgnn": "RTXGNN", "rf": "Random forest", "xgb": "XGBoost", "sage": "GraphSAGE", "sefraud": "SEFraud", "gat": "GAT"}


def load(name):
    p = os.path.join(RESULTS, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def temporal():
    rows = load("elliptic_main.jsonl")
    ret = load("retraining.jsonl")
    if not rows:
        return
    per = defaultdict(lambda: defaultdict(list))
    counts = {}
    for r in rows:
        for s in r["steps"]:
            per[r["model"]][s["step"]].append(s["f1"])
            counts[s["step"]] = s["illicit"]
    steps = list(range(35, 50))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax2 = ax.twinx()
    ax2.bar(steps, [counts[t] for t in steps], color="#dddddd", width=0.7, zorder=0)
    ax2.set_ylabel("illicit transactions", color="#888888")
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    for m in ("rtxgnn", "rf", "sage", "sefraud"):
        if m in per:
            mu = np.array([np.mean(per[m][t]) for t in steps]); sd = np.array([np.std(per[m][t]) for t in steps])
            ax.plot(steps, mu, marker="o", ms=3, color=COL[m], label=LAB[m] + " (static)")
            ax.fill_between(steps, mu - sd, mu + sd, color=COL[m], alpha=0.15)
    if ret:
        rp = defaultdict(lambda: defaultdict(list))
        for r in ret:
            rp[r["model"]][r["step"]].append(r["test"]["f1"])
        for m in ("rtxgnn", "rf"):
            if m in rp:
                ax.plot(steps, [np.mean(rp[m][t]) if rp[m][t] else np.nan for t in steps], ls="--", marker="s", ms=3,
                        color=COL[m], label=LAB[m] + " (retrained)")
    ax.axvline(42.5, color="k", lw=0.8, ls=":")
    ax.text(42.6, 0.95, "dark-market\nshutdown", fontsize=7, va="top")
    ax.set_xlabel("test time step"); ax.set_ylabel("F1 (illicit class)"); ax.set_ylim(0, 1)
    ax.set_xticks(steps)
    ax.legend(fontsize=7, ncol=2, loc="lower left", frameon=False)
    fig.savefig(os.path.join(FIG, "temporal.pdf")); plt.close(fig)


def fidelity_curves():
    rows = load("explanations.jsonl")
    if not rows:
        return
    ks = (5, 10, 20, 40)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r["explainer"] if r["model"] == "rtxgnn" else "sefraud-mask"
        for k in ks:
            agg[key][("+", k)].append(r[f"fid+@{k}_all"]); agg[key][("-", k)].append(r[f"fid-@{k}_all"])
    names = {"mask": "SEAL mask (RTXGNN)", "saliency": "Saliency", "ig": "Integrated gradients",
             "gnnexplainer": "GNNExplainer", "random": "Random", "sefraud-mask": "SEFraud mask (own model)"}
    cols = {"mask": "#c0392b", "saliency": "#2c7fb8", "ig": "#41b6c4", "gnnexplainer": "#7b3294", "random": "#999999",
            "sefraud-mask": "#e69f00"}
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax, sign, title in ((axs[0], "+", "Fid$^+$ (remove top-$k$), higher is better"),
                            (axs[1], "-", "Fid$^-$ (keep only top-$k$), lower is better")):
        for key in names:
            if key not in agg:
                continue
            mu = [np.mean(agg[key][(sign, k)]) for k in ks]
            ax.plot(ks, mu, marker="o", ms=3, color=cols[key], label=names[key], ls="--" if key == "random" else "-")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("explanation size $k$ (features)"); ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("change in predicted-class probability")
    axs[1].legend(fontsize=6.5, frameon=False)
    fig.savefig(os.path.join(FIG, "fidelity.pdf")); plt.close(fig)


def label_efficiency():
    rows = load("label_efficiency.jsonl")
    if not rows:
        return
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][r["frac"]].append(r["test"]["f1"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in ("rtxgnn", "rf", "gat", "mlp"):
        if m not in agg:
            continue
        fr = sorted(agg[m]); mu = np.array([np.mean(agg[m][f]) for f in fr]); sd = np.array([np.std(agg[m][f]) for f in fr])
        ax.errorbar([100 * f for f in fr], mu, yerr=sd, marker="o", ms=3, capsize=2, color=COL.get(m, "#555555"),
                    label=LAB.get(m, m.upper()))
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels([5, 10, 20, 50, 100])
    ax.set_xlabel("labelled training transactions (%)"); ax.set_ylabel("test F1")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(FIG, "label_efficiency.pdf")); plt.close(fig)


def fraud_ring():
    """Annotated explanation of a detected ring on the synthetic benchmark."""
    ck = os.path.join(RESULTS, "ckpt", "synthetic_rtxgnn_s0.pt")
    if not os.path.exists(ck):
        return
    import networkx as nx
    from rtxgnn.synthetic import generate, prepare_synthetic
    from rtxgnn.train import build
    from run_synthetic import PERIODS
    d = prepare_synthetic(generate(seed=0))
    model = build("rtxgnn", d.x.size(1), periods=PERIODS)
    model.load_state_dict(torch.load(ck, map_location="cpu")); model.eval()
    with torch.no_grad():
        out = model(d)
    p = torch.sigmoid(out["logit"]).cpu().numpy()
    last = out["layers"][-1]
    s, t = d.ei_dir
    imp = (last["edge"] * last["node"][s] * last["att"]).cpu().numpy()
    # pick the ring with the most detected test members
    ring_edges = torch.nonzero(d.gt_edge_raw == 1).squeeze(-1)
    src, dst = d.edge_index[:, ring_edges]
    G = nx.DiGraph(); G.add_edges_from(zip(src.tolist(), dst.tolist()))
    rings = [sorted(c) for c in nx.weakly_connected_components(G)]
    test = d.test_mask.cpu().numpy()
    rings.sort(key=lambda c: -sum(test[v] and p[v] > 0.5 for v in c))
    ring = rings[0]
    target = next(v for v in ring if test[v]) if any(test[v] for v in ring) else ring[0]
    nodes = set(ring)
    ei = d.ei_dir.cpu().numpy()
    for v in ring:
        nb = ei[0, ei[1] == v]
        nodes.update(nb[:6].tolist())
    nodes = sorted(nodes)
    H = nx.DiGraph(); H.add_nodes_from(nodes)
    ets = d.edge_t.cpu().numpy(); et = d.etype.cpu().numpy()
    elist = []
    for e in range(ei.shape[1]):
        a, b = ei[0, e], ei[1, e]
        if et[e] == 0 and a in nodes and b in nodes and b in ring:
            elist.append((a, b, imp[e], ets[e], bool(d.gt_edge[e])))
    for a, b, w, tt, g in elist:
        H.add_edge(a, b)
    pos = nx.kamada_kawai_layout(H.to_undirected())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    wmax = max(w for *_, w, _, _ in [(0, 0, e[2], 0, 0) for e in elist]) if elist else 1
    for a, b, w, tt, g in elist:
        nx.draw_networkx_edges(H, pos, edgelist=[(a, b)], ax=ax, width=0.5 + 4 * w / wmax,
                               edge_color="#c0392b" if g else "#aaaaaa", arrowsize=8,
                               style="solid" if g else "dashed", connectionstyle="arc3,rad=0.08")
        if g:
            xm = (pos[a][0] + pos[b][0]) / 2; ym = (pos[a][1] + pos[b][1]) / 2
            ax.text(xm, ym, f"d{tt:.1f}\n{w:.2f}", fontsize=6, color="#7b1d12", ha="center")
    colors = ["#c0392b" if d.y_true[v] == 1 else "#9ecae1" for v in nodes]
    nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_color=colors, node_size=[200 if v == target else 90 for v in nodes],
                           edgecolors=["k" if v == target else "none" for v in nodes], ax=ax)
    nx.draw_networkx_labels(H, pos, labels={v: f"{p[v]:.2f}" for v in ring}, font_size=6, ax=ax,
                            verticalalignment="bottom")
    ax.set_axis_off()
    ax.set_title("Detected laundering ring (synthetic benchmark, seed 0)", fontsize=8)
    fig.savefig(os.path.join(FIG, "fraud_ring.pdf")); plt.close(fig)
    json.dump({"ring": [int(v) for v in ring], "target": int(target), "scores": {int(v): float(p[v]) for v in ring},
               "edges": [(int(a), int(b), float(w), float(tt), g) for a, b, w, tt, g in elist]},
              open(os.path.join(RESULTS, "fraud_ring.json"), "w"))


if __name__ == "__main__":
    for f in (temporal, fidelity_curves, label_efficiency, fraud_ring):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep the other figures
            print("failed", f.__name__, e)
