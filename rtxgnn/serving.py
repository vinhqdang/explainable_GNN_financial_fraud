"""Request-level inference: neighbourhood sampling, prediction and explanation.

A request is a set of target transactions. For each target the 2-hop
neighbourhood is sampled from an in-memory CSR adjacency (at most ``fanout``
neighbours per hop), the sampled subgraphs are merged into one batch, the model
is run once, and the explanation (top-k features and top-k incoming edges of
every target, read from the masks used in the forward pass) is returned.
"""
import numpy as np
import torch
from torch_geometric.data import Data


class CSR:
    def __init__(self, ei, n):
        s, d = ei
        order = torch.argsort(d, stable=True)
        self.src = s[order]
        self.eid = order
        self.ptr = torch.zeros(n + 1, dtype=torch.long)
        self.ptr[1:] = torch.cumsum(torch.bincount(d, minlength=n), 0)

    def in_edges(self, nodes, fanout, gen):
        """Edge ids of (up to ``fanout``) incoming edges of every node in ``nodes``."""
        out = []
        ptr = self.ptr
        for v in nodes.tolist():
            a, b = int(ptr[v]), int(ptr[v + 1])
            if b - a > fanout:
                sel = torch.randperm(b - a, generator=gen)[:fanout] + a
                out.append(self.eid[sel])
            else:
                out.append(self.eid[a:b])
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def sample_batch(data, csr, targets, hops=2, fanout=25, gen=None):
    """Union of the sampled k-hop subgraphs of ``targets`` (directed edges of
    ``data.ei_dir``). Returns the subgraph and the positions of the targets."""
    gen = gen or torch.Generator().manual_seed(0)
    frontier = targets
    nodes = [targets]
    eids = []
    for _ in range(hops):
        e = csr.in_edges(frontier, fanout, gen)
        eids.append(e)
        frontier = torch.unique(data.ei_dir[0, e])
        nodes.append(frontier)
    nodes = torch.unique(torch.cat(nodes))
    e = torch.unique(torch.cat(eids)) if eids else torch.zeros(0, dtype=torch.long)
    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(len(nodes))
    sub = Data(x=data.x[nodes], t=data.t[nodes], vel=data.vel[nodes], num_nodes=len(nodes))
    sub.ei_dir = remap[data.ei_dir[:, e]]
    sub.etype, sub.edt, sub.edge_t = data.etype[e], data.edt[e], data.edge_t[e]
    # ei_dir holds both directions of every payment, so it doubles as the
    # symmetric message-passing graph of GCN/GraphSAGE/GAT
    sub.ei_und = sub.ei_dir
    sub.edge_ids = e
    sub.orig_id = nodes
    return sub, remap[targets]


def explain_targets(out, sub, pos, k_feat=5, k_edge=3):
    """Explanation records read from the masks of an RTXGNN forward pass."""
    feat = out["feat"][pos]
    fv, fi = feat.topk(k_feat, dim=1)
    # edge importance of the last SEAL layer: edge mask x source-node mask x attention
    last = out["layers"][-1]
    s, d = sub.ei_dir
    imp = last["edge"] * last["node"][s] * last["att"]
    recs = []
    prob = torch.sigmoid(out["logit"][pos])
    for j, p in enumerate(pos.tolist()):
        inc = torch.nonzero(d == p).squeeze(-1)
        top = inc[torch.argsort(imp[inc], descending=True)[:k_edge]]
        recs.append({
            "tx": int(sub.orig_id[p]),
            "score": float(prob[j]),
            "features": [(int(a), float(b)) for a, b in zip(fi[j], fv[j])],
            "edges": [(int(sub.orig_id[s[e]]), int(sub.etype[e]), float(imp[e])) for e in top],
        })
    return recs
