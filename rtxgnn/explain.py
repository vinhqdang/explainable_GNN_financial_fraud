"""Feature-attribution explainers and faithfulness metrics.

All explainers return, for every target node, an attribution over the target's
own input features. They operate on a *disjoint* union of the exact 2-hop
subgraphs of the targets, so perturbing one target never affects another.
"""
import time
import numpy as np
import torch
from torch_geometric.data import Data

from .serving import CSR, sample_batch


def disjoint_batch(data, targets, hops=2):
    csr = CSR(data.ei_dir, data.num_nodes)
    parts, pos, off = [], [], 0
    for t in targets.tolist():
        sub, p = sample_batch(data, csr, torch.tensor([t]), hops=hops, fanout=10 ** 9)
        parts.append(sub)
        pos.append(int(p[0]) + off)
        off += sub.num_nodes
    b = Data(x=torch.cat([p.x for p in parts]), t=torch.cat([p.t for p in parts]),
             vel=torch.cat([p.vel for p in parts]), num_nodes=off)
    offs = np.cumsum([0] + [p.num_nodes for p in parts[:-1]])
    b.ei_dir = torch.cat([p.ei_dir + int(o) for p, o in zip(parts, offs)], 1)
    b.ei_und = b.ei_dir
    for k in ("etype", "edt", "edge_t"):
        b[k] = torch.cat([getattr(p, k) for p in parts])
    b.edge_ids = torch.cat([p.edge_ids for p in parts])
    b.graph_of_edge = torch.cat([torch.full((p.ei_dir.size(1),), i) for i, p in enumerate(parts)])
    return b, torch.tensor(pos)


def prob(model, b, x=None, override=None):
    kw = {} if override is None else {"override": override}
    out = model(b, x=x, **kw)
    return torch.sigmoid(out["logit"]), out


# ------------------------------------------------------------------ explainers
def attr_mask(model, b, pos):
    with torch.no_grad():
        return model(b)["feat"][pos]


def attr_random(model, b, pos, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(pos), b.x.size(1), generator=g)


def _signed_score(model, b, x, pos, cls):
    logit = model(b, x=x)["logit"][pos]
    return torch.where(cls == 1, logit, -logit)


def attr_saliency(model, b, pos, cls):
    x = b.x.clone().requires_grad_(True)
    _signed_score(model, b, x, pos, cls).sum().backward()
    return x.grad[pos].abs()


def attr_ig(model, b, pos, cls, steps=32):
    x0 = b.x
    total = torch.zeros(len(pos), x0.size(1))
    for a in (torch.arange(steps) + 0.5) / steps:
        x = x0.clone()
        x[pos] = a * x0[pos]
        x.requires_grad_(True)
        _signed_score(model, b, x, pos, cls).sum().backward()
        total += x.grad[pos]
    return (total / steps * x0[pos]).abs()


def attr_gnnexplainer(model, b, pos, cls, epochs=100, lr=0.01, feat_size=1.0, feat_ent=0.1,
                      edge_size=0.005, edge_ent=1.0):
    """GNNExplainer (Ying et al., 2019) with the regularisation coefficients of
    the PyTorch Geometric implementation: a feature mask on the target node and
    a mask on every edge of its computation graph."""
    n_e = b.ei_dir.size(1)
    fm = torch.nn.Parameter(torch.randn(len(pos), b.x.size(1)) * 0.1)
    em = torch.nn.Parameter(torch.randn(n_e) * 0.1)
    opt = torch.optim.Adam([fm, em], lr=lr)
    has_masks = hasattr(model, "layers") and getattr(model, "use_masks", False)
    with torch.no_grad():
        base = model(b)
    for _ in range(epochs):
        opt.zero_grad()
        x = b.x.clone()
        x[pos] = x[pos] * torch.sigmoid(fm)
        e = torch.sigmoid(em)
        if has_masks:
            ov = {"layers": [{"edge": l["edge"] * e, "node": l["node"]} for l in base["layers"]]}
            logit = model(b, x=x, override=ov)["logit"][pos]
        else:
            logit = model(b, x=x)["logit"][pos]
        p = torch.sigmoid(logit)
        p_c = torch.where(cls == 1, p, 1 - p).clamp(1e-6, 1)
        sf, se = torch.sigmoid(fm), e
        ent = lambda m: -(m * (m + 1e-9).log() + (1 - m) * (1 - m + 1e-9).log())
        loss = (-p_c.log()).sum() + feat_size * sf.mean(1).sum() + feat_ent * ent(sf).mean(1).sum() \
            + edge_size * se.sum() + edge_ent * ent(se).mean() * len(pos)
        loss.backward()
        opt.step()
    return torch.sigmoid(fm).detach(), torch.sigmoid(em).detach()


# -------------------------------------------------------------------- metrics
def fidelity(model, b, pos, attr, ks, cls):
    """Fid+ (necessity): drop of the predicted-class probability when the top-k
    features of the target are set to 0 (the training mean); Fid- (sufficiency):
    drop when *only* the top-k features are kept."""
    with torch.no_grad():
        p0 = prob(model, b)[0][pos]
        pc0 = torch.where(cls == 1, p0, 1 - p0)
        order = torch.argsort(attr, dim=1, descending=True)
        res = {}
        for k in ks:
            top = order[:, :k]
            keep = torch.zeros_like(attr).scatter_(1, top, 1.0)
            x = b.x.clone()
            x[pos] = b.x[pos] * (1 - keep)
            p = prob(model, b, x=x)[0][pos]
            fp = pc0 - torch.where(cls == 1, p, 1 - p)
            x = b.x.clone()
            x[pos] = b.x[pos] * keep
            p = prob(model, b, x=x)[0][pos]
            fm = pc0 - torch.where(cls == 1, p, 1 - p)
            res[k] = (fp.cpu().numpy(), fm.cpu().numpy())
    return res


def jaccard_topk(a, b, k=10):
    ta = torch.topk(a, k, dim=1).indices
    tb = torch.topk(b, k, dim=1).indices
    out = []
    for x, y in zip(ta.tolist(), tb.tolist()):
        sx, sy = set(x), set(y)
        out.append(len(sx & sy) / len(sx | sy))
    return np.array(out)
