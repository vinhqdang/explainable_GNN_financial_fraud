"""RTXGNN: HRAPE temporal encoding + Self-Explainable Aggregation Layers (SEAL).

The implementation follows Algorithm 1 of the revised manuscript line by line:

1. input feature mask       M^feat_i = sigmoid(g_feat(x_i))            (per node)
2. input embedding          h^0_i = W_in (x_i * M^feat_i) + TE(t_i)
3. for each SEAL layer l:
     node (source) mask     M^node_j = sigmoid(g_node(h^{l-1}_j))
     edge mask              M^edge_ij = sigmoid(g_edge([h_i, h_j, e_type, TE(dt_ij)]))
     attention              a_ij = softmax_j(q_i k_j / sqrt(d) + gamma * r(dt_ij))
     aggregation            h^l_i = LN(h^{l-1}_i + W_o sum_j a_ij M^edge_ij M^node_j v_j)
4. prediction               logit_i = MLP([h^0_i, h^L_i])

Masks are consumed by the forward pass, so the explanation returned with a
prediction is exactly the set of gates that produced that prediction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as seg_softmax


def mlp(i, h, o, drop=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, o))


class HRAPE(nn.Module):
    """Hierarchical Recency-Aware Positional Encoding (revised).

    The encoding is invariant to a shift of the time axis: it only uses
    (i) multi-scale sinusoidal encodings of the time gap dt between a
    transaction and the node that receives the message, (ii) a recency score
    r = exp(-lambda * dt) with learnable lambda that biases the attention, and
    (iii) velocity features (log counts of the node's transactions in trailing
    windows). ``absolute=True`` additionally encodes the absolute timestamp with
    the same periods; this was the formulation of the first submission and is
    kept only for the ablation study.
    """

    def __init__(self, dim, periods, n_vel=3, absolute=False):
        super().__init__()
        self.register_buffer("periods", torch.tensor(periods, dtype=torch.float))
        k = len(periods)
        self.absolute = absolute
        if absolute:
            self.W_scale = nn.ModuleList([nn.Linear(2, dim) for _ in range(k)])
        self.W_vel = nn.Linear(n_vel, dim)
        self.W_gap = nn.Linear(2 * k, dim)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def phi(self, t):
        ang = 2 * math.pi * t.unsqueeze(-1) / self.periods
        return torch.stack([ang.sin(), ang.cos()], -1)  # [..., k, 2]

    def node(self, t, vel):
        te = self.W_vel(vel)
        if self.absolute:
            p = self.phi(t)
            te = te + sum(W(p[:, s]) for s, W in enumerate(self.W_scale))
        return te

    def edge(self, dt):
        te = self.W_gap(self.phi(dt).flatten(1))
        r = torch.exp(-F.softplus(self.log_lambda) * dt)
        return te, r


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al., 2019): one linear and k periodic components with
    learnable frequencies. Used in the ablation that replaces HRAPE: it encodes
    the same time gaps dt but has no recency bias and no velocity features.
    ``absolute=True`` encodes the absolute node timestamp instead."""

    def __init__(self, dim, k=8, absolute=False):
        super().__init__()
        self.absolute = absolute
        self.w = nn.Parameter(torch.randn(k + 1) * 0.1)
        self.b = nn.Parameter(torch.zeros(k + 1))
        self.proj = nn.Linear(k + 1, dim)

    def forward(self, t):
        z = t.unsqueeze(-1) * self.w + self.b
        z = torch.cat([z[:, :1], torch.sin(z[:, 1:])], -1)
        return self.proj(z)

    def node(self, t, vel):
        return self(t) if self.absolute else 0

    def edge(self, dt):
        return self(dt), torch.zeros_like(dt)


class SEAL(nn.Module):
    def __init__(self, dim, heads=4, n_etype=2, drop=0.2, use_masks=True, use_time=True):
        super().__init__()
        self.h, self.dk = heads, dim // heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ek = nn.Embedding(n_etype, dim)
        self.ev = nn.Embedding(n_etype, dim)
        # edge-mask generator g_edge([h_i, h_j, e_ij]) = w^T ReLU(A h_i + B h_j + C e_ij) + b,
        # evaluated with the node-level products A h, B h to avoid per-edge matmuls
        self.gA, self.gB = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False)
        self.gC = nn.Embedding(n_etype, dim)
        self.gT = nn.Linear(dim, dim, bias=False)
        self.gw = nn.Linear(dim, 1)
        self.g_node = mlp(dim, dim // 2, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(dim)
        self.ffn = mlp(dim, 2 * dim, dim, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.use_masks, self.use_time = use_masks, use_time

    def forward(self, h, ei, etype, te_e, rec, override=None):
        """override: optional dict with 'edge'/'node' tensors replacing the
        generated masks (used by the fidelity objectives and the evaluation)."""
        s, d = ei
        n = h.size(0)
        if override is not None and "edge" in override and "node" in override:
            m_edge, m_node = override["edge"], override["node"]
        elif self.use_masks:
            z = self.gA(h)[d] + self.gB(h)[s] + self.gC(etype)
            if self.use_time:
                z = z + self.gT(te_e)
            m_edge = torch.sigmoid(self.gw(F.relu(z))).squeeze(-1)
            m_node = torch.sigmoid(self.g_node(h)).squeeze(-1)
        else:
            m_edge = torch.ones(s.size(0), device=h.device)
            m_node = torch.ones(n, device=h.device)
        if override is not None and not ("edge" in override and "node" in override):
            m_edge = override.get("edge", m_edge)
            m_node = override.get("node", m_node)
        q = self.q(h).view(n, self.h, self.dk)
        k = self.k(h)[s] + self.ek(etype)
        if self.use_time:
            k = k + te_e
        k = k.view(-1, self.h, self.dk)
        v = self.v(h)[s] + self.ev(etype)
        if self.use_time:
            v = v + te_e
        v = v.view(-1, self.h, self.dk)
        score = (q[d] * k).sum(-1) / math.sqrt(self.dk)
        if self.use_time:
            score = score + self.gamma * rec.unsqueeze(-1)
        att = seg_softmax(score, d, num_nodes=n)
        gate = (m_edge * m_node[s]).unsqueeze(-1)
        msg = (att * gate).unsqueeze(-1) * v
        agg = torch.zeros(n, self.h, self.dk, device=h.device).index_add_(0, d, msg).view(n, -1)
        h = self.norm(h + self.drop(self.o(agg)))
        h = self.norm2(h + self.ffn(h))
        return h, {"edge": m_edge, "node": m_node, "att": att.mean(-1)}


class RTXGNN(nn.Module):
    def __init__(self, in_dim, dim=64, layers=2, heads=4, drop=0.2, periods=(2.0, 4.0, 13.0, 26.0),
                 temporal="hrape", use_masks=True, n_vel=3):
        super().__init__()
        self.use_masks = use_masks
        self.temporal = temporal
        if temporal in ("hrape", "hrape_abs"):
            self.te = HRAPE(dim, list(periods), n_vel, absolute=temporal == "hrape_abs")
        elif temporal in ("time2vec", "time2vec_abs"):
            self.te = Time2Vec(dim, absolute=temporal == "time2vec_abs")
        else:
            self.te = None
        self.g_feat = mlp(in_dim, dim, in_dim)
        self.inp = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([SEAL(dim, heads, drop=drop, use_masks=use_masks,
                                          use_time=self.te is not None) for _ in range(layers)])
        self.head = mlp(2 * dim, dim, 1, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, data, x=None, override=None):
        x = data.x if x is None else x
        n = x.size(0)
        if override is not None and "feat" in override:
            m_feat = override["feat"]
        elif self.use_masks:
            m_feat = torch.sigmoid(self.g_feat(x) + 2.0)  # initialised close to 1
        else:
            m_feat = torch.ones_like(x)
        h0 = self.inp(x * m_feat)
        if self.te is not None:
            h0 = h0 + self.te.node(data.t, data.vel)
            te_e, rec = self.te.edge(data.edt)
        else:
            te_e, rec = 0, None
        h0 = F.relu(h0)
        h = self.drop(h0)
        masks = []
        for i, layer in enumerate(self.layers):
            ov = None if override is None else override.get("layers", [None] * len(self.layers))[i]
            h, m = layer(h, data.ei_dir, data.etype, te_e, rec, ov)
            masks.append(m)
        logit = self.head(torch.cat([h0, h], -1)).squeeze(-1)
        return {"logit": logit, "feat": m_feat, "layers": masks}
