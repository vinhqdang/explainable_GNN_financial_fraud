"""Baseline models re-implemented for node-level fraud detection.

Every model maps ``data`` to ``{"logit": [N]}`` (optionally ``"aux"``, an
auxiliary loss, and masks for the self-explaining baseline).  Models designed
for multi-relation or continuous-time graphs are adapted to the single-relation
snapshot graph of Elliptic as described in Appendix B of the manuscript.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import softmax as seg_softmax, add_self_loops

from .model import mlp


def mean_agg(h, ei, n, w=None):
    s, d = ei
    m = h[s] if w is None else h[s] * w.unsqueeze(-1)
    out = torch.zeros(n, h.size(1), device=h.device).index_add_(0, d, m)
    deg = torch.zeros(n, device=h.device).index_add_(0, d, torch.ones_like(d, dtype=h.dtype) if w is None else w)
    return out / deg.clamp(min=1e-6).unsqueeze(-1)


# ---------------------------------------------------------------- general GNNs
class MLP(nn.Module):
    def __init__(self, in_dim, dim=64, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(drop), nn.Linear(dim, 1))

    def forward(self, data, x=None):
        return {"logit": self.net(data.x if x is None else x).squeeze(-1)}


class PygGNN(nn.Module):
    def __init__(self, conv, in_dim, dim=64, drop=0.2, layers=2):
        super().__init__()
        self.kind = conv
        dims = [in_dim] + [dim] * layers
        if conv == "gcn":
            self.convs = nn.ModuleList([GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif conv == "sage":
            self.convs = nn.ModuleList([SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif conv == "gat":
            self.convs = nn.ModuleList([GATConv(a, b // 4, heads=4) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(dim, 1)
        self.drop = drop

    def forward(self, data, x=None):
        h = data.x if x is None else x
        for c in self.convs:
            h = F.dropout(F.relu(c(h, data.ei_und)), self.drop, self.training)
        return {"logit": self.out(h).squeeze(-1)}


# ------------------------------------------------------------- temporal GNNs
def time_encode(dt, w, b):
    """Functional time encoding of TGAT: cos(w*dt + b)."""
    return torch.cos(dt.unsqueeze(-1) * w + b)


class TimeEnc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(1 / 10 ** torch.linspace(0, 9, dim).numpy()).float())
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt):
        return time_encode(dt, self.w, self.b)


class TemporalAttn(nn.Module):
    """TGAT layer: attention over temporal neighbours with keys/values built from
    [h_j || Phi(t_i - t_ij)] and query [h_i || Phi(0)]."""

    def __init__(self, dim, tdim, heads=2, drop=0.2):
        super().__init__()
        self.h, self.dk = heads, dim // heads
        self.q = nn.Linear(dim + tdim, dim)
        self.kv = nn.Linear(dim + tdim + 1, 2 * dim)
        self.merge = mlp(2 * dim, dim, dim, drop)

    def forward(self, h, ei, etype, phi_e, phi_0):
        s, d = ei
        n = h.size(0)
        q = self.q(torch.cat([h, phi_0.expand(n, -1)], -1)).view(n, self.h, self.dk)
        k, v = self.kv(torch.cat([h[s], phi_e, etype.float().unsqueeze(-1)], -1)).chunk(2, -1)
        k, v = k.view(-1, self.h, self.dk), v.view(-1, self.h, self.dk)
        a = seg_softmax((q[d] * k).sum(-1) / math.sqrt(self.dk), d, num_nodes=n)
        agg = torch.zeros(n, self.h, self.dk, device=h.device).index_add_(0, d, a.unsqueeze(-1) * v).view(n, -1)
        return F.relu(self.merge(torch.cat([h, agg], -1)))


class TGAT(nn.Module):
    def __init__(self, in_dim, dim=64, tdim=16, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.te = TimeEnc(tdim)
        self.layers = nn.ModuleList([TemporalAttn(dim, tdim, drop=drop) for _ in range(layers)])
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, data, x=None):
        h = F.relu(self.inp(data.x if x is None else x))
        phi_e = self.te(data.edt)
        phi_0 = self.te(torch.zeros(1))
        for l in self.layers:
            h = self.drop(l(h, data.ei_dir, data.etype, phi_e, phi_0))
        return {"logit": self.out(h).squeeze(-1)}


class TGN(nn.Module):
    """TGN (Rossi et al., 2020) with GRU memory, identity message function, mean
    message aggregator and a graph-attention embedding module.  Events (edges)
    are processed in chronological batches, one batch per timestamp; embeddings
    of the nodes active at a timestamp are computed from the memory updated with
    the events of that timestamp."""

    def __init__(self, in_dim, dim=64, mem=64, tdim=16, drop=0.2):
        super().__init__()
        self.mem_dim = mem
        self.te = TimeEnc(tdim)
        self.msg_dim = 2 * mem + tdim + in_dim
        self.gru = nn.GRUCell(self.msg_dim, mem)
        self.inp = nn.Linear(in_dim, dim)
        self.mem_proj = nn.Linear(mem, dim)
        self.emb = TemporalAttn(dim, tdim, drop=drop)
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)

    def _steps(self, data):
        key = (data.num_nodes, data.ei_dir.size(1))
        if getattr(self, "_key", None) != key:
            s, d = data.ei_dir
            # events are processed in batches of equal timestamp (Elliptic) or of
            # ``tgn_bucket`` time units (continuous-time data)
            bucket = getattr(data, "tgn_bucket", None)
            et = data.edge_t if bucket is None else torch.floor(data.edge_t / bucket)
            steps = []
            for t in torch.unique(et):
                em = torch.nonzero(et == t).squeeze(-1)
                nodes = torch.unique(torch.cat([s[em], d[em]]))
                loc = torch.full((data.num_nodes,), -1, dtype=torch.long)
                loc[nodes] = torch.arange(len(nodes))
                steps.append((data.edge_t[em].max(), nodes, em, loc[s[em]], loc[d[em]]))
            self._key, self._cache = key, steps
        return self._cache

    def forward(self, data, x=None):
        x = data.x if x is None else x
        n = x.size(0)
        memory = torch.zeros(n, self.mem_dim)
        last = torch.zeros(n)
        h_all = F.relu(self.inp(x))  # nodes without events: feature embedding only
        phi0 = self.te(torch.zeros(1))
        for t, nodes, em, ls, ld in self._steps(data):
            m = len(nodes)
            xs, mem = x[nodes], memory[nodes]
            msg = torch.cat([mem[ld], mem[ls], self.te(t - last[nodes][ld]), xs[ls]], -1)
            agg = torch.zeros(m, self.msg_dim).index_add_(0, ld, msg)
            cnt = torch.bincount(ld, minlength=m).float()
            upd = cnt > 0
            new_mem = mem.clone()
            new_mem[upd] = self.gru(agg[upd] / cnt[upd].unsqueeze(-1), mem[upd])
            memory = memory.index_put((nodes,), new_mem)
            last = last.index_put((nodes,), torch.where(upd, t, last[nodes]))
            h = F.relu(self.inp(xs) + self.mem_proj(new_mem))
            h = self.emb(h, torch.stack([ls, ld]), data.etype[em], self.te(data.edt[em]), phi0)
            h_all = h_all.index_put((nodes,), h)
        return {"logit": self.out(self.drop(h_all)).squeeze(-1)}


class APAN(nn.Module):
    """APAN (Wang et al., 2021): each node keeps a mailbox of the K most recent
    mails; its embedding is the attention of its own state over the mailbox.
    After an interaction, mails [z_src || z_dst || Phi(dt)] are propagated to the
    1-hop neighbours.  On snapshot data every timestamp is processed with one
    propagation round followed by a second encoding pass."""

    def __init__(self, in_dim, dim=64, tdim=16, K=10, drop=0.2):
        super().__init__()
        self.K = K
        self.te = TimeEnc(tdim)
        self.inp = nn.Linear(in_dim, dim)
        self.mail = nn.Linear(2 * dim + tdim, dim)
        self.att = nn.MultiheadAttention(dim, 2, dropout=drop, batch_first=True)
        self.ffn = mlp(2 * dim, dim, dim, drop)
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)

    def encode(self, z, box, box_mask):
        empty = ~box_mask.any(1)
        bm = box_mask.clone()
        bm[empty, 0] = True
        a, _ = self.att(z.unsqueeze(1), box, box, key_padding_mask=~bm)
        a = torch.where(empty.unsqueeze(-1), torch.zeros_like(a[:, 0]), a[:, 0])
        return F.relu(self.ffn(torch.cat([z, a], -1)))

    def forward(self, data, x=None):
        x = data.x if x is None else x
        n = x.size(0)
        z0 = F.relu(self.inp(x))
        s, d = data.ei_dir
        dim = z0.size(1)
        # mailbox slot of every edge: rank among incoming edges (most recent first)
        from .objective import edge_rank
        rank = edge_rank(data.edge_t, d, n)
        keep = rank < self.K
        box = torch.zeros(n, self.K, dim)
        box_mask = torch.zeros(n, self.K, dtype=torch.bool)
        z = self.encode(z0, box, box_mask)
        mails = self.mail(torch.cat([z[s], z[d], self.te(data.edt)], -1))
        box = box.index_put((d[keep], rank[keep]), mails[keep])
        box_mask = box_mask.index_put((d[keep], rank[keep]), torch.ones(int(keep.sum()), dtype=torch.bool))
        z = self.encode(z0, box, box_mask)
        return {"logit": self.out(self.drop(z)).squeeze(-1)}


class EvolveGCN(nn.Module):
    """EvolveGCN-O (Pareja et al., 2020): the GCN weight matrix of each layer is
    the hidden state of an LSTM that is stepped once per snapshot."""

    def __init__(self, in_dim, dim=64, drop=0.2):
        super().__init__()
        self.W0 = nn.ParameterList([nn.Parameter(torch.randn(in_dim, dim) / math.sqrt(in_dim)),
                                    nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))])
        self.lstm = nn.ModuleList([nn.LSTMCell(dim, dim), nn.LSTMCell(dim, dim)])
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self._cache, self._cache_n = None, None

    def snapshots(self, data):
        if self._cache is None or self._cache_n != data.num_nodes:
            self._cache_n = data.num_nodes
            from .graph import gcn_norm_adj
            snaps = []
            steps = getattr(data, "all_steps", None) or torch.unique(data.t).tolist()
            for t in steps:
                nodes = torch.nonzero(data.t == t).squeeze(-1)
                remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
                remap[nodes] = torch.arange(len(nodes))
                s, d = data.ei_und
                em = (remap[s] >= 0) & (remap[d] >= 0)
                A = gcn_norm_adj(torch.stack([remap[s[em]], remap[d[em]]]), len(nodes))
                snaps.append((nodes, A))
            self._cache = snaps
        return self._cache

    def forward(self, data, x=None):
        x = data.x if x is None else x
        logits = torch.zeros(data.num_nodes)
        W = [w for w in self.W0]
        state = [(w, torch.zeros_like(w)) for w in W]
        for nodes, A in self.snapshots(data):
            new_state = []
            for i, cell in enumerate(self.lstm):
                # rows of the weight matrix are treated as the LSTM batch
                hW, cW = cell(state[i][0], state[i])
                new_state.append((hW, cW))
            state = new_state
            if len(nodes) == 0:
                continue
            h = x[nodes]
            for i in range(2):
                # LSTM outputs lie in (-1, 1); scale by fan-in as in standard initialisation
                W = state[i][0] / math.sqrt(state[i][0].size(0))
                h = self.drop(F.relu(torch.sparse.mm(A, h @ W)))
            logits = logits.index_put((nodes,), self.out(h).squeeze(-1))
        return {"logit": logits}


# -------------------------------------------------------- fraud-specific GNNs
class CAREGNN(nn.Module):
    """CARE-GNN (Dou et al., 2020), single relation: label-aware similarity
    measure, top-p neighbour selection whose p is adapted by the RL rule of the
    paper (p +/- 0.02 when the average neighbour distance falls/rises between
    epochs), and inter-layer aggregation [h_v || mean of selected neighbours]."""

    def __init__(self, in_dim, dim=64, drop=0.2, lam=2.0):
        super().__init__()
        self.sim = nn.ModuleList([nn.Linear(in_dim, 1), nn.Linear(dim, 1)])
        self.W = nn.ModuleList([nn.Linear(2 * in_dim, dim), nn.Linear(2 * dim, dim)])
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self.p = [0.5, 0.5]
        self.prev_dist = [None, None]
        self.lam = lam

    def forward(self, data, x=None):
        h = data.x if x is None else x
        n = h.size(0)
        s, d = data.ei_und
        aux = 0
        for l in range(2):
            score = self.sim[l](h).squeeze(-1)
            prob = torch.sigmoid(score)
            if self.training:
                aux = aux + F.binary_cross_entropy_with_logits(score[data.train_mask], data.y[data.train_mask].float())
            dist = (prob[s] - prob[d]).abs().detach()
            from .objective import edge_rank
            rank = edge_rank(-dist, d, n)
            deg = torch.bincount(d, minlength=n)
            keep = rank.float() < torch.ceil(self.p[l] * deg[d].float())
            if self.training:
                avg = dist[keep].mean().item()
                if self.prev_dist[l] is not None:
                    self.p[l] = float(min(1.0, max(0.1, self.p[l] + (0.02 if avg < self.prev_dist[l] else -0.02))))
                self.prev_dist[l] = avg
            agg = mean_agg(h, (s[keep], d[keep]), n)
            h = self.drop(F.relu(self.W[l](torch.cat([h, agg], -1))))
        return {"logit": self.out(h).squeeze(-1), "aux": self.lam * aux}


class PCGNN(nn.Module):
    """PC-GNN (Liu et al., 2021): label-balanced 'pick' sampler for the training
    nodes of each epoch, 'choose' neighbour filtering by the distance of a
    label-aware predictor, and over-sampling of same-class (minority) neighbours
    for minority training nodes."""

    def __init__(self, in_dim, dim=64, drop=0.2, rho=0.5, lam=2.0):
        super().__init__()
        self.dist = nn.Linear(in_dim, 1)
        self.W = nn.ModuleList([nn.Linear(2 * in_dim, dim), nn.Linear(2 * dim, dim)])
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self.rho, self.lam = rho, lam

    def pick(self, data, mask):
        """Label-balanced sampling: P(v) proportional to deg(v) / |class(v)|."""
        idx = torch.nonzero(mask).squeeze(-1)
        y = data.y[idx]
        deg = torch.bincount(data.ei_und[1], minlength=data.num_nodes)[idx].float() + 1
        cls = torch.bincount(y, minlength=2).float()
        w = deg / cls[y]
        pos = int((y == 1).sum())
        sel = torch.multinomial(w, min(len(idx), 4 * pos), replacement=False)
        m = torch.zeros_like(mask)
        m[idx[sel]] = True
        return m

    def forward(self, data, x=None):
        h = data.x if x is None else x
        n = h.size(0)
        s, d = data.ei_und
        score = self.dist(h).squeeze(-1)
        prob = torch.sigmoid(score)
        aux = F.binary_cross_entropy_with_logits(score[data.train_mask], data.y[data.train_mask].float()) if self.training else 0
        keep = (prob[s] - prob[d]).abs().detach() < self.rho
        es, ed = s[keep], d[keep]
        if self.training:
            # choose: add same-class minority neighbours for minority training nodes
            pos = torch.nonzero(data.train_mask & (data.y == 1)).squeeze(-1)
            if len(pos) > 1:
                extra_src = pos[torch.randint(len(pos), (len(pos) * 2,))]
                extra_dst = pos.repeat(2)
                es, ed = torch.cat([es, extra_src]), torch.cat([ed, extra_dst])
        for l in range(2):
            agg = mean_agg(h, (es, ed), n)
            h = self.drop(F.relu(self.W[l](torch.cat([h, agg], -1))))
        return {"logit": self.out(h).squeeze(-1), "aux": self.lam * aux}


class GAS(nn.Module):
    """GAS (Li et al., 2019) adapted to a homogeneous graph: GCN aggregation on
    the transaction graph concatenated with GCN aggregation on a k-nearest-
    neighbour feature-similarity graph built within each time step."""

    def __init__(self, in_dim, dim=64, drop=0.2):
        super().__init__()
        self.g1 = nn.ModuleList([GCNConv(in_dim, dim), GCNConv(dim, dim)])
        self.g2 = nn.ModuleList([GCNConv(in_dim, dim), GCNConv(dim, dim)])
        self.out = mlp(2 * dim, dim, 1, drop)
        self.drop = drop

    @staticmethod
    def knn_graph(data, k=5):
        rows, cols = [], []
        for t in torch.unique(data.t):
            nodes = torch.nonzero(data.t == t).squeeze(-1)
            z = F.normalize(data.x[nodes], dim=1)
            sim = z @ z.T
            sim.fill_diagonal_(-2)
            nb = sim.topk(k, dim=1).indices
            rows.append(nodes[nb].flatten())
            cols.append(nodes.repeat_interleave(k))
        ei = torch.stack([torch.cat(rows), torch.cat(cols)])
        from torch_geometric.utils import to_undirected
        return to_undirected(ei)

    def forward(self, data, x=None):
        x = data.x if x is None else x
        if not hasattr(data, "ei_knn"):
            data.ei_knn = self.knn_graph(data)
        a, b = x, x
        for c1, c2 in zip(self.g1, self.g2):
            a = F.dropout(F.relu(c1(a, data.ei_und)), self.drop, self.training)
            b = F.dropout(F.relu(c2(b, data.ei_knn)), self.drop, self.training)
        return {"logit": self.out(torch.cat([a, b], -1)).squeeze(-1)}


class FRAUDRE(nn.Module):
    """FRAUDRE (Zhang et al., 2021): fraud-aware graph convolution combining the
    node, the mean of its neighbours and the mean node-neighbour difference
    (to cope with heterophily), combination of intermediate layer outputs, and
    an imbalance-oriented cost-sensitive classifier."""

    def __init__(self, in_dim, dim=64, drop=0.2):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(3 * in_dim, dim), nn.Linear(3 * dim, dim)])
        self.out = nn.Linear(in_dim + 2 * dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, data, x=None):
        h = data.x if x is None else x
        n = h.size(0)
        s, d = data.ei_und
        outs = [h]
        for W in self.W:
            nb = mean_agg(h, (s, d), n)
            diff = mean_agg(h, (s, d), n) - h
            h = self.drop(F.relu(W(torch.cat([h, nb, diff.abs()], -1))))
            outs.append(h)
        return {"logit": self.out(torch.cat(outs, -1)).squeeze(-1)}


# ------------------------------------------------- self-explaining baseline
class SEFraud(nn.Module):
    """SEFraud (Li et al., 2024) on a homogeneous graph: a feature-mask generator
    and an edge-mask generator feed a graph-transformer encoder; a triplet loss
    pulls the embedding of the masked graph towards the original embedding and
    pushes the embedding of the complementary graph away."""

    def __init__(self, in_dim, dim=64, heads=4, drop=0.2, margin=1.0, lam_tri=0.5, lam_sp=0.01):
        super().__init__()
        self.g_feat = mlp(in_dim, dim, in_dim)
        self.inp = nn.Linear(in_dim, dim)
        self.g_edge = mlp(2 * dim, dim, 1)
        self.h, self.dk = heads, dim // heads
        self.q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.k = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.v = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.out = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self.margin, self.lam_tri, self.lam_sp = margin, lam_tri, lam_sp

    def encode(self, x, m_feat, ei, m_edge):
        n = x.size(0)
        s, d = ei
        h = F.relu(self.inp(x * m_feat))
        for l in range(2):
            q = self.q[l](h).view(n, self.h, self.dk)
            k = self.k[l](h)[s].view(-1, self.h, self.dk)
            v = self.v[l](h)[s].view(-1, self.h, self.dk)
            a = seg_softmax((q[d] * k).sum(-1) / math.sqrt(self.dk), d, num_nodes=n)
            msg = (a * m_edge.unsqueeze(-1)).unsqueeze(-1) * v
            agg = torch.zeros(n, self.h, self.dk).index_add_(0, d, msg).view(n, -1)
            h = self.norm[l](h + self.drop(agg))
        return h

    def masks(self, x, ei):
        m_feat = torch.sigmoid(self.g_feat(x) + 2.0)
        h = F.relu(self.inp(x))
        s, d = ei
        m_edge = torch.sigmoid(self.g_edge(torch.cat([h[d], h[s]], -1))).squeeze(-1)
        return m_feat, m_edge

    def forward(self, data, x=None, override=None):
        x = data.x if x is None else x
        ei = data.ei_dir
        m_feat, m_edge = self.masks(x, ei)
        if override is not None:
            m_feat = override.get("feat", m_feat)
            m_edge = override.get("edge", m_edge)
        z = self.encode(x, m_feat, ei, m_edge)
        out = {"logit": self.out(z).squeeze(-1), "feat": m_feat, "edge": m_edge}
        if self.training and override is None:
            ones_f, ones_e = torch.ones_like(m_feat), torch.ones_like(m_edge)
            z_full = self.encode(x, ones_f, ei, ones_e)
            z_neg = self.encode(x, 1 - m_feat, ei, 1 - m_edge)
            sel = data.train_mask
            tri = F.triplet_margin_loss(z_full[sel], z[sel], z_neg[sel], margin=self.margin)
            out["aux"] = self.lam_tri * tri + self.lam_sp * (m_feat.mean() + m_edge.mean())
        return out
