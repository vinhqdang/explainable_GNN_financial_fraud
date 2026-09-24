"""Graph preprocessing shared by all graph models."""
import torch
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops


def prepare(data, windows=(1.0, 2.0, 4.0)):
    """Attach the tensors used by the models to ``data`` (in place).

    * ``ei_dir``/``etype``: every payment edge in both directions; ``etype`` is 0
      for the original direction (payer -> payee) and 1 for the reverse one.
    * ``edt``: time gap t_dst - t_src of each directed edge (always 0 on
      Elliptic because edges never cross time steps).
    * ``ei_und``: symmetrised edge index for GCN/GraphSAGE/GAT.
    * ``t``: node timestamp (float); ``vel``: log(1+number of incident edges
      whose timestamp lies in (t-w, t]) for each window w.
    """
    t = data.time_step.float() if hasattr(data, "time_step") else data.t.float()
    data.t = t
    ei = data.edge_index
    ei, _ = remove_self_loops(ei)
    src, dst = ei
    data.ei_dir = torch.cat([ei, ei.flip(0)], 1)
    data.etype = torch.cat([torch.zeros(ei.size(1)), torch.ones(ei.size(1))]).long()
    et = getattr(data, "edge_time", None)
    if et is None:
        et = torch.maximum(t[src], t[dst])
    data.edge_t = torch.cat([et, et]).float()
    s, d = data.ei_dir
    # time elapsed between the transaction and the receiving node's timestamp
    data.edt = (t[d] - data.edge_t).abs()
    data.ei_und = to_undirected(ei, num_nodes=data.num_nodes)
    # velocity: number of incident transactions in trailing windows of the node's time
    n = data.num_nodes
    vel = []
    for w in windows:
        gap = t[d] - data.edge_t
        m = (gap >= 0) & (gap < w)
        cnt = torch.zeros(n).index_add_(0, d[m], torch.ones(int(m.sum())))
        vel.append(torch.log1p(cnt))
    data.vel = torch.stack(vel, 1)
    data.all_steps = torch.unique(t).tolist()
    return data


def gcn_norm_adj(edge_index, n):
    """Sparse symmetric-normalised adjacency with self loops (for EvolveGCN)."""
    ei, _ = add_self_loops(edge_index, num_nodes=n)
    deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.size(1)))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei.flip(0), w, (n, n)).coalesce()


NODE_KEYS = ("x", "x_raw", "y", "t", "time_step", "vel", "train_mask", "val_mask", "test_mask", "gt_node")
DIR_KEYS = ("etype", "edt", "edge_t", "gt_edge")


def induce(data, nodes):
    """Induced subgraph on ``nodes`` (sorted LongTensor), keeping all attributes
    produced by :func:`prepare`."""
    from torch_geometric.data import Data
    n = data.num_nodes
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(len(nodes))
    out = Data()
    for k in NODE_KEYS:
        if hasattr(data, k) and getattr(data, k) is not None:
            out[k] = getattr(data, k)[nodes]
    out.num_nodes = len(nodes)
    out.orig_id = nodes
    s, d = data.ei_dir
    em = (remap[s] >= 0) & (remap[d] >= 0)
    out.ei_dir = torch.stack([remap[s[em]], remap[d[em]]])
    for k in DIR_KEYS:
        if hasattr(data, k) and getattr(data, k) is not None:
            out[k] = getattr(data, k)[em]
    for k in ("ei_und", "ei_knn", "edge_index"):
        if hasattr(data, k) and getattr(data, k) is not None:
            s, d = getattr(data, k)
            em = (remap[s] >= 0) & (remap[d] >= 0)
            out[k] = torch.stack([remap[s[em]], remap[d[em]]])
    if hasattr(data, "all_steps"):
        out.all_steps = data.all_steps
    return out


def khop_nodes(data, seeds, k=2):
    from torch_geometric.utils import k_hop_subgraph
    sub, _, _, _ = k_hop_subgraph(seeds, k, data.ei_und, num_nodes=data.num_nodes)
    return torch.sort(sub).values


def split_regions(data, k=2, train_mask=None):
    """Training subgraph (k-hop neighbourhood of the labelled training nodes) and
    evaluation subgraph (k-hop neighbourhood of the validation and test nodes).
    For models with at most k message-passing layers this is exact: every
    prediction depends only on the node's k-hop neighbourhood."""
    train_mask = data.train_mask if train_mask is None else train_mask
    tr = khop_nodes(data, torch.nonzero(train_mask).squeeze(-1), k)
    ev = khop_nodes(data, torch.nonzero(data.val_mask | data.test_mask).squeeze(-1), k)
    dtr, dev = induce(data, tr), induce(data, ev)
    dtr.train_mask = train_mask[tr]
    dtr.val_mask = torch.zeros(len(tr), dtype=torch.bool)
    dtr.test_mask = torch.zeros(len(tr), dtype=torch.bool)
    dev.train_mask = torch.zeros(len(ev), dtype=torch.bool)
    return dtr, dev
