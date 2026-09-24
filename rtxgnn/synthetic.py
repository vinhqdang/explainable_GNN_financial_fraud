"""Controlled synthetic benchmark with temporal money-laundering rings.

Generation process (all random draws use ``numpy.random.default_rng(seed)``):

1. Accounts and relationships. ``n`` accounts; a Barabasi-Albert graph with
   ``m_ba`` edges per new node defines which pairs of accounts transact. Each
   relationship (u, v) gets a direction at random and a Poisson number of
   transactions with mean ``tx_per_pair``; timestamps are uniform on
   [0, horizon] days and amounts are log-normal(mu=5, sigma=1).
2. Laundering rings (label 1). ``n_rings`` cycles of length L ~ U{4..8}
   are laid on distinct accounts. A ring is executed at a random start time
   s ~ U[0, horizon - burst]: hop k happens at s + k * burst / L (+ jitter), so
   the whole cycle is completed within ``burst`` days; the amount decays by a
   1-3% fee per hop. Ring members keep their ordinary background transactions
   (camouflage).
3. Decoy cycles (label 0). ``n_decoys`` cycles with the same length
   distribution and amounts are laid on other accounts but their hops are
   spread uniformly over the whole horizon. Cycles alone therefore do not
   identify laundering: the timing of consecutive hops does.
4. Node features (16): in/out transaction counts, mean and std of incoming
   and outgoing amounts, number of distinct counterparties (7 behavioural
   statistics computed from all transactions of the account, log-scaled and
   standardised), one constant, and 8 i.i.d. N(0, 1) noise features.
   ``shift`` adds a constant to the 8 noise features of laundering accounts
   (default 0, i.e. no feature signal).
5. ``label_noise`` flips that fraction of labels uniformly at random.

Ground truth for explanations: for every ring account, its incoming and
outgoing ring transactions.
All accounts are classified at t = horizon; the split is a stratified random
60/20/20 split of the accounts.
"""
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def generate(n=5000, m_ba=2, tx_per_pair=3.0, horizon=180.0, n_rings=50, n_decoys=50, burst=2.0,
             shift=0.0, label_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m_ba, seed=int(rng.integers(1 << 30)))
    src, dst, ts, amt, gt = [], [], [], [], []
    for u, v in G.edges():
        if rng.random() < 0.5:
            u, v = v, u
        k = rng.poisson(tx_per_pair)
        for _ in range(k):
            src.append(u); dst.append(v); ts.append(rng.uniform(0, horizon))
            amt.append(rng.lognormal(5, 1)); gt.append(0)
    perm = rng.permutation(n)
    y = np.zeros(n, dtype=np.int64)
    pos = 0

    def lay_cycle(timed):
        nonlocal pos
        L = int(rng.integers(4, 9))
        nodes = perm[pos:pos + L]
        pos += L
        a = rng.lognormal(7, 0.5)
        if timed:
            start = rng.uniform(0, horizon - burst)
            times = start + np.arange(L) * burst / L + rng.uniform(0, burst / (4 * L), L)
        else:
            times = np.sort(rng.uniform(0, horizon, L))
        for k in range(L):
            src.append(nodes[k]); dst.append(nodes[(k + 1) % L]); ts.append(times[k])
            amt.append(a); gt.append(1 if timed else 0)
            a *= 1 - rng.uniform(0.01, 0.03)
        return nodes

    for _ in range(n_rings):
        y[lay_cycle(True)] = 1
    for _ in range(n_decoys):
        lay_cycle(False)

    src, dst = np.array(src), np.array(dst)
    ts, amt, gt = np.array(ts), np.array(amt), np.array(gt)
    feats = np.zeros((n, 16))
    for side, idx in ((0, src), (1, dst)):
        cnt = np.bincount(idx, minlength=n)
        s1 = np.bincount(idx, weights=np.log(amt), minlength=n)
        s2 = np.bincount(idx, weights=np.log(amt) ** 2, minlength=n)
        mean = s1 / np.maximum(cnt, 1)
        std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0))
        feats[:, 3 * side:3 * side + 3] = np.stack([np.log1p(cnt), mean, std], 1)
    cp = [set() for _ in range(n)]
    for a_, b_ in zip(src, dst):
        cp[a_].add(b_); cp[b_].add(a_)
    feats[:, 6] = np.log1p([len(c) for c in cp])
    feats[:, :7] = (feats[:, :7] - feats[:, :7].mean(0)) / (feats[:, :7].std(0) + 1e-9)
    feats[:, 7] = 1.0
    feats[:, 8:] = rng.normal(size=(n, 8))
    feats[y == 1, 8:] += shift
    y_obs = y.copy()
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y_obs[flip] = 1 - y_obs[flip]

    # stratified 60/20/20 split
    train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
    for c in (0, 1):
        idx = rng.permutation(np.nonzero(y_obs == c)[0])
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        train[idx[:a]] = True; val[idx[a:b]] = True; test[idx[b:]] = True

    data = Data(x=torch.tensor(feats, dtype=torch.float), y=torch.tensor(y_obs),
                edge_index=torch.tensor(np.stack([src, dst])), num_nodes=n)
    data.edge_time = torch.tensor(ts, dtype=torch.float)
    data.gt_edge_raw = torch.tensor(gt)
    data.y_true = torch.tensor(y)
    data.time_step = torch.full((n,), float(horizon))
    data.train_mask = torch.tensor(train)
    data.val_mask = torch.tensor(val)
    data.test_mask = torch.tensor(test)
    data.amount = torch.tensor(amt, dtype=torch.float)
    return data


def prepare_synthetic(data, windows=(1.0, 7.0, 30.0)):
    """Like :func:`rtxgnn.graph.prepare` but keeps multi-edges and their times."""
    from torch_geometric.utils import to_undirected
    t = data.time_step.float()
    data.t = t
    ei = data.edge_index
    data.ei_dir = torch.cat([ei, ei.flip(0)], 1)
    data.etype = torch.cat([torch.zeros(ei.size(1)), torch.ones(ei.size(1))]).long()
    data.edge_t = torch.cat([data.edge_time, data.edge_time])
    s, d = data.ei_dir
    data.edt = (t[d] - data.edge_t).abs()
    data.gt_edge = torch.cat([data.gt_edge_raw, data.gt_edge_raw])
    data.ei_und = to_undirected(ei, num_nodes=data.num_nodes)
    n = data.num_nodes
    vel = []
    for w in windows:
        m = data.edt < w
        vel.append(torch.log1p(torch.zeros(n).index_add_(0, d[m], torch.ones(int(m.sum())))))
    data.vel = torch.stack(vel, 1)
    data.all_steps = [float(t[0])]
    data.tgn_bucket = 1.0  # TGN processes the events of each day as one batch
    return data
