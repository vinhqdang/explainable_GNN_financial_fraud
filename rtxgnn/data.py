"""Dataset loading.

Elliptic is read from the raw CSV files so that the official time step of every
transaction (column 1 of ``elliptic_txs_features.csv``) is preserved. Labels:
0 = licit, 1 = illicit, -1 = unknown.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

DATA_ROOT = os.environ.get("RTX_DATA", "/tmp/data")

TRAIN_STEPS = (1, 30)
VAL_STEPS = (31, 34)
TEST_STEPS = (35, 49)


def _download_elliptic(root):
    from torch_geometric.datasets import EllipticBitcoinDataset
    EllipticBitcoinDataset(root=root)


def load_elliptic(root=None, cache=True):
    root = root or os.path.join(DATA_ROOT, "Elliptic")
    cache_file = os.path.join(root, "rtx_elliptic.pt")
    if cache and os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)
    raw = os.path.join(root, "raw")
    if not os.path.exists(os.path.join(raw, "elliptic_txs_features.csv")):
        _download_elliptic(root)
    feat = pd.read_csv(os.path.join(raw, "elliptic_txs_features.csv"), header=None)
    cls = pd.read_csv(os.path.join(raw, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(raw, "elliptic_txs_edgelist.csv"))

    tx_ids = feat[0].values
    idx = {t: i for i, t in enumerate(tx_ids)}
    step = torch.tensor(feat[1].values, dtype=torch.long)
    # columns 2..166: 93 remaining local features + 72 aggregated features
    x = torch.tensor(feat.loc[:, 2:].values, dtype=torch.float)
    lab = cls.set_index("txId").loc[tx_ids, "class"].astype(str).values
    y = torch.full((len(tx_ids),), -1, dtype=torch.long)
    y[torch.tensor(lab == "1")] = 1
    y[torch.tensor(lab == "2")] = 0
    src = torch.tensor(edges["txId1"].map(idx).values)
    dst = torch.tensor(edges["txId2"].map(idx).values)
    edge_index = torch.stack([src, dst])  # directed payment flow src -> dst

    data = Data(x=x, y=y, edge_index=edge_index, time_step=step)
    lab_mask = y >= 0
    data.train_mask = lab_mask & (step >= TRAIN_STEPS[0]) & (step <= TRAIN_STEPS[1])
    data.val_mask = lab_mask & (step >= VAL_STEPS[0]) & (step <= VAL_STEPS[1])
    data.test_mask = lab_mask & (step >= TEST_STEPS[0]) & (step <= TEST_STEPS[1])
    data.num_local = 93
    if cache:
        torch.save(data, cache_file)
    return data


def describe(data):
    rows = []
    for t in range(1, 50):
        m = data.time_step == t
        rows.append({
            "step": t,
            "nodes": int(m.sum()),
            "licit": int(((data.y == 0) & m).sum()),
            "illicit": int(((data.y == 1) & m).sum()),
            "unknown": int(((data.y == -1) & m).sum()),
        })
    return pd.DataFrame(rows)


def load_tfinance(root=None, split=0, max_neighbors=10, seed=0):
    """T-Finance (Tang et al., 2022) as distributed by GADBench (Tang et al.,
    2023): 39,357 accounts of a financial platform, 10 features, fraud label.
    ``split`` selects one of GADBench's fixed splits (0-9: 40/20/40 supervised).
    Graph models see at most ``max_neighbors`` uniformly sampled neighbours per
    node (the full graph has 21.2M undirected edges)."""
    root = root or os.path.join(DATA_ROOT, "tfin")
    z = np.load(os.path.join(root, "tfinance.npz"))
    x = torch.tensor(z["x"], dtype=torch.float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    y = torch.tensor(z["y"], dtype=torch.long)
    src, dst = torch.tensor(z["src"]), torch.tensor(z["dst"])
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # the stored graph is symmetric; keep one direction then sample per receiver
    one = src < dst
    src, dst = src[one], dst[one]
    g = torch.Generator().manual_seed(seed)
    s2 = torch.cat([src, dst]); d2 = torch.cat([dst, src])
    score = torch.rand(len(s2), generator=g)
    from .objective import edge_rank
    r = edge_rank(score, d2, len(y))
    m = r < max_neighbors
    s2, d2 = s2[m], d2[m]
    # store as a single direction per pair (prepare() adds the reverse)
    und = torch.stack([torch.minimum(s2, d2), torch.maximum(s2, d2)])
    und = torch.unique(und, dim=1)
    data = Data(x=x, y=y, edge_index=und, num_nodes=len(y))
    data.time_step = torch.zeros(len(y))
    data.train_mask = torch.tensor(z["train_masks"][:, split]).bool()
    data.val_mask = torch.tensor(z["val_masks"][:, split]).bool()
    data.test_mask = torch.tensor(z["test_masks"][:, split]).bool()
    return data


def quantile_normalize(x, fit_mask, seed=0):
    """Map every feature to a standard normal with a quantile transform fitted
    on ``fit_mask`` rows only (no labels are used)."""
    from sklearn.preprocessing import QuantileTransformer
    q = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200000,
                            random_state=seed).fit(x[fit_mask].cpu().numpy())
    return torch.tensor(q.transform(x.cpu().numpy()), dtype=torch.float)
