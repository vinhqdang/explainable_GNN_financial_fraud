import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

from rtxgnn.data import load_elliptic, quantile_normalize, TRAIN_STEPS  # noqa: E402
from rtxgnn.graph import prepare, split_regions  # noqa: E402
from rtxgnn.baselines import GAS  # noqa: E402

TABULAR = {"lr", "rf", "xgb"}


def elliptic(k=2):
    d = prepare(load_elliptic())
    # quantile normalisation fitted on the transactions of the training steps
    # (tree ensembles use the raw features, which they do not need normalised)
    d.x_raw = d.x.clone()
    d.x = quantile_normalize(d.x, d.t <= TRAIN_STEPS[1])
    d.ei_knn = GAS.knn_graph(d)
    return d


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_keys(path, keys=("model", "seed")):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out.add(tuple(r.get(k) for k in keys))
    return out


def n_params(model):
    return sum(p.numel() for p in model.parameters())


class slot:
    """At most ``n`` memory-heavy trainings at a time across processes (file locks)."""

    def __init__(self, n=2, prefix="/tmp/rtx_slot"):
        self.n, self.prefix, self.f = n, prefix, None

    def __enter__(self):
        import fcntl
        while True:
            for i in range(self.n):
                f = open(f"{self.prefix}{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except OSError:
                    f.close()
            time.sleep(10)

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
