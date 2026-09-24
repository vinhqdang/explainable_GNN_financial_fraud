"""End-to-end CPU latency of scoring + explaining incoming transactions.

For every request (a batch of B target transactions) we time:
  sample   - 2-hop neighbourhood sampling from the in-memory CSR adjacency
  build    - gathering features / edge attributes into tensors
  forward  - model forward pass (no gradient)
  explain  - reading top-k features / edges from the masks (RTXGNN, SEFraud)
             or running a post-hoc explainer (GNNExplainer, 100 iterations)
and report p50/p95/p99 latency per request, throughput and peak memory.
"""
import argparse
import json
import os
import platform
import resource
import time

import numpy as np
import torch

from common import RESULTS, elliptic
from rtxgnn.graph import split_regions
from rtxgnn.train import build
from rtxgnn.serving import CSR, sample_batch, explain_targets
from rtxgnn.explain import attr_gnnexplainer


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def bench(name, model, dev, targets, B, gen, posthoc=False, n_req=300):
    csr = CSR(dev.ei_dir, dev.num_nodes)
    times = []
    for r in range(n_req):
        tg = targets[torch.randint(len(targets), (B,), generator=gen)]
        t0 = time.perf_counter()
        sub, pos = sample_batch(dev, csr, tg, hops=2, fanout=25, gen=gen)
        t1 = time.perf_counter()
        # tensors are already materialised by sample_batch; build = dtype/contiguity checks
        sub.x = sub.x.contiguous()
        t2 = time.perf_counter()
        with torch.no_grad():
            out = model(sub)
        t3 = time.perf_counter()
        if name == "rtxgnn":
            explain_targets(out, sub, pos)
        elif name == "sefraud":
            out["feat"][pos].topk(5, dim=1)
        if posthoc:
            p = torch.sigmoid(out["logit"][pos]).detach()
            attr_gnnexplainer(model, sub, pos, (p > 0.5).long(), epochs=100)
        t4 = time.perf_counter()
        if r >= 10:  # warm-up
            times.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    a = np.array(times) * 1000
    tot = a.sum(1)
    return dict(model=name + ("+gnnexplainer" if posthoc else ""), batch=B,
                p50=float(np.percentile(tot, 50)), p95=float(np.percentile(tot, 95)),
                p99=float(np.percentile(tot, 99)), mean=float(tot.mean()),
                sample_ms=float(a[:, 0].mean()), build_ms=float(a[:, 1].mean()),
                forward_ms=float(a[:, 2].mean()), explain_ms=float(a[:, 3].mean()),
                throughput_tps=float(B / (tot.mean() / 1000)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batches", default="1,32,256")
    ap.add_argument("--models", default="mlp,gcn,gat,sefraud,rtxgnn")
    ap.add_argument("--nreq", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = elliptic()
    dtr, dev = split_regions(d)
    targets = torch.nonzero(dev.test_mask).squeeze(-1)
    out_path = os.path.join(RESULTS, "latency.jsonl")
    for name in a.models.split(","):
        model = build(name, d.x.size(1))
        ck = os.path.join(RESULTS, "ckpt", f"{name}_s0.pt")
        if os.path.exists(ck):
            model.load_state_dict(torch.load(ck))
        model.eval()
        gen = torch.Generator().manual_seed(0)
        for B in [int(b) for b in a.batches.split(",")]:
            rows = [bench(name, model, dev, targets, B, gen, n_req=a.nreq)]
            if name in ("gat", "rtxgnn") and B == 1:
                rows.append(bench(name, model, dev, targets, B, gen, posthoc=True, n_req=60))
            for row in rows:
                row.update(threads=a.threads, cpu=cpu_name(),
                           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
