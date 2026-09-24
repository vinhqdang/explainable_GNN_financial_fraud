"""Training / evaluation utilities shared by all experiments."""
import copy
import time
import numpy as np
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, precision_recall_curve)

from .objective import pred_loss, rtx_loss
from .model import RTXGNN
from . import baselines as B

DEFAULT_RTX = dict(loss="bce", alpha=0.75, gamma=2.0, lam_suf=0.5, lam_nec=0.5, lam_sp=0.05, k_feat=10, k_edge=2,
                   warmup=20, curriculum=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def best_threshold(p, y):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def metrics(p, y, thr):
    pred = (p >= thr).astype(int)
    out = dict(f1=f1_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0))
    if len(np.unique(y)) > 1:
        out.update(auc=roc_auc_score(y, p), ap=average_precision_score(y, p))
    else:
        out.update(auc=float("nan"), ap=float("nan"))
    return out


def build(name, in_dim, **kw):
    dim = kw.pop("dim", 64)
    if name.startswith("rtxgnn"):
        return RTXGNN(in_dim, dim=dim, **kw)
    table = {
        "mlp": lambda: B.MLP(in_dim, dim),
        "gcn": lambda: B.PygGNN("gcn", in_dim, dim),
        "sage": lambda: B.PygGNN("sage", in_dim, dim),
        "gat": lambda: B.PygGNN("gat", in_dim, dim),
        "evolvegcn": lambda: B.EvolveGCN(in_dim, dim),
        "tgat": lambda: B.TGAT(in_dim, dim),
        "tgn": lambda: B.TGN(in_dim, dim),
        "apan": lambda: B.APAN(in_dim, dim),
        "caregnn": lambda: B.CAREGNN(in_dim, dim),
        "pcgnn": lambda: B.PCGNN(in_dim, dim),
        "gas": lambda: B.GAS(in_dim, dim),
        "fraudre": lambda: B.FRAUDRE(in_dim, dim),
        "sefraud": lambda: B.SEFraud(in_dim, dim),
    }
    return table[name]()


def train_model(model, data, deval, epochs=300, patience=40, lr=5e-3, wd=1e-5,
                rtx_cfg=None, verbose=False, log_every=20, eval_every=1):
    """Full-batch training on the training subgraph ``data`` with early stopping
    on the validation average precision computed on ``deval``.

    Returns the probabilities of all nodes of ``deval`` from the best epoch, the
    decision threshold chosen on the validation nodes, and a training history."""
    device = data.x.device
    model.to(device)
    with torch.device(device):
        return _train(model, data, deval, epochs, patience, lr, wd, rtx_cfg, verbose, log_every, eval_every)


def _train(model, data, deval, epochs, patience, lr, wd, rtx_cfg, verbose, log_every, eval_every):
    train_mask = data.train_mask
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    y_val = deval.y[deval.val_mask].cpu().numpy()
    prior = float(data.y[train_mask].float().mean())
    best, best_state, bad, hist = -1, None, 0, []
    cfg = None
    if isinstance(model, RTXGNN):
        cfg = dict(DEFAULT_RTX, **(rtx_cfg or {}))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        mask = train_mask
        if hasattr(model, "pick"):
            mask = model.pick(data, train_mask)
        out = model(data)
        if cfg is not None:
            loss, parts = rtx_loss(model, data, out, mask, cfg, ep, prior)
        else:
            loss = pred_loss(out["logit"][mask], data.y[mask].float(), prior)
            if "aux" in out:
                loss = loss + out["aux"]
            parts = {}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % eval_every and ep != epochs - 1:
            continue
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(deval)["logit"]).cpu().numpy()
        ap = average_precision_score(y_val, p[deval.val_mask.cpu().numpy()])
        hist.append(dict(epoch=ep, loss=float(loss.detach()), val_ap=ap, **parts))
        if verbose and ep % log_every == 0:
            print(f"ep {ep} loss {loss.item():.4f} val_ap {ap:.4f} {parts} {time.time()-t0:.0f}s", flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = copy.deepcopy(model.state_dict())
            best_p = p
        else:
            bad += eval_every
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    thr = best_threshold(best_p[deval.val_mask.cpu().numpy()], y_val)
    return best_p, thr, hist


def evaluate(p, data, thr, mask=None):
    mask = data.test_mask if mask is None else mask
    return metrics(p[mask.cpu().numpy()], data.y[mask].cpu().numpy(), thr)


def per_step(p, data, thr, steps):
    rows = []
    for t in steps:
        m = data.test_mask & (data.t == t)
        y = data.y[m].cpu().numpy()
        r = metrics(p[m.cpu().numpy()], y, thr)
        r.update(step=int(t), n=int(m.sum()), illicit=int(y.sum()))
        rows.append(r)
    return rows
