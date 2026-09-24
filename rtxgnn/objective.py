"""Training objective of RTXGNN.

L = L_pred + w(e) * (lam_suf * L_suf + lam_nec * L_nec + lam_sp * L_sp)

* L_pred: class-balanced binary cross-entropy on labelled training nodes
  (focal loss is available as an ablation).
* L_suf (sufficiency): the prediction obtained when only the top-k explanation
  is kept (hard top-k masks, straight-through gradients) should match the
  model's own prediction (KL divergence).
* L_nec (necessity): the prediction obtained from the complement of the
  explanation (1 - mask) should be uninformative, i.e. close to the class prior
  (KL(prior || p_complement)).  Together with L_suf this rules out the trivial
  all-ones mask that a sufficiency-only ("masked = full") objective admits.
* L_sp: mean L1 norm of the masks.
* w(e): linear warm-up of the explanation terms (curriculum).

The temporal-consistency loss of the first submission is not used: it
supervised the temporal gate with labels of fraudulent edges.
"""
import torch
import torch.nn.functional as F


def balanced_bce(logit, y, prior):
    """Class-balanced binary cross-entropy: each class contributes half of the loss."""
    w = torch.where(y > 0.5, 0.5 / prior, 0.5 / (1 - prior))
    return (w * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()


def pred_loss(logit, y, prior, cfg=None):
    cfg = cfg or {}
    if cfg.get("loss", "bce") == "focal":
        return focal_loss(logit, y, cfg.get("alpha", 0.75), cfg.get("gamma", 2.0))
    return balanced_bce(logit, y, prior)


def focal_loss(logit, y, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


def bern_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def hard_topk_rows(m, k):
    """Keep the k largest entries of each row (straight-through estimator)."""
    idx = m.topk(k, dim=1).indices
    hard = torch.zeros_like(m).scatter_(1, idx, 1.0)
    return hard + m - m.detach()


def edge_rank(score, dst, n):
    """Rank (0 = highest score) of every edge among the incoming edges of its
    receiving node."""
    order = torch.argsort(score, descending=True)
    idx = order[torch.sort(dst[order], stable=True).indices]  # grouped by dst, score desc
    cnt = torch.bincount(dst, minlength=n)
    first = torch.cumsum(cnt, 0) - cnt
    rank = torch.empty_like(dst)
    rank[idx] = torch.arange(len(idx), device=dst.device) - first[dst[idx]]
    return rank


def hard_topk_edges(m, dst, n, k):
    """Keep, for every receiving node, its k highest-scoring incoming edges."""
    hard = (edge_rank(m.detach(), dst, n) < k).float()
    return hard + m - m.detach()


def explanation_views(out, data, k_feat, k_edge):
    """Masks restricted to the explanation (top-k) and to its complement."""
    n = data.x.size(0)
    feat = hard_topk_rows(out["feat"], k_feat)
    layers_keep, layers_drop = [], []
    for m in out["layers"]:
        e = hard_topk_edges(m["edge"], data.ei_dir[1], n, k_edge)
        layers_keep.append({"edge": e, "node": m["node"]})
        layers_drop.append({"edge": 1 - e, "node": m["node"]})
    return {"feat": feat, "layers": layers_keep}, {"feat": 1 - feat, "layers": layers_drop}


def rtx_loss(model, data, out, mask, cfg, epoch, prior):
    y = data.y[mask].float()
    loss = pred_loss(out["logit"][mask], y, prior, cfg)
    parts = {"pred": loss.item()}
    if not model.use_masks:
        return loss, parts
    w = min(1.0, epoch / max(1, cfg["warmup"])) if cfg["curriculum"] else 1.0
    p = torch.sigmoid(out["logit"]).detach()
    need_views = cfg["lam_suf"] > 0 or cfg["lam_nec"] > 0
    if need_views and w > 0:
        keep, drop = explanation_views(out, data, cfg["k_feat"], cfg["k_edge"])
        sel = mask
        if cfg["lam_suf"] > 0:
            p_keep = torch.sigmoid(model(data, override=keep)["logit"])
            l_suf = bern_kl(p[sel], p_keep[sel]).mean()
            loss = loss + w * cfg["lam_suf"] * l_suf
            parts["suf"] = l_suf.item()
        if cfg["lam_nec"] > 0:
            p_drop = torch.sigmoid(model(data, override=drop)["logit"])
            l_nec = bern_kl(torch.full_like(p_drop[sel], prior), p_drop[sel]).mean()
            loss = loss + w * cfg["lam_nec"] * l_nec
            parts["nec"] = l_nec.item()
    if cfg["lam_sp"] > 0:
        sp = out["feat"].mean() + sum(m["edge"].mean() + m["node"].mean() for m in out["layers"]) / len(out["layers"])
        loss = loss + w * cfg["lam_sp"] * sp
        parts["sp"] = sp.item()
    return loss, parts
