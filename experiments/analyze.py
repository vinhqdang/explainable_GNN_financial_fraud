"""Builds all tables (LaTeX) and the number macros of the manuscript from the
JSONL result files, so that every number in the text comes from a run record."""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

from common import RESULTS, ROOT

OUT = os.path.join(ROOT, "manuscript", "revision1", "tables")
os.makedirs(OUT, exist_ok=True)
MACROS = {}

NAMES = {
    "lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost", "mlp": "MLP",
    "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT",
    "evolvegcn": "EvolveGCN-O", "tgat": "TGAT", "tgn": "TGN", "apan": "APAN",
    "caregnn": "CARE-GNN", "pcgnn": "PC-GNN", "gas": "GAS", "fraudre": "FRAUDRE", "sefraud": "SEFraud",
    "rtxgnn": "RTXGNN (ours)",
}
GROUPS = [("Feature-only models", ["lr", "rf", "xgb", "mlp"]),
          ("General GNNs", ["gcn", "sage", "gat"]),
          ("Temporal GNNs", ["evolvegcn", "tgat", "tgn", "apan"]),
          ("Fraud-specific GNNs", ["caregnn", "pcgnn", "gas", "fraudre", "sefraud"]),
          ("Proposed", ["rtxgnn"])]
GNNS = ["gcn", "sage", "gat", "evolvegcn", "tgat", "tgn", "apan", "caregnn", "pcgnn", "gas", "fraudre", "sefraud"]


def load(name):
    p = os.path.join(RESULTS, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def macro(name, value):
    """LaTeX macro names may only contain letters: digits are spelled out."""
    import re
    name = "".join(DIGITS.get(c, c) for c in name)
    MACROS[re.sub("[^A-Za-z]", "", name)] = value


def ms(v, d=3):
    v = np.asarray(v, float)
    if len(v) == 1 or np.allclose(v, v[0]):
        return f"{v.mean():.{d}f}"
    return f"{v.mean():.{d}f}\\,$\\pm$\\,{v.std(ddof=1):.{d}f}"


def holm(pvals):
    keys = list(pvals)
    order = sorted(keys, key=lambda k: pvals[k])
    m = len(order)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def by_model(rows, key="test"):
    out = defaultdict(dict)
    for r in rows:
        out[r["model"]][r["seed"]] = r[key]
    return out


def paired(a, b, metric):
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s][metric] for s in seeds]); y = np.array([b[s][metric] for s in seeds])
    return x, y


def main_rows():
    evo = load("elliptic_evo.jsonl")
    rows = load("elliptic_main.jsonl")
    seen = {(r["model"], r["seed"]) for r in rows}
    rows += [r for r in load("elliptic_main_gpu.jsonl") if (r["model"], r["seed"]) not in seen]
    if evo:
        rows = [r for r in rows if r["model"] != "evolvegcn"] + evo
    return rows


def main_table():
    rows = main_rows()
    if not rows:
        return
    R = by_model(rows)
    ref = R.get("rtxgnn", {})
    pv = {m: {} for m in ("f1", "ap")}
    for met in ("f1", "ap"):
        raw = {}
        for m in R:
            if m == "rtxgnn" or not ref:
                continue
            x, y = paired(ref, R[m], met)
            if len(x) >= 5 and not np.allclose(x - y, 0):
                raw[m] = wilcoxon(x, y).pvalue
        pv[met] = holm(raw) if raw else {}
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Model & F1 & Precision & Recall & AUC & AP\\\\", "\\midrule"]
    best = {met: max(np.mean([v[met] for v in R[m].values()]) for m in R) for met in ("f1", "precision", "recall", "auc", "ap")}
    for g, ms_ in GROUPS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{g}}}}}\\\\")
        for m in ms_:
            if m not in R:
                continue
            cells = []
            for met in ("f1", "precision", "recall", "auc", "ap"):
                vals = [v[met] for v in R[m].values()]
                c = ms(vals)
                if abs(np.mean(vals) - best[met]) < 1e-9:
                    c = f"\\textbf{{{c}}}"
                if met in pv and m in pv[met]:
                    p = pv[met][m]
                    mean_ref = np.mean([v[met] for v in ref.values()])
                    if p < 0.05:
                        c += "$^{\\dagger}$" if mean_ref > np.mean(vals) else "$^{\\ddagger}$"
                cells.append(c)
            lines.append(f"{NAMES[m]} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "main.tex"), "w").write("\n".join(lines) + "\n")
    # macros
    mean = lambda m, met: np.mean([v[met] for v in R[m].values()])
    std = lambda m, met: np.std([v[met] for v in R[m].values()], ddof=1)
    if "rtxgnn" in R:
        macro("rtxFone", f"{mean('rtxgnn','f1'):.3f}"); macro("rtxFoneSd", f"{std('rtxgnn','f1'):.3f}")
        macro("rtxAP", f"{mean('rtxgnn','ap'):.3f}"); macro("rtxAUC", f"{mean('rtxgnn','auc'):.3f}")
        macro("rtxPrec", f"{mean('rtxgnn','precision'):.3f}"); macro("rtxRec", f"{mean('rtxgnn','recall'):.3f}")
        macro("rtxMissPct", f"{100*(1-mean('rtxgnn','recall')):.0f}")
        macro("rtxFDRPct", f"{100*(1-mean('rtxgnn','precision')):.0f}")
        macro("nSeedsMain", str(len(R["rtxgnn"])))
    for m in ("rf", "xgb", "sefraud", "sage", "gat", "mlp", "tgat", "gcn"):
        if m in R:
            macro(f"{m}Fone", f"{mean(m,'f1'):.3f}"); macro(f"{m}AP", f"{mean(m,'ap'):.3f}")
    gn = [m for m in GNNS if m in R]
    if gn:
        b = max(gn, key=lambda m: mean(m, "f1"))
        macro("bestgnnName", NAMES[b]); macro("bestgnnFone", f"{mean(b,'f1'):.3f}")
        macro("bestgnnAP", f"{mean(b,'ap'):.3f}")
        b2 = max(gn, key=lambda m: mean(m, "ap"))
        macro("bestgnnAPName", NAMES[b2]); macro("bestgnnAPval", f"{mean(b2,'ap'):.3f}")
        if "rtxgnn" in R:
            for met in ("f1", "ap"):
                for m in ("rf", "xgb", "sefraud", b, b2, "gat"):
                    if m in pv[met]:
                        macro(f"p{met.upper()}{m.replace('-', '')}", f"{pv[met][m]:.3f}")
    json.dump({m: {met: [v[met] for v in R[m].values()] for met in ("f1", "ap", "auc")} for m in R},
              open(os.path.join(OUT, "main_raw.json"), "w"))
    return R


ABL = [("full", "RTXGNN (full)"), ("no_hrape", "without HRAPE"), ("time2vec", "HRAPE $\\rightarrow$ Time2Vec (gaps)"),
       ("hrape_abs", "HRAPE + absolute-time encoding"), ("time2vec_abs", "Time2Vec on absolute time"),
       ("no_seal", "without SEAL masks (plain attention)"), ("no_sparsity", "without $\\mathcal{L}_{sp}$"),
       ("no_fidelity", "without $\\mathcal{L}_{suf}$ and $\\mathcal{L}_{nec}$"), ("no_suf", "without $\\mathcal{L}_{suf}$"),
       ("no_nec", "without $\\mathcal{L}_{nec}$"), ("no_curriculum", "without curriculum"),
       ("focal_loss", "focal loss instead of balanced BCE")]
REG = [("full", "masks + $\\mathcal{L}_{sp}$ ($\\lambda_{sp}=0.05$), dropout 0.2, wd $10^{-5}$"),
       ("sparsity_x4", "masks + $\\mathcal{L}_{sp}$ ($\\lambda_{sp}=0.2$)"),
       ("no_sparsity", "masks, no $\\mathcal{L}_{sp}$"),
       ("no_sparsity_drop0", "masks, no $\\mathcal{L}_{sp}$, dropout 0"),
       ("no_sparsity_drop5", "masks, no $\\mathcal{L}_{sp}$, dropout 0.5"),
       ("no_sparsity_wd1e3", "masks, no $\\mathcal{L}_{sp}$, weight decay $10^{-3}$")]
SENS = [("dim32", "$D=32$"), ("full", "$D=64$ (default)"), ("dim128", "$D=128$"), ("layers1", "$L=1$ layer")]


def variant_rows(source):
    R = defaultdict(dict)
    if source == "cpu":
        for r in load("elliptic_ablation.jsonl"):
            R[r["model"].split(":", 1)[1]][r["seed"]] = r
        for r in main_rows():
            if r["model"] == "rtxgnn" and r["seed"] < 5:
                R["full"][r["seed"]] = r
    else:
        for r in load("elliptic_ablation2.jsonl"):
            R[r["model"].split(":", 1)[1]][r["seed"]] = r
    return R


def variant_block(R, spec, prefix, lines):
    ref = R.get("full", {})
    for key, label in spec:
        if key not in R or not R[key]:
            continue
        seeds = sorted(R[key])
        f1 = [R[key][s]["test"]["f1"] for s in seeds]; ap = [R[key][s]["test"]["ap"] for s in seeds]
        if key == "full":
            lines.append(f"{label} & {len(seeds)} & {ms(f1)} & {ms(ap)} & --\\\\")
            macro(f"{prefix}fullFone", f"{np.mean(f1):.3f}")
            continue
        common = [s for s in seeds if s in ref]
        x = np.array([R[key][s]["test"]["f1"] for s in common]); y = np.array([ref[s]["test"]["f1"] for s in common])
        pval = ttest_rel(x, y).pvalue if len(common) >= 3 else float("nan")
        delta = (x - y).mean() if len(common) else float("nan")
        lines.append(f"{label} & {len(seeds)} & {ms(f1)} & {ms(ap)} & {delta:+.3f} ({pval:.2f})\\\\")
        k = key.replace("_", "")
        macro(f"{prefix}{k}Delta", f"{delta:+.3f}"); macro(f"{prefix}{k}Fone", f"{np.mean(f1):.3f}")
        macro(f"{prefix}{k}AP", f"{np.mean(ap):.3f}"); macro(f"{prefix}{k}P", f"{pval:.2f}")


HEAD = ["\\begin{tabular}{lcccc}", "\\toprule", "Variant & seeds & F1 & AP & $\\Delta$F1 (paired $p$)\\\\", "\\midrule"]


def variant_table(spec, fname, prefix):
    """spec entries of the primary study (CPU, 5 seeds) are followed by the
    secondary variants (GPU, 3 seeds), each compared with the full model
    trained in the same setting."""
    cpu, gpu = variant_rows("cpu"), variant_rows("gpu")
    lines = list(HEAD)
    prim = [(k, l) for k, l in spec if k in cpu]
    sec = [(k, l) for k, l in spec if k in gpu and (k not in cpu or k == "full")]
    if len(prim) > 1:
        lines.append("\\multicolumn{5}{l}{\\textit{Primary study (5 seeds)}}\\\\")
        variant_block(cpu, prim, prefix, lines)
    if len(sec) > 1:
        lines.append("\\midrule\\multicolumn{5}{l}{\\textit{Secondary study (3 seeds, trained on GPU; own reference)}}\\\\")
        variant_block(gpu, sec, prefix + "G", lines)
    if len(lines) == len(HEAD):
        return
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def operational():
    """Alert counts at the validation threshold and precision at fixed alert budgets."""
    rows = main_rows()
    ids_p = os.path.join(RESULTS, "elliptic_eval_ids.npy")
    if not rows or not os.path.exists(ids_p):
        return
    from rtxgnn.data import load_elliptic
    from rtxgnn.graph import prepare, split_regions
    d = prepare(load_elliptic())
    _, dev = split_regions(d)
    te = dev.test_mask.cpu().numpy(); y = dev.y.cpu().numpy()[te]
    npos = int(y.sum())
    macro("nTestIllicit", f"{npos:,}"); macro("nTest", f"{int(te.sum()):,}")
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{at validation threshold} & \\multicolumn{4}{c}{precision at an alert budget of}\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}",
             "Model & alerts & missed & false alerts & 250 & 500 & 1,000 & 2,000\\\\", "\\midrule"]
    for m in ("rtxgnn", "sage", "gat", "sefraud", "rf", "xgb"):
        recs = [r for r in rows if r["model"] == m]
        if not recs:
            continue
        al, mi, fa, pk = [], [], [], defaultdict(list)
        for r in recs:
            f = os.path.join(RESULTS, "preds", f"{m}_s{r['seed']}.npy")
            if not os.path.exists(f):
                continue
            p = np.load(f)[te]
            pred = p >= r["thr"]
            al.append(pred.sum()); mi.append(((~pred) & (y == 1)).sum()); fa.append((pred & (y == 0)).sum())
            order = np.argsort(-p)
            for k in (250, 500, 1000, 2000):
                pk[k].append(y[order[:k]].mean())
        if not al:
            continue
        cell = lambda v: f"{np.mean(v):,.0f}"
        lines.append(f"{NAMES[m]} & {cell(al)} & {cell(mi)} & {cell(fa)} & " +
                     " & ".join(f"{np.mean(pk[k]):.2f}" for k in (250, 500, 1000, 2000)) + "\\\\")
        if m == "rtxgnn":
            macro("rtxAlerts", cell(al)); macro("rtxMissed", cell(mi)); macro("rtxFalse", cell(fa))
            macro("rtxPrecAtThousand", f"{np.mean(pk[1000]):.2f}"); macro("rtxPrecAtFive", f"{np.mean(pk[500]):.2f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "operational.tex"), "w").write("\n".join(lines) + "\n")


def temporal_table():
    rows = main_rows()
    ret = load("retraining.jsonl")
    if not rows:
        return
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    info = {}
    for r in rows:
        for s in r["steps"]:
            for met in ("precision", "recall", "f1"):
                per[r["model"]][s["step"]][met].append(s[met])
            info[s["step"]] = (s["n"], s["illicit"])
    rp = defaultdict(lambda: defaultdict(list))
    for r in ret:
        rp[r["model"]][r["step"]].append(r["test"]["f1"])
    lines = ["\\begin{tabular}{rrrccccc}", "\\toprule",
             " & & & \\multicolumn{3}{c}{RTXGNN, static} & RF & RTXGNN / RF\\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){8-8}",
             "Step & Licit & Illicit & Precision & Recall & F1 & F1 static & F1 retrained\\\\", "\\midrule"]
    for t in range(35, 50):
        n, ill = info[t]
        R = per["rtxgnn"][t]
        rf = per["rf"][t]["f1"] if "rf" in per else []
        rr = rp["rtxgnn"].get(t, []); rrf = rp["rf"].get(t, [])
        retr = (f"{np.mean(rr):.2f} / {np.mean(rrf):.2f}" if rr and rrf else "--")
        lines.append(f"{t} & {n-ill:,} & {ill:,} & {ms(R['precision'],2)} & {ms(R['recall'],2)} & {ms(R['f1'],2)} & "
                     f"{np.mean(rf):.2f} & {retr}\\\\" if rf else f"{t} & {n-ill:,} & {ill:,} & -- & -- & -- & -- & --\\\\")
    # pooled vs mean of per-step
    main = [r for r in rows if r["model"] == "rtxgnn"]
    if main:
        pooled = np.mean([r["test"]["f1"] for r in main])
        mean_step = np.mean([np.mean([s["f1"] for s in r["steps"]]) for r in main])
        pre = np.mean([r["test"]["f1"] for r in main])
        macro("rtxMeanStepFone", f"{mean_step:.3f}")
        early = [np.mean([s["f1"] for s in r["steps"] if s["step"] <= 42]) for r in main]
        late = [np.mean([s["f1"] for s in r["steps"] if s["step"] >= 43]) for r in main]
        macro("rtxEarlyFone", f"{np.mean(early):.2f}"); macro("rtxLateFone", f"{np.mean(late):.2f}")
        ill_early = sum(info[t][1] for t in range(35, 43)); ill_late = sum(info[t][1] for t in range(43, 50))
        macro("illEarly", f"{ill_early:,}"); macro("illLate", f"{ill_late:,}")
        macro("illLatePct", f"{100*ill_late/(ill_early+ill_late):.0f}")
    if rp:
        for m in ("rtxgnn", "rf"):
            if m in rp:
                late_r = np.mean([np.mean(rp[m][t]) for t in range(43, 50) if rp[m][t]])
                early_r = np.mean([np.mean(rp[m][t]) for t in range(35, 43) if rp[m][t]])
                macro(f"{m}RetrLateFone", f"{late_r:.2f}"); macro(f"{m}RetrEarlyFone", f"{early_r:.2f}")
                late_s = np.mean([np.mean(per[m][t]["f1"]) for t in range(43, 50)])
                macro(f"{m}StaticLateFone", f"{late_s:.2f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "temporal.tex"), "w").write("\n".join(lines) + "\n")


def label_eff_table():
    rows = load("label_efficiency.jsonl")
    if not rows:
        return
    agg = defaultdict(lambda: defaultdict(list)); ntr = {}
    for r in rows:
        agg[r["model"]][r["frac"]].append(r["test"]["f1"]); ntr[r["frac"]] = (r["n_train"], r["n_train_illicit"])
    fr = sorted(ntr)
    models = [m for m in ("rtxgnn", "rf", "gat", "mlp") if m in agg]
    lines = ["\\begin{tabular}{rr" + "c" * len(models) + "}", "\\toprule",
             "Labels & train (illicit) & " + " & ".join(NAMES[m].replace(" (ours)", "") for m in models) + "\\\\", "\\midrule"]
    for f in fr:
        cells = []
        for m in models:
            v = agg[m].get(f, [])
            full = np.mean(agg[m].get(1.0, [np.nan]))
            cells.append(f"{ms(v)} ({100*np.mean(v)/full:.0f}\\%)" if v else "--")
        lines.append(f"{100*f:.0f}\\% & {ntr[f][0]:,} ({ntr[f][1]:,}) & " + " & ".join(cells) + "\\\\")
    for m in models:
        full = np.mean(agg[m].get(1.0, [np.nan]))
        for f in fr:
            if agg[m].get(f):
                macro(f"le{m}{int(round(100*f))}", f"{np.mean(agg[m][f]):.3f}")
                macro(f"le{m}{int(round(100*f))}Pct", f"{100*np.mean(agg[m][f])/full:.0f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "label_efficiency.tex"), "w").write("\n".join(lines) + "\n")


def explanation_table():
    rows = load("explanations.jsonl")
    if not rows:
        return
    names = [("mask", "rtxgnn", "SEAL masks (self-explaining)"), ("saliency", "rtxgnn", "Saliency (post hoc)"),
             ("ig", "rtxgnn", "Integrated Gradients (post hoc)"), ("gnnexplainer", "rtxgnn", "GNNExplainer (post hoc)"),
             ("random", "rtxgnn", "Random ranking"), ("mask", "sefraud", "SEFraud masks (on SEFraud)"),
             ("mask", "rtxgnn_no_fidelity", "SEAL masks, trained w/o $\\mathcal{L}_{suf},\\mathcal{L}_{nec}$"),
             ("mask", "rtxgnn_no_sparsity", "SEAL masks, trained w/o $\\mathcal{L}_{sp}$")]
    lines = ["\\begin{tabular}{lcccccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Fid$^{+}$ $\\uparrow$} & \\multicolumn{3}{c}{Fid$^{-}$ $\\downarrow$} & Stab. & Time\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Explainer & $k$=5 & 10 & 20 & 5 & 10 & 20 & J@10 & ms\\\\", "\\midrule"]
    cls_lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 " & \\multicolumn{2}{c}{illicit targets} & \\multicolumn{2}{c}{licit targets}\\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                 "Explainer & Fid$^{+}$@10 & Fid$^{-}$@10 & Fid$^{+}$@10 & Fid$^{-}$@10\\\\", "\\midrule"]
    for ex, model, label in names:
        rs = [r for r in rows if r["explainer"] == ex and r["model"] == model]
        if not rs:
            continue
        g = lambda k: [r[k] for r in rs]
        stab = [r["stability_jaccard10"] for r in rs if r["stability_jaccard10"] is not None]
        lines.append(f"{label} & " + " & ".join(f"{np.mean(g(f'fid+@{k}_all')):.3f}" for k in (5, 10, 20)) + " & " +
                     " & ".join(f"{np.mean(g(f'fid-@{k}_all')):.3f}" for k in (5, 10, 20)) + " & " +
                     (f"{np.mean(stab):.2f}" if stab else "--") + f" & {np.mean(g('time_per_target_ms')):.2f}\\\\")
        cls_lines.append(f"{label} & " + " & ".join(f"{np.mean(g(f'fid{s}@10_{c}')):.3f}" for c in ("illicit", "licit") for s in ("+", "-")) + "\\\\")
        import re
        key = re.sub("[^A-Za-z]", "", (model + ex).replace("rtxgnn", "").replace("no_", "No"))
        macro(f"fidP{key}", f"{np.mean(g('fid+@10_all')):.3f}"); macro(f"fidM{key}", f"{np.mean(g('fid-@10_all')):.3f}")
        macro(f"fidTime{key}", f"{np.mean(g('time_per_target_ms')):.2f}")
        if stab:
            macro(f"stab{key}", f"{np.mean(stab):.2f}")
    lines += ["\\bottomrule", "\\end{tabular}"]; cls_lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "explanations.tex"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT, "explanations_class.tex"), "w").write("\n".join(cls_lines) + "\n")
    p = os.path.join(RESULTS, "explanations_cross_seed.json")
    if os.path.exists(p):
        c = json.load(open(p))
        macro("crossSeedSeal", f"{c['seal_mean']:.2f}"); macro("crossSeedRandom", f"{c['random_mean']:.2f}")
    macro("nExplTargets", str(2 * 300))


def synthetic_table():
    rows = load("synthetic.jsonl")
    if not rows:
        return
    sets = ["default", "no_decoy", "wide_burst", "label_noise", "shift"]
    labels = {"default": "default", "no_decoy": "no decoys", "wide_burst": "wide burst", "label_noise": "5\\% label noise",
              "shift": "feature shift"}
    models = ["mlp", "rf", "xgb", "gcn", "sage", "gat", "tgat", "tgn", "sefraud", "rtxgnn:no_seal", "rtxgnn:no_hrape",
              "rtxgnn:time2vec", "rtxgnn"]
    mname = dict(NAMES, **{"rtxgnn:no_seal": "RTXGNN w/o SEAL", "rtxgnn:no_hrape": "RTXGNN w/o HRAPE",
                           "rtxgnn:time2vec": "RTXGNN, Time2Vec", "rtxgnn": "RTXGNN"})
    agg = defaultdict(list); stats = defaultdict(list)
    for r in rows:
        agg[(r["setting"], r["model"])].append(r["test"]["f1"])
        stats[r["setting"]].append((r["nodes"], r["edges"], r["pos_rate"]))
    lines = ["\\begin{tabular}{l" + "c" * len(sets) + "}", "\\toprule",
             "Model & " + " & ".join(labels[s] for s in sets) + "\\\\", "\\midrule"]
    for m in models:
        cells = []
        for s in sets:
            v = agg.get((s, m))
            cells.append(ms(v, 2) if v else "--")
        lines.append(f"{mname[m]} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    lines.append("edges / positive rate & " + " & ".join(
        f"{np.mean([x[1] for x in stats[s]])/1000:.1f}k / {100*np.mean([x[2] for x in stats[s]]):.1f}\\%" if stats[s] else "--" for s in sets) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "synthetic.tex"), "w").write("\n".join(lines) + "\n")
    for m in models:
        for s in sets:
            if agg.get((s, m)):
                macro(f"syn{s.replace('_','')}{m.replace(':','').replace('_','')}", f"{np.mean(agg[(s, m)]):.2f}")
    ea = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "edge_auc" in r:
            for k, v in r["edge_auc"].items():
                ea[(r["setting"], r["model"])][k].append(v)
    if ea:
        el = ["\\begin{tabular}{llcccc}", "\\toprule",
              "Setting & Model & SEAL importance & attention only & GNNExplainer & random\\\\", "\\midrule"]
        for (s, m), d in sorted(ea.items()):
            el.append(f"{labels[s]} & {mname[m]} & " + " & ".join(ms(d[k], 2) if d.get(k) else "--" for k in ("seal", "attention", "gnnexplainer", "random")) + "\\\\")
            key = f"{s.replace('_','')}{m.replace(':','').replace('_','')}"
            for k in ("seal", "attention", "gnnexplainer", "random"):
                if d.get(k):
                    macro(f"edgeAuc{key}{k}", f"{np.mean(d[k]):.2f}")
        el += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(OUT, "synthetic_edges.tex"), "w").write("\n".join(el) + "\n")


def tfinance_table():
    rows = load("tfinance.jsonl")
    if not rows:
        return
    R = by_model(rows)
    order = ["lr", "rf", "xgb", "mlp", "gcn", "sage", "gat", "caregnn", "pcgnn", "fraudre", "sefraud", "rtxgnn"]
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Model & F1 & AUC & AP\\\\", "\\midrule"]
    for m in order:
        if m not in R:
            continue
        lines.append(f"{NAMES[m]} & " + " & ".join(ms([v[k] for v in R[m].values()]) for k in ("f1", "auc", "ap")) + "\\\\")
        macro(f"tf{m}Fone", f"{np.mean([v['f1'] for v in R[m].values()]):.3f}")
        macro(f"tf{m}AUC", f"{np.mean([v['auc'] for v in R[m].values()]):.3f}")
        macro(f"tf{m}AP", f"{np.mean([v['ap'] for v in R[m].values()]):.3f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "tfinance.tex"), "w").write("\n".join(lines) + "\n")


def latency_table():
    rows = load("latency.jsonl")
    if not rows:
        return
    lines = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "Model & threads & $B$ & p50 & p95 & p99 & sample & forward & explain & tx/s\\\\", "\\midrule"]
    for r in sorted(rows, key=lambda r: (r["threads"], r["model"], r["batch"])):
        nm = NAMES.get(r["model"].split("+")[0], r["model"]) + (" + GNNExplainer" if "+" in r["model"] else "")
        lines.append(f"{nm} & {r['threads']} & {r['batch']} & {r['p50']:.1f} & {r['p95']:.1f} & {r['p99']:.1f} & "
                     f"{r['sample_ms']:.1f} & {r['forward_ms']:.1f} & {r['explain_ms']:.1f} & {r['throughput_tps']:,.0f}\\\\")
        words = {1: "one", 4: "four", 32: "thirtytwo", 256: "twofiftysix", 2: "two"}
        mname = {"rtxgnn": "rtx", "gat": "gat", "gcn": "gcn", "mlp": "mlp", "sefraud": "sef",
                 "rtxgnn+gnnexplainer": "rtxgnnex", "gat+gnnexplainer": "gatgnnex"}[r["model"]]
        tag = f"{mname}Lat{'T' + words[r['threads']] if r['threads'] != 1 else ''}B{words[r['batch']]}"
        macro(tag + "P", f"{r['p99']:.1f}"); macro(tag + "Med", f"{r['p50']:.1f}"); macro(tag + "Pnf", f"{r['p95']:.1f}")
        macro(tag + "Fwd", f"{r['forward_ms']:.1f}"); macro(tag + "Smp", f"{r['sample_ms']:.1f}")
        macro(tag + "Exp", f"{r['explain_ms']:.2f}"); macro(tag + "Tps", f"{r['throughput_tps']:,.0f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "latency.tex"), "w").write("\n".join(lines) + "\n")
    macro("cpuName", rows[0]["cpu"].replace("(R)", "").replace("(TM)", ""))
    macro("peakRss", f"{max(r['peak_rss_mb'] for r in rows):,.0f}")


def preprocessing_table():
    raw = by_model(load("elliptic_raw.jsonl"))
    main = by_model(load("elliptic_main.jsonl"))
    if not raw:
        return
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", " & \\multicolumn{2}{c}{raw features} & \\multicolumn{2}{c}{quantile-normalised}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Model & F1 & AP & F1 & AP\\\\", "\\midrule"]
    for m in ("mlp", "gcn", "sage", "gat", "sefraud", "rtxgnn"):
        if m in raw and m in main and 0 in raw[m] and 0 in main[m]:
            a, b = raw[m][0], main[m][0]
            lines.append(f"{NAMES[m]} & {a['f1']:.3f} & {a['ap']:.3f} & {b['f1']:.3f} & {b['ap']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "preprocessing.tex"), "w").write("\n".join(lines) + "\n")


def ring_macros():
    p = os.path.join(RESULTS, "fraud_ring.json")
    if os.path.exists(p):
        r = json.load(open(p))
        macro("ringHitPct", f"{100 * r['hit_rate_detected_members']:.0f}")
        macro("ringNdet", str(r["n_detected_members"]))


def write_macros():
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% generated by experiments/analyze.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main_table()
    variant_table(ABL, "ablation.tex", "abl")
    variant_table(REG, "regularisation.tex", "reg")
    variant_table(SENS, "sensitivity.tex", "sens")
    for f in (operational, temporal_table, label_eff_table, explanation_table, synthetic_table, tfinance_table,
              latency_table, preprocessing_table, ring_macros):
        try:
            f()
        except Exception as e:
            print("failed", f.__name__, repr(e))
    write_macros()
    print(open(os.path.join(OUT, "numbers.tex")).read())
