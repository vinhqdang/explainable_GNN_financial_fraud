# Handoff note: where RTXGNN still performs poorly

This note is for whoever continues work on this repository. It says where to look first and states
exactly which results are still weak. It deliberately does not prescribe fixes.

All numbers below come from `results/*.jsonl` via `experiments/analyze.py`
(`manuscript/revision1/tables/numbers.tex`). Nothing is hand-typed in the manuscript. Re-running
`python experiments/analyze.py` regenerates every table and macro.

## 1. What to read first, in this order

1. **The task.** Manuscript DAJOUR-D-26-00332 received major revisions. The full reviewer and editor
   comments are provided separately by the author. Abbreviated quotes of every point (R1.1–R1.15,
   R2.1–R2.11, AE) are in `manuscript/revision1/response/response.tex`.
2. **The original submission.** `manuscript/submission0/`: sections `1introduction.tex` to
   `6conclusions.tex`, with main file `elsarticle-template-num.tex`.
3. **The current revision.** `manuscript/revision1/RTXGNN_revision1.pdf` (sources `*.tex`, built by
   `build.sh`) and the response letter `manuscript/revision1/response/response_to_reviewers.pdf`.
   `RTXGNN_revision1_marked.pdf` shows the changes against submission 0.
4. **The method code.**
   - `rtxgnn/model.py`: HRAPE, SEAL and RTXGNN.
   - `rtxgnn/objective.py`: loss terms.
   - `rtxgnn/train.py`: training loop, early stopping and threshold selection.
   - `rtxgnn/graph.py`: 2-hop region subgraphs.
   - `rtxgnn/data.py`: data loading, splits and quantile normalisation.
5. **The baselines.** `rtxgnn/baselines.py` (all graph baselines, re-implemented) and
   `rtxgnn/tabular.py` (LR, RF, XGBoost). How each was adapted is described in
   `manuscript/revision1/7appendix.tex`, Appendix B.
6. **The experiment scripts.** `experiments/run_*.py`, `experiments/common.py`,
   `experiments/latency.py`. The Colab helpers `experiments/colab_*.sh|py` were used to run GPU jobs.
7. **The protocol**, as defined in `rtxgnn/data.py` and `rtxgnn/train.py`:
   - Elliptic split: train on time steps 1–30, validate on 31–34, test on 35–49.
   - Early stopping on validation AP; the threshold is the one that maximises validation F1.
   - 10 seeds; Wilcoxon signed-rank test with Holm correction.
   - Hidden size 64, 2 layers, no per-model tuning.

## 2. Verified: the data pipeline is not the problem

`experiments/sanity_weber.py` reproduces the protocol of Weber et al. (2019): train on steps 1–34,
threshold 0.5, random forest with 50 trees and 50 features per split. Output:

| Setting | F1 (illicit) |
|---|---|
| RF, all features, train 1–34, 3 seeds | 0.809 / 0.815 / 0.812 (Weber et al. report about 0.79) |
| RF, local features only, train 1–34 | 0.780 |
| LR (sklearn default), train 1–34 | 0.440 (Weber et al. report about 0.48) |
| **RF, all features, train 1–30** (manuscript protocol), threshold 0.5 | 0.678 / 0.737 / 0.703 |

Loading, labels, features and splits are therefore correct. The same forest loses about 0.1 F1 when
steps 31–34 are held out for validation. In the current protocol, steps 31–34 are never used to fit
the final models, neither RTXGNN nor any baseline.

## 3. Where performance is still weak (exact locations)

### 3.1 Detection accuracy on Elliptic (main table)

- **Data:** `results/elliptic_main.jsonl`, `results/elliptic_main_gpu.jsonl`, `results/elliptic_evo.jsonl`
  (EvolveGCN only). Table: `manuscript/revision1/tables/main.tex`.
- RTXGNN test F1 is 0.700 ± 0.045 and AP 0.771.
  - XGBoost reaches F1 0.790 and AP 0.792, and is significantly better.
  - Random forest reaches F1 0.733.
- After Holm correction, RTXGNN is not significantly better than any GNN except PC-GNN on F1.
  - GAS: 0.701.
  - GCN: 0.690.
  - TGN: 0.685.
  - GraphSAGE: 0.684.
- RTXGNN is unstable across seeds: F1 s.d. 0.045 and precision s.d. 0.088. Its precision is 0.680
  against 0.862 for XGBoost, so the gap is mostly in precision.
- Calibration is poor.
  - The validation-chosen threshold averages 0.82.
  - At threshold 0.5, RTXGNN's test F1 is 0.596.
  - For comparison, the random forest's threshold is 0.50 and GCN's is 0.52. GAS and SEFraud behave
    like RTXGNN, with thresholds of about 0.8.

### 3.2 The explanation mechanism (SEAL masks)

- **Code:** `rtxgnn/model.py` (mask generators), `rtxgnn/objective.py` (sufficiency, necessity,
  sparsity), `rtxgnn/explain.py` (metrics and post-hoc explainers), `experiments/run_explanations.py`.
- **Data:** `results/explanations.jsonl`. Tables: `tables/explanations.tex`,
  `tables/explanations_class.tex`.
- **Necessity is weak.** Removing the top-10 SEAL features changes the prediction by only
  Fid+@10 = 0.032.
  - Integrated Gradients reaches 0.142 on the same model.
  - GNNExplainer reaches 0.279.
  - A random ranking reaches 0.011.
- **Sufficiency is not better than Integrated Gradients.** SEAL's Fid−@10 is 0.076 against 0.069 for IG.
  - For illicit targets the gap is larger: 0.088 against 0.037.
  - At k=5, saliency (0.123) beats SEAL (0.143).
- **The masks are near-binary and not selective.** About 75–82 of the 165 features are "open"
  (mask value above 0.5) in every case study (`tables/case_studies.tex`).
  - Removing the five leading features barely changes the score (for example 1.00 becomes 0.99).
  - On Elliptic, the edge importances of the top neighbours are mostly 0.00.
- **Cross-seed agreement is low.** The mean Jaccard similarity of the top-10 features is 0.12 across
  independently trained models, against 0.03 for random rankings.
- **The sparsity term L_sp does not improve fidelity.** The model without L_sp has the same
  Fid−@10 (0.076) and a better Fid−@5 (0.111).

### 3.3 The components give no measurable accuracy gain (ablations)

- **Data:** `results/elliptic_ablation.jsonl` (primary study, 5 seeds, CPU) and
  `results/elliptic_ablation2.jsonl` (secondary study, 3 seeds, GPU). Tables: `tables/ablation.tex`,
  `tables/regularisation.tex`, `tables/sensitivity.tex`.
- The full model (0.696) is beaten by three of its own ablations:
  - without SEAL masks: 0.704;
  - without L_sp: 0.718;
  - without L_suf and L_nec: 0.708.
- Removing HRAPE costs only −0.030 F1 (p = 0.26).
- Secondary study (3 seeds): removing L_suf alone gives −0.067, while the primary study shows +0.012
  when both L_suf and L_nec are removed. These results are inconsistent.
- The capacity sensitivity is large:
  - hidden size 32: 0.606;
  - one layer: 0.613;
  - default (hidden size 64, two layers): 0.724.

### 3.4 The temporal component (HRAPE)

- **Code:** `rtxgnn/model.py` (HRAPE, Time2Vec), `rtxgnn/synthetic.py`, `experiments/run_synthetic.py`.
- **Data:** `results/synthetic.jsonl`, `results/fraud_ring.json`. Tables: `tables/synthetic.tex`,
  `tables/synthetic_edges.tex`.
- On Elliptic, all edges lie within a single time step, so HRAPE carries almost only degree
  (velocity) information.
- On the synthetic benchmark, which was designed to require timing:
  - **Default setting:** TGAT reaches 0.59 against 0.52 for RTXGNN. RTXGNN without SEAL (0.55) is
    also better than the full model.
  - **Wide burst:** RTXGNN reaches 0.49, the same as without HRAPE (0.49), and below TGAT (0.54).
  - **Feature shift:** RTXGNN without HRAPE (0.77) and without SEAL (0.80) both beat the full model (0.75).
  - Standard deviations are 0.06–0.12, so most differences are not resolved with 5 graphs.
- Edge-explanation AUC for the ring transactions:
  - The s.d. is 0.2–0.3.
  - For the model without HRAPE, raw attention (0.84) beats the SEAL importances (0.60).

### 3.5 Second dataset (T-Finance)

- **Data:** `results/tfinance.jsonl`. Table: `tables/tfinance.tex`. Script: `experiments/run_tfinance.py`.
- RTXGNN reaches F1 0.837, below GraphSAGE (0.849) and FRAUDRE (0.842).
- HRAPE is disabled because there are no timestamps.
- GAS and the temporal baselines were not run.
- Graph models see at most 10 sampled neighbours per account.

### 3.6 Drift after step 43

- **Data:** `results/retraining.jsonl`, and the per-step records inside the main results. Table:
  `tables/temporal.tex`.
- For RTXGNN, late-period F1 is 0.03 with the static model and 0.10 when retrained with a one-step
  label delay. The random forest reaches 0.02 and 0.18. All models collapse, and retraining recovers little.

### 3.7 Label efficiency

- **Data:** `results/label_efficiency.jsonl`. Table: `tables/label_efficiency.tex`.
- With 10% of the labels, RTXGNN keeps 87% of its full-label F1. GAT keeps 96% and the MLP 104%,
  because the MLP's full-label F1 is itself low.
- RTXGNN is therefore not more label-efficient than the simple baselines.

### 3.8 Latency

- **Data:** `results/latency.jsonl`. Table: `tables/latency.tex`. Script: `experiments/latency.py`.
- RTXGNN is the slowest model apart from the GNNExplainer rows:
  - at B=1, p50 is 2.8 ms, against 1.5 ms for GAT and 1.7 ms for SEFraud;
  - its forward pass takes twice as long as GAT's;
  - at B=256, throughput is 2,013 transactions/s against 7,064 for GAT;
  - at B=256, the explanation step takes 27 ms of 126 ms.

### 3.9 Baseline implementations to double-check

- **EvolveGCN-O** (`rtxgnn/baselines.py`) reaches F1 0.332 ± 0.170. This is very unstable: per-seed F1 ranges
  from 0.136 to 0.582, and 4 of the 10 seeds are below 0.17.
- **Logistic regression** (`rtxgnn/tabular.py`) reaches F1 0.326. The sklearn default in Section 2
  reaches 0.440 under a different protocol.
- **TGN and APAN:** on Elliptic the memory and the mailbox are always empty before a node's
  time step (see Appendix B).
- **Tuning.** The main comparison uses one configuration for all models (hidden size 64, 2 layers, dropout 0.2).
  An equal-budget tuning study (`experiments/run_tuning.py`; records `results/tuning_*.jsonl`, final 10-seed runs
  `results/tuned_*.jsonl`; table `tables/tuned.tex`) gives RTXGNN and ten baselines 20 configurations each,
  selected on validation AP. Tuned RTXGNN reaches F1 0.703 (default 0.700). Tuned XGBoost reaches 0.789 and tuned
  RF 0.769; tuned GNNs are between 0.671 and 0.711, with no significant F1 difference to RTXGNN after Holm correction.

### 3.10 Reproducibility caveats

- Runs are split across a 4-vCPU CPU machine and Colab T4 GPUs. The exact split is given in
  `manuscript/revision1/4setup.tex`, paragraph "Hardware".
- The full-model F1 changes with the study because of different seeds and hardware:
  - 0.700 in the main comparison;
  - 0.696 in the primary ablation;
  - 0.724 in the secondary ablation;
  - 0.718 in the label-efficiency study.
- Quantile normalisation and the shift-invariant HRAPE were chosen during development. Both choices
  were re-checked on the validation steps only (Section 5.4 and Appendix C of the manuscript).

## 4. Constraints to respect

- The revision must address the reviewers' points and must not introduce new unsupported claims.
- Every number in the manuscript and the letter must come from the result records via
  `experiments/analyze.py`. Do not hand-edit `tables/`.
- `manuscript/submission0/` is the frozen original. Do not modify it.
- Commit as `Vinh <dqvinh87@gmail.com>` and push to `main`.
