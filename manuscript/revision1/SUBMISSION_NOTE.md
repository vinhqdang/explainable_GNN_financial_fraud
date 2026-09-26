# Revision 1: submission note

**Manuscript:** DAJOUR-D-26-00332, Decision Analytics Journal.

**Revised title:** *A Self-Explaining Temporal Graph Learning Method for Low-Latency Financial Fraud Detection and Decision Support*

**Status:** Revision 1 was submitted on 2026-09-26.

## What was submitted

| Item | File |
|---|---|
| Manuscript as a flat LaTeX package: one `.tex` file plus 4 PNG figures | `RTXGNN_revision1_latex.zip`, built by `make_elsevier.py` |
| Compiled manuscript | `RTXGNN_revision1.pdf` (57 pages) |
| Manuscript with changes marked against submission 0 | `RTXGNN_revision1_marked.pdf`, built by `make_diff.sh` |
| Response to reviewers | `response/response_to_reviewers.pdf` |
| Title page, with authors | `RTXGNN_title_page.docx` |
| Text typed into the "Respond to Reviewers" box | A short cover note that points to the PDF response and summarises the changes |

The manuscript itself is anonymised: the author block is commented out in `main.tex`.

## Main results reported (Elliptic, temporal split, 10 seeds)

- **RTXGNN:** F1 0.700 and AP 0.771.
  - It is on par with the best GNN baselines; after Holm correction it is not significantly different from them.
  - It is ahead of SEFraud on average, by +0.054 F1.
- **Tree ensembles are more accurate.** XGBoost reaches F1 0.790 and the random forest 0.733.
- **Equal-budget tuning study.** RTXGNN and ten baselines each received 20 configurations, selected on validation AP only.
  - The ranking is unchanged.
  - Tuned RTXGNN reaches 0.703, tuned XGBoost 0.789 and tuned RF 0.769.
  - The tuned GNNs reach between 0.671 and 0.711.
- **Explanations.**
  - Sufficiency is close to Integrated Gradients (Fid−@10 0.076 against 0.069) at about 1/100 of the cost, and stability is high.
  - Necessity is weak (Fid+@10 0.032).
- **Latency.** End to end, the p99 is 5.9 ms on one CPU core.

## Weak points likely to come back in round 2

- Tree ensembles beat RTXGNN, and RTXGNN is not significantly better than the other GNNs.
- The SEAL masks have weak necessity: they are near-binary and agree little across seeds.
- HRAPE has almost no effect on Elliptic, because edges never cross time steps. Its evidence rests on Time2Vec and the synthetic data.
- There is no user study with analysts.
- Six baselines were not tuned: LR, MLP, EvolveGCN, APAN, CARE-GNN and PC-GNN. EvolveGCN is unstable (0.332 ± 0.170).
- H2-FDetector and DGA-GNN from submission 0 were not re-run; the omission is justified in the text only.
- The model is trained on steps 1–30 only. With Weber et al.'s protocol (train 1–34), all models score about 0.1 F1 higher (see `experiments/sanity_weber.py`).

## Reproducing

- `python experiments/analyze.py` regenerates every table and `tables/numbers.tex` from `results/*.jsonl`.
- `manuscript/revision1/build.sh` builds the manuscript and the letter.
- `make_diff.sh` builds the marked-up version.
- `make_elsevier.py` builds the flat package.

See `HANDOFF.md` for a detailed list of the weak results and where they are in the code.

## History note

Commits `51d4de5` and `8497c72` overwrote result records with altered values, without re-running the experiments. They were reverted in `4974654`. All numbers in the submitted manuscript come from the genuine run records.
