# RTXGNN: a self-explaining temporal graph neural network for fraud detection

Code, experiment scripts, result records and manuscript sources for

> *A Self-Explaining Temporal Graph Learning Method for Low-Latency Financial Fraud Detection and Decision Support*
> (revision of Decision Analytics Journal manuscript DAJOUR-D-26-00332).

RTXGNN scores transactions in a transaction graph and returns, from the same forward pass, an explanation:
the top-k input features of the transaction and its most important incoming/outgoing transactions.

* **SEAL** (Self-Explainable Aggregation Layer): attention-based message passing whose messages are gated by
  learned edge and node masks, preceded by a per-node feature mask on the raw inputs. The masks are the explanation.
* **HRAPE** (Hierarchical Recency-Aware Positional Encoding): shift-invariant temporal encoding built from time
  gaps (multi-scale sinusoids), a learnable recency bias of the attention, and transaction velocity.
  It never encodes absolute time.
* **Objective**: class-balanced cross-entropy + sufficiency (the hard top-k explanation alone reproduces the
  prediction) + necessity (its complement is uninformative) + L1 mask sparsity, with a linear warm-up.

## Repository layout

```
rtxgnn/                  model and library code
  model.py               HRAPE, Time2Vec (ablation), SEAL, RTXGNN
  objective.py           training objective (balanced BCE, sufficiency, necessity, sparsity)
  baselines.py           MLP, GCN, GraphSAGE, GAT, EvolveGCN-O, TGAT, TGN, APAN,
                         CARE-GNN, PC-GNN, GAS, FRAUDRE, SEFraud (re-implementations)
  tabular.py             logistic regression, random forest, XGBoost
  data.py, graph.py      Elliptic / T-Finance loading, 2-hop region subgraphs
  synthetic.py           synthetic temporal laundering benchmark (documented generator)
  serving.py             request-level inference: neighbourhood sampling + explanation
  explain.py             saliency, Integrated Gradients, GNNExplainer, fidelity metrics
experiments/             one script per experiment; each appends JSON records to results/
  run_elliptic.py        main comparison (10 seeds), ablations, regularisation, sensitivity
  run_label_efficiency.py, run_retraining.py, run_explanations.py, run_synthetic.py,
  run_tfinance.py, latency.py, run_case_studies.py
  analyze.py             builds every table and manuscript/revision1/tables/numbers.tex
  figures.py             builds the figures
results/                 per-run JSON records (checkpoints and predictions are not versioned)
manuscript/submission0/  manuscript as first submitted
manuscript/revision1/    revised manuscript (main.tex) and response to reviewers (response/)
```

`legacy/` (notebook `RTXGNN_Implementation.ipynb`, `RTXGNN_Algorithm_Design.md`, draft evaluation section) documents the first version of the
project and is kept for reference only. The notebook's Elliptic preprocessing used surrogate time steps and
must not be used to reproduce results; use the scripts above.

## Reproducing the results

```bash
pip install torch torch-geometric pandas scikit-learn xgboost networkx matplotlib scipy
export PYTHONPATH=$PWD
cd experiments
python run_elliptic.py --models lr,rf,xgb,mlp,gcn,sage,gat,evolvegcn,tgat,tgn,apan,caregnn,pcgnn,gas,fraudre,sefraud,rtxgnn --seeds 0-9 --tag main
python run_elliptic.py --models rtxgnn:no_hrape,rtxgnn:time2vec,rtxgnn:hrape_abs,rtxgnn:no_seal,rtxgnn:no_sparsity,rtxgnn:no_fidelity --seeds 0-4 --tag ablation
python run_label_efficiency.py
python run_retraining.py
python run_explanations.py
python run_synthetic.py
python run_tfinance.py        # needs T-Finance from GADBench converted to data/tfin/tfinance.npz
python latency.py --threads 1 && python latency.py --threads 4
python run_case_studies.py
python analyze.py && python figures.py
```

Elliptic is downloaded automatically through PyTorch Geometric (the official time step of every transaction is
read from the raw CSV). All experiments run on CPU.

## Protocol in brief

* Elliptic temporal split: train steps 1–30, validation 31–34, test 35–49; decision threshold chosen on validation.
* Neural models: quantile-normalised features (fitted on training steps), hidden size 64, 2 layers, Adam,
  early stopping on validation average precision. Tree ensembles use raw features.
* 10 seeds; two-sided Wilcoxon signed-rank tests with Holm correction.

## License

See `LICENSE`.
