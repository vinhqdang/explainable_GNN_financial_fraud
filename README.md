# RTXGNN: Real-Time eXplainable Graph Neural Network for Financial Fraud Detection

A self-explainable temporal GNN architecture with regulatory-compliant explanations, targeting the [Applied Computing and Informatics (ACI)](https://www.emeraldgrouppublishing.com/journal/aci) journal.

## Overview

RTXGNN jointly learns **fraud prediction** and **multi-granularity explanations** on dynamic financial transaction graphs. Key design goals:

- **< 50ms** end-to-end inference (prediction + explanation)
- **Regulatory compliance**: GDPR Article 22, ECOA/FCRA reason codes
- **Self-explainable**: no post-hoc explanation step — masks are generated inline during the forward pass
- **Temporal awareness**: multi-scale encoding capturing hourly, daily, weekly, and monthly fraud patterns

## Architecture

```
INPUT → Temporal Encoding (HRAPE) → Self-Explainable Aggregation Layers (SEAL)
                                              ↓
                              Dual-Head Output (Prediction + Explanation)
                                              ↓
                              Regulatory Compliance Module
                              (Reason Codes · GDPR Summary · Audit Trail)
```

### Core Components

| Component | Description |
|-----------|-------------|
| **HRAPE** | Hierarchical Recency-Aware Positional Encoding — multi-scale temporal encoding with learnable Fourier features and recency-weighted attention |
| **SEAL** | Self-Explainable Aggregation Layer — jointly generates node/edge/feature importance masks with information bottleneck regularization |
| **Dual-Head** | Prediction head (Focal Loss) + Explanation head (reason code classification, subgraph scoring, confidence estimation) |
| **Regulatory Module** | Maps technical explanations to ECOA/FCRA reason codes with GDPR Article 22 summaries and full audit trails |

## Datasets

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| **Elliptic Bitcoin** | 203,769 | 234,355 | Real-world transaction graph with 49 temporal snapshots |
| **YelpChi** | 45,954 | 3,846,979 | Review fraud network for domain generalization |
| **Synthetic Money Laundering** | 5,000 | — | Injected fraud ring typologies on a scale-free graph |

## Results

### Elliptic Bitcoin Dataset

| Model | F1-Score | AUC |
|-------|----------|-----|
| **RTXGNN (Ours)** | **0.5125** | **0.8662** |
| MLP | 0.4538 | 0.8638 |
| GAT | 0.4285 | 0.8659 |
| GraphSAGE | 0.4141 | 0.8735 |
| GCN | 0.2697 | 0.8217 |

RTXGNN achieves the highest F1-score while maintaining competitive AUC, with superior temporal stability across future time steps (concept drift robustness).

## Repository Structure

```
.
├── RTXGNN_Implementation.ipynb      # Full implementation: training, evaluation, experiments
├── RTXGNN_Algorithm_Design.md       # Detailed algorithm design and pseudocode
├── Section_Evaluation.tex           # LaTeX evaluation section for the paper
├── literature_review.md             # Literature review and identified research gaps
├── learning_curves.png              # Training loss and test F1 over epochs
└── temporal_stability.png           # F1-score across test time steps
```

## Key Research Contributions

| Research Gap | RTXGNN Solution |
|---|---|
| No standardized XAI metrics for fraud | Fraud-specific evaluation (domain consistency, actionability, compliance score) |
| No human-centered evaluation | Built-in analyst study protocol with validated dimensions |
| Temporal explainability lacking | HRAPE + recency-weighted attention masks |
| No ground-truth explanation datasets | Self-supervised explanation learning |
| Regulatory non-compliance | Dedicated module for GDPR/ECOA/FCRA compliance |
| Scalability limitations | Tiered async inference pipeline < 50ms |

## Getting Started

```bash
# Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn matplotlib

# Open the notebook
jupyter notebook RTXGNN_Implementation.ipynb
```

The notebook covers:
1. Dataset loading and preprocessing (Elliptic, YelpChi, Synthetic)
2. Model training with curriculum learning
3. Ablation studies
4. Sensitivity, runtime, and temporal stability experiments
5. t-SNE visualization of learned embeddings

## Training

RTXGNN uses a **curriculum learning** strategy that gradually introduces explanation objectives:

```
Epoch 0-5:   Prediction loss only
Epoch 5-10:  + Explanation fidelity (λ=0.3)
Epoch 10-20: + Sparsity + Temporal consistency
Epoch 20+:   Full multi-task loss (prediction + explanation + sparsity + temporal + reason codes)
```

Loss function:

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{exp} + \lambda_2 \mathcal{L}_{sparse} + \lambda_3 \mathcal{L}_{temporal} + \lambda_4 \mathcal{L}_{reason}$$

## Inference

Tiered explanation depth based on fraud score:

| Score | Action | Explanation |
|-------|--------|-------------|
| < 0.3 | Approve | None (latency optimized) |
| 0.3–0.7 | Review | Light (top-3 features + primary reason) |
| ≥ 0.7 | Decline | Full async (subgraph + NL + regulatory output) |

Target latency: **< 30ms** on GPU for real-time path.

## Citation

```bibtex
@article{rtxgnn2024,
  title   = {RTXGNN: Real-Time eXplainable Graph Neural Network for Financial Fraud Detection},
  author  = {Dang, Vinh},
  journal = {Applied Computing and Informatics},
  year    = {2024}
}
```
