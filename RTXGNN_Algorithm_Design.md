# RTXGNN: Real-Time eXplainable Graph Neural Network for Financial Fraud Detection

## A Self-Explainable Temporal Architecture with Regulatory-Compliant Explanations

---

## 1. Problem Formulation

### 1.1 Formal Definition

Let $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t, \mathbf{X}^t, \mathbf{E}^t)$ be a dynamic heterogeneous financial graph at time $t$, where:
- $\mathcal{V} = \{v_1, ..., v_n\}$: Set of entities (accounts, devices, merchants, cards)
- $\mathcal{E}^t \subseteq \mathcal{V} \times \mathcal{V} \times \mathcal{R} \times \mathbb{R}^+$: Timestamped edges with relation types $\mathcal{R}$
- $\mathbf{X}^t \in \mathbb{R}^{n \times d_v}$: Node feature matrix (account attributes, behavioral features)
- $\mathbf{E}^t \in \mathbb{R}^{|\mathcal{E}^t| \times d_e}$: Edge feature matrix (transaction amount, frequency, etc.)

**Objective**: For a target transaction $e_{ij}^t$ (edge-level) or account $v_i$ (node-level), jointly learn:
1. **Fraud prediction**: $\hat{y} = f(\mathcal{G}^t, v_i) \in [0,1]$
2. **Multi-granularity explanation**: $\mathcal{X} = \{(\mathbf{M}^{node}, \mathbf{M}^{edge}, \mathbf{M}^{temp}, \mathbf{M}^{feat})\}$
3. **Regulatory reason codes**: $\mathcal{R}_c = \{(r_k, s_k, \text{desc}_k)\}_{k=1}^K$ mapping to compliance templates

### 1.2 Design Requirements

| Requirement | Specification | Gap Addressed |
|-------------|---------------|---------------|
| Latency | < 50ms end-to-end (prediction + explanation) | Gap 6: Scalability |
| Temporal awareness | Capture patterns across multiple time scales | Gap 3: Temporal explainability |
| Explanation granularity | Node, edge, feature, temporal, subgraph levels | Gap 1: Standardized metrics |
| Regulatory compliance | GDPR Art. 22, ECOA/FCRA reason codes | Gap 5: Regulatory requirements |
| Human interpretability | Natural language + visual explanations | Gap 2: Human-centered evaluation |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RTXGNN ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │   INPUT      │    │  TEMPORAL        │    │  EXPLAINABLE AGGREGATION    │   │
│  │   LAYER      │───▶│  ENCODING        │───▶│  LAYERS (L layers)          │   │
│  │              │    │  MODULE          │    │                             │   │
│  └──────────────┘    └──────────────────┘    └──────────────┬──────────────┘   │
│        │                     │                              │                   │
│        │                     │                              ▼                   │
│        │                     │              ┌───────────────────────────────┐   │
│        │                     │              │  DUAL-HEAD OUTPUT             │   │
│        │                     │              │  ┌───────────┬───────────┐    │   │
│        │                     │              │  │ Prediction│Explanation│    │   │
│        │                     │              │  │ Head      │ Head      │    │   │
│        │                     │              │  └───────────┴───────────┘    │   │
│        │                     │              └───────────────┬───────────────┘   │
│        │                     │                              │                   │
│        ▼                     ▼                              ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                  REGULATORY COMPLIANCE MODULE                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Reason Code │  │ Template    │  │ Natural     │  │ Audit Trail │     │   │
│  │  │ Mapper      │  │ Generator   │  │ Language    │  │ Logger      │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Multi-Scale Temporal Encoding Module

Unlike existing temporal GNNs that use single-scale time encoding, we propose **Hierarchical Recency-Aware Positional Encoding (HRAPE)** that captures fraud-relevant temporal patterns across multiple granularities.

#### 3.1.1 Temporal Feature Engineering

For each edge $e_{ij}^t$ with timestamp $t$:

```python
def compute_temporal_features(edge, current_time, history_window):
    """
    Compute multi-scale temporal features for fraud detection
    """
    t_delta = current_time - edge.timestamp
    
    # Recency decay (exponential with learnable rate)
    recency_weight = exp(-lambda * t_delta)
    
    # Periodic encodings for cyclical patterns
    hour_enc = [sin(2π * hour/24), cos(2π * hour/24)]
    day_enc = [sin(2π * day_of_week/7), cos(2π * day_of_week/7)]
    month_enc = [sin(2π * day_of_month/30), cos(2π * day_of_month/30)]
    
    # Velocity features (burst detection)
    velocity_1h = count_edges(source, target, window=1h) / 1
    velocity_24h = count_edges(source, target, window=24h) / 24
    velocity_7d = count_edges(source, target, window=7d) / 168
    
    # Acceleration (velocity change rate)
    acceleration = (velocity_1h - velocity_24h) / velocity_24h
    
    return concat([recency_weight, hour_enc, day_enc, month_enc, 
                   velocity_1h, velocity_24h, velocity_7d, acceleration])
```

#### 3.1.2 Learnable Temporal Positional Encoding

$$\mathbf{TE}(t) = \sum_{s \in \{h, d, w, m\}} \mathbf{W}_s \cdot \phi_s(t) + \mathbf{b}_s$$

where:
- $\phi_s(t)$ = Fourier features at scale $s$ (hourly, daily, weekly, monthly)
- $\mathbf{W}_s \in \mathbb{R}^{d_{te} \times d_\phi}$ = learnable projection per scale

**Recency-Weighted Attention**:

$$\alpha_{ij}^{(t)} = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{h}_i)(\mathbf{W}_K \mathbf{h}_j)^\top}{\sqrt{d_k}} + \gamma \cdot \text{RecencyScore}(t_{ij}, t_{now})\right)$$

where $\gamma$ is a learnable temperature parameter controlling recency influence.

---

### 3.2 Self-Explainable Aggregation Layer (SEAL)

The core innovation: **jointly learning prediction and explanation through differentiable mask generation**.

#### 3.2.1 Architecture per Layer

```python
class SEALLayer(nn.Module):
    """
    Self-Explainable Aggregation Layer
    Jointly learns node embeddings and importance masks
    """
    def __init__(self, in_dim, out_dim, num_relations, num_heads=4):
        self.num_relations = num_relations
        self.num_heads = num_heads
        
        # Relation-specific transformations
        self.W_rel = nn.ParameterList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        
        # Attention mechanism
        self.attn = nn.MultiheadAttention(out_dim, num_heads)
        
        # Explanation mask generators (differentiable)
        self.node_mask_gen = MaskGenerator(out_dim, activation='sigmoid')
        self.edge_mask_gen = MaskGenerator(2 * out_dim, activation='sigmoid')
        self.feat_mask_gen = MaskGenerator(in_dim, activation='sigmoid')
        
        # Temporal importance scorer
        self.temp_scorer = TemporalImportanceScorer(out_dim)
        
    def forward(self, x, edge_index, edge_attr, edge_time, batch):
        # Generate feature-level masks (which features matter)
        feat_mask = self.feat_mask_gen(x)  # [N, d_in]
        x_masked = x * feat_mask
        
        # Relation-aware message passing
        messages = []
        edge_masks = []
        
        for r in range(self.num_relations):
            # Get edges of relation type r
            rel_edges = edge_index[:, edge_attr[:, 0] == r]
            
            if rel_edges.size(1) == 0:
                continue
            
            # Transform source nodes
            src_nodes = x_masked[rel_edges[0]]
            tgt_nodes = x_masked[rel_edges[1]]
            
            # Compute edge importance mask
            edge_repr = torch.cat([src_nodes, tgt_nodes], dim=-1)
            e_mask = self.edge_mask_gen(edge_repr)  # [E_r, 1]
            edge_masks.append((rel_edges, e_mask))
            
            # Temporal importance weighting
            temp_weight = self.temp_scorer(edge_time[edge_attr[:, 0] == r])
            
            # Relation-specific transformation
            msg = self.W_rel[r](src_nodes) * e_mask * temp_weight.unsqueeze(-1)
            messages.append((rel_edges, msg))
        
        # Aggregate messages with attention
        h_agg = self.attention_aggregate(x_masked, messages)
        
        # Generate node-level importance mask
        node_mask = self.node_mask_gen(h_agg)  # [N, 1]
        
        return h_agg, {
            'node_mask': node_mask,
            'edge_masks': edge_masks,
            'feat_mask': feat_mask,
            'temp_weights': temp_weight
        }
```

#### 3.2.2 Mask Generator with Information Bottleneck

To ensure explanations are sparse and meaningful, we use **Graph Information Bottleneck (GIB)**:

$$\mathcal{L}_{GIB} = -I(\mathbf{M}; Y) + \beta \cdot I(\mathbf{M}; \mathcal{G})$$

where:
- $I(\mathbf{M}; Y)$: Mutual information between mask and label (maximize predictiveness)
- $I(\mathbf{M}; \mathcal{G})$: Mutual information between mask and input graph (minimize complexity)
- $\beta$: Trade-off parameter (higher = sparser explanations)

```python
class MaskGenerator(nn.Module):
    """
    Generates differentiable importance masks with sparsity regularization
    """
    def __init__(self, input_dim, hidden_dim=64, activation='sigmoid'):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.activation = activation
        
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.mlp(x)
        
        if self.activation == 'sigmoid':
            # Continuous relaxation during training
            mask = torch.sigmoid(logits / temperature)
        elif self.activation == 'gumbel':
            # Gumbel-softmax for discrete mask sampling
            mask = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        
        return mask
    
    def compute_sparsity_loss(self, mask, target_sparsity=0.3):
        """Encourage masks to be sparse"""
        current_sparsity = (mask < 0.5).float().mean()
        return F.mse_loss(current_sparsity, torch.tensor(target_sparsity))
```

---

### 3.3 Dual-Head Output Module

#### 3.3.1 Prediction Head

```python
class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=2):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, node_embeddings, node_mask=None):
        if node_mask is not None:
            # Apply node importance for interpretable prediction
            weighted_emb = node_embeddings * node_mask
        else:
            weighted_emb = node_embeddings
            
        logits = self.classifier(weighted_emb)
        return F.softmax(logits, dim=-1)
```

#### 3.3.2 Explanation Head

```python
class ExplanationHead(nn.Module):
    """
    Generates multi-granularity explanations
    """
    def __init__(self, hidden_dim, num_reason_codes=20):
        self.hidden_dim = hidden_dim
        self.num_reason_codes = num_reason_codes
        
        # Subgraph extractor
        self.subgraph_scorer = nn.Linear(hidden_dim, 1)
        
        # Reason code classifier
        self.reason_classifier = nn.Linear(hidden_dim, num_reason_codes)
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_emb, edge_emb, masks, subgraph_nodes):
        """
        Generate comprehensive explanation package
        """
        explanation = {}
        
        # 1. Top-k important nodes
        node_importance = masks['node_mask'].squeeze()
        top_k_nodes = torch.topk(node_importance, k=min(10, len(node_importance)))
        explanation['important_nodes'] = {
            'indices': top_k_nodes.indices,
            'scores': top_k_nodes.values
        }
        
        # 2. Top-k important edges
        all_edge_scores = torch.cat([m[1] for m in masks['edge_masks']])
        top_k_edges = torch.topk(all_edge_scores.squeeze(), k=min(20, len(all_edge_scores)))
        explanation['important_edges'] = {
            'indices': top_k_edges.indices,
            'scores': top_k_edges.values
        }
        
        # 3. Feature importance
        feat_importance = masks['feat_mask'].mean(dim=0)  # Average across nodes
        explanation['feature_importance'] = feat_importance
        
        # 4. Temporal importance
        explanation['temporal_importance'] = masks['temp_weights']
        
        # 5. Reason code probabilities
        pooled_emb = node_emb[top_k_nodes.indices].mean(dim=0)
        reason_logits = self.reason_classifier(pooled_emb)
        explanation['reason_codes'] = F.softmax(reason_logits, dim=-1)
        
        # 6. Explanation confidence
        explanation['confidence'] = self.confidence_net(pooled_emb)
        
        return explanation
```

---

### 3.4 Regulatory Compliance Module

This module bridges the gap between technical explanations and regulatory requirements.

#### 3.4.1 Reason Code Mapping

```python
# Predefined reason code templates (ECOA/FCRA compliant)
REASON_CODE_TEMPLATES = {
    'RC001': {
        'code': 'VELOCITY_ANOMALY',
        'template': 'Transaction frequency {velocity_ratio:.1f}x higher than historical pattern',
        'category': 'Behavioral',
        'regulatory_text': 'Unusual account activity pattern detected'
    },
    'RC002': {
        'code': 'NETWORK_RISK',
        'template': 'Connected to {num_risky} accounts with prior fraud indicators',
        'category': 'Network',
        'regulatory_text': 'Association with accounts exhibiting suspicious patterns'
    },
    'RC003': {
        'code': 'TEMPORAL_ANOMALY',
        'template': 'Transaction at {time} deviates from typical activity window ({usual_window})',
        'category': 'Temporal',
        'regulatory_text': 'Transaction timing inconsistent with established patterns'
    },
    'RC004': {
        'code': 'AMOUNT_DEVIATION',
        'template': 'Amount ${amount:.2f} is {deviation:.1f} std deviations from mean',
        'category': 'Transactional',
        'regulatory_text': 'Transaction amount outside normal range'
    },
    'RC005': {
        'code': 'NEW_RELATIONSHIP',
        'template': 'First transaction with {entity_type} {entity_id} (account age: {age} days)',
        'category': 'Relationship',
        'regulatory_text': 'New counterparty relationship'
    },
    'RC006': {
        'code': 'GEOGRAPHIC_ANOMALY',
        'template': 'Location {current_loc} inconsistent with profile location {profile_loc}',
        'category': 'Geographic',
        'regulatory_text': 'Geographic location mismatch'
    },
    'RC007': {
        'code': 'DEVICE_RISK',
        'template': 'Device {device_id} associated with {num_accounts} accounts',
        'category': 'Device',
        'regulatory_text': 'Device sharing pattern detected'
    },
    'RC008': {
        'code': 'STRUCTURING_PATTERN',
        'template': '{num_txns} transactions totaling ${total:.2f} within {window}',
        'category': 'Structuring',
        'regulatory_text': 'Transaction pattern consistent with structuring behavior'
    },
    # ... additional codes
}

class RegulatoryComplianceModule(nn.Module):
    """
    Converts technical explanations to regulatory-compliant format
    """
    def __init__(self, hidden_dim, num_reason_codes=20, top_k_reasons=4):
        self.reason_encoder = nn.Linear(hidden_dim, num_reason_codes)
        self.top_k = top_k_reasons  # ECOA requires up to 4 principal reasons
        
        # Trainable reason code embeddings
        self.reason_embeddings = nn.Embedding(num_reason_codes, hidden_dim)
        
        # Template parameter extractors
        self.param_extractors = nn.ModuleDict({
            'velocity': VelocityExtractor(hidden_dim),
            'network': NetworkRiskExtractor(hidden_dim),
            'temporal': TemporalExtractor(hidden_dim),
            'amount': AmountExtractor(hidden_dim),
        })
        
    def forward(self, explanation, node_features, edge_features, graph_context):
        """
        Generate regulatory-compliant explanation package
        """
        # Get reason code probabilities
        reason_probs = explanation['reason_codes']
        top_k_reasons = torch.topk(reason_probs, k=self.top_k)
        
        compliant_explanations = []
        
        for idx, (reason_idx, prob) in enumerate(zip(top_k_reasons.indices, top_k_reasons.values)):
            reason_code = list(REASON_CODE_TEMPLATES.keys())[reason_idx]
            template_info = REASON_CODE_TEMPLATES[reason_code]
            
            # Extract parameters for template
            params = self.extract_template_params(
                reason_code, 
                node_features, 
                edge_features, 
                graph_context,
                explanation
            )
            
            # Fill template
            filled_template = template_info['template'].format(**params)
            
            compliant_explanations.append({
                'rank': idx + 1,
                'code': reason_code,
                'probability': prob.item(),
                'category': template_info['category'],
                'technical_description': filled_template,
                'regulatory_text': template_info['regulatory_text'],
                'parameters': params
            })
        
        return {
            'principal_reasons': compliant_explanations,
            'gdpr_summary': self.generate_gdpr_summary(compliant_explanations),
            'audit_record': self.generate_audit_record(explanation, compliant_explanations)
        }
    
    def generate_gdpr_summary(self, reasons):
        """
        Generate GDPR Article 22 compliant explanation
        "meaningful information about the logic involved"
        """
        summary_parts = []
        for r in reasons[:3]:  # Top 3 for summary
            summary_parts.append(f"{r['category']}: {r['regulatory_text']}")
        
        return "This decision was based on: " + "; ".join(summary_parts) + "."
    
    def generate_audit_record(self, technical_exp, compliant_exp):
        """
        Generate complete audit trail for regulatory examination
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': self.model_version,
            'technical_explanation': {
                'node_importance': technical_exp['important_nodes'],
                'edge_importance': technical_exp['important_edges'],
                'feature_importance': technical_exp['feature_importance'].tolist(),
                'temporal_factors': technical_exp['temporal_importance'].tolist()
            },
            'regulatory_explanation': compliant_exp,
            'confidence_score': technical_exp['confidence'].item(),
            'explanation_method': 'RTXGNN_self_explainable'
        }
```

---

### 3.5 Natural Language Explanation Generator

```python
class NLExplanationGenerator(nn.Module):
    """
    Generates human-readable explanations from technical outputs
    Uses template-based generation with learned slot filling
    """
    
    EXPLANATION_TEMPLATES = {
        'high_risk': """
This transaction was flagged as potentially fraudulent (confidence: {confidence:.1%}) based on the following factors:

**Primary Indicators:**
{primary_reasons}

**Supporting Evidence:**
- Network Analysis: {network_summary}
- Temporal Pattern: {temporal_summary}
- Behavioral Deviation: {behavioral_summary}

**Recommendation:** {recommendation}
        """,
        
        'medium_risk': """
This transaction shows elevated risk indicators (confidence: {confidence:.1%}):

{reason_summary}

**Suggested Action:** {recommendation}
        """,
        
        'low_risk': """
Transaction approved. Minor anomalies noted:
{minor_anomalies}
        """
    }
    
    def generate(self, prediction, explanation, regulatory_output):
        risk_level = self.categorize_risk(prediction)
        template = self.EXPLANATION_TEMPLATES[risk_level]
        
        # Fill template slots
        filled = template.format(
            confidence=prediction['fraud_probability'],
            primary_reasons=self.format_reasons(regulatory_output['principal_reasons'][:2]),
            network_summary=self.summarize_network(explanation),
            temporal_summary=self.summarize_temporal(explanation),
            behavioral_summary=self.summarize_behavioral(explanation),
            recommendation=self.generate_recommendation(prediction, explanation),
            reason_summary=self.format_reasons(regulatory_output['principal_reasons']),
            minor_anomalies=self.format_minor_anomalies(explanation)
        )
        
        return filled
```

---

## 4. Training Procedure

### 4.1 Multi-Task Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{exp} + \lambda_2 \mathcal{L}_{sparse} + \lambda_3 \mathcal{L}_{temporal} + \lambda_4 \mathcal{L}_{reason}$$

#### Component Losses:

```python
class RTXGNNLoss(nn.Module):
    def __init__(self, lambda_exp=0.5, lambda_sparse=0.1, lambda_temp=0.2, lambda_reason=0.3):
        self.lambdas = {
            'exp': lambda_exp,
            'sparse': lambda_sparse, 
            'temp': lambda_temp,
            'reason': lambda_reason
        }
        
        # Class weights for imbalanced fraud detection
        self.pred_loss = FocalLoss(alpha=0.75, gamma=2.0)
        
    def forward(self, outputs, targets, explanation_targets=None):
        losses = {}
        
        # 1. Prediction Loss (Focal Loss for class imbalance)
        losses['pred'] = self.pred_loss(outputs['prediction'], targets['labels'])
        
        # 2. Explanation Fidelity Loss
        # Ensure masked prediction matches full prediction
        pred_full = outputs['prediction']
        pred_masked = outputs['prediction_masked']  # Prediction using only top-k important elements
        losses['exp'] = F.kl_div(
            F.log_softmax(pred_masked, dim=-1),
            F.softmax(pred_full, dim=-1),
            reduction='batchmean'
        )
        
        # 3. Sparsity Loss (encourage concise explanations)
        node_mask = outputs['masks']['node_mask']
        edge_masks = torch.cat([m[1] for m in outputs['masks']['edge_masks']])
        
        target_node_sparsity = 0.2  # Top 20% of nodes should be important
        target_edge_sparsity = 0.1  # Top 10% of edges
        
        losses['sparse'] = (
            F.mse_loss((node_mask > 0.5).float().mean(), torch.tensor(target_node_sparsity)) +
            F.mse_loss((edge_masks > 0.5).float().mean(), torch.tensor(target_edge_sparsity))
        )
        
        # 4. Temporal Consistency Loss
        # Recent frauds should attend to recent transactions
        temp_weights = outputs['masks']['temp_weights']
        if targets.get('fraud_timestamp') is not None:
            fraud_recency = compute_recency_to_fraud(temp_weights, targets['fraud_timestamp'])
            losses['temp'] = F.mse_loss(temp_weights, fraud_recency)
        else:
            losses['temp'] = torch.tensor(0.0)
        
        # 5. Reason Code Supervision (if ground truth available)
        if explanation_targets is not None and 'reason_codes' in explanation_targets:
            losses['reason'] = F.cross_entropy(
                outputs['explanation']['reason_codes'],
                explanation_targets['reason_codes']
            )
        else:
            # Self-supervised: reason codes should be consistent with explanation masks
            losses['reason'] = self.reason_consistency_loss(outputs)
        
        # Total weighted loss
        total_loss = losses['pred']
        for key in ['exp', 'sparse', 'temp', 'reason']:
            total_loss += self.lambdas[key] * losses[key]
        
        losses['total'] = total_loss
        return losses
```

### 4.2 Training Algorithm

```python
def train_rtxgnn(model, train_loader, val_loader, config):
    """
    Training procedure with curriculum learning and explanation refinement
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = RTXGNNLoss(**config.loss_weights)
    
    # Curriculum learning: start with prediction, gradually add explanation objectives
    curriculum_schedule = {
        0: {'pred': 1.0, 'exp': 0.0, 'sparse': 0.0, 'temp': 0.0, 'reason': 0.0},
        5: {'pred': 1.0, 'exp': 0.3, 'sparse': 0.05, 'temp': 0.1, 'reason': 0.0},
        10: {'pred': 1.0, 'exp': 0.5, 'sparse': 0.1, 'temp': 0.2, 'reason': 0.1},
        20: {'pred': 1.0, 'exp': 0.5, 'sparse': 0.1, 'temp': 0.2, 'reason': 0.3},
    }
    
    best_val_auc = 0
    best_explanation_fidelity = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        
        # Update loss weights based on curriculum
        current_weights = get_curriculum_weights(epoch, curriculum_schedule)
        loss_fn.update_weights(current_weights)
        
        epoch_losses = defaultdict(float)
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr,
                batch.edge_time,
                batch.batch
            )
            
            # Compute losses
            losses = loss_fn(outputs, batch.y, batch.get('explanation_labels'))
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v.item()
        
        # Validation
        val_metrics = validate(model, val_loader, loss_fn)
        
        # Logging
        log_metrics(epoch, epoch_losses, val_metrics)
        
        # Save best model (considering both prediction and explanation quality)
        combined_score = (
            0.7 * val_metrics['auc'] + 
            0.3 * val_metrics['explanation_fidelity']
        )
        if combined_score > best_val_auc + best_explanation_fidelity:
            save_checkpoint(model, optimizer, epoch, val_metrics)
            best_val_auc = val_metrics['auc']
            best_explanation_fidelity = val_metrics['explanation_fidelity']
        
        scheduler.step()
    
    return model
```

---

## 5. Inference Pipeline

### 5.1 Real-Time Inference Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REAL-TIME INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Transaction    ┌──────────┐   ┌───────────┐   ┌──────────────────────────┐ │
│  Stream ───────▶│  Graph   │──▶│  Feature  │──▶│  RTXGNN Inference        │ │
│                 │  Builder │   │  Cache    │   │  (GPU-accelerated)       │ │
│                 └──────────┘   └───────────┘   └───────────┬──────────────┘ │
│                                                            │                 │
│                                      ┌─────────────────────┴──────────────┐ │
│                                      ▼                                     │ │
│              ┌────────────────────────────────────────────────────────────┐│ │
│              │                    OUTPUT ROUTER                           ││ │
│              │  ┌─────────────┬─────────────┬─────────────────────────┐  ││ │
│              │  │ score < 0.3 │ 0.3 ≤ s < 0.7│     score ≥ 0.7         │  ││ │
│              │  │   APPROVE   │   REVIEW     │      DECLINE            │  ││ │
│              │  │ (no explain)│(light explain)│  (full explain)        │  ││ │
│              │  └─────────────┴─────────────┴─────────────────────────┘  ││ │
│              └────────────────────────────────────────────────────────────┘│ │
│                                                                              │
│  Async Explanation    ┌──────────────────────────────────────────────────┐  │
│  Queue ──────────────▶│  Batch Explanation Generator                     │  │
│                       │  (Regulatory Compliance + NL Generation)         │  │
│                       └──────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Optimized Inference Code

```python
class RTXGNNInference:
    """
    Production inference engine with tiered explainability
    """
    def __init__(self, model_path, config):
        self.model = self.load_optimized_model(model_path)
        self.feature_cache = FeatureCache(max_size=1_000_000)
        self.graph_builder = IncrementalGraphBuilder(config)
        self.explanation_queue = AsyncQueue(max_size=10_000)
        
        # Pre-computed entity embeddings (updated hourly)
        self.entity_embeddings = self.load_entity_embeddings()
        
        # Explanation level thresholds
        self.thresholds = {
            'no_explanation': 0.3,
            'light_explanation': 0.7,
            'full_explanation': 1.0
        }
        
    @torch.inference_mode()
    def predict(self, transaction: Transaction) -> PredictionResult:
        """
        Real-time prediction with adaptive explanation depth
        Target: < 50ms end-to-end
        """
        start_time = time.perf_counter()
        
        # Step 1: Build local subgraph (cached entity embeddings)
        subgraph = self.graph_builder.build_local_subgraph(
            transaction,
            hop_limit=2,
            max_neighbors=50,
            entity_embeddings=self.entity_embeddings
        )
        
        # Step 2: Feature extraction (with caching)
        features = self.feature_cache.get_or_compute(
            transaction.id,
            lambda: self.extract_features(transaction, subgraph)
        )
        
        # Step 3: Model inference
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = self.model.forward_fast(
                features['node_features'],
                features['edge_index'],
                features['edge_attr'],
                features['edge_time']
            )
        
        fraud_score = outputs['prediction'][0, 1].item()
        
        # Step 4: Tiered explanation generation
        if fraud_score < self.thresholds['no_explanation']:
            explanation = None
            explanation_level = 'none'
        elif fraud_score < self.thresholds['light_explanation']:
            # Light explanation: top 3 features + attention summary
            explanation = self.generate_light_explanation(outputs)
            explanation_level = 'light'
        else:
            # Queue for full explanation (async)
            self.explanation_queue.put({
                'transaction_id': transaction.id,
                'outputs': outputs,
                'subgraph': subgraph,
                'priority': 'high' if fraud_score > 0.9 else 'normal'
            })
            # Return quick explanation immediately
            explanation = self.generate_light_explanation(outputs)
            explanation_level = 'pending_full'
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResult(
            transaction_id=transaction.id,
            fraud_score=fraud_score,
            decision=self.get_decision(fraud_score),
            explanation=explanation,
            explanation_level=explanation_level,
            latency_ms=latency_ms,
            model_version=self.model.version
        )
    
    def generate_light_explanation(self, outputs) -> LightExplanation:
        """
        Quick explanation for real-time response (< 5ms overhead)
        """
        # Top-3 feature importance
        feat_importance = outputs['masks']['feat_mask'].mean(dim=0)
        top_features = torch.topk(feat_importance, k=3)
        
        # Attention-based risk factors
        attention_summary = self.summarize_attention(outputs['attention_weights'])
        
        # Pre-computed reason code (from classification head)
        top_reason = outputs['explanation']['reason_codes'].argmax().item()
        
        return LightExplanation(
            top_features=[
                (FEATURE_NAMES[idx], score.item()) 
                for idx, score in zip(top_features.indices, top_features.values)
            ],
            attention_summary=attention_summary,
            primary_reason=REASON_CODE_TEMPLATES[list(REASON_CODE_TEMPLATES.keys())[top_reason]]
        )
    
    async def process_explanation_queue(self):
        """
        Async worker for full explanation generation
        """
        while True:
            batch = await self.explanation_queue.get_batch(max_size=32, timeout=1.0)
            
            if not batch:
                continue
            
            # Batch process full explanations
            full_explanations = self.generate_full_explanations_batch(batch)
            
            # Store in explanation database
            await self.store_explanations(full_explanations)
            
            # Notify downstream systems (case management, audit)
            await self.notify_explanation_ready(full_explanations)
```

---

## 6. Evaluation Framework

### 6.1 Prediction Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | > 0.95 | Overall discrimination ability |
| AUC-PR | > 0.70 | Precision-recall for imbalanced data |
| Recall@5%FPR | > 0.80 | Fraud catch rate at operational FPR |
| F1 Score | > 0.75 | Harmonic mean of precision and recall |

### 6.2 Explanation Quality Metrics

```python
class ExplanationEvaluator:
    """
    Comprehensive explanation quality assessment
    Addresses Gap 1: Standardized explanation metrics for fraud
    """
    
    def evaluate(self, model, test_data, ground_truth_explanations=None):
        metrics = {}
        
        # 1. Fidelity (does explanation reflect model behavior?)
        metrics['fidelity_plus'] = self.compute_fidelity_plus(model, test_data)
        metrics['fidelity_minus'] = self.compute_fidelity_minus(model, test_data)
        
        # 2. Sparsity (are explanations concise?)
        metrics['sparsity'] = self.compute_sparsity(model, test_data)
        
        # 3. Stability (similar inputs → similar explanations?)
        metrics['stability'] = self.compute_stability(model, test_data)
        
        # 4. Consistency (explanation aligns with domain knowledge?)
        metrics['domain_consistency'] = self.compute_domain_consistency(model, test_data)
        
        # 5. Actionability (can analyst act on explanation?)
        metrics['actionability'] = self.compute_actionability(model, test_data)
        
        # 6. Regulatory Compliance Score
        metrics['compliance_score'] = self.compute_compliance_score(model, test_data)
        
        # 7. Ground truth alignment (if available)
        if ground_truth_explanations is not None:
            metrics['ground_truth_precision'] = self.compute_gt_precision(model, test_data, ground_truth_explanations)
            metrics['ground_truth_recall'] = self.compute_gt_recall(model, test_data, ground_truth_explanations)
        
        return metrics
    
    def compute_fidelity_plus(self, model, test_data):
        """
        Prediction change when keeping only important elements
        Higher = explanation captures essential information
        """
        fidelity_scores = []
        
        for batch in test_data:
            # Full prediction
            outputs_full = model(batch)
            pred_full = outputs_full['prediction']
            
            # Masked prediction (only important elements)
            masks = outputs_full['masks']
            outputs_masked = model.forward_with_mask(batch, masks, keep_important=True)
            pred_masked = outputs_masked['prediction']
            
            # Fidelity+ = 1 - |pred_full - pred_masked|
            fidelity = 1 - torch.abs(pred_full - pred_masked).mean()
            fidelity_scores.append(fidelity.item())
        
        return np.mean(fidelity_scores)
    
    def compute_fidelity_minus(self, model, test_data):
        """
        Prediction change when removing important elements
        Higher = important elements are truly important
        """
        fidelity_scores = []
        
        for batch in test_data:
            outputs_full = model(batch)
            pred_full = outputs_full['prediction']
            
            # Remove important elements
            masks = outputs_full['masks']
            outputs_removed = model.forward_with_mask(batch, masks, keep_important=False)
            pred_removed = outputs_removed['prediction']
            
            # Fidelity- = |pred_full - pred_removed|
            fidelity = torch.abs(pred_full - pred_removed).mean()
            fidelity_scores.append(fidelity.item())
        
        return np.mean(fidelity_scores)
    
    def compute_domain_consistency(self, model, test_data):
        """
        Fraud-specific: Do explanations align with known fraud patterns?
        """
        consistency_scores = []
        
        # Known fraud indicators
        fraud_indicators = {
            'velocity_spike': lambda x: x['velocity_1h'] > 3 * x['velocity_24h'],
            'new_device': lambda x: x['device_age_days'] < 1,
            'unusual_amount': lambda x: abs(x['amount'] - x['avg_amount']) > 3 * x['std_amount'],
            'night_transaction': lambda x: x['hour'] < 6 or x['hour'] > 22,
            'foreign_ip': lambda x: x['ip_country'] != x['account_country'],
        }
        
        for batch in test_data:
            outputs = model(batch)
            
            # Check if known indicators are highlighted when present
            for indicator_name, indicator_fn in fraud_indicators.items():
                indicator_present = indicator_fn(batch.node_features)
                indicator_highlighted = self.check_indicator_in_explanation(
                    indicator_name, outputs['explanation']
                )
                
                if indicator_present.any():
                    # Should be highlighted for fraud cases
                    fraud_mask = batch.y == 1
                    consistency = (indicator_highlighted[fraud_mask & indicator_present]).float().mean()
                    consistency_scores.append(consistency.item())
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def compute_compliance_score(self, model, test_data):
        """
        Regulatory compliance assessment
        """
        compliance_checks = {
            'has_principal_reasons': 0,
            'reasons_count_valid': 0,  # 1-4 reasons per ECOA
            'reasons_are_specific': 0,
            'gdpr_summary_present': 0,
            'audit_trail_complete': 0,
        }
        
        total_samples = 0
        
        for batch in test_data:
            outputs = model(batch)
            regulatory_output = model.regulatory_module(outputs['explanation'])
            
            for i in range(len(batch)):
                total_samples += 1
                
                # Check compliance criteria
                reasons = regulatory_output['principal_reasons'][i]
                
                if len(reasons) > 0:
                    compliance_checks['has_principal_reasons'] += 1
                
                if 1 <= len(reasons) <= 4:
                    compliance_checks['reasons_count_valid'] += 1
                
                if all(r['parameters'] for r in reasons):
                    compliance_checks['reasons_are_specific'] += 1
                
                if regulatory_output['gdpr_summary'][i]:
                    compliance_checks['gdpr_summary_present'] += 1
                
                if self.validate_audit_record(regulatory_output['audit_record'][i]):
                    compliance_checks['audit_trail_complete'] += 1
        
        # Compute overall compliance score
        compliance_score = sum(
            v / total_samples for v in compliance_checks.values()
        ) / len(compliance_checks)
        
        return compliance_score
```

### 6.3 Human Evaluation Protocol

```python
class HumanEvaluationProtocol:
    """
    Protocol for human-centered evaluation with fraud analysts
    Addresses Gap 2: Human-centered evaluation
    """
    
    EVALUATION_DIMENSIONS = {
        'understandability': {
            'question': 'How easy was it to understand why this transaction was flagged?',
            'scale': (1, 5),
            'anchors': {1: 'Very difficult', 3: 'Neutral', 5: 'Very easy'}
        },
        'completeness': {
            'question': 'Does the explanation cover all relevant factors for this decision?',
            'scale': (1, 5),
            'anchors': {1: 'Missing critical factors', 3: 'Adequate', 5: 'Comprehensive'}
        },
        'actionability': {
            'question': 'Based on this explanation, how confident are you in taking action?',
            'scale': (1, 5),
            'anchors': {1: 'Need more info', 3: 'Somewhat confident', 5: 'Very confident'}
        },
        'trust': {
            'question': 'How much do you trust this explanation?',
            'scale': (1, 5),
            'anchors': {1: 'Do not trust', 3: 'Neutral', 5: 'Fully trust'}
        },
        'efficiency': {
            'question': 'Did this explanation help you reach a decision faster?',
            'scale': (1, 5),
            'anchors': {1: 'Slowed me down', 3: 'No difference', 5: 'Much faster'}
        }
    }
    
    def design_study(self, num_analysts=20, cases_per_analyst=50):
        """
        Design human evaluation study
        """
        study_design = {
            'participants': {
                'target_n': num_analysts,
                'criteria': [
                    'Minimum 2 years fraud investigation experience',
                    'Currently active in fraud operations',
                    'Mix of seniority levels'
                ],
                'recruitment': 'Partner financial institutions'
            },
            'conditions': {
                'A': 'No explanation (baseline)',
                'B': 'Feature importance only',
                'C': 'RTXGNN full explanation',
                'D': 'RTXGNN + natural language'
            },
            'design': 'Within-subjects, counterbalanced',
            'cases': {
                'total_per_analyst': cases_per_analyst,
                'fraud_ratio': 0.5,  # Balanced for evaluation
                'difficulty_distribution': {
                    'easy': 0.3,
                    'medium': 0.4,
                    'hard': 0.3
                }
            },
            'metrics': {
                'objective': [
                    'Decision accuracy',
                    'Decision time',
                    'Decision confidence calibration'
                ],
                'subjective': list(self.EVALUATION_DIMENSIONS.keys())
            },
            'analysis_plan': {
                'primary': 'Mixed-effects regression with analyst random effects',
                'secondary': 'Qualitative analysis of think-aloud protocols'
            }
        }
        
        return study_design
```

---

## 7. Complexity Analysis

### 7.1 Time Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Graph construction | O(E) | Linear in edges |
| SEAL layer (L layers) | O(L × E × d²) | E edges, d hidden dim |
| Mask generation | O(N × d) | N nodes |
| Attention aggregation | O(E × H × d/H) | H attention heads |
| Explanation head | O(N × d) | Linear |
| Regulatory mapping | O(K) | K reason codes |
| **Total inference** | **O(L × E × d²)** | Dominated by GNN layers |

### 7.2 Space Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Node embeddings | O(N × d) | Per layer |
| Edge embeddings | O(E × d_e) | Transaction features |
| Attention matrices | O(E × H) | Sparse storage |
| Explanation masks | O(N + E + d) | Node + edge + feature masks |
| **Total** | **O(N × d × L + E × d_e)** | L layers |

### 7.3 Latency Breakdown (Target System)

| Stage | Target Latency | Optimization |
|-------|---------------|--------------|
| Graph construction | < 5ms | Pre-computed entity embeddings |
| Feature extraction | < 3ms | Feature caching |
| GNN inference | < 15ms | TensorRT optimization, INT8 |
| Mask generation | < 5ms | Fused kernels |
| Light explanation | < 2ms | Template matching |
| **Total (real-time)** | **< 30ms** | GPU inference |
| Full explanation | < 200ms | Async processing |

---

## 8. Full Algorithm Pseudocode

```
Algorithm: RTXGNN Training and Inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAINING PHASE:

Input: Training graph G = (V, E, X, E_attr, timestamps), Labels Y
Output: Trained model θ

1. Initialize model parameters θ randomly
2. Initialize curriculum schedule C = {epoch → loss_weights}

3. FOR epoch = 1 to max_epochs:
   
   4. weights ← GetCurriculumWeights(epoch, C)
   
   5. FOR each batch B in DataLoader(G):
      
      6. // Forward pass with explanation generation
      7. FOR l = 1 to L layers:
         
         8. // Temporal encoding
         9. T_enc ← HRAPE(B.timestamps)
         
         10. // Feature mask generation
         11. M_feat ← σ(MLP_feat(H^{l-1}))
         12. H_masked ← H^{l-1} ⊙ M_feat
         
         13. // Relation-aware message passing with edge masks
         14. FOR each relation r in R:
            15. E_r ← GetEdgesOfType(B.edge_index, r)
            16. M_edge^r ← σ(MLP_edge(concat(H[src], H[tgt])))
            17. W_temp ← TemporalScorer(T_enc[E_r])
            18. MSG_r ← W_r(H_masked[src]) ⊙ M_edge^r ⊙ W_temp
         
         19. // Attention aggregation
         20. H^l ← MultiHeadAttention(H_masked, MSG, T_enc)
         
         21. // Node mask generation
         22. M_node^l ← σ(MLP_node(H^l))
      
      23. // Dual-head output
      24. pred ← PredictionHead(H^L ⊙ M_node^L)
      25. exp ← ExplanationHead(H^L, masks)
      
      26. // Compute multi-task loss
      27. L_pred ← FocalLoss(pred, Y[B])
      28. L_exp ← KL(pred_masked, pred_full)
      29. L_sparse ← SparsityLoss(M_node, M_edge, targets)
      30. L_temp ← TemporalConsistencyLoss(W_temp, fraud_timestamps)
      31. L_reason ← ReasonCodeLoss(exp.reason_codes, labels)
      
      32. L_total ← weights.pred × L_pred + weights.exp × L_exp + 
                   weights.sparse × L_sparse + weights.temp × L_temp +
                   weights.reason × L_reason
      
      33. // Backward pass
      34. θ ← θ - η∇_θ L_total

   35. // Validation and checkpoint
   36. IF Validate(θ) improves THEN SaveCheckpoint(θ)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INFERENCE PHASE:

Input: Transaction t, Trained model θ, Entity embeddings E_cache
Output: Prediction p, Explanation X, Regulatory output R

1. // Build local subgraph (using cached embeddings)
2. subgraph ← BuildLocalSubgraph(t, hop=2, max_neighbors=50, E_cache)

3. // Feature extraction with caching
4. features ← FeatureCache.GetOrCompute(t.id, ExtractFeatures(t, subgraph))

5. // Fast forward pass (FP16)
6. WITH mixed_precision:
   7. outputs ← model.forward_fast(features)

8. fraud_score ← outputs.prediction[0, 1]

9. // Tiered explanation
10. IF fraud_score < 0.3:
    11. explanation ← None
    12. level ← "none"
    
13. ELSE IF fraud_score < 0.7:
    14. explanation ← GenerateLightExplanation(outputs)
    15. level ← "light"
    
16. ELSE:
    17. explanation ← GenerateLightExplanation(outputs)
    18. level ← "pending_full"
    19. AsyncQueue.Put(t.id, outputs, priority="high" if fraud_score > 0.9)

20. // Regulatory compliance mapping
21. IF level != "none":
    22. R ← RegulatoryModule(explanation)
    23. R.gdpr_summary ← GenerateGDPRSummary(R.principal_reasons)
    24. R.audit_record ← GenerateAuditRecord(outputs, R)

25. RETURN {
       score: fraud_score,
       decision: GetDecision(fraud_score),
       explanation: explanation,
       regulatory: R,
       latency: elapsed_time
    }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ASYNC EXPLANATION WORKER:

1. LOOP forever:
   2. batch ← AsyncQueue.GetBatch(max_size=32, timeout=1s)
   
   3. IF batch not empty:
      4. // Full explanation generation
      5. FOR each item in batch:
         6. full_exp ← GenerateFullExplanation(item.outputs, item.subgraph)
         7. nl_exp ← NLGenerator.Generate(full_exp)
         8. regulatory ← RegulatoryModule.FullProcess(full_exp)
         
      9. // Store and notify
      10. ExplanationDB.Store(batch, full_explanations)
      11. NotifyDownstream(batch.transaction_ids)
```

---

## 9. Research Contributions Summary

| Gap Identified | RTXGNN Solution | Novelty |
|---------------|-----------------|---------|
| **Gap 1**: No standardized explanation metrics | Comprehensive evaluation framework with fraud-specific metrics (domain consistency, actionability, compliance score) | First fraud-specific XAI evaluation protocol |
| **Gap 2**: No human-centered evaluation | Built-in human evaluation protocol with validated dimensions | Integration of HCI methods in fraud GNN |
| **Gap 3**: Temporal explainability nascent | HRAPE + temporal attention + recency-weighted masks | First temporal explanation for fraud GNNs |
| **Gap 4**: No ground-truth datasets | Self-supervised explanation learning + synthetic benchmark generation | Reduces dependence on unavailable ground truth |
| **Gap 5**: Regulatory non-compliance | Dedicated regulatory module mapping to GDPR/ECOA requirements | First GNN with built-in regulatory compliance |
| **Gap 6**: Scalability limitations | Tiered explanation + async processing + cached embeddings | Production-ready architecture < 50ms |

---

## 10. Implementation Roadmap

### Phase 1: Core Model (Months 1-2)
- [ ] Implement SEAL layers with mask generation
- [ ] Implement HRAPE temporal encoding
- [ ] Implement dual-head output module
- [ ] Basic training loop with curriculum learning

### Phase 2: Explanation & Compliance (Months 2-3)
- [ ] Implement regulatory compliance module
- [ ] Implement natural language generator
- [ ] Define reason code templates
- [ ] Build evaluation framework

### Phase 3: Production Optimization (Months 3-4)
- [ ] TensorRT optimization
- [ ] Implement tiered inference pipeline
- [ ] Build async explanation worker
- [ ] Feature caching system

### Phase 4: Evaluation & Paper (Months 4-5)
- [ ] Benchmark on public datasets (YelpChi, Amazon, Elliptic)
- [ ] Conduct human evaluation study
- [ ] Ablation studies
- [ ] Write paper targeting ACI

---

## References

[Key references to be added based on literature review]

---

*Document Version: 1.0*
*Last Updated: November 2024*
*Target Venue: Emerald Applied Computing and Informatics (ACI)*
