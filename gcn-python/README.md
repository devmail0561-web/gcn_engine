# gcn-python — GCN Causal Engine (Python ML layers)

[![PyPI version](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](https://pypi.org/project/gcn-python/)
[![Python](https://img.shields.io/pypi/pyversions/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-206-passing)](tests/)

**Extract, model and reason over causal structure in natural language and code.**

---

## What it does

`gcn-python` is a **trainable causal extraction engine**. Given a sentence, it produces a `CausalIR` — a typed directed graph describing who causes what, under which conditions, and with what consequences.

```
"Les ventes baissent car la demande recule."
          │
          ▼
CausalIR {
  nodes: [
    {type: "processus", label: "baisser(ventes)"},
    {type: "processus", label: "reculer(demande)"}
  ],
  edges: [
    {source: n001, target: n002, relation: "cause", confidence: 0.87}
  ]
}
```

Unlike LLM-based approaches, the output is a **verifiable, serializable, queryable structure** — not prose.

---

## Installation

```bash
pip install gcn-python

# With PyTorch support (RGCNLayerPT, RGCNLayerGAT):
pip install "gcn-python[torch]"
```

---

## Quick start

### 1. Inference with a pre-trained checkpoint

```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.training.checkpoint import load_checkpoint

# Build pipeline (must match checkpoint architecture)
vocab   = FeatureVocabulary()
d_eff   = vocab.d_clause          # 79 features per clause
d_edge  = vocab.d_edge_closed_loop(d_eff, 7)
encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph   = RGCNLayer(d_in=d_eff, d_out=d_eff)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

# Load trained weights
load_checkpoint(pipeline, "model.npz")
pipeline.encoder.training = False

# Build UDRepresentation for each clause (from your UD parser)
clause1 = UDRepresentation(
    tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {"Tense": "Pres"}}],
    root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
    root_morph={"Tense": "Pres"}, subject_pos="NOUN",
    has_object=False, has_advcl=False, has_temporal_obl=False,
    token_span=(1, 3),
)
clause2 = UDRepresentation(
    tokens=[{"lemma": "reculer", "pos": "VERB", "dep_rel": "root", "morph": {"Tense": "Pres"}}],
    root_lemma="reculer", root_pos="VERB", root_dep_rel="root",
    root_morph={"Tense": "Pres"}, subject_pos="NOUN",
    has_object=False, has_advcl=False, has_temporal_obl=False,
    token_span=(5, 8),
)

# Inference
cir = pipeline.forward([clause1, clause2], "Les ventes baissent car la demande recule.")
print(cir["nodes"][0]["node_type"])   # "processus"
print(cir["edges"][0][2]["relation"]) # "cause"
```

### 2. Training on your own annotated corpus

```bash
# Prepare your dataset in gcn-nl JSON format (see gcn-datasets/examples/)
gcn-train \
  --data-dir  my_corpus/train/ \
  --val-dir   my_corpus/val/ \
  --epochs    100 \
  --lr        0.001 \
  --use-attention \
  --bidirectional \
  --embedding-dim 50 \
  --edge-loss-weight 2.0 \
  --weight-decay 0.001 \
  --patience 20 \
  --output    model.npz \
  --log-csv   training.csv
```

### 3. Training in Python

```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.training.checkpoint import save_checkpoint
import numpy as np

vocab   = FeatureVocabulary()
d_eff   = vocab.d_clause
d_edge  = vocab.d_edge_closed_loop(d_eff, 7)
encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph   = RGCNLayer(d_in=d_eff, d_out=d_eff)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

loader = GCNDataLoader("my_corpus/train/")

for epoch in range(50):
    for sample in loader:
        reps, idxs, conn = reps_from_sentence(sample.sentence)
        if not reps:
            continue
        pipeline.forward(reps, sample.sentence.text, clause_positions=idxs,
                         n_total_clauses=len(sample.sentence.clauses),
                         connector_reps=conn)
        node_logits = pipeline._cached_node_logits
        edge_logits = pipeline._cached_edge_logits
        gold_node   = sample.gold_node_labels[np.array(idxs)]

        loss, d_node, d_edge = pipeline.loss(node_logits, edge_logits, gold_node)
        pipeline.backward(d_node, d_edge, lr=0.001)

save_checkpoint(pipeline, "model.npz")
```

### 4. Substituting your own encoder (PyTorch, JAX…)

```python
from gcn_python.layer2.interface import CausalEncoder
import torch, torch.nn as nn

class MyTransformerEncoder(nn.Module, CausalEncoder):
    """Replace the reference MLP with your own architecture."""
    def __init__(self, d_clause: int, d_edge: int):
        super().__init__()
        self.node_head = nn.Linear(d_clause, 7)
        self.edge_head = nn.Linear(d_edge, 11)

    def forward_node(self, x):
        return self.node_head(torch.tensor(x)).detach().numpy()

    def forward_edge(self, x):
        return self.edge_head(torch.tensor(x)).detach().numpy()

    def parameters(self):
        return [p.detach().numpy() for p in super().parameters()]

# Plug into the pipeline — everything else stays the same
encoder = MyTransformerEncoder(vocab.d_clause, d_edge)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
```

---

## Dataset format (gcn-nl)

```json
{
  "document": {
    "sentences": [
      {
        "id": "s001",
        "text": "Les ventes baissent car la demande recule.",
        "causal_pattern": "cause",
        "tokens": [
          {"id": 1, "form": "Les",    "lemma": "le",      "pos": "DET",  "dep_rel": "det",   "dep_head": 2, "morph": {}},
          {"id": 2, "form": "ventes", "lemma": "vente",   "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 3, "morph": {}},
          {"id": 3, "form": "baissent","lemma":"baisser", "pos": "VERB", "dep_rel": "root",  "dep_head": 0, "morph": {"Tense": "Pres"}}
        ],
        "cir": {
          "nodes": [
            {"id": "n001", "type": "processus", "label": "baisser(ventes)", "token_span": [1, 3]},
            {"id": "n002", "type": "processus", "label": "reculer(demande)", "token_span": [5, 8]}
          ],
          "edges": [
            {"source": "n001", "target": "n002", "relation": "cause",
             "attributes": {"confidence": 1.0, "explicit": true, "negated": false}}
          ]
        }
      }
    ]
  }
}
```

**Node types (7):** `etat` · `action` · `transition` · `processus` · `condition` · `entite` · `etat_systemique`

**Edge relations (11):** `cause` · `enable` · `prevent` · `condition` · `concession` · `sequence` · `motivation` · `filter` · `opposition` · `data_dependency` · `control_dependency`

---

## Architecture

```
UD tokens (pos, dep_rel, morph, lemma)
          │
   Layer 1 — FeatureVocabulary
   vectorize_clause() → 79-dim vector per clause
          │
   Layer 2 — MLPEncoder  (replaceable)
   forward_node()  →  node_logits  (N × 7)
   forward_edge()  →  edge_logits  (E × 11)
          │
   Layer 3 — RGCNLayer  (replaceable)
   message_pass()  →  enriched clause vectors
          │
   CGNPipeline.forward() → CausalIR dict
```

**Extensibility:** replace `MLPEncoder` with any `CausalEncoder` implementation (PyTorch, JAX, ONNX). Replace `RGCNLayer` with `RGCNLayerGAT` (attention) or your own `CausalGraph`.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `gcn-train --data-dir DIR --epochs N --output model.npz` | Train on annotated corpus |
| `gcn-train --val-dir DIR --patience 20` | With validation + early stopping |
| `gcn-train --use-attention --bidirectional` | GAT + 22-relation message passing |
| `gcn-train --embedding-dim 50` | Learnable word embeddings |
| `gcn-train --weight-decay 0.001 --rgcn-dropout 0.1 --label-smoothing 0.05` | Regularization |
| `gcn-eval --data-dir DIR --checkpoint model.npz` | Evaluate checkpoint |
| `gcn-forward --text "..." --checkpoint model.npz` | Single-sentence inference |

---

## Evaluation metrics

```python
from gcn_python.evaluation.metrics import (
    node_macro_f1,        # macro-averaged F1 over 7 node types
    edge_macro_f1,        # macro-averaged F1 over 11 relation types
    graph_exact_match,    # fraction of sentences with fully correct graph
    confusion_matrix,     # N×N confusion matrix
    per_class_report,     # precision / recall / F1 per class
)
```

---

## License

MIT — see [LICENSE](../LICENSE)

## Author

Michel Tendeng — GCN Causal Engine
