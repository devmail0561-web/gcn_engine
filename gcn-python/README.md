# gcn-python — GCN Causal Engine

[![PyPI version](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![Version](https://img.shields.io/badge/version-2.1.1-blue.svg)](https://pypi.org/project/gcn-python/)
[![Python](https://img.shields.io/pypi/pyversions/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-206-passing)](tests/)

---

## Vision

Les LLMs et les Transformers ont révolutionné la compréhension du texte naturel. Leur
limite fondamentale : ils produisent du texte **plausible**, pas de la connaissance
**vérifiable**. Quand un LLM dit *"X cause Y"*, on ne peut pas prouver que c'est
correct, l'interroger formellement, le combiner avec d'autres affirmations, ni raisonner
dessus au sens de Pearl.

**GCN Causal Engine vise le même domaine d'usage que les LLMs — comprendre et raisonner
sur du texte — avec une approche radicalement différente : produire une structure causale
vérifiable, interrogeable et raisonnnable au lieu de texte probabiliste.**

```
Texte brut (toute langue, tout domaine)
          │
          ▼
    GCN Causal Engine
          │
          ▼
CausalIR — graphe causal structuré
  ├── Nœuds typés (7 types : état, action, processus…)
  ├── Relations typées (11 types : cause, enable, condition…)
  ├── Confiance par arête
  ├── Détection de cycles
  └── Origine (explicite / inférée / hypothétique)
          │
          ├── Interrogeable via GCN-QL
          ├── Raisonnable via Pearl (niveaux 1-2-3)
          ├── Verbalisable → texte explicatif
          └── Sérialisable JSON → intégrable dans tout système
```

### Ce que GCN produit qu'un LLM ne peut pas garantir

| Capacité | LLM | GCN Causal Engine |
|----------|-----|-------------------|
| Comprend la causalité dans le texte | ✅ (approximatif) | ✅ (structuré) |
| Sortie vérifiable et auditable | ❌ (texte plausible) | ✅ (JSON typé) |
| Requêtes formelles sur la structure | ❌ | ✅ (GCN-QL) |
| Raisonnement interventionnel (Pearl) | ❌ | ✅ (do-calculus) |
| Combinaison de plusieurs CIR | ❌ | ✅ (graphes composables) |
| Détection de contradictions causales | ❌ | ✅ |
| Entraînement sur corpus spécifique | Coûteux (fine-tuning LLM) | ✅ (léger, NumPy pur) |

### Pour qui ?

- **Chercheurs** : extraire des claims causaux de la littérature scientifique
- **Ingénieurs** : analyser les dépendances causales dans du code source (Python, Rust, JS)
- **Data scientists** : construire des modèles causaux vérifiables pour l'XAI
- **Juristes / médecins** : tracer des chaînes causales dans des documents complexes
- **Tout système** qui a besoin de comprendre *pourquoi* quelque chose se produit,
  pas seulement *quoi*

---

## Installation

```bash
pip install gcn-python

# Avec support PyTorch (RGCNLayerGAT) :
pip install "gcn-python[torch]"
```

---

## Usage

### 1. Inférence depuis du texte brut — API haut niveau

```python
from gcn_python import GCNEngine

# Charger depuis un checkpoint — l'architecture est déduite automatiquement
engine = GCNEngine.from_pretrained("model.npz")

# Analyser une phrase
cir = engine.analyze("Les ventes baissent car la demande recule.")
print(cir["nodes"][0]["node_type"])        # "processus"
print(cir["edges"][0][2]["relation"])      # "cause"
print(cir["edges"][0][2]["confidence"])    # 0.87

# Analyser plusieurs phrases
cirs = engine.analyze_batch([
    "Les prix augmentent car la demande dépasse l'offre.",
    "Si les températures montent, la banquise fond.",
    "Bien que les coûts soient élevés, la production continue.",
])

# Itérer sur un fichier (mémoire constante)
with open("corpus.txt") as f:
    for cir in engine.stream(f):
        if cir["edges"]:
            print(cir["source_text"], "→", cir["edges"][0][2]["relation"])
```

### 2. Interroger le CIR via GCN-QL (gcn-backend)

```bash
# Depuis la ligne de commande Rust
gcn analyze "Les ventes baissent car la demande recule." | \
  gcn query "SELECT causes OF ventes"
```

### 3. Entraîner sur votre corpus

```bash
# Préparer vos données annotées au format gcn-nl (voir section Dataset)
gcn-train \
  --data-dir  corpus/train/ \
  --val-dir   corpus/val/ \
  --epochs    100 \
  --use-attention \
  --bidirectional \
  --embedding-dim 50 \
  --output    model.npz
```

### 4. Entraînement en Python

```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.training.checkpoint import save_checkpoint
import numpy as np

vocab    = FeatureVocabulary()
d_eff    = vocab.d_clause
d_edge   = vocab.d_edge_closed_loop(d_eff, 7)
encoder  = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph    = RGCNLayer(d_in=d_eff, d_out=d_eff)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

loader = GCNDataLoader("corpus/train/")
for epoch in range(100):
    for sample in loader:
        reps, idxs, conn = reps_from_sentence(sample.sentence)
        if not reps:
            continue
        pipeline.forward(reps, sample.sentence.text,
                         clause_positions=idxs,
                         n_total_clauses=len(sample.sentence.clauses),
                         connector_reps=conn)
        loss, d_node, d_edge = pipeline.loss(
            pipeline._cached_node_logits,
            pipeline._cached_edge_logits,
            sample.gold_node_labels[np.array(idxs)],
        )
        pipeline.backward(d_node, d_edge, lr=0.001)

save_checkpoint(pipeline, "model.npz")
```

### 5. Substituer votre propre encodeur (PyTorch, JAX…)

```python
from gcn_python.layer2.interface import CausalEncoder

class MyEncoder(CausalEncoder):
    """Remplacez le MLP de référence par votre architecture."""
    def forward_node(self, x): ...
    def forward_edge(self, x): ...
    def parameters(self): ...

pipeline = CGNPipeline(encoder=MyEncoder(), graph=graph, vocabulary=vocab)
```

---

## Format de données (gcn-nl)

```json
{
  "document": {
    "sentences": [{
      "id": "s001",
      "text": "Les ventes baissent car la demande recule.",
      "causal_pattern": "cause",
      "tokens": [
        {"id": 1, "form": "Les", "lemma": "le", "pos": "DET",
         "dep_rel": "det", "dep_head": 2, "morph": {}}
      ],
      "cir": {
        "nodes": [
          {"id": "n001", "type": "processus", "label": "baisser(ventes)",
           "token_span": [1, 3]},
          {"id": "n002", "type": "processus", "label": "reculer(demande)",
           "token_span": [5, 8]}
        ],
        "edges": [{
          "source": "n001", "target": "n002", "relation": "cause",
          "attributes": {"confidence": 1.0, "explicit": true, "negated": false}
        }]
      }
    }]
  }
}
```

**7 types de nœuds :** `etat` · `action` · `transition` · `processus` · `condition` · `entite` · `etat_systemique`

**11 types de relations :** `cause` · `enable` · `prevent` · `condition` · `concession` · `sequence` · `motivation` · `filter` · `opposition` · `data_dependency` · `control_dependency`

---

## Architecture

```
Texte brut (fr/en/code)
     │
     ▼  [gcn-frontend-fr / gcn-frontend-en / gcn-frontend-code — Rust]
UD tokens (pos, dep_rel, morph, lemma)
     │
     ▼  Layer 1 — FeatureVocabulary (gcn-python)
79-dim vector par clause
     │
     ▼  Layer 2 — MLPEncoder (remplaçable)
node_logits (N×7) + edge_logits (E×11)
     │
     ▼  Layer 3 — RGCNLayer / RGCNLayerGAT (remplaçable)
message passing — enrichissement des représentations
     │
     ▼  CGNPipeline.forward()
CausalIR (JSON)
     │
     ├──▶ gcn-middleend : construction graphe, cycles, validation
     ├──▶ gcn-backend   : Pearl reasoning, GCN-QL
     └──▶ gcn-verbalizer : CausalIR → texte explicatif
```

---

## Métriques d'évaluation

```python
from gcn_python.evaluation.metrics import (
    node_macro_f1,     # F1 macro sur les 7 types de nœuds
    edge_macro_f1,     # F1 macro sur les 11 types de relations
    graph_exact_match, # fraction de phrases avec graphe complet correct
    confusion_matrix,  # matrice de confusion N×N
    per_class_report,  # précision / rappel / F1 par classe
)
```

---

## Stack complet

| Package | Rôle | Langage |
|---------|------|---------|
| `gcn-python` | Couches ML (MLP + R-GCN), entraînement, évaluation | Python |
| `gcn-ir` | Types fondamentaux CausalIR | Rust |
| `gcn-knowledge` | Taxonomies, lexique causal, inférence | Rust |
| `gcn-frontend-fr` | Parser causal français | Rust |
| `gcn-frontend-en` | Parser causal anglais | Rust |
| `gcn-frontend-code` | Parser code source (Python/Rust/JS) | Rust |
| `gcn-middleend` | Graphe causal, cycles, propagation | Rust |
| `gcn-backend` | Pearl niveaux 1-2-3, GCN-QL, export | Rust |
| `gcn-verbalizer` | CausalIR → texte | Rust/Python |

---

## Licence

MIT — © Michel Tendeng
