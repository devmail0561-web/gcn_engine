# gcn-python — GCN Causal Engine

[![PyPI version](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![Version](https://img.shields.io/badge/version-2.1.1-blue.svg)](https://pypi.org/project/gcn-python/)
[![Python](https://img.shields.io/pypi/pyversions/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-206-passing)](tests/)

---

## Ce qu'est GCN

GCN est un **moteur d'extraction et de raisonnement causal vérifiable**.

Il prend du texte brut, extrait la structure causale, et répond à des questions sur cette structure — avec traçabilité jusqu'aux sources.

```
Texte brut (FR, EN, code — toute langue)
          │
          ▼  GCN Causal Engine
          │
          ▼
CausalIR — graphe causal structuré
  ├── Nœuds typés  (7 types : etat, action, processus…)
  ├── Relations typées  (11 types : cause, enable, prevent…)
  ├── Confiance par arête
  └── Source exacte par relation
          │
          ├── Interrogeable : "what causes X ?"
          ├── Traceable    : "selon quel document ?"
          ├── Pearl niveau 2 : "sans X, que se passe-t-il ?"
          └── Contradiction detection entre sources
```

**Ce que GCN fait qu'un LLM ne garantit pas :**

| | LLM | GCN |
|-|-----|-----|
| Réponse causale | Plausible, non vérifiable | Tracée jusqu'à la source |
| Requêtes formelles sur le graphe | ❌ | ✅ |
| Détection de contradictions entre sources | ❌ | ✅ |
| Raisonnement Pearl (do-calculus) | ❌ | ✅ |
| Entraînable sur corpus spécifique | Coûteux | ✅ léger (NumPy) |

**Pour qui :**
- Analystes CTI / SOC — chaînes d'attaque depuis des rapports de menace
- Auditeurs — obligations causales dans des référentiels réglementaires
- Investigateurs — chaînes de responsabilité depuis des dossiers
- Chercheurs — extraction de claims causaux depuis la littérature
- Ingénieurs — dépendances causales dans le code source

---

## Installation

```bash
pip install gcn-python

# Avec support PyTorch (RGCNLayerGAT) :
pip install "gcn-python[torch]"
```

---

## Usage

### 1. Session interactive — analyser des fichiers et poser des questions

```bash
gcn-discuss --checkpoint model.npz
```

```
  GCN Causal Engine
  ─────────────────────────────────────────────────────
  Corpus : vide  —  utilisez /analyze pour charger des documents

  > /analyze incident_report.txt
    incident_report.txt : 47 relation(s) extraite(s)

  > /analyze threat_reports/
    APT28_2024.txt   : 23 relation(s)
    Mandiant_Q3.txt  : 31 relation(s)
    Total session : 101 relation(s)

  > What causes data exfiltration?
    causes_of: 'data exfiltration'
    ────────────────────────────────────────────────────
    1. [action] authentication_bypass  --[enable]-->  (conf=91%)
       source: APT28_2024.txt
    2. [processus] credential_theft  --[cause]-->  (conf=87%)
       source: Mandiant_Q3.txt
    ⚠ CONTRADICTION : firewall_rule --[prevent]--> (SecPolicy.txt)
    ────────────────────────────────────────────────────

  > /save session.json
    Graphe sauvegardé : 101 relations

  > /quit

# Reprendre la session précédente
gcn-discuss --checkpoint model.npz --graph session.json
```

**Commandes disponibles :**

| Commande | Description |
|----------|-------------|
| `/analyze <fichier_ou_répertoire>` | Analyser du texte brut, enrichir le graphe |
| `/save <path.json>` | Persister le graphe de session |
| `/load <path.json>` | Charger un graphe existant |
| `/summarize` | Résumé du corpus courant |
| `/help` | Aide |
| `/quit` | Quitter |

**Questions causales (sans préfixe) :**
```
What causes X?           explain: X
Effects of X?            effects: X
Chain from A to B?       chain: A B
Without X?               counterfactual: X
```

### 2. Indexer un corpus en batch

```bash
gcn-index --corpus rapports/ --checkpoint model.npz --output graph.json
```

Construit le graphe causal depuis un répertoire entier sans interaction.
Utile pour pré-indexer avant une session `gcn-discuss --graph graph.json`.

### 3. Entraîner sur votre corpus

```bash
gcn-train \
  --data-dir  corpus/train/ \
  --val-dir   corpus/val/ \
  --epochs    100 \
  --use-attention \
  --bidirectional \
  --embedding-dim 50 \
  --output    model.npz
```

### 4. API Python

```python
from gcn_python import GCNEngine
from gcn_python.verbalizer.instructions import CausalGraph
from gcn_python.verbalizer.query_report import QueryVerbalizer

# Charger le moteur
engine = GCNEngine.from_pretrained("model.npz")

# Analyser un corpus
cirs  = engine.analyze_batch(open("corpus.txt").readlines())
graph = CausalGraph.from_cirs(cirs)
graph.save("graph.json")

# Interroger
vb = QueryVerbalizer(graph)
print(vb.causes("data_exfiltration"))
print(vb.path("phishing", "ransomware"))
print(vb.contradictions())

# Reprendre une session
graph2 = CausalGraph.load("graph.json")
```

### 5. Substituer votre propre encodeur

```python
from gcn_python.layer2.interface import CausalEncoder

class MyEncoder(CausalEncoder):
    def forward_node(self, x): ...
    def forward_edge(self, x): ...
    def parameters(self): ...

from gcn_python.pipeline.cgnp import CGNPipeline
pipeline = CGNPipeline(encoder=MyEncoder(), graph=graph, vocabulary=vocab)
```

---

## Format de données (gcn-nl)

Vos données d'entraînement : fichiers JSON avec tokens UD et CIR gold.

```json
{
  "document": {
    "sentences": [{
      "id": "s001",
      "text": "The auth bypass enables data exfiltration.",
      "tokens": [
        {"id": 1, "form": "The",   "lemma": "the",    "pos": "DET",  "dep_rel": "det",   "dep_head": 3, "morph": {}},
        {"id": 2, "form": "auth",  "lemma": "auth",   "pos": "NOUN", "dep_rel": "compound","dep_head": 3,"morph": {}},
        {"id": 3, "form": "bypass","lemma": "bypass", "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 4, "morph": {}}
      ],
      "cir": {
        "nodes": [
          {"id": "n001", "type": "entite",   "label": "auth_bypass",        "token_span": [1, 3]},
          {"id": "n002", "type": "processus","label": "data_exfiltration",   "token_span": [5, 7]}
        ],
        "edges": [{
          "source": "n001", "target": "n002", "relation": "enable",
          "attributes": {"confidence": 1.0, "explicit": true, "negated": false}
        }]
      }
    }]
  }
}
```

**7 types de nœuds :** `etat` · `action` · `transition` · `processus` · `condition` · `entite` · `etat_systemique`

**11 relations :** `cause` · `enable` · `prevent` · `condition` · `concession` · `sequence` · `motivation` · `filter` · `opposition` · `data_dependency` · `control_dependency`

---

## Architecture

```
Texte brut (fr/en/code)
     │
     ▼  [gcn-frontend-fr / gcn-frontend-en / gcn-frontend-code — Rust]
UD tokens (pos, dep_rel, morph, lemma)
     │
     ▼  Layer 1 — FeatureVocabulary  (gcn-python)
79-dim vector par clause
     │
     ▼  Layer 2 — MLPEncoder  (remplaçable)
node_logits (N×7) + edge_logits (E×11)
     │
     ▼  Layer 3 — RGCNLayer / RGCNLayerGAT  (remplaçable)
message passing — enrichissement des représentations
     │
     ▼  CGNPipeline.forward() → CausalIR
```

---

## CLI

| Commande | Description |
|----------|-------------|
| `gcn-discuss` | Session interactive — `/analyze`, Q&A causale |
| `gcn-index` | Indexation batch d'un corpus → graphe JSON |
| `gcn-train` | Entraîner sur un corpus annoté |
| `gcn-eval` | Évaluer un checkpoint |
| `gcn-bootstrap` | Générer des données d'entraînement depuis texte brut |
| `gcn-verbalize` | CIR → texte (TrainableDecoder) |

---

## Métriques d'évaluation

```python
from gcn_python.evaluation.metrics import (
    node_macro_f1,      # F1 macro sur les 7 types de nœuds
    edge_macro_f1,      # F1 macro sur les 11 relations
    graph_exact_match,  # fraction de phrases avec graphe complet correct
    confusion_matrix,
    per_class_report,
)
```

---

## Stack complète

| Package | Rôle | Langage |
|---------|------|---------|
| `gcn-python` | Couches ML, entraînement, évaluation, gcn-discuss | Python |
| `gcn-ir` | Types fondamentaux CausalIR | Rust |
| `gcn-knowledge` | Taxonomies, lexique causal | Rust |
| `gcn-frontend-fr` | Parser causal français | Rust |
| `gcn-frontend-en` | Parser causal anglais | Rust |
| `gcn-frontend-code` | Parser code (Python/Rust/JS) | Rust |
| `gcn-middleend` | Graphe causal, cycles, validation | Rust |
| `gcn-backend` | Pearl niveaux 1-2-3, GCN-QL | Rust |
| `gcn-verbalizer` | CausalIR → texte | Rust/Python |

---

## Licence

MIT — © Michel Tendeng
