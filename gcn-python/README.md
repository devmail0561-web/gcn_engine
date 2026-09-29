# gcn-python — Moteur de raisonnement causal GCN-Core

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://img.shields.io/badge/tests-685-passing)](tests/)

Implémentation de référence NumPy du moteur causal GCN-Core. Extrait la structure causale de texte et répond à des requêtes Pearl sur le graphe résultant.

---

## Architecture (4 couches)

```
Couche 0 — UDRepresentation
  Entrée : tokens UD {id, form, lemma, pos, dep_rel, dep_head, morph}
  Sortie : UDRepresentation (root_pos, root_morph, token_span, flags structurels)

Couche 1 — Vectorisation
  vectorize_clause → float[d_clause=106]
    UPOS(18) + dep_rel(38) + subject_pos(5) + Tense(5) + Aspect(4) + Mood(5)
    + Voice(3) + PronType(7) + polarity(1) + flags(3)
    + 12 positionnels Éq.9 + 5 ternaires
  sentence_type.classify() → SentenceProfile (7 champs, couche 1 UD pure, 0 lemme)

Couche 2 — MLPEncoder (3 têtes supervisées)
  ├── Nœuds  : d_clause → mlp_hidden → 64 → 8   (D5 ETUDE)
  ├── Arêtes : d_edge   → 256 → 128 → 64 → 19  (v3.0)
  └── Intent : d_clause → mlp_hidden → 64 → 21  (Éq.6, n_intent_types=0 = désactivé)

Couche 3 — R-GCN message passing
  RGCNLayer ou RGCNLayerGAT (optionnel), bidirectionnel (38 types)

Couche 4 — Pearl (gcn-backend Rust)
  21+ requêtes : WHY/WHAT/CHAIN/chain_t/before/delay/DO/COUNTERFACTUAL
                 EXPLAIN/ANALOGY/SPOF/SUPER-SPOF/CENTRALITY/DENSITY/COVERAGE/RELIABILITY
                 DIFF/ZOOM_IN/ZOOM_OUT/AGGREGATE/NORM_DIFF
```

---

## Types

### NODE_TYPES D5 — 8 types (ETUDE §8)

```python
from gcn_python.constants import NODE_TYPES
# ["processus", "etat_local", "etat_global", "entite",
#  "condition", "concept", "evenement", "contrainte"]
```

Aliases lecture v2 (migration transparente) : `etat→etat_local`, `action→processus`,
`transition→processus`, `etat_systemique→etat_global`.

### RELATION_TYPES — 19 types (ETUDE §9)

```python
from gcn_python.constants import RELATION_TYPES
# 11 directes : cause, enable, prevent, condition, concession, sequence,
#               motivation, filter, opposition, data_dependency, control_dependency
# 8 ternaires v3.0 : analogy, counterfactual,
#                    conditional_cause, mediated_cause, joint_cause,
#                    conditional_prevent, mediated_prevent, joint_prevent
```

### INTENT_TYPES — 21 types (Éq.6)

```python
from gcn_python.constants import INTENT_TYPES
# Causal    : explain, effects, abduct, counterfactual
# Path      : chain, chain_t, before, delay
# Structure : spof, centrality, analogy
# Méta      : summarize, density, coverage, reliability, diff
# Navigation: zoom_in, zoom_out, aggregate
# Génération: verbalize
# Extraction: none
```

---

## Format d'annotation v4 (ETUDE §11)

```json
{
  "id": "s0001",
  "text": "Si les traitements échouent, le médicament est prescrit.",
  "causal_pattern": "condition",
  "intent": "",
  "tokens": [
    {"id": 0, "form": "Si", "lemma": "si", "pos": "SCONJ",
     "dep_rel": "mark", "dep_head": 2, "morph": {}},
    {"id": 2, "form": "échouent", "lemma": "échouer", "pos": "VERB",
     "dep_rel": "advcl", "dep_head": 5, "morph": {"Mood": "Ind", "Tense": "Pres"}},
    {"id": 5, "form": "prescrit", "lemma": "prescrire", "pos": "VERB",
     "dep_rel": "root", "dep_head": -1, "morph": {"Voice": "Pass"}}
  ],
  "cir": {
    "sentence_profile": {"sentence_type": "declarative", "subordination": "condition"},
    "salience": {"focus_node": "n001", "condition_prominence": "foreground"},
    "nodes": [
      {"id": "n001", "type": "processus", "label": "échec traitements",
       "token_span": [0, 2], "pos": "VERB", "morph": {"Mood": "Ind"}},
      {"id": "n002", "type": "evenement", "label": "prescription médicament",
       "token_span": [3, 5], "pos": "VERB", "morph": {"Voice": "Pass"}}
    ],
    "edges": [
      {
        "sources": ["n001"],
        "target": "n002",
        "relation": "conditional_cause",
        "third": {"role": "condition", "node": "n001"},
        "confidence": 0.85,
        "polarity": "positive",
        "voice": "passive",
        "modality": "indicative",
        "has_restriction": false,
        "condition_prominence": "foreground"
      }
    ]
  }
}
```

**Règles critiques :**
- `sources` : liste obligatoire — jamais `source` singulier
- `third` : `{"role": "condition"|"mediator", "node": "id"}` pour les ternaires, `null` sinon
- `joint_cause` : deux arêtes séparées avec même `joint_group_id`, `third: null`
- `intent` : un des 21 INTENT_TYPES pour les questions, `""` pour les déclaratifs

---

## Installation

```bash
# Depuis la racine du dépôt
pip install -e gcn-python/

# Avec support PyTorch (RGCNLayerGAT)
pip install -e "gcn-python/[torch]"
```

---

## Entraînement

```bash
cd gcn-python
gcn-train \
  --data-dir gcn-datasets/real/train/ \
  --val-dir gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional \
  --output model.npz

# Mode coarse D4 (Mood/Tense masqués, labels coarse)
gcn-train --data-dir ... --coarse-phase --coarse-n-min 400 ...

# Tête intention Éq.6 (si données intent annotées)
gcn-train --data-dir ... --n-intent-types 21 ...
```

Options notables :
- `--all-pairs` (True par défaut) — toutes les paires de clauses supervisées
- `--no-positional` / `--no-ternary` — ablation features (gate C.7)
- `--n-intent-types 21` — active la tête NLU routing
- `--coarse-phase` — entraînement au niveau coarse D4

---

## Inférence

### GCNEngine — point d'entrée haut niveau

```python
from gcn_python import GCNEngine

# Charger un checkpoint entraîné
engine = GCNEngine.from_pretrained("checkpoints/model_v4.npz")

# Analyser une phrase
cir = engine.analyze("Si les traitements échouent, le médecin prescrit un autre médicament.")
print(cir["edges"][0][2]["relation"])   # "condition"
print(cir["nodes"][0]["type"])          # "processus"

# Analyser plusieurs phrases
results = engine.analyze_batch([
    "La pluie cause des inondations.",
    "Si la demande baisse, les ventes chutent.",
])
```

### Inférence via CLI

```bash
# Session interactive
gcn-discuss --checkpoint model.npz

# Indexation batch
gcn-index --corpus rapports/ --checkpoint model.npz --output graph.json

# Évaluation
gcn-eval --data-dir gcn-datasets/real/val/ --model-path model.npz
```

---

## CGNPipeline — paramètres clés

```python
from gcn_python.pipeline.cgnp import CGNPipeline

pipeline = CGNPipeline(
    encoder=encoder,          # MLPEncoder
    graph=graph,              # RGCNLayer
    vocabulary=vocab,         # FeatureVocabulary (d_clause=106)
    all_pairs=True,           # Obligatoire — toutes paires supervisées
    bidirectional=True,       # 38 types de relations (19 forward + 19 inv)
    n_intent_types=0,         # 0 = tête intent désactivée
    no_mood=False,            # True = mode coarse D4
    no_tense=False,           # True = mode coarse D4
    theta_ambiguity=0.65,     # Seuil Éq.11 (THETA_AMBIGUITY_DEFAULT)
    edge_threshold=0.0,       # Seuil confiance arête émise
    temperature=1.0,          # Température softmax (T5-min)
)
```

---

## NLU routing (Éq.6)

```python
from gcn_python.pipeline.nlu_routing import nlu_route

# Priorité 1 : tête apprise (si n_intent_types > 0 et pipeline forward exécuté)
# Priorité 2 : heuristique surface UD (SentenceType + 2 concepts)
cmd = nlu_route("Pourquoi les ventes baissent ?", pipeline=pipeline)
# → "explain: ventes" ou "chain: pourquoi ventes"
```

---

## Modules clés — exemples

### FeatureVocabulary + vectorize_clause

```python
from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause
from gcn_python.data.loader import reps_from_sentence

vocab = FeatureVocabulary()
print(vocab.d_clause)                           # 106
print(vocab.d_clause_effective(d_emb=128))      # 234

# Vectoriser une clause depuis un SentenceRecord
reps, idxs, conn_reps = reps_from_sentence(sample)
vec = vectorize_clause(reps[0], vocab)          # ndarray (106,)
```

### SentenceProfile — classify()

```python
from gcn_python.layer1.sentence_type import classify, LangMarkers

tokens = [
    {"id": 0, "form": "Si",      "lemma": "si",     "pos": "SCONJ",
     "dep_rel": "mark", "dep_head": 2, "morph": {}},
    {"id": 2, "form": "échoue",  "lemma": "échouer", "pos": "VERB",
     "dep_rel": "advcl", "dep_head": 4, "morph": {"Mood": "Ind"}},
]
profile = classify(tokens)
print(profile.sentence_type)      # SentenceType.DECLARATIVE
print(profile.subordination)      # SubordinationType.CONDITION
print(profile.is_complex)         # True
```

### MLPEncoder + RGCNLayer

```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.constants import RELATION_TYPES, NODE_TYPES

vocab  = FeatureVocabulary()
d_eff  = vocab.d_clause_effective(d_emb=128)        # 234
d_edge = vocab.d_edge_closed_loop(d_eff, 8, 128)   # 1115

encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph   = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=len(RELATION_TYPES))  # 19
```

### GCNDataLoader

```python
from pathlib import Path
from gcn_python.data.loader import GCNDataLoader

loader = GCNDataLoader(Path("gcn-datasets/splits/train/"))
for sample in loader:
    print(sample.sentence.text)
    print(sample.gold_node_labels)   # ndarray (N,)
    print(sample.edge_map)           # {(i,j): relation_idx}
```

### Évaluation

```python
from gcn_python.evaluation.metrics import edge_macro_f1, node_f1_per_class
from gcn_python.constants import RELATION_TYPES_V2

# F1 macro sur les 11 classes v2 partagées (métrique K2)
f1 = edge_macro_f1(y_true, y_pred, class_subset=RELATION_TYPES_V2)

# F1 par classe avec support
per_class = node_f1_per_class(y_true, y_pred)
for name, (p, r, f, sup) in per_class.items():
    if sup > 0:
        print(f"{name}: F1={f:.3f} (n={sup})")
```

### Checkpoint

```python
from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(pipeline, Path("checkpoints/model_v4.npz"))
pipeline2 = load_checkpoint(Path("checkpoints/model_v4.npz"))
```

---

## Tests

```bash
cd gcn-python
PYTHONPATH=src python -m pytest tests/ -v
# 685 passed, 3 xfailed (checkpoints v2 incompatibles d_clause 79→106)
```

Les 3 xfailed sont attendus : les checkpoints v2 (`prod_v1.npz`, d_clause=79) sont incompatibles
avec v4.0 (d_clause=106). Migration via `scripts/init_v3_stub.py`.

---

## Structure du paquet

```
gcn_python/
├── constants.py          # NODE_TYPES D5, RELATION_TYPES 19, INTENT_TYPES 21,
│                         # COARSE_RELATION_GROUPS, NEGATION_PREVENT_MAP,
│                         # RELATION_TYPES_V2 (11 types K2), NODE_TYPE_ALIASES
├── layer0/interface.py   # Protocol TextParser, GCNBridgeParser
├── layer1/
│   ├── features.py       # vectorize_clause (d_clause=106)
│   ├── sentence_type.py  # classify() → SentenceProfile, LangMarkers
│   ├── representation.py # UDRepresentation
│   └── embedding.py      # WordEmbedding (D10)
├── layer2/
│   ├── reference.py      # MLPEncoder (3 têtes : nœuds+arêtes+intent)
│   └── interface.py      # Protocol CausalEncoder
├── layer3/
│   ├── reference.py      # RGCNLayer NumPy
│   ├── gat.py            # RGCNLayerGAT
│   └── pytorch_rgcn.py   # RGCNLayerPT (optionnel PyTorch)
├── pipeline/
│   ├── cgnp.py           # CGNPipeline (forward/loss/backward)
│   ├── ir_emitter.py     # emit() → CIR dict, apply_negation_algebra(),
│   │                     # detect_ternary(), orient_edge_d7(), confidence_d6()
│   ├── discourse.py      # Éq.10 — extract_discourse_relation()
│   ├── nlu_routing.py    # Éq.6 — nlu_route()
│   ├── label_builder.py  # build_label() → label + attributes
│   └── cli.py            # gcn-forward CLI
├── data/
│   ├── schema.py         # TokenRecord, ClauseRecord, EdgeRecord, SentenceRecord
│   ├── json_reader.py    # _parse_edge, _parse_clause_node, NODE_TYPE_ALIASES
│   ├── loader.py         # GCNDataLoader (all_pairs=True)
│   ├── edge_norm.py      # normalize_edge, normalize_node_id
│   └── graph_vecs.py     # Cache SHA256 vecteurs enrichis
├── training/
│   ├── train.py          # gcn-train CLI (46 options)
│   ├── checkpoint.py     # save/load checkpoint (encoder_*, intent_layer_*)
│   └── bootstrap.py      # gcn-bootstrap CLI
├── evaluation/
│   ├── metrics.py        # node_macro_f1, edge_macro_f1(class_subset=),
│   │                     # RELATION_TYPES_V2, _f1_per_class
│   ├── eval_runner.py    # gcn-eval CLI
│   └── calibration.py    # optimize_temperature, ece_score, apply_isotonic_params
├── frontend/
│   └── bridge.py         # GCNBridgeParser, NODE_TYPE_TO_POS/DEP (D5)
├── verbalizer/           # Décodeur CIR→texte, InstructionHandler, QueryVerbalizer
├── discuss.py            # gcn-discuss CLI (session interactive + nlu_route)
└── index.py              # gcn-index CLI (indexation batch + Éq.10 discours)
```
