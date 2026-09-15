# gcn-python — Couches ML du moteur GCN-Core

[![PyPI version](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![Python](https://img.shields.io/pypi/pyversions/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Moteur de raisonnement causal — extraire, modéliser et inférer la causalité dans le texte naturel et le code source.**

---

## Vision

GCN-Core est un **moteur**, pas un modèle pré-entraîné. Comme le Transformer est une architecture que l'on entraîne sur ses propres données, GCN-Core est une architecture de raisonnement causal que chaque utilisateur entraîne sur son corpus.

### Quel problème résout-il ?

Extraire la structure causale d'un texte ou d'un programme — *qui fait quoi, pourquoi, avec quelle conséquence* — est un problème mal résolu par les LLM génériques : ils produisent du texte vraisemblable, pas une structure vérifiable. GCN-Core produit une **Représentation Intermédiaire Causale** (`CausalIR`) : un graphe orienté typé, sérialisable en JSON, interrogeable par GCN-QL, et raisonnable au sens de Pearl (niveaux 1-2-3).

### Pour qui ?

- **Chercheurs en NLP causal** : annotation et évaluation de relations de cause-effet
- **Ingénieurs** : extraction de dépendances causales depuis de la documentation ou du code
- **Data scientists** : construction de systèmes d'explication (XAI) basés sur des graphes causaux vérifiables
- **Quiconque** veut comprendre *pourquoi* quelque chose se produit, pas seulement *quoi*

### Quand l'utiliser ?

Utilise `gcn-python` quand tu as besoin de :
1. Entraîner les couches ML (MLP + R-GCN) sur ton propre corpus annoté
2. Intégrer le pipeline de vectorisation et d'inférence dans ton code Python
3. Évaluer les performances de classification causale
4. Construire un décodeur texte depuis un graphe causal

---

## Architecture

```
Texte naturel (fr/en) ou Code source (Python/Rust/JS)
                        │
              [gcn-cli — Rust]
                        │
               CausalIR (JSON)
                        │
         ┌──────────────┴──────────────┐
         │                             │
  Layer 1 — Features            Tokens annotés
  FeatureVocabulary                   │
  vectorize_clause()           reps_from_sentence()
         │                             │
         └──────────────┬──────────────┘
                        │
              Layer 2 — Encodeur
              MLPEncoder
              NodeType classifier  (7 classes)
              RelationType classifier  (11 classes)
                        │
              Layer 3 — Graphe causal
              RGCNLayer / RGCNLayerPT
              Message passing R-GCN
                        │
              CGNPipeline.forward()
                        │
               CausalIR enrichi
                        │
              [Optionnel] TrainableDecoder
                        │
                  Surface texte
```

Les **7 types de nœuds** : `etat`, `action`, `transition`, `processus`, `condition`, `entite`, `etat_systemique`

Les **11 types de relations** : `cause`, `enable`, `prevent`, `condition`, `concession`, `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency`

---

## Installation

```bash
# Package Python seul
pip install gcn-python

# Avec support GPU/MPS (R-GCN PyTorch)
pip install gcn-python torch

# Avec le moteur Rust (CLI gcn-analyze, gcn-query, gcn-export)
git clone https://github.com/devmail0561-web/gcn_engine.git
cd gcn_engine && make install
```

**Prérequis :** Python ≥ 3.10, NumPy ≥ 1.24

---

## Démarrage rapide

### Inférence depuis un corpus annoté

```python
from pathlib import Path
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.training.checkpoint import load_checkpoint

# Construire le pipeline
vocab = FeatureVocabulary()
encoder = MLPEncoder(vocab.d_clause, vocab.d_edge)
graph = RGCNLayer(vocab.d_clause, vocab.d_clause)   # d_out == d_clause (contrainte)
pipeline = CGNPipeline(encoder, graph, lang="fr", vocabulary=vocab)

# Charger un checkpoint entraîné
load_checkpoint(pipeline, Path("model.npz"))

# Inférence
loader = GCNDataLoader(Path("mon_corpus/"), lang="fr")
for sample in loader:
    reps, valid_idxs, connector_reps = reps_from_sentence(sample.sentence)
    cir = pipeline.forward(
        reps,
        text=sample.sentence.text,
        connector_reps=connector_reps,
    )
    # cir est un dict JSON-serializable conforme au schéma CausalIR
    print(cir["nodes"])   # [{id, node_type, label, ...}, ...]
    print(cir["edges"])   # [[src_idx, dst_idx, {relation, confidence, ...}], ...]
```

### Entraînement

```bash
gcn-train \
  --data-dir mon_corpus/ \
  --epochs 100 \
  --lr 0.001 \
  --output model.npz \
  --log-csv courbe.csv
```

```python
# Accès programmatique aux métriques après entraînement
import json
with open("courbe.json") as f:
    curve = json.load(f)
# [{"epoch": 1, "loss": 2.3, "node_accuracy": 0.41, "edge_macro_f1": 0.28}, ...]
```

### Évaluation

```bash
gcn-eval --data-dir mon_corpus/ --model-path model.npz
```

Sortie :
```json
{
  "n_samples": 120,
  "n_skipped": 2,
  "node_accuracy": 0.87,
  "node_macro_f1": 0.83,
  "edge_accuracy": 0.79,
  "edge_macro_f1": 0.74
}
```

### Bootstrap — générer un corpus depuis du texte brut

```bash
# 1. Préparer phrases_fr.txt (une phrase par ligne)
# 2. Lancer gcn-bootstrap (requiert gcn-cli Rust installé)
gcn-bootstrap \
  --input phrases_fr.txt \
  --out-dir corpus/ \
  --taxonomy-dir gcn-references/taxonomies/ \
  --lang fr

# 3. Réviser manuellement les JSON générés
# 4. Entraîner
gcn-train --data-dir corpus/ --epochs 50 --output model.npz
```

### R-GCN PyTorch (GPU/MPS)

```python
from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT

# Détection automatique cuda / mps / cpu
graph_pt = RGCNLayerPT(vocab.d_clause, vocab.d_clause)
pipeline = CGNPipeline(encoder, graph_pt, lang="fr", vocabulary=vocab)

# Pour un entraînement natif PyTorch (avec autograd)
import torch
H = torch.from_numpy(clause_vecs).float().to(graph_pt._device)
enriched = graph_pt.forward_torch(H, edge_index, edge_types)
# utiliser optimizer.step() — NE PAS appeler pipeline.backward() avec PyTorch
```

---

## Format de données

Les données d'entraînement sont des fichiers JSON conformes au schéma `gcn-nl` :

```json
{
  "document": {
    "id": "doc-001",
    "lang": "fr",
    "sentences": [
      {
        "id": "s001",
        "text": "Si les ventes baissent, on réduit les coûts.",
        "tokens": [
          {
            "id": 1, "form": "Si", "lemma": "si", "pos": "SCONJ",
            "dep_rel": "mark", "dep_head": 4,
            "morph": {},
            "gcn": {"causal_type": "conjonction", "causal_class": "condition"}
          }
        ],
        "cir": {
          "nodes": [
            {
              "id": "n001", "type": "processus",
              "label": "décroissance(ventes)",
              "token_span": [3, 4], "origin": "explicit",
              "scope": "universal", "temporal_index": 0,
              "attributes": {"entity": "ventes"}
            },
            {
              "id": "n002", "type": "action",
              "label": "réduire(coûts)",
              "token_span": [6, 8], "origin": "explicit",
              "scope": "universal", "temporal_index": 1,
              "attributes": {}
            }
          ],
          "edges": [
            {
              "source": "n001", "target": "n002",
              "relation": "condition",
              "attributes": {
                "confidence": 1.0, "explicit": true,
                "negated": false, "marker_token": 1
              }
            }
          ]
        }
      }
    ]
  }
}
```

Trois schémas disponibles dans `gcn-datasets/schemas/` :
- `gcn-nl.schema.yaml` — texte naturel (fr/en)
- `gcn-pl.schema.yaml` — code source (Python/Rust/JS)
- `gcn-verbalize.schema.yaml` — paires CausalIR ↔ surface texte (entraînement décodeur)

---

## Référence API

### `gcn_python.constants`

Constantes synchronisées avec les types Rust de `gcn-ir`.

```python
from gcn_python.constants import NODE_TYPES, RELATION_TYPES

NODE_TYPES      # ['etat', 'action', 'transition', 'processus',
                #  'condition', 'entite', 'etat_systemique']  — 7 valeurs

RELATION_TYPES  # ['cause', 'enable', 'prevent', 'condition', 'concession',
                #  'sequence', 'motivation', 'filter', 'opposition',
                #  'data_dependency', 'control_dependency']  — 11 valeurs

# Également disponibles :
# SCOPE_VALUES, NODE_ORIGIN_VALUES, AGENT_TYPE_VALUES
# UPOS_TAGS (19), UD_DEP_RELS (38)
# UD_TENSE_VALUES (5), UD_ASPECT_VALUES (4), UD_MOOD_VALUES (5)
# SUBJECT_POS_CATS (5)
```

---

### `gcn_python.data.schema` — Structures de données

```python
from gcn_python.data.schema import TokenRecord, ClauseRecord, EdgeRecord, SentenceRecord
```

#### `TokenRecord`
```python
@dataclass
class TokenRecord:
    id: int               # position 1-based dans la phrase
    form: str             # forme de surface
    lemma: str
    pos: str              # tag UPOS (ex. "VERB", "SCONJ")
    dep_rel: str          # relation Universal Dependencies (ex. "nsubj", "mark")
    dep_head: int         # 0 = racine
    morph: dict[str, str] # {"Tense": "Past", "Mood": "Ind", ...}
    gcn_causal_type: str | None   # "verbe", "conjonction", ...
    gcn_causal_class: str | None  # "processus", "condition", ...
```

#### `ClauseRecord`
```python
@dataclass
class ClauseRecord:
    node_id: str                # "n001"
    node_type: str              # valeur de NODE_TYPES
    label: str                  # étiquette humaine du nœud
    token_span: tuple[int, int] # indices de tokens (1-based, inclus)
    scope: str                  # valeur de SCOPE_VALUES
    temporal_index: int         # ordre temporel dans la phrase
    origin: str                 # "explicit" | "inferred" | "hypothetical"
    attributes: dict            # entité, qualité, agent, patient, ...
    modifiers: list[dict]       # modificateurs aspectuels, modaux, ...
```

#### `EdgeRecord`
```python
@dataclass
class EdgeRecord:
    source: str       # "n001"
    target: str       # "n002"
    relation: str     # valeur de RELATION_TYPES
    confidence: float # [0.0, 1.0]
    explicit: bool    # marqueur lexical présent
    negated: bool
    marker_token: int | None  # id du token marqueur
```

#### `SentenceRecord`
```python
@dataclass
class SentenceRecord:
    id: str
    text: str
    lang: str
    tokens: list[TokenRecord]
    clauses: list[ClauseRecord]
    edges: list[EdgeRecord]
```

---

### `gcn_python.data.json_reader`

```python
from gcn_python.data.json_reader import load_sentences, load_all_sentences
from pathlib import Path

# Lire un fichier JSON unique
sentences = load_sentences(Path("corpus/doc001.json"), lang="fr")
# -> list[SentenceRecord]

# Lire un répertoire entier
sentences = load_all_sentences(Path("corpus/"), lang="fr")
# Parcourt *.json (trié), concatène tous les SentenceRecord
```

Supporte deux formats JSON :
- **Format dataset** : `document.sentences` avec tokens + CIR
- **Format exemples** : `examples[].expected_cir` (CIR sans tokens)

---

### `gcn_python.data.loader`

```python
from gcn_python.data.loader import GCNDataLoader, TrainingSample, reps_from_sentence
```

#### `TrainingSample`
```python
@dataclass
class TrainingSample:
    sentence: SentenceRecord
    gold_node_labels: np.ndarray   # shape (N,) int64 — indices dans NODE_TYPES
    edge_map: dict                 # {(src_clause_idx, tgt_clause_idx): rel_idx}
    # edge_map : arêtes consécutives forward uniquement (gap=1, src < tgt)
```

#### `GCNDataLoader`
```python
loader = GCNDataLoader(
    data_dir=Path("corpus/"),
    lang="fr",          # code langue
    repeat=False,       # True = itérateur infini
)

len(loader)             # nombre de SentenceRecord chargés
for sample in loader:   # yield TrainingSample
    ...
```

#### `reps_from_sentence`
```python
reps, valid_idxs, connector_reps = reps_from_sentence(sentence_record)
# reps           : list[UDRepresentation] — une par clause valide
# valid_idxs     : list[int] — indices originaux dans sentence_record.clauses
# connector_reps : list[UDRepresentation | None] — len = len(reps) - 1
#                  token connecteur (SCONJ/CCONJ/ADP) entre chaque paire de clauses
# Retourne ([], [], []) si pas de tokens ou pas de clauses
```

---

### `gcn_python.data.verbalize_loader`

```python
from gcn_python.data.verbalize_loader import VerbalizerDataLoader, VerbalizeSample
```

#### `VerbalizeSample`
```python
@dataclass
class VerbalizeSample:
    ir_json: str                      # CausalIR JSON (string)
    node_type_embeddings: np.ndarray  # shape (N, 7) — one-hot NODE_TYPES
    gold_tokens: np.ndarray           # shape (T,) int64 — indices SurfaceVocabulary
    source_text: str
```

#### `VerbalizerDataLoader`
```python
loader = VerbalizerDataLoader(
    data_dir=Path("corpus/"),
    vocab=None,   # None = construit le vocab depuis les surfaces gold/silver
)

loader.vocab                # SurfaceVocabulary construit
loader.source_text_map()    # dict[str, list[np.ndarray]] pour joint training
len(loader)
for sample in loader:       # yield VerbalizeSample
    ...
```

---

### `gcn_python.layer1.representation`

```python
from gcn_python.layer1.representation import UDRepresentation
```

#### `UDRepresentation`
```python
@dataclass
class UDRepresentation:
    tokens: list[dict]           # [{lemma, pos, dep_rel, morph}, ...]
    root_lemma: str
    root_pos: str                # UPOS du token racine
    root_dep_rel: str
    root_morph: dict[str, str]   # {"Tense": ..., "Aspect": ..., "Mood": ...}
    subject_pos: str | None      # UPOS du sujet (nsubj), ou None
    has_object: bool             # obj/iobj/nobj dans le span
    has_advcl: bool              # advcl dans le span
    has_temporal_obl: bool       # obl / obl:tmod dans le span
    token_span: tuple[int, int]
    lang: str

# Propriétés calculées
rep.tense    # root_morph.get("Tense", "_absent")
rep.aspect   # root_morph.get("Aspect", "_absent")
rep.mood     # root_morph.get("Mood", "_absent")
rep.is_negative  # root_morph.get("Polarity", "") == "Neg"
```

---

### `gcn_python.layer1.features`

```python
from gcn_python.layer1.features import (
    FeatureVocabulary,
    vectorize_clause,
    vectorize_connector,
    vectorize_edge,
)
```

#### `FeatureVocabulary`
```python
vocab = FeatureVocabulary()  # utilise les constantes par défaut

vocab.d_clause  # 80  (19 UPOS + 38 DEP_REL + 5 subj_pos + 5 tense
                #      + 4 aspect + 5 mood + 1 polarity + 3 flags)
vocab.d_conn    # 21  (19 UPOS + 2 flags directionnels)
vocab.d_edge    # 181 (2 * d_clause + d_conn)

# Sérialisation pour checkpoint
json_str = vocab.to_json()
vocab2 = FeatureVocabulary.from_json(json_str)
```

#### `vectorize_clause`
```python
vec = vectorize_clause(rep, vocab)
# rep  : UDRepresentation
# vocab: FeatureVocabulary
# -> np.ndarray shape (d_clause,) float32
# Concaténation : one_hot(root_pos) + one_hot(root_dep_rel) + one_hot(subject_pos)
#   + one_hot(tense) + one_hot(aspect) + one_hot(mood)
#   + [is_negative] + [has_object, has_advcl, has_temporal_obl]
```

#### `vectorize_connector`
```python
vec = vectorize_connector(marker_rep, src_idx, dst_idx, n_clauses, vocab)
# marker_rep : UDRepresentation du token connecteur, ou None
# -> np.ndarray shape (d_conn,) float32
```

#### `vectorize_edge`
```python
vec = vectorize_edge(src_rep, dst_rep, connector_rep, src_idx, dst_idx, n_clauses, vocab)
# -> np.ndarray shape (d_edge,) float32
# = concat(vectorize_clause(src), vectorize_clause(dst), vectorize_connector(...))
```

---

### `gcn_python.layer2.interface`

```python
from gcn_python.layer2.interface import CausalEncoder
```

Protocol `@runtime_checkable`. Implémenter pour substituer le MLP de référence.

```python
class MonEncoder:
    def forward_node(self, x: np.ndarray) -> np.ndarray:
        # x: (d_clause,) -> (7,) logits NODE_TYPES
        ...

    def forward_edge(self, x: np.ndarray) -> np.ndarray:
        # x: (d_edge,) -> (11,) logits RELATION_TYPES
        ...

    def parameters(self) -> list[np.ndarray]: ...
    def update_node(self, grads, lr: float) -> None: ...
    def update_edge(self, grads, lr: float) -> None: ...
    def update(self, grads, lr: float) -> None: ...

assert isinstance(MonEncoder(), CausalEncoder)  # True
```

---

### `gcn_python.layer2.reference`

```python
from gcn_python.layer2.reference import MLPEncoder
```

#### `MLPEncoder`
Implémentation NumPy de référence de `CausalEncoder`.

**Architecture :**
- Node MLP : `d_clause → 128 → 64 → 7` (initialisation He, ReLU)
- Edge MLP : `d_edge → 256 → 128 → 11` (initialisation He, ReLU)

```python
encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge, seed=42)

# Inférence
node_logits = encoder.forward_node(clause_vec)  # (7,)
edge_logits = encoder.forward_edge(edge_vec)    # (11,)

# Backward (utilisé par CGNPipeline.backward())
grads_node = encoder.backward_node(d_logits)    # list[(dW, db)] — 3 couches
grads_node, d_input = encoder.backward_node_dx(d_logits)  # + gradient en entrée

# Snapshots (pour backward multi-nœuds sans re-forward)
snap = encoder.snapshot_node_cache()
encoder.restore_node_cache(snap)

# SGD manuel
encoder.update_node(grads_node, lr=0.001)
encoder.update_edge(grads_edge, lr=0.001)

# Tous les paramètres (pour checkpoint)
params = encoder.parameters()  # [W1,b1,W2,b2,W3,b3] nœud + idem arête = 12 arrays
```

---

### `gcn_python.layer3.interface`

```python
from gcn_python.layer3.interface import CausalGraph
```

Protocol `@runtime_checkable`. Formule R-GCN :

```
h_i^(l+1) = σ( Σ_r  Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j  +  W_0 h_i )
```

```python
class MonGraph:
    d_out: int  # dimension de sortie (doit == vocab.d_clause)

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_in)
        edge_index: np.ndarray,     # (2, E) — [sources, targets]
        edge_types: np.ndarray,     # (E,) int — indices RELATION_TYPES
    ) -> np.ndarray: ...            # (N, D_out)

    def parameters(self) -> list[np.ndarray]: ...
    def update(self, grads, lr: float) -> None: ...
```

---

### `gcn_python.layer3.reference`

```python
from gcn_python.layer3.reference import RGCNLayer
```

#### `RGCNLayer`
Implémentation NumPy de référence de `CausalGraph`. Supporte les cycles.

```python
graph = RGCNLayer(
    d_in=vocab.d_clause,
    d_out=vocab.d_clause,   # contrainte : d_out == d_clause
    n_relations=11,          # défaut : len(RELATION_TYPES)
    seed=42,
)

# Forward
enriched = graph.message_pass(node_features, edge_index, edge_types)
# node_features : (N, d_in) — vecteurs de clauses
# edge_index    : (2, E)
# edge_types    : (E,)
# -> (N, d_out)

# Backward
d_input, [dW_r, dW_0] = graph.backward_message_pass(d_output)

# SGD
graph.update([dW_r, dW_0], lr=0.001)

# Paramètres : [W_r (n_rel, d_out, d_in), W_0 (d_out, d_in)]
graph.parameters()
```

---

### `gcn_python.layer3.pytorch_rgcn`

```python
from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
```

#### `RGCNLayerPT`
Implémentation PyTorch avec support GPU/MPS.

```python
graph_pt = RGCNLayerPT(
    d_in=vocab.d_clause,
    d_out=vocab.d_clause,
    n_relations=11,
    device=None,   # auto-détecte cuda > mps > cpu
    seed=42,
)

# Inférence (retourne numpy, sans grad)
enriched = graph_pt.message_pass(node_features, edge_index, edge_types)

# Entraînement natif PyTorch (garde le graphe de calcul)
import torch
H = torch.from_numpy(node_features).float().to(graph_pt._device)
enriched_t = graph_pt.forward_torch(H, edge_index, edge_types)  # Tensor (N, d_out)
# Utiliser loss.backward() + optimizer.step()

# Paramètres PyTorch pour optimizer
params = graph_pt.torch_parameters()  # list[nn.Parameter]
optimizer = torch.optim.Adam(params, lr=0.001)

# Déplacer sur un autre device
graph_pt.to_device("cuda")
```

---

### `gcn_python.pipeline.cgnp`

```python
from gcn_python.pipeline.cgnp import CGNPipeline
```

#### `CGNPipeline`
Compose les couches 1-3 en un pipeline complet.

```python
pipeline = CGNPipeline(
    encoder=encoder,    # CausalEncoder
    graph=graph,        # CausalGraph
    lang="fr",
    vocabulary=vocab,
    decoder=None,       # TrainableDecoder optionnel
)
# Précondition : graph.d_out == vocab.d_clause (ValueError sinon)
```

#### `forward`
```python
cir = pipeline.forward(
    reps,                          # list[UDRepresentation]
    text="",                       # texte source
    clause_positions=None,         # list[int] — indices des clauses dans la phrase complète
    n_total_clauses=None,          # nombre total de clauses (pour feature distance)
    connector_reps=None,           # list[UDRepresentation | None], len == len(reps)-1
)
# -> dict CausalIR JSON-sérialisable
# Remplit tous les attributs _cached_*
```

#### `loss`
```python
total_loss, d_node, d_edge = pipeline.loss(
    node_logits=pipeline._cached_node_logits,  # (N, 7)
    edge_logits=pipeline._cached_edge_logits,  # (E, 11) ou None
    gold_node=gold_node_labels,                # (N,) int
    gold_edge=gold_edge_labels,                # (E,) int ou None
    edge_loss_weight=1.0,
    gold_surface=None,                         # (T,) int pour décodeur
)
# total_loss : float
# d_node     : (N, 7)  gradient logits nœuds
# d_edge     : (E, 11) gradient logits arêtes
```

#### `backward`
```python
pipeline.backward(
    d_node_logits=d_node,   # (N, 7)
    d_edge_logits=d_edge,   # (E, 11)
    lr=0.001,
)
# Étapes : backward MLP nœuds (par snapshot) → backward MLP arêtes
#        → backward décodeur (si présent) → backward R-GCN → update poids
# No-op si l'encodeur n'implémente pas backward_node_dx (ex. implémentation custom)
```

#### `filter_edge_cache`
```python
pipeline.filter_edge_cache(valid_edge_idxs)
# Filtre les caches arêtes APRÈS forward() pour aligner edge_logits avec gold_edge.
# valid_edge_idxs : np.ndarray d'indices (produit par GCNDataLoader)
```

---

### `gcn_python.pipeline.ir_emitter`

```python
from gcn_python.pipeline.ir_emitter import emit

cir = emit(
    text="Si les ventes baissent, on réduit les coûts.",
    lang="fr",
    node_types=["processus", "action"],
    node_labels=["décroissance(ventes)", "réduire(coûts)"],
    token_spans=[(3, 4), (6, 8)],
    scopes=["universal", "universal"],
    edge_triples=[
        # (src_idx, dst_idx, relation, confidence, negated, marker_token)
        (0, 1, "condition", 1.0, False, 1),
    ],
    node_origins=["explicit", "explicit"],  # optionnel
)
# -> dict CausalIR JSON-sérialisable
```

---

### `gcn_python.pipeline.label_builder`

```python
from gcn_python.pipeline.label_builder import build_label

label = build_label(
    rep=ud_rep,                     # UDRepresentation
    node_type="action",             # type prédit
    taxonomies_dir=None,            # Path vers taxonomies (nominalizations.json)
)
# -> str  ex. "réduire(coûts)", "décroissance(ventes)", "hidden_cause(?)"
```

---

### `gcn_python.taxonomy.loader`

```python
from gcn_python.taxonomy.loader import TaxonomyIndex

tax = TaxonomyIndex.load(
    taxonomies_dir=Path("gcn-references/taxonomies/"),
    lang_code="fr",
)

tax.membership("provoquer")
# -> {"verbes.cause": True, "verbes.etat": False, ...}

tax.keys()
# -> ["verbes.cause", "verbes.condition", "verbes.enable", ...]

len(tax)  # nombre de classes chargées
```

---

### `gcn_python.training.checkpoint`

```python
from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
from pathlib import Path

# Sauvegarder
save_checkpoint(pipeline, Path("model.npz"))
# Contenu .npz : encoder_0..N, graph_0..1, _vocab_json
#                + decoder_0..N et _decoder_meta_json si décodeur présent

# Restaurer (atomique — lève ValueError si shapes incompatibles)
load_checkpoint(pipeline, Path("model.npz"))
# Restaure aussi FeatureVocabulary et TrainableDecoder depuis le checkpoint
```

---

### `gcn_python.training.train`

```
gcn-train [OPTIONS]

Options :
  --data-dir PATH        Répertoire des données d'entraînement  [requis]
  --lang TEXT            Code langue (défaut: fr)
  --epochs INT           Nombre d'époques (défaut: 50)
  --lr FLOAT             Taux d'apprentissage (défaut: 0.001)
  --output PATH          Fichier checkpoint .npz (défaut: model.npz)
  --log-csv PATH         Log CSV par époque (optionnel)
  --verbalize-dir PATH   Répertoire verbalize pour entraînement conjoint (optionnel)
```

Avec `--log-csv loss.csv`, le fichier `loss.json` est aussi généré avec `node_accuracy` et `edge_macro_f1` par époque.

---

### `gcn_python.training.bootstrap`

```
gcn-bootstrap [OPTIONS]

Options :
  --input PATH           Fichier .txt (une phrase par ligne)  [requis]
  --lang TEXT            Code langue (défaut: fr)
  --out-dir PATH         Répertoire de sortie JSON  [requis]
  --taxonomy-dir PATH    Répertoire taxonomies (ou env GCN_TAXONOMY_DIR)
  --gcn-bin TEXT         Chemin vers le binaire gcn (défaut: gcn)
```

Génère un `generated_NNNN.json` par phrase. Les JSON produits sont à réviser manuellement avant entraînement.

---

### `gcn_python.evaluation.metrics`

Toutes les fonctions sont pures NumPy, sans dépendances externes.

```python
from gcn_python.evaluation.metrics import (
    node_accuracy, node_f1_per_class, node_macro_f1,
    edge_accuracy, edge_f1_per_class, edge_macro_f1,
    causal_graph_similarity,
    decoder_causal_fidelity,
    cross_modal_consistency,
    roundtrip_similarity,
    generation_bleu,
)

# Métriques nœuds
pred = ["action", "processus", "action"]
gold = ["action", "action", "condition"]
node_accuracy(pred, gold)          # -> 0.333...
node_macro_f1(pred, gold)          # -> float
node_f1_per_class(pred, gold)
# -> {"action": {"precision": 0.5, "recall": 1.0, "f1": 0.67, "support": 2}, ...}

# Métriques arêtes (mêmes signatures)
edge_accuracy(pred_rels, gold_rels)
edge_macro_f1(pred_rels, gold_rels)

# Similarité de graphes causaux
sim = causal_graph_similarity(pred_cir_dict, gold_cir_dict)
# -> {"node_count_ratio": 1.0, "node_type_accuracy": 0.8,
#     "edge_count_ratio": 1.0, "edge_relation_accuracy": 0.75, "overall": 0.89}

# Fidélité du décodeur (re-parser la sortie du décodeur)
decoder_causal_fidelity(decoded_cir, gold_cir)
# -> même structure + "causal_fidelity" == "overall"

# Consistance cross-modale (fr vs python sur le même CIR)
cross_modal_consistency(ir_fr, ir_python)
# -> même structure + "consistency" == "overall"

# Fidélité roundtrip (texte → CIR → texte → CIR)
roundtrip_similarity(source_cir, decoded_cir)
# -> même structure + "roundtrip" == "overall"

# BLEU simplifié (NumPy pur)
generation_bleu("on réduit les coûts", ["on réduit les coûts de production"])
# -> float [0.0, 1.0]
```

---

### `gcn_python.evaluation.recorder`

```python
from gcn_python.evaluation.recorder import TrainingRecorder, EpochRecord

recorder = TrainingRecorder()
recorder.record(epoch=1, loss=2.31, metrics={"node_accuracy": 0.41, "edge_macro_f1": 0.28})
recorder.record(epoch=2, loss=1.87, metrics={"node_accuracy": 0.58, "edge_macro_f1": 0.45})

recorder.learning_curve()
# -> {"epoch": [1, 2], "loss": [2.31, 1.87], "node_accuracy": [0.41, 0.58], ...}

recorder.best_epoch(metric="loss", mode="min")
# -> EpochRecord(epoch=2, loss=1.87, metrics={...})

recorder.summary()
# -> {"n_epochs": 2, "first_loss": 2.31, "last_loss": 1.87, "best_loss": 1.87, ...}

recorder.to_csv(Path("curve.csv"))
recorder.to_json(Path("curve.json"))

len(recorder)  # 2
```

---

### `gcn_python.evaluation.eval_runner`

```python
from gcn_python.evaluation.eval_runner import run_eval
from pathlib import Path

report = run_eval(
    data_dir=Path("corpus/"),
    model_path=Path("model.npz"),
    lang="fr",
)
# -> {"n_samples": 120, "n_skipped": 2,
#     "node_accuracy": 0.87, "node_macro_f1": 0.83,
#     "edge_accuracy": 0.79, "edge_macro_f1": 0.74}
```

```
gcn-eval --data-dir corpus/ --model-path model.npz [--lang fr] [--output rapport.json]
```

---

### `gcn_python.verbalizer.interface`

```python
from gcn_python.verbalizer.interface import VerbalizerDecoder
```

Protocol `@runtime_checkable`. Une seule méthode :

```python
class MonDecoder:
    def decode(self, ir_json: str) -> str:
        # CausalIR JSON string -> surface texte
        # Le format de sortie dépend entièrement des données d'entraînement
        ...
```

---

### `gcn_python.verbalizer.decoder`

```python
from gcn_python.verbalizer.decoder import ReferenceDecoder

decoder = ReferenceDecoder()
surface = decoder.decode(json.dumps(cir_dict))
# -> "décroissance(ventes) -[condition]-> réduire(coûts)"
# Linéarisation structurelle — ne nécessite pas d'entraînement
```

---

### `gcn_python.verbalizer.trainable`

```python
from gcn_python.verbalizer.trainable import SurfaceVocabulary, TrainableDecoder
```

#### `SurfaceVocabulary`
```python
vocab = SurfaceVocabulary()
vocab.build(["on réduit les coûts", "si les ventes baissent"])

vocab.encode("on réduit les coûts")  # -> [2, 3, 4, 5]
vocab.decode([2, 3, 4, 5])           # -> "on réduit les coûts"
len(vocab)                            # nombre de tokens

json_str = vocab.to_json()
vocab2 = SurfaceVocabulary.from_json(json_str)
```

#### `TrainableDecoder`
Décodeur NumPy entraînable. Architecture : mean-pool(node_embeddings) → MLP 2 couches → logits vocabulaire.

```python
decoder = TrainableDecoder(vocab=surface_vocab, d_hidden=64, seed=0)

# Entraînement
logits = decoder.forward_decode(node_embeddings)   # (N, D_in) -> (|V|,)
loss, d_logits = decoder.loss_decode(logits, gold_tokens)
d_mean, layer_grads = decoder.backward_decode(d_logits)
decoder.update(layer_grads, lr=0.001)

# Inférence
surface = decoder.decode(ir_json_str)              # -> str

# Checkpoint
json_str = decoder.to_json()
decoder2 = TrainableDecoder.from_json(json_str)
```

**Entraînement conjoint avec CGNPipeline :**

```python
pipeline = CGNPipeline(encoder, graph, lang="fr", vocabulary=vocab, decoder=decoder)
cir = pipeline.forward(reps, text=text)
loss, d_node, d_edge = pipeline.loss(
    pipeline._cached_node_logits,
    pipeline._cached_edge_logits,
    gold_node,
    gold_edge,
    gold_surface=gold_surface_tokens,  # active la loss décodeur
)
pipeline.backward(d_node, d_edge, lr=0.001)
# Le gradient du décodeur se propage vers le R-GCN (couplage encodeur-décodeur)
```

---

### CLI `gcn-verbalize`

```bash
# Depuis un fichier CausalIR JSON
gcn-verbalize cir.json

# Depuis stdin
gcn analyze "Si les ventes baissent, on réduit les coûts." | gcn-verbalize -
# -> "décroissance(ventes) -[condition]-> réduire(coûts)"
```

---

## Contraintes de conception

| Contrainte | Raison |
|---|---|
| `d_out == d_clause` obligatoire | Les sorties R-GCN sont réinjectées dans le MLP nœud, qui attend `d_clause` dimensions. `CGNPipeline.__init__` lève `ValueError` si non respecté. |
| Pas de spaCy à l'inférence | `reps_from_sentence` lit les annotations directement depuis le JSON. spaCy est déclaré comme dépendance mais aucune ligne de code du moteur ne l'appelle. |
| Supervision arêtes consécutives uniquement | Le pipeline prédit les arêtes entre clauses adjacentes (gap=1, direction croissante). Les arêtes longue-distance ou inverses déclenchent un `UserWarning` et sont exclues du calcul de la loss. |
| Backward par snapshot | `MLPEncoder` sauvegarde les activations (`snapshot_node_cache`) pour permettre le backward par nœud sans re-exécuter le forward. Cela garantit des gradients corrects lors de l'accumulation sur N nœuds. |
| Protocols extensibles | `CausalEncoder` et `CausalGraph` sont des `@runtime_checkable` Protocols. Toute implémentation PyTorch, JAX ou custom peut être branchée dans `CGNPipeline` sans modification. |

---

## Licence

MIT — voir [LICENSE](https://github.com/devmail0561-web/gcn_engine/blob/master/LICENSE)
