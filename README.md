# GCN-Core — Grammaire Causale Naturelle

[![crates.io](https://img.shields.io/crates/v/gcn-ir?label=gcn-ir)](https://crates.io/crates/gcn-ir)
[![PyPI](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust tests](https://img.shields.io/badge/tests%20Rust-137%20%E2%9C%85-brightgreen)](https://github.com/devmail0561-web/gcn_engine)
[![Python tests](https://img.shields.io/badge/tests%20Python-231%20%E2%9C%85-brightgreen)](https://github.com/devmail0561-web/gcn_engine)
[![Version](https://img.shields.io/badge/version-2.4.0-blue.svg)](https://pypi.org/project/gcn-python/)

**Moteur de raisonnement causal** — infrastructure sur laquelle les data scientists et analystes construisent et entraînent leurs propres modèles causaux.

> *"La causalité n'est pas extérieure au langage — elle y est encodée de manière systématique à travers chaque partie du discours."*
> — Michel Tendeng, UN-CHK, Septembre 2026

---

## Qu'est-ce que GCN ?

GCN est un **engine**, pas un modèle. Il fournit le cadre formel et les briques techniques pour :

1. **Extraire** la structure causale depuis du texte naturel ou du code
2. **Raisonner** sur des graphes causaux (niveaux de Pearl)
3. **Générer** du texte et du code depuis une vérité causale vérifiée

Contrairement aux LLMs qui corrèlent des tokens, GCN représente **explicitement** les relations causales — qui cause quoi, à quel moment, sous quelle condition.

---

## Architecture

```
Texte / Code
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│  Frontend   │────▶│   CausalIR       │  Représentation Intermédiaire Causale
│  (par lang) │     │   (graphe JSON)  │  100% sérialisable
└─────────────┘     └──────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Middle-end   │  Enrichissement : cycles, contraintes, validation
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Backend     │  Raisonnement Pearl + GCN-QL
                    └───────────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
               Texte généré   Code généré
               (gcn-verbalizer, Phase 5)
```

**Principe directeur :** le graphe causal est le seul objet de vérité. Toute génération en aval est un acte de verbalisation — pas de raisonnement.

---

## Structure du projet

```
projet_CNM/
├── BENCHMARK.md        ← Benchmark complet ML (17 runs, meilleures combos, analyse)
├── PROGRESS.md         ← Suivi d'avancement détaillé par phase
├── TODO.md             ← Plan d'action A→E (complété en v2.4.0)
│
├── gcn-core/           ← Moteur Rust pur (workspace Cargo)
│   ├── crates/
│   │   ├── gcn-ir/             Représentation causale intermédiaire (types purs)
│   │   ├── gcn-knowledge/      Chargement des taxonomies YAML
│   │   ├── gcn-frontend-fr/    Parser français symbolique (bootstrap annotation)
│   │   ├── gcn-frontend-en/    Parser anglais symbolique (bootstrap annotation)
│   │   ├── gcn-frontend-code/  Parser code AST (Python, Rust, JS via tree-sitter)
│   │   ├── gcn-middleend/      Construction graphe, cycles, validation
│   │   ├── gcn-backend/        Raisonnement Pearl 1-2-3 + GCN-QL + export
│   │   ├── gcn-verbalizer/     Décodeur CausalIR → surface (pont Rust)
│   │   └── gcn-cli/            Interface ligne de commande
│   └── SAD.md                  Software Architecture Document
│
├── gcn-python/         ← Couches ML Python (framework-agnostique)
│   ├── models/
│   │   └── wiki.fr.vec         Embeddings FastText français (22 218 mots, 300d)
│   └── src/gcn_python/
│       ├── layer1/     UDRepresentation + vectorisation (depuis tokens JSON annotés)
│       ├── layer2/     CausalEncoder Protocol (MLP référence NumPy)
│       ├── layer3/     CausalGraph Protocol (R-GCN NumPy + RGCNLayerGAT PyTorch)
│       ├── pipeline/   CGNPipeline.forward() + loss() + backward()
│       ├── data/       GCNDataLoader — itère sur gcn-datasets/ → TrainingSample
│       ├── training/   gcn-train + gcn-bootstrap CLI + checkpoint save/load
│       ├── verbalizer/ Décodeur NumPy référence + gcn-verbalize CLI
│       ├── frontend/   GCNBridgeParser (bridge heuristique ~80-85% qualité)
│       └── evaluation/ Métriques + TrainingRecorder
│
├── gcn-tools/         ← Outils externes (hors moteur)
│   └── gcn-annotate/  Outil d'annotation LLM (Anthropic/OpenAI)
│
├── gcn-references/     ← Références linguistiques (hors moteur)
│   └── taxonomies/
│       ├── fr/         11 taxonomies GCN (YAML) — français
│       ├── en/         10 taxonomies GCN (YAML) — anglais
│       ├── python/     Mappings AST Python → types causaux
│       ├── rust/       Mappings AST Rust → types causaux
│       └── js/         Mappings AST JavaScript → types causaux
│
└── gcn-datasets/       ← Données annotées (hors moteur)
    ├── schemas/        gcn-nl.schema.yaml, gcn-pl.schema.yaml, gcn-verbalize.schema.yaml
    ├── examples/       Exemples illustratifs annotés (FR, Python, Rust)
    ├── corpus/         Textes bruts pour bootstrap (generated_1000.json, phrases_fr.txt)
    ├── raw/            Textes bruts non annotés + annotations intermédiaires (phase4/)
    ├── splits/         Ancien split alternatif (693/148/149 phrases)
    ├── scripts/        Scripts Python d'annotation et de gestion
    │   ├── build_annotations.py    Génère JSON complet (CIR + tokens UD) depuis candidats manuels
    │   ├── candidates_c1.py        171 phrases annotées pour 6 types de relations rares
    │   ├── generate_dataset.py     Bootstrap depuis textes bruts
    │   ├── merge_datasets.py       Fusionne plusieurs splits JSON
    │   └── oversample_rare.py      Oversample les classes rares
    └── real/           ← Données annotées officielles
        ├── source/     Fichiers sources originaux (avant split)
        ├── train/      Split officiel — 536 phrases annotées
        ├── val/        Split officiel — 114 phrases annotées
        ├── test/       Split officiel — 117 phrases annotées
        └── augmented/  Datasets dérivés (construits à partir des splits officiels)
            ├── c1_annotations/   171 nouvelles phrases (6 types de relations rares)
            ├── c1_merged/        707 phrases (train + C1)
            ├── c1_oversampled/   735 phrases ← MEILLEUR DATASET D'ENTRAÎNEMENT ✅
            ├── final/            849 phrases (c1_oversampled + val — pour checkpoint prod)
            └── oversampled_v0/   647 phrases (oversampling sans C1 — référence)
```

---

## Prérequis

### Rust (moteur symbolique)
- Rust **1.85+** (édition 2024)
- Cargo (inclus avec Rust)

### Python (couches ML)
- Python **3.10+**
- `pip install -e gcn-python/`
- `pip install torch` (optionnel — pour `RGCNLayerPT` GPU/MPS)

---

## Installation

### Depuis les registres publics (recommandé)

```bash
# Package Python (couches ML)
pip install gcn-python

# Crates Rust (dans votre projet)
cargo add gcn-ir gcn-backend gcn-frontend-fr
```

### Depuis les sources (moteur complet + CLI `gcn`)

```bash
git clone https://github.com/devmail0561-web/gcn_engine.git
cd gcn_engine

make install        # build release + pip install gcn-python
# ou
make install-dev    # build debug + pip install -e gcn-python
```

---

## Utilisation rapide

### Analyser un texte
```bash
gcn analyze "Si les ventes baissent, on réduit les coûts." \
    --data-dir ./gcn-references/taxonomies
```

Sortie JSON :
```json
{
  "source_lang": { "natural": { "lang": "french" } },
  "nodes": [
    { "id": 0, "node_type": "processus", "label": "décroissance(ventes)", ... },
    { "id": 1, "node_type": "action",    "label": "réduire(coûts)", ... }
  ],
  "edges": [
    [0, 1, { "relation": "condition", "confidence": 1.0, ... }]
  ],
  ...
}
```

### Requêtes causales (GCN-QL)
```bash
# Sauvegarder l'IR
gcn analyze "Les ventes baissent parce que la qualité a chuté." \
    --data-dir ./gcn-references/taxonomies > graph.json

# Niveau 1 — Association
gcn query "WHY ventes?" --ir graph.json        # ancêtres causaux
gcn query "WHAT qualité?" --ir graph.json       # descendants causaux
gcn query "CHAIN qualité -> ventes?" --ir graph.json  # chemin causal
gcn query "CYCLES?" --ir graph.json             # boucles de rétroaction
gcn query "GAPS?" --ir graph.json               # lacunes causales

# Niveau 2 — Intervention do-calculus
gcn query "DO qualité" --ir graph.json
# → severed_count (causes coupées), effects (propagation forcée)

# Niveau 3 — Contrefactuel
gcn query "COUNTERFACTUAL qualité" --ir graph.json
# → actual_effects (monde réel), unique_effects (n'auraient pas eu lieu sans qualité)
```

### Analyser un texte anglais
```bash
gcn analyze "Sales fell because costs rose." \
    --data-dir ./gcn-references/taxonomies --format json
# source_lang: { natural: { lang: "english" } }
# Même CausalIR que l'équivalent français — isomorphisme fr↔en
```

### Export Graphviz
```bash
gcn analyze "..." --data-dir ./gcn-references/taxonomies --format dot | dot -Tpng > graph.png
```

### Pipeline Python (couches ML)

Le moteur ne dépend pas de spaCy. Il opère sur des `UDRepresentation` construites depuis des fichiers JSON annotés (format GCN-NL). La CLI prend un fichier dataset en entrée.

```bash
# Inférence ML via l'API Python (le binaire `gcn-forward` a été supprimé,
# tout comme la sous-commande `gcn forward` — redondants avec `gcn-discuss`)
python3 -c "
from gcn_python import GCNEngine
engine = GCNEngine.from_pretrained('model.npz', trusted=True)
cir = engine.analyze('Les ventes baissent.')
print(cir)
"

# Session interactive sur corpus
gcn-discuss --checkpoint model.npz
```

### Entraîner un modèle

```bash
# Générer des données d'entraînement (JSON annoté) depuis des textes bruts
gcn-bootstrap --input phrases_fr.txt --out-dir gcn-datasets/generated/

# Lancer l'entraînement — configuration de référence v2.4.0 (val_edge_f1=0.468)
gcn-train \
  --data-dir  gcn-datasets/real/augmented/c1_oversampled/ \
  --val-dir   gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional \
  --output model.npz

# Charger un checkpoint entraîné
# (GCNEngine reconstruit l'architecture depuis les métadonnées du .npz)
from gcn_python import GCNEngine
engine = GCNEngine.from_pretrained("model.npz")
```

> **Voir [BENCHMARK.md](BENCHMARK.md) pour le benchmark complet de toutes les configurations testées.**

Le data scientist substitue `MLPEncoder` et `RGCNLayer` par ses propres implémentations PyTorch/JAX via les Protocol `CausalEncoder` et `CausalGraph`.

---

## GCN-QL — Langage de requêtes causales

| Syntaxe | Description | Niveau Pearl |
|---|---|---|
| `WHY <label>?` | Ancêtres causaux d'un nœud | 1 — Association |
| `WHAT <label>?` | Descendants causaux d'un nœud | 1 — Association |
| `CHAIN <a> -> <b>?` | Chemin causal entre deux nœuds | 1 — Association |
| `CYCLES?` | Liste les boucles de rétroaction | 1 — Association |
| `GAPS?` | Liste les lacunes causales non résolues | 1 — Association |
| `DO <label>?` | Intervention : court-circuite les causes de X, propage ses effets | 2 — Intervention |
| `COUNTERFACTUAL <label>?` | "Que se serait-il passé si X n'avait pas eu lieu ?" | 3 — Contrefactuel |

**Pearl niveau 2 — `DO X`** : coupe toutes les arêtes entrantes du nœud X (ses causes naturelles sont court-circuitées), puis propage les effets en avant depuis X. Retourne les arêtes coupées (`severed`) et les effets aval.

**Pearl niveau 3 — `COUNTERFACTUAL X`** : compare le monde actuel (effets réels de X) avec le monde hypothétique sans X. `unique_effects` = nœuds qui ne seraient PAS atteints si X n'avait pas eu lieu (aucun chemin alternatif depuis les racines du graphe).

---

## API Rust

### `gcn-ir` — Types de base

```rust
// Représentation causale intermédiaire
pub struct CausalIR {
    pub source_lang: SourceLanguage,
    pub source_text: String,
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(NodeId, NodeId, CausalEdge)>,
    pub cycles: Vec<CausalCycle>,
    pub unresolved: Vec<Ambiguity>,
    pub metadata: IrMetadata,
}

// Types de nœuds
pub enum NodeType { Etat, Action, Transition, Processus, Condition, Entite, EtatSystemique }

// Types de relations
pub enum RelationType {
    Cause, Enable, Prevent, Condition, Concession, Sequence,
    Motivation, Filter, Opposition, DataDependency, ControlDependency
}
```

### `gcn-frontend-fr` / `gcn-frontend-en` — Parsers symboliques

```rust
// Français
let parser = FrenchParser::new(Path::new("gcn-references/taxonomies"))?;
let ir: CausalIR = parser.parse("Si les ventes baissent, on réduit les coûts.")?;

// Anglais — même API, même CausalIR produit
let parser = EnglishParser::new(Path::new("gcn-references/taxonomies"))?;
let ir: CausalIR = parser.parse("If costs rise, sales fall.")?;
// Les deux parsers produisent un CIR isomorphe (même RelationType::Condition)
```

### `gcn-middleend` — Enrichissement

```rust
let result = gcn_middleend::process(ir)?;
// result.ir      : CausalIR avec cycles détectés et gaps propagés
// result.diagnostics : avertissements et erreurs structurelles
```

### `gcn-backend` — Raisonnement et export

```rust
// GCN-QL niveau 1 (association)
let query = Query::parse("WHY ventes?")?;
let result = execute(&query, &ir)?;

// GCN-QL niveau 2 (intervention do-calculus)
let query = Query::parse("DO crise")?;
// -> QueryResult::Intervention { severed_count, severed, effects, .. }

// GCN-QL niveau 3 (contrefactuel)
let query = Query::parse("COUNTERFACTUAL hausse")?;
// -> QueryResult::CounterfactualDiff { actual_effects, unique_effects, .. }

// Export
let json = to_json(&ir)?;
let dot  = to_dot(&ir)?;
```

---

## API Python (couches ML)

### Implémenter un `CausalEncoder` (Couche 2)

```python
import numpy as np
from gcn_python.layer2.interface import CausalEncoder

class MyEncoder:
    """Implémentation PyTorch, JAX ou autre — à vous de choisir."""

    def forward_node(self, x: np.ndarray) -> np.ndarray:
        # x: (D_clause = 80,) → retourne (7,) logits sur NodeType
        ...

    def forward_edge(self, x: np.ndarray) -> np.ndarray:
        # x: (D_edge ≈ 346,) → retourne (11,) logits sur RelationType
        ...

    def parameters(self) -> list[np.ndarray]: ...
    def update(self, grads, lr): ...
```

### Utiliser la couche 3 R-GCN (NumPy ou PyTorch)

```python
# Référence NumPy (pas de dépendance ML)
from gcn_python.layer3.reference import RGCNLayer
layer = RGCNLayer(d_in=64, d_out=128)

# PyTorch optimisé (GPU/MPS, autograd)
from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
layer = RGCNLayerPT(d_in=64, d_out=128, device="cuda")

# Les deux implémentent le même Protocol CausalGraph
out = layer.message_pass(node_features, edge_index, edge_types)  # (N, D_out)

# Entraînement PyTorch natif
optimizer = torch.optim.Adam(layer.torch_parameters(), lr=1e-3)
out = layer.forward_torch(H, edge_index, edge_types)
loss.backward()
optimizer.step()
```

### Implémenter son propre `CausalGraph` R-GCN

```python
from gcn_python.layer3.interface import CausalGraph

class MyRGCN:
    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_node)
        edge_index: np.ndarray,     # (2, E)
        edge_types: np.ndarray,     # (E,) int
    ) -> np.ndarray:                # (N, D_out)
        # R-GCN : h_i = σ( Σ_r Σ_{j∈N_r(i)} (1/c_{ir}) W_r h_j + W_0 h_i )
        ...
```

### Utiliser le pipeline complet

```python
from pathlib import Path
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence

vocab = FeatureVocabulary()

pipeline = CGNPipeline(
    encoder=MyEncoder(),
    graph=MyRGCN(),
    vocabulary=vocab,
    # Paramètres optionnels (Phase 10)
    temperature=1.0,        # S12 : temperature softmax sur les logits
    n_rgcn_layers=1,        # S5  : nombre de couches R-GCN empilées
    all_pairs=False,        # S4  : toutes les paires de clauses (True) ou consécutives seulement
    word_embedding=None,    # S1/S2/S9 : WordEmbedding apprenante (lookup root_lemma)
)

loader = GCNDataLoader(Path("gcn-datasets/examples/"))
for sample in loader:
    reps, valid_idxs, connector_reps = reps_from_sentence(sample.sentence)
    causal_ir_dict = pipeline.forward(
        reps,
        text=sample.sentence.text,
        connector_reps=connector_reps,
    )
```

### Évaluation

```python
from gcn_python.evaluation.metrics import causal_graph_similarity, node_f1_per_class
from gcn_python.evaluation.recorder import TrainingRecorder

recorder = TrainingRecorder()
for epoch in range(100):
    pipeline.forward(text)
    node_logits = pipeline._cached_node_logits
    edge_logits = pipeline._cached_edge_logits
    loss_val, d_node, d_edge = pipeline.loss(
        node_logits, edge_logits, gold_node_labels, gold_edge_labels
    )
    pipeline.backward(d_node, d_edge, lr=0.001)
    metrics = causal_graph_similarity(pred_ir, gold_ir)
    recorder.record(epoch, loss_val, metrics)

recorder.to_csv("training_log.csv")
print(recorder.best_epoch("overall", "max"))
```

---

## Format des datasets

Les datasets suivent le schéma `gcn-datasets/schemas/gcn-nl.schema.yaml`. Le format de stockage est **JSON** (depuis v0.9.4).

Structure minimale d'un exemple annoté :

```json
{
  "document": {
    "id": "doc-fr-001",
    "lang": "fr",
    "sentences": [
      {
        "id": "s001",
        "text": "Si les ventes baissent, on réduit les coûts.",
        "tokens": [
          { "id": 1, "form": "Si", "lemma": "si", "pos": "SCONJ",
            "gcn": { "causal_type": "conjonction", "causal_class": "condition" } }
        ],
        "cir": {
          "nodes": [
            { "id": "n001", "type": "processus", "label": "décroissance(ventes)",
              "token_span": [3, 4], "attributes": { "entity": "ventes" } },
            { "id": "n002", "type": "action", "label": "réduire(coûts)",
              "token_span": [6, 9], "attributes": { "agent": "on" } }
          ],
          "edges": [
            { "source": "n001", "target": "n002", "relation": "condition",
              "attributes": { "confidence": 1.0, "explicit": true, "marker_token": 1 } }
          ]
        }
      }
    ]
  }
}
```

---

## Tests

```bash
# Suite complète Rust (137 tests)
cd gcn-core && cargo test --workspace

# Par crate
cargo test -p gcn-ir
cargo test -p gcn-frontend-fr      # 21 tests
cargo test -p gcn-frontend-en      # 16 tests (dont isomorphisme fr↔en)
cargo test -p gcn-frontend-code    # 26 tests (Python, Rust, JS)
cargo test -p gcn-middleend        # 17 tests
cargo test -p gcn-backend          # 34 tests (Pearl 1-2-3)

# Python (231 tests)
cd gcn-python && python -m pytest
# Dont :
#   test_regression_v230.py  — 17 tests régression correctifs v2.3.0
#   test_e2e_pipeline.py     —  8 tests e2e texte brut → CIR JSON
```

---

## Phases d'implémentation

| Phase | Composants | Statut |
|---|---|---|
| 1 | `gcn-ir` + `gcn-knowledge` | ✅ Terminé |
| 2a | `gcn-frontend-fr` (21 tests) | ✅ Terminé |
| 2b | `gcn-python` couches 1-3 (référence NumPy) | ✅ Terminé |
| 2c | `gcn-python/evaluation` (métriques, TrainingRecorder) | ✅ Terminé |
| 3 | `gcn-middleend` (17 tests) | ✅ Terminé |
| 4 | `gcn-backend` Pearl 1 + `gcn-cli` (34 tests) | ✅ Terminé |
| 5 | `gcn-verbalizer` + décodeur entraînable (21 tests) | ✅ Terminé |
| 6 | `gcn-frontend-code` — AST Python/Rust/JS (26 tests) | ✅ Terminé |
| 7 | Pearl 2-3, R-GCN PyTorch, `gcn-frontend-en` (16 tests) | ✅ Terminé |
| 8 | Mise en production — Makefile, gcn-eval, publication | ✅ Terminé |
| 9 | Corrections pipeline ML (14 problèmes, 120 tests Python) | ✅ Terminé |
| 10 | Correctifs structurels moteur (12 défauts, 192 tests Python) | ✅ Terminé |

---

## Performances ML — État actuel (v2.4.0)

### Checkpoint de production

| Fichier | `gcn-datasets/checkpoints/model_v2.4.0.npz` |
|---------|---------------------------------------------|
| Métriques | `gcn-datasets/checkpoints/model_v2.4.0_metrics.json` |
| Chargement | `GCNEngine.from_pretrained("model_v2.4.0.npz")` |

### Résultats sur le val set (114 phrases françaises)

| Métrique | Valeur | Cible prod | Statut |
|---------|--------|-----------|--------|
| `val_edge_macro_f1` | **0.468** | > 0.40 | ✅ |
| `val_node_macro_f1` | 0.274 | > 0.60 | ✗ (données insuffisantes) |
| `val_graph_exact_match` | 0.123 | > 0.20 | ✗ (bloqué par node) |
| gap train−val (arêtes) | 0.069 | < 0.15 | ✅ |

### Meilleure configuration d'entraînement

```bash
gcn-train \
  --data-dir  gcn-datasets/real/augmented/c1_oversampled/ \
  --val-dir   gcn-datasets/real/val/ \
  --epochs    100 \
  --lr        0.0005 \
  --weighted-loss \
  --use-attention \
  --bidirectional \
  --output    model.npz
```

**Leviers validés expérimentalement :**
- `--use-attention` (GAT) : +0.09 sur val_edge_f1 vs R-GCN NumPy (facteur le plus important)
- `--bidirectional` : +0.07 (aide GAT, nuit à R-GCN — ne pas combiner avec R-GCN seul)
- Oversampling C1 (classes rares à 30 ex.) : +0.07
- LR bas (0.001 → 0.0005) : convergence stable, +0.08

**Ce qui ne marche pas :** `--edge-loss-weight` (0 effet), `--label-smoothing` (−0.05),
embeddings aléatoires (−0.10 vs GAT sans emb), `--rgcn-dropout` (−0.05).

> **Benchmark complet (17 runs, toutes les combinaisons testées) → [BENCHMARK.md](BENCHMARK.md)**

### Bloquant restant pour la production complète

`val_node_macro_f1 = 0.274` (cible 0.60) est un **problème de données, pas de modèle**.
Les 5 types de nœuds rares (etat, action, transition, etat_systemique, condition) n'ont que
9 à 35 exemples chacun. Aucun hyperparamètre ne peut compenser l'absence de données.
Solution : annoter ~800 phrases supplémentaires ciblant ces types.

---

## Nouveautés v2.4.0

- **Checkpoint de production** `model_v2.4.0.npz` — val_edge_f1=0.468 sur 849 phrases
- **Dataset C1** : 171 exemples annotés pour 6 types de relations rares (filter, data_dependency,
  control_dependency, motivation, sequence, opposition) — `gcn-datasets/real/augmented/c1_annotations/`
- **Oversampling** : script `gcn-datasets/oversample_rare.py` — classes rares portées à 30 ex.
- **Robustesse moteur** : 0 `assert` dans le moteur (remplacés par ValueError/RuntimeError),
  issues MEDIUM #4-5 sur `token_span` corrigées
- **Tests e2e** : 8 tests texte brut → CIR JSON (phrase simple/complexe/sans causalité)
- **231 tests Python** (vs 192 en v2.3.0) dont 17 tests de régression v2.3.0
- **Audit 1** : 7 correctifs scripts post-restructuration — `build_annotations.py`, `oversample_rare.py`, `generate_dataset.py`, `make_verbalize_pairs.py`, `merge_datasets.py`

---

## Phases d'implémentation

| Phase | Composants | Statut |
|---|---|---|
| 1 | `gcn-ir` + `gcn-knowledge` | ✅ Terminé |
| 2a | `gcn-frontend-fr` (21 tests) | ✅ Terminé |
| 2b | `gcn-python` couches 1-3 (référence NumPy) | ✅ Terminé |
| 2c | `gcn-python/evaluation` (métriques, TrainingRecorder) | ✅ Terminé |
| 3 | `gcn-middleend` (17 tests) | ✅ Terminé |
| 4 | `gcn-backend` Pearl 1 + `gcn-cli` (34 tests) | ✅ Terminé |
| 5 | `gcn-verbalizer` + décodeur entraînable (21 tests) | ✅ Terminé |
| 6 | `gcn-frontend-code` — AST Python/Rust/JS (26 tests) | ✅ Terminé |
| 7 | Pearl 2-3, R-GCN PyTorch, `gcn-frontend-en` (16 tests) | ✅ Terminé |
| 8 | Mise en production — Makefile, gcn-eval, publication | ✅ Terminé |
| 9 | Corrections pipeline ML (14 problèmes, 120 tests Python) | ✅ Terminé |
| 10 | Correctifs structurels moteur (12 défauts, 192 tests Python) | ✅ Terminé |
| A-v2.3.0 | Audit max — 11 correctifs gradient/reproductibilité (231 tests) | ✅ Terminé |
| B–E | Mesure, tuning, oversampling, robustesse, checkpoint v2.4.0 | ✅ Terminé |
| Audit 1 | 7 correctifs scripts post-restructuration (paths, spans, edges) | ✅ Terminé |

---

## Références

- **Benchmark complet (17 runs ML) :** [BENCHMARK.md](BENCHMARK.md)
- **Suivi d'avancement détaillé :** `PROGRESS.md`
- **Papier de recherche :** `docs/grammaire_causale_naturelle_v2.docx`
- **Architecture détaillée :** `gcn-core/SAD.md`
- **Limitations connues GCNBridgeParser :** `gcn-python/README.md#limitations`
- **Auteur :** Michel Tendeng — Université Numérique Cheikh Hamidou Kane (UN-CHK), L3 Cybersécurité, Ziguinchor, Sénégal
