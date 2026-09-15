# GCN-Core — Grammaire Causale Naturelle

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
│   └── src/gcn_python/
│       ├── layer1/     UDRepresentation + vectorisation (depuis tokens JSON annotés)
│       ├── layer2/     CausalEncoder Protocol (MLP référence NumPy)
│       ├── layer3/     CausalGraph Protocol (R-GCN NumPy + RGCNLayerPT PyTorch)
│       ├── pipeline/   CGNPipeline.forward() + loss() + backward() + gcn-forward CLI
│       ├── data/       GCNDataLoader — itère sur gcn-datasets/ → TrainingSample
│       ├── training/   gcn-train + gcn-bootstrap CLI + checkpoint save/load
│       ├── verbalizer/ Décodeur NumPy référence + gcn-verbalize CLI
│       └── evaluation/ Métriques + TrainingRecorder
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
    └── examples/       Exemples annotés FR, Python, Rust, cross-modal (format JSON)
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

```bash
# Cloner le dépôt
git clone https://github.com/devmail0561-web/gcn_engine.git projet_CNM
cd projet_CNM

# Compiler le moteur Rust
cd gcn-core
cargo build --release

# Installer le package Python (optionnel — couches ML)
cd ../gcn-python
pip install -e .
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
# Inférence depuis un fichier JSON annoté (poids aléatoires sans --model-path)
gcn-forward gcn-datasets/examples/fr_causal_basic.json \
    --lang fr --taxonomy-dir ./gcn-references/taxonomies

# Cibler une sentence spécifique
gcn-forward gcn-datasets/examples/fr_causal_basic.json \
    --sentence-id s001 --lang fr

# Inférence avec un modèle entraîné
gcn-forward gcn-datasets/examples/fr_causal_basic.json \
    --lang fr \
    --taxonomy-dir ./gcn-references/taxonomies \
    --model-path model.npz

# Via gcn-cli (interface Rust↔Python)
gcn forward "Les ventes baissent." --lang fr --enrich
```

### Entraîner un modèle

```bash
# Générer des données d'entraînement (JSON annoté) depuis des textes bruts
gcn-bootstrap --input phrases_fr.txt --lang fr --out-dir gcn-datasets/generated/

# Lancer l'entraînement (SGD NumPy référence) — requiert des JSON avec tokens annotés
gcn-train --data-dir gcn-datasets/generated/ \
          --taxonomy-dir ./gcn-references/taxonomies \
          --epochs 50 --lr 0.001 --output model.npz
```

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
        # x: (D_clause ≈ 136,) → retourne (7,) logits sur NodeType
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
from gcn_python.taxonomy.loader import TaxonomyIndex

tax = TaxonomyIndex.load(Path("gcn-references/taxonomies"), "fr")
vocab = FeatureVocabulary.build(tax)

pipeline = CGNPipeline(
    encoder=MyEncoder(),
    graph=MyRGCN(),
    taxonomy_dir=Path("gcn-references/taxonomies"),
    lang="fr",
    vocabulary=vocab,
)

causal_ir_dict = pipeline.forward("Les ventes baissent parce que la qualité a chuté.")
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

Les datasets suivent le schéma `gcn-datasets/schemas/gcn-nl.schema.yaml`.

Structure minimale d'un exemple annoté :

```yaml
document:
  id: "doc-fr-001"
  lang: fr
  sentences:
    - id: "s001"
      text: "Si les ventes baissent, on réduit les coûts."
      tokens:
        - { id: 1, form: "Si", lemma: "si", pos: "SCONJ",
            gcn: { causal_type: conjonction, causal_class: condition } }
        # ...
      cir:
        nodes:
          - { id: "n001", type: processus, label: "décroissance(ventes)",
              token_span: [3, 4], attributes: { entity: "ventes" } }
          - { id: "n002", type: action,    label: "réduire(coûts)",
              token_span: [6, 9], attributes: { agent: "on" } }
        edges:
          - { source: "n001", target: "n002", relation: condition,
              attributes: { confidence: 1.0, explicit: true, marker_token: 1 } }
```

---

## Tests

```bash
# Suite complète Rust (103 tests)
cd gcn-core && cargo test --workspace

# Par crate
cargo test -p gcn-ir
cargo test -p gcn-frontend-fr      # 16 tests
cargo test -p gcn-frontend-en      # 16 tests (dont isomorphisme fr↔en)
cargo test -p gcn-frontend-code    # 27 tests (Python, Rust, JS)
cargo test -p gcn-middleend        # 17 tests
cargo test -p gcn-backend          # 26 tests (Pearl 1-2-3)

# Python (76 collectés, 71 passent, 5 skippés sans spaCy fr)
cd gcn-python && python -m pytest
```

---

## Phases d'implémentation

| Phase | Composants | Statut |
|---|---|---|
| 1 | `gcn-ir` + `gcn-knowledge` | ✅ Terminé |
| 2a | `gcn-frontend-fr` (16 tests) | ✅ Terminé |
| 2b | `gcn-python` couches 1-3 (référence NumPy) | ✅ Terminé |
| 2c | `gcn-python/evaluation` (métriques, TrainingRecorder) | ✅ Terminé |
| 3 | `gcn-middleend` (17 tests) | ✅ Terminé |
| 4 | `gcn-backend` Pearl 1 + `gcn-cli` (26 tests) | ✅ Terminé |
| 5 | `gcn-verbalizer` — CausalIR → surface (texte et code) | ✅ Terminé |
| 6 | `gcn-frontend-code` — AST Python/Rust/JS (27 tests) | ✅ Terminé |
| 7 | Pearl 2-3, R-GCN PyTorch, `gcn-frontend-en` (16 tests) | ✅ Terminé |

---

## Références

- **Papier de recherche :** `docs/grammaire_causale_naturelle_v2.docx`
- **Architecture détaillée :** `gcn-core/SAD.md`
- **Auteur :** Michel Tendeng — Université Numérique Cheikh Hamidou Kane (UN-CHK), L3 Cybersécurité, Ziguinchor, Sénégal
