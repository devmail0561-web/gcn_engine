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
│   │   ├── gcn-frontend-fr/    Parser français symbolique
│   │   ├── gcn-frontend-code/  Parser code (stub — Phase 6)
│   │   ├── gcn-middleend/      Construction graphe, cycles, validation
│   │   ├── gcn-backend/        Raisonnement Pearl + GCN-QL + export
│   │   └── gcn-cli/            Interface ligne de commande
│   └── SAD.md                  Software Architecture Document
│
├── gcn-python/         ← Couches ML Python (framework-agnostique)
│   └── src/gcn_python/
│       ├── layer1/     Extraction UD (spaCy) → vecteurs de clauses
│       ├── layer2/     CausalEncoder Protocol (MLP référence NumPy)
│       ├── layer3/     CausalGraph Protocol (R-GCN référence NumPy)
│       ├── pipeline/   CGNPipeline.forward() + gcn-forward CLI
│       └── evaluation/ Métriques + TrainingRecorder
│
├── gcn-references/     ← Références linguistiques (hors moteur)
│   └── taxonomies/     11 taxonomies GCN (YAML) — français
│
└── gcn-datasets/       ← Données annotées (hors moteur)
    ├── schemas/        gcn-nl.schema.yaml, gcn-pl.schema.yaml
    └── examples/       Exemples annotés FR, Python, Rust
```

---

## Prérequis

### Rust (moteur symbolique)
- Rust **1.85+** (édition 2024)
- Cargo (inclus avec Rust)

### Python (couches ML)
- Python **3.10+**
- `pip install -e gcn-python/`
- `python -m spacy download fr_core_news_sm`

---

## Installation

```bash
# Cloner le dépôt
git clone <url> projet_CNM
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

# WHY : remonter les causes
gcn query "WHY ventes?" --ir graph.json

# WHAT : descendre les effets
gcn query "WHAT qualité?" --ir graph.json

# CHAIN : y a-t-il un chemin causal ?
gcn query "CHAIN qualité -> ventes?" --ir graph.json

# CYCLES : boucles de rétroaction
gcn query "CYCLES?" --ir graph.json

# GAPS : lacunes causales
gcn query "GAPS?" --ir graph.json
```

### Export Graphviz
```bash
gcn analyze "..." --data-dir ./gcn-references/taxonomies --format dot | dot -Tpng > graph.png
```

### Pipeline Python (couches ML)
```bash
# Via la CLI Python (référence NumPy)
gcn-forward --lang fr --taxonomy-dir ./gcn-references/taxonomies "Les ventes baissent."

# Via gcn-cli (interface Rust↔Python)
gcn forward "Les ventes baissent." --lang fr --enrich
```

---

## GCN-QL — Langage de requêtes causales

| Syntaxe | Description | Niveau Pearl |
|---|---|---|
| `WHY <label>?` | Ancêtres causaux d'un nœud | 1 — Association |
| `WHAT <label>?` | Descendants causaux d'un nœud | 1 — Association |
| `CHAIN <a> -> <b>?` | Chemin causal entre deux nœuds | 1 — Association |
| `CYCLES?` | Liste les boucles de rétroaction | 1 — Association |
| `GAPS?` | Liste les lacunes causales non résolues | 1 — Association |

Les niveaux 2 (intervention `do(X)`) et 3 (contrefactuels) seront ajoutés en Phase 7.

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

### `gcn-frontend-fr` — Parser français

```rust
let parser = FrenchParser::new(Path::new("gcn-references/taxonomies"))?;
let ir: CausalIR = parser.parse("Si les ventes baissent, on réduit les coûts.")?;
```

### `gcn-middleend` — Enrichissement

```rust
let result = gcn_middleend::process(ir)?;
// result.ir      : CausalIR avec cycles détectés et gaps propagés
// result.diagnostics : avertissements et erreurs structurelles
```

### `gcn-backend` — Raisonnement et export

```rust
// GCN-QL
let query = Query::parse("WHY ventes?")?;
let result = execute(&query, &ir)?;

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

### Implémenter un `CausalGraph` R-GCN (Couche 3)

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
    loss = pipeline.loss(pred_ir, gold_ir)
    metrics = causal_graph_similarity(pred_ir, gold_ir)
    recorder.record(epoch, loss, metrics)

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
# Suite complète (51 tests)
cd gcn-core && cargo test

# Par crate
cargo test -p gcn-ir
cargo test -p gcn-frontend-fr
cargo test -p gcn-middleend
cargo test -p gcn-backend

# Python
cd gcn-python && python -m pytest
```

---

## Phases d'implémentation

| Phase | Composants | Statut |
|---|---|---|
| 1 | `gcn-ir` + `gcn-knowledge` | ✅ Terminé |
| 2a | `gcn-frontend-fr` (16 tests) | ✅ Terminé |
| 2b | `gcn-python` couches 1-3 | 🔄 En cours |
| 2c | `gcn-python/evaluation` | ✅ Terminé |
| 3 | `gcn-middleend` (17 tests) | ✅ Terminé |
| 4 | `gcn-backend` + `gcn-cli` (18 tests) | ✅ Terminé |
| 5 | `gcn-verbalizer` — graphe → texte **et** code | ⬜ |
| 6 | `gcn-frontend-code` — AST Python/Rust/JS | ⬜ |
| 7 | Pearl niveaux 2-3, R-GCN optimisé, wolof/arabe | ⬜ |

---

## Références

- **Papier de recherche :** `docs/grammaire_causale_naturelle_v2.docx`
- **Architecture détaillée :** `gcn-core/SAD.md`
- **Auteur :** Michel Tendeng — Université Numérique Cheikh Hamidou Kane (UN-CHK), L3 Cybersécurité, Ziguinchor, Sénégal
