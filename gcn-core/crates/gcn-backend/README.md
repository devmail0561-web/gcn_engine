# gcn-backend — Raisonnement Pearl, GCN-QL et Export

Version: 2.5.0

[![Crates.io](https://img.shields.io/crates/v/gcn-backend)](https://crates.io/crates/gcn-backend)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Backend de raisonnement du moteur **GCN-Core**. Implémente les trois niveaux de causalité de Judea Pearl, le langage de requête GCN-QL, et l'export du graphe causal en JSON ou DOT (Graphviz).

---

## Rôle dans GCN-Core

```
CausalIR  (après gcn-middleend)
            │
     gcn-backend
     ├── pearl::why()           → Pearl niveau 1 : observation inverse
     ├── pearl::what()          → Pearl niveau 1 : observation avant
     ├── pearl::chain()         → Pearl niveau 1 : chemin causal
     ├── pearl::intervene()     → Pearl niveau 2 : do-calculus
     ├── pearl::counterfactual()→ Pearl niveau 3 : contrefactuel
     ├── query::parse() + execute() → GCN-QL
     └── export::to_json() / to_dot()
```

---

## Dépendance

```toml
[dependencies]
gcn-backend = "2.5.0"
```

---

## Raisonnement Pearl

### Niveau 1 — Observation

```rust
use gcn_backend::pearl::{why, what, chain};

// WHY : causes d'un nœud (BFS inverse)
let causes = why(&ir, "inondations");
// -> Vec<(node_label, Vec<CausalLink>)>
for (node, links) in &causes {
    for link in links {
        println!("{} -[{:?}]-> {}", link.from_label, link.relation, link.to_label);
    }
}

// WHAT : effets d'un nœud (BFS avant)
let effects = what(&ir, "pluie");

// CHAIN : chemin le plus court entre deux nœuds
if let Some(path) = chain(&ir, "pluie", "inondations") {
    for link in &path {
        println!("{} -[{:?}]-> {}", link.from_label, link.relation, link.to_label);
    }
}
```

### Niveau 2 — Intervention (do-calculus)

```rust
use gcn_backend::pearl::intervene;

// DO(X) : coupe les causes naturelles de X, propage en avant
if let Some((label, result)) = intervene(&ir, "pluie") {
    println!("Intervention sur '{}'", label);
    println!("{} arêtes entrantes coupées", result.severed.len());
    for effect in &result.effects {
        println!("Effet : {}", effect.to_label);
    }
}
```

### Niveau 3 — Contrefactuels

```rust
use gcn_backend::pearl::counterfactual;

// "Que se serait-il passé si X n'avait pas eu lieu ?"
if let Some((label, result)) = counterfactual(&ir, "pluie") {
    println!("Effets réels de '{}' : {}", label, result.actual_effects.len());
    println!("Effets uniquement dus à '{}' : {:?}", label, result.unique_effects);
    // unique_effects = effets sans chemin alternatif depuis les vraies racines
}
```

### Structures de données Pearl

```rust
pub struct CausalLink {
    pub from_label: String,
    pub to_label: String,
    pub from_id: NodeId,
    pub to_id: NodeId,
    pub relation: RelationType,
    pub confidence: f32,
    pub negated: bool,
    // Éq.12 — ternaire
    pub joint_group_id: Option<String>,  // co-nécessité JointCause/JointPrevent
    pub third_node: Option<NodeId>,      // médiateur ou condition
}

pub struct SpofScore {
    pub label: String,
    pub score: usize,
    pub is_super_spof: bool,  // nœud source d'un groupe JointCause
}

pub struct InterventionResult {
    pub severed: Vec<CausalLink>,
    pub effects: Vec<CausalLink>,
}

pub struct CounterfactualResult {
    pub actual_effects: Vec<CausalLink>,
    pub unique_effects: Vec<String>,  // gère JointCause (co-nécessité) et MediatedCause
}
```

`CausalLink.relation` est sérialisé en snake_case via serde : `"cause"`, `"joint_cause"`, etc.
(Fix C4 audit — plus le format Debug `"Cause"`.)

### Éq.12 — Comportement ternaire Pearl

- **WHY/CHAIN** : `CausalLink.joint_group_id` propagé pour identifier les co-sources
- **COUNTERFACTUAL** : `JointCause([A,A₂]→C)` — si A supprimé, C disparaît même si A₂ reste
- **SPOF** : `is_super_spof=true` si nœud dans au moins 1 groupe `joint_group_id`

---

## GCN-QL — Langage de Requête Causal

GCN-QL est un mini-langage de requête causal. Les requêtes sont insensibles à la casse et acceptent un `?` terminal optionnel.

```rust
use gcn_backend::query::{Query, execute};

// Parser une requête GCN-QL
let q = Query::parse("WHY inondations?")?;
let result = execute(&q, &ir)?;

// Sérialisation JSON du résultat
println!("{}", serde_json::to_string_pretty(&result)?);
```

### Syntaxe GCN-QL (21+ requêtes)

| Requête | Niveau Pearl | Description |
|---------|-------------|-------------|
| `WHY <label>` | 1 | Causes directes et indirectes (joint_group_id propagé) |
| `WHAT <label>` | 1 | Effets directs et indirects |
| `CHAIN <a> -> <b>` | 1 | Chemin causal le plus court |
| `CHAIN_T <a> -> <b>` | 1 | Chemin causal avec info temporelle |
| `BEFORE <a>, <b>` | 1 | A précède-t-il B ? |
| `DELAY <a> -> <b>` | 1 | Délai estimé A→B |
| `EXPLAIN <label>` | 1 | Raisonnement abductif — causes candidates |
| `CYCLES` | — | Cycles de rétroaction |
| `GAPS` | — | Lacunes temporelles causales |
| `SPOF` | — | Points de défaillance unique (+ SUPER-SPOF JointCause) |
| `CENTRALITY <label>` | — | Centralité causale d'un nœud |
| `DENSITY` | — | Densité du graphe |
| `COVERAGE <label>` | — | Couverture causale d'un concept |
| `RELIABILITY <label>` | — | Fiabilité des réponses |
| `ANALOGY <a> -> <b>` | — | Patterns causaux similaires |
| `NORM_DIFF <a>, <b>` | — | Écart normatif A→B |
| `DO <label>` | 2 | Intervention do-calculus |
| `COUNTERFACTUAL <label>` | 3 | Contrefactuel (JointCause, MediatedCause gérés) |
| `ZOOM_IN <label>` | — | Sous-graphe enfants (hiérarchie multi-échelle) |
| `ZOOM_OUT <label>` | — | Nœud parent |
| `AGGREGATE <label>` | — | Vue agrégée nœud + enfants |

### `Query` (enum)

```rust
pub enum Query {
    Why(String), What(String),
    Chain(String, String),
    Cycles, Gaps,
    Intervene(String),
    Counterfactual(String),
}

impl Query {
    pub fn parse(input: &str) -> Result<Self, BackendError>
}
```

### `QueryResult` (sérialisable)

```rust
// Serde tag = "type", rename_all = "snake_case"
pub enum QueryResult {
    Causes        { target: String, links: Vec<LinkDto> },
    Effects       { source: String, links: Vec<LinkDto> },
    Path          { from: String, to: String, found: bool, links: Vec<LinkDto> },
    CycleList     { count: usize, cycles: Vec<CycleDto> },
    GapList       { gap_edges: Vec<GapDto>, unresolved_nodes: usize },
    Intervention  { target: String, severed_count: usize,
                    severed: Vec<LinkDto>, effects: Vec<LinkDto> },
    CounterfactualDiff { target: String, actual_effects: Vec<LinkDto>,
                         unique_effects: Vec<String> },
}

pub struct LinkDto { pub from: String, pub to: String,
                     pub relation: String, pub confidence: f32, pub negated: bool }
pub struct CycleDto { pub id: u32, pub kind: String, pub nodes: Vec<String> }
pub struct GapDto   { pub from: String, pub to: String, pub relation: String }
```

---

## Export

```rust
use gcn_backend::export::{to_json, to_dot};

// JSON (serde_json pretty-printed)
let json = to_json(&ir)?;
std::fs::write("graph.json", &json)?;

// DOT (Graphviz)
let dot = to_dot(&ir)?;
std::fs::write("graph.dot", &dot)?;
// Rendu : dot -Tpng graph.dot -o graph.png
```

Le DOT généré :
- Nœuds : `diamond` pour `Condition`, `ellipse` pour `EtatSystemique`, `box` pour les autres
- Arêtes colorées par `RelationType`, tiretées si `negated`

---

## Erreurs

```rust
pub enum BackendError {
    NodeNotFound(String),
    QueryParseError(String),
    Serialization(serde_json::Error),
    NoPath(String, String),
}
```

---

## Licence

Apache-2.0 — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
