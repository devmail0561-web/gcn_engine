# gcn-backend — Raisonnement Pearl, GCN-QL et Export

Version: 2.0.0

[![Crates.io](https://img.shields.io/crates/v/gcn-backend)](https://crates.io/crates/gcn-backend)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
gcn-backend = "2.0"
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
    pub to_id: NodeId,
    pub relation: RelationType,
    pub confidence: f32,
    pub negated: bool,
}

pub struct InterventionResult {
    pub severed: Vec<CausalLink>,  // arêtes coupées par l'intervention
    pub effects: Vec<CausalLink>,  // effets propagés en avant
}

pub struct CounterfactualResult {
    pub actual_effects: Vec<CausalLink>, // effets réels de X
    pub unique_effects: Vec<String>,     // effets sans chemin sans X
}
```

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

### Syntaxe GCN-QL

| Requête | Niveau Pearl | Description |
|---------|-------------|-------------|
| `WHY <label>` | 1 | Causes directes et indirectes d'un nœud |
| `WHAT <label>` | 1 | Effets directs et indirects d'un nœud |
| `CHAIN <a> -> <b>` | 1 | Chemin causal le plus court de a vers b |
| `CYCLES` | — | Liste tous les cycles de rétroaction détectés |
| `GAPS` | — | Liste les arêtes portant une lacune temporelle causale |
| `DO <label>` | 2 | Intervention do-calculus sur un nœud |
| `COUNTERFACTUAL <label>` | 3 | Analyse contrefactuelle : que sans X ? |

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

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
