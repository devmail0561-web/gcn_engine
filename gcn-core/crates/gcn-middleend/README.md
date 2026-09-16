# gcn-middleend — Construction, Propagation et Validation du Graphe Causal

Version: 2.0.0

[![Crates.io](https://img.shields.io/crates/v/gcn-middleend)](https://crates.io/crates/gcn-middleend)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Middle-end du moteur **GCN-Core**. Enrichit un `CausalIR` produit par un frontend : détecte les cycles de rétroaction (algorithme de Tarjan), propage les lacunes temporelles causales, et valide la cohérence structurelle.

---

## Rôle dans GCN-Core

```
CausalIR  (produit par FrenchParser / EnglishParser / parse_python...)
            │
     gcn-middleend
     ├── graph::build()         → CausalGraph (petgraph DiGraph)
     ├── cycle::detect_and_classify()  → Vec<CausalCycle>
     ├── propagate::run()       → TemporalGap sur Concession/Opposition
     └── validate::run()        → Diagnostics (self-loop, orphelin, ...)
            │
            ▼
     MiddleendResult { ir: CausalIR enrichi, diagnostics }
            │
            ▼
     gcn-backend (raisonnement Pearl, GCN-QL)
```

---

## Dépendance

```toml
[dependencies]
gcn-middleend = "2.0"
```

---

## Utilisation

```rust
use gcn_middleend::process;

let result = process(ir)?;

// CausalIR enrichi : cycles annotés, temporal_gap propagés
let enriched_ir = result.ir;

// Diagnostics structurels
for diag in &result.diagnostics {
    match diag.severity {
        DiagnosticSeverity::Error   => eprintln!("ERREUR : {:?}", diag.kind),
        DiagnosticSeverity::Warning => eprintln!("AVERT.  : {:?}", diag.kind),
    }
}
```

---

## API publique

### `process`

```rust
pub fn process(ir: CausalIR) -> Result<MiddleendResult, MiddleendError>
```

Pipeline complet en 4 étapes :
1. `graph::build(&ir)` — construction du `CausalGraph`
2. `cycle::detect_and_classify(&ir, &g)` — Tarjan SCC, classification des cycles
3. `propagate::run(&mut ir, &g, &mut diagnostics)` — propagation des gaps temporels
4. `validate::run(&ir, &g, &mut diagnostics)` — validations structurelles

### `MiddleendResult`

```rust
pub struct MiddleendResult {
    pub ir: CausalIR,
    pub diagnostics: Vec<Diagnostic>,
}
```

### `MiddleendError`

```rust
pub enum MiddleendError {
    GraphBuildError(String),
}
```

---

## Diagnostics

```rust
pub struct Diagnostic {
    pub node_id: Option<NodeId>,
    pub severity: DiagnosticSeverity,  // Error | Warning
    pub kind: DiagnosticKind,
}

pub enum DiagnosticKind {
    SelfLoop { node: NodeId },
    // Severity: Error — arête d'un nœud vers lui-même

    OrphanedNode { node: NodeId },
    // Severity: Warning — nœud sans aucune arête (graphe multi-nœuds)

    LowConfidenceEdge { src: NodeId, dst: NodeId, confidence: f32 },
    // Severity: Warning — confidence < 0.3

    DanglingCondition { node: NodeId },
    // Severity: Warning — nœud Condition sans arête sortante

    TemporalOrderViolation { src: NodeId, dst: NodeId },
    // Severity: Warning — arête Sequence avec indices temporels incohérents
}
```

---

## Détection de cycles

```rust
use gcn_middleend::{graph::build, cycle::detect_and_classify};

let g = build(&ir);
let (cycles, edge_to_cycle) = detect_and_classify(&ir, &g);

for cycle in &cycles {
    println!("Cycle {:?} : {:?} — nœuds {:?}", cycle.id, cycle.cycle_type, cycle.path);
}
// CycleType : FeedbackPositive | FeedbackNegative | Oscillation
```

Les cycles sont annotés dans `ir.cycles` et les arêtes participantes reçoivent `in_cycle: Some(CycleId)`.

---

## Propagation temporelle

`propagate::run` effectue deux passes :

1. **Gaps causaux** — toute arête `Concession` ou `Opposition` sans `temporal_gap` reçoit `TemporalGap { nature: Deferred }`
2. **Ordre séquentiel** — toute arête `Sequence` dont `temporal_index(src) >= temporal_index(dst)` génère un `TemporalOrderViolation`

---

## Validation structurelle

`validate::run` vérifie quatre propriétés :

| Vérification | Sévérité | Condition |
|---|---|---|
| `SelfLoop` | Error | arête `(n, n, _)` |
| `OrphanedNode` | Warning | nœud sans aucune arête, graphe ≥ 2 nœuds |
| `LowConfidenceEdge` | Warning | `confidence < 0.3` |
| `DanglingCondition` | Warning | `NodeType::Condition` sans arête sortante |

---

## `CausalGraph` (accès direct)

```rust
use gcn_middleend::graph::{build, CausalGraph};

let g: CausalGraph = build(&ir);

// NodeIndex petgraph depuis un NodeId
if let Some(idx) = g.node_index(NodeId(0)) {
    println!("NodeIndex petgraph : {:?}", idx);
}

// Accès au DiGraph petgraph sous-jacent
let _digraph = &g.g;
```

---

## Licence

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
