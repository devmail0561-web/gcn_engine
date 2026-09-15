# gcn-ir — Représentation Intermédiaire Causale

[![crates.io](https://img.shields.io/crates/v/gcn-ir)](https://crates.io/crates/gcn-ir)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Crate fondatrice du moteur **GCN-Core**. Définit le contrat central `CausalIR` : le graphe causal typé qui traverse tout le pipeline (parsers → middle-end → backend → verbalizer).

---

## Rôle dans GCN-Core

```
FrenchParser / EnglishParser / parse_python()
            │
            ▼
        CausalIR   ◄── gcn-ir définit ce type
            │
    gcn-middleend → gcn-backend → gcn-verbalizer
```

Tous les frontends produisent un `CausalIR`. Tous les backends le consomment. `gcn-ir` ne dépend d'aucune autre crate GCN.

---

## Dépendance

```toml
[dependencies]
gcn-ir = "1.0"
```

---

## Types principaux

### `CausalIR`

Structure racine — le graphe causal complet d'un texte ou d'un programme.

```rust
use gcn_ir::{CausalIR, CausalNode, CausalEdge, NodeId};

pub struct CausalIR {
    pub source_lang: SourceLanguage,
    pub source_text: String,
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(NodeId, NodeId, CausalEdge)>,
    pub cycles: Vec<CausalCycle>,
    pub unresolved: Vec<Ambiguity>,
    pub metadata: IrMetadata,
}
```

**Méthodes :**

```rust
ir.node_count() -> usize     // nombre de nœuds
ir.edge_count() -> usize     // nombre d'arêtes
ir.has_cycles() -> bool      // présence de boucles de rétroaction
ir.has_gaps()   -> bool      // présence de lacunes temporelles causales
```

---

## Types de nœuds

### `NodeType` — 7 types causaux

```rust
pub enum NodeType {
    Etat,           // état stable d'une entité
    Action,         // action délibérée d'un agent
    Transition,     // passage d'un état à un autre
    Processus,      // processus continu ou naturel
    Condition,      // condition nécessaire ou suffisante
    Entite,         // entité non-causale (acteur, objet)
    EtatSystemique, // propriété systémique d'un ensemble
}
```

Méthode associée :
```rust
node_type.causal_direction() -> CausalDirection
// Forward | Backward | Both | Accumulative | Suspended | None
```

### `CausalNode`

```rust
pub struct CausalNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub label: String,
    pub source_span: SourceSpan,   // TokenSpan | CodeSpan | Synthetic
    pub scope: Scope,
    pub modifiers: SmallVec<[Modifier; 4]>,
    pub temporal_ref: TemporalRef,
    pub temporal_index: Option<i32>,
    pub origin: NodeOrigin,        // Explicit | Inferred | Hypothetical
    pub attributes: NodeAttributes,
}
```

`NodeAttributes` : `entity`, `quality`, `agent`, `patient`, `agent_type`, `reversible`

---

## Types d'arêtes

### `RelationType` — 11 relations causales

```rust
pub enum RelationType {
    Cause, Enable, Prevent, Condition, Concession,
    Sequence, Motivation, Filter, Opposition,
    DataDependency, ControlDependency,
}

// Détecte les relations signalant une lacune causale cachée
relation.signals_causal_gap() -> bool  // true pour Concession et Opposition
```

### `CausalEdge`

```rust
pub struct CausalEdge {
    pub relation: RelationType,
    pub confidence: f32,             // [0.0, 1.0]
    pub temporal_gap: Option<TemporalGap>,
    pub explicit: bool,              // marqueur lexical présent
    pub negated: bool,
    pub marker_token: Option<u32>,
    pub in_cycle: Option<CycleId>,
}
```

---

## Modificateurs

12 variants pour annoter les nœuds :

```rust
pub enum Modifier {
    Intensity    { value: f32, source_form: String },
    Negation     { total: bool },
    Probability  { value: f32, source_form: String },
    Temporality  { anchor: TemporalAnchor, source_form: String },
    Frequency    { kind: FrequencyKind, source_form: String },
    Manner       { description: String, source_form: String },
    Location     { scope: LocationScope, source_form: String },
    IntrinsicProperty { property: String, source_form: String },
    TemporaryState    { state: String, source_form: String },
    Relational        { relation: String, source_form: String },
    Quantitative      { impact: f32, source_form: String },
    TemporalQuality   { maturity: Maturity, source_form: String },
}
```

---

## Portée, temporalité, erreurs

```rust
pub enum Scope { Universal, Existential, Partial, Null, Specific, Unknown }

pub enum TemporalRef { Absolute(i64), Relative { base: i32, offset: i32 },
                       Epsilon, Range { start: i32, end: i32 }, Unresolved }

pub struct TemporalGap { pub min: Option<i32>, pub max: Option<i32>, pub nature: GapNature }
pub enum GapNature { Immediate, Deferred, Continuous }

pub enum GcnError {
    Parse(String), NodeNotFound(u32), InvalidTaxonomy(String),
    UnsupportedLanguage(String), Serialization(serde_json::Error),
    CausalConstraintViolation(String), InvalidCycle(String), Io(std::io::Error),
}
pub type GcnResult<T> = Result<T, GcnError>;
```

---

## Langues supportées

```rust
pub enum SourceLanguage {
    Natural     { lang: NaturalLanguage },     // French | Wolof | Arabic | English
    Programming { lang: ProgrammingLanguage }, // Python | Rust | JavaScript
}
```

---

## Sérialisation

`CausalIR` et tous ses types dérivent `Serialize` / `Deserialize` (serde). Export JSON :

```rust
let json = serde_json::to_string_pretty(&ir)?;
let ir2: CausalIR = serde_json::from_str(&json)?;
```

---

## Licence

MIT
