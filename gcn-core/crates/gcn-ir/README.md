# gcn-ir — Représentation Intermédiaire Causale

Version: 2.5.0

[![Crates.io](https://img.shields.io/crates/v/gcn-ir)](https://crates.io/crates/gcn-ir)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

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
gcn-ir = "2.5.0"
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

### `NodeType` — 8 types D5 (ETUDE §8)

```rust
pub enum NodeType {
    #[serde(alias = "action", alias = "transition")]
    Processus,      // processus/action continu — absorbe action+transition (D5)
    #[serde(rename = "etat_local", alias = "etat")]
    EtatLocal,      // état local stable d'une entité
    #[serde(rename = "etat_global", alias = "etat_systemique")]
    EtatGlobal,     // propriété systémique d'un ensemble
    Entite,         // entité non-causale (acteur, objet)
    Condition,      // condition nécessaire ou suffisante
    Concept,        // concept abstrait (D5 nouveau)
    Evenement,      // événement ponctuel (D5 nouveau)
    Contrainte,     // contrainte réglementaire ou physique
}
```

Aliases serde pour backward compat : `"etat"→EtatLocal`, `"etat_systemique"→EtatGlobal`,
`"action"→Processus`, `"transition"→Processus`.

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

### `RelationType` — 19 relations (v3.0 ETUDE §9)

```rust
pub enum RelationType {
    // 11 relations directes
    Cause, Enable, Prevent, Condition, Concession,
    Sequence, Motivation, Filter, Opposition,
    DataDependency, ControlDependency,
    // 8 relations ternaires v3.0
    Analogy, Counterfactual,
    ConditionalCause, MediatedCause, JointCause,
    ConditionalPrevent, MediatedPrevent, JointPrevent,
}

relation.signals_causal_gap() -> bool  // true pour Concession et Opposition
relation.is_joint() -> bool            // true pour JointCause et JointPrevent
```

Sérialisé en snake_case via serde : `"cause"`, `"joint_cause"`, `"conditional_prevent"`, etc.

### `TernaryThird` — tiers d'une relation ternaire

```rust
pub enum TernaryRole { Condition, Mediator }

pub struct TernaryThird {
    pub role: TernaryRole,
    pub node: u64,                      // NodeId.0 du nœud tiers
    pub polarity: Option<String>,       // "negative" si Règle 2 §9.4
}
```

### `CausalEdge`

```rust
pub struct CausalEdge {
    pub relation: RelationType,
    pub confidence: f32,                // [0.0, 1.0]
    pub temporal_gap: Option<TemporalGap>,
    pub explicit: bool,                 // marqueur lexical présent
    pub negated: bool,
    pub marker_token: Option<u32>,
    pub in_cycle: Option<CycleId>,
    pub provenance: Option<Provenance>, // traçabilité source
    pub derivation: Option<Derivation>,
    pub joint_group_id: Option<String>, // sha256[:16], identique sur 2 arêtes JointCause
    pub third: Option<TernaryThird>,    // tiers pour relations ternaires
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

Apache-2.0
