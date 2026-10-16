# gcn-backend — Raisonnement Pearl, GCN-QL et Export

Version: 4.0.0

[![Crates.io](https://img.shields.io/crates/v/gcn-backend)](https://crates.io/crates/gcn-backend)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Backend de raisonnement du moteur **GCN-Core**. Implémente les requêtes causales sur graphe
(association, chirurgie de graphe, contrefactuel sur assertions), le langage de requête GCN-QL,
et l'export du graphe causal en JSON ou DOT (Graphviz).

> **Portée (borné 2026-10-11).** Le graphe stocke des **assertions sourcées, pas des faits**.
> `intervene()` (coupe les entrantes, propage) et `counterfactual()` (monde-sans-X) sont des
> calculs sur assertions citées — **sans contrôle de confondeurs ni modèle structurel**,
> donc pas du do-calculus identifié au sens statistique. Voir `docs/refactory/ETUDE_SA.md §2.1`.

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
gcn-backend = "4.0.0"
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

### SPOF — Points de défaillance unique

```rust
use gcn_backend::pearl::spof_all;

let (total_pairs, scores) = spof_all(&ir).expect("graphe < 500 nœuds");
for score in &scores {
    if score.score > 0 {
        println!("{} coupe {} paires causales (super_spof={})",
                 score.label, score.score, score.is_super_spof);
    }
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

### Contexte partagé Ph10-M4 — un seul calcul de graphe par requête

```rust
let ctx = pearl::GraphCtx::new(&ir);   // petgraph + index, UNE fois
let r = execute(&query, &ir);          // execute() construit ce ctx en interne
```

`execute()` (et `chain_strict`) construisent un `GraphCtx` à l'arrivée de la
requête et le passent à toutes les fonctions d'analyse (`*_ctx` : WHY/WHAT,
CHAIN, DO, contrefactuel, EXPLAIN, ASK, SPOF, filtres, caveats, DTO). Les
variantes `ir` restent en wrappers fins (appelants directs + tests inchangés).
Mesuré (release, 3000 nœuds, 5 requêtes) : ~1140 ms avant → ~870 ms après
(~1,3× ; sous la cible 3-4× du plan car Dijkstra/BFS dominent, pas les builds —
le vrai goulot reste M1 côté Python).

Convention S2 : les cœurs `*_ctx` sont `pub(crate)` (usage interne `execute()`
uniquement) ; l'API publique reste les variantes `ir` + `execute()` +
`chain_strict` + `filter_links`/`caveats_for`. `build_maps_pub` supprimé
(0 usage) ; `proptest` retiré des dev-deps (inutilisé).

### Éq.12 — Comportement ternaire Pearl

- **WHY/CHAIN** : `CausalLink.joint_group_id` propagé pour identifier les co-sources
  (groupe valide ssi ≥ 2 arêtes — orphelin = joint invalide, traité en arête simple)
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
| `WHY <label>` | 1 | Causes directes et indirectes (joint_group_id propagé ; coupures → `blocked_by` + `motif` ; exacts d'abord, flous ≤ `AMBIGUITY_LIMIT` (défaut 5), excès → erreur `Ambiguous` ; `TOPK=N` = N meilleurs + `total` ; filtres `WHERE`/`SCOPE=`/etc.) |
| `WHAT <label>` | 1 | Effets directs et indirects (même politique exact-puis-flou ; `TOPK=N` ; filtres) |
| `CHAIN <a> -> <b>` | 1 | Chemin confiance-max × poids origine (Dijkstra −ln(conf), jamais via coupure ; `TOPK=N` défaut 3, `path_confidence` + `ranked_paths` ; `found:false` + `blocked_by` si coupure, vide si aucun chemin ; filtres) |
| `CHAIN_T <a> -> <b>` | 1 | Chemin causal avec info temporelle (oscillation sans phase → rejeté) |
| `BEFORE <a>, <b>` | 1 | A précède-t-il B ? |
| `DELAY <a> -> <b>` | 1 | Délai estimé A→B |
| `EXPLAIN <label>` | 1 | Raisonnement abductif — causes candidates (`CONTRAST <autre>` = distinctives vs partagées ; scindé par `agent_type`) |
| `CYCLES` | — | Cycles de rétroaction (typés : damped/amplifying/oscillating) |
| `GAPS` | — | Lacunes temporelles causales |
| `SPOF` | — | Points de défaillance unique (+ SUPER-SPOF JointCause) |
| `CENTRALITY <label>` | — | Centralité causale d'un nœud |
| `DENSITY` | — | Densité du graphe |
| `COVERAGE <label>` | — | Couverture causale d'un concept |
| `RELIABILITY <label>` | — | Fiabilité des réponses |
| `ANALOGY <a> -> <b>` | — | Patterns causaux similaires (`WEIGHTS=a,b,c,d` défaut 0.40/0.30/0.20/0.10, `MIN_SCORE=x` défaut 0.3, `HOPS=N` défaut 1 = mono-saut, ≥ 2 = contexte Jaccard 50/50) |
| `ASK <label>` | — | Fiche nœud (type + causes/effets directs 1 saut, sans propagation) |
| `WHY-NOT <a> -> <b>` | — | Pourquoi aucun chemin : `coupure_negation` (bloqueurs = `blocked_by`) ou `aucun_lien` (proches), `chemin_existe` sinon |
| `NORM_DIFF <a>, <b>` | — | Écart normatif A→B |
| `DO <label>` | 2 | Intervention do-calculus (entrantes niées incluses dans `severed`, effets jamais via coupure ; cible irréversible → warning `irreversible` FIGÉ, effets calculés ; filtre `AGENT=`) |
| `COUNTERFACTUAL <label>` | 3 | Contrefactuel (JointCause, MediatedCause gérés ; coupures non traversées) |
| `DOUBTS` | — | Liste brute des non-résolus (nœuds, champs, candidats) |
| `PROVENANCE <label>` | — | Fiche des arêtes incidentes (méthode, modèle, span texte/code, dates) |
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
    Causes        { target: String, links: Vec<LinkDto>, blocked_by: Vec<LinkDto>,
                    match_scores: Vec<(String, u8)>, fuzzy: bool, total: usize, .. },
    Effects       { source: String, links: Vec<LinkDto>, blocked_by: Vec<LinkDto>,
                    match_scores: Vec<(String, u8)>, fuzzy: bool, total: usize, .. },
    Path          { from: String, to: String, found: bool,
                    links: Vec<LinkDto>, blocked_by: Vec<LinkDto>,
                    path_confidence: f32, ranked_paths: Vec<RankedPathDto>,
                    caveats: Vec<Caveat> },
    CycleList     { count: usize, cycles: Vec<CycleDto> },
    GapList       { gap_edges: Vec<GapDto>, unresolved_nodes: usize },
    Intervention  { target: String, severed_count: usize,
                    severed: Vec<LinkDto>, effects: Vec<LinkDto>,
                    blocked_by: Vec<LinkDto>, caveats: Vec<Caveat>,
                    warnings: Vec<String> },
    CounterfactualDiff { target: String, actual_effects: Vec<LinkDto>,
                         unique_effects: Vec<String>, blocked_by: Vec<LinkDto>,
                         caveats: Vec<Caveat> },
    Doubts        { count: usize, doubts: Vec<Caveat> },
    ProvenanceReport { label: String, entries: Vec<ProvenanceEntryDto> },
}

// Ph7 — gisement CIR :
// [7a] concession/opposition = BLOQUEUSES (même `blocked_by`, motif `causal_gap`) ;
//   feedback-negative → `damped`, positive → `amplifying`, oscillation → CHAIN_T
//   exige la phase ; [7b] ranking × poids origine (1.0/0.7/0.3 FIGÉS, D-ORIG :
//   9a côté session) + `caveats` systématiques + `DOUBTS` ; [7f] provenance sur
//   les liens + filtres (`WHERE scope=`, `AGENT=`, `ONLY_EXPLICIT`, `METHOD=`,
//   `SINCE=`) + `PROVENANCE` ; [7d] `irreversible` FIGÉ + agents + `by_agent`;
//   [7e] scope null = BLOQUEUR (`null_scope`), témoins non généralisés (7c enterré : NO-GO Ph5).

// Ph1 [D1] — sémantique négation : arête niée = propagation COUPÉE (pas
// d'inversion). `blocked_by` liste les niées qui auraient porté une
// propagation (triées, déterministes). `prevent` reste une relation positive.
// `chain_strict` (CLI `--strict`) exige en plus le `third_node` des liens
// ternaires sur le chemin, sinon `NoPath`.
//
// Ph2 [D2] — résolution exact-puis-flou : exact (score 0) toujours répondu
// (même multiples) ; flou (préfixe/contient, puis similarité injectable via
// `pearl::LabelSim`, backend sans ML) seulement si 0 exact ;
// `WHY x AMBIGUITY_LIMIT=N` (défaut 5, flous seuls comptés) ; excès →
// `Err(Ambiguous { candidates })`, convertie en partiel+question par Ph8.
// `match_scores` = (label, score) des retenus, `fuzzy` = warning flou.
//
// Ph3 [D3] — ranking : CHAIN = Dijkstra confiance-max (coût −ln(conf),
// conf ≤ 0 → epsilon), k-chemins (Yen, `TOPK=N` défaut 3) ; `links` = rang 1,
// `path_confidence` = produit du rang 1, `ranked_paths` ssi TOPK > 1 ;
// WHY/WHAT triés par force (produit depuis la cible) ; EXPLAIN déjà trié par
// score ; ANALOGY poids/seuil paramétrables (défauts 0.40/0.30/0.20/0.10 et 0.3).
// Shift assumé : `chain()` (et NormDiff/Before/Delay/ChainT/strict) utilise le
// chemin confiance-max, plus le plus-court BFS.

pub struct LinkDto { pub from: String, pub to: String,
                     pub relation: String, pub confidence: f32, pub negated: bool,
                     pub derived_from: Option<Vec<(String, String)>> /* D3 Ph6b */,
                     pub motif: Option<String> /* negated|causal_gap|null_scope */,
                     pub cycle_effect: Option<String> /* damped|amplifying */,
                     pub origin: Option<String> /* explicit|inferred|hypothetical */,
                     pub explicit: bool, pub extraction_method: Option<String>,
                     pub model_version: Option<String>, pub span: Option<String>,
                     pub kind: Option<String> /* text|code|synthetic */,
                     pub scope: Option<String>, pub agent: Option<String>,
                     pub patient: Option<String>, pub agent_type: Option<String>,
                     pub witness_only: bool }
pub struct CycleDto { pub id: u32, pub kind: String, pub nodes: Vec<String>,
                      pub effect: Option<String> /* damped|amplifying|oscillating */ }
pub struct GapDto   { pub from: String, pub to: String, pub relation: String }
// AgentType : human | collective | institutional | natural (forces/phénomènes)
//             | system (le système lui-même — phase 9). Champ vide partout
//             aujourd'hui (émetteur Python : None) : l'attribution viendra de
//             l'annotation gold-3000, sinon filtre et groupes restent décoratifs.
```

---

## Always-answer — `respond()` (Ph8)

```rust
use gcn_backend::{Status, UserResponse, respond};

let resp: UserResponse = respond("WHY ventes?", &ir);
// UserResponse { status, phrase, lang, result: Option<QueryResult>,
//                caveats, suggestions } — sérialisable (pont M1).
```

Jamais d'Err nue à la frontière interactive : `execute()` reste inchangé
(Err internes), `respond()` enveloppe tout — parse, introuvable, ambigu,
sans-chemin (bloqueurs niés = WHY-NOT, sinon proches), trop-gros (> 500
nœuds : partiel + top connectés), graphe vide, incident.
Règle de statut : Degrade = parse / introuvable / vide / incident ; Partiel =
ambigu / sans-chemin / flou / coupures niées / trop-gros / ordre temporel
rejeté ; Ok sinon. `gcn query` imprime l'enveloppe (exit 0 sauf Degrade → 1,
contrat batch préservé ; `--strict` inchangé).

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
    Ambiguous { candidates: Vec<String> },
}
```

---

## Licence

Apache-2.0 — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
