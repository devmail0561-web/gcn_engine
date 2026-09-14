# SAD — Software Architecture Document : GCN-Core

**Grammaire Causale Naturelle : Implémentation de Référence**

Basé sur le papier de recherche de Michel Tendeng, Université Numérique Cheikh Hamidou Kane (UN-CHK), Septembre 2026

## Contexte

Le projet GCN (Grammaire Causale Naturelle) de Michel Tendeng propose un cadre formel pour extraire la causalité depuis la structure grammaticale du langage naturel. L'architecture retenue (CGNP → GCN-Core) suit un modèle de **compilateur** : Frontend → CIR → Middle-end → Backend. L'utilisateur demande la création du SAD complet avec stack technique, choix du langage, et exemples de datasets pour langues naturelles et de programmation.

## Objectif et usages

GCN-Core est une **architecture encodeur-décodeur avec CausalIR comme représentation pivot**. Son objectif est de fournir une base pour construire des modèles qui raisonnent sur la causalité plutôt que de l'apprendre comme effet émergent.

### Objectif central

Poser la causalité comme **structure de premier plan**, explicite et vérifiable — pas comme corrélation statistique enfouie dans des poids. Le CausalIR est la représentation commune à toutes les langues et modalités : ce que le modèle a appris à encoder et décoder.

### Ce que peut produire un modèle entraîné sur GCN-Core

La sortie dépend entièrement des données d'entraînement. Selon le corpus utilisé, la même architecture peut servir à construire :

- **Un modèle de langage** — entraîné sur des textes en langue naturelle
- **Un modèle de codage** — entraîné sur du code source
- **Un modèle de traduction** — entraîné sur des paires multilingues ou cross-modal
- **Un modèle agentique** — entraîné sur des traces d'actions et leurs conséquences causales
- **Un modèle de raisonnement contrefactuel** — via Pearl niveaux 2-3

L'architecture ne présuppose pas le type de sortie. C'est l'entraînement qui détermine ce que le modèle sait faire.

### Ce que GCN apporte que les LLMs n'ont pas

| LLM | GCN-Core |
|---|---|
| Causalité implicite dans les poids | Causalité explicite dans le CausalIR |
| Ininterprétable | Graphe auditable nœud par nœud |
| Confond corrélation et causalité | Distingue `Cause`, `Concession`, `Opposition`, `Enable`, `Prevent` |
| Pas de formalisme Pearl natif | Pearl niveaux 1-2-3 intégrés |

## Philosophie de l'architecture

GCN-Core est une **architecture encodeur-décodeur**, à l'image du Transformer — pas un moteur de base de données ni une application utilisateur finale.

| Composant | Rôle | Analogie Transformer |
|---|---|---|
| Frontends + couches 1-3 | Encodeur : Texte/Code → CausalIR | Encoder stack |
| CausalIR | Représentation pivot — structurée, sérialisable, interrogeable | Hidden states / latent representation |
| gcn-verbalizer | Décodeur : CausalIR → Texte/Code | Decoder stack |
| gcn-datasets | Corpus d'entraînement pour les deux côtés simultanément | Training corpus |

Le décodeur est **partie intégrante de l'architecture** au même titre que l'encodeur. Les mêmes paires `(texte, CausalIR)` entraînent les deux côtés : l'encodeur apprend à produire le CausalIR, le décodeur apprend à en produire la surface.

Le CLI `gcn` est un outil de développement et d'inspection — pas l'interface finale. Les applications se construisent sur l'API Rust/Python en consommant le CausalIR.

## Pipeline d'annotation

Les taxonomies de `gcn-references/` guident la création des données d'entraînement. Elles ne sont **jamais lues par le moteur à l'inférence** — elles servent uniquement aux annotateurs.

```
gcn-references/taxonomies/    ← consulté par l'annotateur (pas par le modèle)
          ↓
Annotateur (humain ou outil bootstrap)
          ↓ produit
gcn-datasets/                 ← corpus annoté (texte/code + CausalIR)
          ↓ entraîne
Couches ML (layer1, layer2, layer3)   ← apprennent les patterns depuis les données
```

**Exemple :** `conjonctions.yaml` indique que "si" → `RelationType::Condition`. L'annotateur étiquette `"Si les ventes baissent, on réduit les coûts."` → CausalIR avec un Condition edge. Le modèle apprend ensuite le pattern `si + VERB(advcl)` → Condition depuis les annotations — sans jamais lire `conjonctions.yaml`.

**`gcn-frontend-fr` et `gcn-frontend-code`** sont des outils de bootstrap : ils auto-annotent du texte/code pour aider à constituer `gcn-datasets/`. Une fois le modèle entraîné sur suffisamment de données, il les remplace pour l'inférence.

## Langage retenu : Rust

| Contrainte | Justification Rust |
|---|---|
| **Sûr** | Borrow checker, enums exhaustifs, pas de null (`Option`), pas de UB |
| **Rapide** | Compilation native, zero-cost abstractions, `rayon` pour parallélisme |
| **Léger** | Binaire unique ~10MB, modèles NLP ~20MB, pas de runtime Python/JVM |
| **Type system** | Enums algébriques = isomorphes aux types causaux GCN (4 classes verbales, 7 adverbiales, etc.) |

Les couches ML (Couches 1-3 CGNP) sont implémentées dans le package Python `gcn-python/` — indépendant du workspace Rust. L'interface est JSON via stdout/subprocess.

## Stack technique

### Rust (gcn-core/)

| Crate | Version | Usage |
|---|---|---|
| `serde` | 1.0 | Sérialisation de toutes les structures |
| `yaml_serde` | 0.10.7 | Chargement taxonomies YAML |
| `serde_json` | 1.0.151 | Export JSON + interface Python↔Rust |
| `petgraph` | 0.8.3 | Graphe causal orienté avec cycles (DiGraph) |
| `thiserror` | 2.0.20 | Types d'erreurs |
| `clap` | 4.6.6 | CLI |
| `rayon` | 1.12.0 | Parallélisme data-parallel |
| `smallvec` | 1.13 | Stockage inline modificateurs |
| `tree-sitter` | 0.27.0 | Parsing code (AST) |
| `tree-sitter-python` | 0.25.0 | Grammaire Python |
| `tree-sitter-rust` | 0.24.2 | Grammaire Rust |
| `tree-sitter-javascript` | 0.25.0 | Grammaire JavaScript |
| `proptest` | 1.11.0 | Tests property-based |

### Python (gcn-python/) — couches ML de l'architecture CGNP

| Package | Usage |
|---|---|
| `spacy>=3.7` | Couche 1 — parsing UD universel (fr/en/…) |
| `numpy>=1.24` | Opérations numériques — implémentations de référence des couches 2-3 |
| `pyyaml>=6.0` | Chargement taxonomies GCN |
| `click>=8.1` | CLI `gcn-forward` |

`torch>=2.0` — disponible via `RGCNLayerPT` (couche 3 PyTorch, GPU/MPS). Optionnel : `ImportError` explicite si non installé.
`jax`, `tensorflow` — dépendances du data scientist, pas du moteur.

## Fichiers à créer

### Structure du workspace Cargo

```
projet_CNM/
├── gcn-core/                         # Moteur Rust pur — workspace Cargo
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── SAD.md
│   └── crates/
│       ├── gcn-ir/                   # Types CIR (noyau, 0 dépendances internes)
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── node.rs           # CausalNode, NodeType, AgentType
│       │       ├── edge.rs           # CausalEdge, RelationType
│       │       ├── modifier.rs       # Intensity, Negation, Probability, etc.
│       │       ├── scope.rs          # Scope enum (Universal, Existential, Partial, Null)
│       │       ├── ir.rs             # CausalIR struct (conteneur principal)
│       │       ├── temporal.rs       # TemporalRef, TemporalAnchor
│       │       ├── code.rs           # Types spécifiques code (CodeCausalType, DataFlow, etc.)
│       │       └── error.rs          # GcnError, GcnResult
│       ├── gcn-knowledge/            # Base de connaissances
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── taxonomy.rs       # Chargeur taxonomies YAML → structs typés
│       │       ├── lexicon.rs        # Lexique causal par langue
│       │       ├── inference.rs      # Règles d'inférence de types
│       │       └── loader.rs         # Validation YAML
│       ├── gcn-frontend-fr/          # Bootstrap d'annotation français — remplacé par les couches ML à l'inférence
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── tokenizer.rs
│       │       ├── tagger.rs
│       │       ├── rules.rs
│       │       ├── annotator.rs
│       │       ├── resources.rs      # Charge les taxonomies gcn-references/fr/ (annotation uniquement)
│       │       └── emitter.rs        # Tokens annotés → CausalIR
│       ├── gcn-frontend-en/          # Bootstrap d'annotation anglais — même architecture que gcn-frontend-fr
│       │   └── src/
│       │       ├── lib.rs            # EnglishParser::new(taxonomies_root) + parse(text)
│       │       ├── tokenizer.rs
│       │       ├── tagger.rs
│       │       ├── rules.rs          # Morphologie anglaise (lemmatize_verb, looks_like_verb_morphologically)
│       │       ├── annotator.rs      # Marqueurs multi-mots prioritaires sur verbes causaux
│       │       ├── resources.rs      # Charge gcn-references/en/ via TaxonomyClass.examples (champ générique)
│       │       └── emitter.rs        # Tokens annotés → CausalIR (source_lang: English)
│       ├── gcn-frontend-code/        # Bootstrap d'annotation code — remplacé par les couches ML à l'inférence
│       │   └── src/lib.rs
│       ├── gcn-middleend/            # Graphe causal, cycles, contraintes, validation
│       │   └── src/
│       │       ├── lib.rs            # process(CausalIR) → MiddleendResult
│       │       ├── graph.rs          # DiGraph depuis CausalIR
│       │       ├── cycle.rs          # Détection SCC (Tarjan) + classification
│       │       ├── propagate.rs      # TemporalGap, ordre temporel
│       │       ├── validate.rs       # Self-loop, orphelins, confidence, Condition
│       │       └── error.rs          # Diagnostic, MiddleendError
│       ├── gcn-backend/              # Raisonnement Pearl + export (Phase 4)
│       │   └── src/lib.rs
│       ├── gcn-verbalizer/           # Décodeur de l'architecture (Phase 5)
│       │   └── src/lib.rs
│       └── gcn-cli/                  # CLI
│           └── src/main.rs
│
├── gcn-references/                   # Référentiels pour annotateurs — guide la création de gcn-datasets/ (pas lu à l'inférence)
│   └── taxonomies/                   # Taxonomies par langue/modalité
│       ├── fr/                       # Taxonomies françaises (11 fichiers : verbes, conjonctions, etc.)
│       ├── en/                       # Taxonomies anglaises (10 fichiers)
│       ├── python/                   # Mappings AST Python → types causaux
│       ├── rust/                     # Mappings AST Rust → types causaux
│       └── js/                       # Mappings AST JavaScript → types causaux
│
├── gcn-datasets/                     # Données annotées — hors moteur
│   ├── schemas/
│   │   ├── gcn-nl.schema.yaml        # Schéma dataset langues naturelles
│   │   └── gcn-pl.schema.yaml        # Schéma dataset langues de programmation
│   └── examples/
│       ├── fr_causal_basic.yaml
│       ├── fr_causal_cycles.yaml
│       ├── python_basic.yaml
│       └── rust_basic.yaml
│
├── gcn-python/                       # Couches ML (Python, framework-agnostique)
└── docs/                             # SAD papier de recherche (.docx)
```

### Dépendances inter-crates

`gcn-ir` → 0 dépendances internes (noyau pur). Tout dépend de lui, il ne dépend de rien.

```
gcn-cli → gcn-frontend-fr → gcn-ir
        → gcn-frontend-en → gcn-ir
        → gcn-frontend-code → gcn-ir
        → gcn-middleend → gcn-ir + gcn-knowledge
        → gcn-backend → gcn-ir + gcn-middleend
gcn-knowledge → gcn-ir
```

## Types Rust clés (gcn-ir)

### NodeType — les 4 classes verbales + types structurels

```rust
pub enum NodeType {
    Etat,           // regarde en arrière — "Qu'est-ce qui a produit cet état?"
    Action,         // regarde en avant — "Qu'est-ce que ça va produire?"
    Transition,     // bidirectionnel — charnière entre passé et futur
    Processus,      // accumulation temporelle — cause continue
    Condition,      // suspend la chaîne jusqu'à déclenchement
    Entite,         // agent/patient — pas opérateur causal, peuple la chaîne
    EtatSystemique, // contexte filtrant (crise, stabilité) — affecte toutes les chaînes
}
```

### RelationType — les 6 classes de conjonctions + types code

```rust
pub enum RelationType {
    Cause,           // "parce que", "car"
    Enable,          // "grâce à" — facilite sans forcer
    Prevent,         // "empêcher" — bloque la chaîne
    Condition,       // "si" — suspend jusqu'à évaluation
    Concession,      // "bien que" — signale lacune causale
    Sequence,        // "quand", "après" — temporel
    Motivation,      // "pour", "afin de" — causalité intentionnelle inversée
    Filter,          // état systémique → filtre global
    Opposition,      // "mais", "or" — rupture causale
    DataDependency,  // code: flux de données
    ControlDependency, // code: flux de contrôle
}
```

### CausalNode, CausalEdge, CausalIR

```rust
pub struct CausalNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub label: String,
    pub source_span: SourceSpan,
    pub scope: Scope,                    // ∀, ∃, ¬∃, partiel
    pub modifiers: SmallVec<[Modifier; 4]>,
    pub temporal_index: Option<i32>,
    pub attributes: NodeAttributes,      // entity, quality, agent, patient, agent_type
}

pub struct CausalEdge {
    pub relation: RelationType,
    pub confidence: f32,
    pub temporal_gap: Option<i32>,
    pub explicit: bool,                  // marqueur causal explicite dans le texte?
    pub marker_token: Option<u32>,
    pub in_cycle: Option<CycleId>,
}

pub struct CausalIR {
    pub source_lang: SourceLanguage,     // Natural{fr} | Programming{python}
    pub source_text: String,
    pub graph: DiGraph<CausalNode, CausalEdge>,  // petgraph
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(NodeId, NodeId, CausalEdge)>,
    pub cycles: Vec<CausalCycle>,        // boucles de rétroaction
    pub metadata: IrMetadata,
}
```

### Modifier — les 7 classes adverbiales + 5 adjectivales

```rust
pub enum Modifier {
    Intensity { value: f32, source_form: String },
    Negation { total: bool },
    Probability { value: f32, source_form: String },
    Temporality { anchor: TemporalAnchor, source_form: String },
    Frequency { kind: FrequencyKind, source_form: String },
    Manner { description: String, source_form: String },
    Location { scope: LocationScope, source_form: String },
    IntrinsicProperty { property: String, source_form: String },
    TemporaryState { state: String, source_form: String },
    Relational { relation: String, source_form: String },
    Quantitative { impact: f32, source_form: String },
    TemporalQuality { maturity: Maturity, source_form: String },
}
```

### CodeCausalType — mapping AST → types causaux

```rust
pub enum CodeCausalType {
    VariableState,    // x → État d'une entité
    Assignment,       // x = expr → Action (transition d'état)
    FunctionCall,     // f(args) → Action avec agent et patient
    Conditional,      // if/match → Condition causale
    Loop,             // for/while → Processus (cause continue)
    ReturnValue,      // return → Conséquence
    Exception,        // try/except → Concession/rupture
    ImportDependency, // import → Relation de dépendance
    TypeDefinition,   // class/struct → Agent structurel
    Mutation,         // .push()/.pop() → Transition d'état
    Assertion,        // assert → Contrainte causale
}
```

## Schéma de dataset : GCN-NL (Langues Naturelles)

3 niveaux : token → phrase (CIR) → document (graphe complet)

### Exemple français

```yaml
document:
  id: "doc-fr-001"
  lang: fr
  sentences:
    - id: "s001"
      text: "Si les ventes baissent, on réduit les coûts."
      tokens:
        - {id: 1, form: "Si", lemma: "si", pos: "SCONJ", dep_rel: "mark", dep_head: 4,
           gcn: {causal_type: conjonction, causal_class: condition, attributes: {direction: forward}}}
        - {id: 2, form: "les", lemma: "le", pos: "DET", dep_rel: "det", dep_head: 3,
           gcn: {causal_type: determinant, causal_class: defini, attributes: {scope: universal}}}
        - {id: 3, form: "ventes", lemma: "vente", pos: "NOUN", dep_rel: "nsubj", dep_head: 4,
           gcn: {causal_type: nom, causal_class: processus}}
        - {id: 4, form: "baissent", lemma: "baisser", pos: "VERB", dep_rel: "advcl", dep_head: 7,
           gcn: {causal_type: verbe, causal_class: processus,
                 attributes: {direction: both, reversibility: true}}}
        - {id: 6, form: "on", lemma: "on", pos: "PRON", dep_rel: "nsubj", dep_head: 7,
           gcn: {causal_type: pronom, causal_class: indefini, attributes: {scope: universal}}}
        - {id: 7, form: "réduit", lemma: "réduire", pos: "VERB", dep_rel: "root", dep_head: 0,
           gcn: {causal_type: verbe, causal_class: action,
                 attributes: {direction: forward, reversibility: true}}}
        - {id: 9, form: "coûts", lemma: "coût", pos: "NOUN", dep_rel: "obj", dep_head: 7,
           gcn: {causal_type: nom, causal_class: patient}}
      cir:
        nodes:
          - {id: "n001", type: processus, label: "décroissance(ventes)",
             token_span: [3, 4], attributes: {entity: "ventes", quality: "décroissance",
             temporal_index: 0, scope: universal}}
          - {id: "n002", type: action, label: "réduire(coûts)",
             token_span: [6, 9], attributes: {agent: "on", patient: "coûts",
             temporal_index: 1, scope: universal}}
        edges:
          - {source: "n001", target: "n002", relation: condition,
             attributes: {confidence: 1.0, temporal_gap: 1, explicit: true, marker_token: 1}}
```

### Exemple wolof

```yaml
    - id: "s001"
      text: "Bu jaay yi wàcci, dinanu wàññi njëg yi."
      tokens:
        - {id: 1, form: "Bu", lemma: "bu", pos: "SCONJ",
           gcn: {causal_type: conjonction, causal_class: condition,
                 attributes: {direction: forward}}}
        - {id: 2, form: "jaay", lemma: "jaay", pos: "NOUN",
           gcn: {causal_type: nom, causal_class: processus}}
        - {id: 3, form: "yi", lemma: "yi", pos: "DET",
           gcn: {causal_type: determinant, causal_class: defini,
                 attributes: {scope: universal}}}
        - {id: 4, form: "wàcci", lemma: "wàcci", pos: "VERB",
           gcn: {causal_type: verbe, causal_class: processus,
                 attributes: {direction: both}}}
```

## Schéma de dataset : GCN-PL (Langages de Programmation)

### Exemple Python

```yaml
snippet:
  id: "py-001"
  lang: python
  # def adjust_costs(sales):
  #     total = sum(sales)
  #     if total < threshold:
  #         reduction = total * 0.1
  #         return reduction
  #     return 0.0
  ast_nodes:
    - {id: 0, kind: "function_definition", text: "def adjust_costs(sales):",
       gcn: {causal_type: function_call, attributes: {pure: true, side_effect: false}}}
    - {id: 1, kind: "assignment", text: "total = sum(sales)",
       gcn: {causal_type: assignment, attributes: {data_flow: write}}}
    - {id: 2, kind: "if_statement", text: "if total < threshold:",
       gcn: {causal_type: conditional, attributes: {control_flow: branch}}}
    - {id: 3, kind: "assignment", text: "reduction = total * 0.1",
       gcn: {causal_type: assignment, attributes: {data_flow: write}}}
    - {id: 4, kind: "return_statement", text: "return reduction",
       gcn: {causal_type: return_value, attributes: {data_flow: read}}}
  cir:
    nodes:
      - {id: "n001", type: etat, label: "total = sum(sales)",
         attributes: {variable: "total", type_info: "float"}}
      - {id: "n002", type: condition, label: "total < threshold",
         attributes: {value_constraint: "less_than(total, threshold)"}}
      - {id: "n003", type: action, label: "reduction = total * 0.1",
         attributes: {variable: "reduction"}}
      - {id: "n004", type: etat, label: "return reduction",
         attributes: {variable: "reduction"}}
    edges:
      - {source: "n001", target: "n002", relation: data_dependency, attributes: {confidence: 1.0}}
      - {source: "n002", target: "n003", relation: condition, attributes: {confidence: 1.0}}
      - {source: "n003", target: "n004", relation: sequence, attributes: {confidence: 1.0, data_flow: true}}
```

### Exemple Rust

```yaml
snippet:
  id: "rs-001"
  lang: rust
  # fn process_market(market: &mut Market) -> Result<Report, Error> {
  #     let status = market.evaluate();
  #     match status {
  #         Status::Declining => { market.reduce_costs(0.1); Ok(Report::adjusted()) }
  #         Status::Stable => Ok(Report::unchanged()),
  #     }
  # }
  ast_nodes:
    - {id: 0, kind: "function_item", text: "fn process_market(...)",
       gcn: {causal_type: function_call, attributes: {pure: false, side_effect: true}}}
    - {id: 1, kind: "let_declaration", text: "let status = market.evaluate();",
       gcn: {causal_type: assignment, attributes: {data_flow: write}}}
    - {id: 2, kind: "match_expression", text: "match status { ... }",
       gcn: {causal_type: conditional, attributes: {control_flow: branch}}}
    - {id: 3, kind: "call_expression", text: "market.reduce_costs(0.1)",
       gcn: {causal_type: mutation, attributes: {side_effect: true}}}
  cir:
    nodes:
      - {id: "n001", type: etat, label: "market.evaluate() → status",
         attributes: {variable: "status", type_info: "Status"}}
      - {id: "n002", type: condition, label: "status == Declining",
         attributes: {value_constraint: "equals(status, Status::Declining)"}}
      - {id: "n003", type: transition, label: "market.reduce_costs(0.1)",
         attributes: {variable: "market", function: "reduce_costs"}}
    edges:
      - {source: "n001", target: "n002", relation: data_dependency}
      - {source: "n002", target: "n003", relation: condition}
```

## Les 8 taxonomies GCN (structure YAML)

Chaque fichier `gcn-references/taxonomies/*.yaml` suit cette structure et contient les entrées françaises du papier :

1. **verbes.yaml** — 4 classes (état, action, transition, processus) + règles compositionnelles (aspect modifie le type)
2. **adverbes.yaml** — 7 classes (intensité, temps, fréquence, manière, lieu, négation, probabilité) avec valeurs numériques
3. **adjectifs.yaml** — 5 classes (intrinsèque, temporaire, relationnel, quantitatif, temporel)
4. **conjonctions.yaml** — 6 classes (cause, conséquence, condition, concession, temps, opposition) — concession/opposition marquent les lacunes causales
5. **prepositions.yaml** — 5 classes (spatiale, temporelle, cause, moyen, but) — "but" inverse la direction causale
6. **noms.yaml** — 6 classes (agent [humain/collectif/institutionnel/naturel], patient, processus, état_systémique, abstrait, relation)
7. **pronoms.yaml** — 4 classes (personnel, démonstratif, relatif, indéfini) — indéfinis signalent lacunes causales
8. **determinants.yaml** — 7 classes (défini, indéfini, partitif, démonstratif, possessif, quantitatif, interrogatif) avec scope ∀/∃/¬∃

## Lacunes identifiées et corrections

### L1 — Représentation temporelle incomplète

Le plan mentionne `TemporalRef` et `TemporalAnchor` sans les définir. Le papier utilise T, T-n, T+n, T+ε. Ajout dans `gcn-ir/src/temporal.rs` :

```rust
pub enum TemporalRef {
    Absolute(i64),                    // Timestamp absolu (pour ancrage)
    Relative { base: i32, offset: i32 }, // T+n, T-n
    Epsilon,                          // T+ε (immédiat, "dès que")
    Range { start: i32, end: i32 },   // [T0, T1] pour les processus (accumulation)
    Unresolved,                       // Temporalité non encore déterminée
}

pub struct TemporalGap {
    pub min: Option<i32>,             // Écart minimal
    pub max: Option<i32>,             // Écart maximal
    pub nature: GapNature,            // Immédiat | Différé | Continu
}

pub enum GapNature {
    Immediate,  // T+ε — "il tombe et se casse le bras"
    Deferred,   // T+n — "les ventes baissent → on réduit les coûts (semaines)"
    Continuous, // [T, T+n] — processus d'accumulation
}
```

### L2 — Type Ambiguity manquant

Le `CausalIR` a un champ `unresolved: Vec<Ambiguity>` mais `Ambiguity` n'est jamais défini. Ajout :

```rust
pub struct Ambiguity {
    pub node_id: NodeId,
    pub field: AmbiguousField,        // Quel champ est ambigu
    pub candidates: Vec<AmbiguityCandidate>,
    pub context_hint: Option<String>, // Indice contextuel pour la résolution
}

pub enum AmbiguousField {
    NodeType,       // "baisser" → Processus ou Action?
    Scope,          // "un marché" → Universel ou Existentiel?
    AgentType,      // "on" → Humain ou Indéfini?
    CorefTarget,    // "il" → quel antécédent?
}

pub struct AmbiguityCandidate {
    pub value: String,        // ex: "processus"
    pub confidence: f32,      // 0.6
    pub rule_source: String,  // "compositional_rule: verbe + depuis → processus"
}
```

### L3 — Feedback loop non typé

L'architecture promet un feedback middleend → frontend mais aucun type ne le porte. Ajout dans `gcn-middleend` :

```rust
pub struct ValidationFeedback {
    pub inconsistencies: Vec<Inconsistency>,
    pub suggested_reannotations: Vec<Reannotation>,
}

pub struct Inconsistency {
    pub kind: InconsistencyKind,
    pub node_ids: Vec<NodeId>,
    pub message: String,
}

pub enum InconsistencyKind {
    TemporalContradiction,    // Cause après effet dans le graphe
    BrokenCausalChain,        // Effet sans cause accessible
    ScopeConflict,            // ∀ sur un noeud lié à ¬∃ sur un autre
    TypeMismatch,             // Action sans agent, État sans entité
    OrphanCycle,              // Cycle dont un noeud n'a pas de source
}

pub struct Reannotation {
    pub token_id: u32,
    pub current_type: String,
    pub suggested_type: String,
    pub reason: String,
}
```

Le pipeline devient : `frontend.emit() → middleend.validate() → if feedback.has_inconsistencies() → frontend.reannotate(feedback) → retry`.

### L4 — Graphe incrémental : stratégie de fusion non spécifiée

Le plan dit "incrémental" mais ne dit pas comment fusionner les CIR de phrases successives. Ajout de la spec dans `gcn-middleend/src/graph.rs` :

**Règles de fusion :**
1. **Entités identiques** : si deux noeuds ont la même `entity` et un `scope` compatible, ils fusionnent (merge). Le noeud résultant conserve le `temporal_index` le plus récent.
2. **Coréférences résolues** : pronom → antécédent déjà dans le graphe → nouvelle edge depuis l'antécédent existant.
3. **Nouveaux noeuds** : ajoutés au graphe avec edges temporelles vers les noeuds existants selon `temporal_index`.
4. **Détection de cycles** : après chaque fusion, `petgraph::algo::tarjan_scc()` détecte les composantes fortement connexes de taille > 1 → cycles.

### L5 — Causalité implicite non traitée

Le papier mentionne des relations causales sans marqueur explicite (juxtaposition : "les ventes baissent. On réduit les coûts."). Le plan ne traite que la causalité marquée par des conjonctions/prépositions.

**Ajout** : dans `gcn-frontend-fr/annotator.rs`, règle de causalité par proximité :
- Si deux clauses adjacentes partagent une entité et que la seconde contient une Action dont le patient est lié à l'état de la première → edge `Cause` avec `confidence: 0.5` et `explicit: false`.
- La confiance basse signale que c'est inféré, pas marqué.

### L6 — Négation au niveau des edges

Le plan modélise la négation comme `Modifier::Negation` sur un noeud. Mais la négation peut porter sur la **relation** elle-même : "La crise n'a PAS causé le chômage" nie l'edge, pas le noeud.

**Ajout** dans `CausalEdge` :
```rust
pub struct CausalEdge {
    // ... champs existants ...
    pub negated: bool,  // La relation elle-même est niée
}
```

### L7 — Sérialisation du DiGraph

Le plan met `#[serde(skip)]` sur `graph: DiGraph<...>` dans `CausalIR`, ce qui signifie que le graphe n'est pas sérialisé. C'est fragile — `rebuild_graph()` doit être appelé manuellement.

**Correction** : supprimer le champ `graph` de `CausalIR`. Le `DiGraph` est construit à la demande par le middle-end depuis `nodes` + `edges`. `CausalIR` ne contient que les vecteurs sérialisables. Le graphe vit dans `CausalGraph` (middle-end), pas dans `CausalIR` (ir).

```rust
// gcn-ir/src/ir.rs — sérialisable à 100%
pub struct CausalIR {
    pub source_lang: SourceLanguage,
    pub source_text: String,
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(NodeId, NodeId, CausalEdge)>,
    pub cycles: Vec<CausalCycle>,
    pub unresolved: Vec<Ambiguity>,
    pub metadata: IrMetadata,
}

// gcn-middleend/src/graph.rs — construit à la demande
pub struct CausalGraph {
    inner: DiGraph<NodeId, CausalEdge>,
    node_index: HashMap<NodeId, petgraph::graph::NodeIndex>,
    entity_index: HashMap<String, Vec<NodeId>>,
    temporal_index: BTreeMap<i32, Vec<NodeId>>,
}

impl CausalGraph {
    pub fn from_cir(cir: &CausalIR) -> Self { ... }
    pub fn integrate(&mut self, new_cir: &CausalIR) -> ValidationFeedback { ... }
}
```

### L8 — Configuration du système absente

Aucune mention de comment configurer quel frontend, quelle langue, ML on/off. Ajout d'un fichier `gcn-core/gcn.toml` :

```toml
[frontend]
language = "fr"            # fr | wo | en | ar
parser = "udpipe"          # udpipe | rules_only

[middleend]
enable_cycle_detection = true
max_propagation_depth = 10
merge_strategy = "entity"  # entity | temporal | aggressive

[backend]
pearl_levels = [1]         # [1] | [1, 2] | [1, 2, 3]
export_format = "json"     # json | dot | graphml

[ml]
enabled = false            # Nécessite feature "ml" à la compilation
model = "camembert-base"
confidence_threshold = 0.6 # En dessous → signalé comme ambiguïté
```

Ajout d'un fichier `crates/gcn-cli/src/config.rs` dans la structure.

### L9 — Distribution des modèles UDPipe

Le modèle UDPipe français (~20MB) n'est pas inclus dans le binaire. Il faut une stratégie de distribution.

**Solution** : le modèle est téléchargé au premier lancement via `gcn-cli setup` ou inclus optionnellement via `include_bytes!` avec une feature Cargo `embed-models`. Par défaut, le binaire est léger et le modèle est téléchargé.

Ajout dans la structure : `crates/gcn-cli/src/setup.rs` pour le téléchargement initial.

### L10 — Scope `Specific` manquant

L'enum `Scope` ne distingue pas "le marché" (spécifique, identifié) de "un marché" (indéfini). Le déterminant défini pointe vers une instance connue.

**Correction** : Scope inclut déjà `Universal, Existential, Partial, Null` — ajout de `Specific` et `Unknown` :
```rust
pub enum Scope {
    Universal,    // ∀ — "tous les marchés"
    Existential,  // ∃ — "certains marchés"
    Partial,      // quantité partielle — "du travail"
    Null,         // ¬∃ — "aucun marché"
    Specific,     // Le/La — instance identifiée
    Unknown,      // Non résolu
}
```

### L11 — Noeuds explicites vs inférés

Le champ `explicit` existe sur les edges mais pas sur les noeuds. Un agent implicite ("on réduit les coûts" — qui est "on"?) doit être marqué différemment d'un agent explicite.

**Ajout** dans `CausalNode` :
```rust
pub struct CausalNode {
    // ... champs existants ...
    pub origin: NodeOrigin,
}

pub enum NodeOrigin {
    Explicit,     // Présent littéralement dans le texte
    Inferred,     // Déduit par résolution (coréf, agent implicite)
    Hypothetical, // Postulé par une lacune causale (concession → cause cachée)
}
```

### L12 — Langage de requêtes GCN-QL (implémenté)

Syntaxe GCN-QL complète (tous niveaux de Pearl) :

```
# Pearl niveau 1 — Association (observationnel)
WHY <label>?                    # Ancêtres causaux (BFS inverse)
WHAT <label>?                   # Descendants causaux (BFS avant)
CHAIN <a> -> <b>?               # Chemin causal entre deux nœuds
CYCLES?                         # Boucles de rétroaction (Tarjan SCC)
GAPS?                           # Lacunes causales (TemporalGap)

# Pearl niveau 2 — Intervention (do-calculus)
DO <label>?                     # Coupe les causes de X, propage ses effets
                                # → QueryResult::Intervention { severed_count, severed, effects }

# Pearl niveau 3 — Contrefactuel
COUNTERFACTUAL <label>?         # "Si X n'avait pas eu lieu, quels effets disparaissent ?"
                                # → QueryResult::CounterfactualDiff { actual_effects, unique_effects }
```

Implémenté comme un parser à la main dans `gcn-backend/src/query.rs` (pas de dépendance externe).

## Architecture CGNP — Package Python (gcn-python/)

Le package `gcn-python/` (sibling de `gcn-core/`) implémente les **3 couches ML de l'architecture CGNP**. Il ne dépend d'aucun framework ML — les implémentations de référence sont en NumPy pur. Le data scientist substitue ses propres implémentations via des contrats d'interface (Protocol Python).

```
projet_CNM/
├── gcn-core/        # Rust : moteur pur (IR, middle-end, backend Pearl, CLI)
├── gcn-references/  # Références linguistiques (taxonomies, lexicons)
├── gcn-datasets/    # Données annotées d'entraînement
└── gcn-python/      # Python : couches ML 1-3 de l'architecture CGNP
    ├── pyproject.toml
    └── src/gcn_python/
        ├── constants.py          # Sync avec gcn-ir enums (snake_case serde)
        ├── taxonomy/loader.py    # TaxonomyIndex par langue
        ├── data/                 # Chargement datasets YAML annotés
        ├── layer1/               # Couche 1 : représentation UD universelle
        │   ├── representation.py # UDRepresentation (abstraction UD-agnostique)
        │   ├── extractor.py      # spaCy Doc → List[UDRepresentation]
        │   └── features.py       # FeatureVocabulary + vectorize() → ndarray[D_clause≈135]
        ├── layer2/               # Couche 2 : encodage causal MLP
        │   ├── interface.py      # CausalEncoder Protocol
        │   └── reference.py      # MLPEncoder NumPy (fc→ReLU→fc→ReLU→fc)
        ├── layer3/               # Couche 3 : graphe causal R-GCN
        │   ├── interface.py      # CausalGraph Protocol
        │   ├── reference.py      # RGCNLayer NumPy (message passing relationnel)
        │   └── pytorch_rgcn.py   # RGCNLayerPT PyTorch (GPU/MPS, autograd, scatter_add_)
        ├── pipeline/
        │   ├── cgnp.py           # CGNPipeline.forward() → CIR JSON
        │   │                     # .loss(node_logits, edge_logits, gold_node, gold_edge) → (float, d_node, d_edge)
        │   │                     # .backward(d_node, d_edge, lr) → SGD + snapshots par nœud
        │   ├── label_builder.py
        │   ├── ir_emitter.py     # → JSON conforme schéma serde Rust CausalIR
        │   └── cli.py            # gcn-forward --lang fr [--model-path model.npz] "texte"
        ├── data/
        │   ├── loader.py         # GCNDataLoader — itère sur YAML → TrainingSample
        │   ├── yaml_reader.py    # load_all_sentences(data_dir, lang) → List[SentenceRecord]
        │   └── schema.py         # SentenceRecord, ClauseRecord, EdgeRecord
        ├── training/
        │   ├── train.py          # gcn-train CLI — boucle SGD multi-epoch
        │   ├── checkpoint.py     # save/load checkpoint .npz
        │   └── bootstrap.py      # gcn-bootstrap CLI — génération YAML via gcn analyze
        └── evaluation/
            ├── metrics.py        # Métriques NumPy pures (framework-agnostiques)
            │                     #   node_accuracy, node_f1_per_class
            │                     #   edge_accuracy, edge_f1_per_class
            │                     #   causal_graph_similarity (nœuds + arêtes)
            └── recorder.py       # TrainingRecorder — suivi loss + métriques par epoch
                                  #   .record(epoch, loss, metrics)
                                  #   .learning_curve() → dict {epoch: [], loss: [], …}
                                  #   .to_csv(path)
```

### Contrats d'interface (Couches 2 et 3)

```python
# Couche 2 — CausalEncoder
class CausalEncoder(Protocol):
    def forward_node(self, x: np.ndarray) -> np.ndarray: ...  # (D_clause,) → (7,) logits
    def forward_edge(self, x: np.ndarray) -> np.ndarray: ...  # (D_edge,)   → (11,) logits
    def parameters(self) -> list[np.ndarray]: ...
    def update(self, grads: list[np.ndarray], lr: float) -> None: ...

# Couche 3 — CausalGraph (R-GCN)
class CausalGraph(Protocol):
    def message_pass(
        self, node_features: np.ndarray,  # (N, D_node)
        edge_index: np.ndarray,           # (2, E)
        edge_types: np.ndarray,           # (E,) int
    ) -> np.ndarray: ...                  # (N, D_out)
    def backward_message_pass(
        self, d_output: np.ndarray,       # (N, D_out) — gradient aval
    ) -> tuple[np.ndarray, list[np.ndarray]]: ...  # (d_input, [dW_r, dW_0])
    def parameters(self) -> list[np.ndarray]: ...
    def update(self, grads: list[np.ndarray], lr: float) -> None: ...
```

Formule R-GCN : `h_i^(l+1) = σ( Σ_r Σ_{j∈N_r(i)} (1/c_{i,r}) W_r^(l) h_j^(l) + W_0^(l) h_i^(l) )`  
Support natif des cycles (pas de restriction DAG).

### Flux complet couches 1→3→Rust couche 4

```
text + lang_code
  ↓ [Couche 1] spaCy UD → UDRepresentation[] → ndarray[N, D_clause]
  ↓ [Couche 2] CausalEncoder → NodeType[] + RelationType[]
  ↓ [Couche 3] CausalGraph.message_pass() → représentations enrichies
  ↓ ir_emitter → CausalIR JSON
  ↓ stdout → gcn-cli → serde_json::from_slice::<CausalIR>()
  ↓ [Couche 4 Rust] Pearl inference (do-calculus)
```

### Features Couche 1 (D_clause ≈ 135, universelles)

| Groupe | Dims |
|---|---|
| UPOS racine (one-hot 18 + unk) | 19 |
| dep_rel racine (one-hot 37 UD + unk) | 38 |
| UPOS sujet (one-hot 5) | 5 |
| UD Morph Tense/Aspect/Mood/Polarity | 15+1 |
| Flags UD structurels (has_obj, has_advcl, has_temporal_obl) | 3 |
| Appartenance taxonomie GCN (~56 gcn_class_keys) | ~56 |

`D_edge = 2 × D_clause + D_conn = 346` (D_conn = 76)

## Décisions architecturales

- **DA-1** : La CIR est le contrat central. Tout frontend produit une CIR, le middle-end/backend ne connaissent que la CIR.
- **DA-2** : Les couches ML (2-3) sont dans `gcn-python/` — framework-agnostique (NumPy de référence, `RGCNLayerPT` PyTorch disponible, DS substitue son implémentation via Protocol). L'interface Python↔Rust est CausalIR JSON via subprocess.
- **DA-3** : Le graphe est orienté AVEC cycles (petgraph DiGraph, pas Dag) — section 4.1.bis du papier.
- **DA-4** : Les lacunes causales sont des données de première classe (concession, opposition, pronoms indéfinis).
- **DA-5** : La compositionnalité verbale est implémentée — le type causal d'un verbe dépend de l'aspect et du contexte.
- **DA-6** : Le `DiGraph` vit dans `CausalGraph` (middle-end), PAS dans `CausalIR` (ir). La CIR est 100% sérialisable.
- **DA-7** : Chaque noeud et edge porte son `origin` (Explicit/Inferred/Hypothetical) et `negated` pour la traçabilité.
- **DA-8** : La causalité implicite (juxtaposition) est détectée avec une confiance réduite (0.5), signalée comme non-explicite.

## Phases d'implémentation

| Phase | Composants | Statut |
|---|---|---|
| 1 | `gcn-ir` + `gcn-knowledge` (11 taxonomies YAML fr) | ✅ Terminé |
| 2a | `gcn-frontend-fr` (règles, CIR, 16 tests) | ✅ Terminé |
| 2b | `gcn-python/` couches 1-3 + `pipeline/` + `training/` (boucle SGD NumPy) | ✅ Terminé |
| 2c | `gcn-python/evaluation/` : métriques, `TrainingRecorder` | ✅ Terminé |
| 3 | `gcn-middleend` (graphe DiGraph, cycles Tarjan, propagation, validation) — 17 tests | ✅ Terminé |
| 4 | `gcn-backend` (Pearl niveau 1 + GCN-QL) + `gcn-cli` + interface Rust↔Python — 26 tests | ✅ Terminé |
| 5 | `gcn-verbalizer` (décodeur CausalIR → surface) + métriques décodeur | ✅ Terminé |
| 6 | `gcn-frontend-code` (Python/Rust/JS AST via tree-sitter) — 27 tests | ✅ Terminé |
| 7 | Pearl 2-3 (intervention, contrefactuel), `RGCNLayerPT` PyTorch, `gcn-frontend-en` — 16 tests | ✅ Terminé |

## Vérification

1. **Phase 1** : `cargo test -p gcn-ir -p gcn-knowledge` — les types compilent, les taxonomies se chargent, proptest valide les invariants (ex: tout CausalNode sérialisé→désérialisé = identique, tout Scope a un quantificateur logique associé)
2. **Phase 2** : Tous les exemples du papier (`paper_examples.yaml`) passent le pipeline frontend-fr → CIR et produisent les graphes attendus. Test spécifique : causalité implicite sur phrases juxtaposées.
3. **Phase 3** : Tests d'intégration phrase → CIR → graphe. Détection des cycles sur l'exemple "ventes → coûts → qualité → ventes". Test feedback loop : annotation incohérente → middleend retourne `Inconsistency::TemporalContradiction`. Test fusion incrémentale : 3 phrases → graphe unifié.
4. **Phase 4** : `gcn-cli analyze "Si les ventes baissent, on réduit les coûts."` → JSON CIR correct. `gcn-cli query "WHY ventes?"` → chaîne causale. `gcn-cli query "GAPS?"` → lacunes.
5. **Phase 5** : `Verbalizer::decode(ir)` retourne une surface non vide pour tout CausalIR valide. `roundtrip_similarity` ≥ 0.7 sur les exemples gold du dataset. `cross_modal_consistency` ≥ 0.7 entre deux surfaces du même graphe. La sortie dépend de l'entraînement, pas de l'architecture.
6. **Phase 6** : Snippet Python → CIR produit le même graphe causal qu'une description française équivalente. Test : `if x < y: reduce(z)` et "Si x est inférieur à y, on réduit z" → CIR isomorphe. ✅
7. **Phase 7** : `gcn query "DO X"` et `gcn query "COUNTERFACTUAL X"` retournent les résultats attendus. `gcn-frontend-en` produit un CIR isomorphe à `gcn-frontend-fr` pour la même structure conditionnelle. `RGCNLayerPT.message_pass()` implémente le Protocol `CausalGraph`. ✅
8. **End-to-end** : `cargo test --workspace` passe (103 tests), `cargo clippy -- -D warnings` propre, `python -m pytest gcn-python/tests/` passe (76 collectés, 71 passent, 5 skippés sans spaCy fr).

## Architecture du décodeur

Le décodeur `gcn-verbalizer` est la partie génératrice de l'architecture encodeur-décodeur GCN.

**Principe** : CausalIR JSON → surface. Ce que le décodeur produit dépend entièrement de son entraînement — aucun type de sortie n'est présupposé par l'architecture.

**Entraînement** : même corpus que l'encodeur (`gcn-datasets/`). Les paires `(CausalIR, surface)` du schéma `gcn-verbalize.schema.yaml` servent les deux côtés.

**Évaluation** :
- `decoder_causal_fidelity` : re-parser la sortie et comparer au CausalIR source
- `cross_modal_consistency` : comparer deux surfaces (fr + python) du même CausalIR
- `roundtrip_similarity` : mesurer la fidélité du cycle complet
- `generation_bleu` : qualité de surface pour le langage naturel

**Interface Rust** : `gcn_verbalizer::decode(ir: &CausalIR) -> Result<String, VerbalizerError>`
**Interface Python** : `VerbalizerDecoder.decode(ir_json: str) -> str`

### Composants

| Composant | Rôle |
|---|---|
| `gcn-verbalizer/src/lib.rs` | Pont Rust — appelle `gcn-verbalize` (subprocess), retourne `String` |
| `gcn-python/verbalizer/interface.py` | Protocol `VerbalizerDecoder` |
| `gcn-python/verbalizer/decoder.py` | Implémentation NumPy référence (remplaçable par PyTorch/JAX) |
| `gcn-python/verbalizer/cli.py` | CLI `gcn-verbalize` — entrée JSON stdin, sortie texte stdout |
| `gcn-datasets/schemas/gcn-verbalize.schema.yaml` | Schéma des paires d'entraînement `(CausalIR, surface_text, lang)` |
