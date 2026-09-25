# gcn-frontend-fr — Parser Causal Français

Version: 2.5.0

[![Crates.io](https://img.shields.io/crates/v/gcn-frontend-fr)](https://crates.io/crates/gcn-frontend-fr)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Frontend symbolique français du moteur **GCN-Core**. Analyse du texte naturel français et produit un `CausalIR` : tokenisation, étiquetage POS, détection de marqueurs causaux, annotation de clauses, émission du graphe.

---

## Rôle dans GCN-Core

```
"Si les ventes baissent, on réduit les coûts."
            │
     gcn-frontend-fr
     ├── tokenizer  → Vec<Token>
     ├── tagger     → Vec<TaggedToken>  (POS + lemme + flags)
     ├── annotator  → SentenceAnnotation  (clauses + arêtes)
     └── emitter    → CausalIR
            │
            ▼
    gcn-middleend / gcn-python
```

---

## Dépendance

```toml
[dependencies]
gcn-frontend-fr = "2.5.0"
```

---

## Utilisation

```rust
use gcn_frontend_fr::FrenchParser;
use std::path::Path;

// Initialisation (charge les taxonomies françaises depuis taxonomies_root/fr/)
let parser = FrenchParser::new(Path::new("gcn-references/taxonomies"))?;

// Analyse d'une phrase
let ir = parser.parse("La pluie provoque des inondations.")?;

println!("{} nœuds, {} arêtes", ir.node_count(), ir.edge_count());

// Sérialisation JSON
let json = serde_json::to_string_pretty(&ir)?;
```

---

## API publique

### `FrenchParser`

```rust
pub struct FrenchParser { /* private */ }

impl FrenchParser {
    /// Charge les ressources depuis `taxonomies_root/fr/`.
    pub fn new(taxonomies_root: &Path) -> Result<Self, ParserInitError>;

    /// Tokenise, étiquette, annote, et émet un CausalIR.
    pub fn parse(&self, text: &str) -> Result<CausalIR, FrParseError>;
}
```

### Erreurs

```rust
pub enum ParserInitError {
    Knowledge(KnowledgeError), // échec de chargement des taxonomies
}

pub enum FrParseError {
    EmptyInput,
    NoParseable(String), // aucune clause causale détectée dans le texte
}
```

---

## Composants internes (réexportés pour extension)

### `Token` et `tokenize`

```rust
pub struct Token {
    pub index: u32,   // position 1-based
    pub form: String, // forme de surface
    pub lower: String,
}

pub fn tokenize(text: &str) -> Vec<Token>
// Gère : séparation des clitiques (n', l', qu', ...), ponctuation isolée
```

### `TaggedToken` et `tag`

```rust
pub enum Pos { Verb, Noun, Det, Pron, Conj, Prep, Adv, Punct, Other }

pub struct TaggedToken {
    pub token: Token,
    pub pos: Pos,
    pub lemma: String,
    pub scope_hint: Option<Scope>,
    pub is_imparfait: bool,
    pub is_negation_particle: bool,
    pub is_negation_completer: bool,
}

pub fn tag(tokens: &[Token], res: &LexicalResources) -> Vec<TaggedToken>
// 2 passes : classification puis heuristique positionnelle
```

### `SentenceAnnotation` et `annotate`

```rust
pub struct ClauseAnnotation {
    pub span: (u32, u32),
    pub node_type: NodeType,
    pub label: String,
    pub scope: Scope,
    pub origin: NodeOrigin,
    pub agent: Option<String>,
    pub patient: Option<String>,
    pub entity: Option<String>,
    pub has_depuis: bool,       // aspect duratif "depuis"
    pub is_imparfait: bool,
    pub neg_on_node: bool,
    // ...
}

pub struct EdgeAnnotation {
    pub src_clause: usize,
    pub dst_clause: usize,
    pub relation: RelationType,
    pub confidence: f32,
    pub explicit: bool,
    pub negated: bool,
    pub marker_token_idx: Option<u32>,
}

pub struct SentenceAnnotation {
    pub clauses: Vec<ClauseAnnotation>,
    pub edges: Vec<EdgeAnnotation>,
}

pub fn annotate(tagged: &[TaggedToken], res: &LexicalResources) -> SentenceAnnotation
// Deux stratégies : verbe causal → marqueur causal
// Arêtes implicites inter-phrases (confiance 0.5)
```

### `emit`

```rust
pub fn emit(ann: SentenceAnnotation, source_text: String) -> CausalIR
// Assigne les temporal_index par propagation max sur les arêtes
// source_lang = Natural { lang: French }
```

### Fonctions utilitaires (`rules`)

```rust
// Lemmatisation verbale (NFC, réfléchis, passé composé, imparfait, -ent)
pub fn lemmatize_verb(form: &str) -> String

// Détection morphologique
pub fn is_imparfait(form: &str) -> bool      // -ait/-aient/-ais
pub fn is_infinitive(form: &str) -> bool     // -er/-ir/-re/-oir

// Conversion YAML → types GCN
pub fn causal_direction_to_node_type(dir: &str) -> Option<NodeType>
pub fn relation_type_str_to_enum(rt: &str) -> Option<RelationType>
pub fn scope_str_to_enum(s: &str) -> Option<Scope>
pub fn noun_class_to_node_type(class_name: &str) -> Option<NodeType>

// Direction du marqueur causal
pub enum MarkerDir { Backward, Forward, GoalToAction }
pub fn direction_str_to_enum(s: &str) -> MarkerDir
pub fn conjunction_class_to_direction(class_name: &str) -> MarkerDir

// Nominalisation via table pré-chargée
pub fn nominalize_with_table<'a>(verb_lemma: &'a str, table: &'a HashMap<String, String>) -> &'a str
```

### `LexicalResources`

```rust
pub struct LexicalResources {
    pub causal_markers: Vec<CausalMarkerEntry>,  // triés longueur décroissante
    pub causal_verb_relations: HashMap<String, RelationType>,
    pub verb_classes: HashMap<String, NodeType>,
    pub det_scope: HashMap<String, Scope>,
    pub noun_node_types: HashMap<String, NodeType>,
    pub pron_agent_types: HashMap<String, AgentType>,
    pub negation_particles: HashSet<String>,
    pub negation_completers: HashSet<String>,
    pub nominalizations: HashMap<String, String>,
    pub auxiliary_forms: HashSet<String>,
    // ...
}

impl LexicalResources {
    pub fn load(data_dir: &Path) -> Result<Self, KnowledgeError>
}

pub struct CausalMarkerEntry {
    pub lemma: String,
    pub words: Vec<String>,         // mots du marqueur (pré-découpés)
    pub relation: RelationType,
    pub direction: MarkerDir,
    pub signals_gap: bool,
    pub requires_infinitive: bool,
}
```

---

## Limites connues

- **Coréférence inter-phrases** : non implémentée. `FrenchParser` est un outil de bootstrap. Dans le pipeline ML de production, les labels sont résolus manuellement dans les datasets `gcn-datasets/`.
- **Couverture** : limitée aux marqueurs et verbes présents dans les taxonomies YAML.
- **Compositionnalité verbale** : aspect partiellement implémenté (`depuis` → Processus, imparfait → état duratif).

---

## Licence

Apache-2.0 — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
