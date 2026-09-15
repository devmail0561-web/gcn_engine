# gcn-frontend-en — English Causal Parser

[![crates.io](https://img.shields.io/crates/v/gcn-frontend-en)](https://crates.io/crates/gcn-frontend-en)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

English symbolic frontend for the **GCN-Core** engine. Parses natural language English text and produces a `CausalIR`: tokenization, POS tagging, causal marker detection, clause annotation, graph emission.

The English and French frontends are **isomorphic** — the same causal sentence in French and English produces structurally equivalent `CausalIR` graphs (same node count, same edge types).

---

## Role in GCN-Core

```
"If sales drop, we reduce costs."
            │
     gcn-frontend-en
     ├── tokenizer  → Vec<Token>
     ├── tagger     → Vec<TaggedToken>
     ├── annotator  → SentenceAnnotation
     └── emitter    → CausalIR
            │
            ▼
    gcn-middleend / gcn-python
```

---

## Dependency

```toml
[dependencies]
gcn-frontend-en = "1.0"
```

---

## Usage

```rust
use gcn_frontend_en::EnglishParser;
use std::path::Path;

// Loads English resources from taxonomies_root/en/
let parser = EnglishParser::new(Path::new("gcn-references/taxonomies"))?;

let ir = parser.parse("Heavy rain causes flooding.")?;
println!("{} nodes, {} edges", ir.node_count(), ir.edge_count());

// JSON export
let json = serde_json::to_string_pretty(&ir)?;
```

### Isomorphism with French frontend

```rust
let fr_parser = gcn_frontend_fr::FrenchParser::new(&taxonomies)?;
let en_parser = EnglishParser::new(&taxonomies)?;

let ir_fr = fr_parser.parse("Si les ventes baissent, on réduit les coûts.")?;
let ir_en = en_parser.parse("If sales drop, we reduce costs.")?;

// Both produce: 2 nodes (processus + action), 1 edge (condition)
assert_eq!(ir_fr.node_count(), ir_en.node_count());
```

---

## Public API

### `EnglishParser`

```rust
pub struct EnglishParser { /* private */ }

impl EnglishParser {
    /// Loads resources from `taxonomies_root/en/`.
    pub fn new(taxonomies_root: &Path) -> Result<Self, ParserInitError>;

    /// Tokenizes, tags, annotates, and emits a CausalIR.
    pub fn parse(&self, text: &str) -> Result<CausalIR, EnParseError>;
}
```

### Errors

```rust
pub enum ParserInitError {
    Knowledge(KnowledgeError),
}

pub enum EnParseError {
    EmptyInput,
    NoParseable(String),
}
```

---

## Internal components (re-exported for extension)

### `Token` and `tokenize`

```rust
pub struct Token {
    pub form: String,
    pub lower: String,
    pub index: u32,  // 0-based (unlike French which is 1-based)
}

pub fn tokenize(text: &str) -> Vec<Token>
// Splits trailing punctuation (.,;:!?) as separate tokens
```

### `TaggedToken` and `tag`

```rust
pub enum Pos { Verb, Noun, Det, Pron, Conj, Prep, Adv, Punct, Other }

pub struct TaggedToken {
    pub token: Token,
    pub pos: Pos,
    pub lemma: String,
    pub scope_hint: Option<Scope>,
    pub is_past: bool,              // past tense (-ed suffix)
    pub is_negation_particle: bool, // "not", "n't"
    pub is_negation_completer: bool,
}

pub fn tag(tokens: &[Token], res: &LexicalResources) -> Vec<TaggedToken>
// Single-pass (no positional heuristic, unlike French)
```

### `SentenceAnnotation` and `annotate`

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
    pub is_progressive: bool,  // progressive aspect (ongoing process)
    pub neg_on_node: bool,
    // ...
}

// EdgeAnnotation and SentenceAnnotation: identical to French

pub fn annotate(tagged: &[TaggedToken], res: &LexicalResources) -> SentenceAnnotation
// Strategy order: causal marker first, then causal verb (reversed from French)
```

### `emit`

```rust
pub fn emit(ann: SentenceAnnotation, source_text: String) -> CausalIR
// source_lang = Natural { lang: English }
// Progressive aspect → Modifier::Temporality { anchor: Present }
```

### Utility functions (`rules`)

```rust
// English verb lemmatization
pub fn lemmatize_verb(form: &str) -> String
// Handles: n't, -ied→y, -ed double consonant, -ing, -ies, bare -s

// Morphological detection
pub fn is_past_tense(form: &str) -> bool         // -ed suffix
pub fn is_infinitive_or_base(form: &str) -> bool // negation of -ing/-ed/-s

// Same mapping functions as French frontend:
pub fn causal_direction_to_node_type(dir: &str) -> Option<NodeType>
pub fn relation_type_str_to_enum(rt: &str) -> Option<RelationType>
pub fn scope_str_to_enum(s: &str) -> Option<Scope>
pub fn noun_class_to_node_type(class_name: &str) -> Option<NodeType>
pub enum MarkerDir { Backward, Forward, GoalToAction }
pub fn nominalize_with_table<'a>(verb_lemma: &'a str, table: &'a HashMap<String, String>) -> &'a str
```

### `LexicalResources`

```rust
pub struct LexicalResources { /* same fields as French */ }

impl LexicalResources {
    pub fn load(data_dir: &Path) -> Result<Self, KnowledgeError>
    // Reads `examples` field (language-neutral) instead of `examples_fr`
}
```

---

## Known limitations

- **Coreference resolution**: not implemented. `EnglishParser` is a bootstrap tool.
- **Coverage**: limited to markers and verbs present in `en/` taxonomy YAML files.
- **Aspect**: progressive aspect partially implemented (`is_progressive` flag).

---

## License

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
