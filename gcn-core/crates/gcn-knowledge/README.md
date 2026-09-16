# gcn-knowledge — Taxonomies, Lexicon et Moteur d'Inférence

Version: 2.0.0

[![Crates.io](https://img.shields.io/crates/v/gcn-knowledge)](https://crates.io/crates/gcn-knowledge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Crate de connaissance symbolique du moteur **GCN-Core**. Charge les taxonomies YAML, fournit un lexicon pour la classification POS → classe causale, et enrichit les `CausalIR` via un moteur d'inférence structurel.

---

## Rôle dans GCN-Core

```
gcn-references/taxonomies/*.yaml
            │
            ▼
     gcn-knowledge
     ├── Lexicon        ← classification POS + lemme → NodeType
     ├── TaxonomyLoader ← lecture YAML
     └── InferenceEngine ← enrichissement CausalIR
            │
            ▼
  gcn-frontend-fr / gcn-frontend-en
  (utilisent Lexicon pour le tagging et l'annotation)
```

---

## Dépendance

```toml
[dependencies]
gcn-knowledge = "2.0"
```

---

## Chargement des taxonomies

### `load_taxonomy` / `load_all_taxonomies`

```rust
use gcn_knowledge::loader::{load_taxonomy, load_all_taxonomies};
use std::path::Path;

// Un seul fichier
let taxonomy = load_taxonomy(Path::new("taxonomies/fr/verbes.yaml"))?;

// Répertoire entier → HashMap<String, Taxonomy> clé = taxonomy.taxonomy
let taxonomies = load_all_taxonomies(Path::new("taxonomies/fr/"))?;
```

### `Taxonomy`

```rust
pub struct Taxonomy {
    pub taxonomy: String,
    pub version: String,
    pub description: String,
    pub classes: HashMap<String, TaxonomyClass>,
    pub compositional_rules: Option<Vec<CompositionalRule>>,
}

pub struct TaxonomyClass {
    pub description: Option<String>,
    pub causal_direction: Option<String>, // "production", "maintien", ...
    pub causal_effect: Option<String>,
    pub relation_type: Option<String>,
    pub signals_gap: Option<bool>,
    pub scope: Option<String>,
    pub examples_fr: Option<Vec<LexicalEntry>>, // entrées lexicales françaises
    pub examples: Option<Vec<LexicalEntry>>,    // entrées langue-neutre (EN, ...)
    pub subtypes: Option<HashMap<String, SubType>>,
    // ...
}

pub struct LexicalEntry {
    pub lemma: String,
    pub note: Option<String>,
    pub value: Option<f32>,
    pub kind: Option<String>,
    pub anchor: Option<String>,
    pub scope: Option<String>,
    pub total: Option<bool>,
}
```

---

## Lexicon

Indexe toutes les taxonomies chargées pour des lookups rapides lemme+POS → classe causale.

```rust
use gcn_knowledge::lexicon::Lexicon;
use std::path::Path;

let lexicon = Lexicon::load_from_dir(Path::new("taxonomies/fr/"))?;

// Lookup par lemme (verbes uniquement)
if let Some(class) = lexicon.lookup_verb("provoquer") {
    println!("{:?}", class.causal_direction); // Some("production")
}

// Lookup par POS + lemme → (nom_classe, entrée)
// POS mappings : VERB→verbes, NOUN→noms, DET→determinants, PRON→pronoms
if let Some((class_name, entry)) = lexicon.lookup_by_pos("DET", "tous") {
    println!("{} → {:?}", class_name, entry.scope); // "universel" → Some("universal")
}

// Accès direct à une taxonomie nommée
if let Some(tax) = lexicon.taxonomy("verbes") {
    println!("{} classes", tax.classes.len());
}

// Toutes les taxonomies chargées
let all: &HashMap<String, Taxonomy> = lexicon.taxonomies();
```

---

## Moteur d'inférence

Enrichit un `CausalIR` en 3 passes sans re-emprunts conflictuels.

```rust
use gcn_knowledge::inference::InferenceEngine;
use std::path::Path;

// Depuis un répertoire de taxonomies
let engine = InferenceEngine::from_dir(Path::new("taxonomies/fr/"))?;

// Ou depuis un Lexicon déjà chargé
let engine = InferenceEngine::new(lexicon);

// Enrichissement (mutate l'IR en place)
let notes = engine.enrich(&mut ir);
for note in &notes {
    match note {
        InferenceNote::NodeTypeResolved { node_id, from, to, rule } =>
            println!("Nœud {:?} : {:?} → {:?} (règle: {})", node_id, from, to, rule),
        InferenceNote::ConfidenceAdjusted { from_id, to_id, from, to, rule } =>
            println!("Arête {:?}→{:?} : {:.2} → {:.2}", from_id, to_id, from, to),
        InferenceNote::CausalGapSignaled { from_id, to_id, relation } =>
            println!("Lacune causale : {:?}", relation),
    }
}
```

### Méthodes de l'`InferenceEngine`

```rust
// Inférer le NodeType depuis lemme + POS
engine.infer_node_type("provoquer", "VERB") -> Option<NodeType>
// Retourne None pour les classes non-causales (auxiliaires, etc.)

// Inférer la portée depuis un déterminant/pronom
engine.infer_scope("tous", "DET") -> Option<Scope>
// Retourne Some(Scope::Universal), Some(Scope::Existential), etc.

// Score de confiance structurel [0.1, 1.0]
engine.score_confidence(NodeType::Action, NodeType::Etat, RelationType::Cause, true)
// explicit=true → base 1.0, ajusté ±0.15/±0.10 selon la triple (from, to, rel)
// explicit=false → base 0.5
```

### Ce que `enrich()` fait

1. **Résolution des nœuds inférés** : les nœuds `origin=Inferred` sans type explicite sont reclassifiés via lookup NOUN de `attributes.entity`
2. **Snapshot des types** : capture l'état après résolution pour les calculs de confiance
3. **Recalibration de confiance** : les arêtes implicites (`explicit=false`) reçoivent un score structurel ; les arêtes `Concession`/`Opposition` sont signalées comme lacunes causales

---

## Erreurs

```rust
pub enum KnowledgeError {
    Io(std::io::Error, PathBuf),      // lecture échouée
    Yaml(yaml_serde::Error, PathBuf), // YAML mal formé
}
```

---

## Licence

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
