# gcn-frontend-code — Code Source → CausalIR

[![crates.io](https://img.shields.io/crates/v/gcn-frontend-code)](https://crates.io/crates/gcn-frontend-code)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Frontend code source du moteur **GCN-Core**. Analyse l'AST de code Python, Rust ou JavaScript via tree-sitter et produit un `CausalIR` structurellement isomorphe à celui produit par les frontends texte.

---

## Rôle dans GCN-Core

```
if x < y:
    reduce(z)
        │
 gcn-frontend-code
 ├── tree-sitter (Python/Rust/JS)
 ├── CodeResources  ← YAML AST taxonomies
 ├── walk_block     ← parcours AST → CausalNode + CausalEdge
 └── CausalIR { source_lang: Programming { lang: Python } }
        │
        ▼
 Isomorphe avec FrenchParser sur "Si x < y, on réduit z"
```

---

## Dépendance

```toml
[dependencies]
gcn-frontend-code = "1.0"
```

---

## Utilisation

```rust
use gcn_frontend_code::{parse_python, parse_rust, parse_js};
use std::path::Path;

let taxonomies = Path::new("gcn-references/taxonomies");

// Python
let ir = parse_python("if x < y:\n    reduce(z)", taxonomies)?;

// Rust
let ir = parse_rust("if x < y { reduce(z); }", taxonomies)?;

// JavaScript
let ir = parse_js("if (x < y) { reduce(z); }", taxonomies)?;

println!("{} nœuds, {} arêtes", ir.node_count(), ir.edge_count());
```

### Isomorphisme fr ↔ code

```rust
// Les deux produisent : 2 nœuds, 1 arête Condition
let ir_fr   = fr_parser.parse("Si x est inférieur à y, on réduit z.")?;
let ir_code = parse_python("if x < y:\n    reduce(z)", &taxonomies)?;

assert_eq!(ir_fr.node_count(), ir_code.node_count());
// edges[0].2.relation == RelationType::Condition dans les deux cas
```

---

## API publique

```rust
pub fn parse_python(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError>
pub fn parse_rust(source: &str, taxonomies_root: &Path)   -> Result<CausalIR, CodeParserError>
pub fn parse_js(source: &str, taxonomies_root: &Path)     -> Result<CausalIR, CodeParserError>
```

Chaque fonction :
1. Charge les ressources AST depuis `taxonomies_root/{lang}/{lang}_ast.yaml`
2. Parse `source` via tree-sitter
3. Parcourt l'AST et émet `CausalNode` + `CausalEdge`
4. Retourne un `CausalIR` avec `SourceLanguage::Programming`

### Fichiers YAML requis

| Langage | Fichier taxonomy |
|---------|-----------------|
| Python | `taxonomies/python/python_ast.yaml` |
| Rust | `taxonomies/rust/rust_ast.yaml` |
| JavaScript | `taxonomies/js/js_ast.yaml` |

Ces fichiers mappent les types de nœuds AST tree-sitter → `NodeType`, `RelationType`, et `LabelStrategy`.

### Erreurs

```rust
pub enum CodeParserError {
    Language(String), // erreur tree-sitter ou lecture YAML
    ParseFailed,      // source non parseable par tree-sitter
}
```

---

## Structure produite

Pour `if x < y: reduce(z)` en Python :

```
CausalIR {
  source_lang: Programming { lang: Python },
  nodes: [
    CausalNode { node_type: Condition, label: "x < y", ... },
    CausalNode { node_type: Action,    label: "reduce(z)", ... },
  ],
  edges: [
    (NodeId(0), NodeId(1), CausalEdge { relation: Condition, confidence: 1.0, ... })
  ]
}
```

Les arêtes séquentielles entre instructions consécutives utilisent `RelationType::Sequence`.

---

## LabelStrategy

Le label de chaque nœud AST est extrait selon la stratégie définie dans le YAML :

| Stratégie | Extraction |
|-----------|-----------|
| `FullText` (défaut) | Texte source complet, tronqué à 64 chars |
| `ConditionField` | Champ tree-sitter `"condition"` (ex. `x < y` depuis `if_statement`) |
| `NameField` | Champ tree-sitter `"name"` (ex. nom de fonction) |

---

## Licence

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
