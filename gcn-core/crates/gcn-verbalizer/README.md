# gcn-verbalizer — Pont Rust vers le Décodeur Python

[![crates.io](https://img.shields.io/crates/v/gcn-verbalizer)](https://crates.io/crates/gcn-verbalizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pont Rust du moteur **GCN-Core** vers le décodeur de verbalisation Python. Sérialise un `CausalIR` en JSON et le pipe au sous-processus `gcn-verbalize`, retournant la surface textuelle produite par le modèle entraîné.

---

## Rôle dans GCN-Core

```
CausalIR  (produit par le pipeline Rust)
            │
     gcn-verbalizer
     ├── serde_json::to_string(ir)
     ├── subprocess "gcn-verbalize -" (stdin)
     └── stdout → String (surface texte)
            │
            ▼
  "La pluie provoque des inondations."
```

Le décodeur Python (`gcn-python` — `TrainableDecoder`) est complètement indépendant du pont Rust. Le format de sortie dépend entièrement des données d'entraînement — `gcn-verbalizer` n'impose aucun format.

---

## Prérequis

Le binaire `gcn-verbalize` doit être installé et accessible dans le `PATH` :

```bash
pip install gcn-python
# ou
pip install -e path/to/gcn-python
```

---

## Dépendance

```toml
[dependencies]
gcn-verbalizer = "1.0"
```

---

## Utilisation

```rust
use gcn_verbalizer::decode;

// Verbalise un CausalIR via le décodeur Python
let surface = decode(&ir)?;
println!("{}", surface);
// "La pluie provoque des inondations." (selon l'entraînement)
```

---

## API publique

### `decode`

```rust
pub fn decode(ir: &CausalIR) -> Result<String, VerbalizerError>
```

1. Sérialise `ir` en JSON (`serde_json::to_string`)
2. Lance `gcn-verbalize -` en sous-processus
3. Écrit le JSON sur stdin
4. Lit stdout (trimmed) et le retourne

### `VerbalizerError`

```rust
pub enum VerbalizerError {
    Serialization(serde_json::Error),
    // Échec de sérialisation du CausalIR

    ProcessSpawn(std::io::Error),
    // Impossible de lancer le sous-processus gcn-verbalize

    DecoderError(String),
    // Le sous-processus a retourné un code non-zéro ; contient stderr

    Utf8(std::string::FromUtf8Error),
    // UTF-8 invalide dans stdout du sous-processus
}
```

---

## Pipeline complet

```rust
use gcn_frontend_fr::FrenchParser;
use gcn_middleend::process;
use gcn_verbalizer::decode;
use std::path::Path;

let parser = FrenchParser::new(Path::new("gcn-references/taxonomies"))?;
let ir = parser.parse("Si les ventes baissent, on réduit les coûts.")?;

let result = process(ir)?;  // middleend
let surface = decode(&result.ir)?;
println!("{}", surface);
```

---

## Note architecturale

`gcn-verbalizer` est un **pont de processus**, pas une intégration FFI. Cette conception :
- Isole les dépendances Python (NumPy, PyTorch) du binaire Rust
- Permet de remplacer le décodeur sans recompiler le moteur
- Le comportement du décodeur dépend entièrement de l'entraînement sur `gcn-datasets/`

---

## Licence

MIT — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
