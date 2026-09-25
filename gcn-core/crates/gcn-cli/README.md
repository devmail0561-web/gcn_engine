# gcn-cli — Interface en Ligne de Commande GCN-Core

Version: 2.5.0

[![Crates.io](https://img.shields.io/crates/v/gcn-cli)](https://crates.io/crates/gcn-cli)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Binaire `gcn` — point d'entrée CLI du moteur **GCN-Core**. Expose les capacités du moteur : analyse causale, requêtes GCN-QL, export de graphe.

---

## Installation

```bash
# Depuis le dépôt
git clone https://github.com/devmail0561-web/gcn_engine.git
cd gcn_engine
make install          # build release + pip install gcn-python

# Ou build seul
cd gcn-core
cargo build --release
# binaire dans target/release/gcn
```

---

## Commandes

### `gcn analyze` — Analyse causale symbolique

Parse du texte français via le frontend symbolique `FrenchParser`, exécute le middle-end, et retourne le `CausalIR`.

```bash
gcn analyze "Si les ventes baissent, on réduit les coûts." \
  --data-dir gcn-references/taxonomies/ \
  --format json

gcn analyze "La pluie provoque des inondations." \
  --data-dir gcn-references/taxonomies/ \
  --format dot \
  --diagnostics
```

| Argument/Option | Type | Défaut | Description |
|---|---|---|---|
| `<text>` | positional | requis | Texte français à analyser |
| `--data-dir <PATH>` | `PathBuf` (env: `GCN_TAXONOMY_DIR`) | requis | Répertoire des taxonomies |
| `--format <FORMAT>` | `json` \| `dot` | `json` | Format de sortie |
| `--diagnostics` | flag | off | Affiche les diagnostics middleend sur stderr |

Sortie `--format json` : `CausalIR` JSON complet  
Sortie `--format dot` : graphe Graphviz (rendu avec `dot -Tpng`)

---

### `gcn query` — Raisonnement GCN-QL

Interroge un fichier `CausalIR` JSON avec le langage GCN-QL.

```bash
# Pearl niveau 1 — observation
gcn query "WHY inondations?" --ir graph.json
gcn query "WHAT pluie?"      --ir graph.json
gcn query "CHAIN pluie -> inondations" --ir graph.json

# Cycles et lacunes
gcn query "CYCLES" --ir graph.json
gcn query "GAPS"   --ir graph.json

# Pearl niveau 2 — intervention do-calculus
gcn query "DO pluie" --ir graph.json

# Pearl niveau 3 — contrefactuel
gcn query "COUNTERFACTUAL pluie?" --ir graph.json
```

| Argument/Option | Type | Défaut | Description |
|---|---|---|---|
| `<query>` | positional | requis | Requête GCN-QL |
| `--ir <PATH>` | `PathBuf` | requis | Fichier CausalIR JSON |

Sortie : JSON sérialisé du `QueryResult`.

**Syntaxe GCN-QL complète :**

| Requête | Niveau | Description |
|---------|--------|-------------|
| `WHY <label>` | Pearl 1 | Causes (BFS inverse) |
| `WHAT <label>` | Pearl 1 | Effets (BFS avant) |
| `CHAIN <a> -> <b>` | Pearl 1 | Chemin causal |
| `CYCLES` | — | Cycles de rétroaction |
| `GAPS` | — | Lacunes temporelles causales |
| `DO <label>` | Pearl 2 | Intervention (do-calculus) |
| `COUNTERFACTUAL <label>` | Pearl 3 | "Et si X n'avait pas eu lieu ?" |

---

### `gcn export` — Conversion de format

Convertit un fichier `CausalIR` JSON en un autre format.

```bash
gcn export --ir graph.json --format dot > graph.dot
dot -Tpng graph.dot -o graph.png
```

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--ir <PATH>` | `PathBuf` | requis | Fichier CausalIR JSON |
| `--format <FORMAT>` | `json` \| `dot` | `dot` | Format cible |

---

### `gcn forward` — commande retirée

**`gcn forward` n'existe plus.** Le binaire `gcn-forward` a été supprimé de `gcn-python` (redondant avec `gcn-discuss` et l'API Python). Toute tentative retourne une erreur explicite :

```text
gcn forward a été retiré : le binaire `gcn-forward` n'existe plus.
Utilisez l'API Python (GCNEngine.from_pretrained(..., trusted=True)) ou `gcn-discuss`
pour l'inférence ML.
```

Équivalents réels :

```python
from gcn_python import GCNEngine

engine = GCNEngine.from_pretrained("model.npz", trusted=True)
cir = engine.analyze("La pluie cause des inondations.")
```

```bash
gcn-discuss --checkpoint model.npz
```

---

## Workflow complet

```bash
# 1. Analyser du texte
gcn analyze "Si les ventes baissent, on réduit les coûts." \
  --data-dir gcn-references/taxonomies/ > graph.json

# 2. Interroger le graphe
gcn query "WHY coûts?" --ir graph.json
gcn query "COUNTERFACTUAL ventes?" --ir graph.json

# 3. Exporter pour visualisation
gcn export --ir graph.json --format dot | dot -Tpng -o graph.png

# 4. Inférence ML : voir la section « gcn forward — commande retirée »
#    (aucune sous-commande gcn pour le pipeline ML)
```

---

## Variables d'environnement

| Variable | Description |
|---|---|
| `GCN_TAXONOMY_DIR` | Chemin par défaut vers `gcn-references/taxonomies/` |

---

## Licence

Apache-2.0 — [github.com/devmail0561-web/gcn_engine](https://github.com/devmail0561-web/gcn_engine)
