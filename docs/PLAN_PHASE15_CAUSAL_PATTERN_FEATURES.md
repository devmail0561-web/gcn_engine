# Plan — Phase 15 : causal_pattern comme feature d'apprentissage

**Date :** 2026-09-17  
**Statut :** À implémenter

---

## Contexte

**Problème identifié :** le moteur est entraîné sur des phrases dont les tokens sont
annotés `gcn_causal_type` par token. Ces annotations indiquent directement quel token
est "causal" (verbe causal, connecteur…), ce qui pré-mâche le travail du modèle.
Le modèle ne généralise pas — il transcrit une représentation déjà résolue.

**Proposition :** les seuls inputs supervisés doivent être :
- La phrase brute (tokens avec POS / dep_rel / morph standard UD — sans `gcn_causal_type`)
- Le `causal_pattern` au niveau phrase (label global : "cause", "condition", etc.)

**Cibles à prédire (inchangées) :** node types + edge relations (le CIR complet)

**Faits établis par audit du code :**
1. `gcn_causal_type` par token **n'entre jamais dans le réseau de neurones** — il sert
   uniquement à 2 heuristiques de sélection dans `_rep_from_clause` et
   `_connector_between`, qui ont chacune un fallback POS déjà opérationnel.
2. `causal_pattern` est présent dans 100 % du corpus généré (990 phrases) mais
   n'est jamais lu par le moteur — champ ignoré dans `json_reader.py`.
3. Le champ `morph` est présent dans le corpus généré (`generated_1000.json`) mais
   absent des exemples manuels (`fr_causal_basic.json`, `fr_causal_cycles.json`).

**Changements architecturaux :**
L'architecture MLP + R-GCN reste inchangée. Seul `d_clause` grandit de 75 → 86
(+ 11 dimensions pour le one-hot de `causal_pattern` sur `RELATION_TYPES`).

---

## Changement 1 — Lire `causal_pattern` dans le pipeline de données

### `data/schema.py`
Ajouter un champ à `SentenceRecord` :
```python
causal_pattern: str = ""   # ex: "cause", "condition", "motivation", ""
```

### `data/json_reader.py`
Dans `_parse_dataset_sentence`, lire et stocker le champ :
```python
causal_pattern=s.get("causal_pattern", ""),
```

### `data/loader.py` — `reps_from_sentence`
Propager `causal_pattern` à chaque `UDRepresentation` produite :
```python
rep.causal_pattern = rec.causal_pattern   # après construction du rep
```

---

## Changement 2 — Retirer `gcn_causal_type` comme critère de sélection

### `data/loader.py` — `_rep_from_clause` (lignes 209–214 actuelles)
Supprimer la priorité `gcn_causal_type == "verbe"` — utiliser directement le fallback POS :
```python
# AVANT :
root_tok = (
    next((t for t in span_toks
          if t.gcn_causal_type == "verbe" and t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
    or span_toks[0]
)

# APRÈS :
root_tok = (
    next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
    or span_toks[0]
)
```

### `data/loader.py` — `_connector_between` (lignes 172–175 actuelles)
Supprimer la priorité `gcn_causal_type == "conjonction"` :
```python
# AVANT :
tok = (
    next((t for t in gap_toks if t.gcn_causal_type == "conjonction"), None)
    or next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
)

# APRÈS :
tok = next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
```

---

## Changement 3 — Ajouter `causal_pattern` à `UDRepresentation`

### `layer1/representation.py`
Ajouter un champ avec valeur par défaut vide :
```python
causal_pattern: str = ""
```

### `constants.py`
Pas de nouvelle constante — réutiliser `RELATION_TYPES` (11 valeurs).  
Justification : le `causal_pattern` d'une phrase correspond à la relation principale
entre ses clauses — même ontologie. `causal_pattern` absent (chaîne vide) → vecteur
nul, représentation neutre distincte de toute relation connue.

---

## Changement 4 — Vectoriser `causal_pattern` dans la Couche 1

### `layer1/features.py` — `FeatureVocabulary`
Ajouter le champ `relation_types` (synchronisé avec `RELATION_TYPES`) et mettre à jour `d_clause` :
```python
from ..constants import RELATION_TYPES   # ajout import si absent

relation_types: list[str] = field(default_factory=lambda: list(RELATION_TYPES))

@property
def d_clause(self) -> int:
    return (
        len(self.upos_tags)            # 19
        + len(self.dep_rels)           # 38
        + len(self.subject_pos_cats)   #  5
        + len(self.tense_values)       #  5
        + len(self.aspect_values)      #  4
        + len(self.mood_values)        #  5
        + 1                            #  1  polarity
        + 3                            #  3  structural flags
        + len(self.relation_types)     # 11  causal_pattern one-hot  ← NOUVEAU
    )
    # Total : 75 → 86
```

### `layer1/features.py` — `vectorize_clause`
Concaténer le one-hot du `causal_pattern` à la fin du vecteur clause :
```python
parts.append(_one_hot(rep.causal_pattern or "", vocab.relation_types))
# causal_pattern absent (chaîne vide) → vecteur nul
# causal_pattern présent ("cause", "condition"…) → one-hot sur RELATION_TYPES
```

### `layer1/features.py` — `to_json` / `from_json`
Inclure `relation_types` dans la sérialisation / désérialisation du vocabulaire.

---

## Conséquences en cascade (d_clause : 75 → 86)

| Composant | Impact |
|-----------|--------|
| `MLPEncoder(d_clause=86)` | Première couche 86→128 au lieu de 75→128 |
| `d_effective = vocab.d_clause + d_emb` | Grandit automatiquement |
| `vocab.d_edge_closed_loop(d_effective, ...)` | Recalculé automatiquement |
| `RGCNLayer(d_in=86, d_out=86)` | Matrices W_r, W_0 plus grandes |
| **Checkpoints existants** | **Invalidés — incompatibilité de dimension** |

Les tests qui instancient `MLPEncoder(d_clause=vocab.d_clause, ...)` se mettent à
jour automatiquement. Seuls les tests qui hardcodent `75` devront être corrigés.

---

## Changement 5 — Datasets existants sans `causal_pattern`

Les datasets `fr_causal_basic.json` (3 phrases) et `fr_causal_cycles.json` (4 phrases)
n'ont pas de champ `causal_pattern`.

**Option retenue :** ajouter `causal_pattern` manuellement dans les JSON.

| Fichier | Phrases | `causal_pattern` à ajouter |
|---------|---------|---------------------------|
| `fr_causal_basic.json` | s001 | "condition" |
| `fr_causal_basic.json` | s002 | "cause" |
| `fr_causal_basic.json` | s003 | "motivation" |
| `fr_causal_cycles.json` | s001–s004 | "" (causalité inter-phrases, pas intra) |

L'absence de `causal_pattern` est gérée par défaut (`""`) → vecteur nul.

---

## Changement 6 — Mise à jour des tests

Les tests hardcodant `d_clause == 75` doivent être mis à jour vers `86`.
Les tests utilisant `vocab.d_clause` directement sont automatiquement corrects.

Fichiers impactés :
- `gcn-python/tests/test_layer2.py`
- `gcn-python/tests/test_layer1.py` (si tests de `d_clause`)
- `gcn-python/tests/test_pipeline.py`
- `gcn-python/tests/test_checkpoint.py`
- `gcn-python/tests/test_trainable_decoder.py`
- `gcn-python/tests/test_frontend_bridge.py`
- `gcn-python/tests/test_training.py`

---

## Ce qui NE change PAS

- Architecture MLP / R-GCN / GAT : inchangée
- Format des JSON datasets : rétrocompatible (`causal_pattern` absent = `""`)
- `gold_node_labels` et `edge_map` dans `TrainingSample` : inchangés
- Interface `forward()`, `loss()`, `backward()` du pipeline : inchangée
- CLIs `gcn-train`, `gcn-forward`, `gcn-eval` : inchangés

---

## Ordre d'exécution

```
1. data/schema.py         — SentenceRecord.causal_pattern
2. data/json_reader.py    — lecture causal_pattern depuis JSON
3. layer1/representation.py — UDRepresentation.causal_pattern
4. layer1/features.py     — FeatureVocabulary.relation_types + d_clause + vectorize_clause
5. data/loader.py         — propagation causal_pattern + suppression gcn_causal_type
6. gcn-datasets/examples/ — ajout causal_pattern dans fr_causal_basic.json
7. tests/                 — mise à jour d_clause 75 → 86
```

---

## Vérification

```bash
# Suite complète — doit rester ≥195 passed
python -m pytest gcn-python/tests/ -x -q

# Vérification dimension
python -c "
from gcn_python.layer1.features import FeatureVocabulary
v = FeatureVocabulary()
assert v.d_clause == 86, f'Attendu 86, obtenu {v.d_clause}'
print('d_clause OK:', v.d_clause)
"

# Vérification propagation causal_pattern
python -c "
from gcn_python.data.schema import SentenceRecord
s = SentenceRecord(id='test', text='', tokens=[], clauses=[], edges=[])
assert s.causal_pattern == ''
print('causal_pattern par défaut OK')
"

# Entraînement de validation sur le corpus existant
gcn-train --data-dir gcn-datasets/corpus/ --epochs 10 --embedding-dim 50
```
