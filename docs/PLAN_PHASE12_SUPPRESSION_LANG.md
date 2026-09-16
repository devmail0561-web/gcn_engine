# Plan Phase 12 — Suppression du paramètre `lang` (v2)

## Contexte

Le paramètre `lang` est un artifact de conception. Le réseau de neurones (MLP, R-GCN, GAT, embeddings, loss, backward) est entièrement agnostique à la langue. `lang` n'a d'effet computationnel qu'à un seul endroit : `label_builder.py` (labels textuels des nœuds dans le CIR).

**Objectif précis :** retirer `lang` du *pipeline de calcul* pour qu'un dataset mixte (FR, EN, code, maths) s'entraîne sans aucun paramètre linguistique. `SentenceRecord.lang` est conservé comme métadonnée de dataset (compromis de rétrocompatibilité documenté).

**Rupture d'API :** `CGNPipeline.__init__` perd son 3e argument positionnel `lang`. Version bump requis : **v1.2.0 → v2.0.0**.

---

## Ce que `lang` fait réellement aujourd'hui

### Usage computationnel réel (3 lignes dans `label_builder.py` uniquement)

| Fichier | Ligne | Code | Impact |
|---|---|---|---|
| `label_builder.py` | 27 | `_nominalize(rep.root_lemma, rep.lang, ...)` | Chercher dans tous les sous-répertoires |
| `label_builder.py` | 30 | `"hidden_cause(?)" if rep.lang == "en" else "cause_cachée(?)"` | `"hidden_cause(?)"` universel |
| `label_builder.py` | 98 | `"examples_fr" if lang == "fr" else "examples"` | Essayer les deux clés |

Ces 3 lignes affectent uniquement les **labels textuels** du CIR. Elles n'affectent ni les features ML, ni la loss, ni les poids.

### Non modifié dans ce plan — `_SCOPE_HINTS` dans `cgnp.py` (lignes 704-709)

```python
_SCOPE_HINTS = {
    "tous": "universal", "toutes": "universal", ...  # mots français uniquement
}
```

Ce dictionnaire est français-only et hardcodé. La docstring de `_infer_scope()` dit "Support multilingue à ajouter en phase 10". Ce comportement est **inchangé** dans Phase 12 : pour un dataset non-français, `_infer_scope()` retourne toujours `"specific"` (valeur par défaut correcte). Ceci est documenté comme **limitation connue**, pas comme bug.

### Métadonnée pure — `ir_emitter.py` ligne 63

`"source_lang": {"natural": {"lang": lang}}` — jamais lu par le backend Rust ni par aucun code source interne (grep confirmé : seul `ir_emitter.py` produit ce champ). Changé en `{"natural": {"lang": "und"}}` (ISO 639-3 undetermined) pour préserver la structure dict et éviter un `TypeError` sur des consommateurs externes qui feraient `cir["source_lang"]["natural"]["lang"]`.

### Compromis explicite — `SentenceRecord.lang`

`SentenceRecord.lang` est rendu optionnel (`= ""`), **pas supprimé**. C'est un compromis délibéré : les JSON existants avec `"lang": "fr"` restent valides sans migration. Le champ n'est plus utilisé dans les calculs, mais persiste comme métadonnée de provenance. Ce n'est pas une suppression complète — c'est un retrait du pipeline de calcul.

---

## Précision Rust

Le backend Rust ne parse jamais le CIR JSON produit par Python. `NaturalLanguage` enum Rust : `French`, `Wolof`, `Arabic`, `English`. Le côté Rust est **inchangé**.

---

## Modifications — fichiers sources Python

### 1. `layer1/representation.py` — Supprimer le champ `lang`

```python
# AVANT (ligne 21) :
lang: str

# APRÈS : champ supprimé
```

Toute construction `UDRepresentation(lang=...)` dans `loader.py`, `bridge.py` et les tests doit être mise à jour.

### 2. `layer0/interface.py` — Supprimer `lang` du Protocol

```python
# AVANT :
def parse(self, text: str, lang: str = "fr") -> tuple[...]: ...
# APRÈS :
def parse(self, text: str) -> tuple[...]: ...
```

### 3. `data/schema.py` — Rendre `lang` optionnel

```python
# AVANT :
lang: str
# APRÈS :
lang: str = ""   # conservé comme métadonnée, non utilisé en calcul
```

### 4. `data/json_reader.py` — Supprimer `lang` des signatures

```python
# AVANT :
def load_sentences(path: Path, lang: str = "fr") -> list[SentenceRecord]:
    doc_lang = doc["document"].get("lang", lang)
# APRÈS :
def load_sentences(path: Path) -> list[SentenceRecord]:
    doc_lang = doc["document"].get("lang", "")
```

Même changement pour `load_all_sentences`, `_parse_paper_example`, `_parse_dataset_sentence`. `SentenceRecord` toujours construit avec `lang=doc_lang` pour conserver la métadonnée JSON.

### 5. `data/loader.py` — Supprimer `lang` du constructeur et des helpers

```python
# AVANT :
def __init__(self, data_dir: Path, lang: str = "fr", repeat: bool = False, all_pairs: bool = False):
    self.lang = lang
    self._records = load_all_sentences(data_dir, lang)
# APRÈS :
def __init__(self, data_dir: Path, repeat: bool = False, all_pairs: bool = False):
    self._records = load_all_sentences(data_dir)
```

**Code corrigé pour `_rep_from_clause` et `_connector_between`** — les deux fonctions perdent leur paramètre `lang` et ne passent plus `lang=lang` à `UDRepresentation` (champ supprimé) :

```python
# AVANT :
def _rep_from_clause(clause: ClauseRecord, all_tokens: list[TokenRecord], lang: str) -> UDRepresentation | None:
    ...
    return UDRepresentation(
        ...,
        lang=lang,   # ← supprimé
    )

def _connector_between(clause_a, clause_b, all_tokens, lang: str) -> UDRepresentation | None:
    ...
    return UDRepresentation(
        ...,
        lang=lang,   # ← supprimé
    )

# APRÈS : signature sans lang, pas de lang= dans le constructeur
def _rep_from_clause(clause: ClauseRecord, all_tokens: list[TokenRecord]) -> UDRepresentation | None:
    ...
    return UDRepresentation(
        ...
        # lang= retiré — champ n'existe plus dans UDRepresentation
    )
```

Les appels `_rep_from_clause(clause, rec.tokens, rec.lang)` et `_connector_between(..., rec.lang)` aux lignes 145 et 154 deviennent `_rep_from_clause(clause, rec.tokens)` et `_connector_between(..., rec.lang)` avec suppression du dernier argument.

### 6. `pipeline/label_builder.py` — Universaliser + corriger `_load_nominalizations`

**Ligne 27 :**
```python
# AVANT :
nom = _nominalize(rep.root_lemma, rep.lang, taxonomies_dir)
# APRÈS :
nom = _nominalize(rep.root_lemma, taxonomies_dir)
```

**Ligne 30 :**
```python
# AVANT :
label = "hidden_cause(?)" if rep.lang == "en" else "cause_cachée(?)"
# APRÈS :
label = "hidden_cause(?)"
```

**Lignes 81-103 — refactoring complet sans `break` :**

```python
def _nominalize(lemma: str, taxonomies_dir: Path | None) -> str:
    if taxonomies_dir is None:
        return lemma
    cache_key = str(taxonomies_dir)
    if cache_key not in _nom_cache:
        _nom_cache[cache_key] = _load_nominalizations(taxonomies_dir)
    return _nom_cache[cache_key].get(lemma, lemma)


def _load_nominalizations(taxonomies_dir: Path) -> dict[str, str]:
    """Charge et fusionne les nominalizations de TOUS les sous-répertoires disponibles."""
    table: dict[str, str] = {}
    dirs_to_scan = [taxonomies_dir]
    if taxonomies_dir.is_dir():
        dirs_to_scan += sorted(d for d in taxonomies_dir.iterdir() if d.is_dir())
    for search_dir in dirs_to_scan:
        nom_path = search_dir / "nominalizations.json"
        if not nom_path.exists():
            continue
        with open(nom_path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            continue
        for _cls, cls_data in (doc.get("classes") or {}).items():
            for key in ("examples_fr", "examples"):   # essayer les deux clés
                for entry in (cls_data or {}).get(key) or []:
                    if isinstance(entry, dict) and "lemma" in entry and "note" in entry:
                        table.setdefault(entry["lemma"].lower(), entry["note"])
        # PAS DE break — tous les répertoires sont scannés pour la fusion multi-langues
    return table
```

**Cache key** : passe de `(str(taxonomies_dir), lang)` à `str(taxonomies_dir)`.

### 7. `pipeline/ir_emitter.py` — Émettre `"und"` au lieu de `lang`

```python
# AVANT :
def emit(text: str, lang: str, node_types: ...) -> dict:
    ...
    "source_lang": {"natural": {"lang": lang}},

# APRÈS :
def emit(text: str, node_types: ...) -> dict:
    ...
    "source_lang": {"natural": {"lang": "und"}},
```

`"und"` = ISO 639-3 undetermined. Préserve la structure dict — les consommateurs qui accèdent à `cir["source_lang"]["natural"]["lang"]` reçoivent `"und"` au lieu d'un `TypeError`. Le Rust ne parse pas ce champ.

### 8. `pipeline/cgnp.py` — Supprimer `lang`

```python
# AVANT :
def __init__(self, encoder, graph, lang: str, vocabulary, *, ...):
    self.lang = lang
# APRÈS :
def __init__(self, encoder, graph, vocabulary, *, ...):
    # self.lang supprimé
```

- Lignes 162, 282 : `emit(text, self.lang, ...)` → `emit(text, ...)`
- Ligne 337 : `text_parser.parse(text, self.lang)` → `text_parser.parse(text)`
- Ligne 362 : `_cir_to_reps_and_connectors(cir, self.lang)` → `_cir_to_reps_and_connectors(cir)`

### 9. `frontend/bridge.py` — Supprimer `lang`

| Fonction | Changement |
|---|---|
| `_rep_from_cir_node(node, lang)` | Supprimer `lang`, retirer `lang=lang` de `UDRepresentation` |
| `_build_connector_rep(marker_token_id, lang)` | Supprimer `lang` |
| `_cir_to_reps_and_connectors(cir, lang)` | Supprimer `lang` |
| `GCNBridgeParser.parse(text, lang="fr")` | `parse(text)` |
| `reps_from_text(text, lang="fr")` | `reps_from_text(text)` |

### 10. CLIs — Supprimer `--lang`

| Fichier | Changement |
|---|---|
| `training/train.py` | Supprimer `--lang`, retirer `lang=lang` de `CGNPipeline` et `GCNDataLoader` |
| `training/bootstrap.py` | Supprimer `--lang`, supprimer `"lang": lang` du JSON produit (champ absent du document) |
| `evaluation/eval_runner.py` | Supprimer `--lang`, retirer `lang=lang` de `CGNPipeline` et `GCNDataLoader` |
| `pipeline/cli.py` | Supprimer `--lang`, retirer `lang=lang` de `load_sentences` et `CGNPipeline` |

**Impact format bootstrap.py :** les JSON générés par `gcn-bootstrap` n'auront plus de champ `"lang"` dans `document`. `json_reader.py` fait `doc["document"].get("lang", "")` — compatible. Tout outil externe qui lisait `document.lang` recevra `KeyError` ou `""`.

### 11. `taxonomy/loader.py` — Supprimer `lang_code`

```python
# AVANT :
def load(cls, taxonomies_dir: Path, lang_code: str = "fr") -> "TaxonomyIndex":
    lang_dir = taxonomies_dir / lang_code
    if lang_dir.is_dir():
        dirs_to_scan.append(lang_dir)  # seulement fr/ (ou le lang_code demandé)

# APRÈS :
def load(cls, taxonomies_dir: Path) -> "TaxonomyIndex":
    dirs_to_scan = sorted(d for d in taxonomies_dir.iterdir() if d.is_dir())
    dirs_to_scan.append(taxonomies_dir)  # racine en dernier (fallback)
```

**Changement de comportement documenté :** l'ancien `load(lang_code="fr")` ne chargeait que `taxonomies_dir/fr/`. Le nouveau charge **tous** les sous-répertoires. Effet : les lemmes des taxonomies anglaises, wolof, ou code sont maintenant indexés. Conséquence : `membership()` peut retourner `True` pour des lemmes d'autres langues présents dans les fichiers YAML. C'est le comportement attendu pour un dataset mixte.

---

## Modifications — tests (patterns, pas numéros de lignes)

Les numéros de lignes peuvent avoir évolué. Les patterns à mettre à jour :

| Pattern actuel | Pattern corrigé |
|---|---|
| `CGNPipeline(enc, gr, "fr", vocab)` | `CGNPipeline(enc, gr, vocab)` |
| `CGNPipeline(enc, gr, lang="fr", ...)` | `CGNPipeline(enc, gr, ...)` |
| `CGNPipeline(enc2, gr2, pipeline.lang, vocab)` | `CGNPipeline(enc2, gr2, vocab)` — `pipeline.lang` n'existe plus |
| `UDRepresentation(..., lang="fr")` | `UDRepresentation(...)` — champ supprimé |
| `SentenceRecord(..., lang="fr")` | Inchangé — champ optionnel conservé |
| `GCNDataLoader(data_dir, lang="fr")` | `GCNDataLoader(data_dir)` |
| `emit(text, "fr", ...)` | `emit(text, ...)` |
| `assert result["source_lang"] == {"natural": {"lang": "fr"}}` | `assert result["source_lang"] == {"natural": {"lang": "und"}}` |
| `assert rep.lang == "fr"` | Supprimer — champ n'existe plus |
| `reps_from_text(text, lang="fr")` | `reps_from_text(text)` |
| `assert doc["document"]["lang"] == "fr"` | `assert "lang" not in doc["document"]` |

---

## Tableau des fichiers modifiés

| Fichier | Action | Rupture |
|---|---|---|
| `layer1/representation.py` | Supprimer champ `lang` | Oui — constructeurs |
| `layer0/interface.py` | Supprimer `lang` du Protocol | Oui — implémentations TextParser |
| `data/schema.py` | `lang: str` → `lang: str = ""` | Non — optionnel |
| `data/json_reader.py` | Supprimer `lang` des signatures | Oui |
| `data/loader.py` | Supprimer `lang` du constructeur et helpers | Oui |
| `pipeline/label_builder.py` | Universel + fix `break` + fusion multi-dirs | Non (comportement amélioré) |
| `pipeline/ir_emitter.py` | Supprimer `lang`, émettre `"und"` | Oui (changement de valeur) |
| `pipeline/cgnp.py` | Supprimer `lang` du constructeur | **Oui — rupture API** |
| `frontend/bridge.py` | Supprimer `lang` des fonctions | Oui |
| `training/train.py` | Supprimer `--lang` | Oui (CLI) |
| `training/bootstrap.py` | Supprimer `--lang`, retirer `"lang"` du JSON | Oui (format JSON) |
| `evaluation/eval_runner.py` | Supprimer `--lang` | Oui (CLI) |
| `pipeline/cli.py` | Supprimer `--lang` | Oui (CLI) |
| `taxonomy/loader.py` | Supprimer `lang_code`, scanner tous les dirs | Comportement élargi |
| `tests/` (11 fichiers) | ~50 substitutions par patterns | — |

**Rust (`gcn-core/`)** : **inchangé**.

---

## Version bump

Ce plan est un **breaking change** sur l'API publique de `CGNPipeline`. Requis avant merge :
- Bump `v1.2.0 → v2.0.0` dans `pyproject.toml`
- Entrée CHANGELOG.md : "Breaking: CGNPipeline supprime le paramètre lang"
- Note de migration : `CGNPipeline(enc, gr, "fr", vocab)` → `CGNPipeline(enc, gr, vocab)`

---

## Ce qui ne change pas

- Features ML : `FeatureVocabulary`, `vectorize_clause`, `vectorize_edge` — inchangés
- MLP, R-GCN, GAT, WordEmbedding — inchangés
- Loss et backward — inchangés
- Rust backend Pearl / GCN-QL — inchangé
- `_SCOPE_HINTS` dans `cgnp.py` — inchangé (français-only, comportement par défaut `"specific"` correct)

---

## Vérification end-to-end

```bash
# 1. Entraînement sans --lang sur dataset FR
gcn-train --data-dir gcn-datasets/examples/ --epochs 5 --output /tmp/ckpt_nolang.npz
# Attendu : pas d'erreur, loss décroissante

# 2. Entraînement sur dataset mixte FR + code
gcn-train --data-dir gcn-datasets/ --epochs 5 --output /tmp/ckpt_mixed.npz
# Attendu : pas d'erreur, aucun paramètre lang requis

# 3. Tests complets
python -m pytest gcn-python/tests/ -v
# Attendu : 177+ passed, 0 failed

# 4. Non-régression labels
python -c "
from gcn_python.pipeline.label_builder import build_label
from gcn_python.layer1.representation import UDRepresentation
rep = UDRepresentation(tokens=[], root_lemma='chuter', root_pos='VERB',
    root_dep_rel='root', root_morph={}, subject_pos=None,
    has_object=False, has_advcl=False, has_temporal_obl=False, token_span=(0,0))
label, _ = build_label(rep, 'condition')
assert label == 'hidden_cause(?)', f'Attendu hidden_cause(?), obtenu {label}'
print('Labels OK')
"

# 5. eval_runner sans --lang
gcn-eval --data-dir gcn-datasets/examples/ --model /tmp/ckpt_nolang.npz
```

**Métriques de succès :**
- `node_accuracy` identique avant et après (features ML inchangées)
- `result["source_lang"] == {"natural": {"lang": "und"}}` dans les tests ir_emitter
- `"lang"` absent des JSON produits par `gcn-bootstrap`
- Tous les lemmes FR + EN trouvables dans `TaxonomyIndex.load()` sur dataset mixte
- `label == "hidden_cause(?)"` pour tous les nœuds `condition`, toutes langues
