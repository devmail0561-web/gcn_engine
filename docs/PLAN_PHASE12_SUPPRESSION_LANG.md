# Plan Phase 12 — Suppression du paramètre `lang`

## Contexte

Le paramètre `lang` est un artifact de conception. Le réseau de neurones (MLP, R-GCN, GAT, embeddings, loss, backward) est déjà entièrement agnostique à la langue. `lang` n'a d'effet computationnel qu'à un seul endroit : `label_builder.py` (labels textuels des nœuds dans le CIR). Partout ailleurs, c'est un paramètre passé en cascade sans effet.

**Objectif :** supprimer `lang` du pipeline de calcul pour qu'un dataset mixte (FR, EN, code, maths) s'entraîne sans aucun paramètre linguistique.

---

## Ce que `lang` fait réellement aujourd'hui

Audit exhaustif — chaque occurrence classée par type d'usage :

### Usage computationnel réel (3 endroits dans `label_builder.py`)

| Fichier | Ligne | Usage | Impact de la suppression |
|---|---|---|---|
| `label_builder.py` | 27 | `_nominalize(rep.root_lemma, rep.lang, ...)` — cherche dans `taxonomies_dir/lang/` | Chercher dans tous les sous-répertoires |
| `label_builder.py` | 30 | `"hidden_cause(?)" if rep.lang == "en" else "cause_cachée(?)"` | Utiliser `"hidden_cause(?)"` universellement |
| `label_builder.py` | 98 | `"examples_fr" if lang == "fr" else "examples"` | Essayer les deux clés |

Ces 3 lignes affectent uniquement les **labels textuels** du CIR (chaînes lisibles par humain). Elles n'affectent ni les features ML, ni la loss, ni les poids du réseau.

### Métadonnée pure (aucun effet computationnel)

| Fichier | Ligne | Usage |
|---|---|---|
| `ir_emitter.py` | 63 | `"source_lang": {"natural": {"lang": lang}}` dans le JSON CIR — jamais lu par le backend Rust |

### Pass-through (aucun calcul, juste propagation)

Tous les autres : `UDRepresentation.lang`, `SentenceRecord.lang`, `CGNPipeline.lang`, `GCNDataLoader.lang`, paramètres CLI `--lang`, fonctions `load_sentences`, `load_all_sentences`, `_rep_from_clause`, `_connector_between`, `bridge.py`, `eval_runner.py`, `pipeline/cli.py`.

---

## Précision sur la compatibilité Rust

Le Rust `NaturalLanguage` enum a 4 variants : `French`, `Wolof`, `Arabic`, `English` (sérialisés `"french"`, `"wolof"`, etc.). Le Python émet actuellement `"lang": "fr"` (code ISO, pas le nom complet) — il y a déjà une incohérence de format. **Le backend Rust ne parse jamais le CIR JSON produit par Python** : il construit son propre IR en mémoire. La modification du champ `source_lang` dans `ir_emitter.py` n'a aucun impact Rust.

**Le côté Rust n'est pas modifié.**

---

## Modifications — fichiers sources Python

### 1. `layer1/representation.py` — Supprimer le champ `lang`

```python
# AVANT (ligne 21) :
lang: str

# APRÈS : champ supprimé entièrement
```

**Conséquence** : toute construction de `UDRepresentation` avec `lang=...` doit être mise à jour (tests + `loader.py` + `bridge.py`).

### 2. `layer0/interface.py` — Supprimer `lang` du Protocol

```python
# AVANT (ligne 25) :
def parse(self, text: str, lang: str = "fr") -> tuple[...]: ...

# APRÈS :
def parse(self, text: str) -> tuple[...]: ...
```

### 3. `data/schema.py` — Rendre `lang` optionnel (non-breaking)

```python
# AVANT (ligne 47) :
lang: str

# APRÈS :
lang: str = ""
```

**Raison** : les JSON existants contenant `"lang": "fr"` sont toujours parsés sans erreur. Le champ reste dans le schéma pour la rétrocompatibilité des datasets annotés. Il n'est plus utilisé dans les calculs.

### 4. `data/json_reader.py` — Supprimer `lang` des signatures

```python
# AVANT :
def load_sentences(path: Path, lang: str = "fr") -> list[SentenceRecord]:
    doc_lang = doc["document"].get("lang", lang)

# APRÈS :
def load_sentences(path: Path) -> list[SentenceRecord]:
    doc_lang = doc["document"].get("lang", "")
```

Même changement pour `load_all_sentences`, `_parse_paper_example`, `_parse_dataset_sentence` — supprimer le paramètre `lang` de leurs signatures. `SentenceRecord` est toujours construit avec `lang=doc_lang` (pour conserver la métadonnée lue depuis le JSON).

### 5. `data/loader.py` — Supprimer `lang` du constructeur et des helpers

```python
# AVANT (ligne 41) :
def __init__(self, data_dir: Path, lang: str = "fr", repeat: bool = False, all_pairs: bool = False):
    self.lang = lang
    self._records = load_all_sentences(data_dir, lang)

# APRÈS :
def __init__(self, data_dir: Path, repeat: bool = False, all_pairs: bool = False):
    self._records = load_all_sentences(data_dir)
```

Supprimer `lang` des signatures de `_connector_between` (ligne 165) et `_rep_from_clause` (ligne 197). Ces fonctions construisaient `UDRepresentation(lang=lang)` — le champ n'existe plus. Supprimer les passages `rec.lang` aux lignes 145 et 154.

### 6. `pipeline/label_builder.py` — Universaliser les labels

**Ligne 27** — supprimer `rep.lang` de l'appel :
```python
# AVANT :
nom = _nominalize(rep.root_lemma, rep.lang, taxonomies_dir)
# APRÈS :
nom = _nominalize(rep.root_lemma, taxonomies_dir)
```

**Ligne 30** — label universel :
```python
# AVANT :
label = "hidden_cause(?)" if rep.lang == "en" else "cause_cachée(?)"
# APRÈS :
label = "hidden_cause(?)"
```

**Lignes 81-103** — refactoring de `_nominalize` et `_load_nominalizations` :

```python
def _nominalize(lemma: str, taxonomies_dir: Path | None) -> str:
    if taxonomies_dir is None:
        return lemma
    cache_key = str(taxonomies_dir)
    if cache_key not in _nom_cache:
        _nom_cache[cache_key] = _load_nominalizations(taxonomies_dir)
    return _nom_cache[cache_key].get(lemma, lemma)


def _load_nominalizations(taxonomies_dir: Path) -> dict[str, str]:
    """Charge et fusionne les nominalizations de tous les sous-répertoires disponibles."""
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
            for key in ("examples_fr", "examples"):  # essayer les deux clés
                for entry in (cls_data or {}).get(key) or []:
                    if isinstance(entry, dict) and "lemma" in entry and "note" in entry:
                        table.setdefault(entry["lemma"].lower(), entry["note"])
        break  # premier fichier trouvé par répertoire
    return table
```

**Cache key** : passe de `(str(taxonomies_dir), lang)` à `str(taxonomies_dir)`.

### 7. `pipeline/ir_emitter.py` — Émettre `null` pour `source_lang`

```python
# AVANT (ligne 6) :
def emit(text: str, lang: str, node_types: ...) -> dict:
    ...
    "source_lang": {"natural": {"lang": lang}},

# APRÈS :
def emit(text: str, node_types: ...) -> dict:
    ...
    "source_lang": None,
```

### 8. `pipeline/cgnp.py` — Supprimer `lang` du constructeur et des appels

```python
# AVANT (lignes 38, 61) :
def __init__(self, encoder, graph, lang: str, vocabulary, *, ...):
    self.lang = lang

# APRÈS :
def __init__(self, encoder, graph, vocabulary, *, ...):
    # self.lang supprimé
```

Lignes 162, 282 : `emit(text, self.lang, ...)` → `emit(text, ...)`

Ligne 337 : `text_parser.parse(text, self.lang)` → `text_parser.parse(text)`

Ligne 362 : `_cir_to_reps_and_connectors(cir, self.lang)` → `_cir_to_reps_and_connectors(cir)`

**RUPTURE D'API** : `CGNPipeline.__init__` perd son 3e argument positionnel `lang`. Tout code qui instancie `CGNPipeline(enc, gr, "fr", vocab)` doit devenir `CGNPipeline(enc, gr, vocab)`.

### 9. `frontend/bridge.py` — Supprimer `lang` des fonctions internes et publiques

| Fonction | Changement |
|---|---|
| `_rep_from_cir_node(node, lang)` | Supprimer `lang`, ne plus passer `lang=lang` à `UDRepresentation` |
| `_build_connector_rep(marker_token_id, lang)` | Supprimer `lang` |
| `_cir_to_reps_and_connectors(cir, lang)` | Supprimer `lang` |
| `GCNBridgeParser.parse(text, lang="fr")` | `parse(text)` |
| `reps_from_text(text, lang="fr")` | `reps_from_text(text)` |

### 10. `training/train.py` — Supprimer `--lang`

```python
# SUPPRIMER :
@click.option("--lang", default="fr", show_default=True)
# et le paramètre lang: str du callback

# Lignes 107, 115 — supprimer lang=lang des constructeurs
```

### 11. `training/bootstrap.py` — Supprimer `--lang`

```python
# SUPPRIMER :
@click.option("--lang", default="fr", show_default=True)

# Ligne 151 — "lang": lang dans le JSON produit → supprimer la clé
# Le document JSON ne contiendra plus "lang" — compatible car SentenceRecord.lang = ""
```

### 12. `evaluation/eval_runner.py` — Supprimer `--lang`

```python
# SUPPRIMER :
@click.option("--lang", default="fr", show_default=True)
# Supprimer lang= de CGNPipeline et GCNDataLoader
```

### 13. `pipeline/cli.py` — Supprimer `--lang`

```python
# SUPPRIMER :
@click.option("--lang", default="fr", show_default=True, help="Code langue (fr, en, …)")
# Supprimer lang= de load_sentences et CGNPipeline
```

### 14. `taxonomy/loader.py` — Supprimer `lang_code`

```python
# AVANT :
def load(cls, taxonomies_dir: Path, lang_code: str = "fr") -> "TaxonomyIndex":
    lang_dir = taxonomies_dir / lang_code
    if lang_dir.is_dir():
        dirs_to_scan.append(lang_dir)

# APRÈS :
def load(cls, taxonomies_dir: Path) -> "TaxonomyIndex":
    # Scanner tous les sous-répertoires disponibles + le répertoire racine
    dirs_to_scan = sorted(d for d in taxonomies_dir.iterdir() if d.is_dir())
    dirs_to_scan.append(taxonomies_dir)  # racine en dernier (fallback)
```

---

## Modifications — tests

Toutes les mises à jour sont mécaniques (substitutions sans logique nouvelle).

| Test | Lignes | Changement |
|---|---|---|
| `test_pipeline.py` | 10, 22, 26, 30, 85, 106, 126, 127, 137, 181, 202, 231, 239, 257, 283 | Supprimer `lang=` des constructeurs `CGNPipeline` et `UDRepresentation` |
| `test_pipeline.py` | 37, 129 | `assert "source_lang" in result` → garder (la clé existe, vaut `None`) |
| `test_layer1.py` | 6, 18 | Supprimer `lang=` de `make_rep` et `UDRepresentation` |
| `test_ir_emitter.py` | 8, 15 | Supprimer `lang="fr"` de `emit()` ; `assert result["source_lang"] is None` |
| `test_frontend_bridge.py` | 141, 377 | Supprimer `assert rep.lang == "fr"` (champ supprimé) |
| `test_frontend_bridge.py` | 373, 393 | Supprimer `lang="fr"` des appels |
| `test_training.py` | 23, 121, 128, 155, 162, 201, 216, 250, 297, 336, 368, 391, 455 | Supprimer `lang=` ; remplacer `pipeline.lang` par `""` (ligne 201, 455) |
| `test_bootstrap.py` | 111 | `assert doc["document"]["lang"] == "fr"` → `assert "lang" not in doc["document"]` |
| `test_checkpoint.py` | 19, 30 | Supprimer `lang="fr"` de `CGNPipeline` |
| `test_trainable_decoder.py` | 257, 264, 281, 304, 312 | Supprimer `lang=` des constructeurs |
| `test_taxonomy.py` | 19 | Test `test_unknown_lang_fallback` → adapter : sans `lang_code`, vérifier que tous les sous-répertoires sont scannés |

---

## Tableau des fichiers modifiés

| Fichier | Action |
|---|---|
| `layer1/representation.py` | Supprimer champ `lang` |
| `layer0/interface.py` | Supprimer `lang` du Protocol `TextParser.parse` |
| `data/schema.py` | `lang: str` → `lang: str = ""` (optionnel) |
| `data/json_reader.py` | Supprimer `lang` des 4 signatures de fonctions |
| `data/loader.py` | Supprimer `lang` du constructeur et des 2 helpers |
| `pipeline/label_builder.py` | Universaliser labels + refactoring `_load_nominalizations` |
| `pipeline/ir_emitter.py` | Supprimer `lang`, émettre `"source_lang": null` |
| `pipeline/cgnp.py` | Supprimer `lang` du constructeur et des 4 call sites |
| `frontend/bridge.py` | Supprimer `lang` des 5 fonctions |
| `training/train.py` | Supprimer `--lang` CLI |
| `training/bootstrap.py` | Supprimer `--lang` CLI, supprimer `"lang"` du JSON produit |
| `evaluation/eval_runner.py` | Supprimer `--lang` CLI |
| `pipeline/cli.py` | Supprimer `--lang` CLI |
| `taxonomy/loader.py` | Supprimer `lang_code`, scanner tous les sous-répertoires |
| `tests/` (11 fichiers) | ~50 substitutions mécaniques |

**Rust (`gcn-core/`)** : **inchangé**.

---

## Ce qui ne change pas

- Features ML : `FeatureVocabulary`, `vectorize_clause`, `vectorize_edge` — inchangés
- MLP, R-GCN, GAT, WordEmbedding — inchangés
- Loss et backward — inchangés
- Format JSON des datasets (`SentenceRecord.lang` reste lisible, optional)
- Backend Rust Pearl / GCN-QL — inchangé

---

## Vérification end-to-end

```bash
# 1. Entraînement sans --lang sur dataset FR
gcn-train --data-dir gcn-datasets/examples/ --epochs 5 --output /tmp/ckpt_nolang.npz
# Attendre : pas d'erreur, loss décroissante

# 2. Entraînement sur dataset mixte (FR + code dans le même répertoire)
gcn-train --data-dir gcn-datasets/ --epochs 5 --output /tmp/ckpt_mixed.npz
# Attendre : pas d'erreur, aucun paramètre lang requis

# 3. Suite complète de tests
python -m pytest gcn-python/tests/ -v
# Attendre : 177+ passed, 0 failed
```

**Métriques de succès :**
- `gcn-train` sans `--lang` fonctionne sur dataset FR, EN, code
- `node_accuracy` identique avant et après (les features ML sont inchangées)
- `result["source_lang"] is None` dans les tests `ir_emitter`
- 0 occurrence de `lang` dans les signatures publiques de `CGNPipeline`, `GCNDataLoader`, `TextParser`
- `test_taxonomy.py` : tous les lemmes de tous les sous-répertoires sont trouvés
