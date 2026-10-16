# Issues GitHub — Audit Post-Phase 9

**Date** : 2026-09-16  
**Context** : Audit complet après finalisation phase 9 (commits d0d01a0 à c16b01f)  
**Findings** : 10 issues (3 CRITIQUES, 3 CORRECTNESS, 4 PERFORMANCE)

---

## 🔴 ISSUES CRITIQUES (Priority: HIGH)

### Issue #1 : bootstrap_cmd passe text via stdin au lieu d'argument positionnel

**Labels** : `bug`, `critical`, `training`, `bootstrap`

**Description** :

Le pipeline bootstrap utilise `subprocess.run` avec `input=text` pour passer le texte à analyser au CLI Rust `gcn analyze`, mais le CLI attend le texte comme argument positionnel.

**Fichier** : `gcn-python/src/gcn_python/training/bootstrap.py:42-48`

**Code actuel** :
```python
result = subprocess.run(
    [gcn_bin, "analyze", "--lang", lang],
    input=text,  # ← stdin
    text=True,
    capture_output=True,
    check=True,
)
```

**Problème** :
- CLI Rust : `gcn analyze <TEXT>` (argument positionnel requis)
- Code Python : envoie via stdin → CLI échoue avec "missing required argument: <TEXT>"

**Impact** :
- ❌ Bootstrap workflow complètement cassé
- ❌ Impossible de générer des fichiers d'entraînement depuis texte brut
- ❌ Pipeline ML non fonctionnel pour nouveaux datasets

**Scénario de reproduction** :
```bash
echo "Les ventes baissent." | gcn analyze --lang fr
# ❌ Error: missing required argument: <TEXT>

gcn analyze --lang fr "Les ventes baissent."
# ✅ Fonctionne
```

**Solution proposée** :
```python
result = subprocess.run(
    [gcn_bin, "analyze", "--lang", lang, text],  # text comme dernier arg
    text=True,
    capture_output=True,
    check=True,
)
```

**Tests à ajouter** :
- Test d'intégration `test_bootstrap_cmd_cli_integration` avec vrai binaire gcn
- Mock test vérifiant que text est passé comme arg positionnel

---

### Issue #2 : Encodage subprocess par défaut (locale) provoque corruption UTF-8

**Labels** : `bug`, `critical`, `i18n`, `bootstrap`

**Description** :

`subprocess.run` avec `text=True` utilise l'encodage locale par défaut au lieu de UTF-8 explicite, causant corruption de texte français sur systèmes non-UTF-8.

**Fichier** : `gcn-python/src/gcn_python/training/bootstrap.py:42-48`

**Code actuel** :
```python
result = subprocess.run(
    [...],
    input=text,
    text=True,  # ← utilise locale.getpreferredencoding()
    capture_output=True,
)
```

**Problème** :
- Sur systèmes avec locale ISO-8859-1 : texte UTF-8 encodé incorrectement
- Accents français corrompus : "été" → "Ã©tÃ©"
- CIR produit contient des caractères invalides

**Impact** :
- ❌ Datasets français corrompus sur serveurs CI non-UTF-8
- ❌ Noms d'entités illisibles dans CIR
- ❌ Potentiel crash JSON parsing si caractères invalides

**Scénario de reproduction** :
```python
# Système avec LANG=fr_FR.ISO-8859-1
text = "L'été provoque l'évaporation"
# subprocess avec text=True tente encoder UTF-8 → ISO-8859-1
# → UnicodeEncodeError ou corruption silencieuse
```

**Solution proposée** :
```python
result = subprocess.run(
    [...],
    input=text.encode('utf-8'),  # bytes explicites
    capture_output=True,
    check=True,
)
# Puis décoder stdout :
stdout_text = result.stdout.decode('utf-8')
```

Ou :

```python
result = subprocess.run(
    [...],
    input=text,
    text=True,
    encoding='utf-8',  # ← explicite
    capture_output=True,
)
```

**Tests à ajouter** :
- Test avec texte français contenant accents
- Mock test simulant locale ISO-8859-1

---

### Issue #3 : int() crash avec valeurs null dans token_span JSON

**Labels** : `bug`, `critical`, `bootstrap`, `robustness`

**Description** :

`_extract_token_span` utilise `ts.get('start', 0)` qui retourne la valeur réelle `None` si `"start": null` dans le JSON, pas le défaut `0`. `int(None)` lève TypeError.

**Fichier** : `gcn-python/src/gcn_python/training/bootstrap.py:80-85`

**Code actuel** :
```python
def _extract_token_span(span_obj) -> list[int]:
    if isinstance(span_obj, dict):
        ts = span_obj.get("token_span", {})
        return [int(ts.get('start', 0)), int(ts.get('end', 0))]
        # ↑ si ts = {"start": null, "end": 5} 
        #   ts.get('start', 0) retourne None (valeur présente)
        #   int(None) → TypeError
```

**Problème** :
- `dict.get(key, default)` retourne `default` **seulement si clé absente**
- Si clé présente avec valeur `null` : retourne `None`, pas `default`
- JSON malformé du CLI Rust peut contenir `null`

**Impact** :
- ❌ Bootstrap crash sur JSON malformé avec `{"start": null}`
- ❌ Pas de fallback gracieux
- ❌ Message d'erreur cryptique pour utilisateur

**Scénario de reproduction** :
```python
ts = {"start": null, "end": 5}  # JSON valide mais sémantiquement incorrect
span = _extract_token_span({"token_span": ts})
# → TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

**Solution proposée** :
```python
def _extract_token_span(span_obj) -> list[int]:
    if isinstance(span_obj, dict):
        ts = span_obj.get("token_span", {})
        start = ts.get('start', 0)
        end = ts.get('end', 0)
        # Fallback pour null
        start = 0 if start is None else int(start)
        end = 0 if end is None else int(end)
        return [start, end]
    # ... reste inchangé
```

Ou avec opérateur `or` :

```python
return [int(ts.get('start') or 0), int(ts.get('end') or 0)]
```

**Tests à ajouter** :
- Test avec `{"token_span": {"start": null, "end": 5}}`
- Test avec `{"token_span": {"start": "invalid"}}` (ValueError)

---

## 🟡 ISSUES CORRECTNESS (Priority: MEDIUM)

### Issue #4 : _extract_token_span retourne liste de chars strings au lieu d'ints

**Labels** : `bug`, `medium`, `bootstrap`, `type-safety`

**Description** :

Si `token_span` est une string (JSON hand-edited incorrect), `list("01")` retourne `['0', '1']` (chars) pas `[0, 1]` (ints).

**Fichier** : `gcn-python/src/gcn_python/training/bootstrap.py:85`

**Code actuel** :
```python
def _extract_token_span(span_obj) -> list[int]:
    # ...
    if isinstance(span_obj, (list, str)):
        return list(span_obj)  # ← si str, retourne chars
```

**Problème** :
- `list("01")` → `['0', '1']` (strings de longueur 1)
- Type hint dit `list[int]` mais retourne `list[str]`
- Downstream code attend ints → TypeError ou comportement incorrect

**Impact** :
- ❌ Type safety violée
- ❌ Erreurs cryptiques dans code appelant
- ❌ Validation JSON schema échoue

**Scénario de reproduction** :
```python
span = _extract_token_span("01")
# Retourne ['0', '1'] pas [0, 1]
start, end = span
# start = '0' (string) pas 0 (int)
# range(start, end) → TypeError: 'str' object cannot be interpreted as an integer
```

**Solution proposée** :
```python
if isinstance(span_obj, str):
    # Si string, tenter parser comme "start,end" ou rejeter
    try:
        parts = span_obj.split(',')
        return [int(p.strip()) for p in parts]
    except (ValueError, AttributeError):
        warnings.warn(f"token_span string invalide : {span_obj}")
        return [0, 0]
if isinstance(span_obj, list):
    return [int(x) for x in span_obj]  # ← cast explicite
```

**Tests à ajouter** :
- Test avec `token_span: "01"` (string invalide)
- Test avec `token_span: "0,5"` (string valide comma-separated)
- Test avec `token_span: [0, 5]` (liste correcte)

---

### Issue #5 : _extract_token_span ne valide pas exactement 2 éléments

**Labels** : `bug`, `medium`, `bootstrap`, `validation`

**Description** :

Aucune validation que `token_span` contient exactement 2 éléments. Retourne liste de longueur variable → unpacking `start, end = span` crash.

**Fichier** : `gcn-python/src/gcn_python/training/bootstrap.py:85`

**Code actuel** :
```python
if isinstance(span_obj, (list, str)):
    return list(span_obj)  # ← pas de validation longueur
```

**Problème** :
- JSON malformé : `{"token_span": [5]}` → retourne `[5]`
- Downstream code : `start, end = _extract_token_span(...)` → ValueError

**Impact** :
- ❌ Crash avec message cryptique "not enough values to unpack"
- ❌ Pas de diagnostic clair sur JSON invalide

**Scénario de reproduction** :
```python
span = _extract_token_span([5])  # seulement 1 élément
start, end = span  # ValueError: not enough values to unpack (expected 2, got 1)
```

**Solution proposée** :
```python
if isinstance(span_obj, list):
    if len(span_obj) != 2:
        warnings.warn(
            f"token_span doit contenir 2 éléments (start, end), "
            f"reçu {len(span_obj)} : {span_obj}"
        )
        return [0, 0]
    return [int(span_obj[0]), int(span_obj[1])]
```

**Tests à ajouter** :
- Test avec `[5]` (1 élément)
- Test avec `[0, 5, 10]` (3 éléments)
- Test avec `[]` (liste vide)

---

### Issue #6 : Assert pour validation d'entrée désactivable avec python -O

**Labels** : `bug`, `medium`, `backward`, `production`

**Description** :

`_LinearLayer.backward()` utilise `assert d_out.ndim == 1` pour validation, mais assert est désactivé en mode optimisé (`python -O`).

**Fichier** : `gcn-python/src/gcn_python/layer2/reference.py:26-28`

**Code actuel** :
```python
def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert d_out.ndim == 1, "backward attend d_out 1-D"
    x = self._cache.get("x")
    # ...
    dW = np.outer(d_out, x)
```

**Problème** :
- Production déployée avec `python -O` → assert ignoré
- Array batché 2-D passe validation → `np.outer(d_out=(2,2), x=(2,))` produit shape incorrecte
- Gradient de mauvaise forme se propage silencieusement

**Impact** :
- ❌ Bug silencieux en production
- ❌ Training incorrect avec gradients malformés
- ❌ Difficile à débugger (symptôme éloigné de la cause)

**Scénario de reproduction** :
```bash
python -O -m gcn_python.training.train --data-dir ...
# Assert désactivé → validation sautée
# Si code appelant passe array 2-D par erreur → crash ou gradient incorrect
```

**Solution proposée** :
```python
def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_out.ndim != 1:
        raise ValueError(
            f"_LinearLayer.backward attend d_out 1-D, reçu shape {d_out.shape}"
        )
    x = self._cache.get("x")
    # ...
```

**Alternative** : Documenter que layer attend 1-D, laisser np.outer crash naturellement (mais message moins clair).

**Tests à ajouter** :
- Test backward avec array 2-D (doit lever ValueError)
- CI : vérifier tests passent aussi avec `python -O`

---

## 🟢 ISSUES PERFORMANCE (Priority: LOW)

### Issue #7 : Conversions int() répétées sans cache (node labels)

**Labels** : `performance`, `low`, `evaluation`

**Description** :

`eval_runner.py` appelle `int(lbl)` deux fois par label node sans cacher le résultat.

**Fichier** : `gcn-python/src/gcn_python/evaluation/eval_runner.py:100`

**Code actuel** :
```python
gold_lbl_count = [0] * len(NODE_TYPES)
for lbl in sample.gold_node_types:
    if 0 <= int(lbl) < len(NODE_TYPES):  # ← conversion 1
        gold_lbl_count[int(lbl)] += 1     # ← conversion 2
```

**Problème** :
- 1000 samples × 10 nodes = 20k conversions au lieu de 10k
- CPU gaspillé (mineur mais inutile)
- Si `lbl.__int__()` a side-effects : comportement non-déterministe

**Impact** :
- ⚠️ Performance dégradée sur gros datasets (mineur)
- ⚠️ Code peu idiomatique

**Solution proposée** :
```python
for lbl in sample.gold_node_types:
    idx = int(lbl)
    if 0 <= idx < len(NODE_TYPES):
        gold_lbl_count[idx] += 1
```

**Tests** : Aucun nouveau test requis (refactor neutre)

---

### Issue #8 : Conversions int() répétées sans cache (edge relations)

**Labels** : `performance`, `low`, `evaluation`

**Description** :

Même problème que #7 pour les relations d'arêtes.

**Fichier** : `gcn-python/src/gcn_python/evaluation/eval_runner.py:104`

**Code actuel** :
```python
for rel in sample.gold_edge_relations:
    if int(rel) >= 0:           # ← conversion 1 (filtre)
        gold_rel_count[int(rel)] += 1  # ← conversion 2 (indexing)
```

**Solution proposée** :
```python
for rel in sample.gold_edge_relations:
    idx = int(rel)
    if idx >= 0:
        gold_rel_count[idx] += 1
```

---

### Issue #9 : data_dir.glob("*.json") appelé deux fois

**Labels** : `performance`, `low`, `loader`

**Description** :

`VerbalizerDataLoader.__init__` traverse le filesystem deux fois : une fois pour `verbalize_*.json`, une fois pour tous `*.json`.

**Fichier** : `gcn-python/src/gcn_python/data/verbalize_loader.py:79-86`

**Code actuel** :
```python
# Ligne 79
for p in sorted(data_dir.glob("verbalize_*.json")):
    # ...

# Ligne 86 (warning)
all_jsons = set(data_dir.glob("*.json"))
```

**Problème** :
- Filesystem traversé 2 fois pour même répertoire
- Lent sur montages réseau ou répertoires avec milliers de fichiers

**Impact** :
- ⚠️ I/O gaspillé (mineur sur disque local, significatif sur NFS)

**Solution proposée** :
```python
all_jsons = set(data_dir.glob("*.json"))
verbalize_jsons = {p for p in all_jsons if p.name.startswith("verbalize_")}

for p in sorted(verbalize_jsons):
    # ... load

ignored = all_jsons - verbalize_jsons
if ignored:
    warnings.warn(...)
```

---

### Issue #10 : Gold CIR reconstruit dans boucle chaude

**Labels** : `performance`, `low`, `evaluation`

**Description** :

`eval_runner.py` reconstruit `gold_nodes` et `gold_edges` depuis `sample` à chaque itération au lieu de précalculer.

**Fichier** : `gcn-python/src/gcn_python/evaluation/eval_runner.py:99`

**Code actuel** :
```python
for sample in samples:
    # Reconstruit gold CIR chaque fois
    gold_nodes = [...]
    gold_edges = [...]
    sim = causal_graph_similarity(pred_cir, {"nodes": gold_nodes, "edges": gold_edges})
```

**Problème** :
- 1000 samples → 5000+ allocations éphémères (nodes + edges)
- Pression mémoire et GC overhead
- Pourrait être précalculé lors du load

**Impact** :
- ⚠️ Performance dégradée sur évaluations massives

**Solution proposée** :

Ajouter méthode `TrainingSample.to_gold_cir()` :
```python
@dataclass
class TrainingSample:
    # ... fields existants
    
    def to_gold_cir(self) -> dict:
        """Lazy-cache gold CIR pour réutilisation."""
        if not hasattr(self, '_gold_cir_cache'):
            self._gold_cir_cache = {
                "nodes": [...],
                "edges": [...],
            }
        return self._gold_cir_cache
```

Puis dans eval_runner :
```python
for sample in samples:
    gold_cir = sample.to_gold_cir()
    sim = causal_graph_similarity(pred_cir, gold_cir)
```

---

## 📊 Récapitulatif

| Priorité | Issues | Fichiers touchés |
|----------|--------|------------------|
| 🔴 CRITICAL | 3 | `training/bootstrap.py` |
| 🟡 MEDIUM | 3 | `training/bootstrap.py`, `layer2/reference.py` |
| 🟢 LOW | 4 | `evaluation/eval_runner.py`, `data/verbalize_loader.py` |

**Actions recommandées** :
1. Corriger les 3 CRITICAL avant merge vers `master`
2. Créer issues GitHub pour les 7 autres (MEDIUM + LOW)
3. Planifier corrections MEDIUM en phase 10
4. Optimisations LOW : nice-to-have, pas bloquantes

---

## 🔗 Liens

- **Audit source** : Code-review high-level (agent background task)
- **Phase 9 commits** : d0d01a0 à c16b01f
- **Tests** : 127/127 passent (avant corrections de ces issues)
