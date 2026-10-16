# Audit Moteur GCN — gcn-core + gcn-python

**Date :** 2026-09-18  
**Périmètre :** `gcn-core/crates/*` (Rust) + `gcn-python/src/*` (Python) + loaders `gcn-knowledge/loader.rs`, `taxonomy.rs`, `gcn-python/taxonomy/loader.py`  
**Exclus :** `gcn-references/taxonomies/*.yaml` (contenu linguistique), `gcn-datasets/`, `gcn-tools/`, `docs/`  
**Méthode :** lecture du code source + exécution `cargo test --workspace`, `cargo clippy -- -D warnings`, `python -m pytest tests/ -q`, analyse statique manuelle par phase A→F

---

## Tableau synthétique

| ID  | Sévérité | Phase | Fichier:ligne | Description |
|-----|----------|-------|---------------|-------------|
| B01 | **Critique** | B | `pyproject.toml` | `pyyaml` absent des dépendances → ImportError sur install fraîche |
| B02 | **Majeur** | B | `pipeline/cgnp.py:295` | `edge_logit / self.temperature` → div/0 si `temperature=0` |
| B03 | **Majeur** | B | `layer3/pytorch_rgcn.py` + `pipeline/cgnp.py:615` | `RGCNLayerPT` sans `backward_message_pass` → poids R-GCN jamais mis à jour sans warning |
| A01 | **Majeur** | A | `gcn-backend/src/pearl.rs:43,57,68,194,224` | Matching sous-chaîne → mauvais nœud si labels ambigus |
| A02 | **Majeur** | A | `gcn-middleend/src/graph.rs:25-29` | Arêtes dangling ignorées silencieusement |
| D02 | **Majeur** | D | `gcn-backend/src/pearl.rs:164-175` | `edge_link` O(E) dans BFS → O(E²) total |
| D01 | **Majeur** | D | `gcn-backend/src/pearl.rs:38,52,66,193,224` | `build(ir)` reconstruit le graphe petgraph à chaque requête |
| F02 | **Majeur** | F | `gcn-frontend-fr/src/annotator.rs:444` | `map_or(false,…)` → clippy `-D warnings` bloque la compilation |
| A03 | Mineur | A | `gcn-verbalizer/src/lib.rs:37` | `expect("stdin piped")` non-recoverable si OS défaillant |
| A07 | Mineur | A | `gcn-backend/src/error.rs:9` | `BackendError::NoPath` jamais retourné — code mort |
| A08 | Mineur | A | `gcn-middleend/src/error.rs:6` | `MiddleendError::GraphBuildError` jamais construite — code mort |
| A11 | Mineur | A | `gcn-backend/src/pearl.rs` | `build(ir)` dupliqué dans chaque fonction Pearl |
| A12 | Mineur | A | `gcn-cli/src/main.rs:136` | Texte sans séparateur `--` → flag injection CLI |
| B04 | Mineur | B | `evaluation/eval_runner.py:60-63` | `except Exception` + continue → erreurs forward noyées |
| B05 | Mineur | B | `taxonomy/loader.py:41` | `except Exception: continue` → YAML malformé ignoré |
| B06 | Mineur | B | `layer2/reference.py:63-64` | Double RNG même seed → corrélation init ↔ dropout |
| B07 | Mineur | B | `pipeline/cgnp.py:529-536` | Re-run backward sans dropout → gradients approximatifs |
| C01 | Mineur | C | `frontend/bridge.py:249` | Texte sans `--` → clap peut interpréter comme flag |
| C02 | Mineur | C | `gcn-backend/src/export.rs:59` | `escape_dot` n'échappe pas les backslashes |
| D03 | Mineur | D | `pipeline/cgnp.py:94-121` | Caches non libérés entre epochs — croissance mémoire |
| E01 | Mineur | E | `training/train.py:478` | Skip `dec_loss` non-finie sans compteur |
| E02 | Mineur | E | `evaluation/metrics.py:114` | Alignement position pour `causal_graph_similarity` → biais |
| F01 | Mineur | F | `gcn-ir/`, `gcn-verbalizer/`, `gcn-cli/` | 0 tests dans ces crates |
| F03 | Mineur | F | `Cargo.toml`, `pyproject.toml` | Versions divergentes : Rust 2.1.0 vs Python 2.4.0 |
| F04 | Mineur | F | `gcn-ir/Cargo.toml`, `gcn-knowledge/Cargo.toml` | `proptest` déclaré mais 0 tests proptest |

---

## Phase A — Robustesse / Correction (Rust)

### A01 — Majeur — Matching sous-chaîne dans le backend Pearl

**Fichiers :** `gcn-backend/src/pearl.rs:43,57,68,194,224`

Les fonctions `why()`, `what()`, `chain()`, `intervene()`, `counterfactual()` filtrent les nœuds via `.contains(&lower)` :

```rust
// pearl.rs:41-43
let lower = label.to_lowercase();
ir.nodes
    .iter()
    .filter(|n| n.label.to_lowercase().contains(&lower))
```

**Impact :** Si un IR contient les nœuds `"ventes"` et `"hausse_ventes"`, la requête `WHY ventes` retourne les deux résultats (ou uniquement le premier via `query.rs:133`). L'utilisateur obtient silencieusement la mauvaise réponse causal. Aucune ambiguïté signalée.

**Correctif :**

```rust
// Avant (pearl.rs:43)
.filter(|n| n.label.to_lowercase().contains(&lower))

// Après — exact match prioritaire, sinon substring avec warning
.filter(|n| {
    let lbl = n.label.to_lowercase();
    lbl == lower || lbl.contains(&lower)
})
// Puis, dans query.rs, si results.len() > 1 et aucun exact match → Err(BackendError::AmbiguousLabel)
```

Plus simple — match exact obligatoire, avec fallback substring documenté dans le message d'erreur :

```rust
// pearl.rs — remplacement drop-in
.filter(|n| n.label.to_lowercase() == lower)
// Si vide → retry with contains, ou retourner NodeNotFound explicite
```

---

### A02 — Majeur — Arêtes dangling ignorées silencieusement

**Fichier :** `gcn-middleend/src/graph.rs:25-29`

```rust
for (i, (src, dst, _edge)) in ir.edges.iter().enumerate() {
    if let (Some(&si), Some(&di)) = (node_indices.get(src), node_indices.get(dst)) {
        g.add_edge(si, di, i);
    }
    // Arête ignorée si src ou dst absent — aucun diagnostic
}
```

**Impact :** Un IR mal formé (nœud supprimé après ajout d'arête, ou serde partiel) perd silencieusement des arêtes. Les requêtes Pearl retournent des résultats incomplets. Aucun moyen de le détecter.

**Correctif :**

```rust
for (i, (src, dst, _edge)) in ir.edges.iter().enumerate() {
    match (node_indices.get(src), node_indices.get(dst)) {
        (Some(&si), Some(&di)) => { g.add_edge(si, di, i); }
        _ => {
            // Option A : retourner Err
            return Err(MiddleendError::GraphBuildError(
                format!("dangling edge #{i}: src={:?} dst={:?}", src, dst)
            ));
            // Option B (permissif) : émettre un Diagnostic::DanglingEdge
        }
    }
}
```

---

### A03 — Mineur — `expect` non-recoverable dans verbalizer

**Fichier :** `gcn-verbalizer/src/lib.rs:37`

```rust
child
    .stdin
    .take()
    .expect("stdin piped")  // panic si stdin est None malgré Stdio::piped()
```

**Impact :** Dans le cas improbable où l'OS ne fournit pas stdin après `Stdio::piped()`, le process panique au lieu de retourner `VerbalizerError`. Non-recoverable.

**Correctif :**

```rust
child
    .stdin
    .take()
    .ok_or_else(|| VerbalizerError::ProcessSpawn(
        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "stdin non disponible")
    ))?
    .write_all(ir_json.as_bytes())
    .map_err(VerbalizerError::ProcessSpawn)?;
```

---

### A04 — Mineur — `expect("non-empty")` dans les frontends code

**Fichiers :** `frontend-code/src/python.rs:88`, `rust.rs:90`, `js.rs:86`

```rust
if !block_ids.is_empty() {
    let prev = *block_ids.last().expect("non-empty");  // invariant vérifié ci-dessus
```

L'invariant est correct (`!block_ids.is_empty()` juste avant), mais le style défensif préféré est :

```rust
if let Some(&prev) = block_ids.last() {
    edges.push((prev, id, control_edge(RelationType::Sequence)));
}
```

---

### A07 — Mineur — `BackendError::NoPath` jamais retourné

**Fichier :** `gcn-backend/src/error.rs:9`

```rust
#[error("no path from '{0}' to '{1}'")]
NoPath(String, String),
```

`chain()` retourne `Option<Vec<CausalLink>>` sans jamais lever `NoPath`. `query.rs` le retransmet comme `found: false`. Le variant est déclaré mais mort.

---

### A08 — Mineur — `MiddleendError::GraphBuildError` jamais construite

**Fichier :** `gcn-middleend/src/error.rs:6`

```rust
GraphBuildError(String),
```

`graph.rs` ne retourne jamais d'erreur (voir A02). Ce variant est inutilisé.

---

### A11 — Mineur — `build(ir)` dupliqué dans chaque requête Pearl

**Fichier :** `gcn-backend/src/pearl.rs:39,53,67,206,226`

Chaque appel à `why`, `what`, `chain`, `intervene`, `counterfactual` appelle `build(ir)` qui reconstruit le graphe petgraph complet. Pour N requêtes séquentielles sur le même IR, cela est O(N × (|nodes| + |edges|)) au lieu de O(|nodes| + |edges|) + O(N × query).

---

### A12 — Mineur — Texte sans `--` dans gcn-cli forward

**Fichier :** `gcn-cli/src/main.rs:136`

```rust
cmd.arg(&text);
```

Si `text` commence par `--`, certains parseurs CLI interprètent cela comme un flag. Rust `Command` n'utilise pas le shell (pas d'injection), mais le comportement du sous-process `gcn-forward` peut être inattendu. Correctif : `cmd.arg("--").arg(&text)` si le binaire cible supporte le séparateur POSIX.

---

## Phase B — Robustesse / Correction (Python)

### B01 — Critique — `pyyaml` absent des dépendances Python

**Fichier :** `pyproject.toml` + `taxonomy/loader.py:10`

```python
# taxonomy/loader.py:10
import yaml  # pyyaml
```

```toml
# pyproject.toml — dependencies actuelles
dependencies = [
    "numpy>=1.24,<3.0",
    "click>=8.1",
]
# pyyaml ABSENT
```

**Impact :** `pip install gcn-python && python -c "from gcn_python.taxonomy.loader import TaxonomyIndex"` → `ModuleNotFoundError: No module named 'yaml'`. Toute utilisation des features basées sur les taxonomies (layer1 avec `TaxonomyIndex`) crashe sur une installation standard.

**Correctif :**

```toml
# pyproject.toml
dependencies = [
    "numpy>=1.24,<3.0",
    "click>=8.1",
    "pyyaml>=6.0",
]
```

---

### B02 — Majeur — Division par zéro sur `temperature=0`

**Fichier :** `pipeline/cgnp.py:295`

```python
# cgnp.py:295
rel_conf = float(_softmax((edge_logit / self.temperature).reshape(1, -1))[0, rel_idx])
```

Si l'utilisateur construit `CGNPipeline(..., temperature=0)` ou `temperature<0`, `edge_logit / 0` produit `inf` ou `NaN`. `_softmax(inf)` retourne `NaN` (0/0 après exp-max trick). La valeur de confiance dans le CIR de sortie est corrompue silencieusement.

**Impact :** Aucune ValueError dans le constructeur (lignes 42 et 64). Le CIR produit a des `confidence: NaN` → serde_json Rust rejette les NaN flottants.

**Correctif :**

```python
# cgnp.py:64 — dans __init__, après self.temperature = float(temperature)
if self.temperature <= 0.0:
    raise ValueError(
        f"temperature doit être > 0, reçu {temperature}. "
        "Utilisez temperature=1.0 pour désactiver le scaling."
    )
```

---

### B03 — Majeur — `RGCNLayerPT` sans `backward_message_pass` : poids R-GCN gelés sans avertissement

**Fichiers :** `layer3/pytorch_rgcn.py` + `pipeline/cgnp.py:611-617`

```python
# cgnp.py:611-617
if (self._cached_edge_index is not None
        and self._cached_enriched_vecs is not None):
    d_curr = d_enriched
    for _layer in reversed(self._graph_layers):
        if hasattr(_layer, 'backward_message_pass'):   # False pour RGCNLayerPT
            d_curr, graph_grads = _layer.backward_message_pass(d_curr)
            _layer.update(graph_grads, lr)
        # Si pas de backward_message_pass : aucun update, aucun warning
```

`RGCNLayerPT` n'implémente pas `backward_message_pass` (il attend l'utilisation de PyTorch autograd via `optimizer.step()`). Mais `backward()` de `CGNPipeline` est la voie NumPy — les poids R-GCN PyTorch ne sont jamais mis à jour lors d'un entraînement avec le pipeline NumPy.

**Contraste :** L'absence de `backward_node_dx` sur l'encodeur émet un `UserWarning` (cgnp.py:492-500). L'absence de `backward_message_pass` sur la couche R-GCN est silencieuse.

**Impact :** Un utilisateur qui substitue `RGCNLayer` par `RGCNLayerPT` dans le pipeline NumPy entraîne un modèle dont les poids R-GCN ne bougent jamais. L'encodeur est mis à jour mais les features graph restent gelées. Les métriques s'améliorent légèrement (via l'encodeur) mais le R-GCN ne contribue pas à l'apprentissage.

**Correctif :**

```python
# cgnp.py — dans la boucle backward R-GCN
for _layer in reversed(self._graph_layers):
    if hasattr(_layer, 'backward_message_pass'):
        d_curr, graph_grads = _layer.backward_message_pass(d_curr)
        _layer.update(graph_grads, lr)
    else:
        import warnings as _w
        _w.warn(
            f"{type(_layer).__name__} n'implémente pas backward_message_pass — "
            "les poids de cette couche R-GCN ne sont pas mis à jour par backward(). "
            "Utilisez torch_parameters() + optimizer.step() pour l'entraînement PyTorch.",
            UserWarning, stacklevel=3,
        )
        break  # Une seule fois suffit
```

---

### B04 — Mineur — `except Exception` avalant les erreurs de forward

**Fichier :** `evaluation/eval_runner.py:60-63`

```python
except Exception as exc:
    warnings.warn(f"[{sample.sentence.id}] forward ignoré : {exc}", stacklevel=2)
    n_skipped += 1
    continue
```

Un bug systématique dans le pipeline (ex: shape mismatch après refactoring) skipperait tous les samples sans jamais lever d'exception. Le run se terminerait avec `n_skipped=N` et 0 résultat.

**Correctif :** Restreindre l'exception aux erreurs attendues et ajouter un seuil d'arrêt :

```python
except (RuntimeError, IndexError, KeyError, TypeError) as exc:
    warnings.warn(...)
    n_skipped += 1
    if n_skipped > max_skipped:
        raise RuntimeError(f"Trop d'échecs ({n_skipped}) — vérifier le pipeline.") from exc
    continue
```

---

### B05 — Mineur — YAML malformé ignoré silencieusement

**Fichier :** `taxonomy/loader.py:41`

```python
try:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
except Exception:
    continue  # fichier YAML corrompu → silencieusement ignoré
```

Un fichier de taxonomie mal formé (erreur de syntaxe, encodage) est ignoré sans aucun avertissement. Les features basées sur ce fichier seront absentes, dégradant l'accuracy sans diagnostic.

**Correctif :**

```python
except Exception as exc:
    warnings.warn(
        f"TaxonomyIndex: impossible de charger {yaml_path.name} : {exc}",
        UserWarning, stacklevel=3,
    )
    continue
```

---

### B06 — Mineur — Double RNG avec même seed dans MLPEncoder

**Fichier :** `layer2/reference.py:63-64`

```python
rng = np.random.default_rng(seed)        # utilisé pour init des poids
self._rng = np.random.default_rng(seed)  # utilisé pour dropout — MÊME seed
```

Les deux générateurs démarrent dans le même état. La première séquence de nombres aléatoires générée par `self._rng` (dropout) est identique à celle générée par `rng` (init poids). Corrélation non intentionnelle entre init et dropout.

**Correctif :**

```python
rng = np.random.default_rng(seed)
self._rng = np.random.default_rng(seed + 1)  # seed différent pour dropout
```

---

### B07 — Mineur — Re-run backward sans dropout → gradients approximatifs

**Fichier :** `pipeline/cgnp.py:529-536`

```python
# Sans snapshot : re-execute forward pour recréer le cache interne
_was_tr = getattr(self.encoder, 'training', False)
if _was_tr and hasattr(self.encoder, 'training'):
    self.encoder.training = False
self.encoder.forward_node(vecs[i])
if _was_tr and hasattr(self.encoder, 'training'):
    self.encoder.training = True
```

Si le forward original avait du dropout actif (`training=True`), le re-run pour backward est fait sans dropout. Les activations recalculées diffèrent de celles du forward original → gradients calculés sur la mauvaise passe. Documenté en commentaire mais non propagé comme UserWarning lors de l'entraînement avec dropout.

---

## Phase C — Sécurité

### C01 — Mineur — Texte sans séparateur `--` dans bridge.py

**Fichier :** `frontend/bridge.py:246-249`

```python
cmd = [gcn_bin, "analyze"]
if taxonomy_dir is not None:
    cmd += ["--data-dir", str(taxonomy_dir)]
cmd.append(text)  # texte sans séparateur --
```

Si le texte commence par `-` ou `--`, clap (gcn-cli) peut l'interpréter comme un flag inconnu et retourner une erreur confuse. Pas d'injection shell (subprocess sans `shell=True`).

**Correctif :**

```python
cmd.extend(["--", text])
# Vérifier que gcn analyze accepte le séparateur -- (clap le supporte nativement)
```

---

### C02 — Mineur — `escape_dot` incomplet dans export.rs

**Fichier :** `gcn-backend/src/export.rs:58-60`

```rust
fn escape_dot(s: &str) -> String {
    s.replace('"', "\\\"").replace('\n', "\\n")
    // Manque : '\\' → "\\\\"
}
```

Un label contenant un backslash (`\`) n'est pas échappé, produisant une séquence invalide dans le fichier DOT. Certains renderers DOT (graphviz, dot) peuvent interpréter `\n` dans les labels différemment selon l'OS.

**Correctif :**

```rust
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
}
```

---

### Findings anticipés non confirmés (Phase C)

- **GCN-QL injection** : `query.rs:21-55` utilise `strip_prefix` et pattern matching — les labels sont des strings Rust ordinaires, jamais évalués dynamiquement. Aucune injection possible.
- **YAML bomb** : `yaml.safe_load()` (Python) et `yaml_serde::from_str` (Rust) sont des parsers sécurisés. Pas de `yaml.load()` dangereux dans le code.
- **Path traversal gcn-knowledge** : Les paths sont fournis par l'utilisateur CLI authentifié, pas par des données non fiables. Pas de vulnérabilité dans le contexte d'usage.

---

## Phase D — Performance

### D01 — Majeur — `build(ir)` reconstruit le graphe à chaque requête Pearl

**Fichier :** `gcn-backend/src/pearl.rs:38,52,66,193,224`

Chacune des 5 fonctions Pearl appelle `build(ir)` individuellement :

```rust
pub fn why(ir: &CausalIR, label: &str) -> Vec<...> { let g = build(ir); ... }
pub fn what(ir: &CausalIR, label: &str) -> Vec<...> { let g = build(ir); ... }
pub fn chain(ir: &CausalIR, ...) -> ... { let g = build(ir); ... }
pub fn intervene(ir: &CausalIR, ...) -> ... { /* build implicite via bfs_descendants */ let g = build(ir); ... }
pub fn counterfactual(ir: &CausalIR, ...) -> ... { let g = build(ir); ... }
```

Pour un session `gcn query` sur plusieurs requêtes successives (ex: batch de WHY+WHAT+CHAIN), le graphe est reconstruit N fois. À N=5 requêtes sur un IR de 100 nœuds/300 arêtes, c'est 5×O(400) au lieu de 1×O(400).

---

### D02 — Majeur — `edge_link()` O(E) par arête dans le BFS → O(E²)

**Fichier :** `gcn-backend/src/pearl.rs:164-175`

```rust
fn edge_link(ir: &CausalIR, src: NodeId, dst: NodeId) -> Option<CausalLink> {
    ir.edges.iter()                         // scan linéaire O(E)
        .find(|(s, d, _)| *s == src && *d == dst)
        ...
}
```

Cette fonction est appelée dans `bfs_ancestors`, `bfs_descendants` et `bfs_path_links` pour **chaque arête traversée** dans le BFS. Pour un graphe à E arêtes, la complexité totale est O(E × E) = O(E²) au lieu de O(E) si un HashMap était utilisé.

**Correctif :**

```rust
// Construire un HashMap<(NodeId,NodeId), &CausalEdge> avant le BFS
let edge_map: HashMap<(NodeId, NodeId), &CausalEdge> = ir.edges
    .iter()
    .map(|(s, d, e)| ((*s, *d), e))
    .collect();

fn edge_link_fast(edge_map: &HashMap<(NodeId,NodeId), &CausalEdge>, src: NodeId, dst: NodeId, ir: &CausalIR) -> Option<CausalLink> {
    edge_map.get(&(src, dst)).map(|e| CausalLink {
        from_label: node_label(ir, src),
        to_label: node_label(ir, dst),
        to_id: dst,
        relation: e.relation,
        confidence: e.confidence,
        negated: e.negated,
    })
}
```

---

### D03 — Mineur — Caches `_cached_*` non libérés entre epochs

**Fichier :** `pipeline/cgnp.py:94-121`

Les caches `_cached_clause_vecs`, `_cached_enriched_vecs`, etc. sont réinitialisés à chaque `forward()` mais les objets précédents restent en mémoire jusqu'au GC. Pour des large batches (N=500+ nœuds), cela peut accumuler des centaines de MB. Pas de `del self._cached_*` explicite ni `gc.collect()` entre les epochs dans `train.py`.

---

## Phase E — Qualité ML / Données-pipeline

### E01 — Mineur — Skip `dec_loss` non-finie sans compteur

**Fichier :** `training/train.py:478-479`

```python
if not np.isfinite(dec_loss):
    continue  # ignoré silencieusement
```

Si le décodeur produit des losses non-finies de façon répétée (poids divergents), l'entraînement continue sans signal. Recommandation : compteur de skip + avertissement si skip_rate > 10%.

---

### E02 — Mineur — Alignement positionnel dans `causal_graph_similarity`

**Fichier :** `evaluation/metrics.py:114-128`

L'alignement nœuds/arêtes par position est documenté comme potentiellement biaisé (warning émis). Mais pour des graphes dont l'ordre de nœuds diffère (prédiction vs gold), la métrique `node_type_accuracy` est systématiquement sous-estimée. Pour une évaluation fiable, un alignement par identité (bipartite matching) serait nécessaire.

### Findings anticipés non confirmés (Phase E)

- **`n_total_clauses=0`** : `features.py:165` utilise `max(n_clauses, 1)` — pas de division par zéro.
- **`edge_types hors-bornes`** : `layer3/reference.py` itère `for r in range(self.n_relations)` et filtre par `edge_types == r` — les types inconnus sont simplement ignorés sans crash.

---

## Phase F — Tests / Dette / Docs-drift

### F01 — Mineur — Zéro tests pour gcn-ir, gcn-verbalizer, gcn-cli

Résultats `cargo test --workspace` :
- `gcn-ir` : 0 tests (1 doc-test nominal)
- `gcn-verbalizer` : 0 tests
- `gcn-cli` : 0 tests
- Crates avec tests : backend (21), frontend-fr (27), frontend-en (29), frontend-code (17), middleend (23), knowledge (2)
- Total Rust : 137 tests passés

Résultats `python -m pytest tests/ -q` : **231 passed, 4 skipped** — bonne couverture Python.

---

### F02 — Majeur — `map_or(false,…)` bloque `cargo clippy -D warnings`

**Fichier :** `gcn-frontend-fr/src/annotator.rs:444`

```rust
if subject.as_deref().map_or(false, |s| FR_UNIVERSAL_SUBJECT_LEMMAS.contains(&s)) {
```

Clippy `unnecessary-map-or` refuse avec `-D warnings`. Bloque tout CI utilisant `clippy -D warnings`.

**Correctif :**

```rust
if subject.as_deref().is_some_and(|s| FR_UNIVERSAL_SUBJECT_LEMMAS.contains(&s)) {
```

---

### F03 — Mineur — Versions divergentes Rust vs Python

`gcn-core/Cargo.toml` : `version = "2.1.0"`  
`gcn-python/pyproject.toml` : `version = "2.4.0"`

Les deux composants évoluent séparément mais un utilisateur lisant le dépôt peut être confus sur la compatibilité.

---

### F04 — Mineur — `proptest` déclaré mais 0 tests property-based

**Fichiers :** `gcn-ir/Cargo.toml:17`, `gcn-knowledge/Cargo.toml:18`

```toml
proptest = { workspace = true }
```

Aucune utilisation de `proptest!` ou `proptest::proptest!` dans les sources. Dépendance de dev inutile, ralentit la compilation.

---

### Findings anticipés non confirmés (Phase F)

- **`rayon` déclaré et inutilisé** : `rayon` est déclaré dans `[workspace.dependencies]` mais **aucun crate ne le référence** dans son propre `[dependencies]` — il n'est donc pas compilé. Non-trouvé comme dépendance active.
- **`L3 ValidationFeedback/reannotate absent`** : Ces types/méthodes ne sont pas présents dans le code actuel. Drift SAD confirmé mais pas de régression fonctionnelle.
- **`L4 integrate()/entity_index absent`** : Idem — décrit dans la SAD mais non implémenté. Dette documentaire.

---

## Correctifs prioritaires (Critique + Majeur)

### Fix B01 — `pyproject.toml` : ajouter `pyyaml`

```toml
# gcn-python/pyproject.toml
dependencies = [
    "numpy>=1.24,<3.0",
    "click>=8.1",
    "pyyaml>=6.0",          # ← ajouter
]
```

---

### Fix B02 — `cgnp.py:64` : garde temperature > 0

```python
# pipeline/cgnp.py — dans __init__, après ligne 64
self.temperature = float(temperature)
if self.temperature <= 0.0:
    raise ValueError(
        f"temperature doit être > 0 (reçu : {temperature}). "
        "Utilisez temperature=1.0 pour désactiver le scaling de confiance."
    )
```

---

### Fix B03 — `cgnp.py:615` : warning R-GCN gelé

```python
# pipeline/cgnp.py — remplacer lignes 611-617
if (self._cached_edge_index is not None
        and self._cached_enriched_vecs is not None):
    d_curr = d_enriched
    _rgcn_warned = False
    for _layer in reversed(self._graph_layers):
        if hasattr(_layer, 'backward_message_pass'):
            d_curr, graph_grads = _layer.backward_message_pass(d_curr)
            _layer.update(graph_grads, lr)
        elif not _rgcn_warned:
            import warnings as _w
            _w.warn(
                f"CGNPipeline.backward() : {type(_layer).__name__} sans backward_message_pass "
                "— les poids R-GCN ne sont pas mis à jour. "
                "Utilisez torch_parameters() + optimizer.step() pour l'entraînement PyTorch natif.",
                UserWarning, stacklevel=2,
            )
            _rgcn_warned = True
```

---

### Fix A01 — `pearl.rs` : exact match prioritaire

```rust
// pearl.rs — why(), what(), chain(), intervene(), counterfactual()
// Remplacement du filtre .contains() par exact match avec fallback

let lower = label.to_lowercase();
let exact: Vec<_> = ir.nodes.iter()
    .filter(|n| n.label.to_lowercase() == lower)
    .collect();
let candidates = if !exact.is_empty() { exact } else {
    ir.nodes.iter()
        .filter(|n| n.label.to_lowercase().contains(&lower))
        .collect()
};
// Utiliser candidates à la place du filter précédent
```

---

### Fix A02 — `graph.rs:25-29` : diagnostic dangling edge

```rust
// gcn-middleend/src/graph.rs — dans build()
for (i, (src, dst, _edge)) in ir.edges.iter().enumerate() {
    match (node_indices.get(src), node_indices.get(dst)) {
        (Some(&si), Some(&di)) => { g.add_edge(si, di, i); }
        _ => {
            // Retourner une erreur (option stricte) ou
            // émettre un diagnostic (option permissive — cohérent avec le reste du middleend)
            eprintln!("[WARN] dangling edge #{i}: src={:?} dst={:?} — ignorée", src, dst);
        }
    }
}
```

---

### Fix D02 — `pearl.rs` : HashMap pour `edge_link`

```rust
// Dans les fonctions bfs_* — construire l'index avant le BFS
// pearl.rs — ajouter avant bfs_ancestors/bfs_descendants :
fn build_edge_map(ir: &CausalIR) -> HashMap<(NodeId, NodeId), usize> {
    ir.edges.iter().enumerate()
        .map(|(i, (s, d, _))| ((*s, *d), i))
        .collect()
}

fn edge_link_indexed(ir: &CausalIR, edge_map: &HashMap<(NodeId,NodeId), usize>, src: NodeId, dst: NodeId) -> Option<CausalLink> {
    edge_map.get(&(src, dst)).map(|&i| {
        let (s, d, e) = &ir.edges[i];
        CausalLink {
            from_label: node_label(ir, *s),
            to_label: node_label(ir, *d),
            to_id: *d,
            relation: e.relation,
            confidence: e.confidence,
            negated: e.negated,
        }
    })
}
```

---

### Fix F02 — `annotator.rs:444` : `is_some_and`

```rust
// gcn-frontend-fr/src/annotator.rs:444
// Avant :
if subject.as_deref().map_or(false, |s| FR_UNIVERSAL_SUBJECT_LEMMAS.contains(&s)) {
// Après :
if subject.as_deref().is_some_and(|s| FR_UNIVERSAL_SUBJECT_LEMMAS.contains(&s)) {
```

---

## Dette technique (Mineurs sans correctif immédiat)

| ID  | Description | Effort estimé |
|-----|-------------|---------------|
| A03 | `expect("stdin piped")` → `ok_or_else(...)` dans verbalizer | 5 min |
| A07 | Supprimer `BackendError::NoPath` ou l'utiliser dans `chain()` | 15 min |
| A08 | Utiliser `MiddleendError::GraphBuildError` dans `graph.rs` (voir Fix A02) | 10 min |
| A11 | Refactorer Pearl pour accepter un `CausalGraph` pré-construit | 30 min |
| A12 | Ajouter `--` avant le texte dans `gcn forward` et `bridge.py` | 5 min |
| B04 | Seuil d'erreurs dans `eval_runner.py` | 15 min |
| B05 | Ajouter `UserWarning` dans `taxonomy/loader.py` pour YAML malformé | 5 min |
| B06 | Seed distinct pour `self._rng` dans `MLPEncoder` | 2 min |
| B07 | Ajouter warning quand re-run backward avec dropout | 5 min |
| C02 | Échapper `\\` dans `escape_dot()` | 2 min |
| D03 | `gc.collect()` ou `del` des caches entre epochs dans `train.py` | 10 min |
| E01 | Compteur de skip `dec_loss` non-finie dans `train.py` | 10 min |
| E02 | Alignement bipartite dans `causal_graph_similarity` | 2-4h |
| F01 | Tests unitaires pour gcn-ir, gcn-verbalizer, gcn-cli | 2-4h |
| F03 | Synchroniser versions Rust/Python | 5 min |
| F04 | Retirer `proptest` des dev-deps inutilisés | 5 min |

---

## Résumé exécutif

**Critiques (1) :** `pyyaml` manquant → crash install fraîche. Fix : 1 ligne dans `pyproject.toml`.

**Majeurs (7) :**
- `temperature=0` → NaN dans le CIR (Fix : guard dans `__init__`)
- `RGCNLayerPT` backward silencieux → poids gelés (Fix : UserWarning)
- Matching sous-chaîne Pearl → mauvais nœud (Fix : exact match prioritaire)
- Dangling edges silencieuses dans `graph.rs` (Fix : diagnostic)
- `edge_link` O(E²) dans BFS (Fix : HashMap)
- `build(ir)` dupliqué par requête Pearl (Fix : refactoring API)
- `map_or(false,…)` → clippy bloque CI (Fix : `is_some_and`, 1 ligne)

**Tests :** 137 Rust (tous passés) + 231 Python (231 passés, 4 skipped). Pas de régression.
