# Plan d'implémentation — Verbalizer : CIR → texte

## Contexte et objectif

Le moteur prédit un CIR (nœuds typés + arêtes typées). L'objectif est d'apprendre l'inverse : étant donné le CIR prédit, générer du texte naturel. `TrainableDecoder` (RNN + attention, `verbalizer/trainable.py`) existe mais est **mort en inférence** et entraîné sur une **mauvaise distribution**.

**Décisions confirmées :**
- **R1** : enriched_vecs réels (pas one-hot) — seul régime d'entraînement du décodeur
- **Teacher forcing** : sans réinjection de tokens pour l'instant (Phase 3) ; N5 documenté comme limite architecturale
- **Checkpoint RNG** : reprise non-exacte acceptable (poids restaurés, RNG frais)
- **Inférence opt-in** : `engine.verbalize(text, use_neural=True)` — template reste le défaut

**Valeur exacte du mismatch** : 7 dims (one-hot) vs **79 dims** (`d_clause = 18+38+5+5+4+5+1+3`, la valeur 75 dans les commentaires est périmée).

---

## Clarification bug (a) — Checkpoint layer0/1 : NON-BUG

`_LinearLayer.W` et `.b` sont des arrays numpy directs (layer2/reference.py:18-19), pas des propriétés. `parameters()` retourne des références directes. `load_checkpoint()` fait `p[:] = data[f"decoder_{i}"]` (in-place) → les poids sont correctement restaurés.

**Action unique :** renforcer `test_checkpoint_roundtrip_with_decoder` pour vérifier les valeurs numériques des poids, pas seulement les shapes.

---

## Phase 0 — Corrections bloquantes dans `trainable.py`

### Fix 1 — Double normalisation du gradient `/T²` (lignes 236, 323-329)

`loss_decode` retourne `d_logits / valid_steps` → `backward_decode` redivise par `n_steps` → gradient en `/T²`. Seul le cas 1D est épargné (`n_steps=1`).

**Fix :** `loss_decode` retourne le scalaire loss divisé mais `d_logits` **brut** :
```python
# ligne 216 — cas 1D
return total_loss / N, d_logits        # d_logits sans /N

# ligne 236 — cas 2D
return total_loss / valid_steps, d_logits   # d_logits sans /valid_steps
```
`backward_decode` normalise tout par `n_steps` → une seule division, cohérente.

### Fix 2 — `d_node_embs` non moyenné, échelle incohérente (ligne 330)

`backward_decode` retourne `d_node_embs_total` sans diviser par `n_steps`, alors que `grads` et `d_attn_vec` le sont → gradient S11 vers l'encodeur à l'échelle `T×` trop grand.

**Fix :** `return d_node_embs_total / n_steps, grads, d_attn_vec` (ligne 330).

### Fix 3 — EOS jamais supervisé (`encode()` ligne 36-37)

`encode()` n'ajoute jamais EOS. Le modèle ne prend jamais EOS comme cible → `decode()` boucle toujours jusqu'à `max_decode_len`.

**Fix :** dans la boucle training Path B, ajouter EOS au gold **après** l'encodage :
```python
_eos = pipeline.decoder.vocab._t2i.get(SurfaceVocabulary.EOS, -1)
_gold = (np.append(vsample.gold_tokens, _eos).astype(np.int64)
         if _eos >= 0 else vsample.gold_tokens)
```
Ne pas modifier `encode()` (rétrocompat inférence inchangée).

### Fix 4 — Init zéro de `W_query` tue le gradient au pas 0 (lignes 116-117)

`W_query = zeros` → `d_h_from_attn = W_query @ d_query = 0` au step 0 → le gradient ne remonte pas vers `h_prev` via l'attention.

**Fix :** Xavier init dans `_init_layers()` :
```python
scale = np.sqrt(2.0 / (self.d_hidden + d_in))
self._W_query = self._rng.normal(0, scale, (self.d_hidden, d_in)).astype(np.float32)
```
`attn_vec` reste à zéro (attention uniforme au départ = comportement sensé).

### Fix 5 — Cache périmé sur entrée vide — N2 (ligne 148-149)

`forward_decode([])` retourne sans vider `_rnn_step_cache` → backward suivant rétropropage sur l'ancien exemple.

**Fix :** vider les caches avant le `return` précoce :
```python
if len(node_embeddings) == 0:
    self._rnn_step_cache = []
    self._cached_node_embs = None
    return np.zeros(len(self.vocab), dtype=np.float32)
```

### Fix 6 — Double sémantique de `decode()` — N3/f (lignes 375-381)

`decode()` non-initialisé → top-10 single-step (comportement différent du greedy multi-step post-init) → non-idempotent, deux sorties pour la même entrée.

**Fix :** supprimer la branche non-initialisée ; appeler `forward_decode(node_embeddings)` pour initialiser, puis enchaîner le greedy :
```python
def decode(self, node_embeddings: np.ndarray) -> str:
    if len(node_embeddings) == 0:
        return ""
    if self._layers is None:
        self._init_layers(node_embeddings.shape[1])
    # greedy multi-step unifié (lignes 387-403 existantes)
    ...
```

### Fix 7 — `_d_W_query` écrasé en accumulation mini-batch (ligne 323)

`backward_decode` fait `self._d_W_query = d_W_query_total / n_steps` (écrasement). En mini-batch, N appels successifs écrasent au lieu d'accumuler → W_query n'apprend que sur le dernier sample.

**Fix :** accumuler dans `backward_decode`, normaliser dans `update()` :
```python
# backward_decode ligne 323 — AVANT :
self._d_W_query = d_W_query_total / n_steps
# APRÈS :
self._d_W_query = (
    (self._d_W_query + d_W_query_total) if self._d_W_query is not None
    else d_W_query_total.copy()
)
self._n_wq_steps = getattr(self, '_n_wq_steps', 0) + n_steps

# update() ligne 350 — normaliser au moment du pas :
self._W_query -= lr * (self._d_W_query / max(self._n_wq_steps, 1))
self._d_W_query = None
self._n_wq_steps = 0
```

### Fix 8b — `update()` troncature silencieuse — N4 (ligne 353)

`zip(layers, grads)` coupe sans erreur si `len(grads) != 2`.

**Fix :** ajouter une garde :
```python
if len(grads) != len(self._layers):
    raise ValueError(
        f"update() : {len(grads)} groupes de gradients pour {len(self._layers)} couches."
    )
```

### Fix 8 — `d_in` strict — mismatch silencieux (lignes 114-115, 150-151)

`_init_layers` se verrouille au premier appel. Appels suivants avec `d_in` différent sont ignorés silencieusement → crash différé au matmul.

**Fix :** vérifier `d_in` à chaque appel de `forward_decode` :
```python
if self._last_d_in is not None and node_embeddings.shape[1] != self._last_d_in:
    raise ValueError(
        f"forward_decode : d_in={node_embeddings.shape[1]} incompatible "
        f"avec d_in={self._last_d_in} verrouillé à l'init."
    )
```

### Fix 9 — N1 : |V| gelé si vocab vide à la construction

`layer1` = `_LinearLayer(d_hidden, len(vocab))` → si vocab non construit, `len=2` (PAD+UNK) et les tokens gold >= 2 sont silencieusement ignorés dans `loss_decode`.

**Fix :** documenter le contrat de construction dans `__init__` (guard + warning si `len(vocab) <= 2`) :
```python
if len(vocab) <= 2:
    warnings.warn(
        "TrainableDecoder : vocab contient seulement PAD/UNK. "
        "Appeler vocab.build(surfaces) avant de construire le décodeur.",
        UserWarning, stacklevel=2,
    )
```

---

### Fix 10 — B3 : `backward_accumulate` ne propage pas S11 (`cgnp.py:836-849`)

`backward()` additionne `d_node_embs` dans `d_enriched` et le propage vers le R-GCN (lignes 663-676). `backward_accumulate` accumule `dec_grads`/`d_attn_vec` mais ne touche jamais `d_enriched` → en mini-batch (`mini_batch_size > 1`), le gradient décodeur→encodeur est silencieusement perdu.

**Fix :** dans `backward_accumulate`, après le bloc décodeur (lignes 797-810), propager `d_node_embs` vers `d_enriched` avec le même pattern que `backward()` :
```python
# Après backward_decode en backward_accumulate :
if d_node_embs is not None and d_enriched is not None and d_node_embs.shape == d_enriched.shape:
    d_enriched += d_node_embs
```
Ajouter un test `test_joint_mini_batch_equiv` : mini_batch=1 vs mini_batch=4 produisent les mêmes poids à tolérance numérique.

---

## Phase 1 — Corriger l'entraînement Path B

### `gcn-python/src/gcn_python/data/verbalize_loader.py`

Ajouter `node_labels: list[str]` à `VerbalizeSample`. Extraire `[n["label"] for n in nodes]` dans le loader. Garder `node_type_embeddings` (rétrocompat tests).

```python
@dataclass
class VerbalizeSample:
    ir_json: str
    node_type_embeddings: np.ndarray   # (N, 7) one-hot — conservé
    gold_tokens: np.ndarray
    source_text: str
    node_labels: list[str]             # NOUVEAU
```

Corriger également : le loader prend actuellement `_surfaces[0]` seulement (train.py:419) et jette les surfaces supplémentaires. Utiliser toutes les surfaces via `verb_source_map` (itérer sur la liste complète, pas juste `[0]`).

### `gcn-python/src/gcn_python/training/train.py`

**Helper module-level :**
```python
def _minimal_reps_from_labels(node_labels: list[str], node_types: list[str]) -> list:
    """UDRepresentation minimaux depuis labels CIR (approximation — voir note plan).
    root_lemma = premier token du label. Distribution train ≈ inférence (même pipeline,
    même dimension 79). Amélioration cible : stocker enriched_vecs dans verbalize_real.json.
    """
    from ..layer1.representation import UDRepresentation
    reps = []
    for label, ntype in zip(node_labels, node_types):
        lemma = label.split()[0] if label.strip() else ntype
        reps.append(UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None,
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, max(1, len(label.split()))),
        ))
    return reps
```

**Remplacement boucle verbalize (lignes 472-483) :**
```python
if verb_loader is not None and pipeline.decoder is not None:
    from ..verbalizer.trainable import SurfaceVocabulary as _SV
    for vsample in verb_loader:
        if len(vsample.gold_tokens) == 0:
            continue
        import json as _json
        _cir = _json.loads(vsample.ir_json)
        _ntypes = [n.get("node_type", "entite") for n in _cir.get("nodes", [])]
        _reps = _minimal_reps_from_labels(vsample.node_labels, _ntypes)
        if not _reps:
            continue
        pipeline.forward(_reps, vsample.source_text)
        _enriched = pipeline.get_enriched_vectors()
        if _enriched is None:
            continue
        # Fix 3 : ajouter EOS au gold
        _eos = pipeline.decoder.vocab._t2i.get(_SV.EOS, -1)
        _gold = (np.append(vsample.gold_tokens, _eos).astype(np.int64)
                 if _eos >= 0 else vsample.gold_tokens)
        dec_logits = pipeline.decoder.forward_decode(_enriched, _gold)
        dec_loss, d_dec = pipeline.decoder.loss_decode(dec_logits, _gold)
        if not np.isfinite(dec_loss):
            continue
        _, dec_grads, d_attn_vec = pipeline.decoder.backward_decode(d_dec)
        pipeline.decoder.update(dec_grads, d_attn_vec, lr)
        epoch_loss += dec_loss
        n_samples += 1
```

Propager `gold_surface` en **validation** également (actuellement absent, `dec_loss` non suivi).

---

## Phase 2 — Brancher l'inférence (`engine.py`)

```python
def verbalize(self, text: str, use_neural: bool = False) -> dict:
    """
    Texte brut → CIR prédit + texte verbalisé.

    use_neural=False (défaut) : ReferenceDecoder (templates, toujours disponible).
    use_neural=True : TrainableDecoder si chargé depuis checkpoint, sinon repli template.

    Requiert gcn-cli (ou text_parser passé au constructeur).
    Retourne {"cir": <CausalIR dict>, "text": <str>}.

    Note N5 : le décodeur est un RNN piloté par pooling de nœuds sans réinjection de
    tokens — qualité de génération limitée. Vrai teacher forcing = Phase 3.
    """
    cir = self._pipeline.analyze(text, text_parser=self._text_parser)
    if use_neural and self._pipeline.decoder is not None:
        enriched = self._pipeline.get_enriched_vectors()
        if enriched is not None:
            return {"cir": cir, "text": self._pipeline.decoder.decode(enriched)}
    from .verbalizer.decoder import ReferenceDecoder
    return {"cir": cir, "text": ReferenceDecoder().decode_cir(cir) or ""}
```

---

## Phase 3 — Tests

### `tests/test_verbalize_e2e.py` (NOUVEAU, 6 tests)

1. **`test_node_labels_in_loader`** — `node_labels` est `list[str]`, `len == N`
2. **`test_path_b_uses_enriched_vecs`** — shape `(N, 79)`, `forward_decode` sans crash
3. **`test_gradient_single_normalization`** — après fix 1+2 : `d_node_embs`, `grads[0][0]`, `d_attn_vec` ont la même échelle (ratio ~1, pas ~T)
4. **`test_decode_idempotent`** — deux appels identiques à `decode()` retournent la même chaîne
5. **`test_engine_verbalize_neural`** — `{"cir": dict, "text": str}` avec décodeur + text_parser mock
6. **`test_engine_verbalize_fallback`** — `pipeline.decoder is None` → repli template, pas de crash

### Renforcements dans les tests existants

- `test_trainable_decoder.py::test_checkpoint_roundtrip_with_decoder` : vérifier les valeurs numériques des poids (pas seulement les shapes)
- `test_verbalize_loader.py` : ajouter `test_node_labels_aligned` (`len(node_labels) == N`)

---

## Ce qui ne change PAS

- `cgnp.py`, couches 1–3, `ir_emitter.py` — **intouchés**
- `training/checkpoint.py` — **inchangé** (correct, `p[:]=data[...]` fonctionne)
- Path A (`pipeline.loss(..., gold_surface=)`) — **inchangé**
- Format CIR, `discuss.py`, `cli.py` — **inchangés**
- Tests existants — **conservés**, certains renforcés

---

## Limitation connue (N5) et améliorations futures

**N5 — LM non-conditionné :** ni le token gold (train) ni le token prédit (inférence) ne sont réinjectés. Le décodeur est un RNN piloté uniquement par pooling attentionnel des nœuds. Plafond de qualité bas. Phase 3 : embedding du token précédent + vrai teacher forcing.

**Distribution train/inférence :** `_minimal_reps_from_labels` produit des enriched_vecs synthétiques (79 dims, bonne dimension). Amélioration cible : stocker `"enriched_vecs"` pré-calculés dans `verbalize_real.json` lors de l'annotation.

---

## Vérification

```bash
cd gcn-python

# Phase 0 : fixes trainable.py + tests existants renforcés
python -m pytest tests/test_trainable_decoder.py -v

# Phase 1-3 : nouveaux tests e2e
python -m pytest tests/test_verbalize_e2e.py -v

# Suite complète
python -m pytest tests/ -q
# Attendu : 244+ passed, 0 failed
```
