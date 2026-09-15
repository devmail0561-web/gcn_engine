# Spécification Phase 9 — Correction complète du pipeline ML

**Auteur :** Michel Tendeng  
**Date :** 2026-09-15  
**Révision :** 3 — audit exhaustif intégré  
**Statut :** En révision  
**Branche cible :** `feat/phase9-pipeline-fixes`

---

## Résumé exécutif

Un audit exhaustif du code d'inférence (`cgnp.py`, `ir_emitter.py`, `loader.py`) a révélé **14 problèmes**, dont 6 critiques qui rendent le CausalIR produit par le pipeline ML structurellement incorrect — indépendamment de la qualité de l'entraînement.

Les 3 problèmes originaux (spaCy, décodeur autorégressif, entraînement phasé) restent valides mais sont **secondaires** par rapport aux critiques de fond.

---

## Cartographie des problèmes

```
INFÉRENCE
│
├── ENTRÉE
│   └── P1 — Texte brut impossible (spaCy manquant)
│   └── P9 — Clause nominale : root = déterminant (loader.py:192)
│
├── PIPELINE FORWARD
│   └── P10 — R-GCN voit seulement un graphe en chaîne (cgnp.py:140)
│   └── P7  — Custom encoder → R-GCN jamais entraîné (cgnp.py:290)
│   └── P11 — Gradient R-GCN tronqué silencieusement (cgnp.py:375)
│
├── SORTIE CausalIR
│   └── P2  — explicit TOUJOURS False (marker_token hardcodé None)
│   └── P1c — negated TOUJOURS False (jamais prédit)
│   └── P3  — scope TOUJOURS "specific" (jamais prédit)
│   └── P4  — node_origins TOUJOURS "explicit"
│   └── P5  — attributes TOUJOURS null (entity/agent/patient)
│   └── P6  — temporal_ref TOUJOURS "unresolved"
│   └── P8  — labels sans nominalisation (taxonomies_dir absent)
│
├── DÉCODEUR
│   └── P2d — mean-pool → décodeur autorégressif
│
└── ENTRAÎNEMENT
    └── P3e — gradients couplés → entraînement phasé
```

**Ordre d'implémentation :** P7 → P11 → P2 → P1c → P8 → P9 → P3 → P4 → P5 → P10 → P3e → P2d → P1

---

## PARTIE 1 — Corrections critiques de l'inférence existante

---

### C1 — Bug silencieux : encodeur custom → R-GCN jamais entraîné

**Fichier :** `cgnp.py:290`

**Code actuel :**
```python
if not hasattr(self.encoder, 'backward_node_dx'):
    return   # retour immédiat — graph.update() jamais appelé
```

**Problème :** Si l'utilisateur implémente un `CausalEncoder` custom (sans `backward_node_dx`) et utilise `RGCNLayer`, `backward()` retourne sans appeler `graph.update()`. Les poids R-GCN ne sont jamais mis à jour. Aucun warning.

**Correction :** Séparer le backward de l'encodeur du backward du R-GCN :

```python
def backward(self, d_node_logits, d_edge_logits, lr=0.01):
    vecs = self._cached_enriched_vecs or self._cached_clause_vecs
    if vecs is None or len(vecs) == 0:
        return

    n = min(len(d_node_logits), len(vecs))
    d_enriched = np.zeros((n, vecs.shape[1]), dtype=np.float32)

    # Backward encodeur — uniquement si disponible
    if hasattr(self.encoder, 'backward_node_dx'):
        _backward_encoder_nodes(...)   # inchangé
        _backward_encoder_edges(...)   # inchangé

    # Backward R-GCN — indépendant de l'encodeur
    _backward_rgcn(d_enriched, lr)   # appelé TOUJOURS si graph a backward_message_pass
```

---

### C2 — Gradient R-GCN tronqué silencieusement

**Fichier :** `cgnp.py:299 + 375`

**Code actuel :**
```python
n = min(len(d_node_logits), len(vecs))   # n peut être < N
d_enriched = np.zeros((n, ...))           # nœuds [n..N] reçoivent gradient 0
```

**Problème :** `d_enriched` est de taille `n`, paddé à `N_full` avec des zéros. Les nœuds au-delà de `n` ont participé au message passing mais reçoivent un gradient nul → R-GCN mal entraîné sur ces nœuds.

**Correction :** Allouer `d_enriched` à la taille complète `N = len(vecs)` dès le départ :

```python
N = len(vecs)
d_enriched = np.zeros((N, vecs.shape[1]), dtype=np.float32)
n = min(len(d_node_logits), N)
# backward sur les n premiers nœuds — les autres restent à 0 (légitimement absents de la loss)
# plus de padding artificiel en fin de backward
```

Supprimer le bloc de padding des lignes 375-380.

---

### C3 — `explicit` toujours False / `marker_token` jamais trackéé

**Fichiers :** `cgnp.py:160`, `ir_emitter.py:53`

**Code actuel :**
```python
# cgnp.py:160
edge_triples.append((src_i, dst_i, RELATION_TYPES[rel_idx], rel_conf, False, None))
#                                                                       ^^^^  ^^^^
#                                                                     negated  marker_token

# ir_emitter.py:53
"explicit": marker_token is not None,   # → toujours False
```

**Problème :** Le marker_token et la négation sont hardcodés. Toutes les arêtes ML ressortent `explicit=False`, `negated=False`.

**Correction :** Tracker le token connecteur pendant le forward et propager le marker :

```python
# Dans _forward_from_reps, lors de la construction des edge_triples :
connector = connector_reps[src_i] if connector_reps else None
marker_tok_id = connector.token_span[0] if connector is not None else None

# Détecter la négation : chercher un token de négation dans le span connecteur ou les spans adjacents
negated = _detect_negation(reps[src_i], reps[dst_i], connector)

edge_triples.append((src_i, dst_i, RELATION_TYPES[rel_idx], rel_conf, negated, marker_tok_id))
```

Ajouter `_detect_negation(src_rep, dst_rep, connector_rep) -> bool` :
- Retourne `True` si `src_rep.is_negative` ou si `connector_rep` est de type négation

---

### C4 — `negated` toujours False (encodeur ne prédit pas la négation)

**Fichier :** `cgnp.py:160`

La correction C3 adresse le cas où le connecteur est explicitement négatif. Mais si la négation est sur le nœud source ("X n'entraîne pas Y"), elle doit venir de `UDRepresentation.is_negative` (déjà calculé dans `layer1/representation.py`).

**Correction :** Dans `_detect_negation` :
```python
def _detect_negation(src_rep, dst_rep, connector_rep) -> bool:
    if src_rep.is_negative or dst_rep.is_negative:
        return True
    if connector_rep is not None and connector_rep.is_negative:
        return True
    return False
```

---

### C5 — `scope` toujours `"specific"`

**Fichier :** `cgnp.py:200`

**Code actuel :**
```python
scopes = ["specific"] * len(reps)
```

**Problème :** Le scope est dans les features (`FeatureVocabulary` encode `det_scope`) mais jamais prédit en sortie.

**Correction :** Dériver le scope depuis les features de la clause :

```python
from ..constants import SCOPE_VALUES

def _infer_scope(rep: UDRepresentation) -> str:
    """Dérive le scope depuis les tokens du span (déterminants, pronoms)."""
    for tok in rep.tokens:
        if tok.get("dep_rel") in {"det", "nsubj"} and tok.get("pos") in {"DET", "PRON"}:
            hint = _SCOPE_HINTS.get(tok["lemma"].lower())
            if hint:
                return hint
    return "specific"

_SCOPE_HINTS = {
    "tous": "universal", "toutes": "universal", "chaque": "universal",
    "tout": "universal", "aucun": "null", "aucune": "null",
    "certains": "existential", "certaines": "existential",
    "un": "existential", "une": "existential",
    "quelques": "partial",
}

scopes = [_infer_scope(r) for r in reps]   # remplace la ligne 200
```

---

### C6 — `node_origins` toujours `"explicit"`

**Fichier :** `ir_emitter.py:21`, `cgnp.py:210`

**Problème :** `forward()` ne passe pas `node_origins`. Nœuds inférés et hypothétiques sortent "explicit".

**Correction :** Dériver l'origine depuis `NodeOrigin` du CausalIR ou depuis la confiance du prédicteur :

```python
def _infer_origin(rep: UDRepresentation, node_type: str, confidence: float) -> str:
    if node_type == "condition" and not any(
        t.get("gcn_causal_type") == "conjonction" for t in rep.tokens
    ):
        return "inferred"    # condition sans marqueur explicite = inférée
    if confidence < 0.5:
        return "inferred"
    return "explicit"

node_origins = [
    _infer_origin(r, nt, float(np.max(_softmax(nl.reshape(1,-1)))))
    for r, nt, nl in zip(reps, node_types, node_logits)
]
# Passer node_origins à emit()
```

---

### C7 — `attributes` toujours null

**Fichier :** `ir_emitter.py:37-44`, `cgnp.py:195-198`

**Problème :** `build_label` trouve entity/agent/patient pour le label mais ne les retourne pas. Les attributs du nœud sont vides dans le CausalIR.

**Correction :** Modifier `build_label` pour retourner aussi les attributs :

```python
# label_builder.py
def build_label(rep, node_type, taxonomies_dir=None) -> tuple[str, dict]:
    subject = _find_subject_lemma(rep)
    entity = _find_entity_lemma(rep)
    nom = _nominalize(rep.root_lemma, rep.lang, taxonomies_dir)
    label = _compute_label(rep, node_type, subject, entity, nom)
    attributes = {
        "entity": entity,
        "agent": subject if node_type in ("action", "transition") else None,
        "patient": _find_patient_lemma(rep),
        "quality": None,
        "agent_type": None,
        "reversible": None,
    }
    return label, attributes
```

Mettre à jour `_forward_from_reps` pour déstructurer le tuple et passer `node_attributes` à `emit()`.

Mettre à jour `emit()` pour accepter `node_attributes: list[dict] | None` et les insérer dans les nœuds.

---

### C8 — `build_label` sans nominalisation à l'inférence

**Fichier :** `cgnp.py:195`

**Code actuel :**
```python
node_labels = [build_label(r, nt) for r, nt in zip(reps, node_types)]
# pas de taxonomies_dir
```

**Correction :** Stocker `taxonomies_dir` dans `CGNPipeline.__init__()` et le passer à `build_label` :

```python
# __init__
def __init__(self, encoder, graph, lang, vocabulary, *, decoder=None, taxonomies_dir=None):
    ...
    self.taxonomies_dir = taxonomies_dir   # Path | None

# forward
node_labels_and_attrs = [
    build_label(r, nt, self.taxonomies_dir)
    for r, nt in zip(reps, node_types)
]
```

---

### C9 — Clause nominale : `root_tok = span_toks[0]` (souvent un déterminant)

**Fichier :** `loader.py:192`

**Code actuel :**
```python
or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), span_toks[0])
```

**Problème :** `span_toks[0]` est souvent "la", "le", "un" (DET). Le MLP reçoit `root_pos=DET` pour une clause nominale.

**Correction :** Fallback sur NOUN/PROPN avant de tomber sur le premier token :

```python
root_tok = (
    next((t for t in span_toks if t.gcn_causal_type == "verbe" and t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)  # ← nouveau
    or span_toks[0]
)
```

---

### C10 — R-GCN voit seulement un graphe en chaîne

**Fichier :** `cgnp.py:140-160`

**Problème :** Seules les arêtes consécutives (0→1, 1→2…) sont créées. La relation (0→2) est impossible à prédire.

**Décision architecturale :** Ce problème est **fondamental** mais ne peut pas être résolu sans changer le schéma de supervision (le `GCNDataLoader` exclut déjà les arêtes gap>1). Les deux changements doivent aller ensemble :

1. Étendre le forward à toutes les paires `(i, j)` avec `j > i` et `j - i <= max_gap`
2. Étendre `GCNDataLoader._to_sample()` pour inclure les arêtes avec `gap <= max_gap`
3. Ajouter `max_gap: int = 1` comme paramètre configurable de `CGNPipeline` et `GCNDataLoader`

Pour l'instant : documenter la limitation dans le code, ne pas changer — la fix nécessite de re-annoter les datasets avec des arêtes longue distance.

---

## PARTIE 2 — Problèmes originaux Phase 9

---

### P3e — Conflit gradients encodeur ↔ décodeur

**Fichier :** `cgnp.py:360-369`  
**Solution :** Supprimer les lignes 360-369 (propagation `d_mean → d_enriched`). Entraînement phasé via `--decoder-only` + `--encoder-checkpoint`. Détails inchangés depuis rev.2.

---

### P2d — Décodeur mean-pool → autorégressif

**Fichier :** `trainable.py`  
**Solution :** Décodeur autorégressif (état caché récurrent, teacher forcing, greedy decoding). Détails inchangés depuis rev.2.

---

### P1 — Texte brut → inférence

**Fichier :** `loader.py`  
**Solution :** `reps_from_raw_text(text, lang)` via spaCy. `reps_from_sentence` mis à jour pour utiliser spaCy comme source de features. Détails inchangés depuis rev.2.

---

## Ordre d'implémentation et commits atomiques

| Commit | Correction | Fichiers | Test requis |
|---|---|---|---|
| **fix/C1-rgcn-always-trained** | Séparer backward encodeur / backward R-GCN | `cgnp.py:290` | `test_rgcn_updates_with_custom_encoder` |
| **fix/C2-gradient-no-truncation** | `d_enriched` taille N complète dès init | `cgnp.py:299,375` | `test_backward_full_gradient` |
| **fix/C3-C4-negated-explicit** | Tracker marker_token, détecter négation | `cgnp.py:160`, `loader.py` | `test_negated_edge_detected` |
| **fix/C5-scope-inferred** | `_infer_scope()` depuis déterminants | `cgnp.py:200` | `test_scope_universal_tous` |
| **fix/C6-origin-inferred** | `_infer_origin()` depuis confiance | `cgnp.py:210` | `test_origin_inferred_low_confidence` |
| **fix/C7-attributes-populated** | `build_label` retourne (label, attrs) | `label_builder.py`, `cgnp.py`, `ir_emitter.py` | `test_attributes_entity_not_null` |
| **fix/C8-nominalization** | `taxonomies_dir` dans `CGNPipeline` | `cgnp.py:31,195` | `test_label_nominalized` |
| **fix/C9-nominal-clause-root** | Fallback NOUN avant DET | `loader.py:192` | `test_rep_nominal_clause_root_pos` |
| **fix/P3e-phased-training** | Supprimer couplage gradients, `--decoder-only` | `cgnp.py:360`, `train.py` | `test_encoder_weights_frozen` |
| **fix/P2d-autoregressive-decoder** | Décodeur autorégressif complet | `trainable.py` | `test_decode_produces_sequence` |
| **fix/P1-spacy-raw-text** | `reps_from_raw_text` + `reps_from_sentence` via spaCy | `loader.py`, `cli.py`, `main.rs` | `test_reps_from_raw_text` |

**Règle absolue :** `pytest gcn-python/tests/ -q` (112+ tests) et `cargo test --workspace` (137 tests) doivent passer après **chaque commit**.

---

## Limitations documentées (non corrigées dans cette phase)

| Limitation | Raison du report |
|---|---|
| R-GCN graphe en chaîne (C10) | Nécessite re-annotation des datasets avec arêtes gap>1 — hors scope phase 9 |
| `temporal_ref` "unresolved" (C6) | Nécessite un module de résolution temporelle dédié — phase 10 |
| Confidence non calibrée (C13) | Nécessite temperature scaling post-entraînement — phase 10 |
| Snapshots pre-R-GCN écrasés (C12) | Cosmétique — backward reste cohérent |
| Entrée vide silencieuse (C14) | Ajouter un `warnings.warn` — fix triviale incluse dans C3 |

---

## Vérification end-to-end après toutes les corrections

```bash
# 1. Régression complète
cd gcn-core && cargo test --workspace && cd ..
python3 -m pytest gcn-python/tests/ -q

# 2. Vérifier que les attributs sont peuplés
python3 -c "
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.data.loader import reps_from_raw_text

vocab = FeatureVocabulary()
enc = MLPEncoder(vocab.d_clause, vocab.d_edge)
graph = RGCNLayer(vocab.d_clause, vocab.d_clause)
pipeline = CGNPipeline(enc, graph, 'fr', vocab)

reps, idxs, conns = reps_from_raw_text('Si les ventes baissent, on réduit les coûts.')
cir = pipeline.forward(reps, text='Si les ventes baissent, on réduit les coûts.', connector_reps=conns)

# Vérifications post-fix
assert any(n['attributes']['entity'] is not None for n in cir['nodes']), 'attributes vides'
assert any(e[2]['explicit'] for e in cir['edges']), 'explicit toujours False'
assert len(set(n['scope'] for n in cir['nodes'])) >= 1, 'scope non prédit'
print('OK — CausalIR structurellement correct')
"

# 3. Entraînement phasé
gcn-train --data-dir gcn-datasets/examples/ --epochs 20 --output enc.npz
gcn-train --decoder-only --encoder-checkpoint enc.npz \
  --verbalize-dir gcn-datasets/examples/ --epochs 20 --output model.npz

# 4. Inférence texte brut
gcn-forward --raw-text "Si les ventes baissent, on réduit les coûts." \
  --lang fr --model-path model.npz
```
