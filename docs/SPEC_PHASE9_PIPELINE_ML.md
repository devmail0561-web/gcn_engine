# Spécification Phase 9 — Correction complète du pipeline ML

**Auteur :** Michel Tendeng  
**Date :** 2026-09-15  
**Révision :** 4 — implémenté + corrections post-audit  
**Statut :** Implémenté (commits 3bafb75, 12691c6)  
**Branche cible :** `master`

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

**Ordre d'implémentation :** P7 → P11 → P2 → P1c → P8 → P9 → P3 → P4 → P5 → P3e → P2d → P1

> **Note :** P10 (R-GCN graphe en chaîne) est retiré de l'ordre d'implémentation — voir table des limitations en fin de document.

### Correspondance numérotation P-x → C-x

| Problème original | Correction | Description |
|---|---|---|
| P7 | C1 | Encodeur custom → R-GCN jamais entraîné |
| P11 | C2 | Gradient R-GCN tronqué |
| P2 | C3 | `explicit` toujours False |
| P1c | C4 | `negated` toujours False |
| P8 | C8 | Labels sans nominalisation |
| P9 | C9 | Clause nominale : root = DET |
| P3 | C5 | `scope` toujours "specific" |
| P4 | C6 | `node_origins` toujours "explicit" |
| P5 | C7 | `attributes` toujours null |
| P10 | — | R-GCN graphe en chaîne (déféré) |
| P3e | P3e | Gradients couplés encodeur/décodeur |
| P2d | P2d | Décodeur mean-pool → autorégressif |
| P1 | P1 | Texte brut → inférence (spaCy) |

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

**Correction :** Remplacer le retour silencieux par un `warnings.warn` explicite. Le backward R-GCN **n'est pas indépendant de l'encodeur** : `d_enriched` est calculé via `encoder.backward_node_dx()`. Sans ce gradient, passer des zéros au R-GCN ne l'entraînerait pas. La contrainte d'interface est donc documentée explicitement.

```python
def backward(self, d_node_logits, d_edge_logits, lr=0.01):
    if not hasattr(self.encoder, 'backward_node_dx'):
        # Le R-GCN ne peut être rétropropagé sans backward_node_dx :
        # d_enriched (gradient vers R-GCN) provient du backward de l'encodeur.
        # Un encodeur sans cette méthode doit gérer son propre backward + graph.update().
        import warnings
        warnings.warn(
            "CGNPipeline.backward() : encodeur sans backward_node_dx — "
            "les poids R-GCN ne sont pas mis à jour par ce backward. "
            "Implémenter backward_node_dx ou appeler graph.update() manuellement.",
            UserWarning,
            stacklevel=2,
        )
        return

    # ... suite inchangée (vecs, d_enriched, backward encodeur, backward R-GCN)
```

**Contrainte d'interface documentée :** Pour utiliser `CGNPipeline.backward()` avec une implémentation custom de `CausalEncoder`, celle-ci doit exposer `backward_node_dx`. Sans cela, le R-GCN n'est pas entraîné via ce pipeline — le data scientist doit gérer le backward lui-même.

---

### C2 — Gradient R-GCN tronqué silencieusement

**Fichier :** `cgnp.py:299 + 375`

**Code actuel :**
```python
n = min(len(d_node_logits), len(vecs))   # n peut être < N
d_enriched = np.zeros((n, ...))           # nœuds [n..N] reçoivent gradient 0
```

**Sévérité reclassée :** amélioration défensive, pas un bug actif. Dans le flow d'entraînement normal, `len(d_node_logits) == len(vecs)` toujours — `d_node_logits` provient de `loss()` qui reçoit les logits du `forward()`, lesquels ont exactement `N` lignes. Le cas n < N ne se produit que si le caller tronque `d_node_logits` manuellement. La correction reste pertinente pour robustesse.

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

La correction C3 adresse le cas où le connecteur est explicitement négatif. Mais si la négation est sur le nœud source ("X n'entraîne pas Y"), elle doit venir de `UDRepresentation.is_negative` (déjà calculé dans `layer1/representation.py` : `root_morph.get("Polarity", "") == "Neg"`).

**Correction :** Dans `_detect_negation` :
```python
def _detect_negation(src_rep, dst_rep, connector_rep) -> bool:
    if src_rep.is_negative or dst_rep.is_negative:
        return True
    if connector_rep is not None and connector_rep.is_negative:
        return True
    return False
```

**Limitation documentée :** `is_negative` repose sur le morphème UD `Polarity=Neg`. Les négations analytiques françaises ("ne...pas") dont "pas" n'est pas le root de la clause ne sont pas détectées par cette heuristique. Pour les couvrir, il faudrait chercher un token avec `dep_rel == "advmod"` et `lemma in {"pas", "jamais", "plus", "guère"}` dans le span — hors scope C4, à traiter en phase 10.

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
_SCOPE_HINTS = {
    "tous": "universal", "toutes": "universal", "chaque": "universal",
    "tout": "universal", "aucun": "null", "aucune": "null",
    "certains": "existential", "certaines": "existential",
    "un": "existential", "une": "existential",
    "quelques": "partial",
}

def _infer_scope(rep: UDRepresentation) -> str:
    """Dérive le scope depuis les tokens du span (déterminants, pronoms)."""
    for tok in rep.tokens:
        if tok.get("dep_rel") in {"det", "nsubj"} and tok.get("pos") in {"DET", "PRON"}:
            hint = _SCOPE_HINTS.get(tok["lemma"].lower())
            if hint:
                return hint
    return "specific"

scopes = [_infer_scope(r) for r in reps]   # remplace la ligne 200

# Note : _SCOPE_HINTS couvre uniquement le français. Support multilingue à ajouter en phase 10.
```

---

### C6 — `node_origins` toujours `"explicit"`

**Fichier :** `ir_emitter.py:21`, `cgnp.py:210`

**Problème :** `forward()` ne passe pas `node_origins`. Nœuds inférés et hypothétiques sortent "explicit".

**Correction :** Dériver l'origine depuis la présence d'un connecteur explicite dans le texte source — pas depuis la confiance du modèle (`confidence` ne détermine pas si un marqueur existe dans la phrase).

```python
def _infer_origin(
    node_type: str,
    connector_rep: UDRepresentation | None,
) -> str:
    # Un nœud "condition" sans connecteur dans le texte = relation inférée.
    # Tous les autres types : explicit par défaut (le nœud correspond à un span annoté).
    if node_type == "condition" and connector_rep is None:
        return "inferred"
    return "explicit"

node_origins = [
    _infer_origin(nt, connector_reps[i] if connector_reps and i < len(connector_reps) else None)
    for i, nt in enumerate(node_types)
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

def _find_patient_lemma(rep: UDRepresentation) -> str | None:
    """Premier token avec dep_rel obj/iobj/nobj — patient syntaxique de la clause."""
    for t in rep.tokens:
        if t.get("dep_rel") in {"obj", "iobj", "nobj"}:
            return t["lemma"]
    return None

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

**Précision :** `build_label` accepte déjà `taxonomies_dir: Path | None = None` dans sa signature actuelle (`label_builder.py:13`). Le seul changement nécessaire est d'ajouter `taxonomies_dir` à `CGNPipeline.__init__()` et de le passer à l'appel.

**Correction :**

```python
# __init__
def __init__(self, encoder, graph, lang, vocabulary, *, decoder=None, taxonomies_dir=None):
    ...
    self.taxonomies_dir = taxonomies_dir   # Path | None

# forward — ligne 197-200 de cgnp.py
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

**Fichier :** `cgnp.py` (bloc `d_mean → d_enriched`)

**Solution :** Supprimer uniquement les 5 lignes qui propagent `d_mean` dans `d_enriched`. Conserver `decoder.backward_decode()` et `decoder.update()` — le décodeur doit continuer à se mettre à jour. Entraînement phasé via flag `--decoder-only` + `--encoder-checkpoint` dans `train.py`.

**Implémentation :**
```python
# Dans backward() — bloc P3e corrigé
# Le décodeur est mis à jour indépendamment du R-GCN.
if (self.decoder is not None
        and self._cached_decode_gradient is not None
        and hasattr(self.decoder, 'backward_decode')):
    _, dec_grads = self.decoder.backward_decode(self._cached_decode_gradient)
    self.decoder.update(dec_grads, lr)
    # NOTE : d_mean n'est PAS propagé vers d_enriched
```

**`train.py` — flags ajoutés :**
```bash
gcn-train --decoder-only --encoder-checkpoint enc.npz \
  --verbalize-dir gcn-datasets/examples/ --epochs 20 --output model.npz
```

---

### P2d — Décodeur mean-pool → autorégressif

**Fichier :** `trainable.py`

**Architecture implémentée :** RNN à un pas avec contexte mean-pool.

```python
# Étape RNN : h_t = tanh(W_rnn @ [context; h_{t-1}] + b_rnn)
# Sortie   : logit_t = W_out @ h_t + b_out
# context  = mean_pool(node_embeddings)  — (D_in,)
# h_0      = zeros(d_hidden)

def _rnn_step(context, h_prev):
    rnn_in = np.concatenate([context, h_prev])  # (D_in + D_hidden,)
    z1 = W_rnn @ rnn_in + b_rnn
    h_new = np.tanh(z1)
    logits = W_out @ h_new + b_out
    return h_new, logits
```

**Teacher forcing (entraînement) :** `loss()` appelle `forward_decode(vecs, gold_surface)` — T pas, retourne `(T, |V|)` logits.

**Greedy decoding (inférence) :** boucle jusqu'à `<eos>` ou `max_decode_len` (défaut : 20).

**BPTT :** gradient `dx_rnn[d_in:]` (w.r.t. `h_prev`) propagé à l'étape précédente via `d_h_next`.

**SurfaceVocabulary :** `<eos>` ajouté en fin de vocabulaire dans `build()` (pas dans `__init__`) pour éviter de décaler les indices des tokens utilisateur existants. `to_json`/`from_json` inclut `max_decode_len`.

---

### P1 — Texte brut → inférence

**Fichier :** `loader.py`  
**Solution :** `reps_from_raw_text(text, lang)` via spaCy. `reps_from_sentence` mis à jour pour utiliser spaCy comme source de features. Détails inchangés depuis rev.2.

---

## Ordre d'implémentation — commits réels

| Commit | Corrections | Tests ajoutés |
|---|---|---|
| **3bafb75** | C1 C2 C3/C4 C5 C6 C7 C8 C9 P3e P2d P1 | +8 tests phase 9 (120 total) |
| **12691c6** | 10 corrections post-audit code-review (max) | — (120 tests maintiennent) |

**Couverture tests Python :** 120 / 120 passent (`pytest gcn-python/tests/`)  
**Couverture tests Rust :** 137 / 137 passent (`cargo test --workspace`)

---

## Corrections post-audit (10 findings — code-review max)

| Finding | Correction | Commit |
|---|---|---|
| P3e supprimait decoder.update() | Restauré — seul d_mean→d_enriched supprimé | 12691c6 |
| BPTT : gradient h_prev ignoré | dx_rnn[d_in:] propagé via d_h_next | 12691c6 |
| forward_decode sans gold_tokens | loss() appelle forward_decode(vecs, gold_surface) | 12691c6 |
| _infer_origin mauvais index | connector_reps[i] OR connector_reps[i-1] | 12691c6 |
| EOS à l'index 2 décalait les tokens | EOS ajouté à la fin dans build(), pas __init__ | 12691c6 |
| to_json sans max_decode_len | Ajouté dans to_json/from_json | 12691c6 |
| reps_from_raw_text : lang silencieuse | UserWarning pour lang hors fr/en | 12691c6 |
| decode() sans garde node_embs vide | Retour "" immédiat si vide | 12691c6 |
| train.py : validation ordre | --decoder-only validé avant load_checkpoint | 12691c6 |
| Cause racine des #1/#8 | Même fix que #1 | 12691c6 |

---

## Limitations documentées (non corrigées dans cette phase)

| Limitation | Raison du report |
|---|---|
| R-GCN graphe en chaîne (C10) | Nécessite re-annotation des datasets avec arêtes gap>1 — hors scope phase 9 |
| `temporal_ref` "unresolved" (P6) | Nécessite un module de résolution temporelle dédié — phase 10 |
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

text = 'Si les ventes baissent, tous les coûts augmentent.'
reps, idxs, conns = reps_from_raw_text(text)
cir = pipeline.forward(reps, text=text, connector_reps=conns)

# Vérifications post-fix
assert any(n['attributes']['entity'] is not None for n in cir['nodes']), 'attributes vides'
assert any(e[2]['explicit'] for e in cir['edges']), 'explicit toujours False'
# "tous" → scope=universal attendu sur au moins un nœud
assert any(n['scope'] != 'specific' for n in cir['nodes']), 'scope non prédit (tous restent specific)'
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
