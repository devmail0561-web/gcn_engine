# Note — Plan de remédiation : constantes bidirectionnel

**Constat :** `constants.py` ne contient que `RELATION_TYPES` (11 types forward).
Les constantes `RELATION_TYPES_INV` et `ALL_RELATION_TYPES` nécessaires à la Phase 2b
(message passing bidirectionnel) sont absentes. Les paramètres `bidirectional` dans
`cgnp.py` et `train.py` ne sont pas encore implémentés non plus.

---

## Étape 1 — `constants.py`

Ajouter après `RELATION_TYPES` :

```python
RELATION_TYPES_INV = [r + "_inv" for r in RELATION_TYPES]
# ["cause_inv", "enable_inv", "prevent_inv", "condition_inv", "concession_inv",
#  "sequence_inv", "motivation_inv", "filter_inv", "opposition_inv",
#  "data_dependency_inv", "control_dependency_inv"]

ALL_RELATION_TYPES = RELATION_TYPES + RELATION_TYPES_INV  # 22 types
```

**Règle d'index :** indices 0–10 = forward, indices 11–21 = backward (`r_inv = r + 11`).

**Invariant à préserver :**
- `RELATION_TYPES` (11) → classification d'arêtes (MLP Layer 2), supervision, CIR output — **synchronisé avec Rust, ne pas modifier**
- `ALL_RELATION_TYPES` (22) → `n_relations` du R-GCN/GAT uniquement — **Python-interne, Rust ignore `_inv`**

---

## Étape 2 — `pipeline/cgnp.py`

Ajouter `bidirectional: bool = False` au constructeur de `CGNPipeline`.

Après la construction de `edge_index` et `edge_type_idxs`, insérer la duplication :

```python
if self.bidirectional and edge_index.shape[1] > 0:
    rev_index    = edge_index[[1, 0], :]
    rev_types    = edge_type_idxs + len(self.relation_types)   # 11–21
    edge_index_mp = np.concatenate([edge_index, rev_index], axis=1)
    edge_types_mp = np.concatenate([edge_type_idxs, rev_types])
else:
    edge_index_mp = edge_index
    edge_types_mp = edge_type_idxs
```

`edge_index_mp` / `edge_types_mp` sont passés au R-GCN/GAT seulement.
`edge_index` / `edge_type_idxs` originaux restent utilisés pour la loss d'arêtes — supervision inchangée.

---

## Étape 3 — `layer3/gat.py` et `layer3/pytorch_rgcn.py`

Passer `n_relations=22` quand `bidirectional=True` :

```python
graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=22)
```

`W_r` → shape `(22, d_out, d_in)`, `a_r` → shape `(22, 2*d_out)`.

Ajouter dans `load_state()` une vérification de shape et une `ValueError` explicite
si `n_relations` du checkpoint ne correspond pas à l'instance courante.

`RGCNLayer` (NumPy, `reference.py`) supporte déjà `n_relations` — aucune modification nécessaire.

---

## Étape 4 — `training/train.py`

Ajouter deux flags CLI :

```
--use-attention/--no-attention       (défaut : --no-attention)
--bidirectional/--no-bidirectional   (défaut : --no-bidirectional)
```

Logique de construction :

```python
n_rel = 22 if bidirectional else len(RELATION_TYPES)
if use_attention:
    graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
elif use_pytorch:
    graph = RGCNLayerPT(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
else:
    graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
pipeline = CGNPipeline(..., bidirectional=bidirectional)
```

---

## Étape 5 — Tests

Modifier `tests/test_gat.py` :

- Shape `W_r == (22, d_out, d_in)` et `a_r == (22, 2*d_out)` avec `n_relations=22`
- `node_repr(bidirectional=True) != node_repr(bidirectional=False)` sur un même graphe
- Checkpoint round-trip avec `n_relations=22`
- `ValueError` levée si shape incompatible au chargement

---

## Ordre d'implémentation recommandé

1. `constants.py` (étape 1) — prérequis de tout le reste
2. `cgnp.py` (étape 2) — logique de duplication des arêtes
3. `gat.py` / `pytorch_rgcn.py` (étape 3) — support 22 relations + garde checkpoint
4. `train.py` (étape 4) — flags CLI
5. Tests (étape 5) — validation

Chaque étape doit être auditée avant de passer à la suivante.
