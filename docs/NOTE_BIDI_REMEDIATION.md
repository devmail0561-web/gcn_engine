# Note — Implémentation Phase 2b : message passing bidirectionnel

**Statut : IMPLÉMENTÉ ✅ (2026-09-16)**

Cette note documente l'état final de l'implémentation. Le plan original a été exécuté intégralement.

---

## Ce qui a été implémenté

### Étape 1 — `constants.py` ✅

Après `RELATION_TYPES` :

```python
RELATION_TYPES_INV = [r + "_inv" for r in RELATION_TYPES]
# ["cause_inv", "enable_inv", "prevent_inv", "condition_inv", "concession_inv",
#  "sequence_inv", "motivation_inv", "filter_inv", "opposition_inv",
#  "data_dependency_inv", "control_dependency_inv"]

ALL_RELATION_TYPES = RELATION_TYPES + RELATION_TYPES_INV  # 22 types
```

**Invariant :**
- `RELATION_TYPES` (11) → classification d'arêtes (MLP Layer 2), supervision, CIR output — **synchronisé avec Rust, NE PAS MODIFIER**
- `ALL_RELATION_TYPES` (22) → `n_relations` du R-GCN/GAT uniquement — **Python-interne, Rust ignore les types `_inv`**
- Indices 0–10 = forward, indices 11–21 = backward (`r_inv = r + 11`)

---

### Étape 2 — `pipeline/cgnp.py` ✅

Paramètre `bidirectional: bool = False` ajouté au constructeur de `CGNPipeline`.

Après la construction de `edge_index` et `edge_type_idxs` (lignes 244–252) :

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

`edge_index_mp` / `edge_types_mp` sont passés au R-GCN/GAT pour le message passing.
`edge_index` / `edge_type_idxs` originaux continuent d'être utilisés pour la loss d'arêtes — supervision inchangée.

**Le MLP edge ne voit jamais les types inverses** : `forward_edge(edge_vec)` prend un vecteur de features, pas `edge_type_idxs`. Il prédit toujours parmi 11 logits.

---

### Étape 3 — `layer3/gat.py` ✅

`RGCNLayerGAT` supporte `n_relations` arbitraire (défaut : `len(RELATION_TYPES)` = 11).

Avec `bidirectional=True` : `n_relations=22`. Shapes résultantes :
- `W_r` : `(22, d_out, d_in)`
- `a_r` : `(22, 2*d_out)`

`load_state()` vérifie `arrays[0].shape == (n_relations, d_out, d_in)` et lève `ValueError` si incompatible. Un checkpoint entraîné avec `n_relations=11` ne peut pas être chargé avec `n_relations=22`.

---

### Étape 4 — `training/train.py` ✅

Deux flags CLI ajoutés :

```
--use-attention/--no-attention       (défaut : --no-attention)
--bidirectional/--no-bidirectional   (défaut : --no-bidirectional)
```

Logique de construction :

```python
n_rel = 22 if bidirectional else len(RELATION_TYPES)
if use_attention:
    graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
else:
    graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
pipeline = CGNPipeline(..., bidirectional=bidirectional)
```

Combinaisons valides :

| Flags | Couche graph | n_relations |
|---|---|---|
| (aucun) | `RGCNLayer` NumPy | 11 |
| `--bidirectional` | `RGCNLayer` NumPy | 22 |
| `--use-attention` | `RGCNLayerGAT` PyTorch | 11 |
| `--use-attention --bidirectional` | `RGCNLayerGAT` PyTorch | 22 |

---

### Étape 5 — `tests/test_gat.py` ✅ (15 tests)

- Shape `W_r == (22, d_out, d_in)` et `a_r == (22, 2*d_out)` avec `n_relations=22`
- `node_repr(bidirectional=True) != node_repr(bidirectional=False)` sur un même graphe
- Checkpoint round-trip avec `n_relations=22`
- `ValueError` levée si shape incompatible au chargement

---

## Couverture après Phase 2b

```
192 / 194 tests Python passent (192 ok, 2 skipped)
137 / 137 tests Rust passent
```
