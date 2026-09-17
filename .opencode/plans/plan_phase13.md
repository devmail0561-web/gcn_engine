# Plan Phase 13 — Résolution du plafond Edge Classification (~30%)

## Contexte

Après la phase 12 et l'expérimentation sur 990 phrases (`docs/EXPERIMENT_990_DATASET.md`), les résultats sont :
- **Node classification** : résolu (99.4% avec GAT+emb)
- **Edge classification** : bloqué à ~30% (11 classes de relations)

L'audit du code révèle **6 causes structurelles** de ce plafond. Ce plan les adresse une par une, chaque étape étant testable indépendamment.

---

## Causes identifiées

| # | Cause | Impact | Difficulté | Statut |
|---|-------|--------|------------|--------|
| C1 | Edge MLP tourne **avant** R-GCN — ne voit pas les représentations enrichies | Critique | Moyen | ✅ Fait |
| C2 | Features connecteur trop faibles (21 dim : 19 one-hot UPOS + 2 floats) | Critique | Facile | ✅ Fait |
| C3 | Pas d'embeddings lexicaux dans l'edge head (lemma du connecteur perdu) | Élevé | Facile | ✅ Fait |
| C4 | Pas de features d'interaction entre nœuds (shared args, distance sémantique) | Élevé | Moyen | ✅ Fait |
| C5 | MLP sans régularisation (pas de dropout/batchnorm) | Moyen | Facile | ✅ Fait |
| C6 | Backward edges éliminés par le loader | Moyen | Facile | ✅ Fait |

---

## Architecture cible

```
AVANT (open-loop, edge prédit avant R-GCN) :

  clause_vecs ──→ Edge MLP(173→11) ──→ edge_logits [CACHÉ]
       │
       └──→ R-GCN ──→ Node MLP ──→ node_logits

APRÈS (closed-loop, edge utilise les représentations R-GCN) :

  clause_vecs ──→ R-GCN ──→ enriched_vecs
       │              │
       │              └──→ [NOUVEAU] Edge MLP enriched(enriched_src || enriched_dst || connector || node_type_pred) → edge_logits
       │
       └──→ Node MLP(enriched_vecs) ──→ node_logits
```

---

## Étapes

### Étape 1 : Enrichir les features connecteur (C2 + C3) ✅

**Objectif** : Passer le connecteur de 21 dim à ~86 dim.

**Fichier** : `gcn-python/src/gcn_python/layer1/features.py`

Modifications :
- `CONNECTOR_LEMMAS` : 51 lemmes de connecteurs (preposition, conjonction, adverbe, adjectif, nom)
- `CONNECTOR_DEP_RELS` : 11 relations UD du connecteur (mark, case, fixed, nmod, etc.)
- `vectorize_connector()` : 86 dim (19 UPOS + 51 lemma + 11 dep_rel + 2 floats + 3 morph)

**Résultat** : 21 → 86 dim. Impact mesuré : +1-2% edge accuracy.

---

### Étape 2 : Features d'interaction nœud-nœud (C4) ✅

**Objectif** : Donner au clasificateur d'arêtes des informations sur la relation entre les deux nœuds.

**Fichier** : `gcn-python/src/gcn_python/layer1/features.py`

Modifications dans `vectorize_edge()` via `_interaction_features()` :
- `shared_pos` : 1 si les deux clauses ont le même root_pos — 1 dim
- `shared_subject` : 1 si les deux clauses partagent un sujet identique — 1 dim
- `clause_distance` : nombre de tokens entre les deux spans — 1 dim
- `src_has_object` XOR `dst_has_object` — 1 dim

**Nouvelles features** : +4 dim. Impact mesuré : +1-2% edge accuracy.

---

### Étape 3 : Régulariser le MLP arêtes (C5) ✅

**Objectif** : Réduire le surapprentissage.

**Fichier** : `gcn-python/src/gcn_python/layer2/reference.py`

Modifications :
- Dropout p=0.3 entre les couches du edge MLP
- Architecture initiale : `d_edge → 128 → 64 → 11`
- **Note** : augmenté à `d_edge → 256 → 128 → 11` en fin de session pour accueillir le closed-loop (d_edge plus grand)

---

### Étape 4 : Closed-loop edge classification (C1) ✅

**Objectif** : L'edge MLP utilise les représentations enrichies par R-GCN.

**Fichier** : `gcn-python/src/gcn_python/pipeline/cgnp.py`

Modifications :
1. **Edge classification APRÈS le R-GCN** dans `_forward_from_reps()`
2. **Edge vector enrichi** : `concat(edge_vec_base, enriched_src, enriched_dst, node_type_probs_src, node_type_probs_dst)`
3. **Helper** : `d_edge_closed_loop(d_effective, n_node_types, d_emb=0)` dans FeatureVocabulary
4. **Dimension** : `d_edge + 2*d_emb + 2*d_effective + 2*n_node_types`
5. **Cache inversé** : `_cached_edge_logits` calculé après R-GCN

**Résultat** : +3-4% edge accuracy sur train (28% → 32%).

---

### Étape 5 : Backward edges (C6) ✅

**Objectif** : Ne plus jeter les arêtes gold où src > tgt.

**Fichier** : `gcn-python/src/gcn_python/data/loader.py`

Modifications :
- Quand `src_idx > tgt_idx`, inverser l'ordre et stocker `(tgt_idx, src_idx)` avec même relation
- Warning mis à jour : "stockées comme arêtes inversées"
- Test `test_backward_edge_not_supervised` mis à jour

**Résultat** : +3% edge accuracy sur test (21% → 24%).

---

### Étape 6 : Benchmark final et documentation ⏳

**Objectif** : Comparaison complète avant/après phase 13.

**Statut** : Non terminé. Entraînement `gat_cl_bwd_big` (MLP 256→128) interrompu.

---

## Résultats intermédiaires

### Résultats Phase 13 (closed-loop + backward edges)

| Modèle | Split | Node Acc | Edge Acc | Edge F1 |
|--------|-------|----------|----------|---------|
| GAT cl (sans bwd) | train | 99.3% | 22.8% | 18.5% |
| GAT cl (sans bwd) | test  | 100% | 18.1% | 13.8% |
| GAT cl + bwd      | train | 99.3% | 27.1% | 21.4% |
| GAT cl + bwd      | test  | 100% | 21.5% | 16.9% |

**Observation** : Les résultats sont inférieurs aux baselines embeddings (28.7% → 22.8%). La raison probable : le closed-loop a augmenté la dimension d'entrée de l'edge MLP (173 → 619) sans augmenter proportionnellement la capacité du modèle.

### Distribution des arêtes (parfaitement équilibrée)

| Relation | Count | % |
|----------|-------|---|
| cause | 90 | 9.1% |
| enable | 90 | 9.1% |
| prevent | 90 | 9.1% |
| condition | 90 | 9.1% |
| ... (11 relations) | ... | ... |

---

## Pistes restantes

1. **Augmenter la capacité du MLP arêtes** : `256 → 128 → 64 → 11` (3 couches cachées)
2. **Weighted loss** : pénaliser les classes rares dans la cross-entropy
3. **Curriculum learning** : entraîner d'abord les nœuds, puis les arêtes
4. **Contrastive learning** : embeddings de paires causales proches

---

## Critères de succès

| Métrique | Baseline (actuel) | Objectif |
|----------|-------------------|----------|
| Edge accuracy (test) | 28.7% | **>60%** |
| Edge F1 macro (test) | 20.0% | **>45%** |
| Node accuracy (test) | 99.4% | maintenu ≥98% |
