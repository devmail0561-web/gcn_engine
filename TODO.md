# Plan d'action — GCN Engine v2.3.0 → Production

**Date :** 2026-09-18
**État actuel :** 206/206 tests ✅ | build Rust ✅ | 11 correctifs gradient appliqués
**Métriques connues (pré-v2.3.0) :** node_acc ~99.4% train | edge_acc ~30% | cibles non atteintes
**Métriques cibles prod :** `val_edge_macro_f1 > 0.40` | `val_node_macro_f1 > 0.60` | `val_graph_exact_match > 0.20` | gap train−val < 0.15

---

## Phase A — Sécuriser les correctifs v2.3.0 (priorité immédiate)

> Les 11 correctifs sont vérifiés par raisonnement. Aucun test de régression ne les couvre.
> Sans filet, une future modification peut les régresser silencieusement.

- [ ] **A1** — Test `backward_edge_dx` : vérifier que le gradient de la loss arête
  remonte bien vers `d_enriched[src]` et `d_enriched[dst]` (non-zéro, bonne direction)
- [ ] **A2** — Test reproductibilité dropout : deux `MLPEncoder(seed=42)` avec les mêmes
  inputs produisent les mêmes masques → mêmes sorties
- [ ] **A3** — Test GAT gradient : `backward_message_pass` produit des gradients non-nuls
  sur `a_r` (vecteurs attention) — preuve que le dénominateur n'est plus détaché
- [ ] **A4** — Test `_cached_d_edge_base` / `_cached_d_eff` : vérifier que les offsets
  cachés correspondent bien à `len(edge_vec_base)` et `len(enriched[src_i])`
- [ ] **A5** — Test `_set_training_mode` : `RGCNLayerGAT.training` est bien `False`
  après appel eval, `True` après retour train (via `.train()` PyTorch)

**Critère de sortie :** 5 nouveaux tests verts, aucune régression.

---

## Phase B — Mesurer l'impact réel des correctifs (≤ 30 min machine)

> L'edge_acc ~30% date d'avant le correctif closed-loop. L'impact est inconnu.

- [ ] **B1** — Lancer un entraînement de référence sur `gcn-datasets/real/train/`
  avec les paramètres standards (50 epochs, lr=0.001, --weighted-loss, --val-dir)
- [ ] **B2** — Enregistrer les métriques finales : `val_edge_macro_f1`,
  `val_node_macro_f1`, `val_graph_exact_match`, gap train−val
- [ ] **B3** — Comparer aux chiffres pré-v2.3.0 et documenter le delta

**Critère de sortie :** chiffres mesurés, documentés dans PROGRESS.md.

---

## Phase C — Atteindre les métriques cibles (conditionnel aux résultats B)

> Si `val_edge_macro_f1 < 0.40` après B2, les actions suivantes s'appliquent.

- [ ] **C1** — Enrichir le dataset : 6 relations absentes (motivation, sequence, filter,
  opposition, data_*, control_*) — annoter au moins 30 exemples par relation manquante
- [ ] **C2** — Rééquilibrer les classes rares (prevent=2%, concession=6%) via
  sur-échantillonnage ou augmentation de données
- [ ] **C3** — Hyperparameter tuning : tester `--edge-loss-weight 2.0–5.0`,
  `--label-smoothing 0.05–0.1`, `--rgcn-dropout 0.1–0.3`
- [ ] **C4** — Évaluer le gain du mode `--bidirectional` et `--use-attention` sur val set

**Critère de sortie :** toutes les métriques cibles atteintes sur val set.

---

## Phase D — Robustesse et qualité prod

> Issues MEDIUM identifiées dans PRODUCTION_READINESS_ASSESSMENT.md (v2.0.0).

- [ ] **D1** — Corriger les `assert` restants dans le moteur (remplacer par ValueError/RuntimeError)
- [ ] **D2** — Vérifier la cohérence des `token_span` dans les sorties CIR
  (issue MEDIUM #4-5 spans)
- [ ] **D3** — Test e2e texte brut → CIR JSON : au moins 3 cas (phrase simple,
  phrase complexe, phrase sans causalité) via `GCNBridgeParser`
- [ ] **D4** — Documenter les limitations connues du `GCNBridgeParser` heuristique
  (~80-85% qualité) dans le README pour les utilisateurs

**Critère de sortie :** 0 assert dans le moteur, 3 tests e2e verts.

---

## Phase E — Packaging checkpoint et déploiement

- [ ] **E1** — Entraîner le checkpoint final sur dataset complet (train + val fusionnés)
  une fois les métriques cibles atteintes en Phase C
- [ ] **E2** — Versionner le checkpoint (`model_v2.3.0.npz`) avec ses métriques associées
- [ ] **E3** — Documenter la procédure de chargement checkpoint dans le README
- [ ] **E4** — Bump version → 2.4.0 (premier checkpoint prod documenté)

**Critère de sortie :** checkpoint publié, métriques reproductibles depuis zéro.

---

## Séquence recommandée

```
A (2-3h) → B (30 min machine) → C si nécessaire (2-8h) → D (2-3h) → E (1h)
```

**Chemin nominal (sans enrichissement dataset) :** 6-8h
**Chemin avec enrichissement dataset (Phase C) :** 12-18h

---

## Bloquant hors scope moteur

- **PyO3 binding** : `GCNBridgeParser` appelle le binaire `gcn` via subprocess (~80-85% qualité).
  Le binding PyO3 direct (qualité optimale) reste non implémenté — hors scope pour la v2.x,
  documenté comme limitation connue.
