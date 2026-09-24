# Benchmark complet — GCN Engine (2026-09-18)

**Objectif :** classifier les relations causales entre clauses françaises en 11 types de relations
et 7 types de nœuds, à partir de 536 phrases originales (735 avec oversampling C1).

**Métriques cibles production :**

| Métrique | Cible | Meilleure valeur atteinte | Statut |
|---------|-------|--------------------------|--------|
| `val_edge_macro_f1` | > 0.40 | **0.468** | ✅ |
| `val_node_macro_f1` | > 0.60 | 0.313 | ✗ |
| `val_graph_exact_match` | > 0.20 | 0.132 | ✗ |
| gap train−val edge | < 0.15 | 0.069 | ✅ |

---

## 1. Résultats triés par val_edge_macro_f1 (meilleur → pire)

> **Seules les lignes avec val monitoring sont comparables.** Les runs sans val (marqués `—`)
> mesurent la mémorisation du dataset d'entraînement, pas la généralisation.

### 1a. Runs AVEC val monitoring

| # | Configuration | Dataset | Epochs | LR | best_val_edge_f1 | @ep | val_node_f1 | val_gem | gap_edge |
|---|--------------|---------|--------|-----|-----------------|-----|-------------|---------|---------|
| 1 | **GAT + bidi + oversamp C1** | 735 oversampled | 100 | 0.0005 | **0.4676** | 92 | 0.273 | 0.123 | +0.069 | ¹ |
| 2 | GAT + bidi (120 epochs) | 536 orig | 120 | 0.001 | 0.4207 | 91 | 0.313 | 0.053 | −0.116 |
| 3 | GAT + bidi + oversamp C1 (lr=0.001) | 735 | 50 | 0.001 | 0.3884 | 31 | 0.269 | 0.105 | +0.180 |
| 4 | GAT + bidi | 536 orig | 50 | 0.001 | 0.3179 | 46 | 0.266 | 0.026 | +0.029 |
| 5 | GAT + bidi + elw=3 | 536 orig | 50 | 0.001 | 0.3179 | 46 | 0.266 | 0.026 | +0.029 |
| 6 | GAT + bidi + dataset enrichi (fork) | ~735 | 100 | 0.0005 | 0.3000 | 61 | 0.255 | 0.132 | +0.299 |
| 7 | GAT seul (baseline v2.3.0) | 536 | 50 | 0.001 | 0.2522 | 39 | 0.178 | 0.018 | −0.095 |
| 8 | GAT + elw=2 | 536 | 50 | 0.001 | 0.2522 | 39 | 0.178 | 0.018 | −0.095 |
| 9 | GAT + elw=5 | 536 | 50 | 0.001 | 0.2522 | 39 | 0.178 | 0.018 | −0.095 |
| 10 | GAT + bidi + label-smoothing=0.05 | 536 | 50 | 0.001 | 0.2401 | 46 | 0.149 | 0.018 | +0.034 |
| 11 | R-GCN NumPy seul (baseline) | 536 | 50 | 0.001 | 0.2333 | 46 | 0.168 | 0.018 | +0.028 |
| 12 | GAT + bidi + emb50 aléatoire | 735 | 100 | 0.0005 | 0.2218 | 96 | 0.236 | **0.132** | +0.427 |
| 13 | GAT + label-smoothing=0.1 | 536 | 50 | 0.001 | 0.2060 | 39 | 0.160 | 0.018 | −0.054 |
| 14 | GAT + rgcn-dropout=0.2 | 536 | 50 | 0.001 | 0.1991 | 29 | 0.123 | 0.000 | +0.067 |
| 15 | GAT + bidi + C1_oversamp (lr=0.0005) | 735 C1 | 100 | 0.0005 | 0.1863 | 42 | 0.190 | 0.070 | +0.385 |
| 16 | R-GCN + bidi + oversamp C1 | 735 | 100 | 0.0005 | 0.1627 | 100 | 0.262 | 0.097 | +0.370 |
| 17 | **GAT + bidi + emb300 wiki.fr** | 735 | **5** | 0.0005 | 0.1202 | 4 | 0.019 | 0.000 | +0.083 |

> ¹ **Note run #1** : Reproductibilité partielle — la seed CLI ne contrôlait pas les inits constructeurs avant v2.5.2 (corrigé). Le résultat 0.4676 peut varier selon l'environnement exact.

> **Note run #17** : 5 epochs seulement (12 min/epoch × 300-dim). Non conclusif — le modèle
> n'a pas convergé. À exclure de la comparaison.

> **Note runs #15 et #16** : Ces runs utilisent le dataset C1 (11 types) mais le val set n'a que
> 8 types (filter/data_dep/control_dep absents). Les 3 types C1 comptent pour 0 dans le macro
> F1, ce qui sous-estime artificiellement `val_edge_macro_f1`. La vraie performance sur les
> 8 types originaux serait ~30% plus haute.

### 1b. Runs SANS val monitoring (métriques train uniquement)

> ⚠️ Ces chiffres mesurent la mémorisation du dataset d'entraînement, PAS la généralisation.
> Ils ne sont PAS comparables aux runs avec val. Fournis uniquement pour référence historique.

| Configuration | Dataset | Epochs | train_edge_f1 | train_node_acc | Remarque |
|--------------|---------|--------|--------------|---------------|---------|
| model_v2.4.0 (final, train+val) | 849 | 92 | 0.518 | 65.3% | checkpoint prod, pas de val dispo |
| GAT + emb50 aléatoire (pré-v2.3.0) | 536 | 30 | 0.485 | **98.6%** | overfitting total sans val |
| bidi + emb50 aléatoire (pré-v2.3.0) | 536 | 30 | 0.484 | 95.6% | idem |
| R-GCN + emb50 aléatoire (pré-v2.3.0) | 536 | 30 | 0.475 | 93.5% | idem |
| R-GCN baseline (pré-v2.3.0) | 536 | 50 | 0.448 | 53.7% | — |
| GAT + closed-loop backward | 536 | 50 | 0.402 | 99.5% | correctif v2.3.0 |

---

## 2. Les 5 meilleures combinaisons — détail complet

### Combinaison #1 — MEILLEURE (référence production)

```
GAT + bidirectionnel + oversampling C1 + lr=0.0005
```

```bash
gcn-train \
  --data-dir gcn-datasets/real/augmented/c1_oversampled/ \
  --val-dir  gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional
```

| Métrique | Valeur |
|---------|--------|
| `best_val_edge_macro_f1` | **0.4676** (ep92) |
| `val_node_macro_f1` | 0.273 |
| `val_graph_exact_match` | 0.123 |
| `gap train−val (edge)` | +0.069 ✅ |
| Dataset | 735 phrases (C1 oversampled) |

**Pourquoi #1 :** LR bas (0.0005) stabilise la convergence oscillante de LR=0.001. 100 epochs
permet d'atteindre le pic vers ep90-92. Oversampling C1 amène les 6 relations rares à 30 exemples.

---

### Combinaison #2 — Alternative haute performance

```
GAT + bidirectionnel, 120 epochs, lr=0.001
```

| Métrique | Valeur |
|---------|--------|
| `best_val_edge_macro_f1` | **0.4207** (ep91) |
| `val_node_macro_f1` | 0.313 |
| `val_graph_exact_match` | 0.053 |
| `gap train−val (edge)` | −0.116 (val > train, bonne généralisation) |

**Remarque :** La courbe oscille fortement avec LR=0.001. Le pic ep91=0.42 est atteint par
chance — reproduire exactement ce résultat est difficile. Préférer #1.
val > train ici : possible effet dropout actif en train (pas nécessairement une meilleure généralisation).

---

### Combinaison #3 — Bonne convergence rapide

```
GAT + bidirectionnel + oversampling C1, lr=0.001, 50 epochs
```

| Métrique | Valeur |
|---------|--------|
| `best_val_edge_macro_f1` | **0.3884** (ep31) |
| `val_node_macro_f1` | 0.269 |
| `val_graph_exact_match` | 0.105 |
| `gap train−val (edge)` | +0.180 ⚠️ |

**Remarque :** Pic précoce (ep31) suivi d'overfitting. LR=0.001 converge vite mais instable.

---

### Combinaison #4 — Baseline GAT sans oversampling

```
GAT + bidirectionnel, 50 epochs, lr=0.001
```

| Métrique | Valeur |
|---------|--------|
| `best_val_edge_macro_f1` | **0.3179** (ep46) |
| `val_node_macro_f1` | 0.266 |
| `val_graph_exact_match` | 0.026 |
| `gap train−val (edge)` | +0.029 ✅ |

**Remarque :** Référence propre sans oversampling. Petit gap = bonne généralisation à ce niveau.

---

### Combinaison #5 — Best val_gem (graph exact match)

```
GAT + bidi + emb50 aléatoire, 100 epochs
OU
GAT + bidi + dataset enrichi fork, 100 epochs
```

| Métrique | val_gem |
|---------|--------|
| GAT + bidi + emb50rand | **0.132** (ep96) |
| GAT + bidi + enriched fork | **0.132** (ep61) |

**Remarque :** Les deux atteignent le val_gem maximum connu (0.132). L'emb50rand atteint ce score
au détriment de val_edge_f1 (0.222 seulement). L'enriched fork est plus équilibré.

---

## 3. Enseignements — Ce qui marche et ce qui ne marche pas

### Ce qui marche (validé par l'expérience)

| Levier | Impact mesuré | Mécanisme |
|--------|---------------|-----------|
| `--use-attention` (GAT) | +0.087 sur val_edge vs R-GCN (0.252→0.318→0.468) | L'attention pondère les voisins par relation → meilleure généralisation |
| `--bidirectional` | +0.066 sur val_edge (0.252→0.318) | 22 types de relations (forward + inverse) → représentations plus riches |
| Oversampling C1 | +0.070 sur val_edge (0.318→0.388) | Classes rares portées à 30 exemples → gradient non-nul sur tous les types |
| LR 0.001 → 0.0005 | +0.080 (0.388→0.468) | Convergence stable, moins d'oscillation autour du minimum |
| 100 epochs (vs 50) | Nécessaire pour atteindre le pic | Avec LR bas, la convergence est plus lente mais plus stable |

### Ce qui ne marche pas (invalidé expérimentalement)

| Levier | Résultat | Pourquoi |
|--------|----------|---------|
| `--edge-loss-weight` (×2, ×3, ×5) | **0 effet** sur val | Amplifie la loss mais ne change pas la direction du gradient (même minimum) |
| `--label-smoothing 0.1` | −0.046 vs baseline | Lisse les labels → modèle moins confiant sur les vraies classes |
| `--rgcn-dropout 0.2` | −0.053 à ep50 | Trop de régularisation pour un dataset de 735 phrases |
| `--bidirectional` **avec R-GCN** | 0.233→0.163 (**régression**) | R-GCN apprend W_r séparé par relation. 22 types × matrices = trop de paramètres sans mécanisme d'attention pour les pondérer. Bidi n'aide que GAT. |
| Embeddings 50d **aléatoires** | 0.318→0.222 | Ajout de 50 dims de bruit. Modèle mémorise les lemmes training → overfitting gap=0.43 |
| Embeddings 300d wiki.fr | Non conclusif (5ep) | 12 min/epoch × 100 = 20h. Architecture trop lourde sans GPU. |

---

## 4. Analyse des fausses pistes pré-v2.3.0

Les anciens runs montraient **99.4% de node_accuracy en entraînement** avec embeddings 50d.
C'est de la **mémorisation pure**, pas de la généralisation.

**Preuve** : le même run avec val monitoring donne val_node_f1 ≈ 0.15-0.20, non 0.99.

La raison technique :
1. Embeddings aléatoires initialisés avec seed fixe
2. Modèle apprend à associer les vecteurs aléatoires aux classes → parfait sur train
3. Aucun test sur val → l'overfitting n'est pas détecté

**Règle absolue** : ne jamais évaluer sans val set séparé.

---

## 5. État actuel vs cibles production

| Métrique | Cible prod | Meilleur atteint | Écart | Bloquant |
|---------|-----------|-----------------|-------|---------|
| `val_edge_macro_f1` | > 0.40 | **0.468** ✅ | +0.068 | — |
| `val_node_macro_f1` | > 0.60 | 0.313 ✗ | −0.287 | **Données** : types rares ont 9-35 exemples. Aucun hyperparamètre ne compense. |
| `val_graph_exact_match` | > 0.20 | 0.132 ✗ | −0.068 | Bloqué par node_f1 |
| gap train−val (edge) | < 0.15 | 0.069 ✅ | — | — |

### Chemin pour atteindre val_node_f1 > 0.60

Le modèle ne peut pas apprendre correctement ces 5 types de nœuds rares :

| Type nœud | Exemples train | F1 estimé | Exemples nécessaires |
|-----------|---------------|-----------|---------------------|
| etat | 35 | ~0.15 | ~200 |
| action | 30 | ~0.12 | ~200 |
| transition | 12 | ~0.05 | ~150 |
| etat_systemique | 9 | ~0.03 | ~150 |
| condition | 4 | ~0.01 | ~100 |

**Ce qu'il faut** : annoter ~800 phrases supplémentaires ciblant ces types de nœuds.
Aucun hyperparamètre (LR, dropout, embeddings, architecture) ne peut compenser l'absence de données.

### Chemin pour tester correctement les embeddings pré-entraînés

Pour valider l'apport de wiki.fr.vec 300d :
```bash
# Option A : réduire à 50d par troncature
head -1 wiki.fr.vec > wiki.fr.50d.vec
tail -n +2 wiki.fr.vec | awk '{printf $1; for(i=2;i<=51;i++) printf " "$i; print ""}' >> wiki.fr.50d.vec

# Option B : lancer overnight (100ep × 12 min = ~20h)
gcn-train --embedding-file wiki.fr.vec --epochs 100 ...

# Option C : GPU (réduirait à ~1-2 min/epoch)
```

---

## 6. Checkpoint de production v2.4.0

| Paramètre | Valeur |
|-----------|--------|
| Fichier | `gcn-datasets/checkpoints/model_v2.4.0.npz` |
| Métriques | `gcn-datasets/checkpoints/model_v2.4.0_metrics.json` |
| Config entraînement | GAT + bidi + lr=0.0005 + weighted-loss + 92ep |
| Dataset | `augmented/final/` (849 phrases = train_c1_oversamp + val fusionnés) |
| `val_edge_macro_f1` (réf.) | 0.468 (mesuré sur le run avec val séparé) |
| Chargement | `GCNEngine.from_pretrained("model_v2.4.0.npz")` |

---

## 7. Recommandations pour v2.5.0

Par ordre de priorité et impact attendu :

1. **[DONNÉES — CRITIQUE]** Annoter 800 phrases supplémentaires ciblant les types de nœuds rares
   (etat, action, transition, etat_systemique, condition). Impact estimé : val_node_f1 0.313 → 0.50+.

2. **[DONNÉES — MOYEN]** Compléter le val set avec les 3 types C1 (filter, data_dep, control_dep)
   pour avoir une évaluation non-biaisée. Les 3 types sont actuellement absents du val set.

3. **[TECHNIQUE — MOYEN]** Tester embeddings wiki.fr.vec 50d (troncature ou PCA depuis 300d).
   Impact attendu si les pré-entraînés généralisent mieux que les aléatoires : val_edge_f1 0.468 → 0.52+.

4. **[TECHNIQUE — MOYEN]** Entraîner le `TrainableDecoder` (CIR → texte) via `gcn-train --train-decoder`.
   La feature est planifiée (plan approuvé) ; les données existent déjà dans `--data-dir`.

5. **[TECHNIQUE — FAIBLE]** Binding PyO3 pour remplacer GCNBridgeParser subprocess.
   Impact : latence d'inférence, pas les métriques ML.

---

## 8. Historique des audits

| Audit | Date | Scope | Findings | Corrigés |
|-------|------|-------|----------|---------|
| Audit 1 | 2026-09-18 | Diff de session (scripts, tests, moteur) | 7 (2 critiques, 2 médium, 2 bas, 1 info) | ✅ 7/7 |

---

## 9. Re-run 2026-09-21 — code actuel, seed fixe (NON COMPARABLE au tableau §1)

**Contexte :** re-exécution des configs #1 et #4 sur le code post-audits (HEAD `54ef614`),
`--seed 7` partout (option inexistante le 2026-09-18). CPU seul (8 cœurs, torch 2.13.0, ~4-9 s/epoch).

```bash
# #1-bis — référence prod (100ep, ~9min)
gcn-train --data-dir gcn-datasets/real/augmented/c1_oversampled/ --val-dir gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 --weighted-loss --use-attention --bidirectional --seed 7 \
  --output gcn-datasets/checkpoints/bench_20260921_gat_bidi_oversamp_lr5e4.npz \
  --log-csv gcn-datasets/checkpoints/bench_20260921_gat_bidi_oversamp_lr5e4.csv
# #4-bis — baseline sans oversampling (50ep, ~4min) : même commande avec
# --data-dir gcn-datasets/real/train/ --epochs 50 --lr 0.001 (seed 7, sorties bench_20260921_gat_bidi.*)
```

### 9a. Résultats (nouvelle échelle — ne pas mélanger avec §1)

| Run | Config | Epochs | best_val_edge_f1 | @ep | val_node_f1 | val_gem | gap_edge | train_edge_acc(fin) |
|-----|--------|--------|-----------------|-----|-------------|---------|---------|---------------------|
| #1-bis | GAT + bidi + oversamp C1, lr=0.0005, seed 7 | 100 | **0.1615** | 45 | 0.162 | 0.009 | +0.218 | 0.426 |
| #4-bis | GAT + bidi, 536 orig, lr=0.001, seed 7 | 50 | **0.2508** | 39 | 0.127 | 0.009 | −0.105 | 0.233 |

Apprentissage sain dans les deux cas : loss ↓ (4.99→2.60 et 4.29→2.46), edge_acc train ↑.
Gap #1-bis +0.218 (tendance overfit) vs #4-bis −0.105 (val > train, bonne généralisation).

### 9b. Calibration croisée — mêmes poids anciens, harnais neuf, val inchangé

`val.json` (114 phrases) est byte-identique depuis le 2026-09-17. Évaluation des checkpoints
du 2026-09-18 avec le code actuel (`gcn-eval`, même val) :

| Poids | val_edge_f1 le 2026-09-18 (contemporain) | val_edge_f1 actuel (même val) |
|-------|------------------------------------------|-------------------------------|
| `gat_bidi_oversamp_lr5e4.npz` (run #1, 0.4676) | 0.4676 | **0.1867** |
| `gat_bidi.npz` (run #4, 0.3179) | 0.3179 | **0.2217** |

Sous la même règle, l'entraînement neuf fait jeu égal avec les poids anciens
(#1-bis 0.1615 vs 0.1867 ; #4-bis **0.2508** vs 0.2217 — le neuf gagne).
**L'apprentissage n'est pas cassé ; l'échelle a changé.**

### 9c. Pourquoi l'échelle a changé (preuves datées, pas d'hypothèses)

1. **Données d'entraînement réécrites après le run #1.** `c1_oversampled/train.json` :
   mtime **2026-09-18 10:35** — 11 min APRÈS la fin du run #1 (checkpoint 10:24).
   `gcn-datasets/real/` est git-ignoré : les octets d'origine sont irrécupérables.
   (Supervision relationnelle identique en counts ; le contenu token a changé.)
2. **Harnais d'évaluation reconstruit.** `run_eval` actuel rebâtit le pipeline depuis
   `_arch_json` (bidi, all_pairs, couches, temperature, edge_threshold…) ; l'ancien
   construisait `n_relations=11` par défaut et échouait même au load des checkpoints
   bidi (`(22,79,79) ≠ (11,79,79)` — vérifié). Ajouts depuis : remap asymétrique,
   restauration D1, `predict_links`/seuils, dédup/oversample.
3. **Reproductibilité vérifiée.** Ancien code (e6d8028) relancé aujourd'hui dans cet env :
   init loss 4.9985 × 3 runs identiques — déterministe, et identique au code neuf (4.9953).
   L'écart avec le 4.33 du 2026-09-18 ne vient donc ni du code archivé ni de l'env
   (torch/numpy installés le 2026-08-24, inchangés) : il vient des données réécrites (point 1).

### 9d. Conséquences pour la suite

- **Ne jamais comparer §1 et §9.** Deux règles différentes. La référence prod reste le
  checkpoint `model_v2.4.0.npz`, pas un chiffre.
- Le biais C1-val (3 types absents du val, §1 note #15/#16) persiste sous le schéma 11 types.
- Prochain run utile : seed-sweep (7, 42, 123) de #4-bis pour mesurer la variance seed
  avant toute conclusion sur les hyperparamètres — ~12 min pour 3×50ep.

## 10. Run supervision v2 2026-09-22 — fusion silver FR+EN+code (NON COMPARABLE §1/§9)

**Données :** `DATA/supervision/fusion_v2.json` — 2398 uniques
(1661 existant + 358 silver FR + 348 silver EN + 31 code, tous ACCEPTés aveugle 0 critique),
split `combined_v2` 1666/350/382 stratifié (relation × langue) seed 42.
**Train :** `DATA/supervision/train_v2/model_v2.npz` — 100ep lr 0.0005 GAT+bidi weighted-loss,
loss 4.62→3.39, edge_acc 0.10→0.40. **Éval** (`gcn-eval`, harnais actuel, `trusted: true`) :

| Modèle (mêmes données) | val edge_f1 (350) | test edge_f1 (382) | val node_f1 |
|------------------------|-------------------|-------------------|-------------|
| model_v2 (supervision) | **0.3124** | **0.3392** | 0.2357 |
| baseline prod mêmes données | 0.1880 | 0.2431 | 0.1225 |
| **Δ supervision** | **+0.124** | **+0.096** | +0.113 |

**Ne pas comparer** au 0.468 (§1) : harnais et données ont changé (§9b).
**Limites actées** (cf `verdicts_supervision.json` + `errata_supervision_v2.json`) :
7 arêtes backward légitimes non supervisables (0.3%, limite harnais forward-consécutif),
3 spans inversées en quarantaine (s0057 train, s0259/s0643 test — garde-fous loader actifs, exclusion au prochain run),
241 existant en tokens provisoires (10%, 0 dans le silver — réinjection UD au prochain run),
IDs sXXXX non uniques inter-langues (pré-existant, silver namespaced).
