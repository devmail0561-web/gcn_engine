# Plan — Production Readiness GCN Engine

**Date :** 2026-09-17  
**Objectif :** Passer de l'état actuel (infrastructure prête, modèle non entraîné sur données réelles)
à un moteur déployable produisant des CIR fiables sur du texte naturel non vu.

---

## État de départ (vérifié)

```
Dataset réel       : 678 phrases, tokens UD = 0, causal_pattern absent
                     Relations : cause(402) enable(167) condition(56) concession(40) prevent(13)
                     6 relations absentes : motivation, sequence, filter, opposition, data_*, control_*

Checkpoints        : 5 existants — tous entraînés sur corpus synthétique (66 templates)
                     → mémorisation confirmée, non utilisables en prod

Tests              : 204 passent, infrastructure solide
Scripts Phase 1    : implémentés, non encore exécutés sur les données
```

**Critères "production ready" :**
```
val_edge_macro_f1 (réel)    > 0.40   (au-dessus de la baseline triviale)
val_graph_exact_match (réel) > 0.20   (au moins 1 graphe complet sur 5 est correct)
val_node_macro_f1 (réel)    > 0.60
Gap train−val node           < 0.15   (pas de mémorisation)
Test robustesse : "si" → condition ≠ "bien que" → concession (connecteur change la relation)
```

---

## Étape 1 — Annoter le dataset réel avec tokens UD

**Durée estimée :** 15–30 min (exécution des scripts)

```bash
cd gcn-tools/gcn-scraper/src/gcn_scraper

# 1a. Diagnostiquer les token_span existants
python diagnose_spans.py \
  --input gcn-datasets/real/annotated.json

# Résultat attendu : taux d'invalides > 50% → confirme que les spans doivent être re-dérivés

# 1b. Re-dériver les token_span via spaCy (structure dépendances syntaxiques)
python rederive_spans.py \
  --input  gcn-datasets/real/annotated.json \
  --output gcn-datasets/real/annotated_spans_fixed.json

# Résultat attendu : >= 400 phrases corrigées sur 678
# Si < 400 : les spans sont trop corrompus → re-annotation CIR complète nécessaire (hors scope)

# 1c. Valider la qualité des CIR après correction spans
python validate_cir.py \
  --input gcn-datasets/real/annotated_spans_fixed.json

# 1d. Annoter avec les tokens UD (spaCy fr_core_news_sm)
python annotate_real_dataset.py \
  --input  gcn-datasets/real/annotated_spans_fixed.json \
  --output gcn-datasets/real/annotated_ud.json \
  --lang   fr

# 1e. Vérifier que reps_from_sentence() retourne des représentations non vides
python validate_reps.py \
  --input gcn-datasets/real/annotated_ud.json

# 1f. Re-splitter en train/val/test stratifié sur la relation des edges
python split_real_dataset.py \
  --input   gcn-datasets/real/annotated_ud.json \
  --out-dir gcn-datasets/real/ \
  --seed    42
```

**Critère de succès :** `gcn-train --data-dir gcn-datasets/real/ --epochs 1` produit `loss > 0`

---

## Étape 2 — Entraînement sur données réelles

**Durée estimée :** 10–30 min (selon matériel)

```bash
gcn-train \
  --data-dir  gcn-datasets/real/train.json \
  --val-dir   gcn-datasets/real/val.json \
  --epochs    150 \
  --lr        0.001 \
  --use-attention \
  --bidirectional \
  --embedding-dim  50 \
  --edge-loss-weight 3.0 \
  --weighted-loss \
  --weight-decay   0.001 \
  --rgcn-dropout   0.1 \
  --label-smoothing 0.05 \
  --patience       20 \
  --output         checkpoints/prod_v1.npz \
  --log-csv        logs/prod_v1.csv
```

**Ce qui peut mal se passer et quoi faire :**

| Symptôme | Cause probable | Action |
|----------|---------------|--------|
| `loss=0` dès la 1ère epoch | tokens UD toujours absents | Re-vérifier étape 1e |
| `val_node_macro_f1 = 0` | Toutes les clauses ont 0 nœud valide | Vérifier alignement token_span |
| `val_edge_macro_f1 < 0.15` | 6 relations absentes + déséquilibre cause (60%) | Normal au 1er run — continuer |
| gap train−val > 0.30 | Mémorisation | Augmenter weight_decay à 0.005 |

---

## Étape 3 — Évaluation et diagnostic

**Durée estimée :** 5 min

```bash
# Métriques sur le test set (jamais vu pendant l'entraînement)
gcn-eval \
  --data-dir    gcn-datasets/real/test.json \
  --checkpoint  checkpoints/prod_v1.npz \
  --output-json logs/prod_v1_test_metrics.json

# Matrice de confusion edges
python -c "
from gcn_python.evaluation.metrics import confusion_matrix, per_class_report
# (après avoir récupéré les prédictions via eval_runner)
"
```

**Lecture des résultats :**

- `val_edge_macro_f1 > 0.40` ET `gap < 0.15` → le modèle généralise → **Prêt pour tests en conditions réelles**
- `val_edge_macro_f1` entre 0.25 et 0.40 → insuffisant mais améliorable → aller à l'étape 4
- `val_edge_macro_f1 < 0.25` → problème fondamental (données, alignement) → diagnostiquer

---

## Étape 4 — Si métriques insuffisantes : enrichir le dataset

**Condition :** `val_edge_macro_f1 < 0.40` après étape 2–3

Le problème principal est la **distribution déséquilibrée et la couverture incomplète** des relations :
- `cause` domine à 60 % → le modèle prédit toujours "cause"
- 6 relations absentes → impossible à prédire

**Action : scraper des phrases ciblées par relation manquante**

```bash
# Scraper 500+ phrases par relation manquante
gcn-scrape \
  --sources wikipedia_fr wikipedia_en hal \
  --out-dir gcn-datasets/raw/targeted/ \
  --limit 5000

# Annoter UD + CIR (gcn-annotate)
gcn-annotate annotate \
  --input  gcn-datasets/raw/targeted/ \
  --out    gcn-datasets/real/targeted_cir.json \
  --model  claude-sonnet-5

# Combiner avec les 678 phrases existantes et re-splitter
python gcn-tools/gcn-scraper/src/gcn_scraper/split_real_dataset.py \
  --input   gcn-datasets/real/targeted_cir.json \
  --out-dir gcn-datasets/real/ \
  --seed    42
```

Répéter l'étape 2 avec le dataset enrichi.

---

## Étape 5 — Test de robustesse

**Condition :** métriques de l'étape 3 satisfaisantes

Valider que le modèle apprend la **sémantique du connecteur**, pas juste la fréquence :

```bash
python -c "
from gcn_python.data.loader import reps_from_sentence
from gcn_python.data.json_reader import load_sentences
from gcn_python.training.checkpoint import load_checkpoint
from gcn_python.pipeline.cgnp import CGNPipeline
# ...

# Test 1 : même phrase, connecteur différent
# 'Les ventes baissent, DONC on réduit les coûts.'  → attendu : cause ou sequence
# 'Les ventes baissent, BIEN QUE on réduit les coûts.' → attendu : concession

# Test 2 : inversion des clauses
# 'A parce que B' → cause(A→B)
# 'B parce que A' → cause(B→A) — relation symétrique doit changer de direction

# Test 3 : phrase hors-template (non vue pendant l'entraînement)
"
```

---

## Étape 6 — Packaging checkpoint prod

**Condition :** étapes 3 et 5 satisfaisantes

```bash
# Renommer le checkpoint validé
cp checkpoints/prod_v1.npz checkpoints/prod_stable.npz

# Documenter les métriques du checkpoint
cat logs/prod_v1_test_metrics.json

# Vérifier que le checkpoint se charge sans erreur
python -c "
from gcn_python.training.checkpoint import load_checkpoint
from gcn_python.pipeline.cgnp import CGNPipeline
# ... charger et faire un forward sur une phrase test
print('Checkpoint OK')
"
```

---

## Résumé du chemin critique

```
Étape 1 : annoter UD dataset réel          [BLOQUANT — 30 min]
    ↓
Étape 2 : entraîner sur données réelles    [20 min]
    ↓
Étape 3 : évaluer — métriques OK ?
    ├── OUI (val_edge_f1 > 0.40) ──────────→ Étape 5 : robustesse → Étape 6 : packaging
    └── NON (val_edge_f1 < 0.40) ──────────→ Étape 4 : enrichir dataset → Étape 2
```

---

## Métriques minimales de production

| Métrique | Minimum acceptable | Idéal |
|----------|-------------------|-------|
| `val_node_macro_f1` (réel) | > 0.60 | > 0.80 |
| `val_edge_macro_f1` (réel) | > 0.40 | > 0.60 |
| `val_graph_exact_match` (réel) | > 0.20 | > 0.40 |
| Gap train−val node | < 0.15 | < 0.10 |
| Test robustesse connecteurs | Pass | Pass |
| Baseline triviale edge (cause 60%) | Battu | Battu |
