# Plan Phase 14 — Remediation Memorisation & Evaluation

**Date** : 2026-09-17
**Objectif** : Prouver que le moteur GCN-Core apprend veritablement la semantique causale, pas simplement des patterns de templates.

---

## Diagnostic

### Constats

| # | Constat | Severite |
|---|---------|----------|
| 1 | Les metriques logguees (node_acc, edge_acc) sont calculees sur les **donnees d'entrainement uniquement** | Critique |
| 2 | Le dataset (990 phrases) est genere a partir de **66 templates fixes** avec **49 noms** et **75 verbes** | Critique |
| 3 | Le vocabulaire est **100% partage** entre train/val/test (memorisation possible) | Haute |
| 4 | Aucune regularisation (pas de weight decay, pas de dropout dans RGCN/GAT) | Haute |
| 5 | Le eval_runner.py est **casse** pour les modeles avec attention/bidirectional/embeddings | Haute |
| 6 | Pas de early stopping, pas de best-model selection | Moyenne |
| 7 | Pas de shuffling des donnees d'entrainement | Basse |

### Preuve de memorisation

- Node accuracy 99% avec embeddings sur dataset synthetique = le modele a memorise le lexique (49 noms -> 7 types)
- Train accuracy ≈ Test accuracy (dans le rapport) = le dataset est trop simple pour distinguer memorisation et apprentissage
- 66 templates * 15 occurrences = memorisation des connecteurs ("parce que" -> cause, "bien que" -> concession)

---

## Plan de remediation

### Phase 14A — Correction de l'evaluation (Priorite 1)

**Objectif** : Mettre en place une evaluation train/val/test correcte pour diagnostiquer le vrai probleme.

#### Etape 1 : Corriger train.py pour evaluer sur val set

**Fichiers** : `gcn-python/src/gcn_python/training/train.py`

Modifications :
1. Ajouter argument CLI `--val-dir` (optionnel, default=None)
2. Si `--val-dir` est fourni, charger un `GCNDataLoader(val_dir)` separe
3. A chaque fin d'epoch, lancer une passe de evaluation sur le val set (forward uniquement, pas de backward)
4. Enregistrer `val_loss`, `val_node_accuracy`, `val_node_macro_f1`, `val_edge_accuracy`, `val_edge_macro_f1` dans le recorder
5. Sauvegarder le best model (selon val F1) au lieu du dernier model

**Estimation** : ~80 lignes a ajouter/modifier

#### Etape 1b : Etendre TrainingRecorder

**Fichiers** : `gcn-python/src/gcn_python/evaluation/recorder.py`

Modifications :
1. Ajouter `record_val(epoch, val_metrics)` ou etendre `record()` avec parametre optionnel `val_metrics`
2. Ajouter `_val_history` dans la structure interne
3. Modifier `to_csv()` pour inclure les colonnes val_*
4. Modifier `to_json()` pour inclure les metriques val
5. Modifier `learning_curve()` pour retourner les series val
6. Ajouter `best_val_epoch(metric="val_node_macro_f1", mode="max")`

**Estimation** : ~60 lignes a ajouter/modifier

#### Etape 1c : Ajouter le shuffling

**Fichiers** : `gcn-python/src/gcn_python/data/loader.py`

Modifications :
1. Ajouter parametre `shuffle=False` au constructeur `GCNDataLoader`
2. Si `shuffle=True`, melanger les samples avec un RNG seed configurable
3. Dans `train.py`, passer `shuffle=True` pour le train set et `shuffle=False` pour le val set

**Estimation** : ~15 lignes

#### Etape 1d : Corriger eval_runner.py

**Fichiers** : `gcn-python/src/gcn_python/evaluation/eval_runner.py`

Modifications :
1. Ajouter les flags `--use-attention`, `--bidirectional`, `--embedding-dim`, `--all-pairs`
2. Utiliser ces flags pour reconstruire le pipeline avec la bonne architecture
3. Permettre `--val-dir` pour evaluer sur un set separe pendant l'entrainement

**Estimation** : ~40 lignes

#### Etape 1e : Ajouter early stopping

**Fichiers** : `gcn-python/src/gcn_python/training/train.py`

Modifications :
1. Ajouter argument CLI `--patience` (default=10)
2. Tracker la val_f1最佳 epoch
3. Si pas d'ameliortion pendant `patience` epochs, arreter l'entrainement
4. Restaurer les poids du best model a la fin

**Estimation** : ~30 lignes

#### Tests Phase 14A

- Test que train.py avec --val-dir produit un CSV avec colonnes train + val
- Test que le best model est sauvegarde (pas le dernier)
- Test que early stopping arrete avant le max epochs
- Test que eval_runner.py charge correctement un model avec attention

---

### Phase 14B — Refonte du dataset (Priorite 2)

**Objectif** : Creer un dataset qui force le modele a apprendre la semantique, pas a memoriser des templates.

#### Etape 2 : Nouveau generateur avec templates hors-distribution

**Fichiers** : `gcn-datasets/generate_dataset_v2.py` (nouveau)

Concept : **Split stratifie par templates**, pas par indices.

1. **Templates d'entrainement** (44 templates) : 4 templates par relation (sur 6 disponibles)
2. **Templates de validation** (11 templates) : 1 template par relation (jamais vu a l'entrainement)
3. **Templates de test** (11 templates) : 1 template par relation (jamais vu ni en train ni en val)

Cela force le modele a generaliser au-dela des patterns de templates.

```
Relation: cause (6 templates dispo)
  -> train: cause_simple, cause_parce_que, cause_relative, cause_c_est
  -> val:   cause_a_cause_de    (jamais vu)
  -> test:  cause_etant_donne   (jamais vu)
```

**Estimation** : ~300 lignes (nouveau script)

#### Etape 3 : Vocabulaire etendu + hors-distribution

**Fichiers** : `gcn-datasets/generate_dataset_v2.py`

Modifications :
1. Etendre le lexique : 200 noms (au lieu de 49), 150 verbes (au lieu de 75)
2. Ajouter des categories lexicales nouvelles (noms tech, noms abstraits, verbes de mouvement)
3. **Split par vocabulaire** :
   - Train : 120 noms, 100 verbes (60% du vocab)
   - Val : 40 noms, 25 verbes (20%, jamais vu)
   - Test : 40 noms, 25 verbes (20%, jamais vu)

Le modele ne peut plus memoriser "marche -> etat", il doit apprendre que les noms sont des entites.

**Estimation** : ~150 lignes (dans le meme script)

#### Etape 4 : Paraphrases via LLM

**Fichiers** : `gcn-datasets/augment_paraphrases.py` (nouveau)

Concept : Pour chaque phrase du dataset, generer 2-3 paraphrases via LLM (Claude/GPT).

1. Prendre les 990 phrases du dataset
2. Pour chaque phrase, demander au LLM : "Reformule cette phrase en gardant la meme relation causale"
3. Annoter automatiquement les paraphrases avec `gcn-annotate`
4. Ajouter les paraphrases au dataset d'entrainement uniquement
5. Valider que les paraphrases ont la meme CIR que l'original

**Contraintes** :
- Utiliser `gcn-annotate` existant (pas de nouvel outil)
- Batch de 20 phrases par appel API
- cout estime : ~50 appels API (990/20)
- Timeout et retry logic deja implementes dans annotator.py

**Estimation** : ~200 lignes (nouveau script)

#### Etape 5 : Nettoyage du dataset original

**Fichiers** : `gcn-datasets/generate_dataset_v2.py`

Le dataset actuel a des problemes de qualite :
- Phrases artificielles ("Etant donne toute production, tous ventes change.")
- Pas de variation morphologique
- Confidence toujours 1.0

Ameliorations :
1. Varier les determiners (le/la/les/un/une/des/ce/cette)
2. Varier l'ordre des constituants quand c'est grammaticalement correct
3. Ajouter de la ponctuation variable
4. Ajouter des phrases avec 2-3 relations (pas toujours 1)
5. Varier la longueur (2-5 clauses au lieu de 2 fixes)

**Estimation** : ~200 lignes (dans le script de generation)

---

### Phase 14C — Regularisation du modele (Priorite 3)

**Objectif** : Reduire la capacite du modele pour eviter la memorisation.

#### Etape 6 : Weight decay (L2 regularization)

**Fichiers** : `gcn-python/src/gcn_python/layer2/reference.py`

Modifications :
1. Ajouter parametre `weight_decay=0.0` au constructeur MLPEncoder
2. Dans `update_node()` et `update_edge()`, ajouter la penalite L2 : `W -= lr * (grad + weight_decay * W)`
3. Dans `train.py`, passer `--weight-decay 0.001` par defaut

**Estimation** : ~20 lignes

#### Etape 7 : Dropout dans RGCN/GAT

**Fichiers** :
- `gcn-python/src/gcn_python/layer3/reference.py`
- `gcn-python/src/gcn_python/layer3/gat.py`

Modifications :
1. Ajouter parametre `dropout=0.0` au constructeur
2. Appliquer dropout sur les features d'entree avant message passing
3. En mode eval (future), desactiver le dropout
4. Dans `train.py`, passer `--dropout 0.3` par defaut

**Estimation** : ~30 lignes par fichier

#### Etape 8 : Label smoothing

**Fichiers** : `gcn-python/src/gcn_python/pipeline/cgnp.py`

Modifications :
1. Ajouter parametre `label_smoothing=0.0` a `_cross_entropy()`
2. Remplacer le one-hot strict par `y_smooth = (1 - epsilon) * y_hot + epsilon / n_classes`
3. Dans `train.py`, passer `--label-smoothing 0.1` par defaut

**Estimation** : ~15 lignes

---

### Phase 14D — Diagnostics avances (Priorite 4)

**Objectif** : Outils pour comprendre CE que le modele apprend.

#### Etape 9 : Matrice de confusion pendant l'entrainement

**Fichiers** : `gcn-python/src/gcn_python/evaluation/metrics.py`

Modifications :
1. Ajouter `confusion_matrix(pred, gold, classes) -> np.ndarray`
2. Ajouter `per_class_report(pred, gold, classes) -> dict` (precision, recall, F1 par classe)
3. Dans `train.py`, loguer la matrice de confusion a chaque epoch (dans le CSV ou un fichier separe)

**Estimation** : ~60 lignes

#### Etape 10 : Evaluation hors-distribution (OOD)

**Fichiers** : `gcn-datasets/generate_dataset_v2.py` + `gcn-python/src/gcn_python/evaluation/eval_runner.py`

Concept : Generer un dataset **complementairement different** pour tester la generalisation :
1. Phrases avec des mots jamais vus (vocabulaire OOD)
2. Phrases avec des structures syntaxiques differentes
3. Phrases avec des relations ambiguës

Ce dataset ne sert qu'a l'evaluation, jamais a l'entrainement.

**Estimation** : ~150 lignes

#### Etape 11 : Test de robustesse

**Fichiers** : `gcn-python/tests/test_robustness.py` (nouveau)

Tests :
1. **Test connecteur** : "La pluie tombe **parce que** le ciel est gris" vs "La pluie tombe **bien que** le ciel soit gris" -> le modele doit changer la relation
2. **Test inversion** : "A cause de X, Y" vs "A cause de Y, X" -> les types de nœuds doivent etre inverses
3. **Test hors-template** : phrases naturelles jamais generees par le template
4. **Test vocabulaire OOD** : mots nouveaux avec la meme structure

**Estimation** : ~200 lignes

---

## Ordre d'execution

```
Phase 14A (Evaluation)          Phase 14B (Dataset)           Phase 14C (Regularisation)
========================        ====================          =========================
Etape 1: train.py + val         Etape 2: Gen v2               Etape 6: Weight decay
Etape 1b: Recorder              Etape 3: Vocab etendu         Etape 7: Dropout RGCN
Etape 1c: Shuffling             Etape 4: Paraphrases LLM      Etape 8: Label smoothing
Etape 1d: eval_runner           Etape 5: Nettoyage
Etape 1e: Early stopping
         |                              |                              |
         v                              v                              v
    Tests 14A                    Tests 14B                    Tests 14C
         |                              |                              |
         +------------------------------+------------------------------+
                                        |
                                        v
                              Phase 14D (Diagnostics)
                              ======================
                              Etape 9: Matrice confusion
                              Etape 10: Dataset OOD
                              Etape 11: Tests robustesse
                                        |
                                        v
                                   Tests 14D
                                        |
                                        v
                              EXPERIMENT_990_V2.md
```

---

## Livrables

| # | Livrable | Fichier | Description |
|---|----------|---------|-------------|
| 1 | train.py avec validation | `training/train.py` | --val-dir, early stopping, best model |
| 2 | Recorder etendu | `evaluation/recorder.py` | train + val metrics |
| 3 | eval_runner corrige | `evaluation/eval_runner.py` | Support attention/bidi/emb |
| 4 | Generateur v2 | `gcn-datasets/generate_dataset_v2.py` | Templates OOD + vocab etendu |
| 5 | Script paraphrases | `gcn-datasets/augment_paraphrases.py` | Paraphrases via LLM |
| 6 | Regularisation | `layer2/reference.py`, `layer3/*.py` | Weight decay, dropout, label smoothing |
| 7 | Matrice confusion | `evaluation/metrics.py` | Per-class diagnostics |
| 8 | Dataset OOD | `gcn-datasets/ood/` | Test de generalisation |
| 9 | Tests robustesse | `gcn-python/tests/test_robustness.py` | Tests de semantique causale |
| 10 | Rapport final | `docs/EXPERIMENT_V2.md` | Resultats avant/apres |

---

## Metriques de succes

| Metrique | Avant (actuel) | Objectif |
|----------|----------------|----------|
| Node accuracy (val) | Inconnu (pas d'eval) | > 80% sur val OOD |
| Edge accuracy (val) | Inconnu (pas d'eval) | > 50% sur val OOD |
| Train-Val gap (node) | Inconnu | < 10% |
| Train-Val gap (edge) | Inconnu | < 15% |
| Robustesse connecteur | Non teste | > 90% de changement correct |
| Vocabulaire OOD (node) | Non teste | > 60% |

---

## Risques

| Risque | Impact | Mitigation |
|--------|--------|------------|
| LLM trop cher pour paraphrases | Cout API | Limiter a 500 phrases, batch size 20 |
| Dataset v2 trop petit | Underfitting | Garder 990 phrases min, ajouter 2000 paraphrases |
| Regularisation trop forte | Underfitting | Cross-valider weight_decay et dropout sur val set |
| eval_runner casse les tests existants | Regression | Garder l'ancien comportement par defaut, ajouter flags |
