# Rapport d'expérimentation — Dataset généré 990 phrases

## Contexte

Après la phase 12 (suppression de `lang`), nous avons créé un dataset d'entraînement de 990 phrases françaises pour valider que l'architecture MLP → R-GCN/GAT apprend effectivement des représentations causales.

## Dataset

- **Génération** : `gcn-datasets/generate_dataset.py`
- **Format** : compatible `load_sentences()` (schema CIR standard)
- **Composition** : 11 relations × 90 variations = 990 phrases
- **Lexique** : 48 noms, 50 verbes (action/processus/transition), 150+ mots
- **Templates** : 6 structures par relation (simple, parce que, relatif, c'est...qui, à cause de, étant donné)
- **Split** : train=693, val=148, test=149

### Distribution des node types (gold)

| Type            | Count | %     |
|-----------------|-------|-------|
| entite          | 894   | 37.8% |
| processus       | 719   | 30.4% |
| action          | 385   | 16.3% |
| etat            | 234   | 9.9%  |
| etat_systemique | 133   | 5.6%  |

## Résultats d'entraînement

30 epochs, lr=0.001, sans word embeddings.

### Évaluation train/test

| Modèle | Split | Node Acc | Node F1 | Edge Acc | Edge F1 |
|--------|-------|----------|---------|----------|---------|
| RGCN   | train | 0.5340   | 0.3389  | 0.2887   | 0.1884  |
| RGCN   | test  | 0.5444   | 0.3405  | 0.2414   | 0.1599  |
| GAT    | train | 0.5340   | 0.3389  | 0.2887   | 0.1819  |
| GAT    | test  | 0.5444   | 0.3405  | 0.2414   | 0.1494  |
| BiDi   | train | 0.5328   | 0.3588  | 0.2887   | 0.1819  |
| BiDi   | test  | 0.5333   | 0.3552  | 0.2414   | 0.1494  |

### Distribution des prédictions (RGCN, test)

| Type     | Pred | Gold  |
|----------|------|-------|
| entite   | 79.7%| 36.9% |
| action   | 16.4%| 17.2% |
| processus| 3.9% | 31.9% |
| etat     | 0.0% | 9.4%  |
| etat_sys | 0.0% | 4.4%  |

## Analyse

### Le modèle apprend-il ?

**Oui, mais partiellement.**

- La loss décroît (2.55 → 1.92 sur 50 epochs RGCN baseline)
- Le node accuracy (0.54) est supérieur au random (1/5 = 0.20)
- Pas de surapprentissage notable (train ≈ test)

### Problème : biais de classe majoritaire

Le modèle prédit "entite" dans **80% des cas** alors que cette classe ne représente que **37%** du gold. C'est un échec de discrimination.

### Causes

1. **Features UD insuffisantes** : pos, dep_rel, morph ne capturent pas la sémantique causale. Un nom "marché" et un nom "pluie" ont les mêmes features UD.

2. **Pas de word embeddings** : sans vecteurs pré-entraînés, le modèle ne sait pas que "réduire" est sémantiquement proche de "diminuer" mais opposé à "augmenter".

3. **Classe majoritaire dominante** : avec 37% de "entite", le modèle minimisait la loss en prédisant cette classe partout. La cross-entropy ne pénalise pas assez ce biais.

4. **Architecture sous-dimensionnée** : le MLP encoder (d_clause=20) est trop petit pour capturer les différences entre 5 types de nœuds à partir de features sparse.

## Pistes d'amélioration

### Court terme (sans changer l'architecture)

| # | Action | Impact attendu |
|---|--------|----------------|
| 1 | **Class weights** dans la loss : pénaliser les classes majoritaires | Réduit le biais de prédiction |
| 2 | **Focal loss** : downweight les exemples faciles, focus sur les difficiles | Améliore la discrimination |
| 3 | **Oversampling** des classes minoritaires (etat, etat_systemique) | Équilibre le training |
| 4 | **Early stopping** sur val F1 (pas accuracy) | Évite le plateau de majorité |

### Moyen terme (features enrichies)

| # | Action | Impact attendu |
|---|--------|----------------|
| 5 | **Word embeddings** (FastText fr) : charger via `--embedding-file` | Capture la sémantique des mots |
| 6 | **Features de position** : position du nœud dans la phrase, distance au connecteur | Indique le rôle syntaxique |
| 7 | **Features de connecteur** : type de marqueur causal (cause, condition, concession) | Signal direct de la relation |
| 8 | **One-hot des lemmes** des verbes causaux | Le modèle apprend les patterns verbe→type |

### Long terme (architecture)

| # | Action | Impact attendu |
|---|--------|----------------|
| 9 | **GAT plus profond** (2-3 couches) avec residual connections | Capture les dépendances longue distance |
| 10 | **Contrastive learning** : embeddings de phrases causales proches | Apprend la structure causale |
| 11 | **Data augmentation** : paraphrases via LLM | Diversifie les patterns |

## Résultats avec word embeddings (FastText wiki.fr.vec, 50 dim)

Entraînement avec `--embedding-dim 50` (features UD + embeddings apprenables).

### Comparaison avant/après embeddings

| Modèle     | Node Acc (test) | Δ    | Node F1 (test) | Δ    | Edge Acc (test) | Edge F1 (test) |
|------------|-----------------|------|----------------|------|-----------------|----------------|
| RGCN       | 0.5444          | —    | 0.3405         | —    | 0.2414          | 0.1599         |
| RGCN+emb   | **0.9611**      | +77% | **0.9361**     | +175%| 0.2874          | 0.2002         |
| GAT        | 0.5444          | —    | 0.3405         | —    | 0.2414          | 0.1494         |
| GAT+emb    | **0.9944**      | +83% | **0.9939**     | +192%| 0.2874          | 0.2002         |
| BiDi       | 0.5333          | —    | 0.3552         | —    | 0.2414          | 0.1494         |
| BiDi+emb   | **0.9750**      | +83% | **0.9569**     | +169%| 0.2989          | 0.2088         |

### Analyse

**Node classification** : les embeddings transforment les résultats.
- GAT+emb atteint **99.4%** de node accuracy sur test (vs 54.4% sans embeddings)
- Le modèle discriminate correctement les 5 types de nœuds causaux
- Pas de surapprentissage notable (train ≈ test)

**Edge classification** : reste le point faible.
- L'accuracy des arêtes plafonne à ~30% (vs 24% sans embeddings)
- Le modèle ne parvient pas à distinguer les 11 types de relations
- Cause probable : les embeddings n'aident pas à la classification de paires de nœuds

## Conclusion

L'expérience confirme que :
1. Le moteur GCN-Core **fonctionne** (loss décroît, accuracy > random)
2. Les features UD seules sont **insuffisantes** pour la classification causale
3. Les **word embeddings** résolvent le problème du node classification (54% → 99%)
4. Le **edge classification** reste un problème ouvert — nécessite une architecture dédiée

Le dataset de 990 phrases est un banc d'essai valide pour tester ces améliorations.
