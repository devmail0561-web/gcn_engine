# Guide — Word Embeddings lexicaux (gcn-python)

**Statut** : embeddings **ACTIFS PAR DÉFAUT** depuis le 2026-09-25 (`--embedding-dim 128`,
cf `REMEDIATION-DIAGNOSTIC.md` et `CHANGELOG.md` `[Unreleased]`).
Ce guide documente l'activation, les variantes pré-entraînées et le dépannage.

---

## 1. Comportement par défaut (recommandé)

```bash
gcn-train --data-dir ... --val-dir ... --epochs 20 --output model.npz
# → embeddings apprenables 128-dim, vocab construit sur les lemmes du train
# → checkpoint contient word_emb_E + _word_emb_vocab_json
# → gcn-eval restaure vocabulaire + poids à l'identique
```

Dimensions résultantes : `d_clause` 79 → **207**, `d_edge` 365 → **877**
(formules : `FeatureVocabulary.d_clause_effective` / `d_edge_closed_loop`).

Désactivation (déconseillée — moteur aveugle au lexique, warning explicite) :
```bash
gcn-train --data-dir ... --embedding-dim 0 --output model_noemb.npz
# AVERTISSEMENT : embeddings lexicaux DÉSACTIVÉS ...
```

## 2. Embeddings pré-entraînés

Fichier local disponible : `gcn-python/models/wiki.fr.vec` (300-dim, ~1,15 M mots, 58 Mo).

```bash
# Initialisation depuis fichier (dimension auto-détectée : 300)
gcn-train --data-dir ... --embedding-file gcn-python/models/wiki.fr.vec --output model_wiki.npz

# Tronqué à 50-dim (recommandation BENCHMARK.md §5, entraînement ~20× plus rapide)
head -1 wiki.fr.vec > wiki.fr.50d.vec
tail -n +2 wiki.fr.vec | awk '{printf $1; for(i=2;i<=51;i++) printf " "$i; print ""}' >> wiki.fr.50d.vec
gcn-train --data-dir ... --embedding-file wiki.fr.50d.vec --output model_wiki50.npz
```

Modèle fastText binaire (nécessite `pip install fasttext-wheel`, impose `d_emb=300`, gelé par défaut) :
```bash
gcn-train --data-dir ... --fasttext cc.fr.300.bin --output model_ft.npz
```

## 3. Variantes d'architecture (requièrent `--embedding-dim > 0`)

| Option | Effet | Coût |
|--------|-------|------|
| `--clause-pooling mean\|max` (A) | Pooling embeddings sur tokens de contenu au lieu du seul `root_lemma` | 0 |
| `--subject-object-emb` (B) | Concatène embeddings sujet + objet (`+2×d_emb`, `_absent` appris) | +2×d_emb dims |
| `--freeze-embeddings` (C) | Gèle les pré-entraînés (requiert `--embedding-file`) ; spéciaux et nouveaux lemmes restent entraînables | — |

## 4. Dépannage

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `ValueError: Dimension du fichier (300) ≠ d_emb (128)` | `--embedding-dim` explicite incompatible avec `--embedding-file` | Omettre `--embedding-dim` (auto-détection) ou passer `--embedding-dim 300` |
| `ClickException: --fasttext impose d_emb=300` | `--embedding-dim 200 --fasttext` | Omettre `--embedding-dim` avec `--fasttext` |
| `AVERTISSEMENT : embeddings DÉSACTIVÉS` | `--embedding-dim 0` explicite | Retirer l'option (défaut 128) sauf besoin de compatibilité |
| Vieux checkpoint sans `word_emb_E` évalué par `gcn-eval` | Checkpoint pré-remédiation | Fonctionne : `_arch_json` sans `d_emb` → 0 → pipeline sans embeddings (warning si applicable) |
| `edge_f1` plus bas AVEC embeddings aléatoires | Mémorisation des lemmes (BENCHMARK.md §4) | Utiliser pré-entraînés à convergence, pas des dims aléatoires élevées |

## 5. Validation empirique

Protocole : `REMEDIATION-DIAGNOSTIC.md` §6–§8 (+ `BENCHMARK.md` §5 pour wiki.fr à convergence).
Résultats mesurés : `BENCHMARK.md` (section remédiation, à compléter après exécution).
Règle absolue : **toujours avec `--val-dir` séparé** (les scores train avec embeddings
sont de la mémorisation, cf `BENCHMARK.md` §4).
