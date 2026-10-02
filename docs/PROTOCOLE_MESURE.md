# Protocole de mesure — comparaisons moteur (gelé 2026-10-01)

Toute comparaison de runs n'est valide que si TOUS les points tiennent.
Sinon : pas de conclusion, pas de chiffre publié.

1. **Dataset figé** : SHA256 du train + val en tête de CSV (`training_data_hash`,
   déjà câblé). Toute réannotation entre runs invalide tout.
2. **Masque N_min figé** : même `--min-class-count`, mêmes classes actives
   (lire `[v5]`/`[v3.0]` en tête de log). Sinon les F1 portent sur des ensembles
   différents — incomparables par construction (`metrics.py` moyenne sur support>0).
3. **Code figé** : SHA du commit en tête de CSV (à ajouter si absent).
4. **`best` unique** : `val_edge_macro_f1`, checkpoint restauré sur ce critère
   (corriger `train.py` qui restaure sur node). Pas de max post-hoc.
5. **Seeds ×3 minimum** (7, 42, 123) + moyenne ± écart-type. Effet < 0.05 = bruit
   (loterie inter-seeds mesurée ~0.06).
6. **Early stopping sur edge**, patience large (30+), ou epoch fixe sans
   early-stopping (jamais sur node : `train.py` arrête sur `val_node_macro_f1`).
7. **Rapports obligatoires** : last-epoch + per-class + supports + `edge_cut_*`.
