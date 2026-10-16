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
   (fait P0 : early-stopping + restore unifiés sur edge). Pas de max post-hoc.
5. **Seeds ×3 minimum** (7, 42, 123) + moyenne ± écart-type. Effet < 0.05 = bruit
   (loterie inter-seeds mesurée ~0.06).
6. **Early stopping sur edge**, patience large (30+), ou epoch fixe sans
   early-stopping (fait P0 : unifié sur edge, jamais node).
7. **Rapports obligatoires** : last-epoch + per-class + supports + `edge_cut_*`.
8. **Seeds déterministes** : `--seed` propagé partout ; défaut 42 si absent
   (`train.py:368`). Dérivations internes fixes (`+100` couches R-GCN,
   `+1` scheduled-sampling, `+9999` dropout GAT, `epoch*1000+n` link-pred) —
   ne pas les toucher entre runs comparés. Métriques vides = `NaN`
   (jamais 0.0) depuis correctif `metrics.py`.
9. **Dims et règles figées** : `--mlp-hidden/--mlp-hidden2/--edge-hidden/
   --qual-hidden`, `--decoupled-wd`, `--theta-ambiguity`, `--temperature`,
   `--strict-directions` identiques entre runs comparés (persistés en
   `_arch_json`, sauf règles d'update). Tout écart invalide la comparaison.
