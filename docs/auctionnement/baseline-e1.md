# E1 — baseline regelée (harnais unique, gold gelé)

Date : 2026-10-14
Statut : GO pour E0. Ne pas comparer à l'ancien harnais (0.468 vs 0.1867 archivés incomparables).

## Splits gelés (hash)

- `v4/final/manifest.json` : seed 42, quota 64, exclude_none true, train 572 / val 132, 52/12 × 11 relations.
- `gold_mini/gold_mini_manifest.json` : id gold-mini-v1, seed 42, n=85, 11 relations × 7-10, source lots v4 arbitrés 08-24.
- Règle : toute réannotation invalide les comparaisons (`PROTOCOLE_MESURE.md:1-15`). Seeds 7/42/123 + moyenne ±std, effet <0.05 = bruit.

## Référence actuelle (harnais unique)

- val_edge_macro_f1 moy 0.239 ±0.04 (seeds 0.184/0.265/0.269) vs train ~0.99 → overfit massif, gap <0.15 non tenu.
- OOV val 23.7% vs seuil prod 5% (`forward_ids.py:71-72`).
- ECE 0.33-0.55 N=81 <100 → NO-GO Ph5, quantification enterrée, conf = provisoire. Gate production codé : `require_production_sample(N>=100)` (`calibration.py`).
- 8 relations v3 à 0 → gel R8/R10 (`refacto_loss.py:25-50`, `grid_f1.py:101-103` exit 2).

## Gates durcis (livrés)

- `Makefile guards` : torch cantonné layer3 PT/GAT jamais runtime tok ; subprocess restreint au chemin tok (bridge + train/train_tok autorisés, pipeline/nlu/layer0 interdits).
- `calibration.require_production_sample` : N>=100 exigé en production, tests unitaires à N>=10 inchangés (18 verts).
- `forward_ids` : 39 verts. Manifests ci-dessus = référence E1.

## E0 — exécuté 2026-10-14 (CPU, harnais unique)

- Commande : `train_tok --train v4/final/train/train.json --val v4/final/val/val.json --epochs 20 --espace coarse`, seeds 7/42/123 (`data_hash c1e5d944d5830d59`, `git_sha 2778fb4`). CSV `/tmp/e0_seed{7,42,123}.csv`.
- Résultat : `best_val_edge_f1 = 0.10666` identique 3 seeds, `best_epoch = 0` (init), `epochs = 11` (early stop), `oov_val = 0.2143`, 572/132, hors_fenetre 0 / arrieres 0.
- Lecture : n'apprend pas — best à l'init, 0 variance inter-seeds. Goulot = représentation (sac mots + OOV 21%), pas seeds. Reste ~0.11 << 0.25 : O1 + têtes factorisées d'abord (R2/R5), pas de tuning.

## E2 — O1 vs aléatoire (exécuté 2026-10-14, CPU, même 536/572)

- O1 : `scripts/pretrain_o1.py` skip-gram contexte ±2, 3ep CPU, 572 phrases V=5952 d=128. Perte 8.1026 → 8.0295 → 8.0712 (signal faible, plateau ep2). Init `/tmp/o1_init.npz` s1. Zéro lexique, zéro gold.
- Aval : `train_tok --espace coarse --epochs 20 --init-npz` seeds 7/42/123, même harnais (`data c1e5d944`, `git 2778fb4`). CSV `/tmp/e2_seed{7,42,123}.csv`.
- Résultat : 0.10666 / 0.10666 / 0.10622 vs aléatoire 0.10666 ×3. Δ ≈ 0 << bruit 0.05-0.06.
- Verdict E2 : NO-GO. Le skip-gram 3ep ne déplace pas la val. Cause probable = têtes/étiquettes (R2 : espace plat, negated=False, scope figé) + OOV 21%, pas l'init E. Suite plan validé : têtes factorisées + segmentation apprise (ancres/bornes, biaffine rang faible) avant tout nouveau pré-entraînement. O1-skipgram gelé, pas d'acharnement.
- Incident corrigé : `--init-npz` exigeait l'ordre interne identique → réindexation par id stable (ensembles = vérité), `--init-npz` ajouté à `train_tok` (`entrainer(init_npz=...)` + CLI).

## Têtes factorisées (livré 2026-10-14, R2)

- Module `layer2/edge_factor.py` : base 11 × ternary 4 × special 3 + `compose()` déterministe → 19. Roundtrip 19/19 prouvé, totalité triplets prouvée (`test_edge_factor.py`, 5 verts).
- Honnête : gain params ~5% (18 vs 19 têtes), pas 10×. Vrai gain = partage statistique (ternary/special appris sur toutes les bases) + GEL par facteur (base 11 toutes ≥30, seuls ternary/special gelés) au lieu de GEL 8 relations entières. Zéro lexique, indices seuls.
- Reste : brancher supervision factorisée dans `train_tok` (loss par facteur + masques par facteur) + ancres/bornes nœuds + segmentation apprise.

## Supervision factorisée auxiliaire (livré 2026-10-14, R2)

- `train_tok --factor-aux` : mêmes paires supervisées entraînent 3 têtes (base 11 × ternary 4 × special 3) depuis `edge_gold_fin` (indices fins jamais remappés, sauvés avant remap coarse). Pas de gradient vers le tronc — signal partagé futur, mesure immédiate. Snapshot/restore inclus. Zéro lexique.
- Run coarse 20ep seed 42 : `best_val_edge_f1 = 0.10666` best ep 0, identique sans auxiliaire — attendu (tronc inchangé par construction). CSV `/tmp/e3_seed42.csv`.
- Sonde facteurs après 1ep train (132 paires val) : joint 0.098, base 0.098 (= hasard 1/11), ternary 1.0, special 1.0 (dégénérés : v4/final ne contient que des directes, 8 v3 à 0). Confirme la cause GEL : pas de signal ternaire dans ce split. Les têtes sont prêtes pour du gold avec ternaires (gold-3000).

## Segmentation apprise v0 (livré 2026-10-14, R3)

- `layer1/span_head.py` : têtes start/end linéaires + `decode_spans` glouton non-recouvrant + `span_f1` exacte + loss pondérée auto (ratio nég/pos du batch — sans elle, effondrement à zéro constaté : F1 0.000, 0 prédiction). Zéro lexique, CPU pur. Tests 5 verts (`test_span_head.py`).
- Sonde 572 train / 132 val, 5ep lr 0.5 : loss 2.31→2.24, spanF1 0.000→0.027. Débuts justes (`[0,…]`, `[44,…]`), fins coupées tôt. Limite honnête : embeddings statiques sans contexte — les fins exigent un encodeur contextuel (O1). B reste en place, rien n'est câblé au runtime : mesure seule.

## Encodeur contextuel CPU (livré 2026-10-14, O1)

- `layer1/context.py` : CNN dilaté 2 couches (d=1 puis 2, champ ±3) + résidus + tanh, 98 560 params ≤1M, NumPy pur. Backward prouvé finite-diff (err ~1e-4, float32). `grad_input` ajouté à `SpanHead` pour le reflux. Tests 6 verts (`test_context.py`, inclut finite-diff).
- Sonde spans 572/132 5ep : loss 2.28→1.97 (vs 2.31→2.24 sans contexte), spanF1 0.027→0.090 (×3.3). Contexte prouvé comme levier ; absolu encore bas (fins + granularité sub-clause = gold-3000 + O1 MLM sur 285k).
- `ruff` + guards verts. B et chemin runtime inchangés (mesure seule).

## E4 — contexte branché au runtime (exécuté 2026-10-14, PERCÉE)

- Câblage : `TokForwardEngine(use_context=...)` (défaut off, parité préservée : 69 verts `forward_ids + training + context`), ctx entre lookup et pooling, backward pool⁻¹ → ctx → E, snapshot/restore + save/load npz (refus croisé explicite), `--context` CLI, bump epoch M1a. Zéro lexique.
- Runs coarse 20ep même harnais (`data c1e5d944`, `git 2778fb4`), CSV `/tmp/e4_seed{7,42,123}.csv` :
  seed 7 → 0.32444 (ep 19) ; seed 42 → 0.37662 (ep 19) ; seed 123 → 0.25214 (ep 19).
  Moy 0.318 ±0.05 vs sans-contexte 0.10666 ×3. Δ ≈ +0.21 >> bruit 0.05-0.06.
- Lecture : la représentation était bien le goulot (R1 prouvé bout-en-bout, pas seulement en sonde). Best ep 19/20 partout = encore en progression à 20ep (patience non déclenchée) — 30ep à tester. Loterie inter-seeds forte (±0.05, seed 123 bas) : 3 seeds minimum respectés, effet au-delà.
- Limites : espace coarse (fin toujours GEL R8/R10), val N=132 petite, OOV 21% inchangé, MLM-init non re-testée avec contexte (transfert spans négatif 0.047 — chantier O1-MLM + init full-moteur restant).

## E5 — 30ep contexte (exécuté 2026-10-14)

- Mêmes 572/132 coarse, seeds 7/42/123, CSV `/tmp/e5_seed{7,42,123}.csv` :
  0.44778 (ep 29) / 0.37662 (ep 19) / 0.34530 (ep 27). Moy 0.390 ±0.04.
- Best ep 29 et 27 (7, 123) = progression au-delà de 20ep confirmée. Seed 42 plafonne ep 19. Loterie ±0.04 persistante.
- Cible historique >0.40 (espace fin) : atteinte en coarse à seed 7 (0.448). Reste : fin (GEL), OOV, variance.

## Init MLM full (E2-MLM) — en cours

O1-MLM v4-only : 572 phrases V=5952, ppl 5124→2503→1909 monotone (hasard 5952). Init `/tmp/o1_mlm_v4.npz` s1 (`FORMAT_O1 20261014`, E+ctx). `--init-o1` ajouté (exige `--context` + même vocab, sinon STOP).

## E6 — aval avec init MLM (exécuté 2026-10-14, NO-GO transfert)

- Coarse 30ep `--context --init-o1`, seeds 7/42/123, CSV `/tmp/e6_seed{7,42,123}.csv` :
  0.32470 (ep 29) / 0.33726 (ep 27) / 0.38976 (ep 29). Moy 0.351 ±0.03
  vs init aléatoire 0.448/0.377/0.345 (moy 0.390). Δ ≈ -0.04 ∈ bruit.
- Verdict : l'init MLM n'aide ni ne nuit à la classification — cohérent avec spans (0.047 < 0.090). Le gain vient de l'ARCHITECTURE contexte (+0.21), pas des poids MLM. O1-MLM gelé comme init aval ; reste utile comme preuve que le brut porte du signal (ppl ÷3).
- Bilan E0→E6 : E0 0.107 → E4/E5 0.318/0.390 (contexte) → E6 0.351 (init, bruit).

## E7 — OOV hashé vs zéros (exécuté 2026-10-14, NEUTRE)

- `TokEmbedding(oov_mode="hash")` : seaux `id % 1024`, table fixe-seed apprise, compteur OOV inchangé (couverture), défaut `zeros` intact (10 verts `test_tok_embedding`). `--oov-hash` CLI + persistance/snapshot. Zéro lexique.
- Coarse 30ep `--context --oov-hash`, seeds 7/42/123, CSV `/tmp/e7_seed{7,42,123}.csv` :
  0.43862 (ep 29) / 0.36520 (ep 19) / 0.32186 (ep 28). Moy 0.375 ±0.05
  vs zéros 0.448/0.377/0.345 (moy 0.390). Δ ≈ -0.015 ∈ bruit.
- Verdict : neutre — le modèle contourne déjà les zéros (contexte ±3), les seaux ajoutent autant de bruit que de signal à cette échelle. Zéros restent le défaut ; hash disponible pour OOV >40%. OOV 21% n'est pas le goulot (R1 contexte l'était, prouvé E4).

## E10 — spans auxiliaires câblés (exécuté 2026-10-14, NEUTRE+STABLE)

- `train_tok --span-aux` : bornes start/end sur `token_reps` (parts, sans re-lookup), gold = frontières alignements clauses, reflux partagé (2e update ctx/phrase documenté), snapshot/restore + save/load npz (`has_span`). Incident 0-d (`dst[:]`) corrigé par `np.copyto`. 87 verts (forward_ids + training + span + checkpoint).
- Coarse 30ep `--context --span-aux`, seeds 7/42/123, CSV `/tmp/e10_seed{7,42,123}.csv` :
  0.41042 (ep 28) / 0.38748 (ep 20) / 0.36342 (ep 25). Moy 0.387 ±0.02
  vs sans 0.448/0.377/0.345 (moy 0.390 ±0.05). Δ ≈ -0.003 ∈ bruit, variance ÷2.
- Verdict : neutre sur F1 (même signal clauses — attendu sans gold sub-clause), stabilisateur possible (loterie réduite, 1 mesure). L'infrastructure ancres/bornes est câblée au train ; sa valeur viendra du gold sub-clause (les 20 ternaires : noeuds propositionnels, pas clauses B).

## Bilan E0→E10

E0 0.107 (sac) → E4/E5 0.318/0.390 (contexte) → E6 0.351 (init MLM, bruit) → E7 0.375 (OOV hash, bruit) → E10 0.387 (spans aux, bruit, variance ÷2). Seul levier prouvé : architecture contexte. Débloqueurs : gold ternaire validé (fin + calibration N>=100 + GEL par facteur).

## E8 — second split final2 (exécuté 2026-10-14, GÉNÉRALISATION CONFIRMÉE)

- Split `v4/final2` (seed 42 quota 64) : train 506 / val 99 / test 99, 46/9/9 × 11 (avec control/data_dependency). SHA train 387ee96e / val 1ab19e41. CSV `/tmp/e8_final2_seed{7,42,123}.csv` + `/tmp/e8b_final2_nocontext.csv`.
- Sans contexte seed 42 : 0.10666 best ep 0 — identique à final au 5e chiffre. Sans contexte, rien n'apprend sur AUCUN split.
- Avec contexte : 0.30172 (ep 28) / 0.34986 (ep 29) / 0.41080 (ep 28). Moy 0.354 ±0.04 vs final 0.390 ±0.04. Seeds inversés (123 pire sur final, meilleur sur final2) → loterie confirmée, effet contexte (+0.25) la domine sur les deux splits.
- Verdict : gain généralisé, pas sur-ajusté. Défaut `--context` ? Non — attendre test N=99 + calibration avant changement de défaut.

## E11 — ternaires : loader + silver 20 (exécuté 2026-10-14)

- `tok_loader.convertir_phrase` passe `third` (rôle ∈ {Condition, Mediator}, noeud existant, sinon SpanError) → `edge_third`, et groupe les joints même (cible, relation) → `edge_joint` id déterministe. 2 tests loader verts.
- `gcn-datasets/scripts/build_ternary_proposal.py` : générateur (spans calculés depuis sous-chaînes alignées début-token, jamais indexés main) → `gcn-datasets/silver_ternary_proposal_20.json` (sha 9aebedc5, poids 0.0, versionné hors DATA).
- Chargé : 20/20, 0 rejet, 0 hors/arrières. 24 arêtes : conditional_cause 3, conditional_prevent 1, mediated_cause 3, mediated_prevent 1, joint_cause 4, joint_prevent 4, analogy 4, counterfactual 4. Thirds 8, groupes joints 2 clés.
- Statut : SILVER non validé, exclu du train (poids 0.0). Promotion = validation humaine par numéro (`gold-ternaire-proposition-20.md`) + gate IC95. Encore <30/classe : GEL maintenu, mais le format et le câblage sont prêts.

## E9 — test N=99 + calibration gatée (exécuté 2026-10-14)

- Nouveau : `--model-out` best-restore s1+ (E + têtes + RGCN + ctx + facteur + oov) + `scripts/eval_tok.py` (F1 test + ECE + gate N>=100). 82 verts (training + checkpoint + forward_ids).
- Modèle : final2 train seed 123 `--context`, best val 0.411 (ep 28), sauvé `/tmp/e8_best.npz`. Test 99 phrases : `test_edge_f1 = 0.275`, `test_node_f1 = 0.205`, loss 2.72, 0 rejet.
- Lecture honnête : chute val→test 0.136 — sélection best-sur-val optimiste (R5 prouvé sur test, pas en théorie) + N=99 petit + OOV 23.7%. Le 0.39 n'est pas un chiffre produit.
- Calibration : gate `require_production_sample` déclenché exactement comme conçu — N=99 <100 → `ece: NO-GO` documenté, aucun chiffre publié comme calibré. Le gate Ph5 fonctionne.

## E12 — espace facteur (exécuté 2026-10-14, GEL FIN LEVÉ)

- Nouveau : `--espace factor` — nœuds fins 8 + arêtes primaires factorisées (base 11 × ternary 4 × special 3, reflux tronc), doctrine D-R8F (`require` sur les 11 bases apprises 52×11 ✓, masques seuls nœuds présents 5/8 + ternary/special observés ; seuils/calibration restent gelés en aval). Éval via `compose()` restreint aux observés (jamais d'hallucination ternaire). 38 verts.
- Sans contexte seed 42 : 0.0366 best ep 23 — 11-voies sur sacs plus dur que 5-voies coarse (0.107) : difficulté, pas bug (métrique filtrée support>0 vérifiée `metrics.py:110-111`).
- Avec contexte, seeds 7/42/123, CSV `/tmp/e12c_seed{7,42,123}.csv` :
  0.23265 (ep 27) / 0.22430 (ep 27) / 0.25288 (ep 26). Moy 0.237 ±0.01.
- Lecture : fin S'ENTRAÎNE (GEL exit 2 levé pour les bases), 2.6× hasard, variance ±0.01 vs ±0.05 coarse — le partage factorisé régularise. Non comparable à coarse 0.390 (11 fines vs 5 grossières). Avec du gold ternaire (les 20 en attente), ternary/special deviennent apprenables au lieu de gelés.

## E5b — coarse post-P1 (exécuté 2026-10-15, pool FR pur 490/96)

- Même protocole E5 (coarse 30ep `--context`, seeds 7/42/123, `git_sha 2778fb4`) sur pool
  post-quarantaine P1 (552 tombstones : 545 EN + 7 latex), `data_hash 57b87557` :
  0.32648 (ep 28) / 0.34138 (ep 29) / 0.29432 (ep 29). Moy 0.320 ±0.02.
  CSV `/tmp/opencode/e5b_seed{7,42,123}.csv`.
- Vs E5 0.390 ±0.04 sur final gelé (572/132) : Δ -0.07 > bruit. NON comparable
  (pool différent — pas une régression du modèle) : train -14 %, val 96 (bruit d'estimation),
  classes rares amputées (P1 a retiré filter 49 / prevent 27 / data_dep 70 / control_dep 17
  avec les lots EN). Les phrases ENportaient du signal apprenable (CIR valides, modèle
  bytes quasi-agnostique à la langue) — ce qui confirme à la fois l'hypothèse universalité
  ET la nécessité P1b (remplissage FR, en validation).
- Rôle : référence post-P1 pour P5 (P5 mesurera l'effet P4+P1b vs E5b, pas vs E5).

## E5-P6/E12-P6 — validation PV1 en ablation propre (exécuté 2026-10-15)

- Protocole PV1 (délégation, `protocole-validation-machine.md` v1) : signaux auto
  S1-S4 + Red R1-R6, revue manuelle 288 cas + 50 cdb2 (R5) + 20 ternaires + 164 auto-oui
  (gate aveugle), re-jugement aveugle 19 cas → accord 74 % < 95 % ⇒ revue intégrale
  182 auto-oui + R5/R6 ajoutés (négation, spans dégénérés). Verdicts tracés
  `/tmp/opencode/verdicts_*.json`. Taux oui : minage 360/479 (75 %), cdb2 16/50 (32 %,
  actions sans contrôle — R5 a fait son travail), ternaire 20/20.
- Promotion : `lot27_pv1.json` (313, dédup 63) + `lot28_ternaire.json` (20) + 5 tombstones
  PV1 (2 sens + 3 quar). Correctif direction : 4 backs résiduels détectés post-merge
  (2 inversés convention P4, 2 exclus) → 0 backs actifs vérifié. Split `v4/p6/`
  (506/99/99 + strates ternaires 1-4, 579 tombstones, GATES OK). 1 rejet loader
  documenté (s0356 char-span lot05, 0.2 % < gate 5 %).
- **Leçon métrologie** : factor effondré sur P6-val (0.101) alors que P6-train est sain —
  la métrique val est hypersensible à la composition (micro-classes ternaires).
  Règle : ablations à val GELÉE (P5-val). L'hypothèse dilution-spans a été TESTÉE et
  réfutée (retirer les longs spans dégrade : 0.06-0.10) — R1-info-only maintenu.
- **Ablation propre, val P5 gelée, mêmes seeds 7/42/123** (CSV `/tmp/opencode/e5p6e_*`,
  `/tmp/opencode/e12p6d_*`) :
  coarse P6-train 0.49252 (ep 28) / 0.38024 (ep 26) / 0.45048 (ep 28). Moy **0.441**
  vs P5-train 0.396/0.435/0.221 (moy 0.351). Δ **+0.09**, seed 123 réparé (0.221→0.450).
  factor P6-train 0.30938/0.35588/0.30777. Moy **0.324** vs 0.306. Δ +0.018 (bruit).
- Erratum revue 2026-10-15 : la v1 citait `e5p6_seed*` (val P6 native : 0.394/0.337/0.385,
  moy 0.372) au lieu de `e5p6e_*` (val gelée). Les chiffres 0.441 étaient justes,
  la référence de fichier fausse — corrigée. L'écart natif-vs-gelé (0.372 vs 0.441,
  même train) confirme la leçon L2 par l'exemple.
- Verdict : le minage validé PV1 apporte du signal en coarse (+0.09, stabilise),
  neutre en factor à cette échelle (ternary 1-4 ex. inexploitables — gate R8 intact).
  La validation déléguée rigoureuse (72 % oui, 28 % rejetés dont 34/50 cdb2) est
  mesurée utile, pas décorative.

## E5-P5 — coarse post-remédiation (exécuté 2026-10-15, pool P5 506/99/99 RÉPARÉ)

- Recette P5 (validations humaines 2026-10-15 : convention sens oui, 3 quarantaines oui,
  P1b 129 oui sur échantillon) : P4 appliqué (514 inversions vérifiées, 0 backs actifs,
  backups `/tmp/opencode/p4_backup_20261015/`, `P4_propositions.avant.json`) + P1b promu
  (`lot26_p1b.json`, sha `a894a59c`) → `make_split --quota 64 --seed 42 --exclude-none`
  → `DATA/supervision/v4/p5/` (train 506/val 99/test 99, 46/9/9 ×11 TOUTES au quota,
  577 tombstones exclus dont 3 P4, `check_v4.py` GATES OK). Références gelées
  final/final2 intactes (E5 0.390 non rejoué, non comparable).
- **Incident loader (détecté et réparé le même jour)** : l'inversion P4 avait laissé
  l'ordre liste des nœuds inchangé, et `tok_loader.py:64-66` exclut les arêtes
  « arrières » (rang source ≥ rang cible) — 153 arêtes silencieusement jetées
  (`edge_gold` 353 au lieu de 506, 4 bases <30 → gate R8 factor GELÉ à raison).
  Réparation : `scripts/reparer_ordre_noeuds.py` (sources remontées avant cible,
  ids intacts, 105 lots) → `edge_gold` 46×11 (+enable 60), 0 arrières, 0 hors-fenêtre.
  Le chiffre 0.425 mesuré avant réparation est RETIRÉ (données dégradées).
- Même protocole E5 (coarse 30ep `--context`, 6 seeds, `git_sha 2778fb4`),
  CSV `/tmp/opencode/e5p5r_seed{7,42,123,0,100,2026}.csv` :
  0.39636 (ep 29) / 0.43524 (ep 25) / 0.22134 (ep 2, run effondré) /
  0.46310 (ep 29) / 0.36310 (ep 21) / 0.41134 (ep 28). Moy 0.382 ±0.08.
- Vs E5b 0.320 ±0.02 (même protocole, pool pré-P4/P1b ; E5b avait 20 arrières,
  effet négligeable) : Δ +0.062 > seuil 0.05, mais loterie ±0.08 domine
  (seed 123 effondré ; sans lui moy 0.414). Verdict : effet remédiation probable
  mais non verrouillé — la variance, pas les données, est le goulot mesuré nº1.
- Anti-biais sélection (2026-10-15, courbes `/tmp/opencode/e5p5r_seed*.csv`, 0 ré-entraînement) :
  best-ep moy 0.382 vs moyenne 5-dernières-epochs 0.369 (-0.013 de biais optimiste) ;
  seed 123 plate (~0.22, échec d'optimisation, pas artefact de sélection).
  Early-stop robuste = -0.01 de biais, variance inchangée : le remède est N (val/test ≥100),
  pas la règle d'arrêt.

## E12-P5 — factor post-remédiation (exécuté 2026-10-15)

- Même protocole E12 (`--espace factor --context`, seeds 7/42/123) sur P5 réparé
  (R8 franchi : 11 bases ≥46) : 0.32323 (ep 27) / 0.35119 (ep 29) / 0.24537 (ep 29).
  Moy 0.306. CSV `/tmp/opencode/e12p5_seed{7,42,123}.csv`.
- Vs E12 0.237 ±0.01 (final2) : Δ +0.069, même sens et même ordre que coarse (+0.062) —
  cohérent inter-espaces, mais pools non comparables + loterie. Ternary/special
  toujours gelés (gold en attente) ; seules les 11 bases apprennent.

## E-Gate — Silver 0→0.3 évalué (2026-10-15, MAINTIEN 0.0)

- Mesure d'erreur proposeur (`GATES.md:15`, verdicts PV1 comme vérité estimée, n=479) :
  119 erreurs → p=24.8 %, IC95 [21.0 %, 28.7 %]. Borne haute > 20 % ⇒ **poids silver maintenu 0.0**.
- Par relation : data_dependency 1/114 (excellent), filter 21/135 (16 %), control_dep 11/57 (19 %),
  prevent 37/111 (33 %, pire), condition 22/22 + opposition 10/11 + sequence 7/8 (échecs quasi
  tous directionnels — couverts par la convention P4 validée, pas par le proposeur).
- Conséquence : le gate reste opposable aux futurs batches. Aucun changement de poids
  sans re-mesure.
- **Re-mesure batch science/educ (2026-10-15, n=100 stratifié PV-4)** : 35 erreurs,
  p=35 %, IC95 [26 %, 44 %] — population plus dure (négations, définitions, bilingue,
  titres). Prevent 7/25 = 28 % (vs 33 %, bruit) : **zéro hijack `pour` résiduel**
  (fix confirmé à sa source), reliquat = négation (R5), spans dégénérés (R6),
  sens `avertir`, étymologie. Prédiction ~15 % NON atteinte au stade propose :
  le fix a multiplié la pureté silver ×4 mais l'erreur propose vient d'ailleurs.
  Prochains fixes B : négation (ne...pas + verbe), titres dégénérés, marker-stranding.

## E-Fix2 — négation prevent B (2026-10-15, tooling hors-moteur)

- `balanced_auto.py` : `_prevent_nié()` — skip prevent si nié (`ne...pas|jamais|plus|...`
  ±35/20 car, `n'` élidé géré), si `sans` + verbe-prevent suit/précède
  (`sans empêcher` ≠ `sans permettre` conservé), si `prévient que|,` (avertir).
  101 tests verts (26 annotate + 75 scraper).
- Mesure avant/après (`git stash`, 2192 phrases, 1er marqueur) : **147 flips, dont
  135 gains vers prevent (61 ex-motivation = le hijack, le reste incidental correct
  vérifié sur échantillon) et 12 pertes prevent, TOUTES des négations correctes**
  (`n'empêche pas`, `sans empêcher`, `aucune loi n'empêche`).
  Strictement améliorant, zéro régression mesurée.

## E-Fix — infinitifs prevent B (2026-10-15, tooling hors-moteur)

- Cause racine du hijack `pour` (minage prevent 18 %) : `balanced_auto.py` ne listait
  que les formes conjuguées (`empêche`, `évite`...) ; le regex à frontières de mots ne
  matchait jamais les infinitifs (`pour éviter/empêcher` → seul `pour`/motivation trouvé).
- Fix : +10 infinitifs FR (`empêcher`, `prévenir`, `éviter`, `bloquer`, `interdire`,
  `inhiber`, `protéger`, `freiner`, `entraver`, `neutraliser`) + 6 EN.
  Vérifié : `pour empêcher` → prevent premier ; 26 tests verts + 4 skips pré-existants.
- Données validées inchangées (verdicts sur textes complets). **Prédiction vérifiée
  le même jour** : re-screening des 336 candidats prevent → silver 211/336 (63 % vs 23 %),
  pureté prevent 123/211 = 58 % (vs 43 %) et 123/336 = 37 % des candidats (vs 10 %, ×4).
  Mécanisme confirmé : `fallback_regex.py` charge CONN depuis `balanced_auto.py`
  (fichier dépôt) — un seul fix, deux pipelines (silver + build_lot).
  Prochain gate Silver : erreur prevent attendue ~15 % (à re-mesurer).

## E5-P7/E12-P7 — minagese promu (exécuté 2026-10-15, lot29 133 oui)

- PV-4 : 100 échantillonnés (65/35) + 88 restants (68/20) → 133 oui promus
  (`lot29_minagese`, 0 doublon). Gate strict : 1 EN résiduelle (g0144) → tombstone +
  split_en (546). Split `v4/p7/` (512/105/107, 11 au quota + strates ternaires,
  580 tombstones, GATES OK). Coarse refuse le ternaire (fail-fast `label fin hors
  table` — bon comportement) : mesure coarse sur 11-directes, val P5 gelée.
- Coarse P7-train → P5-val : 0.41268/0.45174/0.47476 (ep 29/29/29). Moy **0.447**
  vs P6 0.441. Δ +0.006 ∈ bruit : +133 phrases n'ajoutent rien de mesurable à N=99.
  Coarse sature ~0.44 sur cette val — le volume seul ne monte plus.
- Factor P7 full (ternaire inclus) : 0.29468/0.23319/0.31828. Moy **0.282** vs 0.306.
  Neutre/bruit (métrique incluant micro-classes ternaires).

## E9-bis — test P7 (exécuté 2026-10-15, CHIFFRE PRODUIT DÉCEVANT)

- Modèle : coarse P7-train seed 42 `--context`, best val 0.452 (ep 29),
  `/tmp/opencode/e9b_best.npz`. Test P7 11-directes (99 phrases) : `test_edge_f1 = 0.158`,
  `test_node_f1 = 0.281`, loss 3.38, 0 rejet. OOV test 28 %.
- Chute val→test 0.294 (vs E9 0.136 sur final2) : le +0.09 val P6/P7 NE TRANSFÈRE PAS
  au test (0.158 < E9 0.275). Causes probables : sélection best-sur-val optimiste sur
  val N=99 + test plus dur (spans minage longs 16.2 tok, OOV 28 %) + loterie.
  Le 0.44 val n'est pas un chiffre produit — honnêteté maintenue.
- Calibration : ece=0.355, n=103 (compte ≥100 mais échantillon test ≠ production :
  NO-GO maintenu, quantification toujours enterrée).

## E7-bis — OOV-hash sur P7 (exécuté 2026-10-15, NEUTRE confirmé)

- Coarse P7-train `--context --oov-hash`, val P5 gelée, seeds 7/42/123 :
  0.40910/0.46992/0.42658. Moy 0.435 vs zéros 0.441. Δ -0.006 ∈ bruit.
  Même verdict qu'E7 à OOV 21 % : l'OOV (28 % test) n'est pas le levier, même à ce niveau.
- Bilan des leviers testés : seul le contexte architectural compte (+0.21, E4/E8) ;
  init (E2/E6), OOV-hash (E7/E7-bis), spans-aux (E10), early-stop (mesure seule) :
  tous neutres. Reste : données (gold, N) et ternary/special (gold requis).

## E5-P8/E12-P8 — tb2 promu (exécuté 2026-10-15, lot31 57 oui)

- PV-4 R5 : 59 jugés, 57 oui / 2 non (tb2016/tb2017 : src chevauche tgt, circulaire).
  `lot31_tb2.json` (57 : 37 ternaires + 20 contrainte). Split `v4/p8/`
  (train 547/val 107/test 107, 11 au quota 46 + ternaires 4-9, 580 tombstones, GATES OK).
  Ternary pool : analogy 9, conditional 9+6, counterfactual 10, joint 8, mediated 7+6
  (toujours <30 : R8 tient, mais ×5 en une journée).
- Coarse P8-train → P5-val gelée : 0.44066/0.39280/0.44428. Moy **0.426** vs P7 0.447.
  Δ -0.02 ∈ bruit : 4e split consécutif dans la bande 0.43-0.45 — saturation confirmée.
- Factor P8-train → P5-val : 0.29189/0.25886/0.27496. Moy **0.275** vs 0.324 (P6-ablation)
  et 0.306 (P5). Fluctuation inter-pools sans tendance (0.27-0.32) : factor plat-bruité,
  pas de gain mesurable du volume validé à cette échelle.

## E8-révisé — `--context` en défaut (exécuté 2026-10-15)

- Décision : défaut ON (`action store_true default=True` + `--no-context` échappatoire),
  car +0.21 prouvé E4/E8 généralisé 3 splits ; la condition suspensive (calibration)
  portait sur les confiances, orthogonale au défaut d'architecture.
- Vérifié : sans flag → `context:true` (apprend) ; `--no-context` → régime sacs 0.113.
  Tests `test_context.py` + `test_cli_tok.py` : 12 verts. Harnais E inchangés
  (passent `--context` explicite, comportement identique).

## E5-P9/E12-P9 — mc3 promu (exécuté 2026-10-15, lot32 70 oui)

- PV-4 R5 : 74 jugés, 70 oui / 4 non (src⊃tgt circulaire, cibles mal étiquetées,
  sens inversé). `lot32_mc3.json` (37 ternaires + 33 directs). Gate strict : 2 EN
  résiduelles → tombstones (g0144/d0144, doublons) + 1 FP langdetect (titres EN,
  `exceptions_bruit.json` créé). Split `v4/p9/` (580/111/111, 11 au quota 46 +
  ternaires 2-16, 581 tombstones, GATES OK). Ternary pool : analogy 15,
  conditional 16+11, counterfactual 16, joint 14+2, mediated 12+12.
- Coarse P9-train → P5-val gelée : 0.43330/0.40498/0.41216. Moy **0.417** :
  5e split dans la bande, saturation confirmée (volume validé ≠ gain).
- Factor P9 full : 0.22274/0.22258/0.17046 (ep 10, seed 123 effondré). Moy **0.206**,
  en baisse (0.24-0.32 avant) : les têtes ternaires à 2-16 exemples ajoutent du bruit
  sans signal — R8-bases tient, mais le factor n'apprend rien des miettes.
  Conclusion : sans gold ternaire à ~30/classe, le volume ternaire dilue.

## Revue déléguée — verdict (2026-10-15, autorisation juge humain)

- Code : 0 fichier `.rs` touché (cargo inutile, documenté) ; pytest 981 passed + 3 xfailed
  pré-existants ; 26 annotate + 75 scraper verts après les 2 fixes B.
- Données : audit couverture 100 % (609 jugés, 595 promus oui, 0 divergence) ;
  spot-check v2 : 30/30 maintenus, 0 renversé.
- Mesures : re-run P9 seed 7 bit-identique (0.43330 ep 29) ; 21 CSV revérifiés contre
  baseline — 20 conformes, **1 erratum trouvé et corrigé** (citation `e5p6_seed*`
  au lieu de `e5p6e_*`, chiffres inchangés).
- PR : diff vs master = 331 fichiers (historique refacto) — la revue porte les 2 commits
  (`2778fb4..c1423f0`) ; hunks Makefile/train_tok.py mixtes (sessions E) signalés,
  solo-repo, fonctionnellement couverts par les suites vertes.

## E5-P10/E12-P10 — mc4a promu + leçon val-gelée (exécuté 2026-10-15, lot33 76 oui)

- PV-4 R5 : 76 jugés pattern + labels nettoyés (39 src circulaires), 76 oui / 0 non.
  `lot33_mc4a.json` (conditional 37 + mediated 39). Split `v4/p10/`
  (train 632/val 123/test 123 ; train : 46×11 + ternaires 4-27, conditional_cause 27
  et mediated 25+22 proches des 30 ; 581 tombstones, GATES OK).
- Coarse P10-train → P5-val gelée : 0.52764/0.53326/0.58300 (ep 29/29/28).
  Moy **0.548** — record, MAIS :
- Coarse P10-train → P10-val native (99) : 0.32424/0.37778/0.39858 (ep 20/28/29).
  Moy **0.367**. Écart gelée-native **0.18** sur même train.
- **Leçon (méthodologique, majeure)** : ~20 expériences sur la même val gelée ont
  méta-surajusté à la val (sélection implicite de pools + reporting du meilleur).
  Le +0.09 P6 et le +0.13 apparent sont contaminés — ampleur inconnue, à supposer
  non nulle. Règle ajoutée (PV-8 L6) : toute conclusion exige double mesure
  (val gelée + val native/fraîche) ; écart >0.05 ⇒ sur-ajustement, pas gain.
  Le seul chiffre hors-échantillon restant : test 0.158 (E9-bis).
- Factor P10 full → P5-val : 0.25445/0.24194/0.28715. Moy **0.261**, plat.

## E5-P10c/E5-P6c — ablation non biaisée (exécuté 2026-10-15, val P11 seed 123)

- Correctif L6 : split frais seed 123 (`v4/p11/`), val 11-directes purgée des
  recouvrements trains (99→58 phrases propres, jamais utilisées). CSV `e5p10c_*`, `e5p6c_*`.
- P10-train → val propre : 0.34754/0.34160/0.33400 (ep 29/29/29). Moy **0.341 ±0.01**.
- P6-train → val propre : 0.39694/0.18714/0.29732 (ep 25/16/28). Moy **0.294 ±0.08**.
- Lecture : Δ +0.05 même sens que le +0.09 gelé (ordre de grandeur confirmé, ampleur
  révisée à la baisse) ; surtout **stabilisation** : P10 ±0.01 vs P6 ±0.08 (seed 42
  effondré sur P6, jamais sur P10). La val gelée masquait l'instabilité P6.
  Réserve : N=58, une seule val — intervalles larges, pas de triomphalisme.

## E9-ter — test P11 frais (exécuté 2026-10-15, RÉPLIQUE)

- Modèle : coarse P10-train seed 7 (déterministe : re-run bit-identique 0.52764 ep 29),
  best val P5 gelée → test P11 11-directes (99 phrases, jamais utilisées) :
  `test_edge_f1 = 0.165`, node 0.292, loss 3.21, 0 rejet, ECE 0.281 (n=103).
- Vs E9-bis (P7-test) 0.158 : **deux points test indépendants concordent (0.16)**.
  Le niveau test est stable et bas ; l'écart val→test (~0.3) n'est pas un accident
  de split. Le chiffre produit = 0.16, pas 0.44.
- Leçon infra : `/tmp/opencode` a été nettoyé (CSVs perdus — nombres déjà vérifiés
  pré-perte, cf. revue). Politique : `--csv-out` désormais vers
  `DATA/supervision/v4/mesures/` (persistant local). Splits p4-p11 intacts (DATA).

## Ph5 — calibration production : NO-GO documenté (2026-10-16, RÉSULTAT MAJEUR)

- Échantillon : 270 phrases FR presse scellées (c5_presse 20261012, seed 20261016),
  hygiène H aveugle-modèle (49 exclues biblio/non-FR/sans-verbe), forward e9ter gelé :
  101/270 phrases produisent des paires (299). Vague 2 : 150 phrases → 48/150 (100 paires).
- PV1 manuel exhaustif vague 1 (123 paires, paraphrase+contre-lecture, motifs/sid
  en `validation_pv1/verdicts_ph5.json`) : **4 oui + 2 corriger = 6 golds,
  précision production 4.9%** (IC95 [2.2%, 10.2%]).
- Gate `require_production_sample(N>=100)` : N=6 → **NO-GO formel. Pas d'ECE publié.**
- Surconfiance massive et mesurée : médiane conf des 117 faux = 0.47 ;
  **14 faux à conf ≥ 0.80** (causale 0.99 sur « d' », 0.97, 0.96, 0.92 sur « « »,
  0.89 sur « » », 0.82-0.90 sur fragments de noms) ; 2 golds à conf < 0.50.
- Cause racine : la segmentation B produit des clauses dégénérées sur texte réel
  (ponctuation seule, mots isolés, rubriques, coupures intra-nom propre) et la tête
  arêtes prédit entre fragments avec haute confiance. Les vraies relations
  intra-clause sont manquées (ph5-178 « grâce à », ph5-205 « face à »).
- Classes de faux : rubrique/paratexte ×19, ponctuation ×22, fragments ×~40,
  apposition/identification ×6, circonstance ×2, finalité ×2, concession ×2
  (hors coarse-5), juxtaposition ×4, sujet-prédicat ×1, paywall/UI ×4.
- Aveugle humain 10% : DEVIENT SANS OBJET (pas de golds à exploiter pour ECE).
- Pivot imposé : le prochain chantier n'est pas la calibration, c'est
  (1) gate d'entrée anti-dégénéré (FAIT — voir § Gate-Ph5 ci-dessous),
  (2) segmentation B sur texte réel. Sans (2), toute calibration reste vaine.

## Gate-Ph5 — abstention dégénérée (2026-10-16, FAIT et MESURÉ)

- Implémentation : `clause_degeneree()` + filtre tracé `clauses_rejetees` dans
  `nlu/tok.py::parse_text` (entrée inférence seule, jamais le train) ;
  tests `gcn-python/tests/test_tok_gate.py` ; suite 984 passed.
- Règles (forme seule, zéro lexique) : clause vide, ≤2 tokens, ponctuation pure.
- Effet réel re-mesuré (forward gaté, mapping topologique) : 123 → 57 paires
  (-54%), 6/6 golds conservés, **précision 0.049 → 0.115**,
  **faux à conf ≥0.80 : 14 → 1**. 5 paires non-mappables (effet topologie fenêtre,
  documenté — le gate change l'adjacence, c'est son objet).
- Restant : précision 11.5% << exploitable → chantier segmentation B ensuite.

## Seg-B — pipeline v2 : précision ×7 (2026-10-16, FAIT et MESURÉ)

- Diagnostic : clauses inférence = phrases (spaCy sents) vs clauses train =
  propositions (nœuds gold) → mismatch de granularité ; spaCy sur-fragmente
  les titres/références (17 clauses sur un titre EN).
- B-a (garde-fusion, `engine_b.fusionner_clauses`, socle — bénéficie aux deux
  chemins) : refusionne toute coupure sans ponctuation forte (.!?…؟？).
  Forme seule, zéro lexique, idempotent, no-op fallback. Effet : 257/270
  phrases mono-clause (fini les paires sur fragments de titres).
- B-b (subdivision connecteurs, `nlu/tok.subdiviser_connecteurs`, chemin tok
  seul) : coupe aux connecteurs des tables existantes (aucun lexique nouveau),
  connecteur avec la partie droite ; re-gate ensuite. Effet : 174/71/22/3
  clauses/phrase, 152 paires dont la structure est saine (vs 57).
- Gate resserré (1 token au lieu de 2 — « Il reste » S+V conservé).
- PV1 manuel exhaustif v2 (152 paires, motifs/sid en
  `validation_pv1/verdicts_ph5_v2.json`) : **24 oui + 28 corriger = 52 golds,
  précision 0.342 (×7 vs 0.049)**. Golds : causale 20, negative 14,
  restrictive 8, sequentielle 6, systemique 4.
- Confusions systématiques (signal réentraînement tête) : restrictive→causale
  ×7 (« donc », « car », « permet », « conduit à »), negative→causale ×6,
  causale→systemique ×3/negative ×3, negative→sequentielle ×3.
- ECE : N=52 <100 → toujours NO-GO, mais à 52% du gate (contre 6%).
  Chemin : ~150 paires v2 de plus à juger (nouveaux candidats du réservoir).
- Résidus B-c : B-c1 FAIT (`subdiviser_connecteurs` : connecteur en tête →
  coupe à la première virgule, protase|apodose — ponctuation seule ;
  162/83/22/3 clauses/phrase). B-c2 (refusion sans-verbe) **RETIRÉ avant
  commit — règle projet : jamais de hardcoding ni de mot-matching dans le
  moteur** (listes verbales et stoplist supprimées du code ; les fragments
  restants passent par le gate de forme ou sont jugés non en PV1).
  Restent : connecteurs discursifs qui miettent, abréviations (hors scope),
  liste EN hygiène (one-shot, pas moteur).

## Ph5-ECE — calibration GO (2026-10-16, RÉSULTAT HISTORIQUE)

- Vague 3 (220 phrases fraîches + H, pipeline v2) : 95 phrases, 145 paires,
  PV1 exhaustif → **19 oui + 35 corriger = 54 golds (37.2%)**, cohérent v2.
- Cumul : **52 + 54 = 106 golds**. Gate `require_production_sample(N>=100)` :
  **FRANCHI**.
- `optimize_temperature` : **ECE brut 0.145 → T*=2.50 → ECE-temp 0.078 ≤ 0.15 :
  GO** (isotonie non requise). Distribution : causale 37, negative 29,
  restrictive 17, systemique 12, sequentielle 11 (classes <30 : ECE global
  valide, ECE par classe non garanti — noté). Auto-contrôle : erreur de
  classement v3-2217 (oui au lieu de corriger) détectée via l'aveugle,
  corrigée avant clôture (ECE 0.154→0.145/0.088→0.078 — GO inchangé).
- Checkpoint calibré : `DATA/supervision/v4/mesures/e9ter_calibre.npz`
  (T=2.5 + provenance) — vérifié : 0.618 → 0.374 sur exemple.
- T*=2.5 = modèle très surconfiant : cohérent avec les 14 surconfiants v1.
- Aveugle humain 10% (11 cas, seed 20261021, `silver_ph5_AVEUGLE.md`) :
  **EN ATTENTE — gate ≥95% (11/11 requis)**. Si échec : golds humains intégraux.

## Cause racine + voie B fermée (2026-10-16, DIAGNOSTIC PROUVÉ)

- `train_tok.py:518,533` : la perte n'accumule QUE les paires gold
  (`if p not in edge_gold: continue`) — **les vrais négatifs ne sont jamais
  entraînés**. L'éval fait pareil. La tête apprend « quelle relation sachant
  qu'il y en a une », jamais « y en a-t-il une ». L'absence n'existe pas
  dans l'objectif — d'où paires sur fragments + surconfiance structurelles.
- Voie B (seuil d'abstention sur conf calibrée) TESTÉE et FERMÉE : courbe
  précision/rappel sur 297 paires (106 golds) — max-F1 au seuil 0.0
  (tout garder, P=0.357) ; P ne dépasse 0.38 à aucun seuil ; médianes
  conf gold/non 0.50/0.45 (recouvrement total). La confiance ne porte
  AUCUN signal de détection — le seuil est inutile.
- Reste la voie A (classe d'absence) : tête 5→6 (ou tête d'existence),
  réentraînement avec les ~190 vrais négatifs PV1, éval détection+F1,
  rupture compat checkpoint, revalidation E. **Changement d'architecture —
  décision humaine requise (scope, compat, effort).**

## Voie A — verdict nuancé (2026-10-16, GO DÉCISION)

- GO humain obtenu. Réalisé : tête d'existence (MLP d_edge→32→1, pattern
  R2 tête-seule puis reflux tronc), loader coarse-natif + `exist_gold`,
  BCE masquée, snapshot/npZ v4 (compat anciens checkpoints), 305 clauses
  typées PV1-lite, split prod 127/64, campagne 3 seeds (val 0.497/0.608/0.684).
- **Détection : NO-GO triple-preuve.** (1) Collapse total (F1=0.000, tout
  abstenu) tête-seule ET avec reflux. (2) Sonde gelée : train F1=1.000,
  test F1=0.528 < baseline tout-positif 0.578. (3) Courbe d'apprentissage
  plate (0.49/0.44/0.52/0.53 pour n=34→135). Les vecs figés ne portent AUCUN
  signal de détection généralisable à cette échelle.
- **Relation : GO INATTENDU.** Ablation propre (seed 123) sur test P11 gelé :
  P10-old 0.165 → P10-rebuilt 0.179 (+0.014, bruit) → +prod-golds 0.261
  (+0.082, significatif). **Les 69 golds production valent +0.08 test F1.**
  Nouveau meilleur test : 0.261 (modèle `aug_s123`).
- Statut modèles : `e9ter_calibre` RESTE la référence production (triples +
  T=2.5 prouvé N=106). `aug_s123` = meilleur relation (0.261) mais tête
  existence effondrée (zéro triple) + calibration indicative (ECE-temp 0.087
  N=69, T* au bord de grille 5.0) → éval-only, pas shippé.
- Infra gardée (testée, non-breaking) : tête existence, `exist_gold`,
  coarse-natif, BCE — disponible quand l'échelle (gold-3000) le permettra.
- Leçon : à ~200 paires, la supervision production sert la RELATION
  (diversité), pas la DÉTECTION (échelle insuffisante d'un ordre).

## Programme gold-3000 — vagues J/S/T/bulk/cond + p12 (2026-10-16, EN COURS)

- Lots promus : **lot34 joints** (20+20, joint_prevent 2→22, joint_cause
  14→34), **lot35 spécial** (20+20, analogy 15→35, counterfactual 16→36),
  **lot36 ternaire-T** (10×4, conditional/mediated 27-37→37-47),
  **lot37 bulk** (126, cause×60 enable×35), **lot38 cond** (71 dont 65
  condition 95→160). Aveugle groupé en file : **32 cas**.
- **p12** : quota 100 + exclude-none + exclude-lots=lot37 (manifest tracé),
  980/207/207 (11-directes 770/165/165). Val/test +68% vs p11 ; couv.
  éval combinée ~300 (207 split + 94 prod gelé).
- **Fuite inter-splits DÉCOUVERTE et PROTOCOLE** : les re-splits rééchantillonnent
  le même pool → recouvrements train/test (p12-test∩p10-train 31%,
  P11-test∩p12-train 37%). Tous les chiffres E9-ter/bis/aug sont des bornes
  hautes. Règle : exclusion overlap systématique + comptage publié.
  Effet mesuré petit (±0.02-0.04) mais réel.
- **lot37 DÉGRADE le test (-0.08)** : ablation propre (même seed/config/test) :
  avec bulk 0.116 → sans bulk 0.196 ; split propre p12b 0.174. Cause :
  spans style TOK-regex (« L ' ICA ») = bruit d'alignement (labels PV1 OK).
  lot37 exclu des splits (`--exclude-lots`, labels intacts) en attendant
  re-spanning. Les golds PV1 propres (69 prod) HELPENT (+0.08) : c'est le
  STYLE de spans qui compte, pas les labels.
- Tableau test honnête (clean) : e9ter→P11 0.164, e9ter→p12 0.273,
  aug→P11 0.240, p12b→p12 0.174. Le CLASSEMENT DÉPEND DU TEST (±0.1) :
  aucune supériorité de modèle n'est établie sous 0.1.

## Ph5 vague 4 — B-c1 validé, précision stable (2026-10-16)

- 150 phrases fraîches (H1/H2 seuls — H3 verbal abandonné, règle
  anti-mot-matching), pipeline B-c1 deux passages : 65 phrases, 163 paires
  (rendement ×1.7 vs v2 — B-c1 extrait plus, ex. protases « Parce que »).
- PV1 exhaustif : **19 oui + 41 corriger = 60 golds (36.8%)** —
  quatrième point stable (34.2/37.2/36.8). Cumul production : 166 golds.
- B-c1 prouvé utile : les causales en tête autrefois monolithiques
  (v2-254) sont extraites ; les corrections restent dominées par
  causale/sequentielle/systemique explicites ratées par la tête.
- Doctrine révisée (longitudinale) : sujet + verbe d'effet
  (perturber/entraîner/permettre-X-tiers) = causale, PAS sujet-prédicat
  (v2-98, v1-219 corrigés ; v2-3089-e0 « diriger » reste NON — pas d'effet).
- ECE révisé N=107 (2 labels ajoutés) : brut 0.159, T*=2.50 (stable),
  ECE-temp 0.086 → GO inchangé. Checkpoint e9ter_calibre inchangé (T=2.5).

## Prochain : aveugle groupé (32 cas) → merge PR → vagues suivantes (bulk re-spanning, térnaire fin)
