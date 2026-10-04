# Reprise demain — état au 2026-10-03 ~18h35

## Objectif (corrigé 2026-10-04 — l'ancienne formulation est fausse)
Ancien libellé « 100k silver : 50k lignes code + 50k textes » :
AUCUN FONDEMENT dans les docs (ni code annoté — seul du texte est
annoté — ni cible 100k). Il n'y a jamais eu d'objectif v5.
Objectif réel : **v4 texte seul** (gold + mesures), données v5
conservées sur disque mais hors objectif, plus aucun travail v5.
Cibles prod (`TODO.md`) : `val_edge_macro_f1 > 0.40`,
`val_node_macro_f1 > 0.60`, `val_graph_exact_match > 0.20`, gap < 0.15.
Doctrine : gold intact, tombstones jamais suppression, DATA hors git, `git add` fichiers
explicites, quotas ±10 % AU SPLIT (`--quota`), défauts inchangés sans mesure 3 seeds (Δ<0,05=bruit).

## Compteurs
- **Code : 79008 lignes** (inchangé, objectif dépassé). 603 fichiers, 6→5 relations.
- **Texte v5 : 4661** (inchangé) — éduc 1858 · divers 964 · science 915 · presse 807 · doc-code 117.
- **v4 gold : 1597** (+247 session : lot20 5→4+39, lot21 +83, lot22 +80).
  Par relation : cause 295 · concession 192 · condition 74 · control 53 · data 115 ·
  enable 196 · filter 92 · motivation 131 · opposition 150 · prevent 73 · sequence 161.
- **Quarantaines : 123** (dont lot20 10, lot21 10, lot22 13 ; vague1 51).
- Top-up lot11 : 10 isolées (`topup_lot11_pending.json`, corrigées, hors splits).
- Pool restant : ~4000 candidats (5707−1597−123, doublons pools inclus).

## Arbitrages session (assistant délégué, veto humain possible)
- Lot20/top-up (63) : 53 garder (4 retypes, 4 explicit:false, 10 labels re-ancrés,
  4 nœuds [1,1] réparés) + 10 quar. s1679 sorti du gold (`si` interrogatif).
- Lot21 (93, seed 42 rares-first) : 83 gold (15 strict-auto) + 10 quar.
- Lot22 (93) : 80 gold (22 strict-auto) + 13 quar (5 sorties en revue 2 :
  `sans` circonstanciel, `contrôle` mentionné ×2, `not limited to`, titre accolé).
- Motifs quar récurrents : noms propres (`Without Borders`, `Race Against…`),
  `sans` circonstanciel, `si` interrogatif/intensif, `contre` non-préventif
  (ennemi/collision/nom propre), `active` adjectival, `ainsi`→explicit:false (×14).
- Vérif par lot : 0 span >15, 0 chevauchement, 0 doublon, labels⊂spans, gates `check_v4` OK.

## Dette outillée payée (3 bugs pipeline arbitrage, cf. scripts)
- Boucle extension/repli TRIM qui oscillait (hang) ; clamp qui inversait les spans
  nichés (`[16,4]`) — ordre textuel ; extension qui ré-ajoutait la ponctuation.
- Leçon : nœuds n1/n2 ≠ ordre textuel ; scan de sortie obligatoire
  (spans/labels/marqueurs/chevauchements).

## Commits session (branche feat/p2-frontends)
`74ee80c` arbitrage lots 20/21/22 + select_lot exclut topup/pending.
Vérifié 2026-10-04 : `HEAD == origin/feat/p2-frontends` (0 ahead, diff vide)
→ déjà poussé, mention "NON POUSSÉ" caduque.
`select_lot.py` modifié (suivi) : exclusion `topup_*`/`*_pending` + rappel caps =
noms exacts (`control_dependency`, pas `control`).
Précédents : `b580b7f` pools rares2 · `3cddf44` fix proposeur (gate219 0,749).

## En-cours / à faire demain (ordre)
1. **Veto humain** éventuel sur verdicts 20/21/22 (tous marqués `assistant délégué`,
   réversibles via patchs + `ARBITRAGE_63_propositions.md`) ; **push** `74ee80c` si OK.
2. **Lot23** : mêmes caps dispos (filter/control/data/prevent/condition/sequence/
   opposition/cause). `enable/motivation/concession` = 0 dans le pool restant →
   scrape ciblé frais si on les veut.
3. **Volume v5** : science hors rate-limit (HAL=0) ; presse/éduc (~45k) ; doc-code épuisé.
4. **Gates** 808 science-web + `aveugle_200` (200 verdicts null).
5. **Chantier proposeur** (vrai verrou, gate 0,749 < 0,85) : segmentation ou multi-arêtes.
6. Dette : runs merge 2330 non archivés dans MESURES.md ; lot19 data-heavy assumé.

## Fichiers clés
- `scripts/arbitrage_lot{20,21,22}.py` (rejouables) · `scripts/select_lot.py` ·
  `scripts/build_lot.py` · `scripts/check_v4.py` · `scripts/gate_accuracy.py --gold219`.
- Données (hors git) : `v4/annotated/lot21_83.json` · `lot22_80.json` · `lot20_39.json` ·
  `lot20_05.json` (4) · `quarantaine_lot{20,21,22}.json` · `ARBITRAGE_lot{20,21,22}.md` ·
  `ARBITRAGE_63_propositions.md` · `v5/*_propose.json` (4661) · `v5/gates/aveugle_200.json`.
- Règles : `docs/REGLE_EQUILIBRE_DATASET.md` §5 · `docs/PROTOCOLE_MESURE.md`.

## Vérification 2026-10-04 (5 points figés)
1. **Push** : `74ee80c` déjà sur origin, rien à pousser.
2. **Doublons inter-lots (×2)** : `r2-00275` (lot21) = `g0101` (lot12,
   filter, spans ~identiques) ; `s1182` (lot16) = `s1133` (lot10,
   concession, casse seule différence). Tombstones ajoutés
   (`quarantaine_lot21.json`, `quarantaine_lot16.json` créé), garder les
   antérieurs. `make_split.py` exclut désormais quarantaines +
   `*_pending.json` + dédupe textes (1597→1595 phrases livrées).
3. **`check_v4` par lot** : N_min non applicable aux lots unitaires
   (lot20_39 : concession 1, filter 1). Gate N_min = niveau merge/split.
   191 spans >15 sur l'historique (lots 01-11, 19) : gelé, pas de
   réécriture (doctrine : pas de relabellisation mécanique).
4. **65 `_none` gold** : edgeless à part (`make_split --quota` ne les
   plafonne pas), à typer ou exclure explicitement au prochain merge.
5. **Gate proposeur 0,749** confirmé (164,02/219, cause 0,65 point bas).
   Dette MESURES : runs merge ~2330 introuvables → tracés NON ARCHIVÉS
   dans `mesures/MESURES.md` (à rejouer si besoin). Manifest v4 complété
   (`candidates_rares2.json`, 2698 phrases).

## Lot23 arbitré 2026-10-04 (93 seed 42, caps rares-first)
Caps : filter/control_dependency/data_dependency/condition/sequence/
opposition:12, cause:11, prevent:10. Build : 93 LOT, 0 QUAR, 78 LITIG.
**Gold 82** (15 strict-auto, 13 labels re-ancrés, 4 spans imposés,
2 retypes : s1196 condition→cause, s2306 cause→concession,
3 explicit:false `ainsi`, 2 nœuds [1,1] réparés) + **11 quar**
(5 `contrôle`/`active` mentionnels, `contre` ennemi, catalogue,
2 fragments, `si` intensif). Vérif : 0 span>15, 0 chevauchement,
0 doublon, 0 label hors span (mieux que 20/21/22 : 1/6/4).
N_min lot : concession 1 (gate niveau merge, cf. P0c).
Fichiers : `scripts/arbitrage_lot23.py` (rejouable) ·
`v4/annotated/lot23_82.json` · `quarantaine_lot23.json` ·
`ARBITRAGE_lot23.md`. Gold fichiers : 1679 (1597+82).

## Lot24 arbitré 2026-10-04 (103 seed 42, proposeur 0.854)
Nouveau cache propositions (ancien cache pré-0.854 jeté). Caps lot23 +
enable:2 (pool=2 !) + motivation:8 (pool=65 — `pour`+infinitif marche).
Build : 103 LOT, 0 QUAR, 91 LITIG.
**Gold 87** (9 strict, 14 labels re-ancrés, 3 spans imposés,
3 retypes : r2-00393/r2-00909 filter→condition (`if and only if`),
r2-01039 control→prevent (activation niée), s1591 →cause,
6 explicit:false `ainsi`, 3 nœuds [1,1] réparés) + **16 quar**
(4 `contrôle` mentionnels — conflit veto, `calls`/`needs` nominaux,
4 `sans` circonstanciels, `si oui`/`si` intensif, `pour commencer`,
titre+marqueur erroné). Vérif : 0 span>15, 0 chevauchement, 0 doublon,
0 label hors span. Concession 0 (pool vide).
Fichiers : `scripts/arbitrage_lot24.py` ·
`v4/annotated/lot24_87.json` · `quarantaine_lot24.json` ·
`ARBITRAGE_lot24.md`. Gold fichiers : 1766 (1679+87).

## Volume v5 entamé 2026-10-04 (gate 808 OK, batch arxiv2 +27)
- **Gate 808 science-web TENU (structurel)** : `validate_cir.py` corrigeait
  le schéma v1 (`source`) alors que v5 est en v2 (`sources`) → 0 % valide
  bidon. Fix : accepte `sources` + repli `source`
  (`gcn-tools/gcn-scraper/.../validate_cir.py`). Après fix :
  **4661/4661 valides (100 %)**, dont 808 science-web.
- **Batch `science-arxiv2`** : scrape arXiv EN seed 42 (hors HAL) → 97
  phrases → build_v5 → **27 propose** (70 quar "sans proposition" =
  abstracts descriptifs non-causaux, rendement 28 %).
  `v5/science/science-arxiv2_{propose,quar,litig}.json`, 27/27 valides.
  Brut + checkpoint reprise dans `gcn-datasets/DATA/raw/`
  (reprise wiki : `gcn-scrape ... --resume` avec cet output-dir).
- **Leçons volume** : scrape lent (~15 s/phrase : délais 3 s arXiv,
  pagination wiki) ; runs >25 min tués avec les outputs (écrire petit :
  `--max-per-query 10`, cibles ≤150/run) ; arXiv ML peu causal →
  privilégier wikipedia/presse/éduc pour le volume (~45k restants :
  science ~24k, presse ~9,2k, éduc ~8,1k, divers ~4k).

## Aveugle_200 arbitré 2026-10-04 (133 garder / 67 quarantaine)
Verdicts `garder`/`quarantaine` + motif renseignés en place
(`v5/gates/aveugle_200.json`, ids/textes intacts) via
`scripts/verdicts_aveugle.py` (rejouable) + `AVEUGLE_200_verdicts.md`.
Quar/rel : sequence 38, cause 14, data 4, control 3, filter/opposition/
prevent 2, condition/enable 1. Faiblesses systématiques : `first`
ordinal/nominal/propre → sequence (~20), `become` statif → sequence
(~10), `nécessaire` adjectival → data (3), titres/sections accolés,
doublon inter-registres divers-en-news-001703 = presse-en-002171.
Prochaine étape : les merges silver doivent exclure verdict=quarantaine.

## Proposeur 0,749 → 0,854 (gate219 tenu, +23 nets) 2026-10-04
`gcn_annotate/balanced_auto.py` (tests package OK : 26 pass, 4 skipped).
Précision (aveugle_200) : garde ordinal `first`/`primero` (dét/possessif/
`be+participe`/`half`/`-composé`), `becomes?`/`devient` nus retirés
(410 séquences spurieuses silver réduites au silence), `pour`/`para`
exigent infinitif, `end up` intact (non mesuré : à durcir).
Rappel (gate) : `même si`→P0 implicite OK, `Si+néanmoins`→concession,
clivée `c'est parce que` prioritaire, `pas accès`→prevent,
`oppose/opposé(s)`→opposition, `en raison du/des`+`à cause du/des`,
`lorsqu` (élision), `driven by`, `une fois`, `certes` tête,
`alors que/tandis que/subsequently/thereafter`→P0, `temporel>pour`,
`_strip_lead` débris + split `,(?!\d)`, seuils contenu 4→3 / total 4→3.
Limites connues : `contrôle` nominal (s065/s112 — gold lui-même met
control sur mentions définitionnelles s114/s115 : **conflit doctrine à
trancher au veto lot23**, 5 quarantaines `contrôle`/`active` à revoir),
`contributes to`→enable, `para que`→motivation, `llama`, mono-arête.

## Merge1 gold+silver quota140 + 3 seeds MESURES 2026-10-04
Merge rejouable `scripts/merge_goldsilver.py` (`--quota 140 --seed 42`,
`_none` non plafonné, `_methode` conservé, split stratifié 80/20) :
**gold livré 1764** (tombstones exclus 159) + **silver garder 4621**
(aveugle-quar exclus 67) → **train 1284 / val 321**.
Dataset figé hors git : `v4/merge1/{train,val,manifest}.json`
(SHA manifest train `04263584eebeed30`, val `14083013813e86ce`).
Runs config gelée (150 ep, lr 0.0005, edge-loss 3.0, GAT bidir dim 32) :
**s7 0.175 @ep12 · s42 0.292 @ep18 · s123 0.220 @ep9 · moy 0.229**,
train ~1.0 (surapprentissage massif, best 9-18/150).
Checkpoints/CSV hors git : `checkpoints/merge1_s{7,42,123}.npz`,
`v4/mesures/M1_s{7,42,123}.csv`, `MESURES.md` § merge1 archivé en place.
Lecture : moy 0.229 < gold-only 542 (0.278) et 357 (0.313) → silver
quenché dilue le signal gold. Piste : quota silver différencié
(`--silver-weight`) ou filtre confiance. Wrapper CLI :
`PYTHONPATH=gcn-python/src /tmp/opencode/gcntrain.sh` (pip 2.5.0 sans
`--min-class-count`, src 4.0.0).

## V4 finale gold-only quota64 + 3 seeds 2026-10-04 (v4 TERMINÉE sauf veto)
- **Topup lot11 intégré** : `topup_lot11_pending.json` → `lot25_topup11.json`
  (`check_v4` GATES OK, 10 opposition, 0 doublon id/texte). Reste 0 pending.
- **65 `_none`** (lots 01-05) exclus du split, fichiers intacts
  (`make_split.py --exclude-none`, décision documentée).
- **Split figé `v4/final/`** : `make_split.py --quota 64 --seed 42`
  (+ manifest SHA : train `be315eff47b0b96c`, val `497fca8f46607c0c`) →
  **train 572 / val 132**, uniforme exact 52/12 × 11 relations.
  `check_v4` GATES OK (52/classe ≥ N_min 30 ; 41 spans >15 = historique gelé).
- **Runs** : **s7 0.184 @ep119 · s42 0.265 @ep36 · s123 0.269 @ep128 ·
  moy 0.239 ±0.04**, train ~0.98-0.99. CSV `V4F_s{7,42,123}.csv`,
  checkpoints `v4final_s{7,42,123}.npz` (hors git), `MESURES.md` § v4 finale.
- Lecture : 0.239 vs merge1 0.229 (Δ=0.01 = bruit, Δ<0.05) ; cible prod
  0.40 toujours loin. Prochain levier : campagne control/prevent/condition
  (+30-50 phrases) puis quota relevé, ou features (piste données épuisée
  à volume constant).

## Veto humain en attente (lots 20-24, choix actuels conservés par défaut)
Répondre oui/non par point (fichiers `ARBITRAGE_lot{20..24}.md`) :
1. Doctrine `contrôle` nominal : garder les ~10 quarantaines (lots 23-24 :
   r2-01382/02149/02495/02366/02389, r2-01394/01579/02160/02124/02466)
   malgré gold s114/s115 qui met control sur mentions définitionnelles ?
2. Lot20 : `f0073` quar (option B garder fenêtre [7,14]), `s2564` litote
   (0-arête ?), `s'incline contre` opposition (option B quar si désaccord) ?
3. Lot22 revue 2 : confirmer les 5 sorties (r2-01077/01387/01418/01971/
   02221/02447) ?
4. Retypes : r2-00393/00909 filter→condition, r2-01039 control→prevent,
   s1591→cause, s1196 condition→cause, s2306 cause→concession,
   `ainsi`→explicit:false ×14, `s2310 pour commencer` en quar ?
5. Doublons : antériorité gardée (g0101 lot12 > r2-00275, s1133 lot10 >
   s1182) ?
