# Reprise demain — état au 2026-10-03 ~18h35

## Objectif
100k silver : 50k lignes code + 50k textes (science 25k, presse 10k, éduc 10k, divers 5k).
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
