# Table de correspondance 11 indices → 19 relations — v1 validée (phase 0, P0-5)

> Statut : **v1 VALIDÉE 2026-10-11** (arbitrage humain : 5 GARDÉS patchés, 1 GELÉ,
> 1 deux-tiers, 1 REFORMULÉ — minage 601k textes / ~146k uniques / 433 fichiers,
> preuve `/tmp/opencode/minage_triggers.json`, volontairement non versionnée).
> Consommateurs débloqués : fusion C8/J7, silver-B, guide §4.
> Trouvaille corpus : **76 % de doublons exacts** (455k/601k) → dédup C8 confirmée
> critique (MD5 + Jaccard, pas seulement exact).
> Principe acté 2026-10-08 : le moteur ne charge jamais de données configurées —
> cette table est un DOC pour humains + scripts data ; le runtime n'utilise que les
> constantes code (`normalize.py`, `edge.rs`, `constants.py`).
> Précision 2026-10-09 (anti-confusion) : le « 100 % conformes schéma 8/19 » des gates
> = conformité de normalisation (ce fichier + `normalize.py`, mesurée par script),
> PAS le contenu sémantique des déclencheurs (ci-dessous, v0, interdit prod).
> Sources :
> - 11 indices : `RELATION_KEYWORDS` de `gcn-scraper/.../filters/causal_scorer.py`
>   (= les « 11 directes » documentées `gcn-annotate/README.md:139`).
> - Aliases : `RELATION_TYPE_ALIASES` de `gcn_annotate/normalize.py:59-73`.
> - Cibles : RELATION_TYPES v4 (`normalize.py:14-21`) = `edge.rs:11-33` = `constants.py:24-32`.
> - Ternaires/jointes : `TERNARY_RELATIONS` + `JOINT_RELATIONS` (`normalize.py:23-27`),
>   `TernaryThird` (gcn-ir), `joint_group_id` (`pearl.rs:20-23`).

## Correspondances directes (11)

| Indice | Aliases connus | Cible |
|---|---|---|
| cause | Cause | cause |
| enable | Enable, enables | enable |
| prevent | Prevent, prevents | prevent |
| condition | Condition, conditional | condition |
| concession | Concession, concedes | concession |
| sequence | Sequence, sequences | sequence |
| motivation | Motivation, motivates | motivation |
| filter | Filter, filters | filter |
| opposition | Opposition, opposes | opposition |
| data_dependency | data_dep | data_dependency |
| control_dependency | control_dep | control_dependency |

## 8 nouvelles relations — déclencheurs à définir

Pour chacune : écrire le déclencheur observable (marqueur ? tiers ? sources multiples ?) +
exemples or tirés des lots arbitrés. Pistes code entre parenthèses, à valider humain.

> Propositions machine 2026-10-11 (minées sur `combined_v2/train.json`, 1665 phrases) —
> **[VALIDÉ 2026-10-11, arbitrage humain sur minage large 146k uniques]**,
> patchs ci-dessous. Exemples tronqués à 200 car.

- `conditional_cause` — ternaire (tiers rôle=condition) [GARDÉ patché] : `si + proposition causale`
  (conséquence sous condition, ex. « Si l'inoculation a moins de complications [...],
  elle suscite des doutes et entraîne des risques »). Distinguer de `condition` pure
  (condition sans effet asserté) : le tiers porte sur le LIEN, pas sur un nœud.
  Patchs : prose seule (**hors bucket code** : `if err == ... because` exclu),
  effet asserté exigé, exclus `se produire` réflexif, sigle `(SI)`, méta `est une cause`.
- `mediated_cause` — ternaire (tiers rôle=mediator) [GARDÉ patché] : `grâce à / via / au moyen de /
  par l'intermédiaire de + GN moyen` (ex. « Grâce à ces alliances nous pourrions commencer
  à appréhender le monde... », « ...diminuer la pénibilité [...] : utilisation de scans »).
  Le tiers = le moyen/instrument, jamais la cause elle-même.
  Patchs : **frontières de mots obligatoires** (`viaux/viande/Bolivian` = bug sans `\b`),
  exclus `through which/whom` (relatif) et `through` adverbial (`pass through`).
- `joint_cause` — jointe (sources multiples, joint_group_id) [GARDÉ patché] : `combinaison X + Y /
  à la fois / conjoint + effet unique` (ex. « la combinaison précession + obliquité peut
  accroître les précipitations »). Exige ≥ 2 sources nommées convergentes, pas une
  énumération décorative.
  Patchs : effet unique asserté exigé (`à la fois des produits` sans effet = rejeté),
  exclus `recombinaison/recombination` (1 seul processus).
- `conditional_prevent` — ternaire (tiers rôle=condition) [GARDÉ patché] : même patron que
  `conditional_cause` avec effet d'empêchement/blocage (meilleure précision ~4/5 FR :
  « Si ce jour est choisi, c'est pour [...] éviter », « If damaged, a brake stops »).
  Patchs : hors code (`while ... stops`), hors sigle `(SI)`.
- `mediated_prevent` — ternaire (tiers rôle=mediator) [GARDÉ patché] : même patron que
  `mediated_cause` avec effet d'empêchement (ex. « ...incendiée afin de réduire la production
  et d'empêcher les prix d'atteindre... » — ici le moyen = l'incendie volontaire).
  Patchs : mêmes frontières (`viaires/viations` exclus) ; bons spécimens
  (« grâce à la photosynthèse [...] sans », « through photosynthesis without »).
- `joint_prevent` — jointe (sources multiples, joint_group_id) [GELÉ 2026-10-11] : n=15,
  précision ~1-2/5 (`un à la fois, sans séparation` = procédure, pas prévention).
  Seul bon spécimen : « Both plants caused reductions [...] without ».
  Réactivation si campagne rattrapage ≥30 occurrences, sinon gel R8 maintenu.
- `analogy` — autre (comparaison, pas de causalité directe — cf. poids ANALOGY backend) [GARDÉ deux-tiers] :
  **fort** `de même que / tout comme / just as` (ex. « le cancer a des veines étendues de tous
  côtés, de même que le crabe a des p... ») — exclus `just as soon/long` (temporel) ;
  **faible (LITIGE)** `like a/an` (ex. code pédago `like an assignment`) : au cas par cas.
  **Exclure** `tel que` énumératif (introduit des
  exemples, pas une comparaison : « risques tel que la pollution du sol... » ≠ analogy ;
  piège mesuré : 1383 occurrences).
- `counterfactual` — autre (conditionnel contrefactuel) [REFORMULÉ 2026-10-11] :
  patrons stricts `if + PQP + would have` / `sans + aurait` (événement non advenu).
  Patrons larges rejetés : `had ... would` seul (= rapporté : « had determined he would face »),
  `could have` seul (= spéculation). FR quasi-vide (23 hits ≈ EN mal classés) : campagne
  rattrapage FR requise.
  **Exclure le conditionnel journalistique** (`aurait été donné par Hippocrate`,
  `auraient porté leurs conquêtes` = information incertaine, pas monde alternatif —
  piège mesuré : 360 occurrences >> 23 hits FR).
> Statut global inchangé : SQUELETTE, interdit prod.

- `conditional_cause` — ternaire (tiers rôle=condition) [PROPOSÉ] : `si + proposition causale`
  (conséquence sous condition, ex. « Si l'inoculation a moins de complications [...],
  elle suscite des doutes et entraîne des risques »). Distinguer de `condition` pure
  (condition sans effet asserté) : le tiers porte sur le LIEN, pas sur un nœud.
- `mediated_cause` — ternaire (tiers rôle=mediator) [PROPOSÉ] : `grâce à / via / au moyen de /
  par l'intermédiaire de + GN moyen` (ex. « Grâce à ces alliances nous pourrions commencer
  à appréhender le monde... », « ...diminuer la pénibilité [...] : utilisation de scans »).
  Le tiers = le moyen/instrument, jamais la cause elle-même.
- `joint_cause` — jointe (sources multiples, joint_group_id) [PROPOSÉ] : `combinaison X + Y /
  à la fois / conjoint + effet unique` (ex. « la combinaison précession + obliquité peut
  accroître les précipitations »). Exige ≥ 2 sources nommées convergentes, pas une
  énumération décorative.
- `conditional_prevent` — ternaire (tiers rôle=condition) [PROPOSÉ] : même patron que
  `conditional_cause` avec effet d'empêchement/blocage (2 hits seulement sur 1665 phrases —
  à miner sur corpus large avant de figer).
- `mediated_prevent` — ternaire (tiers rôle=mediator) [PROPOSÉ] : même patron que
  `mediated_cause` avec effet d'empêchement (ex. « ...incendiée afin de réduire la production
  et d'empêcher les prix d'atteindre... » — ici le moyen = l'incendie volontaire).
- `joint_prevent` — jointe (sources multiples, joint_group_id) [PROPOSÉ] : même patron que
  `joint_cause` avec effet d'empêchement. Rare attendu — ne pas forcer, gel R8 si 0 occurrence.
- `analogy` — autre (comparaison, pas de causalité directe — cf. poids ANALOGY backend) [PROPOSÉ] :
  `de même que / tout comme / comme + comparé` (ex. « le cancer a des veines étendues de tous
  côtés, de même que le crabe a des p... »). **Exclure** `tel que` énumératif (introduit des
  exemples, pas une comparaison : « risques tel que la pollution du sol... » ≠ analogy).
- `counterfactual` — autre (conditionnel contrefactuel — marqueurs à inventorier) [PROPOSÉ] :
  `si + PQP / conditionnel passé d'événement non advenu` (ex. « Sans X, Y aurait... »).
  **Exclure le conditionnel journalistique** (`aurait été donné par Hippocrate`,
  `auraient porté leurs conquêtes` = information incertaine, pas monde alternatif —
  4/5 hits `aurait` du minage sont journalistiques, pas contrefactuels).

## Cas ambigus — à constituer depuis les LITIGE des lots

Chaque entrée : exemple + règle (ex. condition vs filter, cause vs enable, concession vs
opposition). Liste vide en v0 — remplir en P0-5 avant campagne gold.

## Consommateurs

- dataset C8/J7 (hints 11 → 19) · silver-B (mapping d'annotation) · gold humain (guide §4).
