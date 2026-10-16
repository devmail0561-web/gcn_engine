# Protocole validation machine — v1 (2026-10-15, délégation du juge humain)

> Contexte : le juge humain délègue la validation (~550 cas) en exigeant des mesures
> strictes anti-fausse-validation. Ce protocole est la mesure. Tout écart = STOP + sortie écrite.

## PV-1 Asymétrie (principe cardinal)
- **« oui » exige ≥3 signaux positifs INDÉPENDANTS et ZÉRO drapeau rouge.**
- Tout le reste → « non » (exclu, récupérable) ou « litig » (réservé humain).
- Un faux positif empoisonne le gold ; un faux négatif ne perd qu'une phrase.
  En cas de doute : **non**.

## PV-2 Signaux (auto-calculés, `scripts/signaux_validation.py`)
- **S1 Proposeur** : relation build_lot (`balanced_auto`) == classe visée.
  Faible (même famille que le minage) : ne compte jamais seul, exige S2 ou S3 à côté.
- **S2 Lexique marqueur** : marqueur ∈ lexique FR de la relation (listes §PV-5, issues
  `table-11-19.md` + `RELATION_KEYWORDS`). Marqueur ambigu multi-lexiques → signal ABSENT.
- **S3 Direction** : span source avant span cible (convention P4 validée humain).
- **S4 Types** : compatibilité source/cible par relation (§PV-6).
- Règle auto-oui : **S1 ∧ (S2 ∨ S3) ∧ S4 ∧ ¬Red**. Tout le reste → revue manuelle.

## PV-3 Drapeaux rouges (¬Red si aucun)
- R1 : span >15 tok → FLAG INFORMATIF (pas bloquant) : précédent lots 01-07
  (177 gt15 dans le train final) — la dilution de pooling n'empoisonne pas le label ;
  noté en fiche, re-découpe réservée au gold humain (guide-annotation-v0).
- R2 : ≥2 connecteurs concurrents détectés (silver-B) → manuel.
- R3 : marqueur ambigu (lexiques `pour`, `sous`, `protège`, `bloque`, `contre`) → manuel.
- R4 : EN/LaTeX/code-bucket → quarantaine (jamais oui).
- R5 : négation (ne/n'/pas/jamais/plus/guère/aucun ±4 tok du marqueur) → manuel,
  défaut non (sens nié : pw-1279, pw-771, pw-28).
- R6 : span dégénéré (≤2 tok sur phrase >12 tok) → manuel, défaut non (pw-659, pw-210).
- R7 (leçon test aveugle 2026-10-15, accord 74 %) : l'alignement sémantique
  marqueur-spans n'est PAS auto-vérifiable → re-jugement aveugle 10 % obligatoire ;
  accord <95 % ⇒ revue manuelle intégrale des auto-oui (appliqué : 191 cas).
- R5 : cas que j'ai moi-même rédigé (cdb2, P1b) → S1 inapplicable, revue manuelle
  systématique + exigence S2∧S3∧S4.

## PV-4 Revue manuelle (moi, traçée)
Chaque cas manuel reçoit : test de paraphrase (puis-je énoncer le fait causal en une
phrase ?) + contre-lecture explicite (quelle lecture alternative invaliderait le oui ?)
+ verdict {oui|non|litig} + règle citée. Enregistré en `*_verdicts.json`.
Test aveugle : 10 % des auto-oui rejugés sans voir le verdict auto (fichier séparé) ;
**accord <95 % → revue manuelle intégrale** (STOP auto-oui).

## PV-5 Lexiques marqueurs FR (v1, à étendre uniquement sur preuve lots)
- prevent : empêcher/éviter/stopper/bloquer/interdire/prévenir/neuraliser/écarter/détourner/couper/retient
- filter : exception/sauf/hormis/excepté/seuls/seul/seule/uniquement/seulement/hors/part(à part)
- control_dependency : déclencher/sous contrôle/sous la direction/suspendre/ordonner/limiter/restreindre/censurer/confisquer/expulser/interrompre/rejeter
- data_dependency : dépend/repose sur/découle/résulte/se traduit par
- cause : causer/provoquer/engendrer/entraîner/grâce à (NON : médié → voir ternaires)
- enable : permettre/faciliter/rendre possible/donner lieu
- condition : si/lorsque/quand/à condition/sans (NON : sans+aurait → counterfactual)
- concession : bien que/quoique/toutefois/cependant/malgré/n’en/-même si
- opposition : mais/alors que/tandis que/contre/en revanche/par contre
- sequence : ensuite/puis/après/dès/lors
- motivation : pour/afin de/car/parce que (NON : pour+éviter/empêcher → prevent)
- Ambiguïtés actées : pour, sous, protège, bloque, contre → R3 systématique.

## PV-6 Compatibilité types (source → cible) — v2 corrigée sur lots (2026-10-15)
Constat : la segmentation auto met souvent la clause entière en source (processus)
et l'effet évité en entite/etat_global. Table alignée sur l'observé (lots 01-24) :
- prevent : source (entite|processus|contrainte|etat_global) → cible (evenement|processus|etat_local|etat_global|entite) effet évité
- filter : source (entite|concept|processus|evenement) → exclu (entite|concept|evenement|processus|etat_local)
- control_dependency : contrôleur-autorité (entite) → contrôlé (tout)
- data_dependency : dépendant (tout) → déterminant (evenement|concept|processus|etat_local|etat_global)
- cause : cause → effet (evenement|processus|etat_local|etat_global|entite) ; enable : tout → tout
- condition : tout → tout ; concession/opposition : fait → fait
  (processus|evenement|etat_*|condition, JAMAIS entite→entite : règle P4)
- sequence : (processus|evenement) → (processus|evenement)
- motivation : tout → (processus|evenement)

## PV-7 Traçabilité et gates
- Chaque verdict : `{id, verdict, signaux:[S1..S4], contre_lecture, regle, validateur:"machine-PV1"}`.
- Promotion : unanimité oui uniquement (`promouvoir_p1b.py` refuse les manquants ;
  étendre aux verdicts PV1). Refus par défaut au moindre doute (PV-1).
- L'humain garde : spot-check final ~30 cas adversariaux (30 min) + appel suspensif
  sur tout batch (tombstones réversibles, disque intact).

## PV-8 Leçons d'application 2026-10-15 (ne pas régresser)
- L1 (gate aveugle) : accord 74 % ⇒ revue intégrale 182 auto-oui + R5/R6
  (négation, spans dégénérés). Taux oui final minage 75 %, cdb2 32 %, ternaire 100 %.
- L2 (métrologie) : la métrique val est hypersensible à sa composition —
  **ablations à val gelée**, jamais de comparaison inter-vals.
- L3 (spans) : l'hypothèse dilution a été testée et réfutée (retirer les longs spans
  dégrade) — R1-info-only maintenu empiriquement, pas seulement par précédent.
- L4 (direction) : 4 backs résiduels post-merge (2 inversés, 2 exclus) — le détecteur
  backs tourne APRÈS chaque promotion, avant tout split (gate ajouté au rituel P5).
- L6 (métrologie val, 2026-10-15) : ~20 expériences sur val P5 gelée ont méta-surajusté
  (0.548 gelée vs 0.367 native, même train — écart 0.18). RÈGLE : toute conclusion
  exige double mesure (gelée + native/fraîche) ; écart >0.05 ⇒ sur-ajustement, pas gain.
  Les Δ P5→P8 antérieurs sont contaminés d'ampleur inconnue.
- L5 (spot-check 2026-10-15) : 29/29 uniques MAINTENUS, 0 renversé (fiche
  `SPOTCHECK_PV1_30.md`, doublon g0-1883 #18=#28 signalé, cas inversés vérifiés en lot).
  Règle ≥28/30 : VALIDATION PV1 CONFIRMÉE. Verdicts `/tmp/opencode/verdicts_spotcheck.json`.
