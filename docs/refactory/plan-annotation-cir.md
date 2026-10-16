# Plan annotation — phrases brutes → CIR d'entraînement (volet manquant)

> Note 2026-10-11 : pilotage opérationnel unifié dans
> `plan-collecte-dataset-annotation.md` (T3 : table v1 SQUELETTE, silver-B, contrôle
> LLM, gold-3000, kappa ≥0,70). Ce volet reste la référence de fond ; l'ordre
> d'exécution (tranche verticale gold-219 d'abord) est dans le plan unifié.

> Sans ce volet, le plan dataset produit un tas de texte inutilisable pour le train.
> But : transformer les phrases collectées en CIR supervisés (silver massif + gold
> de contrôle), avec traçabilité complète. Périmètre : outillage d'annotation +
> volumes + contrôle qualité. Le train lui-même reste dans le volet training.

## 1. Chaîne d'annotation (une seule, dans cet ordre)

```text
phrases brutes (dataset 3M)
  → filtre causal (connecteur exigé, sinon quarantaine — dataset §4)
  → SILVER automatique : moteur B (spaCy, T1) — weight 0.0 par défaut (R9)
  → CONTRÔLE : annotateur LLM (gcn-annotate) sur échantillon stratifié
  → GOLD humain : socle 60-100 (socle refacto) + extension multi-domaines
  → validation (validate_cir) + normalisation (8 types / 19 relations)
  → fusion + split stratifié + oversample des rares → prêt gcn-train
```

## 2. Silver : le moteur B annote, le train se méfie

Situation : des millions de phrases ne seront jamais annotées à la main. Le silver
automatique est obligatoire, mais le corpus actuel prouve qu'un silver aveugle
produit un dataset pourri (CIR jetés du legacy).

Règles : le silver vient du moteur B (spaCy + règles, T1) — jamais de l'ancien
moteur UD ; poids 0.0 par défaut dans la loss (le silver n'apprend rien tant que
sa qualité n'est pas prouvée, R9) ; montée à 0.3 seulement si l'erreur mesurée
sur le gold est ≤ 20 %. Chaque annotation silver porte sa source (`source=silver-B`)
et la loss est masquée quand B est en fallback (T1).

Cas particulier du code : les commentaires/docstrings s'annotent comme la prose.
Le code source lui-même ne passe PAS par le silver ML — il relève du frontend
symbolique existant (tree-sitter), pas du train texte→CIR.

Cas particulier maths/physique : extraction LaTeX (latexpand) avant annotation ;
les formules sont gardées comme blocs opaques (jamais découpées en tokens),
seul le texte porteur de causalité autour est annoté.

## 3. Contrôle LLM : échantillon, pas masse

L'annotateur LLM (gcn-annotate) ne sert pas à annoter en masse (coût, dérive).
Il sert à : (a) estimer l'erreur du silver sur échantillon stratifié
(langue × relation × domaine, ~1000 phrases par jalon) — c'est cette mesure qui
autorise ou non le poids 0.3 ; (b) assister l'annotation gold (pré-remplissage
relu par humain, jamais gold direct).

## 4. Gold : la cible chiffrée

Le socle refacto exige 60-100 phrases gold. Les règles R8/R10 exigent 30 exemples
minimum par classe et par split pour démasquer et calibrer : avec 19 relations,
8 types de nœuds et 3 actes, la cible gold est d'environ 3000 phrases
(langue × relation × domaine couverts, aucun zéro). En deçà, les classes rares
restent masquées et la gate gold-3000 ne peut pas passer.

## 5. Table de correspondance 11 → 19 (à produire dans ce volet)

Le compteur causal historique connaît 11 relations, la cible en a 19. Avant toute
campagne d'annotation : écrire la table explicite (chaque indice causal → relation
cible, avec les cas ambigus listés et leur règle de départage). Sans cette table,
le silver et les humains n'annotent pas le même schéma.

## 6. Contrôle qualité (gates, pas d'œil humain exhaustif)

- `validate_cir` sur 100 % : schéma, spans, relations dans le vocabulaire, taux
  de rejet publié par jalon.
- Normalisation 8 types / 19 relations (normalize) : déterministe, testée.
- Distribution causale : les fourchettes du dataset §5 vérifiées sur le silver
  comme sur le gold — un silver à 90 % de « cause » est rejeté, pas entraîné.
- Traçabilité : chaque CIR porte {source, phrase-hash, campagne, outil+version} ;
  errata et retraits possibles sans reconstruire (tombstones).

## 7. Ordre et effort

Outillage (~2j) : table 11→19 (0.5j) + branchement silver-B dans le pipeline
(0.5j) + protocole contrôle LLM (0.5j) + campagne gold 3000 (au fil de l'eau,
hors chemin critique : le socle 60-100 suffit pour démarrer). Puis campagnes
d'annotation au rythme de la collecte, avec gate qualité à chaque jalon avant
fusion — jamais après.
