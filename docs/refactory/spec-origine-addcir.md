# Spec origine dialogue pour `add_cir` (phase 0, P0-5 / backend 9a)

> Constat vérifié : `CausalGraph.add_cir` (`verbalizer/instructions.py:96-124`) fusionne nœuds/arêtes
> SANS origine ni poids (tuples `(src, dst, attrs, src_text)`). Backend 9a exige origine + poids
> 1.0/0.8/0.5 à l'écriture. Composition avec 7b : voir DECISIONS.md D-ORIG (proposition : champs séparés).

## Champ

- `session_origin ∈ {document, affirmation_utilisateur, question_utilisateur}` — porté par nœud ET
  arête (l'arête hérite par défaut de l'origine de son écriture, pas de ses extrémités).
- `session_weight` dérivé : document 1.0 / affirmation 0.8 / question 0.5 (figé, cf. 9a).
- Stockage : clé d'attrs (ex. `attrs["session_origin"]`), jamais en label/span (données intactes).

## API

- `add_cir(cir, session_origin=...)` — origine **requise** (pas de défaut silencieux) pour toute
  écriture post-9a ; `add_discourse_block` propage l'origine du bloc.
- Migration : CIR existants (corpus, gold, silver) → `document` avec flag `migrated:true`
  (distinguer l'attesté du supposé).

## Règles (9a, à câbler)

- Question ne pose jamais un fait : arêtes `question_utilisateur` exclues des preuves des réponses
  suivantes (mention « d'après votre question » si seule source).
- Clarification : 1 en attente max, abandonnée après 3 tours avec annonce (compteur en session).
- Révocation : n'efface que `affirmation_utilisateur|question_utilisateur`, jamais `document`/corpus.

## Tests d'acceptation (repris du plan, à automatiser)

- Chemin doc 0.6 > chemin question 0.9×0.5=0.45 ; mention d'origine en réponse ; révocation
  n'efface que l'apport utilisateur ; jamais 2 clarifications en attente.
