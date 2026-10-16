# Bascule intents 21 → 3 actes (phase 0, P0-4)

> Constat vérifié : `nlu_route` (`nlu_routing.py:45-67`) mappe chaque question vers le DSL Pearl via
> 21 intents (tête apprise prio 1, heuristique surface prio 2) ; `INTENT_TYPES` (21, `constants.py:158-173`)
> est plombé partout (train, checkpoint, eval, engine). `SENTENCE_TYPES` = 4 (`constants.py:179-184`).
> Cible T4 : 3 actes (assert→stocker, question→répondre, request→ack+file-action).
> Point ouvert : le sens de « P1 22 » (`inference` §4[D]) — 22 intents survivants ? À clarifier, pas une preuve.

## Inventaire des sites (devenir à assigner)

| Site | Rôle actuel | Devenir proposé |
|---|---|---|
| `constants.py:158` `INTENT_TYPES` (21) | vocab tête Éq.6 | GARDER comme vocab P1-questions SI option couches, sinon geler |
| `nlu_routing.py:26-42` sets ONE/TWO + `_build_dsl` | question → DSL Pearl | GARDER (cœur non régressable) ; seuils à revoir |
| `nlu_routing.py:45-67` `nlu_route` | prio tête > surface | GARDER ; brancher `cir.intent` en prio 0 (T4) |
| `cgnp.py:75,177,468-471,922,987` tête Éq.6 duck-typée | logits intent optionnels | ADAPTER : produire `cir.intent` (3 actes) ; 21 classes → sous-tête ou suppression (selon option) |
| `layer2/reference.py:72-148` tête MLP | `d_clause → n_intent_types` | REMPLACER par tête act (1,3) + `route_intent` |
| `train.py:15,298,383,514-516,611,634,652,773,916-922,1376-1442` | gold/comptes/masques/métriques intent | ADAPTER (gold `sentence.intent` → actes ; masques/logits) |
| `training/checkpoint.py:186-195` | persiste `n_intent_types` + vocab | VERSIONNER (compat anciens checkpoints : refuser ou convertir, cf R4) |
| `engine.py:165-283`, `eval_runner.py:117-216` | reconstruit `n_intent_types` | ADAPTER |
| `SENTENCE_TYPES` (4) → 3 actes | déclarative→assert, interrogative→question, impérative→request, exclamative→assert saillant (FIGÉ 2026-10-08) |

## Options

- (a) Remplacement pur : 3 actes tuent les 21 (risque : perte requêtes Pearl fines — ex. SPOF/ANALOGY/DOUBTS restent-ils interrogeables ?).
- (b) **Couches (RETENU 2026-10-08, implémenté tête act)** : 3 actes routent au niveau session ; `nlu_route`/21 survivent SOUS `question` ; `cir.intent` porte l'acte, le DSL Pearl reste l'outil de réponse.
- (c) Strangler par type : migrer intent par intent avec double-run comparé (coûteux, le plus sûr).

## Vérification P0-4

- [ ] 0 site sans devenir · [ ] mapping 4 sentence-types → 3 actes figé · [ ] sorts des requêtes Pearl fines écrit (quelles survivent, via quel chemin) · [ ] tests à garder listés (ne pas casser les 685 sans tri).
