# Spec moteur B — socle spaCy + fallback (phase 0, P0-3)

> Référence : [T1] `plan-remodelage-tout-appris.md`. D-SPACY tranchée et appliquée
> le 2026-10-08 (import lazy optionnel, `layer0/interface.py` amendée).
> Implémenté (socle 0a, 2026-10-08) : `layer0/engine_b.py` + `fallback_regex.py`
> + `tests/test_engine_b.py` (11 tests verts). Le silver actuel reste
> `silver-auto-v2` (règles connecteurs) — B-spaCy est le moteur du socle.

## Rôle

B fournit la segmentation/smarts linguistiques au silver d'annotation ET à NLU (inference step 6)
sans réintroduire ni lattice ni UD humain au runtime du moteur appris.

## Interface exigée

- Entrée : `text: str` (+ `lang` budgétée fr/en/es/de/pt/it).
- Sorties : `(clauses, connecteurs, source)` où `source ∈ {silver-B, fallback}` — étiquette
  obligatoire portée jusqu'au CIR (`source=silver-B|fallback`, loss masquée en fallback).
- Contraintes : `spacy==3.8 + fr_core_news_sm` pin ; import **lazy** + **timeout** 30s — FIGÉ 2026-10-08 (même ordre que le pont actuel `bridge.py:96-98`, révisable après spike PDF) + **fallback regex**
  si absent/lent/HS ; **test sans-spacy vert** (suite dédiée).

## Fallback regex (à spécifier, règles minimales proposées)

1. Segmentation phrases (ponctuation forte + familles `?/؟/？`, `¿…?`, `!`).
2. Détection connecteurs via les tables existantes (`causal_scorer.py: RELATION_KEYWORDS` fr/en +
   `balanced_auto.py` es/de) — pas de nouveau lexique sans audit.
3. `source=fallback` systématique + compteur fallback exposé (seuil proposé 5 %, à valider phase 1 — `FALLBACK_ALERT_RATE`).

## Traçabilité et tests d'acceptation

- Chaque sortie B porte `{outil, version, source}` (пин spacy, version modèle).
- Acceptation : suite sans-spacy verte ; fallback déclenché artificiellement → `source=fallback`
  + loss masquée (test) ; pin reproductible (lockfile).
- Non-objectifs : parsing fin, NER, autre langue que le budget D2.

## Traçabilité T1 (relecture, cocher)

- [x] pin spacy+modèle · [x] lazy+timeout · [x] fallback regex · [x] source silver-B|fallback ·
  [ ] loss masquée fallback (côté train, R9 — pas B) · [x] test sans-spacy · [x] D-SPACY tranchée.
