# Plan remédiation données v4 — v3 consolidée (2026-10-15)

## D0. Références d'autorité (toutes vérifiées)
- `plan-collecte-dataset-annotation.md` : T0→T5, quotas 5D, gate J7, gold-3000, F2 (motivation
  sens inverse) / F3 (sur-segmentation spaCy).
- `plan-annotation-cir.md` : chaîne unique brut→filtre→silver B (poids 0.0)→contrôle
  LLM échantillon ~1000/jalon→gold humain 60-100→3000→`validate_cir`+normalize→fusion/split/oversample.
  LLM = estimateur d'erreur + pré-remplissage relu, jamais gold direct. Table 11→19 préalable obligatoire.
- `guide-annotation-v0.md` : spans ±7 tok, `>15tok`→LITIGE, `confidence 1.0` auto bannie du gold,
  double annotation N=100 κ≥0.70, arbitrage tiers GARDER|QUARANTAINE+motif, tombstones jamais supprimés.
- `GATES.md` : Silver 0→0.3 seulement si borne haute IC95 ≤20 %, R8/R10 GEL, Socle0c GO n=85.
- `audit-gold.md` : lots v4 01-24 base saine, pools legacy à normaliser+échantillonner.
- `REGLE_EQUILIBRE_DATASET.md` : quotas 11 directes ±10 %, pas de duplication massive,
  labels arbitrés 01-04 intouchables (journal anti-réécriture).
- `table-11-19.md` + `DECISIONS.md` (D-TABLE, D-AGNOSTIC).

## D1. Diagnostic (prouvé, chiffré)
- Labels `verbe(sujet)` faux dans les lots : `hypothesi(above)` + `displaystyle` comme entité
  causale, `varient(long)` sujet faux, `majeure(revenue)` inversé, `décriver` ; 42 % labels hors-texte.
- Pollution : ~16-19 % EN + résidus LaTeX (3-4 phrases) en données FR.
- Relations douteuses : concession 50/52 et opposition 48/52 à l'envers (souvent entité→entité),
  condition 40/52 à l'envers ; filter/motivation à vérifier ; 14 arêtes sans marqueur.
- Sain : 0 doublon, 0 fuite train/val, spans valides, balance quota exacte, marqueurs OK (1 cas `?`).

## D2. Décisions tranchées
1. **EN → split séparé** : FR pur au train, EN en split à part (actif futur test universalité).
   Drop refusé (perte), tag seul refusé (pollution features).
2. **Arbitrage assisté** : assistant = census + propositions + `ARBITRAGE_lotNN.md` pré-remplis ;
   humain = validation oui/non/corriger par lot (~1-2 h). Arbitrage 100 % LLM refusé
   (auto-circularité, viole `plan-annotation-cir.md` : « pas de LLM, juge=humain »).
3. **Sous-quota → remplacement ciblé** avant split, jamais oversample compensatoire (`REGLE_EQUILIBRE`).
4. **Seuils** : junk certain (LaTeX, EN avéré) → quarantaine immédiate sans seuil ;
   bruit graduel → >5 % par lot (cohérent `audit_spans_tok.py` : rejet ≤5 %) ;
   seuils fins par catégorie fixés au rapport P0.

## D3. Principes intangibles
- Quarantaine avant correction ; tombstones jamais supprimés.
- Manifests sha256 à chaque étape ; `make_split --quota 64 --seed 42` rejoué à l'identique.
- Labels arbitrés 01-04 intouchables. Runs E non bloqués (v4 gelé, comparaisons inter-jeux interdites).

## D4. Phases (ordre strict, chacune gatée)
- **P0 — Détecteur** (`scripts/detect_bruit.py`, stdlib) : LaTeX / EN (`lang_detector.py`) /
  labels à sujet hors-span / coquilles / sens-par-relation / spans >15 / conf 1.0 → `bruit_*.json`.
  Gate : revue rapport, zéro suppression auto. (2-3 j, 0 humain)
- **P1+P3 — Quarantaine + langue** : junk + EN avérés → `quarantaine_nettoyage_*.json` +
  tombstones motif (`verdicts_aveugle.py`), exclus des merges (`merge_datasets.py`) ;
  EN → split à part ; LaTeX → `[FORMULE]` (`pdf_latex.py`).
  Gate : manifests sha, re-mesure quotas → remplacement si sous-quota. (2-4 j, ~0 humain)
- **P4 — Relations** : ~140 arêtes + filter/motivation + 14 sans marqueur ; préalables :
  convention de sens + F2 tranchés (guide §4), double annotation N=100 κ≥0,70.
  Gate : κ atteint avant le reste. (assistant ~3-5 j ; humain ~1-2 h)
- **P2 allégé — Labels** : régénération auto corrigée (sujet = tête `nsubj` dans le span,
  anti-troncation, coquilles) + diff + aveugle 200 (IC95 ≤20 %, sinon humain). (3 j + contrôle)
- **P5 — Rebuild + mesure** : merge → split → `check_v4.py` + `validate_cir.py` 100 % +
  `audit_spans_tok.py` ≤5 % → E5-coarse 3 seeds vs 0,390.
  **Critère d'arrêt : Δ < bruit ⇒ données innocentées.** (2 j)
- **P6 — CI** : `detect_bruit.py` dans `make gates`, rejet conf 1.0 auto, >5 % ⇒ quarantaine. (1 j)

Total : assistant ~10-15 j, humain ~1-2 h.
**Gold-3000 et MLM-large APRÈS P5 GO** (ne pas industrialiser le bruit ; ordre T2→T4).

## D5. Pourquoi cet ordre
Le gradient (CE) supervise types + relations, jamais les chaînes de labels → l'heure humaine va
aux relations (P4), pas aux labels (P2 allégé). Le junk/EN (P1+P3) est mécanique et assainit
toutes les mesures suivantes. P5 avec critère d'arrêt chiffré évite d'incriminer les données
indéfiniment.
