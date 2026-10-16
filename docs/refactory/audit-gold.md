# Audit gold existant vs schéma cible (phase 0, P0-5 / D2)

> Méthode : `scripts/audit_gold_schema.py gcn-datasets/DATA/supervision --max-per-file 500`
> (327 fichiers, rejouable). Verdicts : `v4` (types ⊆ 8, relations ⊆ 19, sans alias v2),
> `v2-legacy/mixte` (aliases v2 ou hors 19), `sans-cir` (phrases brutes).
> Limites : cap 500/fichier (totaux réels supérieurs) ; clé `confidence` seule inspectée
> (lots v4 : conf portée par les ARBITRAGE_*.md — à recoller).

## Constats (vérifiés 2026-10-08)

1. **`cir.intent` = 0 occurrence sur 327 fichiers.** La cible T4 (`cir.intent`, 3 actes) est
   100 % greenfield — aucun gold existant ne la couvre.
2. **Lots v4 arbitrés (01-24 + rares/divers/candidats, `v4/annotated/`, ~2646 CIR capped) : verdict
   `v4`.** Types observés 5/8, relations jusqu'à 11/19 par lot. Base saine pour gold_mini 60-100
   (sélection à faire : couverture relations × langues).
3. **Pools legacy `v2-legacy/mixte`** : `annotees_v1/*`, `silver*/`, `combined_*`, `entrainement_v1`,
   `train_v2`, `mixte_final/*`, `fusion_*` (types=7 avec aliases v2 — ex. `etat_systemique` vérifié
   dans `annotees_v1/fr/lot_001.jsonl` —, relations 11, méthodes `silver-auto-v2`,
   `silver-allowlist-moi-types`, `silver-{code,en,fr}-moi-types`). Réutilisables APRÈS
   `normalize.py` (v2→v4) + revalidation échantillon.
4. **Splits train/val/test existants** : `mixte_final`, `train_v2`, `v4/final`, `v4/final2`,
   `entrainement_v1`, `eval_v2/*`, `smoke` — matière pour held-out (à re-geler HORS docs, seed 42).
5. **Pool à annoter** : `g0/*` (1500 phrases sans CIR) + `bench_*` (100-phrases, sans CIR).
6. **Quarantaines et tombstones** : `quarantaine_*` présents dans presque chaque lot (pratique
   conforme au plan) ; `struct/*` = brut scrapé (sans CIR, matière C8).

## Verdict socle 0c

- gold_mini 60-100 : EXTRAIRE de `v4/annotated` (lots 08-24, rels 7-11) — pas de re-annotation.
- held-out : RE-GELER depuis `v4/final*` + `mixte_final` (hors docs, seed 42).
- Reste (legacy) : normaliser puis échantillonner — ne PAS supposer v4 partout.
- Manquant pour R8 (30/classe) : matrice de COMPTES par classe (l'audit donne distincts, pas
  effectifs) — relancer sans cap + comptage avant toute décision N≥30.
