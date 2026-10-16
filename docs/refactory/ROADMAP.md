# Roadmap intégrée — remodelage moteur (phase 0, P0-7)

> S0 = gate P0 franchie. Durées : jours-homme sauf collecte (calendaires). Owners : GATES.md.
> Estimations refacto : 11-14j hors BCE/S4-opt (vérifié) ; backend P0 2j ; dataset 50-95j à 3 machines.

## Chemin critique

```
P0 décisions+specs (2j) ─┬─▶ slice -1 (1j) ─▶ socle 0 (3j : B+s0 ‖ Rust PR ‖ gold_mini+held-out + table v1)
                         │      │                    │
                         │      └─ STOP si NO-GO ─────┘
                         └─▶ spikes PDF/densité (0.5+0.5j, calendaires, GO/NO-GO seuils prédéfinis)
socle ─▶ L0 (1j) ─▶ L1 (3-5j) ─▶ tête act+route (1j) ─▶ convertisseur (0) ─▶ seuils (1-2j) ─▶ NLU+B (1j) ─▶ [BCE opt]
socle ─▶ backend P0 (2j, parallélisable dès S0) ─▶ Ph8 contrat UserResponse (dû AVANT tête act)
slice GO ─▶ Ph5 calibration ─▶ GO: 6a+7c │ NO-GO: 6b ─▶ Ph7 (7a→7b→7f→7d→7e→7c) ─▶ Ph9+10 (M4,9a-d,M1,M2a,9e+M2b/c,M3)
C0 registry ─▶ C1-doc/code + C2-maths… (fond, 50-95j) ─▶ J7 fusion (EXIGE table 11→19) ─▶ J8 ─▶ campagnes annotation ─▶ gold 3000
```

## Dépendances verrouillées (issues inventaire)

table→J7 · Ph8→tête act (socle 0b) · slice→Ph5 · socle 0c→L0/L1 · B-spec→silver/NLU · bascule→tête act ·
guide+kappa→campagne gold · spikes→C1/C2 · held-out→seuils/R10.

## Bloquants moteur → SA (2026-10-11, miroir de `ETUDE_SA.md §29`)

- **Ph5 NO-GO** (ECE 0.33-0.55, `GATES.md`) → conf moteur non calibrées ; le SA les traite en `provisoire` + `EN RÉVISION` par défaut.
- **Checkpoint non re-mesuré** (0.468 vs 0.1867, harnais distincts, `CHANGELOG Lot A`) → aucun checkpoint consommé par le SA avant re-mesure sur harnais unique.
- **Dataset à venir** (`plan-dataset-3M.md`, `plan-annotation-cir.md`) : 3M visés, ~146k (5 %), 50-95j ; gold 219, 8 relations à 0, gel R8/R10 → calibration SA et levée des gels en attente du gold-3000.
- **Benchmarks à construire** (MVP §12.5 ETUDE_SA : 1000 docs, 200 claims, 50 chaînes, 24h) → gate d'entrée P3.
- **Non bloqués** : scaffolding SA, WAL, boucle, API (ordre §21 ETUDE_SA inchangé).

## Baselines — snapshot À EXÉCUTER avant slice (I2)

```bash
# Latence + taille (local, dater le snapshot)
cargo test -p gcn-backend 2>&1 | tail -3            # temps suite backend (référence P0)
du -sh gcn-datasets/DATA/supervision gcn-python/models gcn-datasets/checkpoints
# Références déjà documentées (à re-mesurer) :
# val_edge_f1=0.468 (v2.4.0, PROGRESS.md) · ECE via evaluation/calibration.py sur gold figé
python3 --version && cargo --version && nvidia-smi -L  # outillage dispo (GPU S4-opt ?)
```

## CI — gardes à ajouter (I1, workflow `.github/workflows/ci.yml` : 6 jobs existants)

`grep -rE "subprocess|gcn_bin|Lattice" gcn-python/src == 0` (acceptation A) · `torch` interdit socle s0 ·
roundtrip CIR+UserResponse · `format_refacto` présent après socle · `make gates` local étendu
(CI note déjà : « DATA/ hors git → make gates en local »).

## Seuils spikes P0-6 (figés 2026-10-08)

- Densité causale code : GO si ≥ 15 % d'unités causales (compteur actuel `causal_scorer.py`), sinon re-scopage prog acté avant C1.
- PDF LaTeX : GO si sections propres + formules en blocs opaques sur échantillon Exo7 (0.5j timeboxé), sinon spike complémentaire ou re-scopage C2/C3.

## Chantiers différés (ne pas oublier — déclencheurs explicites)
- **Embedding char-aware (morphologie OOV)** : plan figé `plan-embedding-char-aware.md`
  (vérifié 2026-10-15 : OOV 21 % dont ~13 % morphologique, gain borné ≤ +0.02).
  DÉCLENCHEUR : gold-3000 livré + val/test ≥ 300 (le goulot N/val déplacé vers la
  couverture). D'ici là : NE PAS exécuter J1 (décision actée 2026-10-15).
