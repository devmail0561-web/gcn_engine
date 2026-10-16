# Registre des gates — qui décide, sur quelle preuve (phase 0)

> Doc locale (docs/ non versionné). Statut : owners nommés le 2026-10-08 (solo).
> Références plans : `plan-remodelage-tout-appris.md` (slice, R8/R9/R10), `plan-backend-durcissement.md` (Ph5),
> `plan-dataset-3M.md` (J7), `plan-annotation-cir.md` (silver 0.3, gold).

## Table des gates

| Gate | Phase | Décideur | Preuve exigible | Règle |
|---|---|---|---|---|
| Slice -1 | refacto -1 | Michel Tendeng (solo) | loss<0.20 ET acc 4/4 ET Δcos≥0.05 (seeds 42/123), log rejouable | GO / WARN-CONTINUE (+100 steps, 2 lr) / STOP sec si hors zone grise |
| Socle 0c gold | refacto 0 | Michel Tendeng (solo) | `gold_mini` 60-100 + held-out HORS docs, inventaire seed | GO si 60-100 conformes schéma 8/19 (via `normalize.py`) |
| Calibration Ph5 | backend 5 | Michel Tendeng (solo) | ECE + monotonie sur gold (script rejouable, N et seed notés) | GO → 6a/7c ; NO-GO → 6b + quantification enterrée (acté) |
| Gel R8/R10 | refacto 4 | Michel Tendeng (solo) | N/classe/split sur gold (matrice publiée) | <30/classe → gel (pas de seuils, pas de calibration) |
| Silver 0→0.3 | annotation | Michel Tendeng (solo) | erreur silver sur échantillon stratifié ~1000/jalon + IC95 | Monte si borne haute IC95 ≤ 20 % ; maintien 0.0 si borne basse > 20 % ; sinon +500 phrases et re-décision (max 2 rounds, puis arbitrage humain) |
| Fusion J7 | dataset 7 | Michel Tendeng (solo) | matrice langue×relation×domaine + fourchettes §5 ±tolérance | Case hors fourchette → campagne de rattrapage AVANT gel |
| Format s0 | refacto 0a | Michel Tendeng (solo) | `format_refacto=20261007` lu/écrit + roundtrip Rust vert | Bloque tête act et socle 0b sinon |

## Done global du remodelage (proposition)

Remodelage déclaré terminé ssi : slice GO + socle vert + backend P0 vert + always-answer 100 % (property Ph8) + J7 fourchettes tenues + (ECE acceptable → 6a/7c livrés **ou** NO-GO acté → 6b livré). Métriques figées dans le snapshot baselines (ROADMAP.md).

## Règles STOP

- Slice STOP → pas de socle, diagnostic écrit avant toute suite.
- Ph5 NO-GO + 6b jugé insuffisant → STOP avec sortie écrite (pas d'acharnement quantification).
- Spike P0-6 NO-GO → re-scopage dataset acté avant tout code.
- J7 bloqué 2 cycles de rattrapage → arbitrage périmètre (variante B ou quotas révisés).

## Verdicts (phase build)

- **Socle 0c gold — GO (2026-10-08)** : `gold_mini` n=85 ∈ [60-100], 11 relations × 7-10, 100 % conformes schéma 8/19 (valid_v4, `scripts/select_gold_mini.py`, seed 42, déterministe prouvé par re-run) + held-out re-gelés hors docs (mixte_final test 380, v4/final2 test) + manifests sha256. Preuve : `gold_mini_manifest.json`, `heldout_manifest.json`.
- **Baselines I2 — snapshot (2026-10-08)** : voir `baselines-2026-10-08.md` (backend 67/67 en 4,2 s ; pas de GPU).
- **Gel R8/R10 — programmé + ACTÉ en démo (2026-10-08)** : recomptage sans cap (`comptes_classes.json`) — 11 relations ≥1 491 occurrences à l'échelle supervision, mais **8 nouvelles relations à 0 occurrence** : gel automatique de ces classes (pas de seuils, pas de calibration) jusqu'au gold-3000. Le gate portera sur le gold par classe/split, pas sur ces totaux bruts. Démo : `grid_f1_thresholds.py` → GEL exit 2 sur gold actuel (11 relations 6-9 < 30) ; `--force` : 250 essais, meilleur ≠ (0.0,0.65,1.0), non adopté. Vrai grid + calibration après gold-3000.
- **Slice -1 — GO (2026-10-08, vérifié par re-run)** : `scripts/slice_minus1_gate.py` (harnais existant réutilisé, 4 phrases gold_mini 2-nœuds/1-arête, 300 steps, arch défauts gcn-train, lr=0.05) — seed 42 : loss_eval=0.0057, acc=4/4, Δcos=+0.130 ; seed 123 : 0.0066, 4/4, +0.094. Seuils gate (0.20/4/4/0.05) tenus des deux seeds, sortie 0. Caveat : phrases jouets (seuil loss facile) — le gate est appliqué tel qu'écrit ; la robustesse vient du socle + held-out. Preuve : `/tmp/slice_minus1/metrics.json` (re-run reproductible : `PYTHONPATH=gcn-python/src python3 scripts/slice_minus1_gate.py`). **Socle autorisé.**
- **Backend P0 — GO (2026-10-08) → P0 LIVRÉ** : `cargo test -p gcn-backend` 102 passés (61 intégration + 11 properties/fuzz + 24 gold + 6 ternaires), `clippy --all-targets -D warnings` 0 erreur, `fmt` propre, workspace OK, 3 snapshots JSON stables (`tests/snapshots/`). Ph1 (négation + tiers) + Ph2 (exact-puis-flou) + Ph3 (Dijkstra + TOPK) + Ph4 (properties + fuzz + validité joints) livrés avec preuves (CHANGELOG + README crate).
- **Backend Ph8 — livré (2026-10-08, D-PH8 : socle 0b débloqué)** : `respond()` + enveloppe `UserResponse` versionnée + verbalizer `say()` FR + `gcn query` pont M1 (enveloppe JSON, exit 0 sauf Degrade → 1) + roundtrip CIR→UserResponse testé. Preuves : 12 tests `test_respond` + 3 unitaires `say`, property always-answer, vérification CLI bout-en-bout.
- **Format s0 — GO (2026-10-08)** : `format_refacto=20261007` écrit par B (`to_record`, garde anti-s1/s2/s3) + vrai CIR Python (schema 2.0) parsé par Rust bout-en-bout (WHY + CYCLES reclassé) + test de compatibilité CI (`test_exchange_contract`, clés futures ignorées). **Socle 0 livré (0a + 0b + 0c).**
- **Calibration Ph5 — NO-GO (2026-10-08) → 6b, quantification enterrée (acté)** : ECE 0.33-0.55 (>> 0.15), monotonie non (seed-dépendante, held-out anti-monotone), T* saturée, N=81 < 100. Preuve : `/tmp/ph5/` (rejouable). 7c conditionné : enterré avec 6a (modifiers sans gate) ; 6b (produit sans chiffres) en route.
