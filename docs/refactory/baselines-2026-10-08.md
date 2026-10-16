# Snapshot baselines — lot A (2026-10-08)

> Référence ROADMAP §Baselines (I2). Rejouable : commandes ci-dessous, seed/gate N/A (mesures, pas d'expérience).
> Commit : `f9a8214` (branche `refacto/arch-cible-refacto-2026-10`). Date : 2026-10-08T11:51:07Z.

## Suite backend (référence P0)

```bash
cargo test -p gcn-backend   # depuis gcn-core/
```

- Résultat : **67 passés (61 + 6), 0 échoué** — `integration_backend.rs` (61) + `test_ternary_pearl.rs` (6).
- Durée : **4,2 s** (real, rebuild chaud ; premier build non mesuré).

## Tailles

```bash
du -sh gcn-datasets/DATA/supervision gcn-python/models gcn-datasets/checkpoints
```

- supervision : **351 M** · models : **56 M** · checkpoints : **253 M**.

## Outillage

- `python3 --version` → **3.11.9** · `cargo/rustc` → **1.98.0**.
- `nvidia-smi` → **absent (pas de GPU local)**. Conséquence : S4-opt (teacher GPU) impossible en local — renforce son exclusion du périmètre.

## Références métriques (documentées, NON re-mesurées — voir caveats)

- `val_edge_f1=0.468` (v2.4.0, PROGRESS.md) **vs** `0.1867` (harnais v2.5.0, README.md:240, « non comparable ») — **réfs contradictoires, re-mesure exigée sur harnais unique** (slice/socle, pas ici : requiert un run d'entraînement, hors lot A).
- ECE : outil dispo (`evaluation/calibration.py` : `ece_score`, `optimize_temperature`) — **mesure différée à la gate Ph5** (requiert sorties modèle sur gold, post-slice).

## Gardes I1 (état au snapshot)

- `gcn_transformers` dans `gcn-python/src/` : **0 occurrence** → garde CI dure ajoutée (lot A).
- `import torch` top-level dans `gcn-python/src/` : **0** (torch = optional-dep `pyproject.toml:41`, imports lazy) → garde CI dure ajoutée.
- `subprocess|gcn_bin|Lattice` (`*.py`) : **8 fichiers sources** (`bridge.py`, `__init__.py`, `cgnp.py`, `index.py`, `train.py`, `bootstrap.py`, `discuss.py`, `engine.py`) → comptage CI informatif (cible acceptation A = 0, fin de refacto).
