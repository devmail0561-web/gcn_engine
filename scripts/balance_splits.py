#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""balance_splits.py — Rééquilibrage des splits v4 par sous-échantillonnage.

Quota = min relation count dans le split (REGLE_EQUILIBRE_DATASET ±10%).
Classe rare : conservée intégralement.
Classe fréquente : sous-échantillonnée aléatoirement (seed fixe).

Usage :
    python scripts/balance_splits.py
    python scripts/balance_splits.py --dry-run
    python scripts/balance_splits.py --seed 42
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "gcn-datasets" / "splits"


def primary_relation(sentence: dict) -> str:
    edges = sentence.get("cir", {}).get("edges", [])
    if edges:
        return edges[0].get("relation", "_none")
    return "_none"


def balance_split(path: Path, seed: int, dry_run: bool) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sents = data.get("document", data).get("sentences", [])

    # Grouper par relation primaire
    groups: dict[str, list] = collections.defaultdict(list)
    for s in sents:
        groups[primary_relation(s)].append(s)

    # Séparer les phrases sans arête (hors quota)
    no_edge = groups.pop("_none", [])

    # Quota = min count sur les 11 relations directes
    quota = min(len(v) for v in groups.values())

    rng = random.Random(seed)
    balanced = list(no_edge)  # garder les phrases sans arête intégralement
    counts_before = {}
    counts_after = {}
    for rel, group in sorted(groups.items()):
        counts_before[rel] = len(group)
        sampled = group if len(group) <= quota else rng.sample(group, quota)
        counts_after[rel] = len(sampled)
        balanced.extend(sampled)

    # Mélanger
    rng.shuffle(balanced)

    if "document" in data:
        data["document"]["sentences"] = balanced
    else:
        data = {"schema_version": "4.0", "document": {"sentences": balanced}}

    print(f"  {'[DRY]' if dry_run else 'OK  '} {path.parent.name}/{path.name} "
          f"{len(sents)}→{len(balanced)} phrases (quota={quota}/relation)")
    for rel in sorted(counts_before):
        diff = counts_after[rel] - counts_before[rel]
        marker = "  " if diff == 0 else f"{diff:+d}"
        print(f"    {rel:25s}: {counts_before[rel]:4d} → {counts_after[rel]:4d}  {marker}")

    if not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"split": path.stem, "before": len(sents), "after": len(balanced), "quota": quota}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Rééquilibrage splits v4 (seed={args.seed}) {'[DRY RUN]' if args.dry_run else ''}")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / split / f"{split}.json"
        balance_split(path, seed=args.seed, dry_run=args.dry_run)
        print()

    print("=" * 65)
    if not args.dry_run:
        print("Rééquilibrage terminé.")
    else:
        print("DRY RUN terminé — aucune modification.")


if __name__ == "__main__":
    main()
