#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Split stratifié train/val du dataset v4 (recette gelée, seed 42).

Stratification sur la première relation d'arête (`_none` si sans arête),
20 % par groupe (min 1), mélange final. Rejoue à l'identique tout split
v4r/v4x/v4xx (même code, même seed) — les splits /tmp ne sont plus la
référence, cette recette l'est.

Usage :
    python3 scripts/make_split.py --out /tmp/opencode/v4xx [--seed 42]
"""
import argparse
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ss = []
    for f in sorted(glob.glob(str(V4 / "annotated" / "lot*.json"))):
        ss.extend(json.load(open(f))["document"]["sentences"])
    groups: dict[str, list] = {}
    for s in ss:
        es = s.get("cir", {}).get("edges", [])
        groups.setdefault(es[0].get("relation", "_none") if es else "_none", []).append(s)
    tr, va = [], []
    for rel, g in sorted(groups.items()):
        rng.shuffle(g)
        n_val = max(1, int(0.2 * len(g)))
        va.extend(g[:n_val])
        tr.extend(g[n_val:])
    rng.shuffle(tr)
    rng.shuffle(va)
    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    json.dump({"schema_version": "4.0", "document": {"id": "train", "sentences": tr}},
              open(out / "train" / "train.json", "w"), ensure_ascii=False)
    json.dump({"schema_version": "4.0", "document": {"id": "val", "sentences": va}},
              open(out / "val" / "val.json", "w"), ensure_ascii=False)
    print(f"train: {len(tr)} val: {len(va)} -> {out}")


if __name__ == "__main__":
    main()
