#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Merge gold v4 + silver v5 (garder) + split stratifié (recette gelée, seed 42).

Sélection au livré, gold intact :
  - gold : `annotated/lot*.json` moins tombstones (`quarantaine*.json`,
    `*_pending.json` — cf. make_split.py) ;
  - silver : `v5/*/*_propose.json` moins verdicts `quarantaine` de
    `v5/gates/aveugle_200.json` (garder par défaut).
--quota N : plafond uniforme par relation AVANT split (règle §1, §2.4 ;
  `_none` non plafonné, strate edgeless à part). Sans --quota : tout le pool.
`_methode` ("silver-auto" vs absent) conservé tel quel pour --silver-weight.

Usage :
    python3 scripts/merge_goldsilver.py --out /tmp/opencode/merge1 --quota 140 [--seed 42]
"""
import argparse
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"
V5 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v5"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_gold():
    excluded = set()
    for f in sorted(glob.glob(str(V4 / "annotated" / "quarantaine*.json"))):
        try:
            for q in json.load(open(f)):
                if isinstance(q, dict) and q.get("id"):
                    excluded.add(q["id"])
        except Exception:
            pass
    for f in sorted(glob.glob(str(V4 / "annotated" / "*_pending.json"))):
        try:
            for s in json.load(open(f))["document"]["sentences"]:
                excluded.add(s.get("id"))
        except Exception:
            pass
    ss = []
    for f in sorted(glob.glob(str(V4 / "annotated" / "lot*.json"))):
        for s in json.load(open(f))["document"]["sentences"]:
            if s.get("id") not in excluded:
                ss.append(s)
    return ss, len(excluded)


def load_silver():
    gate = V5 / "gates" / "aveugle_200.json"
    qav = set()
    if gate.exists():
        for x in json.load(open(gate)):
            if x.get("verdict") == "quarantaine" and x.get("id"):
                qav.add(x["id"])
    ss = []
    for f in sorted(V5.rglob("*_propose.json")):
        for s in json.load(open(f))["document"]["sentences"]:
            if s.get("id") not in qav:
                ss.append(s)
    return ss, len(qav)


def rel_of(s) -> str:
    es = s.get("cir", {}).get("edges", [])
    return es[0].get("relation", "_none") if es else "_none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quota", type=int, default=0)
    args = ap.parse_args()
    if args.quota < 0:
        raise SystemExit("--quota doit être ≥ 0.")
    rng = random.Random(args.seed)
    gold, n_excl = load_gold()
    silver, n_qav = load_silver()
    print(f"gold livré: {len(gold)} (tombstones exclus: {n_excl})")
    print(f"silver garder: {len(silver)} (aveugle-quar exclus: {n_qav})")
    ss = gold + silver
    rng.shuffle(ss)
    if args.quota > 0:
        kept, counts = [], {}
        for s in ss:
            r = rel_of(s)
            if r != "_none" and counts.get(r, 0) >= args.quota:
                continue
            counts[r] = counts.get(r, 0) + 1
            kept.append(s)
        ss = kept
    print("quotas:", dict(sorted(Counter(rel_of(s) for s in ss).items())))
    groups: dict[str, list] = {}
    for s in ss:
        groups.setdefault(rel_of(s), []).append(s)
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
    tr_doc = {"schema_version": "4.0", "document": {"id": "train", "sentences": tr}}
    va_doc = {"schema_version": "4.0", "document": {"id": "val", "sentences": va}}
    (out / "train" / "train.json").write_text(json.dumps(tr_doc, ensure_ascii=False),
                                              encoding="utf-8")
    (out / "val" / "val.json").write_text(json.dumps(va_doc, ensure_ascii=False),
                                          encoding="utf-8")
    manifest = {
        "seed": args.seed, "quota": args.quota,
        "gold_n": len(gold), "silver_n": len(silver),
        "train_n": len(tr), "val_n": len(va),
        "train_sha256": hashlib.sha256(json.dumps(tr_doc, sort_keys=True).encode()).hexdigest()[:16],
        "val_sha256": hashlib.sha256(json.dumps(va_doc, sort_keys=True).encode()).hexdigest()[:16],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"train: {len(tr)} val: {len(va)} -> {out}")
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
