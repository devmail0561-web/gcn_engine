#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Sélection seedée d'un lot d'annotation (lots 11+).

Lit le pool (candidates + fresh), EXCLUT les phrases déjà annotées
(lot*.json) ET les quarantaines (quarantaine_*.json — dette lot09 :
le proposeur re-proposait les refusés), propose via balanced_auto,
plafonne par relation (rares d'abord), mélange et écrit la sélection.

Usage :
    python3 scripts/select_lot.py --caps filter:6,control:2 --n 100 \
        --out /tmp/opencode/lot11_sel.json [--seed 42]
"""
import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gcn-tools" / "gcn-annotate" / "src"))
from gcn_annotate.balanced_auto import annotate  # noqa: E402

V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"
EN_PAT = re.compile(
    r"\b(the|is|are|was|were|this|that|with|from|have|has|will|would|which|"
    r"their|there|been|more|than|also|often|used|when|what|does)\b", re.I)


def load_pool(extra=("candidates/candidates.json", "candidates/candidates_fresh.json", "candidates/candidates_divers.json", "candidates/candidates_rares.json", "candidates/candidates_rares2.json")):
    pool = {}
    for rel in extra:
        p = V4 / rel
        if not p.exists():
            continue
        for s in json.load(open(p))["document"]["sentences"]:
            pool.setdefault(s["id"], s["text"])
    return pool


def load_excluded():
    done, quar = set(), set()
    for f in glob.glob(str(V4 / "annotated" / "lot*.json")):
        for s in json.load(open(f))["document"]["sentences"]:
            done.add(s["id"])
    for f in glob.glob(str(V4 / "annotated" / "quarantaine*.json")):
        for q in json.load(open(f)):
            quar.add(q.get("id"))
    for f in glob.glob(str(V4 / "annotated" / "topup_*.json")) + glob.glob(str(V4 / "annotated" / "*_pending.json")):
        for s in json.load(open(f))["document"]["sentences"]:
            done.add(s["id"])
    return done, quar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True,
                    help="plafonds par relation, ex. filter:6,control:2,concession:22")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--props-cache", default="/tmp/opencode/props_cache.json",
                    help="cache {id: [relation, lang]} des propositions")
    args = ap.parse_args()
    caps = dict(c.split(":") for c in args.caps.split(","))
    caps = {k: int(v) for k, v in caps.items()}
    rng = random.Random(args.seed)

    pool = load_pool()
    done, quar = load_excluded()
    print(f"pool: {len(pool)} | annotées: {len(done)} | quarantaines: {len(quar)}")

    try:
        props = json.load(open(args.props_cache))
    except (FileNotFoundError, json.JSONDecodeError):
        props = {}
    new = 0
    for sid, text in pool.items():
        if sid in done or sid in quar or sid in props:
            continue
        lang = "en" if EN_PAT.search(text[:120]) else "fr"
        try:
            r = annotate(text, lang)
        except Exception:
            r = None
        props[sid] = [r["causal_pattern"], lang] if r and r.get("edges") else [None, lang]
        new += 1
    json.dump(props, open(args.props_cache, "w"))
    print(f"propositions (nouveau: {new}) :",
          dict(Counter(r for r, _ in props.values() if r)))

    by_rel: dict[str, list] = {}
    for sid, (rel, lang) in props.items():
        if sid not in done and sid not in quar and rel:
            by_rel.setdefault(rel, []).append((sid, lang))
    sel = []
    for rel, cap in caps.items():
        ids = by_rel.get(rel, [])
        rng.shuffle(ids)
        take = ids[:cap]
        print(f"{rel}: {len(take)}/{len(ids)} (cap {cap})")
        sel.extend([(sid, rel, lang) for sid, lang in take])
    rng.shuffle(sel)
    sel = sel[:args.n]
    json.dump(sel, open(args.out, "w"), ensure_ascii=False)
    print(f"TOTAL: {len(sel)} -> {args.out}")
    print("LANG:", dict(Counter(l for _, _, l in sel)))


if __name__ == "__main__":
    main()
