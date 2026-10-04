#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Split stratifié train/val du dataset v4 (recette gelée, seed 42).

Stratification sur la première relation d'arête (`_none` si sans arête),
20 % par groupe (min 1), mélange final. Rejoue à l'identique tout split
v4r/v4x/v4xx (même code, même seed) — les splits /tmp ne sont plus la
référence, cette recette l'est.

--quota N : plafond uniforme par relation AVANT split (règle §1 : quotas
uniformes ±10 %, §2.4 : au niveau atteignable). `_none` non plafonné
(strate edgeless à part). Gold intact : le plafond ne s'applique qu'au
livré (tirage seedé). Sans --quota : comportement historique (non
équilibré, déprécié pour l'entraînement).

Tombstones : les ids en `annotated/quarantaine*.json` et
`annotated/*_pending.json` sont exclus du livré (jamais supprimés du
disque — ex. doublon inter-lots r2-00275/g0101, topup lot11 hors splits).
Dédupe inter-lots sur texte normalisé (garde le plus ancien).

Usage :
    python3 scripts/make_split.py --out /tmp/opencode/v4xx [--seed 42]
    python3 scripts/make_split.py --out /tmp/opencode/bal --quota 20
"""
import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quota", type=int, default=0,
                    help="plafond uniforme par relation (0 = aucun). Ex. 20.")
    ap.add_argument("--exclude-none", action="store_true",
                    help="exclure la strate edgeless `_none` du livré "
                         "(décision v4 finale 2026-10-04 : 65 `_none` lots "
                         "01-05 exclus, fichiers intacts).")
    args = ap.parse_args()
    if args.quota < 0:
        raise SystemExit("--quota doit être ≥ 0.")
    rng = random.Random(args.seed)
    ss = []
    for f in sorted(glob.glob(str(V4 / "annotated" / "lot*.json"))):
        ss.extend(json.load(open(f))["document"]["sentences"])
    # Tombstones exclus du livré (disque intact) + dédupe inter-lots.
    excluded: set[str] = set()
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
    if excluded:
        before = len(ss)
        ss = [s for s in ss if s.get("id") not in excluded]
        print(f"tombstones exclus: {before - len(ss)} (quarantaines+pending)")
    seen: set[str] = set()
    deduped = []
    for s in ss:
        key = (s.get("text") or "").strip().lower()
        if key in seen:
            print(f"doublon inter-lots exclu: {s.get('id')}")
            continue
        seen.add(key)
        deduped.append(s)
    ss = deduped
    if args.quota > 0:
        # Plafond uniforme seedé : mélange global puis premiers N par relation.
        # `_none` (edgeless) non plafonné. Gold intact (sélection au livré).
        rng.shuffle(ss)
        kept, counts = [], {}
        for s in ss:
            es = s.get("cir", {}).get("edges", [])
            rel = es[0].get("relation", "_none") if es else "_none"
            if rel != "_none" and counts.get(rel, 0) >= args.quota:
                continue
            counts[rel] = counts.get(rel, 0) + 1
            kept.append(s)
        ss = kept
        from collections import Counter
        print("quotas:", dict(sorted(Counter(
            ((s.get("cir", {}).get("edges") or [{}])[0].get("relation", "_none"))
            for s in ss).items())))
    groups: dict[str, list] = {}
    for s in ss:
        es = s.get("cir", {}).get("edges", [])
        rel = es[0].get("relation", "_none") if es else "_none"
        if rel == "_none" and args.exclude_none:
            continue
        groups.setdefault(rel, []).append(s)
    if args.exclude_none:
        print("strate `_none` exclue du livré (v4 finale : fichiers intacts).")
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
    json.dump(tr_doc, open(out / "train" / "train.json", "w"), ensure_ascii=False)
    json.dump(va_doc, open(out / "val" / "val.json", "w"), ensure_ascii=False)
    print(f"train: {len(tr)} val: {len(va)} -> {out}")
    from collections import Counter
    manifest = {
        "seed": args.seed,
        "quota": args.quota,
        "exclude_none": args.exclude_none,
        "train": len(tr),
        "val": len(va),
        "train_rel": dict(sorted(Counter(
            ((s.get("cir", {}).get("edges") or [{}])[0].get("relation", "_none"))
            for s in tr).items())),
        "val_rel": dict(sorted(Counter(
            ((s.get("cir", {}).get("edges") or [{}])[0].get("relation", "_none"))
            for s in va).items())),
        "train_sha256": hashlib.sha256(
            json.dumps(tr_doc, sort_keys=True).encode()).hexdigest()[:16],
        "val_sha256": hashlib.sha256(
            json.dumps(va_doc, sort_keys=True).encode()).hexdigest()[:16],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
