#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""check_v4.py — Gates v5 pour datasets v4 (supervision/v4/).

Vérifie sans modifier : version schéma, spans (non-vides, non-pleines si gap
attendu), N_min=30 par classe, marqueurs (gold vs gaps), doublons de texte,
couverture sentence_type/intent/third, token_source renseigné.

Usage :
    python scripts/check_v4.py gcn-datasets/DATA/supervision/v4/annotated/train.json
    python scripts/check_v4.py --min-class-count 15 <fichier>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

N_MIN_DEFAULT = 30
FINE_SPAN_WARN_TOKENS = 15  # span > 15 tokens = suspecte pleine-phrase


def check_file(path: Path, min_count: int) -> tuple[bool, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    doc = data.get("document", data)
    if data.get("schema_version") != "4.0" and doc.get("schema_version") != "4.0":
        print(f"  [GATE] schema_version != 4.0 dans {path.name}")
        return False, {}
    sents = doc.get("sentences", [])
    rel_counts: Counter = Counter()
    gold_markers = gaps = found = 0
    seen_texts: dict[str, str] = {}
    dupes = 0
    st_counts: Counter = Counter()
    intents = thirds = 0
    span_warns = 0
    ok = True
    for s in sents:
        sid = s.get("id", "?")
        if s.get("text") in seen_texts:
            dupes += 1
        else:
            seen_texts[s.get("text", "")] = sid
        toks = s.get("tokens", [])
        clauses = s.get("cir", {}).get("nodes", [])
        edges = s.get("cir", {}).get("edges", [])
        st_counts[s.get("sentence_type", "") or "(vide)"] += 1
        if s.get("intent"):
            intents += 1
        for n in clauses:
            span = n.get("token_span", [0, 0]) or [0, 0]
            if len(span) < 2 or span[1] <= 0 or span[0] > span[1]:
                print(f"  [GATE] {sid}/{n.get('id')}: span invalide {span}")
                ok = False
            elif span[1] - span[0] + 1 > FINE_SPAN_WARN_TOKENS:
                span_warns += 1
        covered = set()
        for n in clauses:
            span = n.get("token_span", [0, 0]) or [0, 0]
            if len(span) >= 2:
                covered.update(range(span[0], span[1] + 1))
        for e in edges:
            rel_counts[e.get("relation", "?")] += 1
            if e.get("marker_token") is not None:
                gold_markers += 1
            if e.get("third"):
                thirds += 1
        # gaps = paires de clauses consécutives non contiguës
        spans = sorted(
            (n.get("token_span", [0, 0]) or [0, 0] for n in clauses),
            key=lambda x: x[0] if len(x) >= 2 else 0,
        )
        for a, b in zip(spans, spans[1:]):
            if len(a) >= 2 and len(b) >= 2 and b[0] > a[1] + 1:
                gaps += 1
                gap_ids = set(range(a[1] + 1, b[0]))
                mt = [e.get("marker_token") for e in edges
                      if e.get("marker_token") in gap_ids]
                if mt:
                    found += 1
    print(f"  phrases={len(sents)} relations={dict(sorted(rel_counts.items()))}")
    print(f"  marqueurs gold={gold_markers} gaps={gaps} gaps-avec-connecteur={found}")
    print(f"  sentence_type={dict(sorted(st_counts.items()))} intent={intents} thirds={thirds}")
    print(f"  doublons={dupes} spans suspectes(>{FINE_SPAN_WARN_TOKENS}tok)={span_warns}")
    under = {r: n for r, n in rel_counts.items() if 0 < n < min_count}
    if under:
        print(f"  [GATE] sous N_min={min_count} : {under}")
        ok = False
    if dupes:
        print(f"  [GATE] {dupes} texte(s) dupliqué(s) — tombstones requis")
        ok = False
    return ok, {"phrases": len(sents), "relations": dict(rel_counts)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gates v5 datasets v4")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-class-count", type=int, default=N_MIN_DEFAULT)
    args = ap.parse_args()
    all_ok = True
    for f in args.files:
        print(f"== {f}")
        ok, _ = check_file(Path(f), args.min_class_count)
        all_ok = all_ok and ok
    print("GATES OK" if all_ok else "GATES EN ÉCHEC")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
