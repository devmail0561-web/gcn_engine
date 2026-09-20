#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Migration v1 -> v2 : source -> sources + schema_version 2.0.

Usage : python migrate_v1_v2.py <input.json> <output.json>
Teste la non-perte : compte arêtes avant/après, assert égalité.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _migrate_edge(e: dict) -> dict:
    e = dict(e)
    if "sources" not in e and "source" in e:
        e["sources"] = [e["source"]]
    return e


def _count_edges(doc: dict) -> int:
    n = 0
    for s in doc.get("document", {}).get("sentences", []):
        n += len(s.get("cir", {}).get("edges", []))
    return n


def migrate(doc: dict) -> dict:
    doc = json.loads(json.dumps(doc))
    doc["schema_version"] = "2.0"
    for s in doc.get("document", {}).get("sentences", []):
        cir = s.get("cir", {})
        cir["edges"] = [_migrate_edge(e) for e in cir.get("edges", [])]
    return doc


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage : {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        raise SystemExit(2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    doc = json.loads(src.read_text(encoding="utf-8"))
    before = _count_edges(doc)
    out = migrate(doc)
    after = _count_edges(out)
    assert before == after, f"perte d'arêtes : {before} -> {after}"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Migration OK : {before} arête(s), schema_version=2.0 -> {dst}")


if __name__ == "__main__":
    main()
