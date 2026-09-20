# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
merge_datasets.py — Fusionne plusieurs fichiers JSON dataset en un seul.

Usage :
    python3 scripts/merge_datasets.py \
        real/train/train.json \
        real/augmented/c1_annotations/train.json \
        --output real/augmented/c1_merged/train.json

Vérifie la distribution des relations après fusion.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_sentences(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    doc = data.get("document", data)
    return doc.get("sentences", [])


def count_edges(sentences: list[dict]) -> Counter:
    c = Counter()
    for s in sentences:
        for e in s.get("cir", {}).get("edges", []):
            rel = e.get("relation")
            if rel:
                c[rel] += 1
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    all_sentences = []
    ids_seen: set[str] = set()
    duplicates = 0

    for path in args.inputs:
        sents = load_sentences(path)
        before = len(all_sentences)
        for s in sents:
            if s["id"] in ids_seen:
                duplicates += 1
                continue
            ids_seen.add(s["id"])
            all_sentences.append(s)
        print(f"{path.name}: {len(sents)} phrases → {len(all_sentences)-before} ajoutées")

    if duplicates:
        print(f"  Doublons ignorés : {duplicates}")

    print(f"\nTotal : {len(all_sentences)} phrases")
    counts = count_edges(all_sentences)
    total = sum(counts.values())
    print("Distribution des arêtes :")
    for rel, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<22} {cnt:3d}  ({100*cnt/total:.1f}%)")

    output = {"document": {"sentences": all_sentences}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé : {args.output}")


if __name__ == "__main__":
    main()
