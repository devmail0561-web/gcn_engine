#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""D.0-min : Extrait les phrases OOV depuis le corpus existant sans annotation manuelle.

Stratégie :
  - Construire le lexique fermé (lemmes vus dans real/train)
  - Retenir les phrases de generated_1000.json contenant ≥ 1 lemme hors lexique (OOV)
  - Écrire gcn-datasets/test/oov_split_test.json

Usage :
    python scripts/d0_min_oov_split.py [--min-oov 1] [--max-sentences 50]
"""
import argparse
import json
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "gcn-datasets" / "corpus" / "generated_1000.json"
TRAIN_PATH  = REPO_ROOT / "gcn-datasets" / "real" / "train" / "train.json"
OUT_PATH    = REPO_ROOT / "gcn-datasets" / "test" / "oov_split_test.json"
CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "PROPN", "ADV"})


def build_train_lexicon(train_path: Path) -> frozenset[str]:
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)
    lemmas: set[str] = set()
    sentences = (
        data if isinstance(data, list)
        else data.get("sentences", data.get("document", {}).get("sentences", []))
    )
    for sent in sentences:
        for tok in sent.get("tokens", []):
            if tok.get("pos") in CONTENT_POS:
                lemmas.add(tok["lemma"].lower())
    return frozenset(lemmas)


def extract_oov_sentences(
    corpus_path: Path,
    lexicon: frozenset[str],
    min_oov: int,
    max_sentences: int,
) -> list[dict]:
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    sentences = data["document"]["sentences"]
    results = []
    for sent in sentences:
        oov_lemmas = [
            tok["lemma"].lower()
            for tok in sent.get("tokens", [])
            if tok.get("pos") in CONTENT_POS
            and tok["lemma"].lower() not in lexicon
        ]
        if len(oov_lemmas) >= min_oov:
            results.append({**sent, "oov_lemmas": oov_lemmas})
        if len(results) >= max_sentences:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-oov", type=int, default=1,
                        help="Nombre minimum de lemmes OOV par phrase (défaut: 1)")
    parser.add_argument("--max-sentences", type=int, default=50,
                        help="Nombre maximum de phrases à extraire (défaut: 50)")
    args = parser.parse_args()

    print(f"Lecture lexique train: {TRAIN_PATH}")
    lexicon = build_train_lexicon(TRAIN_PATH)
    print(f"  → {len(lexicon)} lemmes dans le lexique fermé train")

    print(f"Extraction OOV depuis: {CORPUS_PATH}")
    sentences = extract_oov_sentences(CORPUS_PATH, lexicon, args.min_oov, args.max_sentences)
    print(f"  → {len(sentences)} phrases OOV trouvées (min_oov={args.min_oov})")

    output = {
        "schema_version": "1.0",
        "description": f"OOV split automatique — min_oov={args.min_oov}, lexique_train={len(lexicon)} lemmes",
        "source": str(CORPUS_PATH.name),
        "lexicon_source": str(TRAIN_PATH.name),
        "sentences": sentences,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {OUT_PATH} ({len(sentences)} phrases)")

    if sentences:
        sample = sentences[0]
        print(f"\nExemple: {sample['text'][:80]}")
        print(f"  OOV: {sample['oov_lemmas'][:5]}")


if __name__ == "__main__":
    main()
