# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
make_verbalize_pairs.py — Convertit gcn-nl (train.json) → gcn-verbalize.

Pour chaque phrase annotée : (cir, text) → VerbalizeSample.
Sortie : verbalize_real.json prêt pour --verbalize-dir de gcn-train.

Usage :
    python3 scripts/make_verbalize_pairs.py \
        --input real/augmented/c1_oversampled/train.json \
        --output real/verbalize/verbalize_real.json
"""
import argparse
import json
from pathlib import Path


def _node_gcn_nl_to_verbalize(i: int, node: dict) -> dict:
    """ClauseRecord (gcn-nl) → CausalIR node (gcn-verbalize)."""
    return {
        "id": i,
        "node_type": node.get("type", node.get("node_type", "processus")),
        "label": node.get("label", ""),
        "origin": node.get("origin", "explicit"),
    }


def _edge_gcn_nl_to_verbalize(edge: dict, node_id_map: dict) -> list | None:
    """EdgeRecord (gcn-nl) → [src_int, dst_int, attrs], ou None si endpoints inconnus."""
    src = node_id_map.get(edge.get("source", ""))
    dst = node_id_map.get(edge.get("target", ""))
    if src is None or dst is None:
        return None
    attrs = edge.get("attributes", {})
    relation   = attrs.get("relation")   or edge.get("relation")
    confidence = attrs.get("confidence") or edge.get("confidence")
    negated    = attrs.get("negated",    edge.get("negated",    False))
    explicit   = attrs.get("explicit",   edge.get("explicit",   True))
    if relation is None or confidence is None:
        return None
    return [src, dst, {
        "relation":   relation,
        "confidence": float(confidence),
        "negated":    bool(negated),
        "explicit":   bool(explicit),
    }]


def sentence_to_verbalize(s: dict) -> dict | None:
    """Phrase gcn-nl → exemple gcn-verbalize. Retourne None si pas de texte ni de CIR."""
    text = s.get("text", "").strip()
    cir  = s.get("cir", {})
    nodes_raw = cir.get("nodes", [])
    edges_raw = cir.get("edges", [])

    if not text or not nodes_raw:
        return None

    # Construire la map id_string → index_int pour les arêtes
    node_id_map = {n.get("id", str(i)): i for i, n in enumerate(nodes_raw)}

    nodes = [_node_gcn_nl_to_verbalize(i, n) for i, n in enumerate(nodes_raw)]
    edges = [e for e in (_edge_gcn_nl_to_verbalize(r, node_id_map) for r in edges_raw)
             if e is not None]

    return {
        "id": s.get("id", ""),
        "causal_ir": {
            "source_lang": {"natural": {"lang": "fr"}},
            "source_text": text,
            "nodes": nodes,
            "edges": edges,
        },
        "surfaces": [
            {"lang": "fr", "text": text, "quality": "gold"}
        ],
    }


def convert(input_path: Path, output_path: Path) -> None:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    sentences = raw.get("document", {}).get("sentences", [])
    examples = []
    skipped  = 0

    for s in sentences:
        ex = sentence_to_verbalize(s)
        if ex is not None:
            examples.append(ex)
        else:
            skipped += 1

    out = {"schema_version": "1.0", "schema": "gcn-verbalize", "examples": examples}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Converti : {len(examples)} paires (CIR, texte)  —  {skipped} ignorées")
    print(f"Sauvegardé : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
