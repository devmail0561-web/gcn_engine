"""
build_annotations.py — Construit les entrées JSON complètes du dataset à partir
des phrases candidates annotées manuellement (CIR) + tokens UD générés automatiquement.

Usage :
    python3 build_annotations.py --input candidates.py --output real/train_c1/c1_annotations.json

Le script :
1. Lit la liste CANDIDATES depuis le fichier Python fourni
2. Génère les tokens UD via spaCy fr_core_news_sm
3. Assemble la structure JSON complète compatible avec GCNDataLoader
4. Écrit un document {"document": {"sentences": [...]}} prêt à être fusionné dans train/

Format attendu d'une entrée CANDIDATES :
{
    "text": "...",
    "causal_pattern": "filter",
    "n001": {"type": "action", "label": "appliquer", "span_start": 1, "span_end": 5},
    "n002": {"type": "condition", "label": "valider", "span_start": 6, "span_end": 10},
    "edge_relation": "filter",
    "edge_negated": False,
}
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "gcn-tools" / "gcn-scraper" / "src"))

try:
    from gcn_scraper.ud_annotator import annotate_ud
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[1] / "gcn-tools" / "gcn-scraper" / "src" / "gcn_scraper"))
    from ud_annotator import annotate_ud


def _node_entry(node_dict: dict, node_id: str) -> dict:
    return {
        "id": node_id,
        "type": node_dict["type"],
        "label": node_dict["label"],
        "token_span": [node_dict["span_start"], node_dict["span_end"]],
        "origin": "explicit",
        "scope": "specific",
        "temporal_index": int(node_id[1:]) - 1,
        "modifiers": [],
        "attributes": {},
    }


def _edge_entry(source: str, target: str, relation: str, negated: bool = False) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "attributes": {
            "confidence": 0.9,
            "explicit": True,
            "negated": negated,
        },
    }


def _clamp_span(span_start: int, span_end: int, n_tokens: int) -> tuple[int, int]:
    """Cap span to [1, n_tokens], ensure start <= end."""
    s = max(1, min(span_start, n_tokens))
    e = max(s, min(span_end, n_tokens))
    return s, e


def build_sentence(candidate: dict, sentence_id: str) -> dict:
    text = candidate["text"]
    tokens = annotate_ud(text, lang="fr")
    n_tok = len(tokens)

    # Clamp spans to actual token count
    for key in ("n001", "n002"):
        nd = candidate[key]
        s, e = _clamp_span(nd["span_start"], nd["span_end"], n_tok)
        nd["span_start"], nd["span_end"] = s, e

    n001 = _node_entry(candidate["n001"], "n001")
    n002 = _node_entry(candidate["n002"], "n002")
    edge = _edge_entry(
        "n001", "n002",
        candidate["edge_relation"],
        candidate.get("edge_negated", False),
    )

    return {
        "id": sentence_id,
        "text": text,
        "causal_pattern": candidate["causal_pattern"],
        "tokens": tokens,
        "cir": {
            "nodes": [n001, n002],
            "edges": [edge],
        },
    }


def load_candidates(path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location("candidates_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CANDIDATES


def main():
    parser = argparse.ArgumentParser(description="Build annotated JSON from candidates")
    parser.add_argument("--input", required=True, type=Path, help="Python file with CANDIDATES list")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--id-prefix", default="c1_", help="Sentence ID prefix")
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    print(f"Chargé {len(candidates)} candidats depuis {args.input}")

    sentences = []
    errors = []
    for i, cand in enumerate(candidates):
        sid = f"{args.id_prefix}{i:04d}"
        try:
            entry = build_sentence(cand, sid)
            sentences.append(entry)
        except Exception as e:
            errors.append((sid, str(e)))
            print(f"  ERREUR {sid}: {e}")

    print(f"Générés: {len(sentences)} sentences ({len(errors)} erreurs)")

    # Distribution des types de relations
    from collections import Counter
    rel_counts = Counter(s["causal_pattern"] for s in sentences)
    print("Distribution :")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<22} {cnt:3d}")

    output = {"document": {"sentences": sentences}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé : {args.output}")


if __name__ == "__main__":
    main()
