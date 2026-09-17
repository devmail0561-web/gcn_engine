"""
Diagnostic rapide : mesure le pourcentage de phrases où les token_span CIR
sont invalides (chevauchement identique, hors-bornes, start >= end).
"""
import json
import spacy


def diagnose_spans(annotated_path: str, n_samples: int = 100) -> dict:
    nlp = spacy.load("fr_core_news_sm")
    data = json.loads(open(annotated_path).read())
    sents = data["document"]["sentences"]

    n_invalid = 0
    examples = []
    checked = 0
    for s in sents:
        if checked >= n_samples:
            break
        nodes = s.get("cir", {}).get("nodes", [])
        if not nodes:
            continue
        spans = [tuple(n["token_span"]) for n in nodes]
        doc = nlp(s["text"])
        n_tokens = len(doc)
        invalid = (
            any(start >= end for start, end in spans)
            or any(end > n_tokens for start, end in spans)
            or len(spans) != len(set(spans))
        )
        if invalid:
            n_invalid += 1
            if len(examples) < 5:
                examples.append({
                    "id": s.get("id", "?"),
                    "text": s["text"][:100],
                    "spans": spans,
                    "n_tokens": n_tokens,
                })
        checked += 1

    result = {
        "checked": checked,
        "invalid": n_invalid,
        "invalid_pct": (n_invalid / max(checked, 1)) * 100,
        "examples": examples,
    }
    print(f"Phrases vérifiées : {checked}")
    print(f"Invalide : {n_invalid}/{checked} ({result['invalid_pct']:.1f}%)")
    for ex in examples:
        print(f"  [{ex['id']}] tokens={ex['n_tokens']} spans={ex['spans']}")
        print(f"    \"{ex['text']}\"")
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated.json"
    diagnose_spans(path)
