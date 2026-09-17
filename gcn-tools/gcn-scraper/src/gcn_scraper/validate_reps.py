"""
Validation post-annotation : vérifie que reps_from_sentence retourne des reps non vides
pour le dataset réel annoté avec les tokens UD.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "gcn-python", "src"))

from gcn_python.data.schema import SentenceRecord, TokenRecord, ClauseRecord, EdgeRecord
from gcn_python.data.loader import reps_from_sentence


def _dict_to_sentence_record(sent: dict) -> SentenceRecord:
    """Convertit un dict sentence (annotated_ud.json) en SentenceRecord."""
    tokens = []
    for t in sent.get("tokens", []):
        tokens.append(TokenRecord(
            id=t["id"],
            form=t["form"],
            lemma=t["lemma"],
            pos=t["pos"],
            dep_rel=t["dep_rel"],
            dep_head=t.get("dep_head", 0),
            morph=t.get("morph", {}),
        ))

    clauses = []
    for n in sent.get("cir", {}).get("nodes", []):
        ts = n.get("token_span", [0, 0])
        clauses.append(ClauseRecord(
            id=n["id"],
            token_span=(ts[0], ts[1]),
            label=n.get("label", ""),
        ))

    edges = []
    for e in sent.get("cir", {}).get("edges", []):
        attrs = e.get("attributes", {})
        edges.append(EdgeRecord(
            source=e["source"],
            target=e["target"],
            relation=e["relation"],
            confidence=attrs.get("confidence", 1.0),
            explicit=attrs.get("explicit", True),
            negated=attrs.get("negated", False),
            marker_token=attrs.get("marker_token", ""),
        ))

    return SentenceRecord(
        id=sent.get("id", ""),
        text=sent.get("text", ""),
        tokens=tokens,
        clauses=clauses,
        edges=edges,
        lang=sent.get("lang", ""),
    )


def validate_reps(annotated_ud_path: str) -> dict:
    data = json.loads(open(annotated_ud_path).read())
    sents = data["document"]["sentences"]

    n_ok = 0
    n_empty = 0
    empty_examples = []

    for s in sents:
        rec = _dict_to_sentence_record(s)
        reps, idxs, conn = reps_from_sentence(rec)
        if reps:
            n_ok += 1
        else:
            n_empty += 1
            if len(empty_examples) < 5:
                empty_examples.append({"id": s.get("id", "?"), "text": s["text"][:80]})

    total = n_ok + n_empty
    report = {
        "total": total,
        "ok": n_ok,
        "empty": n_empty,
        "ok_pct": (n_ok / max(total, 1)) * 100,
        "empty_examples": empty_examples,
    }

    print(f"Total : {total}, OK : {n_ok}, Vides : {n_empty} ({report['ok_pct']:.1f}%)")
    for ex in empty_examples:
        print(f"  [{ex['id']}] \"{ex['text']}\"")

    if n_ok < n_empty * 10:
        print("ATTENTION : trop de reps vides — vérifier l'alignement token_span")
    else:
        print("OK : majorité de reps non vides")

    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated_ud.json"
    validate_reps(path)
