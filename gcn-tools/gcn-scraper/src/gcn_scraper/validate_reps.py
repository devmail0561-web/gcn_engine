"""
Validation post-annotation : vérifie que reps_from_sentence retourne des reps non vides
pour le dataset réel annoté avec les tokens UD.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gcn-python", "src"))

from gcn_python.data.json_reader import load_sentences
from gcn_python.data.loader import reps_from_sentence


def validate_reps(annotated_ud_path: str) -> dict:
    records = load_sentences(Path(annotated_ud_path))

    n_ok = 0
    n_empty = 0
    empty_examples = []

    for rec in records:
        reps, idxs, conn = reps_from_sentence(rec)
        if reps:
            n_ok += 1
        else:
            n_empty += 1
            if len(empty_examples) < 5:
                empty_examples.append({"id": rec.id, "text": rec.text[:80]})

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
