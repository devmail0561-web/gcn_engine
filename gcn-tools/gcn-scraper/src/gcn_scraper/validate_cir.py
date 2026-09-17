"""
Valide le CIR de chaque phrase du dataset réel :
1. Chaque node a un token_span [start, end] avec start < end
2. Les token_span de nodes distincts ne se chevauchent pas à l'identique
3. Chaque edge référence des node_id qui existent
4. Les relations sont dans RELATION_TYPES

Produit un rapport : nb phrases valides, invalides, exemples d'erreurs.
"""
import json
import sys
from pathlib import Path


from gcn_python.constants import RELATION_TYPES as _RELATION_TYPES_LIST

RELATION_TYPES = set(_RELATION_TYPES_LIST)


def validate_cir(annotated_path: str) -> dict:
    data = json.loads(Path(annotated_path).read_text())
    sents = data["document"]["sentences"]

    valid = 0
    invalid = 0
    errors = []

    for s in sents:
        nodes = s.get("cir", {}).get("nodes", [])
        edges = s.get("cir", {}).get("edges", [])
        sent_id = s.get("id", "?")
        sent_errors = []

        # 1. Vérifier les token_span
        spans = []
        for n in nodes:
            ts = n.get("token_span", [])
            if len(ts) != 2 or ts[0] > ts[1] or ts[0] < 1:
                sent_errors.append(f"node {n['id']}: token_span invalide {ts}")
            spans.append(tuple(ts))

        # 2. Vérifier les doublons de span
        if len(spans) != len(set(spans)):
            sent_errors.append("deux nodes ont le même token_span")

        # 3. Vérifier les edges
        node_ids = {n["id"] for n in nodes}
        for e in edges:
            if e.get("source") not in node_ids:
                sent_errors.append(f"edge source '{e.get('source')}' inexistant")
            if e.get("target") not in node_ids:
                sent_errors.append(f"edge target '{e.get('target')}' inexistant")
            rel = e.get("relation", "")
            if rel not in RELATION_TYPES:
                sent_errors.append(f"relation inconnue '{rel}'")

        if sent_errors:
            invalid += 1
            if len(errors) < 10:
                errors.append({"id": sent_id, "errors": sent_errors[:3]})
        else:
            valid += 1

    total = valid + invalid
    report = {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_pct": (valid / max(total, 1)) * 100,
        "errors": errors,
    }

    print(f"Total : {total}, Valides : {valid}, Invalides : {invalid} ({report['valid_pct']:.1f}%)")
    for ex in errors:
        print(f"  [{ex['id']}] {ex['errors']}")

    return report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated_spans_fixed.json"
    validate_cir(path)
