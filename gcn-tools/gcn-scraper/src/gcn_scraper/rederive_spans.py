"""
Re-derive les token_span des nœuds CIR depuis spaCy.

Stratégie de détection des clauses :
  Une clause = le sous-arbre d'un token dont dep_rel est dans
  {"root", "advcl", "ccomp", "relcl", "acl"}.
  Les clauses sont triées par position de leur token racine dans la phrase.

Mapping CIR nodes → clauses détectées :
  Les nœuds CIR sont supposés être ordonnés par position dans le texte.
  Le nœud CIR i est mappé à la clause i (0-indexed).
  Si n_nodes_cir != n_clauses_detected : la phrase est marquée "ambiguë"
  et exclue du dataset (token_span non fiable).

Résultat : token_span = [min_tok_id, max_tok_id] du sous-arbre de la clause,
  avec tok_id = token.i + 1 (convention 1-based, compatible avec json_reader.py).
"""
import json
import spacy
from typing import Optional


CLAUSE_DEPS = {"root", "advcl", "ccomp", "relcl", "acl"}


def _detect_clauses(doc) -> list[tuple[int, int]]:
    """
    Retourne les spans (start_1based, end_1based) de chaque clause détectée,
    triés par position croissante dans la phrase.
    """
    clauses = []
    for tok in doc:
        if tok.dep_.lower() in CLAUSE_DEPS:
            subtree_ids = [t.i + 1 for t in tok.subtree]  # 1-based
            if subtree_ids:
                clauses.append((min(subtree_ids), max(subtree_ids)))
    clauses = sorted(set(clauses), key=lambda c: c[0])
    return clauses


def rederive_cir_spans(
    sentence: dict,
    doc,
) -> Optional[dict]:
    """
    Prend une sentence (avec CIR existant) et un doc spaCy.
    Retourne la sentence avec les token_span des nœuds CIR corrigés,
    ou None si la re-dérivation est impossible (ambiguïté de mapping).

    Les types de nœuds et les relations d'arêtes sont conservés intacts.
    """
    nodes = sentence.get("cir", {}).get("nodes", [])
    if not nodes:
        return None

    clauses = _detect_clauses(doc)

    if len(clauses) != len(nodes):
        return None

    corrected = dict(sentence)
    corrected["cir"] = dict(sentence["cir"])
    corrected["cir"]["nodes"] = []
    for node, (span_start, span_end) in zip(nodes, clauses):
        corrected_node = dict(node)
        corrected_node["token_span"] = [span_start, span_end]
        corrected["cir"]["nodes"].append(corrected_node)

    return corrected


def rederive_all_spans(annotated_path: str, output_path: str) -> dict:
    """
    Lit annotated.json, re-dérive les token_span, écrit le fichier corrigé.
    Rapport : nb phrases corrigées, nb exclues (ambiguïté), nb ignorées (0 nœuds).
    """
    nlp = spacy.load("fr_core_news_sm")
    data = json.loads(open(annotated_path).read())
    sents = data["document"]["sentences"]

    corrected, excluded, empty = [], [], []
    for s in sents:
        if not s.get("cir", {}).get("nodes"):
            empty.append(s)
            continue
        doc = nlp(s["text"])
        result = rederive_cir_spans(s, doc)
        if result is None:
            excluded.append(s.get("id", "?"))
        else:
            corrected.append(result)

    report = {
        "corrected": len(corrected),
        "excluded": len(excluded),
        "empty": len(empty),
        "excluded_ids": excluded[:20],
    }

    print(f"Corrigées : {len(corrected)}, Exclues : {len(excluded)}, Vides : {len(empty)}")
    if excluded:
        print(f"Phrases exclues (ambiguïté) : {excluded[:10]}")

    output = dict(data)
    output["document"]["sentences"] = corrected
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Écrit : {output_path}")

    return report


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "gcn-datasets/real/annotated_spans_fixed.json"
    rederive_all_spans(inp, out)
