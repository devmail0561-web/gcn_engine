# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""
Re-dérive les token_span des nœuds CIR depuis spaCy.

Stratégies (appliquées dans l'ordre jusqu'au premier succès) :

S1 — Sous-arbres de clauses syntaxiques (CLAUSE_DEPS).
     Fonctionne quand n_spacy_clauses == n_cir_nodes.

S2 — Segmentation par connecteurs (SCONJ/CCONJ).
     Le LLM segmente souvent aux connecteurs plutôt qu'aux limites syntaxiques.
     Découpe la phrase aux positions des SCONJ/CCONJ pour obtenir n+1 segments.
     Fonctionne quand n_connectors + 1 == n_cir_nodes.

S3 — Division positionnelle uniforme.
     Dernier recours : répartit les T tokens en N segments de taille T//N.
     Imprécis mais produit des spans distincts et non-nuls.
     Toujours accepté si n_cir_nodes >= 1.

Résultat : token_span = [start_1based, end_1based], convention 1-based.
"""
import json
import spacy
from pathlib import Path
from typing import Optional


CLAUSE_DEPS = {"root", "advcl", "ccomp", "relcl", "acl"}


def _detect_clauses_s1(doc, n_nodes: int) -> Optional[list[tuple[int, int]]]:
    """S1 : sous-arbres syntaxiques — exact si count == n_nodes."""
    clauses = []
    for tok in doc:
        if tok.dep_.lower() in CLAUSE_DEPS:
            ids = [t.i + 1 for t in tok.subtree]
            if ids:
                clauses.append((min(ids), max(ids)))
    clauses = sorted(set(clauses), key=lambda c: c[0])
    return clauses if len(clauses) == n_nodes else None


def _detect_clauses_s2(doc, n_nodes: int) -> Optional[list[tuple[int, int]]]:
    """S2 : segmentation aux connecteurs SCONJ/CCONJ."""
    tokens = list(doc)
    T = len(tokens)
    if T == 0:
        return None
    # Trouver les positions des connecteurs (frontières de segments)
    boundaries = [0] + [
        tok.i for tok in tokens
        if tok.pos_ in {"SCONJ", "CCONJ"} and tok.i > 0
    ] + [T]
    # Dédupliquer et trier
    boundaries = sorted(set(boundaries))
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1      # 1-based
        end   = boundaries[i + 1]      # 1-based inclusive
        if start <= end:
            segments.append((start, end))
    return segments if len(segments) == n_nodes else None


def _detect_clauses_s3(doc, n_nodes: int) -> list[tuple[int, int]]:
    """S3 : division positionnelle uniforme — toujours applicable, spans valides.

    Les spans sont clampées dans [1, T] : si n_nodes > T, les derniers nœuds
    reçoivent le dernier token (spans dupliquées mais valides, start <= end).
    """
    T = len(doc)
    if T == 0 or n_nodes == 0:
        return [(1, 1)] * max(1, n_nodes)
    size = max(1, T // n_nodes)
    spans = []
    for i in range(n_nodes):
        start = min(i * size + 1, T)                      # 1-based
        end = min((i + 1) * size if i < n_nodes - 1 else T, T)  # 1-based
        spans.append((start, max(start, end)))
    return spans


def rederive_cir_spans(sentence: dict, doc) -> Optional[dict]:
    """
    Re-dérive les token_span des nœuds CIR. Essaie S1 → S2 → S3.
    Retourne None uniquement si le CIR n'a aucun nœud.
    """
    nodes = sentence.get("cir", {}).get("nodes", [])
    if not nodes:
        return None

    n = len(nodes)
    spans = (
        _detect_clauses_s1(doc, n)
        or _detect_clauses_s2(doc, n)
        or _detect_clauses_s3(doc, n)
    )

    corrected = dict(sentence)
    corrected["cir"] = dict(sentence["cir"])
    corrected["cir"]["nodes"] = []
    for node, (s, e) in zip(nodes, spans):
        corrected_node = dict(node)
        corrected_node["token_span"] = [s, e]
        corrected["cir"]["nodes"].append(corrected_node)

    return corrected


def rederive_all_spans(annotated_path: str, output_path: str) -> dict:
    """
    Lit annotated.json, re-dérive les token_span, écrit le fichier corrigé.
    Rapport : nb phrases corrigées, nb exclues (ambiguïté), nb vides (0 nœud,
    conservées telles quelles dans la sortie).
    """
    nlp = spacy.load("fr_core_news_sm")
    data = json.loads(Path(annotated_path).read_text(encoding="utf-8"))
    sents = data["document"]["sentences"]

    corrected, excluded, empty = [], [], []
    for s in sents:
        if not s.get("cir", {}).get("nodes"):
            empty.append(s)
            corrected.append(s)  # conservée telle quelle, pas jetée (Audit 3 C7)
            continue
        doc = nlp(s["text"])
        result = rederive_cir_spans(s, doc)
        if result is None:
            excluded.append(s.get("id", "?"))
            corrected.append(s)  # conservée avec spans d'origine
        else:
            corrected.append(result)

    report = {
        "corrected": len(corrected) - len(empty) - len(excluded),
        "preserved": len(empty) + len(excluded),
        "excluded": len(excluded),
        "empty": len(empty),
        "excluded_ids": excluded[:20],
    }

    print(f"Corrigées : {len(corrected)}, Exclues : {len(excluded)}, Vides : {len(empty)}")
    if excluded:
        print(f"Phrases exclues (ambiguïté) : {excluded[:10]}")

    import copy
    output = copy.deepcopy(data)
    output["document"]["sentences"] = corrected
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Écrit : {output_path}")

    return report


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "gcn-datasets/real/annotated_spans_fixed.json"
    rederive_all_spans(inp, out)
