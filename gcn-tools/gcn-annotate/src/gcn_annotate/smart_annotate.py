# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""
Annotateur causal intelligent — FR/EN.

Produit des annotations CIR avec les 7 types de nœuds et les 11 relations,
en utilisant l'analyse syntaxique spaCy + règles linguistiques.

Logique :
  1. Détecter le connecteur causal et sa relation
  2. Segmenter la phrase en clauses src/dst autour du connecteur
  3. Classifier chaque clause en l'un des 7 NodeTypes via des heuristiques
     syntaxiques et lexicales
  4. Produire le JSON gcn-nl complet avec token_span 1-based
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import Counter

import spacy


# ---------------------------------------------------------------------------
# Connecteurs → relations (FR + EN)
# ---------------------------------------------------------------------------

CONNECTORS: list[tuple[str, str]] = [
    # cause
    ("parce que",       "cause"),
    ("parce qu",        "cause"),
    ("car ",            "cause"),
    ("puisque",         "cause"),
    ("du fait que",     "cause"),
    ("en raison de",    "cause"),
    ("à cause de",      "cause"),
    ("a cause de",      "cause"),
    ("grâce à",         "cause"),
    ("grace a",         "cause"),
    ("étant donné",     "cause"),
    ("vu que",          "cause"),
    ("because ",        "cause"),
    ("since ",          "cause"),
    ("as ",             "cause"),
    ("owing to",        "cause"),
    ("due to",          "cause"),
    # enable
    ("permet ",         "enable"),
    ("permet de",       "enable"),
    ("rend possible",   "enable"),
    ("favorise",        "enable"),
    ("facilite",        "enable"),
    ("contribue à",     "enable"),
    ("enables",         "enable"),
    ("allows",          "enable"),
    ("facilitates",     "enable"),
    # prevent
    ("empêche",         "prevent"),
    ("interdit",        "prevent"),
    ("bloque",          "prevent"),
    ("limite",          "prevent"),
    ("réduit",          "prevent"),
    ("prevents",        "prevent"),
    ("inhibits",        "prevent"),
    ("reduces",         "prevent"),
    # condition
    ("si ",             "condition"),
    ("à condition que", "condition"),
    ("pourvu que",      "condition"),
    ("sous réserve",    "condition"),
    ("if ",             "condition"),
    ("provided that",   "condition"),
    ("unless",          "condition"),
    ("when ",           "condition"),
    ("whenever",        "condition"),
    # concession
    ("bien que",        "concession"),
    ("même si",         "concession"),
    ("malgré",          "concession"),
    ("quoique",         "concession"),
    ("cependant",       "concession"),
    ("néanmoins",       "concession"),
    ("pourtant",        "concession"),
    ("toutefois",       "concession"),
    ("although",        "concession"),
    ("even though",     "concession"),
    ("despite",         "concession"),
    ("however",         "concession"),
    ("nevertheless",    "concession"),
    # sequence
    ("puis ",           "sequence"),
    ("ensuite",         "sequence"),
    ("alors ",          "sequence"),
    ("après quoi",      "sequence"),
    ("d'abord",         "sequence"),
    ("en premier lieu", "sequence"),
    ("then ",           "sequence"),
    ("subsequently",    "sequence"),
    ("afterwards",      "sequence"),
    ("following",       "sequence"),
    # motivation
    ("afin de",         "motivation"),
    ("pour que",        "motivation"),
    ("dans le but de",  "motivation"),
    ("en vue de",       "motivation"),
    ("so that",         "motivation"),
    ("in order to",     "motivation"),
    ("so as to",        "motivation"),
    # opposition
    ("alors que",       "opposition"),
    ("tandis que",      "opposition"),
    ("au contraire",    "opposition"),
    ("en revanche",     "opposition"),
    ("whereas",         "opposition"),
    ("while ",          "opposition"),
    ("on the contrary", "opposition"),
    ("in contrast",     "opposition"),
    # filter
    ("à moins que",     "filter"),
    ("sauf si",         "filter"),
    ("excepté si",      "filter"),
    ("except if",       "filter"),
    ("unless ",         "filter"),
]


# ---------------------------------------------------------------------------
# Heuristiques de classification NodeType
# ---------------------------------------------------------------------------

# Verbes d'état (→ etat)
STATE_VERBS = {
    "être", "avoir", "rester", "demeurer", "paraître", "sembler", "devenir",
    "be", "remain", "stay", "seem", "appear",
}

# Verbes de transition/changement (→ transition)
TRANSITION_VERBS = {
    "devenir", "changer", "évoluer", "passer", "transformer", "modifier",
    "become", "change", "evolve", "transform", "shift", "turn",
}

# Verbes d'action explicites (→ action)
ACTION_VERBS = {
    "faire", "créer", "produire", "construire", "mettre", "prendre",
    "réaliser", "effectuer", "appliquer", "lancer", "décider", "choisir",
    "make", "create", "build", "produce", "apply", "launch", "decide",
    "execute", "run", "deploy", "implement",
}

# Verbes de processus (→ processus — défaut pour verbes génériques)
# Patrons lexicaux etat_systemique
SYSTEMIC_NOUNS = {
    "loi", "règle", "norme", "politique", "système", "protocole", "procédure",
    "mécanisme", "régulation", "réglementation", "standard", "principe",
    "law", "rule", "norm", "policy", "system", "protocol", "regulation",
    "mechanism", "standard", "principle", "framework",
}


def _classify_clause(doc_tokens: list, root_token) -> str:
    """
    Classifie une clause spaCy en l'un des 7 NodeTypes.

    Ordre de priorité :
    1. condition  — si le token racine est SCONJ "si"/"if"
    2. entite     — pas de verbe dans la clause
    3. etat_systemique — nom systémique dans la clause
    4. etat       — verbe d'état
    5. transition — verbe de transition
    6. action     — verbe d'action
    7. processus  — défaut
    """
    if root_token is None:
        return "entite"

    lemma = root_token.lemma_.lower()
    pos   = root_token.pos_

    # 1. Condition : clause introduite par un SCONJ conditionnel
    if pos in {"SCONJ"} and lemma in {"si", "if", "pourvu", "provided"}:
        return "condition"

    # 2. Entité : pas de verbe conjugué dans la clause
    has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc_tokens)
    if not has_verb:
        return "entite"

    # 3. Etat systémique : nom système + copule
    clause_lemmas = {t.lemma_.lower() for t in doc_tokens}
    if clause_lemmas & SYSTEMIC_NOUNS:
        return "etat_systemique"

    # 4. État : verbe d'état
    if lemma in STATE_VERBS and lemma not in TRANSITION_VERBS:
        return "etat"

    # 5. Transition
    if lemma in TRANSITION_VERBS:
        return "transition"

    # 6. Action
    if lemma in ACTION_VERBS:
        return "action"

    # 7. Processus (défaut)
    return "processus"


def _find_connector(text_lower: str) -> tuple[str, str] | None:
    """Retourne (connecteur, relation) ou None."""
    for conn, rel in CONNECTORS:
        if conn in text_lower:
            return conn, rel
    return None


def _split_at_connector(text: str, connector: str) -> tuple[str, str]:
    """Sépare la phrase en src/dst autour du connecteur."""
    idx = text.lower().find(connector)
    if idx == -1:
        mid = len(text) // 2
        return text[:mid].strip(), text[mid:].strip()
    before = text[:idx].strip().rstrip(",;: ")
    after  = text[idx + len(connector):].strip()
    # Si "before" est vide (connecteur en début de phrase), inverser
    if not before:
        return after, before
    return before, after


def _make_node(node_id: str, node_type: str, label: str,
               span_start: int, span_end: int) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "label": label,
        "token_span": [span_start, span_end],
        "origin": "explicit",
        "scope": "specific",
        "temporal_index": int(node_id[1:]) - 1,
        "modifiers": [],
        "attributes": {},
    }


def _make_edge(src: str, dst: str, relation: str) -> dict:
    return {
        "source": src,
        "target": dst,
        "relation": relation,
        "attributes": {"confidence": 0.9, "explicit": True, "negated": False},
    }


def annotate_sentence(text: str, nlp) -> dict | None:
    """
    Annote une phrase et retourne un dict CIR, ou None si pas de relation causale.
    """
    text = text.strip()
    if len(text.split()) < 6:
        return None

    result = _find_connector(text.lower())
    if result is None:
        return None
    connector, relation = result

    src_text, dst_text = _split_at_connector(text, connector)
    if not src_text or not dst_text:
        return None

    # Analyser chaque clause avec spaCy
    doc_src = nlp(src_text)
    doc_dst = nlp(dst_text)
    doc_full = nlp(text)

    # Root token de chaque clause
    root_src = next((t for t in doc_src if t.dep_.lower() == "root"), None)
    root_dst = next((t for t in doc_dst if t.dep_.lower() == "root"), None)

    # Classifier les types de nœuds
    type_src = _classify_clause(list(doc_src), root_src)
    type_dst = _classify_clause(list(doc_dst), root_dst)

    # Trouver les positions 1-based des tokens dans la phrase complète
    tokens_full = list(doc_full)
    T = len(tokens_full)

    # Allouer les spans : chercher où src et dst apparaissent dans doc_full
    connector_idx = text.lower().find(connector)
    tokens_before = [t for t in tokens_full if t.idx < connector_idx]
    tokens_after  = [t for t in tokens_full if t.idx >= connector_idx + len(connector)]

    src_start = 1
    src_end   = max(1, len(tokens_before))
    dst_start = min(T, src_end + 2)
    dst_end   = T

    # Label = lemme du root
    label_src = root_src.lemma_ if root_src else src_text.split()[0]
    label_dst = root_dst.lemma_ if root_dst else dst_text.split()[0]

    # Causal pattern = relation
    return {
        "id": "",  # sera rempli par le caller
        "text": text,
        "causal_pattern": relation,
        "cir": {
            "nodes": [
                _make_node("n001", type_src, label_src, src_start, src_end),
                _make_node("n002", type_dst, label_dst, dst_start, dst_end),
            ],
            "edges": [
                _make_edge("n001", "n002", relation),
            ],
        },
    }


def annotate_file(input_path: Path, output_path: Path,
                  lang: str = "fr", max_sentences: int = 5000) -> dict:
    """
    Lit un fichier texte (une phrase par ligne) et produit un JSON gcn-nl.
    Retourne un rapport {total, annotated, skipped, distribution}.
    """
    model = {"fr": "fr_core_news_sm", "en": "en_core_web_sm"}.get(lang, "fr_core_news_sm")
    try:
        nlp = spacy.load(model)
    except OSError:
        nlp = spacy.load("fr_core_news_sm")

    lines = [l.strip() for l in Path(input_path).read_text("utf-8").splitlines() if l.strip()]
    total = min(len(lines), max_sentences)

    sentences = []
    rel_counts: Counter = Counter()
    node_counts: Counter = Counter()
    skipped = 0

    for i, line in enumerate(lines[:total]):
        ann = annotate_sentence(line, nlp)
        if ann is None:
            skipped += 1
            continue
        ann["id"] = f"s{len(sentences)+1:04d}"
        sentences.append(ann)
        rel_counts[ann["cir"]["edges"][0]["relation"]] += 1
        for n in ann["cir"]["nodes"]:
            node_counts[n["type"]] += 1

    output = {"document": {"sentences": sentences}}
    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "total": total,
        "annotated": len(sentences),
        "skipped": skipped,
        "relations": dict(rel_counts),
        "node_types": dict(node_counts),
    }

    print(f"Phrases traitées : {total}")
    print(f"Annotées : {len(sentences)}  ({100*len(sentences)//max(total,1)}%)")
    print(f"Ignorées  : {skipped}")
    print("\nDistribution relations :")
    for k, v in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:22s}: {v}")
    print("\nDistribution node types :")
    for k, v in sorted(node_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:22s}: {v}")
    print(f"\nFichier : {output_path}")

    return report


if __name__ == "__main__":
    inp  = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/raw/phase4/sentences_raw.txt"
    out  = sys.argv[2] if len(sys.argv) > 2 else "gcn-datasets/raw/phase4/annotated_smart.json"
    lang = sys.argv[3] if len(sys.argv) > 3 else "fr"
    annotate_file(Path(inp), Path(out), lang=lang, max_sentences=15000)
