from __future__ import annotations
from pathlib import Path
import yaml
from .schema import SentenceRecord, TokenRecord, ClauseRecord, EdgeRecord


def load_sentences(path: Path, lang: str = "fr") -> list[SentenceRecord]:
    """Charge un fichier YAML GCN-NL → List[SentenceRecord]."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return []

    # Format paper_examples.yaml
    if "examples" in doc:
        return [_parse_paper_example(ex, lang) for ex in doc["examples"] if "expected_cir" in ex]

    # Format dataset (document.sentences)
    if "document" in doc:
        doc_lang = doc["document"].get("lang", lang)
        sentences = doc["document"].get("sentences") or []
        return [_parse_dataset_sentence(s, doc_lang) for s in sentences if "cir" in s]

    return []


def load_all_sentences(data_dir: Path, lang: str = "fr") -> list[SentenceRecord]:
    """Charge tous les fichiers YAML d'un répertoire."""
    records = []
    for p in sorted(data_dir.glob("*.yaml")):
        records.extend(load_sentences(p, lang))
    return records


def _parse_paper_example(ex: dict, lang: str) -> SentenceRecord:
    cir = ex.get("expected_cir", {})
    clauses = [_parse_clause_node(n) for n in cir.get("nodes", [])]
    edges = [_parse_edge(e) for e in cir.get("edges", [])]
    return SentenceRecord(
        id=ex.get("id", ""),
        text=ex.get("text", ""),
        lang=lang,
        tokens=[],
        clauses=clauses,
        edges=edges,
    )


def _parse_dataset_sentence(s: dict, lang: str) -> SentenceRecord:
    tokens = [_parse_token(t) for t in s.get("tokens", [])]
    cir = s.get("cir", {})
    clauses = [_parse_clause_node(n) for n in cir.get("nodes", [])]
    edges = [_parse_edge(e) for e in cir.get("edges", [])]
    return SentenceRecord(
        id=s.get("id", ""),
        text=s.get("text", ""),
        lang=lang,
        tokens=tokens,
        clauses=clauses,
        edges=edges,
    )


def _parse_token(t: dict) -> TokenRecord:
    gcn = t.get("gcn", {}) or {}
    morph_raw = t.get("morph") or {}
    return TokenRecord(
        id=int(t.get("id", 0)),
        form=t.get("form", ""),
        lemma=t.get("lemma", ""),
        pos=t.get("pos", ""),
        dep_rel=t.get("dep_rel", ""),
        dep_head=int(t.get("dep_head", 0)),
        morph=morph_raw if isinstance(morph_raw, dict) else {},
        gcn_causal_type=gcn.get("causal_type"),
        gcn_causal_class=gcn.get("causal_class"),
    )


def _parse_clause_node(n: dict) -> ClauseRecord:
    span = n.get("token_span", [0, 0])
    attrs = dict(n.get("attributes", {}) or {})
    for f in ("entity", "quality", "agent", "patient", "agent_type", "temporal_index", "scope"):
        if f in n and f not in attrs:
            attrs[f] = n[f]
    return ClauseRecord(
        node_id=n.get("id", ""),
        node_type=n.get("type", "action"),
        label=n.get("label", ""),
        token_span=(span[0], span[1]) if len(span) >= 2 else (0, 0),
        scope=n.get("scope", attrs.get("scope", "specific")),
        temporal_index=int(n.get("temporal_index", attrs.get("temporal_index", 0))),
        origin=n.get("origin", "explicit"),
        attributes=attrs,
        modifiers=n.get("modifiers", []) or [],
    )


def _parse_edge(e: dict) -> EdgeRecord:
    attrs = e.get("attributes") or {}
    return EdgeRecord(
        source=e.get("source", ""),
        target=e.get("target", ""),
        relation=e.get("relation", "cause"),
        confidence=float(attrs.get("confidence", e.get("confidence", 1.0))),
        explicit=bool(attrs.get("explicit", e.get("explicit", True))),
        negated=bool(attrs.get("negated", e.get("negated", False))),
        marker_token=attrs.get("marker_token", e.get("marker_token")),
    )
