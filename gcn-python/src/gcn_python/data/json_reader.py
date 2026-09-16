from __future__ import annotations
from pathlib import Path
import json
import warnings
from .schema import SentenceRecord, TokenRecord, ClauseRecord, EdgeRecord


def load_sentences(path: Path, lang: str = "fr") -> list[SentenceRecord]:
    """Charge un fichier JSON GCN-NL → List[SentenceRecord]."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return []

    # Format paper_examples.json (legacy — préférer le format 'document')
    if "examples" in doc:
        warnings.warn(
            f"Format legacy 'examples' détecté dans {path.name}. "
            f"Migrer vers le format 'document' (GCN-NL standard).",
            DeprecationWarning,
            stacklevel=2,
        )
        return [_parse_paper_example(ex, lang) for ex in doc["examples"] if "expected_cir" in ex]

    # Format dataset (document.sentences)
    if "document" in doc:
        doc_lang = doc["document"].get("lang", lang)
        sentences = doc["document"].get("sentences") or []
        return [_parse_dataset_sentence(s, doc_lang) for s in sentences if "cir" in s]

    # M-PL : Format non reconnu (possiblement "snippets" pour gcn-pl)
    warnings.warn(
        f"Format JSON non reconnu dans {path.name} (ni 'examples', ni 'document') — fichier ignoré. "
        f"Les fichiers gcn-pl (snippets) ne sont pas supportés par le chargeur ML.",
        UserWarning,
        stacklevel=2,
    )
    return []


def load_all_sentences(data_dir: Path, lang: str = "fr") -> list[SentenceRecord]:
    """Charge tous les fichiers JSON d'un répertoire."""
    records = []
    for p in sorted(data_dir.glob("*.json")):
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
    node_type = n.get("type") or ""
    if not node_type:
        warnings.warn(
            f"Nœud {n.get('id', '?')} sans champ 'type' — défaut 'action' appliqué.",
            UserWarning, stacklevel=3,
        )
        node_type = "action"
    return ClauseRecord(
        node_id=n.get("id", ""),
        node_type=node_type,
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
    relation = e.get("relation") or ""
    if not relation:
        warnings.warn(
            f"Arête {e.get('source', '?')}→{e.get('target', '?')} sans champ 'relation' "
            f"— défaut 'cause' appliqué.",
            UserWarning, stacklevel=3,
        )
        relation = "cause"
    return EdgeRecord(
        source=e.get("source", ""),
        target=e.get("target", ""),
        relation=relation,
        confidence=float(attrs.get("confidence", e.get("confidence", 1.0))),
        explicit=bool(attrs.get("explicit", e.get("explicit", True))),
        negated=bool(attrs.get("negated", e.get("negated", False))),
        marker_token=attrs.get("marker_token", e.get("marker_token")),
    )
