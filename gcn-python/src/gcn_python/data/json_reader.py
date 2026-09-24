# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import warnings
from .schema import SentenceRecord, TokenRecord, ClauseRecord, EdgeRecord
from ..constants import RELATION_TYPES


def load_sentences(path: Path, silver_weight: float = 1.0) -> list[SentenceRecord]:
    """Charge un fichier JSON GCN-NL → List[SentenceRecord].

    silver_weight (Amélioration F) : poids appliqué aux phrases silver
    (champ _methode commençant par "silver-"). gold (absent) = 1.0.
    1.0 = rétrocompatible (aucun effet).
    """
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
        return [_parse_paper_example(ex) for ex in doc["examples"] if "expected_cir" in ex]

    # Format dataset (document.sentences)
    if "document" in doc:
        doc_lang = doc["document"].get("lang", "")
        sentences = doc["document"].get("sentences") or []
        return [_parse_dataset_sentence(s, doc_lang, silver_weight) for s in sentences if "cir" in s]

    # M-PL : Format non reconnu (possiblement "snippets" pour gcn-pl)
    warnings.warn(
        f"Format JSON non reconnu dans {path.name} (ni 'examples', ni 'document') — fichier ignoré. "
        f"Les fichiers gcn-pl (snippets) ne sont pas supportés par le chargeur ML.",
        UserWarning,
        stacklevel=2,
    )
    return []


def load_all_sentences(data_dir: Path, silver_weight: float = 1.0) -> list[SentenceRecord]:
    """Charge tous les fichiers JSON d'un répertoire."""
    records = []
    for p in sorted(data_dir.glob("*.json")):
        records.extend(load_sentences(p, silver_weight))
    return records


def _parse_paper_example(ex: dict) -> SentenceRecord:
    cir = ex.get("expected_cir", {})
    clauses = [_parse_clause_node(n) for n in cir.get("nodes", [])]
    edges = [_parse_edge(e) for e in cir.get("edges", [])]
    return SentenceRecord(
        id=ex.get("id", ""),
        text=ex.get("text", ""),
        tokens=[],
        clauses=clauses,
        edges=edges,
    )


def _parse_dataset_sentence(s: dict, lang: str = "", silver_weight: float = 1.0) -> SentenceRecord:
    tokens = [_parse_token(t) for t in s.get("tokens", [])]
    cir = s.get("cir", {})
    clauses = [_parse_clause_node(n) for n in cir.get("nodes", [])]
    edges = [_parse_edge(e) for e in cir.get("edges", [])]
    # Amélioration F : pondération par confiance d'annotation via _methode
    methode = str(s.get("_methode", "gold"))
    weight = silver_weight if methode.startswith("silver") else 1.0
    return SentenceRecord(
        id=s.get("id", ""),
        text=s.get("text", ""),
        lang=lang,
        tokens=tokens,
        clauses=clauses,
        edges=edges,
        causal_pattern=s.get("causal_pattern", ""),
        weight=weight,
    )


def _safe_int(val: Any, default: int = 0) -> int:
    """Convertit une valeur en int de manière sûre (retourne default si échec)."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _parse_token(t: dict) -> TokenRecord:
    gcn = t.get("gcn", {}) or {}
    morph_raw = t.get("morph") or {}
    return TokenRecord(
        id=_safe_int(t.get("id"), 0),
        form=t.get("form", ""),
        lemma=t.get("lemma", ""),
        pos=t.get("pos", ""),
        dep_rel=t.get("dep_rel", ""),
        dep_head=_safe_int(t.get("dep_head"), 0),
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
        temporal_index=_safe_int(n.get("temporal_index") or attrs.get("temporal_index"), 0),
        origin=n.get("origin", "explicit"),
        attributes=attrs,
        modifiers=n.get("modifiers", []) or [],
    )


def _parse_edge(e: dict) -> EdgeRecord:
    attrs = e.get("attributes") or {}
    relation = e.get("relation") or e.get("relation_type") or attrs.get("relation") or ""
    if not relation:
        raise ValueError(
            f"Arête {e.get('source', e.get('sources', '?'))}→{e.get('target', '?')} "
            f"sans champ 'relation' — arête ignorée. "
            f"Vérifier l'annotation (relations valides : {RELATION_TYPES})."
        )
    # sources prioritaire, wrap source, warn si conflit
    sources = e.get("sources")
    legacy_source = e.get("source", "")
    if sources is not None and legacy_source and list(sources) != [legacy_source]:
        warnings.warn(
            f"Arête {legacy_source}→{e.get('target', '?')} : 'sources' {sources} "
            f"et 'source' {legacy_source!r} en conflit — 'sources' gagne.",
            UserWarning, stacklevel=3,
        )
    if sources is None:
        sources = [legacy_source] if legacy_source else []
    sources = [str(s) for s in sources]
    target = str(e.get("target", ""))
    # confidence absente -> None + warn (jamais 0.0 / 1.0 silencieux)
    if "confidence" in attrs or "confidence" in e:
        conf_raw = attrs.get("confidence", e.get("confidence"))
        try:
            confidence = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            warnings.warn("confidence invalide — None appliqué.", UserWarning, stacklevel=3)
            confidence = None
    else:
        confidence = None  # N-3 : absence normale, pas de warning (28% des arêtes)
    # negated absent -> None (détection pipeline via root_morph)
    if "negated" in attrs or "negated" in e:
        negated = bool(attrs.get("negated", e.get("negated", False)))
    else:
        negated = None
    _exp_raw = attrs.get("explicit", e.get("explicit"))
    explicit = bool(_exp_raw) if _exp_raw is not None else None  # N-4 : None si absent
    return EdgeRecord(
        source=sources[0] if sources else "",
        target=target,
        relation=relation,
        confidence=confidence,
        explicit=explicit,
        negated=negated,
        marker_token=attrs.get("marker_token", e.get("marker_token")),
        sources=sources,
    )
