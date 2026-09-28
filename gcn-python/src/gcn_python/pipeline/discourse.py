# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Éq.10 ETUDE — extraction de relations discursives inter-phrasales.

Opère sur des paires de phrases consécutives (Φᵢ₋₁, Φᵢ).
Couche 1 (structurelle) : signal POS + position (CCONJ/ADV, id≤2) sans lemme.
Couche 2 (si LangMarkers fourni) : matching préfixe sur discourse_connectors.
"""
from __future__ import annotations

_DISCOURSE_POS = frozenset({"CCONJ", "ADV"})
_ANAPHORA_RELS = frozenset({"nsubj", "expl"})


def _pivot_node_id(cir: dict) -> int | None:
    """Nœud pivot = dernier nœud par position token_span.end."""
    nodes = cir.get("nodes", [])
    if not nodes:
        return None
    return max(
        nodes,
        key=lambda n: (n.get("source_span", {})
                        .get("token_span", {})
                        .get("end", 0)),
    )["id"]


def _detect_connector(tokens: list[dict], lang_markers) -> tuple[str | None, float, bool]:
    """Retourne (relation, confidence, inverted) ou (None, 0.0, False).

    Couche 1 : premier token CCONJ/ADV avec id≤2 → signal structurel.
    Couche 2 : matching préfixe sur discourse_connectors si lang_markers fourni.
    """
    if not tokens:
        return (None, 0.0, False)

    # Couche 2 — préfixe des 3 premiers lemmes
    if lang_markers is not None and getattr(lang_markers, "discourse_connectors", {}):
        prefix = " ".join(
            t.get("lemma", "").lower()
            for t in tokens
            if t.get("id", 99) <= 3
        )
        for _cat, entry in lang_markers.discourse_connectors.items():
            for lemma in entry.get("lemmas", []):
                if prefix.startswith(lemma):
                    # mapper "cause_inverted" → "cause" avec inverted=True
                    relation = _cat.replace("_inverted", "")
                    return (relation, float(entry.get("confidence", 0.75)),
                            bool(entry.get("inverted", False)))

    # Couche 1 — signal structurel seul (sans lemme)
    for tok in tokens:
        if tok.get("pos") in _DISCOURSE_POS and tok.get("id", 99) <= 2:
            # CCONJ → cause probable, ADV → sequence probable, conf modérée
            if tok.get("pos") == "CCONJ":
                return ("cause", 0.50, False)
            return ("sequence", 0.50, False)

    return (None, 0.0, False)


def _detect_anaphora(tokens: list[dict]) -> bool:
    """Vrai si le premier token pertinent est un pronom sujet (anaphore probable)."""
    for tok in tokens[:3]:
        if tok.get("pos") == "PRON" and tok.get("dep_rel") in _ANAPHORA_RELS:
            return True
    return False


def extract_discourse_relation(
    cir_prev: dict | None,
    cir_curr: dict,
    reps_curr: list,
    lang_markers=None,
) -> list | None:
    """Éq.10 — extrait une relation discursive entre Φᵢ₋₁ et Φᵢ.

    Returns:
        [src_pivot_id, dst_pivot_id, edge_dict] compatible ir_emitter, ou None.
    cir_prev : None ou CIR vide → None retourné.
    reps_curr : list[UDRepresentation] depuis pipeline._cached_reps.
    """
    if not cir_prev or not cir_prev.get("nodes"):
        return None
    if not cir_curr.get("nodes"):
        return None

    tokens_curr = reps_curr[0].tokens if reps_curr else []

    relation, confidence, inverted = _detect_connector(tokens_curr, lang_markers)

    if relation is None:
        # Anaphore — conf 0.50
        if _detect_anaphora(tokens_curr):
            relation, confidence, inverted = "cause", 0.50, False
        else:
            # Juxtaposition — conf 0.35
            relation, confidence, inverted = "cause", 0.35, False

    src_id = _pivot_node_id(cir_prev)
    dst_id = _pivot_node_id(cir_curr)
    if src_id is None or dst_id is None:
        return None

    if inverted:
        src_id, dst_id = dst_id, src_id

    import datetime as _dt
    try:
        from .. import __version__ as _ver
    except Exception:
        _ver = "unknown"

    edge_dict = {
        "relation": relation,
        "confidence": confidence,
        "confidence_ml": confidence,
        "confidence_d6": confidence,
        "temporal_gap": None,
        "explicit": False,
        "negated": False,
        "source_polarity": None,
        "marker_token": None,
        "third": None,
        "joint_group_id": None,
        "ambiguous": False,
        "candidates": None,
        "in_cycle": None,
        "provenance": {
            "ref": None,
            "span": "discourse",
            "extraction_method": "discourse_eq10",
            "model_version": _ver,
            "extracted_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    return [src_id, dst_id, edge_dict]
