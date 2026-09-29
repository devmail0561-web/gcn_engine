# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Normaliseur unifié des arêtes CIR (Étape 0 du plan mise-a-niveau v5).

Centralise les trois normaliseurs divergents :
- training/bootstrap.py::_normalize_edge (défauts silencieux cause/1.0)
- frontend/bridge.py::_parse_edges (int() strict, warn agrégé)
- verbalizer/instructions.py::add_cir (drop silencieux des dicts)

Règles :
- ids : f"n{int(raw):03d}" idempotent, "n001" conservé.
- confidence absent -> None (jamais 0.0), warn agrégé.
- negated absent -> None (jamais False).
- relation absente -> None + warn, edge rejetée (return None).
- sanitization anti-injection : \\n, \\r, ANSI strip via sanitize_text.
"""
from __future__ import annotations

import re
import warnings
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def sanitize_text(s: str) -> str:
    """Supprime \\n/\\r/séquences ANSI, strip. Non-linguistique, sûr."""
    if not isinstance(s, str):
        return s
    s = _ANSI_RE.sub("", s)
    return s.replace("\n", " ").replace("\r", " ").strip()


def normalize_node_id(raw: Any) -> str:
    """Normalise un id de nœud vers 'nNNN'. Idempotent.

    - int -> f"n{int:03d}"
    - str 'n001' / 'n1' -> 'n001'
    - str numérique '1' -> 'n001'
    - autre str non convertible -> conservée (sanitisée), warn.
    """
    if isinstance(raw, bool):
        raise ValueError(f"id de nœud invalide : {raw!r}")  # noqa: TRY004  # API publique : ValueError attendu par les appelants (bootstrap)
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"id de nœud négatif : {raw!r}")
        return f"n{raw:03d}"
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("id de nœud vide")
        if re.fullmatch(r"n\d+", s):
            num = s[1:]
            return f"n{int(num):03d}"
        if re.fullmatch(r"\d+", s):
            return f"n{int(s):03d}"
        try:
            return f"n{int(s):03d}"
        except (TypeError, ValueError):
            warnings.warn(
                f"edge_norm : id non-convertible {raw!r} conservé tel quel.",
                UserWarning,
                stacklevel=3,
            )
            return sanitize_text(s)
    # float entier, etc.
    try:
        return f"n{int(raw):03d}"
    except (TypeError, ValueError) as err:
        raise ValueError(f"id de nœud invalide : {raw!r}") from err


def _get_relation(attrs: dict, outer: dict) -> Any:
    for key in ("relation", "relation_type"):
        if isinstance(attrs, dict) and attrs.get(key):
            return attrs[key]
    for key in ("relation", "relation_type"):
        if isinstance(outer, dict) and outer.get(key):
            return outer[key]
    nested = (outer.get("attributes") or {}) if isinstance(outer, dict) else {}
    if isinstance(nested, dict):
        for key in ("relation", "relation_type"):
            if nested.get(key):
                return nested[key]
    return None


def _get_field(attrs: dict, outer: dict, name: str, default: Any = None) -> Any:
    if isinstance(attrs, dict) and name in attrs:
        return attrs[name]
    if isinstance(outer, dict) and name in outer:
        return outer[name]
    nested = (outer.get("attributes") or {}) if isinstance(outer, dict) else {}
    if isinstance(nested, dict) and name in nested:
        return nested[name]
    return default


def _norm_single(src_raw: Any, dst_raw: Any, attrs: dict, outer: dict,
                 joint_group_id: str | None = None) -> dict | None:
    """Normalise src/dst/attrs déjà extraits → dict canonique ou None."""
    relation = _get_relation(attrs, outer)
    if not relation:
        warnings.warn(
            "edge_norm : relation absente — arête ignorée (pas de défaut silencieux).",
            UserWarning,
            stacklevel=3,
        )
        return None
    relation = sanitize_text(str(relation))

    try:
        src = normalize_node_id(src_raw)
        dst = normalize_node_id(dst_raw)
    except ValueError:
        warnings.warn(
            f"edge_norm : id invalide src={src_raw!r} dst={dst_raw!r} — arête ignorée.",
            UserWarning,
            stacklevel=3,
        )
        return None

    _sentinel = object()
    conf_check = _get_field(attrs, outer, "confidence", default=_sentinel)
    if conf_check is _sentinel:
        warnings.warn(
            f"edge_norm : confidence absente pour {src}→{dst} — None (pas 0.0).",
            UserWarning,
            stacklevel=3,
        )
        confidence = None
    elif conf_check is None:
        confidence = None
    else:
        try:
            confidence = float(conf_check)
        except (TypeError, ValueError):
            warnings.warn(
                f"edge_norm : confidence invalide {conf_check!r} — None.",
                UserWarning,
                stacklevel=3,
            )
            confidence = None

    neg_raw = _get_field(attrs, outer, "negated", default=_sentinel)
    negated = None if neg_raw is _sentinel else bool(neg_raw)

    exp_raw = _get_field(attrs, outer, "explicit", default=_sentinel)
    explicit = True if exp_raw is _sentinel else bool(exp_raw)

    marker = _get_field(attrs, outer, "marker_token", default=None)

    result: dict = {
        "src": src,
        "dst": dst,
        "source": src,
        "target": dst,
        "sources": [src],
        "relation": relation,
        "confidence": confidence,
        "explicit": explicit,
        "negated": negated,
        "marker_token": marker,
    }
    if joint_group_id is not None:
        result["joint_group_id"] = joint_group_id
    return result


def normalize_edge(e: Any) -> dict | list[dict] | None:
    """Normalise une arête tuple [src,dst,attrs] ou dict -> dict canonique.

    Retourne None si relation absente (warn) ou format invalide.
    Pour JOINT_CAUSE/JointPrevent avec 2 sources : retourne list[dict] (2 arêtes).
    confidence absente -> None + warn. negated absent -> None.
    """
    import hashlib

    if isinstance(e, (list, tuple)) and len(e) == 3:
        src_raw, dst_raw, attrs = e
        outer: dict = {}
        if not isinstance(attrs, dict):
            warnings.warn(
                "edge_norm : attrs non-dict ignoré.",
                UserWarning,
                stacklevel=2,
            )
            return None
        src_list = [src_raw]
    elif isinstance(e, dict):
        attrs = e
        outer = e
        src_raw_field = e.get("sources", e.get("source", ""))
        dst_raw = e.get("target", e.get("dst", ""))
        if isinstance(src_raw_field, (list, tuple)):
            if not src_raw_field:
                warnings.warn(
                    "edge_norm : sources vide — arête ignorée.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            src_list = list(src_raw_field)
        else:
            src_list = [src_raw_field]
    else:
        warnings.warn(
            f"edge_norm : format d'arête invalide {type(e).__name__} — ignorée.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if len(src_list) == 1:
        return _norm_single(src_list[0], dst_raw, attrs, outer)

    if len(src_list) == 2:
        # JOINT_CAUSE/JointPrevent : 2 arêtes avec joint_group_id déterministe
        # C1 fix : normaliser AVANT le hash pour cohérence avec bootstrap.py
        _dst_norm = normalize_node_id(dst_raw)
        _src_norms = [normalize_node_id(s) for s in src_list]
        key = f"{_dst_norm}|{'|'.join(sorted(_src_norms))}"
        jgid = hashlib.sha256(key.encode()).hexdigest()[:16]
        relation = _get_relation(attrs, outer)
        if relation and sanitize_text(str(relation)) not in ("joint_cause", "joint_prevent"):
            import warnings as _w
            _w.warn(
                f"edge_norm: relation '{relation}' sur 2 sources remplacée par 'joint_cause'. "
                "Annoter explicitement joint_cause/joint_prevent pour supprimer cet avertissement.",
                UserWarning, stacklevel=4,
            )
            relation = "joint_cause"
        e1 = dict(attrs)
        e1["source"] = src_list[0]
        e1["relation"] = relation
        e2 = dict(attrs)
        e2["source"] = src_list[1]
        e2["relation"] = relation
        r1 = _norm_single(src_list[0], dst_raw, e1, e1, joint_group_id=jgid)
        r2 = _norm_single(src_list[1], dst_raw, e2, e2, joint_group_id=jgid)
        return [r for r in (r1, r2) if r is not None] or None

    warnings.warn(
        f"edge_norm: {len(src_list)} sources, 2 max supportées — seule la première conservée.",
        UserWarning,
        stacklevel=2,
    )
    return _norm_single(src_list[0], dst_raw, attrs, outer)
