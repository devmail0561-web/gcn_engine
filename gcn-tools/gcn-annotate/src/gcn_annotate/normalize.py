# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Normalisation et validation des annotations LLM — schéma v4 (ETUDE §11)."""
from __future__ import annotations

import warnings

# v4 — 8 types de nœuds (D5 ETUDE)
NODE_TYPES = {
    "processus", "etat_local", "etat_global", "entite",
    "condition", "concept", "evenement", "contrainte",
}

# v4 — 19 types de relations (D2 ETUDE)
RELATION_TYPES = {
    "cause", "enable", "prevent", "condition", "concession", "sequence",
    "motivation", "filter", "opposition", "data_dependency", "control_dependency",
    "analogy", "counterfactual",
    "conditional_cause", "mediated_cause", "joint_cause",
    "conditional_prevent", "mediated_prevent", "joint_prevent",
}

TERNARY_RELATIONS = {
    "conditional_cause", "mediated_cause",
    "conditional_prevent", "mediated_prevent",
}
JOINT_RELATIONS = {"joint_cause", "joint_prevent"}

VALID_POLARITY = {"positive", "negative"}
VALID_VOICE = {"active", "passive"}
VALID_MODALITY = {"indicative", "subjunctive", "conditional", "imperative"}
VALID_PROMINENCE = {"foreground", "background"}
VALID_THIRD_ROLES = {"condition", "mediator"}

NODE_TYPE_ALIASES: dict[str, str] = {
    # v2 → v4 (migration)
    "etat": "etat_local",
    "etat_systemique": "etat_global",
    "action": "processus",
    "transition": "processus",
    # accents / espaces
    "état": "etat_local",
    "état_systémique": "etat_global",
    "état systemique": "etat_global",
    "etat systemique": "etat_global",
    "état_local": "etat_local",
    "état_global": "etat_global",
    "événement": "evenement",
    # aliases hors-champ
    "constraint": "contrainte",
    # Pascal case
    "Etat": "etat_local", "Action": "processus", "Transition": "processus",
    "Processus": "processus", "Condition": "condition",
    "Entite": "entite", "Entité": "entite", "Contrainte": "contrainte",
    "Concept": "concept", "Evenement": "evenement", "Événement": "evenement",
    "EtatLocal": "etat_local", "EtatGlobal": "etat_global",
}

RELATION_TYPE_ALIASES: dict[str, str] = {
    "enables": "enable", "prevents": "prevent",
    "concedes": "concession", "sequences": "sequence",
    "motivates": "motivation", "filters": "filter",
    "opposes": "opposition",
    "data_dep": "data_dependency", "control_dep": "control_dependency",
    "conditional": "condition",
    "Cause": "cause", "Enable": "enable", "Prevent": "prevent",
    "Condition": "condition", "Concession": "concession",
    "Sequence": "sequence", "Motivation": "motivation",
    "Filter": "filter", "Opposition": "opposition",
    "ConditionalCause": "conditional_cause",
    "MediatedCause": "mediated_cause",
    "JointCause": "joint_cause",
}


VALID_SENTENCE_TYPES = {"declarative", "interrogative", "imperative", "exclamative"}


def normalize_node_type(raw: str) -> str | None:
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if s in NODE_TYPES:
        return s
    c = NODE_TYPE_ALIASES.get(s)
    if c and c in NODE_TYPES:
        return c
    s2 = s.replace("é", "e").replace("è", "e").replace("ê", "e")
    if s2 in NODE_TYPES:
        return s2
    return None


def normalize_relation_type(raw: str) -> str | None:
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if s in RELATION_TYPES:
        return s
    c = RELATION_TYPE_ALIASES.get(s)
    if c and c in RELATION_TYPES:
        return c
    return None


def _normalize_third(third_raw) -> dict | None:
    if not third_raw or not isinstance(third_raw, dict):
        return None
    role = str(third_raw.get("role", "")).lower()
    if role not in VALID_THIRD_ROLES:
        return None
    node = str(third_raw.get("node", ""))
    polarity = third_raw.get("polarity")
    return {
        "role": role,
        "node": node,
        "polarity": str(polarity).lower() if polarity in VALID_POLARITY else None,
    }


def normalize_annotation(raw: dict, span_base: str = "1") -> dict:
    """Normalise une annotation brute vers le schéma v4 (ETUDE §11).

    - NODE_TYPES 8 / RELATION_TYPES 19
    - source → sources (liste)
    - marker_token hissé de attributes → niveau arête (v4 plat)
    - token_span uniformisés en 1-based inclusifs (span_base="0" si source 0-based,
      ex. sortie LLM — sinon "+1" corromprait des spans déjà 1-based)
    - third normalisé pour les ternaires
    - polarity, voice, modality validés
    - intent transmis tel quel ; sentence_type validé (ignoré sinon, jamais deviné)
    """
    if span_base not in ("0", "1"):
        raise ValueError(f"span_base doit être '0' ou '1' (reçu {span_base!r}).")
    doc = raw.get("document", raw)
    sentences = doc.get("sentences", [])

    for sent in sentences:
        cir = sent.get("cir", {})
        # sentence_type : validé, jamais deviné (dérivation = rôle du loader).
        _st = str(sent.get("sentence_type", "") or "").strip().lower()
        if _st and _st not in VALID_SENTENCE_TYPES:
            warnings.warn(
                f"Phrase '{sent.get('id', '?')}' sentence_type invalide '{_st}' — ignoré.",
                UserWarning, stacklevel=2,
            )
            _st = ""
        sent["sentence_type"] = _st

        # Nœuds
        valid_nodes = []
        node_ids: set[str] = set()
        for node in cir.get("nodes", []):
            norm_type = normalize_node_type(node.get("type", ""))
            if norm_type is None:
                warnings.warn(
                    f"Nœud '{node.get('id', '?')}' type invalide '{node.get('type')}' — supprimé.",
                    UserWarning, stacklevel=2,
                )
                continue
            node["type"] = norm_type
            nid = str(node.get("id", ""))
            node_ids.add(nid)
            # token_span → 1-based inclusifs [start, end]
            _span = node.get("token_span", [0, 0]) or [0, 0]
            try:
                _s0, _s1 = int(_span[0]), int(_span[1])
            except (TypeError, ValueError, IndexError):
                _s0, _s1 = 0, 0
            if span_base == "0" and (_s0 > 0 or _s1 > 0):
                _s0, _s1 = _s0 + 1, _s1 + 1
            node["token_span"] = [max(0, _s0), max(0, _s1)]
            valid_nodes.append(node)

        # Arêtes
        valid_edges = []
        for edge in cir.get("edges", []):
            # Relation
            norm_rel = normalize_relation_type(edge.get("relation", ""))
            if norm_rel is None:
                warnings.warn(
                    f"Arête relation invalide '{edge.get('relation')}' — supprimée.",
                    UserWarning, stacklevel=2,
                )
                continue
            edge["relation"] = norm_rel

            # sources (liste) — shim depuis source singulier
            if "sources" not in edge or not edge["sources"]:
                legacy = edge.get("source")
                edge["sources"] = [str(legacy)] if legacy else []
            else:
                edge["sources"] = [str(s) for s in edge["sources"]]
            edge.pop("source", None)

            # marker_token hissé de attributes → niveau arête (v4 plat).
            # balanced_auto/smart le nichent sous attributes ; le loader lit le plat.
            _attrs = edge.get("attributes") or {}
            if edge.get("marker_token") is None and _attrs.get("marker_token") is not None:
                try:
                    edge["marker_token"] = int(_attrs.get("marker_token"))
                except (TypeError, ValueError):
                    warnings.warn(
                        f"marker_token invalide {_attrs.get('marker_token')!r} — ignoré.",
                        UserWarning, stacklevel=2,
                    )

            # third — ternaires
            if norm_rel in TERNARY_RELATIONS:
                edge["third"] = _normalize_third(edge.get("third"))
            elif norm_rel in JOINT_RELATIONS:
                edge["third"] = None   # joint = deux arêtes, pas de third
            else:
                edge.setdefault("third", None)

            # Qualifications (défauts si absents)
            pol = str(edge.get("polarity", "positive")).lower()
            edge["polarity"] = pol if pol in VALID_POLARITY else "positive"

            voice = str(edge.get("voice", "active")).lower()
            edge["voice"] = voice if voice in VALID_VOICE else "active"

            mod = str(edge.get("modality", "indicative")).lower()
            edge["modality"] = mod if mod in VALID_MODALITY else "indicative"

            edge.setdefault("has_restriction", False)
            edge.setdefault("confidence", None)

            cp = edge.get("condition_prominence")
            if cp and str(cp).lower() in VALID_PROMINENCE:
                edge["condition_prominence"] = str(cp).lower()
            else:
                edge["condition_prominence"] = None

            valid_edges.append(edge)

        cir["nodes"] = valid_nodes
        cir["edges"] = valid_edges

    return raw
