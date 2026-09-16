"""Normalisation et validation des annotations LLM."""
from __future__ import annotations

import warnings


NODE_TYPES = {"etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"}
RELATION_TYPES = {"cause", "enable", "prevent", "condition", "concession", "sequence",
                  "motivation", "filter", "opposition", "data_dependency", "control_dependency"}

# Variantes LLM → valeurs canoniques
NODE_TYPE_ALIASES: dict[str, str] = {
    "état": "etat",
    "état_systémique": "etat_systemique",
    "état systemique": "etat_systemique",
    "etat_systemique": "etat_systemique",
    "etat systemique": "etat_systemique",
    "Etat": "etat",
    "Action": "action",
    "Transition": "transition",
    "Processus": "processus",
    "Condition": "condition",
    "Entite": "entite",
    "Entité": "entite",
}

RELATION_TYPE_ALIASES: dict[str, str] = {
    "enables": "enable",
    "prevents": "prevent",
    "concedes": "concession",
    "sequences": "sequence",
    "motivates": "motivation",
    "filters": "filter",
    "opposes": "opposition",
    "data_dep": "data_dependency",
    "control_dep": "control_dependency",
    "Cause": "cause",
    "Enable": "enable",
    "Prevent": "prevent",
    "Condition": "condition",
    "Concession": "concession",
    "Sequence": "sequence",
    "Motivation": "motivation",
    "Filter": "filter",
    "Opposition": "opposition",
}


def normalize_node_type(raw: str) -> str | None:
    """Normalise un type de nœud LLM en valeur canonique. Retourne None si invalide."""
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in NODE_TYPES:
        return normalized
    canonical = NODE_TYPE_ALIASES.get(normalized)
    if canonical and canonical in NODE_TYPES:
        return canonical
    # Essayer sans accents courants
    normalized = normalized.replace("é", "e").replace("è", "e").replace("ê", "e")
    if normalized in NODE_TYPES:
        return normalized
    return None


def normalize_relation_type(raw: str) -> str | None:
    """Normalise un type de relation LLM en valeur canonique. Retourne None si invalide."""
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in RELATION_TYPES:
        return normalized
    canonical = RELATION_TYPE_ALIASES.get(normalized)
    if canonical and canonical in RELATION_TYPES:
        return canonical
    return None


def normalize_annotation(raw: dict) -> dict:
    """Normalise et valide une annotation LLM brute.

    - Corrige les types de nœuds/relations non standard
    - Supprime les nœuds/arêtes avec des types invalides
    - Retourne le document normalisé

    Émet un warning pour chaque entrée corrigée ou supprimée.
    """
    doc = raw.get("document", raw)
    sentences = doc.get("sentences", [])

    for sent in sentences:
        cir = sent.get("cir", {})

        # Normaliser les nœuds
        valid_nodes = []
        node_ids = set()
        for node in cir.get("nodes", []):
            raw_type = node.get("type", "")
            norm_type = normalize_node_type(raw_type)
            if norm_type is None:
                warnings.warn(
                    f"Nœud '{node.get('id', '?')}' type invalide '{raw_type}' — supprimé.",
                    UserWarning, stacklevel=2,
                )
                continue
            node["type"] = norm_type
            node_id = node.get("id", "")
            if node_id in node_ids:
                warnings.warn(
                    f"Nœud ID dupliqué '{node_id}' — conservé (première occurrence).",
                    UserWarning, stacklevel=2,
                )
            node_ids.add(node_id)
            valid_nodes.append(node)

        # Normaliser les arêtes
        valid_edges = []
        for edge in cir.get("edges", []):
            raw_rel = edge.get("relation", "")
            norm_rel = normalize_relation_type(raw_rel)
            if norm_rel is None:
                warnings.warn(
                    f"Arête {edge.get('source', '?')}→{edge.get('target', '?')} "
                    f"relation invalide '{raw_rel}' — supprimée.",
                    UserWarning, stacklevel=2,
                )
                continue
            edge["relation"] = norm_rel
            # Vérifier que source et target existent dans les nœuds valides
            if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                warnings.warn(
                    f"Arête {edge.get('source', '?')}→{edge.get('target', '?')} "
                    f"réfère un nœud inexistant — conservée (vérification paresseuse).",
                    UserWarning, stacklevel=2,
                )
            valid_edges.append(edge)

        cir["nodes"] = valid_nodes
        cir["edges"] = valid_edges

    return raw
