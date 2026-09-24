# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from ..constants import NODE_ORIGIN_VALUES, TEMPORAL_REF_DEFAULT


def _infer_temporal_ref(rep) -> str:
    """Dérive temporal_ref depuis les features UD déjà calculées (S-1)."""
    tense = getattr(rep, 'tense', '_absent')
    if tense == 'Past':
        return 'past'
    if tense in ('Fut', 'Futur'):
        return 'future'
    if tense == 'Pres':
        return 'present'
    if getattr(rep, 'has_temporal_obl', False):
        return 'anchored'
    return TEMPORAL_REF_DEFAULT


def emit(
    text: str,
    node_types: list[str],
    node_labels: list[str],
    token_spans: list[tuple[int, int]],
    scopes: list[str],
    edge_triples: list[tuple[int, int, str, float, bool, int | None]],
    node_origins: list[str] | None = None,
    node_attributes: list[dict] | None = None,
    node_inferred: list[bool] | None = None,
    temporal_refs: list[str] | None = None,  # S-1 : temporal_ref par nœud
) -> dict:
    """
    Produit un dict CausalIR conforme au schéma serde Rust de gcn-ir.
    JSON-serializable. Tous les noms en snake_case.

    edge_triples : (src_idx, dst_idx, relation, confidence, negated, marker_token)
    node_attributes : list de dicts {entity, agent, patient, quality, agent_type, reversible}
    node_inferred : flag par nœud (origine inférée) — métadonnée Python
      "is_inferred" sur le dict nœud, PAS un token spécial. Ignoré par serde Rust
      (champ supplémentaire côté Python uniquement).
    """
    if node_origins is None:
        node_origins = [NODE_ORIGIN_VALUES[0]] * len(node_types)

    nodes = []
    for i, (nt, label, span, scope, origin) in enumerate(
        zip(node_types, node_labels, token_spans, scopes, node_origins)
    ):
        attrs = node_attributes[i] if node_attributes and i < len(node_attributes) else {
            "entity": None,
            "quality": None,
            "agent": None,
            "patient": None,
            "agent_type": None,
            "reversible": None,
        }
        inferred = bool(node_inferred[i]) if node_inferred and i < len(node_inferred) else False
        nodes.append({
            "id": i,
            "node_type": nt,
            "label": label,
            "source_span": {"token_span": {"start": span[0], "end": span[1]}},
            "scope": scope,
            "modifiers": [],
            "temporal_ref": temporal_refs[i] if temporal_refs and i < len(temporal_refs) else TEMPORAL_REF_DEFAULT,
            "temporal_index": i,
            "origin": origin,
            "is_inferred": inferred,
            "attributes": attrs,
        })

    import math
    edges = []
    for src, dst, relation, confidence, negated, marker_token in edge_triples:
        conf = float(confidence)
        if not math.isfinite(conf):
            raise ValueError(
                f"emit : confidence non finie ({confidence!r}) pour l'arête "
                f"{src}->{dst} — JSON refusé par serde Rust."
            )
        edges.append([src, dst, {
            "relation": relation,
            "confidence": conf,
            "temporal_gap": None,
            "explicit": marker_token is not None,
            "negated": negated,
            "marker_token": marker_token,
            "in_cycle": None,
        }])

    # S-7 : détection de cycles par DFS sur le graphe d'arêtes
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for src, dst, _ in edges:
        adj[src].append(dst)
        adj[dst].append(src)  # non-orienté pour détection de cycles simples

    visited: set[int] = set()
    cycle_node_sets: list[frozenset[int]] = []

    def _dfs_cycle(node: int, parent: int, path: list[int]) -> None:
        visited.add(node)
        path.append(node)
        for nb in adj[node]:
            if nb == parent:
                continue
            if nb in visited:
                # Cycle détecté : extraire la boucle
                idx = path.index(nb)
                cycle_node_sets.append(frozenset(path[idx:]))
            else:
                _dfs_cycle(nb, node, path)
        path.pop()

    for start in range(len(nodes)):
        if start not in visited:
            _dfs_cycle(start, -1, [])

    # Dédupliquer les cycles
    unique_cycles = list({frozenset(c) for c in cycle_node_sets})
    cycles_output = [sorted(c) for c in unique_cycles]

    # Marquer in_cycle sur les arêtes
    in_cycle_nodes: set[int] = set()
    for c in unique_cycles:
        in_cycle_nodes.update(c)
    for edge in edges:
        src_e, dst_e = edge[0], edge[1]
        edge[2]["in_cycle"] = (src_e in in_cycle_nodes and dst_e in in_cycle_nodes)

    import datetime
    return {
        # Moteur language-agnostic : "und" volontaire (la langue n'est jamais
        # passée au modèle, même à l'entraînement).
        "source_lang": {"natural": {"lang": "und"}},
        "source_text": text,
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles_output,
        "unresolved": [],
        "metadata": {
            "schema_version": "2.0",
            "pipeline": ["cgnp-layer1", "cgnp-layer2", "cgnp-layer3"],
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
    }
