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
    doc_ref: str | None = None,
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

    # S-2 : temporal_index = rang dans l'ordre du texte (par position de token start)
    # Ordre de création ≠ ordre temporel quand une cause est mentionnée après son effet.
    text_order = sorted(range(len(token_spans)), key=lambda k: token_spans[k][0])
    temporal_rank = [0] * len(token_spans)
    for rank, orig_idx in enumerate(text_order):
        temporal_rank[orig_idx] = rank

    nodes = []
    for i, (nt, label, span, scope, origin) in enumerate(
        zip(node_types, node_labels, token_spans, scopes, node_origins, strict=False)
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
            "temporal_index": temporal_rank[i],  # S-2 : rang texte, pas rang création
            "origin": origin,
            "is_inferred": inferred,
            "attributes": attrs,
        })

    import datetime as _dt
    _now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from .. import __version__ as _gcn_version
    except Exception:
        _gcn_version = "unknown"

    import math
    edges = []
    for src, dst, relation, confidence, negated, marker_token in edge_triples:
        conf = float(confidence)
        if not math.isfinite(conf):
            raise ValueError(
                f"emit : confidence non finie ({confidence!r}) pour l'arête "
                f"{src}->{dst} — JSON refusé par serde Rust."
            )
        # S-7 temporal_gap : différence d'indices temporels entre les deux clauses
        t_src = temporal_rank[src] if src < len(temporal_rank) else None
        t_dst = temporal_rank[dst] if dst < len(temporal_rank) else None
        t_gap = (t_dst - t_src) if (t_src is not None and t_dst is not None) else None
        edges.append([src, dst, {
            "relation": relation,
            "confidence": conf,
            "temporal_gap": t_gap,
            "explicit": marker_token is not None,
            "negated": negated,
            "marker_token": marker_token,
            "in_cycle": None,
            "provenance": {
                "ref": doc_ref,
                "span": ({"token_span": {"start": marker_token, "end": marker_token}}
                         if marker_token is not None else "synthetic"),
                "extraction_method": "ml_python",
                "model_version": _gcn_version,
                "extracted_at": _now_iso,
            },
        }])

    # S-7 : détection de cycles par DFS sur le graphe d'arêtes (orienté)
    # Utilise un graphe orienté pour éviter de confondre arête retour avec parent.
    adj_directed: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for src, dst, _ in edges:
        adj_directed[src].append(dst)

    visited: set[int] = set()
    in_path: set[int] = set()       # nœuds dans le chemin de récursion courant
    cycle_node_sets: list[frozenset[int]] = []

    def _dfs_cycle(node: int, path: list[int]) -> None:
        visited.add(node)
        in_path.add(node)
        path.append(node)
        for nb in adj_directed[node]:
            if nb in in_path:
                # Arête retour → cycle : extraire la boucle
                idx = path.index(nb)
                cycle_node_sets.append(frozenset(path[idx:]))
            elif nb not in visited:
                _dfs_cycle(nb, path)
        path.pop()
        in_path.discard(node)

    for start in range(len(nodes)):
        if start not in visited:
            _dfs_cycle(start, [])

    # Dédupliquer les cycles
    unique_cycles = list({frozenset(c) for c in cycle_node_sets})
    cycles_output = [sorted(c) for c in unique_cycles]

    node_to_cycle: dict[int, int] = {}
    for cycle_id, c in enumerate(unique_cycles):
        for nid in c:
            node_to_cycle[nid] = cycle_id
    for edge in edges:
        src_e, dst_e = edge[0], edge[1]
        src_cid = node_to_cycle.get(src_e)
        dst_cid = node_to_cycle.get(dst_e)
        edge[2]["in_cycle"] = src_cid if (src_cid is not None and src_cid == dst_cid) else None

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
            "created_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        },
    }
