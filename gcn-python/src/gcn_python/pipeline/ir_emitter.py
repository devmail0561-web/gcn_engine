from __future__ import annotations


def emit(
    text: str,
    lang: str,
    node_types: list[str],
    node_labels: list[str],
    token_spans: list[tuple[int, int]],
    scopes: list[str],
    edge_triples: list[tuple[int, int, str, float, bool, int | None]],
    node_origins: list[str] | None = None,
) -> dict:
    """
    Produit un dict CausalIR conforme au schéma serde Rust de gcn-ir.
    JSON-serializable. Tous les noms en snake_case.

    edge_triples : (src_idx, dst_idx, relation, confidence, negated, marker_token)
    """
    if node_origins is None:
        node_origins = ["explicit"] * len(node_types)

    nodes = []
    for i, (nt, label, span, scope, origin) in enumerate(
        zip(node_types, node_labels, token_spans, scopes, node_origins)
    ):
        nodes.append({
            "id": i,
            "node_type": nt,
            "label": label,
            "source_span": {"token_span": {"start": span[0], "end": span[1]}},
            "scope": scope,
            "modifiers": [],
            "temporal_ref": "unresolved",
            "temporal_index": i,
            "origin": origin,
            "attributes": {
                "entity": None,
                "quality": None,
                "agent": None,
                "patient": None,
                "agent_type": None,
                "reversible": None,
            },
        })

    edges = []
    for src, dst, relation, confidence, negated, marker_token in edge_triples:
        edges.append([src, dst, {
            "relation": relation,
            "confidence": float(confidence),
            "temporal_gap": None,
            "explicit": marker_token is not None,
            "negated": negated,
            "marker_token": marker_token,
            "in_cycle": None,
        }])

    return {
        "source_lang": {"natural": {"lang": lang}},
        "source_text": text,
        "nodes": nodes,
        "edges": edges,
        "cycles": [],
        "unresolved": [],
        "metadata": {
            "schema_version": "1.0",
            "pipeline": ["cgnp-layer1", "cgnp-layer2", "cgnp-layer3"],
            "created_at": None,
        },
    }
