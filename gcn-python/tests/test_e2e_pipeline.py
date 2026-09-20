# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests e2e Phase D3 — texte brut → CIR JSON via pipeline CGNP.

3 cas :
  1. Phrase simple   : 2 clauses, 1 relation causale (cause)
  2. Phrase complexe : 3 clauses, 2 relations (cause + enable)
  3. Sans causalité  : 1 clause, 0 relation

Méthode : _cir_to_reps_and_connectors() remplace l'appel au binaire gcn
(non disponible dans l'environnement de test). Le reste du pipeline est réel.
"""
import json
import numpy as np
import pytest

from gcn_python.frontend.bridge import _cir_to_reps_and_connectors
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline


# ─── CIR synthétiques ────────────────────────────────────────────────────────

def _node(i, ntype, label, start, end):
    return {
        "id": i, "node_type": ntype, "label": label,
        "source_span": {"token_span": {"start": start, "end": end}},
        "scope": "specific", "temporal_index": i, "temporal_ref": "unresolved",
        "origin": "explicit",
        "attributes": {"entity": None, "quality": None, "agent": None, "patient": None},
    }


_CIR_SIMPLE = {
    "source_lang": {"natural": {"lang": "french"}},
    "source_text": "Les ventes baissent donc les prix augmentent.",
    "nodes": [
        _node(0, "action", "baisser", 1, 3),
        _node(1, "processus", "augmenter", 5, 7),
    ],
    "edges": [[0, 1, {
        "relation": "cause", "confidence": 0.95,
        "explicit": True, "negated": False, "marker_token": 4,
    }]],
}

_CIR_COMPLEX = {
    "source_lang": {"natural": {"lang": "french"}},
    "source_text": "Le gel détruit les cultures, ce qui entraîne des pénuries et provoque une hausse des prix.",
    "nodes": [
        _node(0, "processus", "détruire",  1,  5),
        _node(1, "etat",      "pénurie",   7, 12),
        _node(2, "processus", "augmenter", 13, 18),
    ],
    "edges": [
        [0, 1, {"relation": "cause",  "confidence": 0.90, "explicit": True,  "negated": False, "marker_token": 6}],
        [1, 2, {"relation": "enable", "confidence": 0.85, "explicit": False, "negated": False, "marker_token": None}],
    ],
}

_CIR_NO_CAUSAL = {
    "source_lang": {"natural": {"lang": "french"}},
    "source_text": "La météo est agréable aujourd'hui.",
    "nodes": [
        _node(0, "etat", "agréable", 1, 5),
    ],
    "edges": [],
}


# ─── Fixture : pipeline minimal ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    from gcn_python.constants import NODE_TYPES
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    n_node_types = len(NODE_TYPES)  # 7
    d_edge = vocab.d_edge_closed_loop(d_cl, n_node_types)
    enc = MLPEncoder(d_clause=d_cl, d_edge=d_edge, seed=0, n_node_types=n_node_types)
    graph = RGCNLayer(d_in=d_cl, d_out=d_cl, seed=0)
    return CGNPipeline(encoder=enc, graph=graph, vocabulary=vocab)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _reps_from_cir(cir):
    reps, connectors = _cir_to_reps_and_connectors(cir)
    return reps, connectors, cir.get("source_text", "")


def _valid_cir_json(out: dict) -> None:
    """Vérifie la structure minimale d'un CIR JSON."""
    assert isinstance(out, dict), "La sortie doit être un dict"
    assert "nodes" in out, "CIR doit contenir 'nodes'"
    assert "edges" in out, "CIR doit contenir 'edges'"
    assert isinstance(out["nodes"], list), "'nodes' doit être une liste"
    assert isinstance(out["edges"], list), "'edges' doit être une liste"
    # Chaque nœud doit avoir un champ node_type valide
    from gcn_python.constants import NODE_TYPES
    for n in out["nodes"]:
        assert "node_type" in n, f"Nœud sans 'node_type' : {n}"
        assert n["node_type"] in NODE_TYPES, f"node_type inconnu : {n['node_type']}"
    # JSON-serializable
    json.dumps(out)


# ─── Cas 1 : phrase simple ────────────────────────────────────────────────────

def test_e2e_simple_phrase_structure(pipeline):
    """Phrase simple (2 clauses, 1 relation) → CIR valide avec 2 nœuds et ≥0 arêtes."""
    reps, connectors, text = _reps_from_cir(_CIR_SIMPLE)
    assert len(reps) == 2, f"Attendu 2 reps, obtenu {len(reps)}"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 2, f"Attendu 2 nœuds, obtenu {len(out['nodes'])}"


def test_e2e_simple_phrase_json_serializable(pipeline):
    """La sortie de forward() est sérialisable en JSON (pas de numpy scalaires)."""
    reps, connectors, text = _reps_from_cir(_CIR_SIMPLE)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    serialized = json.dumps(out)
    assert len(serialized) > 10


def test_e2e_simple_phrase_relation_field(pipeline):
    """Les arêtes prédites contiennent un champ 'relation' valide."""
    reps, connectors, text = _reps_from_cir(_CIR_SIMPLE)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    from gcn_python.constants import RELATION_TYPES
    for edge in out["edges"]:
        rel = edge[2]["relation"] if isinstance(edge, list) else edge.get("relation")
        assert rel in RELATION_TYPES, f"Relation inconnue : {rel}"


# ─── Cas 2 : phrase complexe ─────────────────────────────────────────────────

def test_e2e_complex_phrase_three_nodes(pipeline):
    """Phrase complexe (3 clauses) → CIR avec 3 nœuds."""
    reps, connectors, text = _reps_from_cir(_CIR_COMPLEX)
    assert len(reps) == 3, f"Attendu 3 reps, obtenu {len(reps)}"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 3, f"Attendu 3 nœuds, obtenu {len(out['nodes'])}"


def test_e2e_complex_phrase_edges_between_valid_nodes(pipeline):
    """Les arêtes référencent des ids de nœuds valides."""
    reps, connectors, text = _reps_from_cir(_CIR_COMPLEX)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    node_ids = {n["id"] for n in out["nodes"]}
    for edge in out["edges"]:
        src, dst = (edge[0], edge[1]) if isinstance(edge, list) else (edge.get("source"), edge.get("target"))
        assert src in node_ids, f"Arête source {src} inexistante dans les nœuds"
        assert dst in node_ids, f"Arête cible {dst} inexistante dans les nœuds"


# ─── Cas 3 : phrase sans causalité ───────────────────────────────────────────

def test_e2e_no_causal_single_node(pipeline):
    """Phrase sans causalité (1 clause) → 1 nœud, 0 arête."""
    reps, connectors, text = _reps_from_cir(_CIR_NO_CAUSAL)
    assert len(reps) == 1, f"Attendu 1 rep, obtenu {len(reps)}"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 1, f"Attendu 1 nœud, obtenu {len(out['nodes'])}"
    assert len(out["edges"]) == 0, f"Attendu 0 arête, obtenu {len(out['edges'])}"


def test_e2e_no_causal_json_serializable(pipeline):
    """La sortie sans causalité est sérialisable en JSON."""
    reps, connectors, text = _reps_from_cir(_CIR_NO_CAUSAL)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    json.dumps(out)


def test_e2e_no_causal_node_type_valid(pipeline):
    """Le type de nœud retourné est dans les 7 types connus."""
    reps, connectors, text = _reps_from_cir(_CIR_NO_CAUSAL)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    from gcn_python.constants import NODE_TYPES
    assert out["nodes"][0]["node_type"] in NODE_TYPES
