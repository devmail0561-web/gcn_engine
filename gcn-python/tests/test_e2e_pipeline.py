from conftest import make_test_pipeline

# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests e2e Phase D3 — lattice → CIR via pipeline CGNP.

3 cas :
  1. Phrase simple   : 2 clauses, 1 connecteur (mark réel)
  2. Phrase complexe : 3 clauses, 2 connecteurs
  3. Sans causalité  : 1 clause, 0 connecteur

Méthode : _reps_from_lattice() sur des lattices synthétiques (tokens
observés, pas de dictionnaire). Le reste du pipeline est réel.
"""
import json

import pytest

from gcn_python.frontend.bridge import _reps_from_lattice
from gcn_python.layer1.features import FeatureVocabulary

# ─── Lattices synthétiques ────────────────────────────────────────────────────

def _tok(i, form, pos, dep, head, lemma, clause, flags=None):
    return {"index": i, "form": form, "pos": pos, "dep_rel": dep,
            "dep_head": head, "lemma": lemma, "clause": clause,
            "flags": flags or []}


_LATTICE_SIMPLE = {
    "source_text": "Les ventes baissent donc les prix augmentent.",
    "tokens": [
        _tok(1, "ventes", "noun", "nsubj", 2, "vente", 0),
        _tok(2, "baissent", "verb", "root", 0, "baisser", 0),
        _tok(3, "donc", "other", "mark", 5, "donc", 1),
        _tok(4, "prix", "noun", "nsubj", 5, "prix", 1),
        _tok(5, "augmentent", "verb", "advcl", 2, "augmenter", 1),
    ],
    "clauses": [[1, 2], [3, 5]],
}

_LATTICE_COMPLEX = {
    "source_text": "Le gel detruit les cultures, ce qui entraine des penuries et provoque une hausse.",
    "tokens": [
        _tok(1, "gel", "noun", "nsubj", 2, "gel", 0),
        _tok(2, "detruit", "verb", "root", 0, "detruire", 0),
        _tok(3, "cultures", "noun", "obj", 2, "culture", 0),
        _tok(4, ",", "punct", "punct", -1, ",", 0),
        _tok(5, "qui", "other", "mark", 7, "qui", 1),
        _tok(6, "penuries", "noun", "nsubj", 7, "penurie", 1),
        _tok(7, "entraine", "verb", "advcl", 2, "entrainer", 1),
        _tok(8, "et", "other", "mark", 10, "et", 2),
        _tok(9, "hausse", "noun", "nsubj", 10, "hausse", 2),
        _tok(10, "provoque", "verb", "advcl", 2, "provoquer", 2),
    ],
    "clauses": [[1, 4], [5, 7], [8, 10]],
}

_LATTICE_NO_CAUSAL = {
    "source_text": "La meteo est agreable.",
    "tokens": [
        _tok(1, "meteo", "noun", "nsubj", 2, "meteo", 0),
        _tok(2, "agreable", "noun", "root", 0, "agreable", 0),
    ],
    "clauses": [[1, 2]],
}


# ─── Fixture : pipeline minimal ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    from gcn_python.constants import NODE_TYPES
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause_effective(4)
    n_node_types = len(NODE_TYPES)  # 7
    vocab.d_edge_closed_loop(d_cl, n_node_types)
    return make_test_pipeline()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _reps_from_lat(lat):
    reps, connectors = _reps_from_lattice(lat)
    return reps, connectors, lat.get("source_text", "")


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
    """Phrase simple (2 clauses, 1 connecteur) → CIR valide avec 2 nœuds et ≥0 arêtes."""
    reps, connectors, text = _reps_from_lat(_LATTICE_SIMPLE)
    assert len(reps) == 2, f"Attendu 2 reps, obtenu {len(reps)}"
    assert connectors[0] is not None and connectors[0].root_lemma == "donc"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 2, f"Attendu 2 nœuds, obtenu {len(out['nodes'])}"


def test_e2e_simple_phrase_json_serializable(pipeline):
    """La sortie de forward() est sérialisable en JSON (pas de numpy scalaires)."""
    reps, connectors, text = _reps_from_lat(_LATTICE_SIMPLE)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    serialized = json.dumps(out)
    assert len(serialized) > 10


def test_e2e_simple_phrase_relation_field(pipeline):
    """Les arêtes prédites contiennent un champ 'relation' valide."""
    reps, connectors, text = _reps_from_lat(_LATTICE_SIMPLE)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    from gcn_python.constants import RELATION_TYPES
    for edge in out["edges"]:
        rel = edge[2]["relation"] if isinstance(edge, list) else edge.get("relation")
        assert rel in RELATION_TYPES, f"Relation inconnue : {rel}"


# ─── Cas 2 : phrase complexe ─────────────────────────────────────────────────

def test_e2e_complex_phrase_three_nodes(pipeline):
    """Phrase complexe (3 clauses) → CIR avec 3 nœuds."""
    reps, connectors, text = _reps_from_lat(_LATTICE_COMPLEX)
    assert len(reps) == 3, f"Attendu 3 reps, obtenu {len(reps)}"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 3, f"Attendu 3 nœuds, obtenu {len(out['nodes'])}"


def test_e2e_complex_phrase_edges_between_valid_nodes(pipeline):
    """Les arêtes référencent des ids de nœuds valides."""
    reps, connectors, text = _reps_from_lat(_LATTICE_COMPLEX)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    node_ids = {n["id"] for n in out["nodes"]}
    for edge in out["edges"]:
        src, dst = (edge[0], edge[1]) if isinstance(edge, list) else (edge.get("source"), edge.get("target"))
        assert src in node_ids, f"Arête source {src} inexistante dans les nœuds"
        assert dst in node_ids, f"Arête cible {dst} inexistante dans les nœuds"


# ─── Cas 3 : phrase sans causalité ───────────────────────────────────────────

def test_e2e_no_causal_single_node(pipeline):
    """Phrase sans causalité (1 clause) → 1 nœud, 0 arête."""
    reps, connectors, text = _reps_from_lat(_LATTICE_NO_CAUSAL)
    assert len(reps) == 1, f"Attendu 1 rep, obtenu {len(reps)}"

    out = pipeline.forward(reps, text, connector_reps=connectors)
    _valid_cir_json(out)
    assert len(out["nodes"]) == 1, f"Attendu 1 nœud, obtenu {len(out['nodes'])}"
    assert len(out["edges"]) == 0, f"Attendu 0 arête, obtenu {len(out['edges'])}"


def test_e2e_no_causal_json_serializable(pipeline):
    """La sortie sans causalité est sérialisable en JSON."""
    reps, connectors, text = _reps_from_lat(_LATTICE_NO_CAUSAL)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    json.dumps(out)


def test_e2e_no_causal_node_type_valid(pipeline):
    """Le type de nœud retourné est dans les types connus."""
    reps, connectors, text = _reps_from_lat(_LATTICE_NO_CAUSAL)
    out = pipeline.forward(reps, text, connector_reps=connectors)
    from gcn_python.constants import NODE_TYPES
    assert out["nodes"][0]["node_type"] in NODE_TYPES
