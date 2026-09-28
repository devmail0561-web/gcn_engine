# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Éq.11 ETUDE — résolution d'ambiguïté (θ_ambiguity)."""
import numpy as np
import pytest
from gcn_python.constants import THETA_AMBIGUITY_DEFAULT
from gcn_python.pipeline.ir_emitter import emit


def _make_simple_cir(relation: str, conf: float, ambiguous: bool = False,
                     candidates=None):
    """Construit un CIR minimal à 2 nœuds pour les tests."""
    tup = ("phrase test", ["processus", "processus"], ["A", "B"],
           [(0, 1), (2, 3)], ["specific", "specific"],
           [(0, 1, relation, conf, False, None, None, ambiguous, candidates)])
    cir = emit(
        tup[0], tup[1], tup[2], tup[3], tup[4], tup[5],
    )
    return cir


# ---------------------------------------------------------------------------
# Valeur par défaut THETA_AMBIGUITY_DEFAULT
# ---------------------------------------------------------------------------

def test_theta_default_value():
    assert THETA_AMBIGUITY_DEFAULT == 0.65


# ---------------------------------------------------------------------------
# Arête non ambiguë — confidence >= θ
# ---------------------------------------------------------------------------

def test_high_confidence_not_ambiguous():
    cir = _make_simple_cir("cause", 0.80, ambiguous=False)
    assert len(cir["edges"]) == 1
    edge = cir["edges"][0][2]
    assert edge["ambiguous"] is False
    assert edge["candidates"] is None


# ---------------------------------------------------------------------------
# Arête ambiguë — confidence < θ, candidates présents
# ---------------------------------------------------------------------------

def test_low_confidence_ambiguous_candidates():
    candidates = [("cause", 0.52), ("condition", 0.48)]
    cir = _make_simple_cir("cause", 0.52, ambiguous=True, candidates=candidates)
    assert len(cir["edges"]) == 1
    edge = cir["edges"][0][2]
    assert edge["ambiguous"] is True
    assert edge["candidates"] is not None
    assert len(edge["candidates"]) == 2
    assert edge["candidates"][0][0] == "cause"
    assert edge["candidates"][1][0] == "condition"


# ---------------------------------------------------------------------------
# Cohérence locale — arête confirmée voisine influence la résolution
# ---------------------------------------------------------------------------

def test_graph_coherence_resolves_to_neighbor_type():
    """Deux arêtes : une confirmée 'condition', une ambiguë entre cause/condition.
    La cohérence locale doit favoriser 'condition'."""
    confirmed = (0, 1, "condition", 0.90, False, None, None, False, None)
    ambiguous = (1, 2, "cause", 0.55, False, None, None, True,
                 [("cause", 0.55), ("condition", 0.45)])
    cir = emit(
        "test", ["processus", "processus", "processus"],
        ["A", "B", "C"],
        [(0, 1), (2, 3), (4, 5)],
        ["specific", "specific", "specific"],
        [confirmed, ambiguous],
    )
    edges = {(e[0], e[1]): e[2] for e in cir["edges"]}
    resolved = edges[(1, 2)]
    # c1_type="condition" est dans les voisins confirmés de node 1 → résolu
    assert resolved["relation"] == "condition"


# ---------------------------------------------------------------------------
# Cohérence locale vs globale — seuls les voisins comptent
# ---------------------------------------------------------------------------

def test_coherence_local_not_global():
    """Arête ambiguë non connectée au nœud confirmé → résolution par c0 (défaut)."""
    confirmed = (0, 1, "condition", 0.90, False, None, None, False, None)
    # arête entre nœuds 3-4, sans lien vers 0-1
    ambiguous = (3, 4, "cause", 0.55, False, None, None, True,
                 [("cause", 0.55), ("condition", 0.45)])
    cir = emit(
        "test",
        ["processus"] * 5,
        ["A", "B", "C", "D", "E"],
        [(i, i + 1) for i in range(5)],
        ["specific"] * 5,
        [confirmed, ambiguous],
    )
    edges = {(e[0], e[1]): e[2] for e in cir["edges"]}
    resolved = edges[(3, 4)]
    # aucun voisin confirmé → c0 conservé
    assert resolved["relation"] == "cause"
