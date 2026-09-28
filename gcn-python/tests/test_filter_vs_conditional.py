# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase E/D3 — FILTER direct vs CONDITIONAL_CAUSE (plan §DECISIONS_NON_COUVERTES).

D3 : phrase pivot restrictive (« n'est prescrit que si… », Agent=∅ en
passif) = FILTER direct (third=None), pas CONDITIONAL_CAUSE.
Le modèle ne doit pas confondre les deux sans que les gates le détectent.
Critère plan : F1 FILTER > 60% sur pivots — ici test structurel émetteur
(stockage de la distinction + third=None pour FILTER).
"""
from __future__ import annotations

from gcn_python.pipeline.ir_emitter import detect_ternary, emit


def _cir(rel, **kw):
    return emit(
        text="t", node_types=["processus", "etat_local"], node_labels=["a", "b"],
        token_spans=[(0, 3), (4, 7)], scopes=["specific", "specific"],
        edge_triples=[(0, 1, rel, 0.8, False, 2)],
        **kw,
    )


def test_filter_is_direct_third_none():
    cir = _cir("filter")
    assert cir["edges"][0][2]["relation"] == "filter"
    assert cir["edges"][0][2]["third"] is None


def test_conditional_cause_carries_third():
    cir = _cir("conditional_cause")
    # Émetteur 6-tuple legacy : third None par défaut (détecteur Phase E).
    assert cir["edges"][0][2]["third"] is None
    # 9-tuple v3 : third stocké et distinguable de FILTER.
    cir9 = emit(
        text="t", node_types=["processus", "etat_local"], node_labels=["a", "b"],
        token_spans=[(0, 3), (4, 7)], scopes=["specific", "specific"],
        edge_triples=[(0, 1, "conditional_cause", 0.8, False, 2,
                       "condition", 1, None)],
    )
    assert cir9["edges"][0][2]["third"] == {"role": "condition", "node": 1}
    assert cir9["edges"][0][2]["relation"] != "filter"


def test_detect_ternary_conditional_needs_sconj():
    sconj = [{"lemma": "si", "pos": "SCONJ", "dep_rel": "mark", "morph": {}}]
    assert detect_ternary("conditional_cause", sconj)[0] == "condition"
    plain = [{"lemma": "pluie", "pos": "NOUN", "dep_rel": "nsubj", "morph": {}}]
    assert detect_ternary("conditional_cause", plain)[0] is None


def test_detect_ternary_mediator_needs_obl():
    obl = [{"lemma": "saturer", "pos": "VERB", "dep_rel": "obl", "morph": {}}]
    assert detect_ternary("mediated_cause", obl)[0] == "mediator"


def test_detect_ternary_joint_passthrough_signature():
    assert detect_ternary("joint_cause", joint_signature="deadbeef")[2] == "deadbeef"
    assert detect_ternary("cause")[0] is None
