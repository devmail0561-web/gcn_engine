# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase C.5 — D6-shadow : formule ETUDE §9.0 transcrite à l'identique.

Vérifie les 3 exemples numériques de l'ETUDE §9.0 (lignes 1402-1408) :
- CAUSE Rung1 + Ind + explicite = 0.50
- CONDITION Rung2 + Sub + explicite = 0.75*0.85 = 0.6375
- FILTER Rung2 + Ind + explicite + Δ+0.10 = 0.85
Et le mode shadow de emit() : confidence ML inchangée + confidence_d6.
"""
from __future__ import annotations

from gcn_python.pipeline.ir_emitter import confidence_d6, emit


def _cir_two_nodes(edge_triples, **kw):
    return emit(
        text="t",
        node_types=["processus", "etat_local"],
        node_labels=["a", "b"],
        token_spans=[(0, 2), (3, 5)],
        scopes=["specific", "specific"],
        edge_triples=edge_triples,
        **kw,
    )


def test_d6_cause_rung1():
    assert confidence_d6("cause", mood="Ind", connector="explicite") == 0.50


def test_d6_condition_sub():
    assert abs(confidence_d6("condition", mood="Sub", connector="explicite") - 0.6375) < 1e-9


def test_d6_filter_frontee_salience():
    assert abs(confidence_d6("filter", mood="Ind", connector="explicite",
                             salience_bonus=0.10) - 0.85) < 1e-9


def test_d6_connector_table():
    assert confidence_d6("cause", connector="verb") == 0.50 * 0.90
    assert confidence_d6("cause", connector="parataxis") == 0.50 * 0.60
    assert confidence_d6("cause", connector="aucun") == 0.50 * 0.50


def test_d6_mood_cnd():
    assert abs(confidence_d6("enable", mood="Cnd", connector="explicite") - 0.75 * 0.80) < 1e-9


def test_d6_clamped_and_unknown_relation():
    assert 0.0 <= confidence_d6("cause", salience_bonus=10.0) <= 1.0
    # Relation inconnue → fallback R2 documenté, jamais de KeyError.
    assert confidence_d6("relation_xyz", mood="Ind", connector="explicite") == 0.75


def test_d6_shadow_dual_fields_ml_unchanged():
    cir = _cir_two_nodes([(0, 1, "cause", 0.42, False, None)])
    attrs = cir["edges"][0][2]
    assert attrs["confidence"] == 0.42
    assert attrs["confidence_ml"] == 0.42
    # marker None → connecteur "aucun" : 0.50*1.0*0.50 = 0.25
    assert abs(attrs["confidence_d6"] - 0.25) < 1e-9


def test_d6_shadow_explicit_marker():
    cir = _cir_two_nodes([(0, 1, "condition", 0.9, False, 7)],
                         mood_by_node={0: "Ind", 1: "Sub"})
    attrs = cir["edges"][0][2]
    assert attrs["confidence"] == 0.9
    assert abs(attrs["confidence_d6"] - 0.6375) < 1e-9


def test_emit_9_tuple_ternary_storage():
    cir = _cir_two_nodes([(0, 1, "conditional_cause", 0.8, False, 4,
                           "condition", 1, None)])
    attrs = cir["edges"][0][2]
    assert attrs["third"] == {"role": "condition", "node": 1}
    assert attrs["joint_group_id"] is None


def test_emit_9_tuple_joint_group_id():
    cir = _cir_two_nodes([(0, 1, "joint_cause", 0.8, False, 2,
                           None, None, "abc123")])
    attrs = cir["edges"][0][2]
    assert attrs["joint_group_id"] == "abc123"
    assert attrs["third"] is None
