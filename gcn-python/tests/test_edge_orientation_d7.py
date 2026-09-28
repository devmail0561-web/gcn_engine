# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase C.5 — D7 orientation + Éq.7 voix (plan §C.5, test obligatoire).

Cas plan : « Le médicament est prescrit si les traitements échouent »
- D7 : source = clause SCONJ « si les traitements échouent ».
- Éq.7 : dans la clause cible passive, Agent/Patient inversés.
Ordre : D7 d'abord, Éq.7 ensuite.
"""
from __future__ import annotations

from gcn_python.pipeline.ir_emitter import (
    apply_voice_eq7,
    emit,
    orient_edge_d7,
)


def _sconj_tokens():
    return [
        {"lemma": "si", "pos": "SCONJ", "dep_rel": "mark", "morph": {}},
        {"lemma": "échouer", "pos": "VERB", "dep_rel": "advcl", "morph": {}},
    ]


def _main_tokens():
    return [
        {"lemma": "médicament", "pos": "NOUN", "dep_rel": "nsubj:pass", "morph": {}},
        {"lemma": "prescrire", "pos": "VERB", "dep_rel": "root",
         "morph": {"Voice": "Pass"}},
    ]


def test_d7_subordinate_becomes_source():
    # src=0 principale, dst=1 subordonnée SCONJ → swap vers (1, 0).
    assert orient_edge_d7(0, 1, "condition", [(0, 2), (3, 5)],
                          {0: _main_tokens(), 1: _sconj_tokens()}) == (1, 0)


def test_d7_already_correct_unchanged():
    assert orient_edge_d7(1, 0, "condition", [(0, 2), (3, 5)],
                          {0: _main_tokens(), 1: _sconj_tokens()}) == (1, 0)


def test_d7_sequence_uses_temporal_order():
    # SEQUENCE : source = clause antérieure, SCONJ ignoré.
    assert orient_edge_d7(1, 0, "sequence", [(0, 2), (3, 5)],
                          {0: _main_tokens(), 1: _sconj_tokens()}) == (0, 1)


def test_d7_no_metadata_identity():
    assert orient_edge_d7(0, 1, "condition") == (0, 1)


def test_eq7_passive_target_swaps():
    assert apply_voice_eq7(0, 1, {0: "Act", 1: "Pass"}) == (1, 0)


def test_eq7_active_unchanged():
    assert apply_voice_eq7(0, 1, {0: "Act", 1: "Act"}) == (0, 1)
    assert apply_voice_eq7(0, 1, None) == (0, 1)


def test_voice_d7_order_passive_conditional():
    """Test obligatoire plan §C.5 : D7 puis Éq.7 sur phrase passive + si."""
    # Clause 0 = principale passive, clause 1 = SCONJ. Arête brute (0, 1).
    src, dst = orient_edge_d7(0, 1, "condition", [(0, 2), (3, 5)],
                              {0: _main_tokens(), 1: _sconj_tokens()})
    assert (src, dst) == (1, 0)
    # La cible (clause 0, Voice=Pass) inverse Agent/Patient → (0, 1) logique.
    src2, dst2 = apply_voice_eq7(src, dst, {0: "Pass", 1: "Act"})
    assert (src2, dst2) == (0, 1)


def test_emit_orientation_opt_in():
    cir = emit(
        text="t", node_types=["processus", "etat_local"], node_labels=["a", "b"],
        token_spans=[(0, 2), (3, 5)], scopes=["specific", "specific"],
        edge_triples=[(0, 1, "sequence", 0.5, False, None)],
        apply_orientation=True,
    )
    assert (cir["edges"][0][0], cir["edges"][0][1]) == (0, 1)


def test_emit_default_no_reorientation():
    cir = emit(
        text="t", node_types=["processus", "etat_local"], node_labels=["a", "b"],
        token_spans=[(0, 2), (3, 5)], scopes=["specific", "specific"],
        edge_triples=[(1, 0, "cause", 0.5, False, None)],
    )
    assert (cir["edges"][0][0], cir["edges"][0][1]) == (1, 0)
