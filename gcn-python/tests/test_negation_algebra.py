# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests §9.4 ETUDE — algèbre de négation ternaire."""
import pytest
from gcn_python.pipeline.ir_emitter import apply_negation_algebra
from gcn_python.constants import NEGATION_PREVENT_MAP


# ---------------------------------------------------------------------------
# RÈGLE 1 — Neg(dst) : relation causale → variante prevent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cause_rel,expected", [
    ("cause",             "prevent"),
    ("conditional_cause", "conditional_prevent"),
    ("mediated_cause",    "mediated_prevent"),
    ("joint_cause",       "joint_prevent"),
])
def test_r1_neg_dst_maps_to_prevent(cause_rel, expected):
    rel, third, sp = apply_negation_algebra(cause_rel, "dst", None, True)
    assert rel == expected
    assert third is None
    assert sp is None


def test_r1_non_cause_relation_unchanged():
    rel, third, sp = apply_negation_algebra("enable", "dst", None, True)
    assert rel == "enable"   # pas dans NEGATION_PREVENT_MAP → inchangé


# ---------------------------------------------------------------------------
# RÈGLE 2 — Neg(condition/third) : polarity="negative" sur third
# ---------------------------------------------------------------------------

def test_r2_neg_condition_sets_polarity():
    third_in = {"role": "condition", "node": 2}
    rel, third_out, sp = apply_negation_algebra("conditional_cause", "condition", third_in, True)
    assert rel == "conditional_cause"
    assert third_out is not None
    assert third_out["polarity"] == "negative"
    assert sp is None


def test_r2_no_third_no_change():
    rel, third_out, sp = apply_negation_algebra("conditional_cause", "condition", None, True)
    assert rel == "conditional_cause"
    assert third_out is None


# ---------------------------------------------------------------------------
# RÈGLE 3 — Neg(src) : cause → counterfactual, joint_cause → source_polarity
# ---------------------------------------------------------------------------

def test_r3_cause_becomes_counterfactual():
    rel, third, sp = apply_negation_algebra("cause", "src", None, True)
    assert rel == "counterfactual"
    assert sp is None


def test_r3_joint_cause_source_polarity():
    rel, third, sp = apply_negation_algebra("joint_cause", "src", None, True)
    assert rel == "joint_cause"   # relation inchangée
    assert sp == "negative"


# ---------------------------------------------------------------------------
# Passthrough — aucune négation
# ---------------------------------------------------------------------------

def test_passthrough_not_negated():
    for rel in ("cause", "conditional_cause", "joint_cause", "enable"):
        out_rel, out_third, out_sp = apply_negation_algebra(rel, "dst", None, False)
        assert out_rel == rel
        assert out_third is None
        assert out_sp is None


# ---------------------------------------------------------------------------
# Cohérence — NEGATION_PREVENT_MAP couvre les 4 relations cause
# ---------------------------------------------------------------------------

def test_prevent_map_coverage():
    expected_keys = {"cause", "conditional_cause", "mediated_cause", "joint_cause"}
    assert set(NEGATION_PREVENT_MAP.keys()) == expected_keys
    for k, v in NEGATION_PREVENT_MAP.items():
        assert "prevent" in v, f"{k} → {v} ne contient pas 'prevent'"
