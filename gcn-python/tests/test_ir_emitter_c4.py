# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase C.4 — constants.py : 19 relations, 8 nœuds ; edge_norm.py : joint_group_id."""
from __future__ import annotations

from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.data.edge_norm import normalize_edge


def make_joint_edge(src1="n001", src2="n002", dst="n003"):
    return {
        "sources": [src1, src2],
        "target": dst,
        "relation": "joint_cause",
        "confidence": 0.8,
        "explicit": True,
        "negated": False,
    }


def test_19_relation_types_in_constants():
    assert len(RELATION_TYPES) == 19


def test_8_node_types_in_constants():
    assert len(NODE_TYPES) == 8
    assert "contrainte" in NODE_TYPES


def test_joint_cause_two_edges_from_two_sources():
    result = normalize_edge(make_joint_edge())
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 2


def test_edge_norm_joint_cause_group_id_preserved():
    result = normalize_edge(make_joint_edge())
    assert isinstance(result, list)
    jgid_0 = result[0].get("joint_group_id")
    jgid_1 = result[1].get("joint_group_id")
    assert jgid_0 is not None
    assert jgid_0 == jgid_1


def test_joint_group_id_is_deterministic():
    e = make_joint_edge()
    r1 = normalize_edge(e)
    r2 = normalize_edge(e)
    assert isinstance(r1, list) and isinstance(r2, list)
    assert r1[0]["joint_group_id"] == r2[0]["joint_group_id"]


def test_joint_group_id_different_for_different_sources():
    r1 = normalize_edge(make_joint_edge(src1="n001", src2="n002"))
    r2 = normalize_edge(make_joint_edge(src1="n010", src2="n020"))
    assert isinstance(r1, list) and isinstance(r2, list)
    assert r1[0]["joint_group_id"] != r2[0]["joint_group_id"]
