# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""R1 — reason() pure et bornée (gate : 10/10 + property + suites vertes)."""
import json

import pytest

from gcn_python.reasoning import (
    REASON_VERSION,
    STEP_CODES,
    dump_trace,
    load_trace,
    reason,
    reason_question,
)
from gcn_python.verbalizer.instructions import CausalGraph


def _g(nodes, edges):
    g = CausalGraph()
    g.add_cir({"nodes": [{"id": n, "label": n} for n in nodes],
               "edges": [[s, d, dict(a)] for s, d, a in edges],
               "source_text": "t"}, session_origin="document")
    return g


def _chain3():
    return _g(["pluie", "inondation", "degats"],
              [("pluie", "inondation", {"relation": "cause", "confidence": 0.9}),
               ("inondation", "degats", {"relation": "cause", "confidence": 0.7})])


def test_direct_ok_conf_min():
    r = reason("pluie", "degats", _chain3())
    assert r["version"] == REASON_VERSION
    assert r["status"] == "ok"
    assert r["answer"]["path"] == ["pluie", "inondation", "degats"]
    assert r["answer"]["confidence"] == pytest.approx(0.7)
    assert r["answer"]["derived"] == []
    assert {s["code"] for s in r["trace"]} <= STEP_CODES


def test_distractors_ignored():
    g = _g(["pluie", "inondation", "degats", "bruit", "vent"],
           [("pluie", "inondation", {"relation": "cause", "confidence": 0.9}),
            ("inondation", "degats", {"relation": "cause", "confidence": 0.7}),
            ("bruit", "vent", {"relation": "cause", "confidence": 0.99}),
            ("pluie", "vent", {"relation": "cause", "confidence": 0.5})])
    r = reason("pluie", "degats", g)
    assert r["status"] == "ok"
    assert r["answer"]["path"] == ["pluie", "inondation", "degats"]


def test_no_path_degrade():
    g = _g(["a", "b"], [])
    r = reason("a", "b", g)
    assert r["status"] == "degrade"
    assert "no causal path" in r["answer"]["summary"]
    assert r["trace"]  # trace conservée même en échec


def test_negated_backtrack():
    g = _g(["a", "b", "c"],
           [("a", "b", {"relation": "cause", "confidence": 0.9, "negated": True}),
            ("a", "c", {"relation": "cause", "confidence": 0.8}),
            ("c", "b", {"relation": "cause", "confidence": 0.6})])
    r = reason("a", "b", g)
    assert r["status"] == "ok"
    assert r["answer"]["path"] == ["a", "c", "b"]
    assert r["answer"]["confidence"] == pytest.approx(0.6)
    d = r["answer"]["derived"][0]
    assert d["derivation"] == "derived" and d["premises"] and d["confidence"] > 0
    assert any(s["code"] == "verify:" and (s.get("motif") or "").startswith("contradiction")
               for s in r["trace"])


def test_budget_exhausted():
    r = reason("pluie", "degats", _chain3(), budget=0)
    assert r["status"] == "degrade"
    assert "budget" in r["answer"]["summary"]


def test_empty_goal():
    r = reason("", "x", _chain3())
    assert r["status"] == "degrade"
    assert r["trace"][-1]["motif"] == "empty-goal"


def test_oversize():
    g = CausalGraph()
    g.add_cir({"nodes": [{"id": f"n{i}", "label": f"n{i}"} for i in range(501)],
               "edges": [], "source_text": "t"}, session_origin="document")
    r = reason("n0", "n1", g)
    assert r["status"] == "degrade"
    assert r["trace"][-1]["motif"] == "oversize:"


def test_question_chain_ok_and_multigoal_unparsed():
    assert reason_question("chain pluie -> degats", _chain3())["status"] == "ok"
    assert reason_question("chain a b; chain c d", _chain3())["trace"][-1]["motif"] == "multigoal:"
    assert reason_question("bonjour", _chain3())["trace"][-1]["motif"] == "unparsed-goal"


def test_deterministic_and_pure():
    g = _chain3()
    before = json.dumps({"nodes": g.nodes, "edges": g.edges}, sort_keys=True, default=str)
    r1, r2 = reason("pluie", "degats", g), reason("pluie", "degats", g)
    assert r1 == r2
    after = json.dumps({"nodes": g.nodes, "edges": g.edges}, sort_keys=True, default=str)
    assert before == after  # lecture seule : graphe intact


def test_infer_always_sourced_property():
    graphs = [_chain3(),
              _g(["a", "b", "c"],
                 [("a", "b", {"relation": "cause", "confidence": 0.9, "negated": True}),
                  ("a", "c", {"relation": "cause", "confidence": 0.8}),
                  ("c", "b", {"relation": "cause", "confidence": 0.6})]),
              _g(["a", "b"], [])]
    for g in graphs:
        for kw in [("pluie", "degats"), ("a", "b"), ("a", "zz")]:
            for s in reason(*kw, g)["trace"]:
                if s["code"] == "infer:":
                    assert s["premises"], "infer: sans prémisse = hallucination"


def test_trivial_self_goal_degrades():
    g = _g(["a"], [("a", "a", {"relation": "cause", "confidence": 0.9})])
    r = reason("a", "a", g)
    assert r["status"] == "degrade"
    assert r["trace"][-1]["motif"] == "trivial:"


def test_load_rejects_non_graph(tmp_path):
    for content, exc in [("[1,2,3]", ValueError), ('{"nodes": [1]}', ValueError),
                         ('{"edges": {}}', ValueError),
                         ('{"edges": [42]}', ValueError),
                         ('{"edges": [{"src": "a"}]}', ValueError)]:
        p = tmp_path / "g.json"
        p.write_text(content)
        with pytest.raises(exc):
            CausalGraph.load(p)


def test_trace_roundtrip(tmp_path):
    r = reason("pluie", "degats", _chain3())
    p = dump_trace(r["trace"], tmp_path / "t.jsonl")
    assert load_trace(p) == r["trace"]


class _NoPathGraph(CausalGraph):
    """Simule une divergence de matchers : arêtes présentes, find_path aveugle."""

    def find_path(self, kw_from, kw_to):
        return []


def test_fallback_bridge_without_path():
    g = _NoPathGraph()
    g.add_cir({"nodes": [{"id": n, "label": n} for n in ["a", "c", "b"]],
               "edges": [["a", "c", {"relation": "cause", "confidence": 0.8}],
                         ["c", "b", {"relation": "cause", "confidence": 0.6}]],
               "source_text": "t"}, session_origin="document")
    r = reason("a", "b", g)
    assert r["status"] == "ok"
    assert r["answer"]["path"] == ["a", "c", "b"]
    assert any(s["code"] == "subgoal:" for s in r["trace"])


class TestR2Strict:
    def test_non_causal_path_rejected_then_optout(self):
        g = _g(["a", "b"],
               [("a", "b", {"relation": "sequence", "confidence": 0.9})])
        r = reason("a", "b", g)
        assert r["status"] == "degrade"
        assert any((s.get("motif") or "").startswith("non-causal:")
                   for s in r["trace"])
        assert reason("a", "b", g, causal_only=False)["status"] == "ok"

    def test_untyped_edge_accepted_under_strict(self):
        g = _g(["a", "b"], [("a", "b", {"confidence": 0.9})])
        assert reason("a", "b", g)["status"] == "ok"

    def test_low_conf_premise_filtered(self):
        g = _g(["a", "b", "c"],
               [("a", "b", {"relation": "cause", "confidence": 0.9, "negated": True}),
                ("a", "c", {"relation": "cause", "confidence": 0.05}),
                ("c", "b", {"relation": "cause", "confidence": 0.9})])
        assert reason("a", "b", g)["status"] == "ok"  # R1 : 0.0 par défaut
        r = reason("a", "b", g, min_premise_conf=0.5)
        assert r["status"] == "degrade"
        assert not [s for s in r["trace"] if s["code"] == "infer:"]

    def test_opposite_prevent_contradiction(self):
        g = _g(["a", "b"],
               [("a", "b", {"relation": "cause", "confidence": 0.9}),
                ("a", "b", {"relation": "prevent", "confidence": 0.8})])
        r = reason("a", "b", g)
        assert r["status"] == "degrade"
        assert any("cause vs prevent" in (s.get("motif") or "")
                   for s in r["trace"])

    def test_opposition_relation_contradiction(self):
        g = _g(["a", "b"],
               [("a", "b", {"relation": "enable", "confidence": 0.9}),
                ("a", "b", {"relation": "opposition", "confidence": 0.8})])
        r = reason("a", "b", g)
        assert r["status"] == "degrade"

    def test_causal_group_content(self):
        from gcn_python.reasoning import CAUSAL_STRICT
        assert {"cause", "enable", "mediated_cause"} <= set(CAUSAL_STRICT)
        assert "sequence" not in CAUSAL_STRICT and "prevent" not in CAUSAL_STRICT


class TestP3GoalTraceability:
    def test_bare_multword_without_sep_ambiguous(self):
        r = reason_question("chain a b c", _chain3())
        assert r["status"] == "degrade"
        assert r["trace"][-1]["motif"] == "ambiguous-goal:"
        assert r["trace"][0]["dropped"] == ["b"]

    def test_explicit_sep_drops_traced(self):
        r = reason_question("chain: pluie -> inondation -> degats", _chain3())
        assert r["status"] == "ok"
        assert r["trace"][0]["dropped"] == ["inondation"]
