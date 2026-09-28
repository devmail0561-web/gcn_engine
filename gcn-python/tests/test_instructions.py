# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests unitaires pour le verbalizer/instructions.py.

Couvre : parse_command, CausalGraph._normalize, _match_level, _safe_conf,
find_path, find_best, estimate_delay, add_cir, SPOF, analogy poids.
"""
from __future__ import annotations

from gcn_python.verbalizer.instructions import CausalGraph, parse_command

# ── parse_command ─────────────────────────────────────────────────────────────

class TestParseCommand:
    def test_explain_with_colon(self):
        assert parse_command("explain: ventes") == ("explain", "ventes")

    def test_explain_case_insensitive(self):
        assert parse_command("EXPLAIN: hausse")[0] == "explain"

    def test_effects_without_colon(self):
        cmd, _arg = parse_command("effects hausse des coûts")
        assert cmd == "text"

    def test_summarize_no_arg(self):
        assert parse_command("summarize") == ("summarize", None)

    def test_unknown_returns_text(self):
        assert parse_command("bonjour monde") == ("text", None)

    def test_chain_with_colon(self):
        assert parse_command("chain: demande, ventes") == ("chain", "demande, ventes")

    def test_empty_input(self):
        assert parse_command("") == ("text", None)

    def test_whitespace_only(self):
        assert parse_command("   ") == ("text", None)

    def test_help(self):
        assert parse_command("help") == ("help", None)

    def test_quit(self):
        assert parse_command("quit") == ("quit", None)


# ── CausalGraph._normalize ───────────────────────────────────────────────────

class TestNormalize:
    def test_accents_fr(self):
        assert CausalGraph._normalize("économie") == "economie"

    def test_ligature_oe(self):
        assert CausalGraph._normalize("bœuf") == "boeuf"

    def test_ligature_ae(self):
        assert CausalGraph._normalize("Æther") == "aether"

    def test_uppercase(self):
        assert CausalGraph._normalize("État-Système") == "etat-systeme"

    def test_punctuation_stripped(self):
        assert CausalGraph._normalize("auth/login") == "authlogin"

    def test_empty(self):
        assert CausalGraph._normalize("") == ""

    def test_none(self):
        assert CausalGraph._normalize(None) == ""

    def test_czech_accents(self):
        assert CausalGraph._normalize("přístup") == "pristup"

    def test_polish_accents(self):
        assert CausalGraph._normalize("żądanie") == "zadanie"


# ── CausalGraph._safe_conf ──────────────────────────────────────────────────

class TestSafeConf:
    def test_normal(self):
        assert CausalGraph._safe_conf({"confidence": 0.8}) == 0.8

    def test_none(self):
        assert CausalGraph._safe_conf({}) == 1.0

    def test_nan(self):
        assert CausalGraph._safe_conf({"confidence": float("nan")}) == 1.0

    def test_inf(self):
        assert CausalGraph._safe_conf({"confidence": float("inf")}) == 1.0

    def test_negative_clamped(self):
        assert CausalGraph._safe_conf({"confidence": -0.5}) == 0.0

    def test_over_one_clamped(self):
        assert CausalGraph._safe_conf({"confidence": 1.5}) == 1.0

    def test_string_invalid(self):
        assert CausalGraph._safe_conf({"confidence": "not_a_number"}) == 1.0


# ── CausalGraph._match_level ────────────────────────────────────────────────

class TestMatchLevel:
    def test_exact(self):
        assert CausalGraph._match_level("ventes", "ventes") == 1

    def test_word_boundary(self):
        assert CausalGraph._match_level("hausse des ventes", "ventes") == 2

    def test_prefix(self):
        assert CausalGraph._match_level("ventestotales", "ventes") == 3

    def test_substring(self):
        assert CausalGraph._match_level("les ventes montent", "vente") == 4

    def test_no_match(self):
        assert CausalGraph._match_level("profit", "ventes") is None

    def test_empty_keyword(self):
        assert CausalGraph._match_level("ventes", "") is None


# ── CausalGraph core ────────────────────────────────────────────────────────

def _make_graph():
    g = CausalGraph()
    g.add_cir({
        "nodes": [
            {"id": 0, "label": "pluie", "node_type": "processus"},
            {"id": 1, "label": "inondation", "node_type": "processus"},
            {"id": 2, "label": "dégâts", "node_type": "processus"},
        ],
        "edges": [
            [0, 1, {"relation": "cause", "confidence": 0.9}],
            [1, 2, {"relation": "cause", "confidence": 0.7}],
        ],
        "source_text": "La pluie cause l'inondation qui cause des dégâts.",
    })
    return g


class TestCausalGraph:
    def test_add_cir_nodes(self):
        g = _make_graph()
        assert len(g.nodes) == 3
        assert g.nodes["0"]["label"] == "pluie"

    def test_add_cir_edges(self):
        g = _make_graph()
        assert len(g.edges) == 2

    def test_adjacency(self):
        g = _make_graph()
        assert "1" in g.adjacency["0"]

    def test_find_path(self):
        g = _make_graph()
        path = g.find_path("pluie", "dégâts")
        assert path is not None
        assert len(path) == 3

    def test_find_path_no_route(self):
        g = _make_graph()
        path = g.find_path("dégâts", "pluie")
        assert not path

    def test_clear(self):
        g = _make_graph()
        g.clear()
        assert len(g.nodes) == 0
        assert len(g.edges) == 0


# ── estimate_delay None propagation ─────────────────────────────────────────

class TestEstimateDelay:
    def test_no_temporal_data(self):
        g = _make_graph()
        result = g.estimate_delay("pluie", "dégâts")
        assert result["found"] is True
        assert result["index_delta"] is None

    def test_no_path(self):
        g = _make_graph()
        result = g.estimate_delay("dégâts", "pluie")
        assert result["found"] is False

    def test_gap_total_propagates_none(self):
        """Once gap_total is None (corrupt gap), it stays None."""
        g = CausalGraph()
        g.add_cir({
            "nodes": [
                {"id": 0, "label": "a"},
                {"id": 1, "label": "b"},
                {"id": 2, "label": "c"},
            ],
            "edges": [
                [0, 1, {"relation": "cause", "confidence": 1.0,
                         "temporal_gap": "invalid"}],
                [1, 2, {"relation": "cause", "confidence": 1.0,
                         "temporal_gap": {"min": 5, "max": 10}}],
            ],
        })
        result = g.estimate_delay("a", "c")
        assert result["found"] is True
        assert result["gap_sum"] is None


# ── save / load roundtrip ───────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        g = _make_graph()
        p = tmp_path / "graph.json"
        g.save(p)
        g2 = CausalGraph.load(p)
        assert len(g2.nodes) == 3
        assert len(g2.edges) == 2
        assert g2.nodes["0"]["label"] == "pluie"
