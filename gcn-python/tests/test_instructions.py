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
    }, session_origin="document")
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
        }, session_origin="document")
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


# ── Origine dialogue 9a (D-ORIG + spec-origine-addcir.md) ────────────────────

def _cir_pair(label_src="stress", label_dst="insomnie", conf=0.9, tag=""):
    return {
        "nodes": [{"id": f"s{tag}", "label": label_src}, {"id": f"d{tag}", "label": label_dst}],
        "edges": [[f"s{tag}", f"d{tag}", {"relation": "cause", "confidence": conf}]],
        "source_text": "t",
    }


class TestSessionOrigin:
    def test_origin_required(self):
        import pytest
        g = CausalGraph()
        with pytest.raises(ValueError):
            g.add_cir(_cir_pair())
        with pytest.raises(ValueError):
            g.add_cir(_cir_pair(), session_origin="archive")

    def test_weight_baked_at_write(self):
        g = CausalGraph()
        g.add_cir(_cir_pair(conf=0.6), session_origin="document")
        g.add_cir(_cir_pair(conf=0.9), session_origin="question_utilisateur")
        g.add_cir(_cir_pair(conf=0.9), session_origin="affirmation_utilisateur")
        confs = [e[2]["confidence"] for e in g.edges]
        assert confs[0] == 0.6
        assert confs[1] == 0.9 * 0.5
        assert confs[2] == 0.9 * 0.8
        assert [e[2]["session_weight"] for e in g.edges] == [1.0, 0.5, 0.8]

    def test_doc_path_outranks_question_path(self):
        # Acceptation spec : chemin doc 0.6 > chemin question 0.9×0.5=0.45
        import warnings
        g = CausalGraph()
        g.add_cir(_cir_pair("doc_src", "cible", 0.6, tag="a"), session_origin="document")
        g.add_cir(_cir_pair("q_src", "cible", 0.9, tag="b"), session_origin="question_utilisateur")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = g.find_causes("cible")
        assert g._label(res[0][0]) == "doc_src"
        assert g._label(res[1][0]) == "q_src"

    def test_origin_mention_in_response(self):
        from gcn_python.verbalizer.instructions import InstructionHandler
        h = InstructionHandler()
        h.add_cir(_cir_pair("q_src", "cible", 0.9), session_origin="question_utilisateur")
        out = h.execute("explain: cible")
        assert "[question]" in out
        assert "origin: question-only" in out

    def test_no_mention_for_pure_document(self):
        from gcn_python.verbalizer.instructions import InstructionHandler
        h = InstructionHandler()
        h.add_cir(_cir_pair("doc_src", "cible", 0.6), session_origin="document")
        out = h.execute("explain: cible")
        assert "[question]" not in out and "origin:" not in out

    def test_no_double_bake_on_readd(self):
        g = CausalGraph()
        cir = _cir_pair(conf=0.9)
        g.add_cir(cir, session_origin="question_utilisateur")
        g.add_cir(cir, session_origin="question_utilisateur")
        assert g.edges[0][2]["confidence"] == g.edges[1][2]["confidence"] == 0.9 * 0.5

    def test_no_aliasing_with_caller_cir(self):
        g = CausalGraph()
        cir = _cir_pair(conf=0.9)
        raw_attrs = cir["edges"][0][2]
        g.add_cir(cir, session_origin="question_utilisateur")
        assert "session_origin" not in raw_attrs
        assert raw_attrs["confidence"] == 0.9

    def test_node_precedence_upgrade_only(self):
        g = CausalGraph()
        g.add_cir(_cir_pair("x", "y"), session_origin="question_utilisateur")
        assert g.nodes["s"]["session_origin"] == "question_utilisateur"
        g.add_cir(_cir_pair("x", "z"), session_origin="document")
        assert g.nodes["s"]["session_origin"] == "document"
        g.add_cir(_cir_pair("x", "w"), session_origin="question_utilisateur")
        assert g.nodes["s"]["session_origin"] == "document"  # jamais de downgrade

    def test_load_migrates_legacy_graph(self, tmp_path):
        import json
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps({
            "nodes": {"n1": {"label": "a"}, "n2": {"label": "b"}},
            "edges": [{"src": "n1", "dst": "n2",
                       "attrs": {"relation": "cause", "confidence": 0.7}, "text": "t"}],
        }), encoding="utf-8")
        g = CausalGraph.load(p)
        assert g.nodes["n1"]["session_origin"] == "document"
        assert g.nodes["n1"]["migrated"] is True
        assert g.edges[0][2]["session_origin"] == "document"
        assert g.edges[0][2]["confidence"] == 0.7  # poids document = 1.0 : intact


class TestAddCirStrict:
    def test_unknown_relation_rejected(self):
        import pytest
        g = CausalGraph()
        with pytest.raises(ValueError, match="relation inconnue"):
            g.add_cir({"nodes": [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}],
                       "edges": [["a", "b", {"relation": "telepathy", "confidence": 0.9}]],
                       "source_text": "t"}, session_origin="document")
        assert len(g.edges) == 0  # rien stocké

    def test_dangling_endpoint_rejected(self):
        import pytest
        g = CausalGraph()
        with pytest.raises(ValueError, match=r"pendante.*zz"):
            g.add_cir({"nodes": [{"id": "a", "label": "a"}],
                       "edges": [["a", "zz", {"relation": "cause", "confidence": 0.9}]],
                       "source_text": "t"}, session_origin="document")

    def test_cross_cir_reference_accepted(self):
        g = CausalGraph()
        g.add_cir({"nodes": [{"id": "a", "label": "a"}],
                   "edges": [], "source_text": "t"}, session_origin="document")
        g.add_cir({"nodes": [{"id": "b", "label": "b"}],
                   "edges": [["a", "b", {"relation": "cause", "confidence": 0.9}]],
                   "source_text": "t2"}, session_origin="document")
        assert len(g.edges) == 1

    def test_untyped_relation_accepted(self):
        g = CausalGraph()
        g.add_cir({"nodes": [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}],
                   "edges": [["a", "b", {"confidence": 0.9}]],
                   "source_text": "t"}, session_origin="document")
        assert len(g.edges) == 1
