# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
import json
from gcn_python.pipeline.ir_emitter import emit


def test_emit_structure():
    result = emit(
        text="Si les ventes baissent, on réduit les coûts.",
        node_types=["processus", "action"],
        node_labels=["décroissance(ventes)", "réduire(coûts)"],
        token_spans=[(3, 4), (6, 9)],
        scopes=["universal", "universal"],
        edge_triples=[(0, 1, "condition", 1.0, False, 1)],
    )
    assert result["source_lang"] == {"natural": {"lang": "und"}}
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    s = json.dumps(result)
    back = json.loads(s)
    assert back["nodes"][0]["node_type"] == "processus"
    assert back["edges"][0][2]["relation"] == "condition"


def test_node_fields_complete():
    result = emit("test", ["action"], ["courir(il)"], [(1, 2)], ["specific"], [])
    node = result["nodes"][0]
    required = {"id", "node_type", "label", "source_span", "scope", "modifiers",
                "temporal_ref", "temporal_index", "origin", "attributes"}
    assert required.issubset(node.keys())
    attr_required = {"entity", "quality", "agent", "patient", "agent_type", "reversible"}
    assert attr_required.issubset(node["attributes"].keys())


def test_empty_emit():
    result = emit("", [], [], [], [], [])
    assert result["nodes"] == []
    assert result["edges"] == []


# ── S-1 temporal_ref ────────────────────────────────────────────────────────

def test_s1_temporal_ref_from_tense():
    """S-1 : temporal_ref calculé depuis UDRepresentation.tense."""
    from gcn_python.pipeline.ir_emitter import _infer_temporal_ref

    class MockRep:
        tense = "Past"
        has_temporal_obl = False

    assert _infer_temporal_ref(MockRep()) == "past"

    MockRep.tense = "Fut"
    assert _infer_temporal_ref(MockRep()) == "future"

    MockRep.tense = "Pres"
    assert _infer_temporal_ref(MockRep()) == "present"

    MockRep.tense = "_absent"
    MockRep.has_temporal_obl = True
    assert _infer_temporal_ref(MockRep()) == "anchored"

    MockRep.has_temporal_obl = False
    assert _infer_temporal_ref(MockRep()) == "unresolved"


def test_s1_temporal_ref_propagated_to_emit():
    """S-1 : temporal_refs passé à emit() remplace 'unresolved'."""
    result = emit("test", ["action"], ["a"], [(0, 1)], ["specific"], [],
                  temporal_refs=["past"])
    assert result["nodes"][0]["temporal_ref"] == "past"


# ── S-2 temporal_index ──────────────────────────────────────────────────────

def test_s2_temporal_index_text_order():
    """S-2 : temporal_index reflète l'ordre du texte, pas l'ordre de création."""
    # Nœud 0 = span (10,15) ; nœud 1 = span (0,5)
    # Dans le texte : nœud 1 apparaît en premier → temporal_index=0
    result = emit("test", ["action", "action"], ["effet", "cause"],
                  [(10, 15), (0, 5)], ["specific", "specific"], [])
    indices = {n["label"]: n["temporal_index"] for n in result["nodes"]}
    assert indices["cause"] == 0, f"cause doit être 0 (est {indices['cause']})"
    assert indices["effet"] == 1, f"effet doit être 1 (est {indices['effet']})"


def test_s2_temporal_index_sequential_when_ordered():
    """S-2 : si spans déjà ordonnés, temporal_index = 0, 1, 2..."""
    result = emit("test", ["action", "action", "action"], ["a", "b", "c"],
                  [(0, 2), (3, 5), (6, 8)], ["specific"] * 3, [])
    for i, node in enumerate(result["nodes"]):
        assert node["temporal_index"] == i


# ── S-3 origin hypothetical ─────────────────────────────────────────────────

def test_s3_infer_origin_hypothetical_cnd():
    """S-3 : mood=Cnd → origin='hypothetical'."""
    from gcn_python.pipeline.cgnp import _infer_origin

    class MockRep:
        mood = "Cnd"

    assert _infer_origin("action", connector_rep=None, rep=MockRep()) == "hypothetical"


def test_s3_infer_origin_hypothetical_sub():
    """S-3 : mood=Sub → origin='hypothetical'."""
    from gcn_python.pipeline.cgnp import _infer_origin

    class MockRep:
        mood = "Sub"

    assert _infer_origin("processus", connector_rep=None, rep=MockRep()) == "hypothetical"


def test_s3_infer_origin_explicit_default():
    """S-3 : mood=Ind → origin='explicit' (défaut)."""
    from gcn_python.pipeline.cgnp import _infer_origin

    class MockRep:
        mood = "Ind"

    assert _infer_origin("action", connector_rep=None, rep=MockRep()) == "explicit"


# ── S-7 cycles + temporal_gap + created_at ──────────────────────────────────

def test_s7_created_at_filled():
    """S-7 : created_at non None dans les metadata."""
    result = emit("test", ["action"], ["a"], [(0, 1)], ["specific"], [])
    assert result["metadata"]["created_at"] is not None
    assert result["metadata"]["created_at"].endswith("Z")


def test_s7_temporal_gap_computed():
    """S-7 : temporal_gap = différence d'indices temporels entre les clauses."""
    # cause span (0,5), effet span (10,15) → temporal_gap = 1
    result = emit("cause effet", ["action", "action"], ["cause", "effet"],
                  [(0, 5), (10, 15)], ["specific", "specific"],
                  [(0, 1, "cause", 1.0, False, None)])
    gap = result["edges"][0][2]["temporal_gap"]
    assert gap == 1, f"gap={gap}"


def test_s7_cycles_detected():
    """S-7 : cycle A→B→A détecté dans cycles[]."""
    result = emit("test", ["action", "action"], ["a", "b"],
                  [(0, 1), (2, 3)], ["specific", "specific"],
                  [(0, 1, "cause", 1.0, False, None),
                   (1, 0, "enable", 1.0, False, None)])
    assert len(result["cycles"]) > 0, "cycle A-B-A non détecté"


def test_s7_in_cycle_marked_on_edges():
    """S-7 : edges participant au cycle ont in_cycle=True."""
    result = emit("test", ["action", "action"], ["a", "b"],
                  [(0, 1), (2, 3)], ["specific", "specific"],
                  [(0, 1, "cause", 1.0, False, None),
                   (1, 0, "enable", 1.0, False, None)])
    assert all(e[2]["in_cycle"] is True for e in result["edges"])


def test_s7_no_cycle_in_cycle_false():
    """S-7 : sans cycle, in_cycle=False sur toutes les arêtes."""
    result = emit("test", ["action", "action"], ["a", "b"],
                  [(0, 1), (2, 3)], ["specific", "specific"],
                  [(0, 1, "cause", 1.0, False, None)])
    assert all(e[2]["in_cycle"] is False for e in result["edges"])


# ── L-5 scope heuristique ───────────────────────────────────────────────────

def test_l5_scope_universal_token():
    """L-5 : lemme 'tout' avec dep_rel=det → scope='universal'."""
    from gcn_python.pipeline.cgnp import _infer_scope

    class MockRep:
        tokens = [{"lemma": "tout", "dep_rel": "det", "pos": "DET"}]

    assert _infer_scope(MockRep(), {}) == "universal"


def test_l5_scope_existential_token():
    """L-5 : lemme 'parfois' avec dep_rel=advmod → scope='existential'."""
    from gcn_python.pipeline.cgnp import _infer_scope

    class MockRep:
        tokens = [{"lemma": "parfois", "dep_rel": "advmod", "pos": "ADV"}]

    assert _infer_scope(MockRep(), {}) == "existential"


def test_l5_scope_specific_default():
    """L-5 : sans marqueur → scope='specific'."""
    from gcn_python.pipeline.cgnp import _infer_scope

    class MockRep:
        tokens = [{"lemma": "chien", "dep_rel": "nsubj", "pos": "NOUN"}]

    assert _infer_scope(MockRep(), {}) == "specific"


def test_l5_scope_hints_override():
    """L-5 : scope_hints externe prioritaire sur détection intégrée."""
    from gcn_python.pipeline.cgnp import _infer_scope

    class MockRep:
        tokens = [{"lemma": "parfois", "dep_rel": "advmod", "pos": "ADV"}]

    assert _infer_scope(MockRep(), {"parfois": "partial"}) == "partial"
