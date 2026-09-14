from __future__ import annotations
import json
import pytest
from gcn_python.verbalizer import ReferenceDecoder, VerbalizerDecoder


def _make_ir(source_lang: dict, nodes: list, edges: list, source_text: str = "") -> str:
    return json.dumps({
        "source_lang": source_lang,
        "source_text": source_text,
        "nodes": nodes,
        "edges": edges,
        "cycles": [],
        "unresolved": [],
        "metadata": {"schema_version": "1.0", "pipeline": [], "created_at": None},
    })


FR_LANG = {"natural": {"lang": "fr"}}
PY_LANG = {"programming": {"lang": "python"}}

NODES = [
    {"id": 0, "node_type": "processus", "label": "ventes",
     "source_span": {"kind": "synthetic"}, "scope": "universal",
     "modifiers": [], "temporal_ref": {"kind": "present"},
     "temporal_index": 0, "origin": "explicit", "attributes": {}},
    {"id": 1, "node_type": "action", "label": "couts",
     "source_span": {"kind": "synthetic"}, "scope": "universal",
     "modifiers": [], "temporal_ref": {"kind": "present"},
     "temporal_index": 1, "origin": "explicit", "attributes": {}},
]
EDGE = [0, 1, {"relation": "condition", "confidence": 1.0,
               "temporal_gap": None, "explicit": True,
               "negated": False, "marker_token": None, "in_cycle": None}]


def test_decoder_implements_protocol():
    assert isinstance(ReferenceDecoder(), VerbalizerDecoder)


def test_decode_returns_string():
    ir_json = _make_ir(FR_LANG, NODES, [EDGE])
    result = ReferenceDecoder().decode(ir_json)
    assert isinstance(result, str)


def test_decode_non_empty():
    ir_json = _make_ir(FR_LANG, NODES, [EDGE])
    assert len(ReferenceDecoder().decode(ir_json)) > 0


def test_decode_contains_node_labels():
    ir_json = _make_ir(FR_LANG, NODES, [EDGE])
    result = ReferenceDecoder().decode(ir_json)
    assert "ventes" in result
    assert "couts" in result


def test_decode_contains_relation():
    ir_json = _make_ir(FR_LANG, NODES, [EDGE])
    result = ReferenceDecoder().decode(ir_json)
    assert "condition" in result


def test_negated_edge_marked():
    edge = [0, 1, {"relation": "prevent", "confidence": 1.0,
                   "temporal_gap": None, "explicit": True,
                   "negated": True, "marker_token": None, "in_cycle": None}]
    result = ReferenceDecoder().decode(_make_ir(FR_LANG, NODES, [edge]))
    assert "¬" in result


def test_empty_edges_falls_back_to_node_labels():
    result = ReferenceDecoder().decode(_make_ir(FR_LANG, NODES, []))
    assert "ventes" in result or "couts" in result


def test_multiple_edges_multiple_lines():
    edge2 = [1, 0, {"relation": "cause", "confidence": 0.8,
                    "temporal_gap": None, "explicit": False,
                    "negated": False, "marker_token": None, "in_cycle": None}]
    result = ReferenceDecoder().decode(_make_ir(FR_LANG, NODES, [EDGE, edge2]))
    assert result.count("\n") >= 1


def test_different_source_lang_same_structure_same_output():
    """The reference decoder is format-agnostic: source_lang doesn't change output."""
    ir_fr = _make_ir(FR_LANG, NODES, [EDGE])
    ir_py = _make_ir(PY_LANG, NODES, [EDGE])
    assert ReferenceDecoder().decode(ir_fr) == ReferenceDecoder().decode(ir_py)


def test_decode_with_programming_lang():
    ir_json = _make_ir(PY_LANG, NODES, [EDGE])
    result = ReferenceDecoder().decode(ir_json)
    assert isinstance(result, str) and len(result) > 0
