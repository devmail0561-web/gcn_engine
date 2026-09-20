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
