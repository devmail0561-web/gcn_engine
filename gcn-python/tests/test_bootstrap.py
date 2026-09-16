"""Tests pour training/bootstrap.py — corrections issues CRITICAL #1-3."""
import json
import pytest
from pathlib import Path
from gcn_python.training.bootstrap import _extract_token_span, _cir_to_doc


# ── Tests Issue #3 CRITICAL : null handling dans token_span ────────────────────


def test_extract_token_span_with_null_start():
    """Issue #3 : token_span avec start=null ne doit pas crasher."""
    node = {
        "source_span": {
            "token_span": {
                "start": None,  # null dans JSON
                "end": 5
            }
        }
    }
    span = _extract_token_span(node)
    assert span == [0, 5], "start=null doit fallback à 0"


def test_extract_token_span_with_null_end():
    """Issue #3 : token_span avec end=null ne doit pas crasher."""
    node = {
        "source_span": {
            "token_span": {
                "start": 2,
                "end": None  # null dans JSON
            }
        }
    }
    span = _extract_token_span(node)
    assert span == [2, 0], "end=null doit fallback à 0"


def test_extract_token_span_with_both_null():
    """Issue #3 : token_span avec start et end=null."""
    node = {
        "source_span": {
            "token_span": {
                "start": None,
                "end": None
            }
        }
    }
    span = _extract_token_span(node)
    assert span == [0, 0], "null values doivent fallback à 0"


def test_extract_token_span_dict_format():
    """Format dict normal (Python) doit fonctionner."""
    node = {
        "source_span": {
            "token_span": {
                "start": 1,
                "end": 5
            }
        }
    }
    span = _extract_token_span(node)
    assert span == [1, 5]


def test_extract_token_span_flat_format():
    """Format flat (Rust) doit fonctionner."""
    node = {
        "token_span": [2, 7]
    }
    span = _extract_token_span(node)
    assert span == [2, 7]


def test_extract_token_span_missing_fields():
    """Champs manquants doivent fallback à [0, 0]."""
    node = {}
    span = _extract_token_span(node)
    assert span == [0, 0]


def test_extract_token_span_empty_token_span():
    """token_span vide doit fallback à [0, 0]."""
    node = {"token_span": None}
    span = _extract_token_span(node)
    assert span == [0, 0]


# ── Tests _cir_to_doc (intégration) ───────────────────────────────────────────


def test_cir_to_doc_basic():
    """Conversion CIR → doc gcn-nl basique."""
    cir = {
        "nodes": [
            {
                "id": "n001",
                "node_type": "processus",
                "label": "baisse",
                "source_span": {"token_span": {"start": 1, "end": 3}},
                "scope": "specific",
                "temporal_index": 0,
                "origin": "explicit"
            }
        ],
        "edges": []
    }
    doc = _cir_to_doc("Les ventes baissent.", "fr", cir)

    assert doc["document"]["lang"] == "fr"
    assert len(doc["document"]["sentences"]) == 1
    sent = doc["document"]["sentences"][0]
    assert sent["text"] == "Les ventes baissent."
    assert len(sent["cir"]["nodes"]) == 1
    assert sent["cir"]["nodes"][0]["type"] == "processus"
    assert sent["cir"]["nodes"][0]["token_span"] == [1, 3]


def test_cir_to_doc_with_null_token_span():
    """Issue #3 : CIR avec token_span null ne doit pas crasher."""
    cir = {
        "nodes": [
            {
                "id": "n001",
                "node_type": "action",
                "label": "test",
                "source_span": {"token_span": {"start": None, "end": None}},
            }
        ],
        "edges": []
    }
    doc = _cir_to_doc("Test.", "fr", cir)

    # Ne doit pas crasher
    assert doc["document"]["sentences"][0]["cir"]["nodes"][0]["token_span"] == [0, 0]


def test_cir_to_doc_with_edges():
    """Conversion avec arêtes."""
    cir = {
        "nodes": [
            {"id": "n001", "node_type": "condition", "label": "A"},
            {"id": "n002", "node_type": "action", "label": "B"},
        ],
        "edges": [
            {
                "source": "n001",
                "target": "n002",
                "relation": "cause",
                "confidence": 0.95,
                "explicit": True,
                "negated": False,
            }
        ]
    }
    doc = _cir_to_doc("Si A alors B.", "fr", cir)

    edges = doc["document"]["sentences"][0]["cir"]["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "n001"
    assert edges[0]["target"] == "n002"
    assert edges[0]["relation"] == "cause"
    assert edges[0]["confidence"] == 0.95


# ── Tests Issue #1 et #2 (intégration CLI) ────────────────────────────────────
# Note : Tests d'intégration nécessitent le binaire gcn-cli Rust.
# Créer tests séparés avec pytest.mark.integration si disponible.


@pytest.mark.skip(reason="Nécessite binaire gcn-cli Rust installé")
def test_bootstrap_cmd_integration(tmp_path):
    """Issue #1 et #2 : bootstrap doit appeler gcn avec text comme arg positionnel."""
    # Test d'intégration complet à implémenter si gcn-cli disponible
    pass
