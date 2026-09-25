# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests pour training/bootstrap.py — corrections issues CRITICAL #1-3."""
import json
from pathlib import Path

import pytest

from gcn_python.training.bootstrap import _cir_to_doc, _extract_token_span

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
    doc = _cir_to_doc("Les ventes baissent.", cir)

    assert len(doc["document"]["sentences"]) == 1
    sent = doc["document"]["sentences"][0]
    assert sent["text"] == "Les ventes baissent."
    assert len(sent["cir"]["nodes"]) == 1
    assert sent["cir"]["nodes"][0]["type"] == "processus"
    # B1 correctif : token_span synthétique [i+1, i+1] (aligné sur le token synthétique)
    assert sent["cir"]["nodes"][0]["token_span"] == [1, 1]
    # B1 correctif : tokens non vides pour que reps_from_sentence fonctionne
    assert len(sent["tokens"]) == 1
    assert sent["tokens"][0]["lemma"] == "baisse"
    assert sent["tokens"][0]["id"] == 1


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
    doc = _cir_to_doc("Test.", cir)

    # Ne doit pas crasher ; token_span synthétique [1, 1] pour le nœud 0
    assert doc["document"]["sentences"][0]["cir"]["nodes"][0]["token_span"] == [1, 1]


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
    doc = _cir_to_doc("Si A alors B.", cir)

    edges = doc["document"]["sentences"][0]["cir"]["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "n001"
    assert edges[0]["target"] == "n002"
    assert edges[0]["relation"] == "cause"
    assert edges[0]["confidence"] == 0.95


def test_normalize_edge_tuple_format():
    """COMMIT 1 : format tuple [src_id, dst_id, edge_obj] → dict correct."""
    from gcn_python.training.bootstrap import _normalize_edge
    e = [0, 1, {"relation": "cause", "confidence": 0.9, "explicit": True, "negated": False}]
    result = _normalize_edge(e)
    assert result is not None
    assert result["source"] == "n000"
    assert result["target"] == "n001"
    assert result["relation"] == "cause"
    assert result["confidence"] == 0.9
    assert result["explicit"] is True


def test_normalize_edge_dict_format():
    """COMMIT 1 : format dict existant → résultat identique à l'ancien code."""
    from gcn_python.training.bootstrap import _normalize_edge
    e = {"source": "n001", "target": "n002", "relation": "condition", "confidence": 1.0,
         "explicit": True, "negated": False}
    result = _normalize_edge(e)
    assert result is not None
    assert result["source"] == "n001"
    assert result["target"] == "n002"
    assert result["relation"] == "condition"


def test_normalize_edge_invalid_returns_none():
    """COMMIT 1 : format inconnu → None (pas de crash)."""
    from gcn_python.training.bootstrap import _normalize_edge
    assert _normalize_edge("invalid") is None
    assert _normalize_edge(42) is None
    assert _normalize_edge([0, 1]) is None  # seulement 2 éléments
    assert _normalize_edge([0, 1, "not_a_dict"]) is None  # edge_obj doit être dict


def test_cir_to_doc_with_tuple_edges():
    """COMMIT 1 : _cir_to_doc ne crashe plus avec des arêtes en format tuple."""
    from gcn_python.training.bootstrap import _cir_to_doc
    cir = {
        "nodes": [
            {"id": 0, "node_type": "action", "label": "A"},
            {"id": 1, "node_type": "etat", "label": "B"},
        ],
        "edges": [
            [0, 1, {"relation": "cause", "confidence": 1.0, "explicit": True, "negated": False}]
        ]
    }
    doc = _cir_to_doc("A cause B.", cir)
    edges = doc["document"]["sentences"][0]["cir"]["edges"]
    assert len(edges) == 1
    nodes = doc["document"]["sentences"][0]["cir"]["nodes"]
    assert edges[0]["source"] == nodes[0]["id"] == "n000"
    assert edges[0]["target"] == nodes[1]["id"] == "n001"
    assert edges[0]["relation"] == "cause"


# ── Tests Issue #1 et #2 (intégration CLI) ────────────────────────────────────
# Nécessitent le binaire gcn-cli Rust : PATH d'abord, sinon target/{release,debug}.


def _find_gcn_bin():
    """Chemin du binaire gcn (PATH, sinon artefact du workspace Rust)."""
    import shutil
    found = shutil.which("gcn")
    if found:
        return found
    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = root / "gcn-core" / "target" / profile / "gcn"
        if candidate.is_file():
            return str(candidate)
    return None


GCN_BIN = _find_gcn_bin()


@pytest.mark.skipif(
    GCN_BIN is None,
    reason="Binaire gcn-cli absent (cargo build --workspace)",
)
def test_bootstrap_cmd_integration(tmp_path):
    """Issue #1 : gcn-bootstrap appelle `gcn analyze --data-dir … -- <texte>`.

    Issue #2 : sortie UTF-8 décodée puis écrite en JSON gcn-nl valide.
    """
    from click.testing import CliRunner

    from gcn_python.training.bootstrap import bootstrap_cmd

    input_file = tmp_path / "phrases.txt"
    input_file.write_text("Les ventes baissent.\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    taxonomy_dir = Path(__file__).resolve().parents[2] / "gcn-references" / "taxonomies"

    result = CliRunner().invoke(
        bootstrap_cmd,
        [
            "--input", str(input_file),
            "--out-dir", str(out_dir),
            "--taxonomy-dir", str(taxonomy_dir),
            "--gcn-bin", GCN_BIN,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    produced = sorted(out_dir.glob("generated_*.json"))
    assert len(produced) == 1, f"1 fichier attendu, {len(produced)} produit(s)\n{result.output}"

    doc = json.loads(produced[0].read_text(encoding="utf-8"))
    sentences = doc["document"]["sentences"]
    assert len(sentences) == 1
    assert sentences[0]["text"] == "Les ventes baissent."
    assert sentences[0]["tokens"], "aucun token synthétique"
    assert sentences[0]["cir"]["nodes"], "aucun nœud CIR"


# ── Tests B1 : données bootstrappées utilisables à l'entraînement ────────────

def test_cir_to_doc_has_tokens():
    """B1 régression : _cir_to_doc doit produire des tokens non vides."""
    from gcn_python.training.bootstrap import _cir_to_doc
    cir = {
        "nodes": [
            {"id": "n001", "node_type": "action", "label": "baisser", "scope": "specific",
             "temporal_index": 0, "origin": "explicit"},
            {"id": "n002", "node_type": "etat", "label": "impact", "scope": "specific",
             "temporal_index": 1, "origin": "explicit"},
        ],
        "edges": []
    }
    doc = _cir_to_doc("test.", cir)
    tokens = doc["document"]["sentences"][0]["tokens"]
    assert len(tokens) == 2, "un token synthétique par nœud"
    assert tokens[0]["id"] == 1
    assert tokens[1]["id"] == 2
    assert tokens[0]["lemma"] == "baisser"
    assert tokens[1]["lemma"] == "impact"
    assert tokens[0]["pos"] == "VERB"
    assert tokens[1]["pos"] == "NOUN"


def test_cir_to_doc_bootstrapped_data_trainable():
    """B1 régression : données bootstrappées doivent produire des reps valides via reps_from_sentence."""
    import json
    import tempfile
    from pathlib import Path

    from gcn_python.data.json_reader import load_sentences
    from gcn_python.data.loader import reps_from_sentence
    from gcn_python.training.bootstrap import _cir_to_doc

    cir = {
        "nodes": [
            {"id": "n001", "node_type": "condition", "label": "hausse des coûts",
             "scope": "specific", "temporal_index": 0, "origin": "explicit"},
            {"id": "n002", "node_type": "action", "label": "réduire budget",
             "scope": "specific", "temporal_index": 1, "origin": "explicit"},
        ],
        "edges": [
            {"source": "n001", "target": "n002", "relation": "cause", "confidence": 0.8}
        ]
    }
    doc = _cir_to_doc("Si les coûts hausse on réduit.", cir)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bootstrapped.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        sentences = load_sentences(path)

    assert len(sentences) == 1, "une phrase dans le doc"
    reps, _valid_idxs, _connector_reps = reps_from_sentence(sentences[0])
    assert len(reps) == 2, (
        f"2 reps attendues, {len(reps)} obtenues — "
        "les données bootstrappées sont ignorées à l'entraînement (correctif B1 manquant)."
    )
    assert reps[0].root_pos == "SCONJ"
    assert reps[1].root_pos == "VERB"
