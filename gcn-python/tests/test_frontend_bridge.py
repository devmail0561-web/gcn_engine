"""Tests pour frontend/bridge.py — implémentation P1 (pont texte brut → UDRepresentation)."""
import json
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gcn_python.frontend.bridge import (
    GCNBridgeError,
    NODE_TYPE_TO_DEP,
    NODE_TYPE_TO_POS,
    _build_connector_rep,
    _cir_to_reps_and_connectors,
    _extract_lemma,
    _extract_span,
    _parse_edges,
    _rep_from_cir_node,
    reps_from_text,
)

# ── Fixture CIR de référence ──────────────────────────────────────────────────

_CIR_TWO_NODES = {
    "source_lang": {"natural": {"lang": "french"}},
    "source_text": "Les ventes baissent donc les prix augmentent.",
    "nodes": [
        {
            "id": 0, "node_type": "action", "label": "baisse(ventes)",
            "source_span": {"token_span": {"start": 1, "end": 3}},
            "scope": "specific", "temporal_index": 0, "temporal_ref": "unresolved",
            "origin": "explicit",
            "attributes": {"entity": "ventes", "quality": None, "agent": None, "patient": None},
        },
        {
            "id": 1, "node_type": "etat", "label": "hausse(prix)",
            "source_span": {"token_span": {"start": 5, "end": 7}},
            "scope": "specific", "temporal_index": 1, "temporal_ref": "unresolved",
            "origin": "explicit",
            "attributes": {"entity": "prix", "quality": "hausse", "agent": None, "patient": None},
        },
    ],
    "edges": [[0, 1, {"relation": "cause", "confidence": 1.0,
                      "explicit": True, "negated": False, "marker_token": 4}]],
}


# ── Tests mappings heuristiques ───────────────────────────────────────────────


def test_node_type_to_pos_all_types():
    """Les 7 node_types produisent VERB/NOUN/SCONJ selon la logique attendue."""
    assert NODE_TYPE_TO_POS["action"] == "VERB"
    assert NODE_TYPE_TO_POS["transition"] == "VERB"
    assert NODE_TYPE_TO_POS["processus"] == "NOUN"
    assert NODE_TYPE_TO_POS["etat"] == "NOUN"
    assert NODE_TYPE_TO_POS["etat_systemique"] == "NOUN"
    assert NODE_TYPE_TO_POS["entite"] == "NOUN"
    assert NODE_TYPE_TO_POS["condition"] == "SCONJ"


def test_node_type_to_dep_all_types():
    """Les 7 node_types produisent root/nsubj/advcl selon la logique attendue."""
    assert NODE_TYPE_TO_DEP["action"] == "root"
    assert NODE_TYPE_TO_DEP["transition"] == "root"
    assert NODE_TYPE_TO_DEP["processus"] == "root"
    assert NODE_TYPE_TO_DEP["etat"] == "nsubj"
    assert NODE_TYPE_TO_DEP["etat_systemique"] == "nsubj"
    assert NODE_TYPE_TO_DEP["entite"] == "nsubj"
    assert NODE_TYPE_TO_DEP["condition"] == "advcl"


# ── Tests _extract_lemma ─────────────────────────────────────────────────────


def test_extract_lemma_with_parenthesis():
    assert _extract_lemma("décroissance(ventes)") == "décroissance"


def test_extract_lemma_simple():
    assert _extract_lemma("hausse") == "hausse"


def test_extract_lemma_with_question():
    assert _extract_lemma("cause_cachée(?)") == "cause_cachée"


def test_extract_lemma_empty():
    assert _extract_lemma("") == "_unknown"


def test_extract_lemma_whitespace():
    assert _extract_lemma("   ") == "_unknown"


def test_extract_lemma_question_mark_only():
    """Label dégénéré '(?)' — ne doit pas crasher."""
    result = _extract_lemma("(?)")
    assert isinstance(result, str)
    assert len(result) > 0


# ── Tests _extract_span ───────────────────────────────────────────────────────


def test_extract_span_dict_format():
    node = {"source_span": {"token_span": {"start": 2, "end": 7}}}
    assert _extract_span(node) == (2, 7)


def test_extract_span_flat_format():
    node = {"token_span": [2, 7]}
    assert _extract_span(node) == (2, 7)


def test_extract_span_null_values():
    node = {"source_span": {"token_span": {"start": None, "end": None}}}
    assert _extract_span(node) == (0, 0)


def test_extract_span_missing():
    assert _extract_span({}) == (0, 0)


# ── Tests _rep_from_cir_node ─────────────────────────────────────────────────


def test_rep_from_cir_node_action():
    """action → VERB, root, has_advcl=False."""
    node = {"id": 0, "node_type": "action", "label": "baisse(ventes)",
            "source_span": {"token_span": {"start": 1, "end": 3}},
            "temporal_ref": "unresolved",
            "attributes": {"entity": "ventes", "agent": None, "patient": None}}
    rep = _rep_from_cir_node(node)
    assert rep.root_pos == "VERB"
    assert rep.root_dep_rel == "root"
    assert rep.has_advcl is False
    assert rep.root_morph == {}


def test_rep_from_cir_node_etat():
    """etat → NOUN, nsubj, has_advcl=False."""
    node = {"id": 1, "node_type": "etat", "label": "hausse(prix)",
            "source_span": {"token_span": {"start": 5, "end": 7}},
            "temporal_ref": "unresolved",
            "attributes": {"entity": "prix", "agent": None, "patient": None}}
    rep = _rep_from_cir_node(node)
    assert rep.root_pos == "NOUN"
    assert rep.root_dep_rel == "nsubj"
    assert rep.has_advcl is False


def test_rep_from_cir_node_condition():
    """condition → SCONJ, advcl, has_advcl=False (CIR ne porte pas l'info sur sous-clauses)."""
    node = {"id": 2, "node_type": "condition", "label": "si(ventes)",
            "source_span": {"token_span": {"start": 0, "end": 2}},
            "temporal_ref": "unresolved",
            "attributes": {"entity": None, "agent": None, "patient": None}}
    rep = _rep_from_cir_node(node)
    assert rep.root_pos == "SCONJ"
    assert rep.root_dep_rel == "advcl"
    assert rep.has_advcl is False  # conservative : CIR ne porte pas cette info


def test_rep_from_cir_node_patient_has_object():
    """patient non-null → has_object=True."""
    node = {"id": 0, "node_type": "action", "label": "réduire(coûts)",
            "temporal_ref": "unresolved",
            "attributes": {"entity": None, "agent": None, "patient": "coûts"}}
    rep = _rep_from_cir_node(node)
    assert rep.has_object is True


def test_rep_from_cir_node_agent_subject_pos():
    """agent non-null → subject_pos='NOUN'."""
    node = {"id": 0, "node_type": "action", "label": "investit",
            "temporal_ref": "unresolved",
            "attributes": {"entity": None, "agent": "entreprise", "patient": None}}
    rep = _rep_from_cir_node(node)
    assert rep.subject_pos == "NOUN"


def test_rep_from_cir_node_no_agent():
    """agent=null → subject_pos=None."""
    node = {"id": 0, "node_type": "etat", "label": "hausse",
            "temporal_ref": "unresolved",
            "attributes": {"entity": None, "agent": None, "patient": None}}
    rep = _rep_from_cir_node(node)
    assert rep.subject_pos is None


def test_rep_morph_always_empty():
    """root_morph={} pour tout node_type → tense/mood/aspect = _absent."""
    for nt in NODE_TYPE_TO_POS:
        node = {"id": 0, "node_type": nt, "label": "test",
                "temporal_ref": "unresolved", "attributes": {}}
        rep = _rep_from_cir_node(node)
        assert rep.root_morph == {}
        assert rep.tense == "_absent"
        assert rep.mood == "_absent"
        assert rep.is_negative is False


def test_rep_temporal_obl_resolved():
    """temporal_ref différent de 'unresolved' → has_temporal_obl=True."""
    node = {"id": 0, "node_type": "processus", "label": "croissance",
            "temporal_ref": {"range": {"start": -30, "end": 0}},
            "attributes": {}}
    rep = _rep_from_cir_node(node)
    assert rep.has_temporal_obl is True


# ── Tests _parse_edges ────────────────────────────────────────────────────────


def test_parse_edges_tuple_format():
    """Format tuple [src, dst, obj] → (int, int, dict)."""
    edges = [[0, 1, {"relation": "cause", "confidence": 1.0}]]
    result = _parse_edges(edges)
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == 1
    assert result[0][2]["relation"] == "cause"


def test_parse_edges_dict_format():
    """Format dict avec source/target string → (int, int, dict)."""
    edges = [{"source": "0", "target": "1", "relation": "enable"}]
    result = _parse_edges(edges)
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == 1


def test_parse_edges_invalid_ignored():
    """Formats invalides → liste vide, pas de crash."""
    assert _parse_edges(["invalid"]) == []
    assert _parse_edges([42]) == []
    assert _parse_edges([[0, 1]]) == []  # seulement 2 éléments
    assert _parse_edges([{}]) == []  # dict sans source/target


# ── Tests connecteurs ─────────────────────────────────────────────────────────


def test_connector_from_marker_token():
    """marker_token=4 → UDRep SCONJ avec token_span=(4,4)."""
    reps, connectors = _cir_to_reps_and_connectors(_CIR_TWO_NODES)
    assert len(connectors) == 1
    assert connectors[0] is not None
    assert connectors[0].root_pos == "SCONJ"
    assert connectors[0].root_dep_rel == "mark"
    assert connectors[0].token_span == (4, 4)


def test_connector_none_without_marker():
    """Pas de marker_token → None dans la liste des connecteurs."""
    cir = {
        "nodes": [
            {"id": 0, "node_type": "action", "label": "A", "temporal_ref": "unresolved", "attributes": {}},
            {"id": 1, "node_type": "etat", "label": "B", "temporal_ref": "unresolved", "attributes": {}},
        ],
        "edges": [[0, 1, {"relation": "cause", "confidence": 1.0, "marker_token": None}]],
    }
    reps, connectors = _cir_to_reps_and_connectors(cir)
    assert len(connectors) == 1
    assert connectors[0] is None


def test_connector_marker_zero_ignored():
    """marker_token=0 → None (valeur fictive, ignorée)."""
    cir = {
        "nodes": [
            {"id": 0, "node_type": "action", "label": "A", "temporal_ref": "unresolved", "attributes": {}},
            {"id": 1, "node_type": "etat", "label": "B", "temporal_ref": "unresolved", "attributes": {}},
        ],
        "edges": [[0, 1, {"relation": "cause", "confidence": 1.0, "marker_token": 0}]],
    }
    reps, connectors = _cir_to_reps_and_connectors(cir)
    assert connectors[0] is None


def test_cir_empty_nodes():
    """nodes=[] → ([], []) sans exception."""
    reps, connectors = _cir_to_reps_and_connectors({"nodes": [], "edges": []})
    assert reps == []
    assert connectors == []


# ── Tests reps_from_text (avec mock subprocess) ───────────────────────────────


def _make_mock_result(cir_dict, returncode=0, stderr=""):
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = json.dumps(cir_dict)
    mock.stderr = stderr
    return mock


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_mocked_success(mock_run):
    """Workflow complet mocké → 2 reps avec les bons root_pos."""
    mock_run.return_value = _make_mock_result(_CIR_TWO_NODES)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        reps = reps_from_text("Les ventes baissent donc les prix augmentent.")
    assert len(reps) == 2
    assert reps[0].root_pos == "VERB"   # action → VERB
    assert reps[1].root_pos == "NOUN"   # etat → NOUN


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_quality_warning(mock_run):
    """UserWarning émis systématiquement pour informer de la qualité approximative."""
    mock_run.return_value = _make_mock_result(_CIR_TWO_NODES)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reps_from_text("test")
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warnings) >= 1
    assert "APPROXIMATIFS" in str(user_warnings[0].message)


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_gcn_not_found(mock_run):
    """FileNotFoundError → GCNBridgeError avec 'introuvable' dans le message."""
    mock_run.side_effect = FileNotFoundError
    with pytest.raises(GCNBridgeError, match="introuvable"):
        with warnings.catch_warnings(record=True):
            reps_from_text("test")


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_timeout(mock_run):
    """TimeoutExpired → GCNBridgeError avec 'Timeout' dans le message."""
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="gcn", timeout=30)
    with pytest.raises(GCNBridgeError, match="Timeout"):
        with warnings.catch_warnings(record=True):
            reps_from_text("test")


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_nonzero_return(mock_run):
    """returncode=1 → GCNBridgeError avec 'échoué' dans le message."""
    mock_run.return_value = _make_mock_result({}, returncode=1, stderr="erreur")
    with pytest.raises(GCNBridgeError, match="échoué"):
        with warnings.catch_warnings(record=True):
            reps_from_text("test")


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_reps_from_text_invalid_json(mock_run):
    """Sortie non-JSON → GCNBridgeError avec 'JSON' dans le message."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "not valid json"
    mock_run.return_value = mock
    with pytest.raises(GCNBridgeError, match="JSON"):
        with warnings.catch_warnings(record=True):
            reps_from_text("test")


@pytest.mark.skipif(shutil.which("gcn") is None, reason="Requiert gcn-cli installé")
def test_reps_from_text_integration():
    """Test d'intégration complet avec vrai binaire gcn."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        reps = reps_from_text("Les ventes baissent.")
    assert len(reps) >= 1
    for rep in reps:
        assert rep.root_pos in {"VERB", "NOUN", "SCONJ", "ADJ", "_unk"}
        assert rep.lang == "fr"


# ── Tests CGNPipeline.analyze() et analyze_or_skip() ─────────────────────────


def _make_pipeline():
    """Helper : pipeline ML minimal pour tests."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    vocab = FeatureVocabulary()
    return CGNPipeline(
        encoder=MLPEncoder(vocab.d_clause, vocab.d_edge),
        graph=RGCNLayer(vocab.d_clause, vocab.d_clause),
        vocabulary=vocab,
    )


@patch("gcn_python.frontend.bridge.subprocess.run")
def test_pipeline_analyze_returns_cir(mock_run):
    """pipeline.analyze() retourne un CausalIR dict avec les champs attendus."""
    mock_run.return_value = _make_mock_result(_CIR_TWO_NODES)
    pipeline = _make_pipeline()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cir = pipeline.analyze("Les ventes baissent.")
    assert "nodes" in cir
    assert "edges" in cir
    assert len(cir["nodes"]) == 2


def test_pipeline_analyze_or_skip_none_without_binary():
    """analyze_or_skip() retourne None si le binaire gcn est absent."""
    pipeline = _make_pipeline()
    result = pipeline.analyze_or_skip("test", gcn_bin="nonexistent-gcn-binary-xyz")
    assert result is None


def test_pipeline_analyze_or_skip_no_double_warning():
    """analyze_or_skip() n'émet aucun UserWarning (dégradation gracieuse silencieuse)."""
    pipeline = _make_pipeline()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pipeline.analyze_or_skip("test", gcn_bin="nonexistent-gcn-binary-xyz")
    assert result is None
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warnings) == 0, f"Warning inattendu : {[str(x.message) for x in user_warnings]}"
