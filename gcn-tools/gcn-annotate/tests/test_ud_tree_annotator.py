# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests UDTreeAnnotator : protocole, qualité spans, confiance, types de nœuds."""
import pytest

try:
    import spacy
    spacy.load("fr_core_news_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

pytestmark = pytest.mark.skipif(not HAS_SPACY, reason="spaCy fr_core_news_sm non disponible")

from gcn_annotate.ud_tree_annotator import UDTreeAnnotator, _CONFIDENCE


CAUSAL_FR = [
    "La pluie provoque l'inondation parce que le sol est saturé.",
    "Si la température dépasse 100 degrés alors l'eau bout.",
    "Le gouvernement empêche l'inflation grâce à des taux élevés.",
    "La chaleur permet la croissance des bactéries.",
]

NON_CAUSAL = [
    "Le soleil brille aujourd'hui.",
    "Paris est la capitale de la France.",
]


@pytest.fixture
def annotator():
    return UDTreeAnnotator()


def test_annotator_returns_list(annotator):
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    assert isinstance(result, list)


def test_annotator_empty_input(annotator):
    assert annotator.annotate([], lang="fr") == []


def test_annotator_causal_detected(annotator):
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    # Au moins 50% des phrases causales doivent être détectées
    assert len(result) >= len(CAUSAL_FR) // 2


def test_annotator_non_causal_skipped(annotator):
    result = annotator.annotate(NON_CAUSAL, lang="fr")
    assert len(result) == 0


def test_annotator_cir_structure(annotator):
    result = annotator.annotate(CAUSAL_FR[:1], lang="fr")
    if not result:
        pytest.skip("phrase non détectée")
    ann = result[0]
    assert "cir" in ann
    cir = ann["cir"]
    assert "nodes" in cir and "edges" in cir
    assert len(cir["nodes"]) == 2
    assert len(cir["edges"]) == 1


def test_annotator_spans_valid(annotator):
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    for ann in result:
        nodes = ann["cir"]["nodes"]
        spans = []
        for n in nodes:
            s, e = n["token_span"]
            assert s >= 1, f"span start < 1 : {n['id']} dans {ann['text']!r}"
            assert s <= e, f"span start > end : {n['id']}"
            spans.append((s, e))
        assert len(spans) == len(set(spans)), "spans dupliqués"


def test_annotator_confidence_in_range(annotator):
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    for ann in result:
        for edge in ann["cir"]["edges"]:
            conf = edge["attributes"]["confidence"]
            assert 0.5 <= conf <= 1.0, f"confiance hors [0.5, 1.0] : {conf}"


def test_annotator_relation_types_valid(annotator):
    from gcn_annotate.annotator import RELATION_TYPES
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    for ann in result:
        for edge in ann["cir"]["edges"]:
            assert edge["relation"] in RELATION_TYPES, f"relation inconnue : {edge['relation']}"


def test_annotator_node_types_valid(annotator):
    from gcn_annotate.annotator import NODE_TYPES
    result = annotator.annotate(CAUSAL_FR, lang="fr")
    for ann in result:
        for node in ann["cir"]["nodes"]:
            assert node["type"] in NODE_TYPES, f"type inconnu : {node['type']}"


def test_confidence_calibration():
    assert _CONFIDENCE["cause"] >= _CONFIDENCE["sequence"]
    assert all(0.5 <= v <= 1.0 for v in _CONFIDENCE.values())


def test_annotator_protocol_compatible(annotator):
    """Vérifie que UDTreeAnnotator respecte le protocole LLMAnnotator."""
    from gcn_annotate.annotator import LLMAnnotator
    assert isinstance(annotator, LLMAnnotator)
