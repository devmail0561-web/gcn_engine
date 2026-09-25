# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests invariants token_span : 1-based, start≤end, spans distincts entre nœuds."""
import pytest
from gcn_annotate.auto_annotate import analyze_sentence as auto_analyze
from gcn_annotate.smart_annotate import annotate_sentence as smart_annotate


def _check_spans(nodes: list[dict], text: str, strict_bound: bool = False) -> None:
    """strict_bound=True vérifie e <= len(text.split()) — valide seulement si les
    tokens sont ceux de text.split() (auto_annotate). smart_annotate utilise les
    tokens spaCy (ponctuation incluse) donc T peut dépasser len(text.split())."""
    T = len(text.split())
    spans = []
    for n in nodes:
        s, e = n["token_span"]
        assert s >= 1, f"span start < 1 : {n['id']} token_span={n['token_span']} dans {text!r}"
        assert s <= e, f"span start > end : {n['id']} token_span={n['token_span']}"
        if strict_bound:
            assert e <= T, f"span end > T={T} : {n['id']} token_span={n['token_span']}"
        spans.append((s, e))
    assert len(spans) == len(set(spans)), f"spans dupliqués : {spans}"


FR_SENTENCES = [
    "La pluie provoque l'inondation parce que le sol est saturé.",
    "Si la température dépasse 100 degrés alors l'eau bout.",
    "Bien que le chômage diminue la pauvreté persiste.",
    "Le gouvernement empêche l'inflation grâce à des taux élevés.",
    "La chaleur permet la croissance des bactéries.",
]

EN_SENTENCES = [
    "The rain causes flooding because the soil is saturated.",
    "If the temperature exceeds 100 degrees then the water boils.",
    "Although unemployment decreases poverty persists.",
]


@pytest.mark.parametrize("text", FR_SENTENCES + EN_SENTENCES)
def test_auto_annotate_spans_valid(text):
    result = auto_analyze(text)
    if result is None:
        pytest.skip("phrase non annotée")
    _check_spans(result["nodes"], text, strict_bound=True)


@pytest.mark.parametrize("text", FR_SENTENCES + EN_SENTENCES)
def test_smart_annotate_spans_valid(text):
    try:
        import spacy
        nlp = spacy.load("fr_core_news_sm")
    except Exception:
        pytest.skip("spaCy fr_core_news_sm non disponible")
    result = smart_annotate(text, nlp)
    if result is None:
        pytest.skip("phrase non annotée")
    _check_spans(result["cir"]["nodes"], text, strict_bound=False)
