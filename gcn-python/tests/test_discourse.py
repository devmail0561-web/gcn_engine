# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Éq.10 ETUDE — relations discursives inter-phrasales."""
from gcn_python.layer1.sentence_type import LangMarkers
from gcn_python.pipeline.discourse import extract_discourse_relation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cir(node_ids: list[int], span_ends: list[int]) -> dict:
    """CIR minimal pour les tests."""
    return {
        "nodes": [
            {"id": nid, "source_span": {"token_span": {"start": 0, "end": end}}}
            for nid, end in zip(node_ids, span_ends, strict=False)
        ],
        "edges": [],
    }


def _tok(lemma: str, pos: str, dep_rel: str, tid: int) -> dict:
    return {"lemma": lemma, "pos": pos, "dep_rel": dep_rel, "id": tid}


def _rep_with_tokens(tokens: list[dict]):
    class _FakeRep:
        def __init__(self, toks):
            self.tokens = toks
    return _FakeRep(tokens)


def _markers() -> LangMarkers:
    """Fixture inline — les markers ne viennent pas d'un fichier de config externe."""
    return LangMarkers.from_json({
        "discourse_connectors": {
            "cause": {"lemmas": ["donc", "ainsi", "therefore"], "confidence": 0.80, "inverted": False},
            "cause_inverted": {"lemmas": ["car", "parce que", "because"], "confidence": 0.75, "inverted": True},
            "concession": {"lemmas": ["cependant", "neanmoins", "however"], "confidence": 0.80, "inverted": False},
            "sequence": {"lemmas": ["ensuite", "puis", "then", "next"], "confidence": 0.80, "inverted": False},
        }
    })


# ---------------------------------------------------------------------------
# Sans cir_prev → None
# ---------------------------------------------------------------------------

def test_no_prev_cir_returns_none():
    cir = _cir([0], [5])
    rep = _rep_with_tokens([_tok("donc", "ADV", "advmod", 0)])
    result = extract_discourse_relation(None, cir, [rep])
    assert result is None


def test_empty_prev_cir_returns_none():
    result = extract_discourse_relation({}, _cir([0], [5]), [])
    assert result is None


# ---------------------------------------------------------------------------
# Couche 1 — signal structurel (CCONJ/ADV id≤2)
# ---------------------------------------------------------------------------

def test_cconj_produces_cause_layer1():
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("et", "CCONJ", "cc", 1)])
    result = extract_discourse_relation(prev, curr, [rep])
    assert result is not None
    assert result[2]["relation"] == "cause"
    assert result[2]["confidence"] == 0.50


def test_adv_high_id_no_layer1():
    """ADV avec id > 2 ne déclenche pas la couche 1."""
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("donc", "ADV", "advmod", 5)])
    result = extract_discourse_relation(prev, curr, [rep])
    # Pas de connecteur → juxtaposition
    assert result is not None
    assert result[2]["confidence"] == 0.35


# ---------------------------------------------------------------------------
# Couche 2 — discourse_connectors depuis LangMarkers
# ---------------------------------------------------------------------------

def test_donc_produces_cause_via_markers():
    markers = _markers()
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("donc", "ADV", "advmod", 0)])
    result = extract_discourse_relation(prev, curr, [rep], lang_markers=markers)
    assert result is not None
    assert result[2]["relation"] == "cause"
    assert result[2]["confidence"] == 0.80


def test_cependant_produces_concession_via_markers():
    markers = _markers()
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("cependant", "ADV", "advmod", 0)])
    result = extract_discourse_relation(prev, curr, [rep], lang_markers=markers)
    assert result is not None
    assert result[2]["relation"] == "concession"


def test_ensuite_produces_sequence_via_markers():
    markers = _markers()
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("ensuite", "ADV", "advmod", 1)])
    result = extract_discourse_relation(prev, curr, [rep], lang_markers=markers)
    assert result is not None
    assert result[2]["relation"] == "sequence"


def test_car_inverted_direction():
    """'car' → cause_inverted : direction src/dst inversée."""
    markers = _markers()
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("car", "CCONJ", "cc", 0)])
    result = extract_discourse_relation(prev, curr, [rep], lang_markers=markers)
    assert result is not None
    assert result[2]["relation"] == "cause"
    # direction inversée : src devient le pivot du curr
    assert result[0] == 1   # pivot curr (inverted)
    assert result[1] == 0   # pivot prev


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def test_anaphora_fallback():
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("il", "PRON", "nsubj", 0)])
    result = extract_discourse_relation(prev, curr, [rep])
    assert result is not None
    assert result[2]["relation"] == "cause"
    assert result[2]["confidence"] == 0.50


def test_juxtaposition_fallback():
    prev = _cir([0], [3])
    curr = _cir([1], [7])
    rep = _rep_with_tokens([_tok("le", "DET", "det", 0)])
    result = extract_discourse_relation(prev, curr, [rep])
    assert result is not None
    assert result[2]["relation"] == "cause"
    assert result[2]["confidence"] == 0.35
