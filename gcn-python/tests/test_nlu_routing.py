# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Éq.6 ETUDE — NLU routing langue-agnostique."""
import numpy as np
import pytest
from gcn_python.constants import INTENT_TYPES
from gcn_python.pipeline.nlu_routing import nlu_route, _build_dsl, _content_words


# ---------------------------------------------------------------------------
# Heuristique surface — priorité 2 (sans pipeline)
# ---------------------------------------------------------------------------

def test_interrogative_surface_routes():
    result = nlu_route("Pourquoi X ?")
    assert result is not None
    assert "explain" in result or "chain" in result or "x" in result


def test_declarative_returns_none():
    assert nlu_route("Le moteur analyse les données.") is None
    assert nlu_route("A cause B.") is None


def test_two_distinct_concepts_chain():
    # Heuristique surface : premier et dernier content word (pas de détection wh-word)
    result = nlu_route("Comment alpha mène à beta ?")
    assert result is not None
    assert result.startswith("chain:")
    assert "beta" in result


def test_single_word_question_explain():
    # Un seul content word → explain
    result = nlu_route("ventes ?")
    assert result is not None
    assert result.startswith("explain:")
    assert "ventes" in result


def test_empty_content_returns_none():
    assert nlu_route("??") is None


# ---------------------------------------------------------------------------
# Couche 1 UD — si tokens fournis (INTERROGATIVE détecté structurellement)
# ---------------------------------------------------------------------------

def _tok_interrogative():
    return [
        {"lemma": "pourquoi", "pos": "ADV", "dep_rel": "advmod",
         "morph": {"PronType": "Int"}, "id": 0},
        {"lemma": "ventes", "pos": "NOUN", "dep_rel": "nsubj",
         "morph": {}, "id": 1},
    ]


def test_tokens_ud_interrogative_routes():
    result = nlu_route("Pourquoi ventes ?", tokens=_tok_interrogative())
    assert result is not None


def test_tokens_ud_declarative_returns_none():
    tokens = [{"lemma": "analyser", "pos": "VERB", "dep_rel": "root",
               "morph": {"Mood": "Ind"}, "id": 0}]
    assert nlu_route("Le moteur analyse.", tokens=tokens) is None


# ---------------------------------------------------------------------------
# Priorité 1 — tête apprise (_cached_intent_logits)
# ---------------------------------------------------------------------------

class _MockPipeline:
    def __init__(self, intent_idx: int):
        logits = np.zeros((1, len(INTENT_TYPES)), dtype=np.float32)
        logits[0, intent_idx] = 10.0
        self._cached_intent_logits = logits


def test_intent_logits_override_surface():
    """La tête apprise prend la priorité sur l'heuristique surface."""
    chain_idx = INTENT_TYPES.index("chain")
    pipeline = _MockPipeline(chain_idx)
    result = nlu_route("Pourquoi X ?", pipeline=pipeline)  # surface → explain
    assert result is not None
    assert result.startswith("chain:")  # tête → chain


def test_intent_none_returns_none():
    none_idx = INTENT_TYPES.index("none")
    pipeline = _MockPipeline(none_idx)
    assert nlu_route("Le moteur analyse.", pipeline=pipeline) is None


# ---------------------------------------------------------------------------
# _build_dsl — tous les 21 intents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", INTENT_TYPES)
def test_build_dsl_all_intents(intent):
    if intent == "none":
        return  # none ne génère pas de DSL
    concepts = ["alpha", "beta"]
    result = _build_dsl(intent, concepts)
    assert result is not None
    assert result.startswith(f"{intent}:")


def test_build_dsl_two_concept_intents():
    for intent in ["chain", "chain_t", "before", "delay", "diff", "analogy"]:
        result = _build_dsl(intent, ["alpha", "beta"])
        assert "alpha" in result and "beta" in result, f"{intent}: {result}"


def test_build_dsl_one_concept_intents():
    for intent in ["explain", "effects", "summarize", "verbalize"]:
        result = _build_dsl(intent, ["alpha"])
        assert "alpha" in result


def test_build_dsl_empty_concepts_returns_none():
    assert _build_dsl("explain", []) is None
