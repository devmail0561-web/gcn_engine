# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase A — sentence_type.py (D1) : classify, LangMarkers, SentenceProfile."""
from __future__ import annotations

from pathlib import Path

import pytest

from gcn_python.layer1.sentence_type import (
    Complexity,
    LangMarkers,
    Modality,
    Polarity,
    SentenceProfile,
    SentenceType,
    SubordinationType,
    Voice,
    classify,
)

LANG_MARKERS_PATH = Path(__file__).resolve().parents[2] / "gcn-datasets/configs/lang_markers.json"
SOURCE_PATH = Path(__file__).resolve().parents[1] / "src/gcn_python/layer1/sentence_type.py"


def tok(lemma="x", pos="VERB", dep_rel="root", morph=None, id=1, dep_head=0, form="x"):
    return {"lemma": lemma, "pos": pos, "dep_rel": dep_rel,
            "morph": morph or {}, "id": id, "dep_head": dep_head, "form": form}


# ---------------------------------------------------------------------------
# TestSentenceProfile
# ---------------------------------------------------------------------------

def test_profile_frozen():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    with pytest.raises((AttributeError, TypeError)):
        p.sentence_type = SentenceType.INTERROGATIVE  # type: ignore[misc]


def test_profile_expects_response_true():
    p = SentenceProfile(SentenceType.INTERROGATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert p.expects_response is True


def test_profile_expects_response_false():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert p.expects_response is False


def test_profile_expects_action():
    p = SentenceProfile(SentenceType.IMPERATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.IMPERATIVE)
    assert p.expects_action is True


def test_profile_asserts_fact():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert p.asserts_fact is True


def test_profile_is_negated():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.NEGATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert p.is_negated is True


def test_profile_is_passive():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.PASSIVE, Modality.INDICATIVE)
    assert p.is_passive is True


def test_profile_is_complex():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE,
                        Modality.INDICATIVE, Complexity.COMPLEX)
    assert p.is_complex is True


def test_profile_has_condition():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE,
                        Modality.INDICATIVE, Complexity.COMPLEX, SubordinationType.CONDITION)
    assert p.has_condition is True


def test_profile_equality():
    p1 = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    p2 = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert p1 == p2


def test_profile_repr_nonempty():
    p = SentenceProfile(SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE, Voice.ACTIVE, Modality.INDICATIVE)
    assert repr(p) != ""


# ---------------------------------------------------------------------------
# TestClassifyTypeUD
# ---------------------------------------------------------------------------

def test_question_mark_interrogative():
    tokens = [tok(form="Viens", dep_rel="root"), tok(form="?", pos="PUNCT", dep_rel="punct")]
    assert classify(tokens).sentence_type == SentenceType.INTERROGATIVE


def test_prontype_int_interrogative():
    tokens = [
        tok(lemma="qui", pos="PRON", dep_rel="advmod", morph={"PronType": "Int"}, id=1),
        tok(lemma="venir", pos="VERB", dep_rel="root", id=2),
    ]
    assert classify(tokens).sentence_type == SentenceType.INTERROGATIVE


def test_mood_imp_imperative():
    tokens = [tok(morph={"Mood": "Imp"}, dep_rel="root")]
    assert classify(tokens).sentence_type == SentenceType.IMPERATIVE


def test_verbform_inf_no_subject_imperative():
    tokens = [tok(morph={"VerbForm": "Inf"}, dep_rel="root")]
    assert classify(tokens).sentence_type == SentenceType.IMPERATIVE


def test_empty_tokens_declarative():
    assert classify([]).sentence_type == SentenceType.DECLARATIVE


def test_inversion_subject_after_root_interrogative():
    tokens = [
        tok(lemma="venir", pos="VERB", dep_rel="root", id=1),
        tok(lemma="tu", pos="PRON", dep_rel="nsubj", id=2),
    ]
    assert classify(tokens).sentence_type == SentenceType.INTERROGATIVE


def test_exclamation_mark_exclamative():
    tokens = [tok(form="x", dep_rel="root"), tok(form="!", pos="PUNCT", dep_rel="punct")]
    assert classify(tokens).sentence_type == SentenceType.EXCLAMATIVE


def test_normal_declarative():
    tokens = [
        tok(lemma="le", pos="DET", dep_rel="det", id=1),
        tok(lemma="chat", pos="NOUN", dep_rel="nsubj", id=2),
        tok(lemma="manger", pos="VERB", dep_rel="root", morph={"Mood": "Ind"}, id=3),
    ]
    assert classify(tokens).sentence_type == SentenceType.DECLARATIVE


# ---------------------------------------------------------------------------
# TestClassifyTypeMarkers
# ---------------------------------------------------------------------------

def _fr_markers():
    return LangMarkers(interrogative_lemmas=frozenset(["est-ce-que", "pourquoi", "comment"]))


def _en_markers():
    return LangMarkers(interrogative_lemmas=frozenset(["what", "who", "where"]))


def test_markers_fr_interrogative():
    tokens = [tok(lemma="est-ce-que", pos="SCONJ", dep_rel="mark", id=1),
              tok(lemma="venir", pos="VERB", dep_rel="root", id=2)]
    assert classify(tokens, _fr_markers()).sentence_type == SentenceType.INTERROGATIVE


def test_markers_en_interrogative():
    tokens = [tok(lemma="what", pos="PRON", dep_rel="obj", id=1),
              tok(lemma="do", pos="AUX", dep_rel="aux", id=2),
              tok(lemma="you", pos="PRON", dep_rel="nsubj", id=3),
              tok(lemma="want", pos="VERB", dep_rel="root", id=4)]
    assert classify(tokens, _en_markers()).sentence_type == SentenceType.INTERROGATIVE


def test_no_markers_lemme_not_detected():
    tokens = [tok(lemma="pourquoi", pos="ADV", dep_rel="advmod", id=1),
              tok(lemma="venir", pos="VERB", dep_rel="root", id=2)]
    # Sans markers, "pourquoi" lemme seul ne déclenche pas INTERROGATIVE (couche 1 seule)
    # (sauf si PronType=Int — ici absent)
    assert classify(tokens, None).sentence_type == SentenceType.DECLARATIVE


def test_empty_markers_no_change():
    tokens = [tok(lemma="x", pos="VERB", dep_rel="root")]
    assert classify(tokens, LangMarkers()).sentence_type == SentenceType.DECLARATIVE


def test_layer2_only_if_layer1_missed():
    # Couche 1 détecte ? → INTERROGATIVE sans markers
    tokens = [tok(dep_rel="root", id=1), tok(form="?", pos="PUNCT", dep_rel="punct", id=2)]
    assert classify(tokens, None).sentence_type == SentenceType.INTERROGATIVE


# ---------------------------------------------------------------------------
# TestClassifyPolarity
# ---------------------------------------------------------------------------

def test_polarity_neg_root_morph():
    tokens = [tok(morph={"Polarity": "Neg"}, dep_rel="root")]
    assert classify(tokens).polarity == Polarity.NEGATIVE


def test_polarity_neg_token_morph():
    tokens = [
        tok(dep_rel="root", id=1),
        tok(pos="PART", dep_rel="advmod", morph={"Polarity": "Neg"}, id=2),
    ]
    assert classify(tokens).polarity == Polarity.NEGATIVE


def test_polarity_neg_via_markers():
    markers = LangMarkers(negation_particles=frozenset(["pas", "ne"]))
    tokens = [
        tok(dep_rel="root", id=1),
        tok(lemma="pas", pos="PART", dep_rel="advmod", id=2),
    ]
    assert classify(tokens, markers).polarity == Polarity.NEGATIVE


def test_polarity_affirmative_default():
    tokens = [tok(dep_rel="root")]
    assert classify(tokens).polarity == Polarity.AFFIRMATIVE


# ---------------------------------------------------------------------------
# TestClassifyVoice
# ---------------------------------------------------------------------------

def test_voice_pass_root_morph():
    tokens = [tok(morph={"Voice": "Pass"}, dep_rel="root")]
    assert classify(tokens).voice == Voice.PASSIVE


def test_voice_aux_pass():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="aux:pass", id=2)]
    assert classify(tokens).voice == Voice.PASSIVE


def test_voice_nsubj_pass():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="nsubj:pass", id=2)]
    assert classify(tokens).voice == Voice.PASSIVE


def test_voice_active_default():
    tokens = [tok(dep_rel="root")]
    assert classify(tokens).voice == Voice.ACTIVE


# ---------------------------------------------------------------------------
# TestClassifyComplexity
# ---------------------------------------------------------------------------

def test_complexity_advcl():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="advcl", morph={"VerbForm": "Fin"}, id=2)]
    assert classify(tokens).complexity == Complexity.COMPLEX


def test_complexity_ccomp():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="ccomp", id=2)]
    assert classify(tokens).complexity == Complexity.COMPLEX


def test_complexity_xcomp():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="xcomp", id=2)]
    assert classify(tokens).complexity == Complexity.COMPLEX


def test_complexity_acl():
    tokens = [tok(dep_rel="root", id=1), tok(dep_rel="acl", id=2)]
    assert classify(tokens).complexity == Complexity.COMPLEX


def test_complexity_mark_sconj():
    tokens = [tok(dep_rel="root", id=1), tok(pos="SCONJ", dep_rel="mark", id=2)]
    assert classify(tokens).complexity == Complexity.COMPLEX


def test_complexity_simple_default():
    tokens = [tok(dep_rel="root")]
    assert classify(tokens).complexity == Complexity.SIMPLE


# ---------------------------------------------------------------------------
# TestClassifySubordination
# ---------------------------------------------------------------------------

def _sub_markers():
    return LangMarkers(subordination_markers={
        "condition": ["si", "unless"],
        "cause": ["parce", "car"],
        "concession": ["bien que", "malgre"],
    })


def test_subordination_none_without_markers():
    tokens = [tok(lemma="si", pos="SCONJ", dep_rel="mark", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens).subordination == SubordinationType.NONE


def test_subordination_condition_si():
    tokens = [tok(lemma="si", pos="SCONJ", dep_rel="mark", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens, _sub_markers()).subordination == SubordinationType.CONDITION


def test_subordination_cause_parce():
    tokens = [tok(lemma="parce", pos="SCONJ", dep_rel="mark", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens, _sub_markers()).subordination == SubordinationType.CAUSE


def test_subordination_concession():
    tokens = [tok(lemma="malgre", pos="SCONJ", dep_rel="mark", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens, _sub_markers()).subordination == SubordinationType.CONCESSION


def test_subordination_unknown_type_ignored():
    markers = LangMarkers(subordination_markers={"unknown_type_xyz": ["xyz"]})
    tokens = [tok(lemma="xyz", pos="SCONJ", dep_rel="mark", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens, markers).subordination == SubordinationType.NONE


def test_subordination_lemma_in_sconj_pos():
    tokens = [tok(lemma="si", pos="SCONJ", dep_rel="advmod", id=1), tok(dep_rel="root", id=2)]
    assert classify(tokens, _sub_markers()).subordination == SubordinationType.CONDITION


def test_subordination_first_match_returned():
    # Both condition and cause markers present — first match in dict iteration
    tokens = [
        tok(lemma="si", pos="SCONJ", dep_rel="mark", id=1),
        tok(lemma="parce", pos="SCONJ", dep_rel="mark", id=2),
        tok(dep_rel="root", id=3),
    ]
    result = classify(tokens, _sub_markers()).subordination
    assert result in (SubordinationType.CONDITION, SubordinationType.CAUSE)


# ---------------------------------------------------------------------------
# TestClassifyRestriction
# ---------------------------------------------------------------------------

def test_restriction_false_without_markers():
    tokens = [tok(lemma="ne", id=1), tok(lemma="que", id=2), tok(dep_rel="root", id=3)]
    assert classify(tokens).has_restriction is False


def test_restriction_true_ne_que():
    markers = LangMarkers(restriction_patterns=(["ne", "que"],))
    tokens = [tok(lemma="ne", id=1), tok(lemma="que", id=2), tok(dep_rel="root", id=3)]
    assert classify(tokens, markers).has_restriction is True


def test_restriction_false_pattern_absent():
    markers = LangMarkers(restriction_patterns=(["ne", "que"],))
    tokens = [tok(lemma="le", id=1), tok(lemma="chat", id=2), tok(dep_rel="root", id=3)]
    assert classify(tokens, markers).has_restriction is False


def test_restriction_false_empty_patterns():
    markers = LangMarkers(restriction_patterns=())
    tokens = [tok(lemma="ne", id=1), tok(lemma="que", id=2), tok(dep_rel="root", id=3)]
    assert classify(tokens, markers).has_restriction is False


# ---------------------------------------------------------------------------
# TestLangMarkers
# ---------------------------------------------------------------------------

def test_lang_markers_from_json():
    data = {
        "interrogative_lemmas": ["POURQUOI", "comment"],
        "negation_particles": ["ne", "pas"],
        "restriction_patterns": [["ne", "que"]],
        "subordination_markers": {"condition": ["si"]},
    }
    m = LangMarkers.from_json(data)
    assert "pourquoi" in m.interrogative_lemmas
    assert "ne" in m.negation_particles
    assert m.restriction_patterns == (["ne", "que"],) or ["ne", "que"] in m.restriction_patterns


def test_lang_markers_from_json_basic():
    """LangMarkers se construit depuis un dict inline — pas de fichier requis."""
    m = LangMarkers.from_json({
        "interrogative_lemmas": ["pourquoi", "comment"],
        "negation_particles": ["ne", "pas"],
    })
    assert len(m.interrogative_lemmas) > 0
    assert len(m.negation_particles) > 0


def test_lang_markers_frozen():
    m = LangMarkers()
    with pytest.raises((AttributeError, TypeError)):
        m.interrogative_lemmas = frozenset(["x"])  # type: ignore[misc]


def test_lang_markers_from_json_lowercase():
    data = {"interrogative_lemmas": ["QUOI", "QUI", "COMMENT"]}
    m = LangMarkers.from_json(data)
    assert "quoi" in m.interrogative_lemmas
    assert "QUOI" not in m.interrogative_lemmas


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

def test_empty_tokens_full_defaults():
    p = classify([])
    assert p.sentence_type == SentenceType.DECLARATIVE
    assert p.polarity == Polarity.AFFIRMATIVE
    assert p.voice == Voice.ACTIVE
    assert p.modality == Modality.INDICATIVE
    assert p.complexity == Complexity.SIMPLE
    assert p.subordination == SubordinationType.NONE
    assert p.has_restriction is False


def test_tokens_four_keys_no_crash():
    tokens = [{"lemma": "x", "pos": "VERB", "dep_rel": "root", "morph": {}}]
    p = classify(tokens)
    assert p.sentence_type == SentenceType.DECLARATIVE


def test_tokens_seven_keys_no_crash():
    tokens = [{"lemma": "x", "pos": "VERB", "dep_rel": "root", "morph": {},
               "id": 1, "dep_head": 0, "form": "x"}]
    p = classify(tokens)
    assert p.sentence_type == SentenceType.DECLARATIVE


def test_morph_none_no_crash():
    tokens = [{"lemma": "x", "pos": "VERB", "dep_rel": "root", "morph": None}]
    p = classify(tokens)
    assert p is not None


def test_default_lang_markers_classify_declarative():
    tokens = [tok(dep_rel="root")]
    p = classify(tokens, LangMarkers())
    assert p.sentence_type == SentenceType.DECLARATIVE


# ---------------------------------------------------------------------------
# TestNoHardcoding
# ---------------------------------------------------------------------------

def test_no_interrogative_lemmas_constant():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_INTERROGATIVE_LEMMAS" not in source


def test_no_estce_que_constant():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_ESTCE_QUE" not in source


def test_no_negative_lemmas_constant():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_NEGATIVE_LEMMAS" not in source
