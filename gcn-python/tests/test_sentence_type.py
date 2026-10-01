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


# ---------------------------------------------------------------------------
# P4a — tête type de phrase apprise (miroir Éq.6 intent)
# ---------------------------------------------------------------------------

import numpy as np

from gcn_python.constants import SENTENCE_TYPES


def test_sentence_types_values():
    assert SENTENCE_TYPES == ["declarative", "interrogative", "imperative", "exclamative"]


def test_derive_sentence_type_rules():
    """Bootstrap par ponctuation : ? → interrogative, ! → exclamative."""
    from gcn_python.data.loader import derive_sentence_type
    assert derive_sentence_type("Ça va ?") == "interrogative"
    assert derive_sentence_type("Quel bruit !") == "exclamative"
    assert derive_sentence_type("Les ventes baissent.") == "declarative"
    # L'impératif exige l'annotation — la règle ne devine pas.
    assert derive_sentence_type("Réduis les coûts.") == "declarative"


def test_loader_annotated_beats_rule():
    """SentenceRecord.sentence_type annoté prime sur la règle."""
    import tempfile
    from pathlib import Path

    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import ClauseRecord, SentenceRecord, TokenRecord

    def tk(i, lemma, pos, dep, head):
        return TokenRecord(id=i, form=lemma, lemma=lemma, pos=pos,
                           dep_rel=dep, dep_head=head, morph={})
    rec = SentenceRecord(
        id="s", text="Ça va ?",
        tokens=[tk(1, "aller", "VERB", "root", 0)],
        clauses=[ClauseRecord(node_id="n1", node_type="processus", label="x",
                              token_span=(1, 1), scope="specific",
                              temporal_index=0, origin="explicit")],
        edges=[], sentence_type="imperative")
    with tempfile.TemporaryDirectory() as d:
        sample = GCNDataLoader(Path(d))._to_sample(rec)
    assert sample.sentence_type == "imperative"
    assert sample.sentence_type_source == "annotated"
    rec2 = SentenceRecord(
        id="s2", text="Ça va ?",
        tokens=[tk(1, "aller", "VERB", "root", 0)],
        clauses=[ClauseRecord(node_id="n1", node_type="processus", label="x",
                              token_span=(1, 1), scope="specific",
                              temporal_index=0, origin="explicit")],
        edges=[])
    with tempfile.TemporaryDirectory() as d:
        sample2 = GCNDataLoader(Path(d))._to_sample(rec2)
    assert sample2.sentence_type == "interrogative"
    assert sample2.sentence_type_source == "rule"


def test_sentence_marks_pooled():
    """"?" / "!" atteignent le pooling (sinon l'interrogative est invisible)."""
    from gcn_python.layer1.features import _pool_lemmas
    from gcn_python.layer1.representation import UDRepresentation

    def rep(form):
        return UDRepresentation(
            tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}},
                    {"lemma": form, "pos": "PUNCT", "dep_rel": "punct", "morph": {},
                     "form": form}],
            root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 2))
    assert "?" in _pool_lemmas(rep("?"), "mean")
    assert "!" in _pool_lemmas(rep("!"), "mean")
    assert "?" not in _pool_lemmas(rep("."), "mean")


def test_sentence_head_shapes_and_update():
    """Tête sentence : forward/backward/update cohérents (miroir intent)."""
    from conftest import make_word_embedding
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder

    vocab = FeatureVocabulary()
    we = make_word_embedding()
    d_eff = vocab.d_clause_effective(we.d_emb)
    enc = MLPEncoder(d_clause=d_eff,
                     d_edge=vocab.d_edge_closed_loop(d_eff, 8, we.d_emb),
                     n_sentence_types=len(SENTENCE_TYPES), seed=7)
    x = np.random.RandomState(7).randn(d_eff).astype(np.float32)
    logits = enc.forward_sentence(x)
    assert logits.shape == (len(SENTENCE_TYPES),)
    before = [p.copy() for p in [_l.W for _l in enc._sentence_layers]]
    grads, dx = enc.backward_sentence(np.ones_like(logits))
    assert dx.shape == (d_eff,)
    enc.update_sentence(grads, lr=0.01)
    assert any(not np.allclose(a, b) for a, b in
               zip(before, [_l.W for _l in enc._sentence_layers], strict=False))


def _mkexp_sentences():
    """Paires minimales : mêmes clauses, seul le final change (. vs ?)."""

    def sent(i, final, typ):
        toks = [
            {"id": 1, "form": "ventes", "lemma": "vente", "pos": "NOUN",
             "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
            {"id": 2, "form": "baissent", "lemma": "baisser", "pos": "VERB",
             "dep_rel": "root", "dep_head": 0, "morph": {}},
            {"id": 3, "form": final, "lemma": final, "pos": "PUNCT",
             "dep_rel": "punct", "dep_head": 0, "morph": {}},
        ]
        return {"id": f"s{i}", "text": f"ventes baissent{final}", "tokens": toks,
                "cir": {"nodes": [
                    {"id": "n1", "type": "processus", "label": "baisse ventes",
                     "token_span": [1, 2]}],
                    "edges": []}}
    return {"document": {"sentences": [
        sent(1, ".", "declarative"), sent(2, ".", "declarative"),
        sent(3, "?", "interrogative"), sent(4, "?", "interrogative")]}}


def test_sentence_learned_end_to_end_real():
    """ÉTAT RÉEL : vrai train puis prédictions — le "?" seul fait la différence."""
    import json
    import tempfile
    from pathlib import Path

    from click.testing import CliRunner

    from gcn_python.training.train import train_cmd

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.json").write_text(json.dumps(_mkexp_sentences()), encoding="utf-8")
        out = tmp_path / "model.npz"
        result = CliRunner().invoke(train_cmd, [
            "--data-dir", str(data_dir), "--epochs", "60", "--embedding-dim", "8",
            "--seed", "7", "--lr", "0.005", "--output", str(out),
            "--n-sentence-types", "4",
            # Données jouet : opt-out explicite du gate v5.
            "--min-class-count", "1"])
        assert result.exit_code == 0, f"train réel échoué :\n{result.output}\n{result.exception}"

        from gcn_python import GCNEngine
        eng = GCNEngine.from_pretrained(out, trusted=True)
        ok = 0
        from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
        for smp in GCNDataLoader(data_dir):
            reps, _, conn = reps_from_sentence(smp.sentence)
            eng._pipeline.forward(reps, smp.sentence.text, connector_reps=conn)
            el = np.asarray(eng._pipeline._cached_sentence_logits)
            pred = SENTENCE_TYPES[int(np.argmax(el[0]))]
            gold = smp.sentence_type
            assert pred == gold, f"{smp.sentence.id} : {pred} != {gold}"
            ok += 1
        assert ok == 4


def test_intent_conditioned_requires_sentence_head():
    """P4b : couplage sans tête sentence = ValueError (jamais silencieux)."""
    import pytest

    from gcn_python.layer2.reference import MLPEncoder
    with pytest.raises(ValueError, match="n_sentence_types"):
        MLPEncoder(d_clause=110, d_edge=50, n_intent_types=3,
                   n_sentence_types=0, intent_conditioned=True)


def test_intent_conditioned_forward_shape():
    """P4b : entrée intent = [vec | probs] quand couplée."""

    from conftest import make_word_embedding
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    we = make_word_embedding()
    d_eff = vocab.d_clause_effective(we.d_emb)
    enc = MLPEncoder(d_clause=d_eff,
                     d_edge=vocab.d_edge_closed_loop(d_eff, 8, we.d_emb),
                     n_intent_types=3, n_sentence_types=4,
                     intent_conditioned=True, seed=7)
    pipe = CGNPipeline(encoder=enc, graph=__import__(
        "gcn_python.layer3.reference", fromlist=["RGCNLayer"]).RGCNLayer(
            d_in=d_eff, d_out=d_eff,
            n_relations=__import__("gcn_python.constants", fromlist=["rgcn_n_relations"]).rgcn_n_relations(19, False)),
        vocabulary=vocab, word_embedding=we,
        n_intent_types=3, n_sentence_types=4)
    from gcn_python.layer1.representation import UDRepresentation
    rep = UDRepresentation(
        tokens=[{"lemma": "x", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="x", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos=None, has_object=False,
        has_advcl=False, has_temporal_obl=False, token_span=(1, 1))
    pipe.forward([rep], "x.")
    assert pipe._cached_sentence_logits is not None
    assert pipe._cached_intent_logits is not None
    assert pipe._cached_intent_logits.shape[1] == 3
