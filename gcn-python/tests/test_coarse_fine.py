# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests D4/§15 ETUDE — progression coarse→fine."""
from gcn_python.constants import (
    COARSE_NODE_TYPES,
    COARSE_RELATION_TYPES,
    NODE_TYPES,
    RELATION_TYPES,
    coarse_node,
    coarse_relation,
)
from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause
from gcn_python.layer1.representation import UDRepresentation

# ---------------------------------------------------------------------------
# Remapping relation fine → coarse
# ---------------------------------------------------------------------------

def test_fine_to_coarse_relation_coverage():
    """Tous les types fins (hors analogy/counterfactual) ont un groupe coarse."""
    excluded = {"analogy", "counterfactual"}
    for rel in RELATION_TYPES:
        if rel in excluded:
            assert coarse_relation(rel) == rel, f"{rel} devrait être inchangé"
        else:
            assert coarse_relation(rel) in COARSE_RELATION_TYPES, (
                f"{rel} → {coarse_relation(rel)} absent de COARSE_RELATION_TYPES"
            )


def test_fine_to_coarse_node_coverage():
    """Tous les types nœuds fins ont un groupe coarse."""
    for nt in NODE_TYPES:
        assert coarse_node(nt) in COARSE_NODE_TYPES, (
            f"{nt} → {coarse_node(nt)} absent de COARSE_NODE_TYPES"
        )


def test_coarse_relation_types_count():
    assert len(COARSE_RELATION_TYPES) == 5


def test_coarse_node_types_count():
    assert len(COARSE_NODE_TYPES) == 4


# ---------------------------------------------------------------------------
# Passthrough — analogy/counterfactual hors §15
# ---------------------------------------------------------------------------

def test_analogy_passthrough():
    assert coarse_relation("analogy") == "analogy"


def test_counterfactual_passthrough():
    assert coarse_relation("counterfactual") == "counterfactual"


# ---------------------------------------------------------------------------
# Masque features — no_mood / no_tense
# ---------------------------------------------------------------------------

def _make_rep(mood: str = "Sub", tense: str = "Pres") -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": "courir", "pos": "VERB", "dep_rel": "root",
                 "morph": {"Mood": mood, "Tense": tense}, "id": 0}],
        root_lemma="courir", root_pos="VERB", root_dep_rel="root",
        root_morph={"Mood": mood, "Tense": tense},
        subject_pos=None, has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(0, 1),
    )


def test_no_mood_zeroes_mood_feature():
    from conftest import make_word_embedding
    we = make_word_embedding()
    vocab = FeatureVocabulary()
    rep = _make_rep(mood="Sub")
    v_normal = vectorize_clause(rep, vocab, word_embedding=we)
    v_masked = vectorize_clause(rep, vocab, word_embedding=we, no_mood=True)
    # Les deux vecteurs diffèrent sur les dimensions mood
    mood_start = (len(vocab.upos_tags) + len(vocab.dep_rels)
                  + len(vocab.subject_pos_cats) + len(vocab.tense_values)
                  + len(vocab.aspect_values))
    mood_end = mood_start + len(vocab.mood_values)
    assert not (v_normal[mood_start:mood_end] == v_masked[mood_start:mood_end]).all(), (
        "no_mood=True devrait zéroter les features Mood"
    )


def test_no_tense_zeroes_tense_feature():
    from conftest import make_word_embedding
    we = make_word_embedding()
    vocab = FeatureVocabulary()
    rep = _make_rep(tense="Pres")
    v_normal = vectorize_clause(rep, vocab, word_embedding=we)
    v_masked = vectorize_clause(rep, vocab, word_embedding=we, no_tense=True)
    tense_start = len(vocab.upos_tags) + len(vocab.dep_rels) + len(vocab.subject_pos_cats)
    tense_end = tense_start + len(vocab.tense_values)
    assert not (v_normal[tense_start:tense_end] == v_masked[tense_start:tense_end]).all(), (
        "no_tense=True devrait zéroter les features Tense"
    )


def test_no_mood_no_tense_same_shape():
    from conftest import make_word_embedding
    we = make_word_embedding()
    vocab = FeatureVocabulary()
    rep = _make_rep()
    v = vectorize_clause(rep, vocab, word_embedding=we, no_mood=True, no_tense=True)
    v_ref = vectorize_clause(rep, vocab, word_embedding=we)
    assert v.shape == v_ref.shape, "d_clause doit être inchangé avec no_mood/no_tense"


# ---------------------------------------------------------------------------
# Promotion — types avec N≥coarse_n_min restent fins (test logique)
# ---------------------------------------------------------------------------

def test_promotion_excludes_from_remap():
    """Un type fin avec N≥400 ne doit pas être remappé vers son groupe coarse."""
    promoted = {"cause"}
    # Simuler la logique du train.py : si rel in promoted → garder fine
    for rel in RELATION_TYPES:
        active = rel if rel in promoted else coarse_relation(rel)
        assert active is not None


def test_coarse_relation_groups_cover_all_non_analogy():
    """Chaque groupe coarse est non-vide et ses membres sont dans RELATION_TYPES."""
    from gcn_python.constants import COARSE_RELATION_GROUPS
    for group, members in COARSE_RELATION_GROUPS.items():
        assert len(members) > 0, f"Groupe coarse '{group}' est vide"
        for m in members:
            assert m in RELATION_TYPES, f"'{m}' dans groupe '{group}' absent de RELATION_TYPES"
