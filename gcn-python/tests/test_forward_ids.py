# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests du chemin tok in-process (pipeline/forward_ids.py).

Doctrine : parité STRUCTURELLE (schéma serde Rust), jamais parité de
valeurs avec le legacy (features différentes — assumé). Têtes non
entraînées : aucun test ne lit une classe ou une confiance comme mesure.
Déterminisme : octets identiques hors horodates wall-clock
(``extracted_at`` / ``created_at``, design legacy de ``emit()``).
"""

import json
import re as _re
import sys

import pytest

sys.path.insert(0, "gcn-python/tests")
from conftest import make_test_pipeline
from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.engine import GCNEngine
from gcn_python.pipeline.forward_ids import (
    D_MODEL_DEFAULT,
    WINDOW_K,
    TokForwardEngine,
    _edge_full_dim,
)

TEXTE_2_PHRASES = "Les ventes baissent. On reduit les couts."
TEXTE_1_PHRASE = "Les ventes baissent."


def _moteur(textes=(TEXTE_2_PHRASES,), **kwargs):
    eng = TokForwardEngine(**kwargs)
    eng.build_vocab(textes)
    return eng


def _sans_horodates(cir: dict) -> str:
    s = json.dumps(cir, sort_keys=True)
    s = _re.sub(r'"extracted_at": "[^"]*"', '"extracted_at": "T"', s)
    s = _re.sub(r'"created_at": "[^"]*"', '"created_at": "T"', s)
    return s


def _valide_structure(cir: dict, texte: str, n_clauses: int) -> None:
    assert cir["source_text"] == texte
    assert cir["source_lang"] == {"natural": {"lang": "und"}}
    assert len(cir["nodes"]) == n_clauses
    for i, noeud in enumerate(cir["nodes"]):
        assert noeud["id"] == i
        assert noeud["node_type"] in NODE_TYPES
        assert isinstance(noeud["label"], str) and noeud["label"]
        debut, fin = (
            noeud["source_span"]["token_span"]["start"],
            noeud["source_span"]["token_span"]["end"],
        )
        assert 0 <= debut <= fin
        assert noeud["scope"] == "unknown"
        assert noeud["origin"] == "explicit"
    for src, dst, arete in cir["edges"]:
        assert 0 <= src < dst < n_clauses
        assert dst - src <= WINDOW_K  # R7 : fenêtre k=2, jamais all_pairs
        assert arete["relation"] in RELATION_TYPES
        assert 0.0 <= arete["confidence"] <= 1.0
        assert arete["negated"] is False  # v0 : aucun signal de négation
        assert arete["provenance"]["extraction_method"] == "ml_python"
        assert str(arete["provenance"]["ref"]).startswith("tok:")
    assert cir["metadata"]["schema_version"] == "2.0"


def test_gate_vocab_vide():
    eng = TokForwardEngine()
    assert eng.vocab_size == 0
    with pytest.raises(ValueError):
        eng.analyze_tok(TEXTE_2_PHRASES)


def test_gate_texte_non_str():
    eng = _moteur()
    with pytest.raises(TypeError):
        eng.analyze_tok(42)


def test_gate_langue_hors_budget():
    eng = TokForwardEngine()
    with pytest.raises(ValueError):
        eng.build_vocab([TEXTE_2_PHRASES], lang="xx")


def test_texte_vide_sans_erreur():
    eng = _moteur()
    cir = eng.analyze_tok("   ")
    assert cir["nodes"] == [] and cir["edges"] == []


def test_structure_deux_phrases():
    eng = _moteur()
    cir = eng.analyze_tok(TEXTE_2_PHRASES)
    _valide_structure(cir, TEXTE_2_PHRASES, 2)
    assert len(cir["edges"]) == 1  # paire (0,1) dans la fenêtre


def test_structure_une_phrase():
    eng = _moteur(textes=(TEXTE_1_PHRASE,))
    cir = eng.analyze_tok(TEXTE_1_PHRASE)
    _valide_structure(cir, TEXTE_1_PHRASE, 1)
    assert cir["edges"] == []


def test_determinisme_hors_horodates():
    eng = _moteur()
    a = _sans_horodates(eng.analyze_tok(TEXTE_2_PHRASES))
    b = _sans_horodates(eng.analyze_tok(TEXTE_2_PHRASES))
    assert a == b


def test_determinisme_inter_instances_meme_seed():
    kwargs = {"seed": 7}
    a = _sans_horodates(_moteur(**kwargs).analyze_tok(TEXTE_2_PHRASES))
    b = _sans_horodates(_moteur(**kwargs).analyze_tok(TEXTE_2_PHRASES))
    assert a == b


def test_oov_warn_et_taux_expose():
    eng = TokForwardEngine()
    eng.build_vocab(["Bonjour le monde."])
    with pytest.warns(UserWarning):
        cir = eng.analyze_tok("Zyxwv qwerty azerty. Paragraphe deuxieme.")
    assert eng.last_oov_rate > 0.05
    # Dégradé mais valide : la segmentation B fait foi (pas de compte supposé).
    _valide_structure(cir, "Zyxwv qwerty azerty. Paragraphe deuxieme.", len(cir["nodes"]))
    assert len(cir["nodes"]) >= 1


def test_dims_coherentes():
    eng = _moteur()
    assert eng.emb.d_model == D_MODEL_DEFAULT
    assert eng.encoder is not None and eng.rgcn is not None
    assert eng.rgcn.d_in == eng.rgcn.d_out == D_MODEL_DEFAULT
    assert _edge_full_dim(D_MODEL_DEFAULT) == 4 * D_MODEL_DEFAULT + 1 + 2 * 8


def test_sans_binaire_ni_subprocess(monkeypatch):
    """Le chemin tok n'appelle jamais de subprocess (acceptation [A], partielle)."""
    import subprocess as _sp

    def _interdit(*args, **kwargs):
        raise AssertionError("subprocess appelé sur le chemin tok")

    monkeypatch.setattr(_sp, "run", _interdit)
    eng = _moteur()
    cir = eng.analyze_tok(TEXTE_2_PHRASES)
    _valide_structure(cir, TEXTE_2_PHRASES, 2)


class TestEngineUseTok:
    """Câblage GCNEngine.analyze(use_tok) — défaut legacy inchangé, gate capacitaire."""

    def test_defaut_legacy_inchange(self):
        from gcn_python.layer1.representation import UDRepresentation

        pipe = make_test_pipeline(d_emb=8)
        engine = GCNEngine(pipe)

        class _Parser:
            def parse(self, text):
                rep = UDRepresentation(
                    tokens=[{"lemma": "w", "pos": "VERB", "dep_rel": "root", "morph": {}}],
                    root_lemma="w",
                    root_pos="VERB",
                    root_dep_rel="root",
                    root_morph={},
                    subject_pos=None,
                    has_object=False,
                    has_advcl=False,
                    has_temporal_obl=False,
                    token_span=(0, 1),
                )
                return [rep], []

        engine._text_parser = _Parser()
        cir = engine.analyze("phrase vue")
        assert cir["source_text"] == "phrase vue"
        assert len(cir["nodes"]) == 1

    def test_use_tok_sans_moteur_refuse(self):
        pipe = make_test_pipeline(d_emb=8)
        engine = GCNEngine(pipe)
        with pytest.raises(ValueError):
            engine.analyze(TEXTE_2_PHRASES, use_tok=True)

    def test_use_tok_vocab_vide_refuse(self):
        pipe = make_test_pipeline(d_emb=8)
        engine = GCNEngine(pipe, tok_engine=TokForwardEngine())
        with pytest.raises(ValueError):
            engine.analyze(TEXTE_2_PHRASES, use_tok=True)

    def test_use_tok_cir_valide(self):
        pipe = make_test_pipeline(d_emb=8)
        tok = TokForwardEngine()
        tok.build_vocab([TEXTE_2_PHRASES])
        engine = GCNEngine(pipe, tok_engine=tok)
        cir = engine.analyze(TEXTE_2_PHRASES, use_tok=True)
        _valide_structure(cir, TEXTE_2_PHRASES, 2)
