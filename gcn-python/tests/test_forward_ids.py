# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests du chemin tok in-process (pipeline/forward_ids.py + nlu/tok.py).

Doctrine : parité STRUCTURELLE (schéma serde Rust), jamais parité de
valeurs avec le legacy (features différentes — assumé). Têtes non
entraînées : aucun test ne lit une classe ou une confiance comme mesure.
Déterminisme : octets identiques hors horodates wall-clock
(``extracted_at`` / ``created_at``, design legacy de ``emit()``).
Séparation : le runtime (``pipeline/``) n'importe jamais B — la
segmentation vit dans ``nlu/tok.py`` (garde
``test_runtime_appris_n_importe_jamais_moteur_b``).
"""

import json
import re as _re
import sys

import pytest

sys.path.insert(0, "gcn-python/tests")

from conftest import make_test_pipeline
from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.engine import GCNEngine
from gcn_python.layer0.tokenizer import split as tok_split
from gcn_python.nlu.tok import analyze_tok_text, parse_text
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


def _segments(texte, lang="fr"):
    seg = parse_text(texte, lang=lang)
    return seg


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
        debut = noeud["source_span"]["token_span"]["start"]
        fin = noeud["source_span"]["token_span"]["end"]
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


def _forward(eng, texte, lang="fr"):
    seg = _segments(texte, lang=lang)
    return eng.forward_from_spans(
        texte,
        seg["ids"],
        seg["offsets"],
        seg["clause_spans"],
        seg["clause_textes"],
        connecteurs=seg["connecteurs"],
        source_b=seg["source"],
    )


def test_gate_vocab_vide():
    eng = TokForwardEngine()
    assert eng.vocab_size == 0
    seg = _segments(TEXTE_2_PHRASES)
    with pytest.raises(ValueError):
        eng.forward_from_spans(
            TEXTE_2_PHRASES,
            seg["ids"],
            seg["offsets"],
            seg["clause_spans"],
            seg["clause_textes"],
        )


def test_gate_texte_non_str():
    eng = _moteur()
    with pytest.raises(TypeError):
        eng.forward_from_spans(42, [], [], [], [])


def test_gate_langue_hors_budget():
    eng = TokForwardEngine()
    with pytest.raises(ValueError):
        eng.build_vocab([TEXTE_2_PHRASES], lang="xx")
    with pytest.raises(ValueError):
        parse_text(TEXTE_2_PHRASES, lang="xx")


def test_clauses_vides_sans_erreur():
    eng = _moteur()
    cir = eng.forward_from_spans("   ", [], [], [], [], source_b="vide")
    assert cir["nodes"] == [] and cir["edges"] == []


def test_structure_deux_phrases():
    eng = _moteur()
    cir = _forward(eng, TEXTE_2_PHRASES)
    _valide_structure(cir, TEXTE_2_PHRASES, 2)
    assert len(cir["edges"]) == 1  # paire (0,1) dans la fenêtre


def test_structure_une_phrase():
    eng = _moteur(textes=(TEXTE_1_PHRASE,))
    cir = _forward(eng, TEXTE_1_PHRASE)
    _valide_structure(cir, TEXTE_1_PHRASE, 1)
    assert cir["edges"] == []


def test_determinisme_hors_horodates():
    eng = _moteur()
    a = _sans_horodates(_forward(eng, TEXTE_2_PHRASES))
    b = _sans_horodates(_forward(eng, TEXTE_2_PHRASES))
    assert a == b


def test_determinisme_inter_instances_meme_seed():
    kwargs = {"seed": 7}
    a = _sans_horodates(_forward(_moteur(**kwargs), TEXTE_2_PHRASES))
    b = _sans_horodates(_forward(_moteur(**kwargs), TEXTE_2_PHRASES))
    assert a == b


def test_oov_warn_et_taux_expose():
    eng = TokForwardEngine()
    eng.build_vocab(["Bonjour le monde."])
    texte = "Zyxwv qwerty azerty. Paragraphe deuxieme."
    seg = _segments(texte)
    with pytest.warns(UserWarning):
        cir = eng.forward_from_spans(
            texte,
            seg["ids"],
            seg["offsets"],
            seg["clause_spans"],
            seg["clause_textes"],
            source_b=seg["source"],
        )
    assert eng.last_oov_rate > 0.05
    # Dégradé mais valide : la segmentation B fait foi (pas de compte supposé).
    _valide_structure(cir, texte, len(cir["nodes"]))
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
    cir = _forward(eng, TEXTE_2_PHRASES)
    _valide_structure(cir, TEXTE_2_PHRASES, 2)


def test_frontend_parse_text():
    seg = parse_text(TEXTE_2_PHRASES)
    assert seg["source"] in ("silver-B", "fallback")
    assert len(seg["clause_spans"]) == len(seg["clause_textes"]) == 2
    ids, offsets, _tokens = tok_split(TEXTE_2_PHRASES)
    assert seg["ids"] == ids and seg["offsets"] == offsets
    assert parse_text("   ")["clause_spans"] == []
    with pytest.raises(TypeError):
        parse_text(42)


def test_frontend_analyze_tok_text():
    eng = _moteur()
    cir = analyze_tok_text(eng, TEXTE_2_PHRASES)
    _valide_structure(cir, TEXTE_2_PHRASES, 2)
    assert analyze_tok_text(eng, "   ")["nodes"] == []


class TestCacheTok:
    """Cache M1a du chemin tok : hit = zéro calcul, invalidation au bump."""

    def test_repetition_sans_recalcul(self, monkeypatch):
        eng = _moteur()
        appels = {"n": 0}
        origine = eng._forward_parts

        def _compte(*args, **kwargs):
            appels["n"] += 1
            return origine(*args, **kwargs)

        monkeypatch.setattr(eng, "_forward_parts", _compte)
        import gcn_python.nlu.tok as _tok

        origine_parse = _tok.parse_text
        segles = {"n": 0}

        def _compte_parse(*args, **kwargs):
            segles["n"] += 1
            return origine_parse(*args, **kwargs)

        monkeypatch.setattr(_tok, "parse_text", _compte_parse)
        c1 = analyze_tok_text(eng, TEXTE_2_PHRASES)
        c2 = analyze_tok_text(eng, TEXTE_2_PHRASES)
        assert appels["n"] == 1  # forward sauté la 2e fois
        assert segles["n"] == 1  # socle sauté aussi (cache au-dessus de B)
        assert c1 == c2
        assert eng.cache_stats() == {"hits": 1, "misses": 1, "size": 1, "max": 1024}

    def test_forward_cached_segment_level(self):
        eng = _moteur()
        seg = parse_text(TEXTE_2_PHRASES)
        kwargs = {
            "text": TEXTE_2_PHRASES,
            "ids": seg["ids"],
            "offsets": seg["offsets"],
            "clause_spans": seg["clause_spans"],
            "clause_textes": seg["clause_textes"],
        }
        c1 = eng.forward_cached(**kwargs)
        c1["edges"] = []
        c2 = eng.forward_cached(**kwargs)
        assert c2["edges"] != []
        assert eng.cache_stats()["hits"] == 1

    def test_copies_independantes(self):
        eng = _moteur()
        c1 = analyze_tok_text(eng, TEXTE_2_PHRASES)
        c1["edges"] = []
        c2 = analyze_tok_text(eng, TEXTE_2_PHRASES)
        assert c2["edges"] != []

    def test_mutation_poids_invalide(self):
        eng = _moteur()
        analyze_tok_text(eng, TEXTE_2_PHRASES)
        assert eng.cache_stats()["misses"] == 1
        eng.emb.update(0.01)  # bump epoch → invalidation (sens sûr)
        analyze_tok_text(eng, TEXTE_2_PHRASES)
        assert eng.cache_stats()["misses"] == 2

    def test_fifo_et_clear(self):
        eng = _moteur(textes=(TEXTE_2_PHRASES, TEXTE_1_PHRASE))
        import gcn_python.pipeline.forward_ids as _mod

        ancien = _mod.TOK_CACHE_MAX
        _mod.TOK_CACHE_MAX = 1
        try:
            analyze_tok_text(eng, TEXTE_2_PHRASES)
            analyze_tok_text(eng, TEXTE_1_PHRASE)
            assert eng.cache_stats()["size"] == 1
            eng.clear_cache()
            assert eng.cache_stats() == {"hits": 0, "misses": 0, "size": 0, "max": 1}
        finally:
            _mod.TOK_CACHE_MAX = ancien


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

    def test_moteur_vide_refuse(self):
        with pytest.raises(TypeError):
            GCNEngine(None, tok_engine=None)

    def test_tok_only_legacy_refuse_avec_conseil(self):
        from gcn_python.pipeline.forward_ids import TokForwardEngine as _T

        tok = _T()
        tok.build_vocab([TEXTE_2_PHRASES])
        engine = GCNEngine(None, tok_engine=tok)
        with pytest.raises(ValueError):
            engine.analyze(TEXTE_2_PHRASES)
        with pytest.raises(ValueError):
            engine.verbalize(TEXTE_2_PHRASES)
        # analyze_or_skip : contrat None-safe (warn + None, jamais raise).
        with pytest.warns(UserWarning):
            assert engine.analyze_or_skip(TEXTE_2_PHRASES) is None
        out = engine.analyze_batch([TEXTE_2_PHRASES, TEXTE_2_PHRASES, "  "], use_tok=True)
        assert len(out) == 2
        _valide_structure(out[0], TEXTE_2_PHRASES, 2)

    def test_tok_only_surface_coherente(self):
        from gcn_python.constants import NODE_TYPES, RELATION_TYPES
        from gcn_python.pipeline.forward_ids import TokForwardEngine as _T

        tok = _T()
        tok.build_vocab([TEXTE_2_PHRASES])
        engine = GCNEngine(None, tok_engine=tok)
        assert engine.node_types == NODE_TYPES
        assert engine.relation_types == RELATION_TYPES
        assert "tok-only" in repr(engine)
        # stream + or_skip via tok.
        flux = list(engine.stream([TEXTE_2_PHRASES, "  "], use_tok=True))
        assert len(flux) == 1
        _valide_structure(flux[0], TEXTE_2_PHRASES, 2)
        cir = engine.analyze_or_skip(TEXTE_2_PHRASES, use_tok=True)
        _valide_structure(cir, TEXTE_2_PHRASES, 2)

    def test_from_pretrained_tok(self, tmp_path):
        tok = TokForwardEngine(seed=9)
        tok.build_vocab([TEXTE_2_PHRASES])
        chemin = str(tmp_path / "tok.npz")
        tok.save_npz(chemin)
        engine = GCNEngine.from_pretrained(chemin, trusted=True)
        assert engine._pipeline is None and engine._tok_engine is not None
        _valide_structure(engine.analyze(TEXTE_2_PHRASES, use_tok=True), TEXTE_2_PHRASES, 2)

    def test_from_pretrained_tok_exige_trusted(self, tmp_path):
        tok = TokForwardEngine()
        tok.build_vocab([TEXTE_2_PHRASES])
        chemin = str(tmp_path / "tok.npz")
        tok.save_npz(chemin)
        with pytest.raises(RuntimeError):
            GCNEngine.from_pretrained(chemin)
        with pytest.raises(FileNotFoundError):
            GCNEngine.from_pretrained(str(tmp_path / "absent.npz"), trusted=True)


class TestCheckpointS1:
    """Persistance s1+ : roundtrip exact, refus fail-fast (jamais partiel)."""

    def test_roundtrip_identique(self, tmp_path):
        from gcn_python.nlu.tok import parse_text

        eng = TokForwardEngine(seed=5)
        eng.build_vocab([TEXTE_2_PHRASES])
        seg = parse_text(TEXTE_2_PHRASES)
        kwargs = {
            "text": TEXTE_2_PHRASES,
            "ids": seg["ids"],
            "offsets": seg["offsets"],
            "clause_spans": seg["clause_spans"],
            "clause_textes": seg["clause_textes"],
        }
        # Mutation des poids (sinon le roundtrip ne prouve rien).
        parts = eng.forward_supervise(**kwargs)
        from gcn_python.training.train_tok import backward_tok

        backward_tok(eng, parts, [0, 1], {(0, 1): 2}, lr=0.01)
        ref = _sans_horodates(eng.forward_from_spans(**kwargs))
        chemin = str(tmp_path / "tok.npz")
        assert eng.save_npz(chemin) == chemin
        charge = TokForwardEngine.load_npz(chemin)
        assert charge.d_model == eng.d_model and charge.seed == eng.seed
        assert charge.vocab_size == eng.vocab_size
        assert _sans_horodates(charge.forward_from_spans(**kwargs)) == ref

    def test_refus_garbage_et_legacy(self, tmp_path):
        import numpy as np

        faux = tmp_path / "faux.npz"
        np.savez(str(faux), autre=np.asarray([1]))
        with pytest.raises(ValueError):
            TokForwardEngine.load_npz(str(faux))
        legacy = tmp_path / "legacy.npz"
        np.savez(str(legacy), E=np.zeros((1, 1), dtype=np.float32))
        with pytest.raises(ValueError):
            TokForwardEngine.load_npz(str(legacy))

    def test_refus_d_model_incoherent(self, tmp_path):
        import numpy as np

        from gcn_python.layer1.tok_embedding import FORMAT_REFACTO

        eng = TokForwardEngine(d_model=8, seed=1)
        eng.build_vocab([TEXTE_2_PHRASES])
        chemin = str(tmp_path / "tok.npz")
        eng.save_npz(chemin)
        with np.load(chemin, allow_pickle=False) as z:
            tables = {k: z[k] for k in z.files}
        tables["d_model"] = np.asarray(999, dtype=np.int64)
        corrompu = str(tmp_path / "corrompu.npz")
        np.savez(corrompu, **tables)
        with pytest.raises(ValueError):
            TokForwardEngine.load_npz(corrompu)
        assert FORMAT_REFACTO == 20261007
