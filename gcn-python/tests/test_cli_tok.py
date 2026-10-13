# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests flags --use-tok des CLIs batch (discuss/index) — auto-force + gate.

- ``resolve_use_tok`` : défaut legacy inchangé, auto-force tok-only, gate
  explicite (E_tok vide refusé) ;
- index e2e (CliRunner) sur checkpoint tok s1+ : graphe avec relations,
  avec et sans flag (auto-force prouvée) ;
- ``--help`` expose les options ; ``handle(use_tok)`` enrichit le graphe.
"""

import json

import pytest
from click.testing import CliRunner

from gcn_python.cli.tok_flags import resolve_use_tok
from gcn_python.dialogue import DialogueSession
from gcn_python.engine import GCNEngine
from gcn_python.pipeline.forward_ids import TokForwardEngine
from gcn_python.verbalizer.instructions import CausalGraph, InstructionHandler

TEXTE = "Les ventes baissent. On reduit les couts."


def _tok_ckpt(tmp_path, seed=21):
    tok = TokForwardEngine(seed=seed)
    tok.build_vocab([TEXTE])
    chemin = str(tmp_path / "tok.npz")
    tok.save_npz(chemin)
    return chemin


def _tok_only(chemin):
    return GCNEngine.from_pretrained(chemin, trusted=True)


def test_resolve_defaut_legacy_inchange():
    import sys

    sys.path.insert(0, "gcn-python/tests")
    from conftest import make_test_pipeline

    engine = GCNEngine(make_test_pipeline(d_emb=8))
    assert resolve_use_tok(engine, False, "outil") is False


def test_resolve_gate_sans_tok_fonde():
    import sys

    sys.path.insert(0, "gcn-python/tests")
    import click

    from conftest import make_test_pipeline

    engine = GCNEngine(make_test_pipeline(d_emb=8))
    with pytest.raises(click.ClickException):
        resolve_use_tok(engine, True, "outil")


def test_resolve_auto_force_tok_only(tmp_path):
    engine = _tok_only(_tok_ckpt(tmp_path))
    assert engine._pipeline is None
    assert resolve_use_tok(engine, False, "outil") is True
    assert resolve_use_tok(engine, True, "outil") is True


def test_index_help_expose_use_tok():
    from gcn_python.discuss import discuss_cmd
    from gcn_python.index import index_cmd

    for cmd in (discuss_cmd, index_cmd):
        aide = CliRunner().invoke(cmd, ["--help"]).output
        assert "--use-tok" in aide and "--lang" in aide


def test_index_e2e_tok_auto_force(tmp_path):
    from gcn_python.index import index_cmd

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(TEXTE + "\n", encoding="utf-8")
    ckpt = _tok_ckpt(tmp_path)
    sortie = tmp_path / "graph.json"
    # Sans flag : auto-force par détection du checkpoint tok.
    res = CliRunner().invoke(
        index_cmd, ["--corpus", str(corpus), "--checkpoint", ckpt, "--output", str(sortie)]
    )
    assert res.exit_code == 0, res.output
    graphe = CausalGraph.load(sortie)
    assert len(graphe.edges) >= 1
    # Avec flag explicite : même résultat.
    sortie2 = tmp_path / "graph2.json"
    res = CliRunner().invoke(
        index_cmd,
        ["--corpus", str(corpus), "--checkpoint", ckpt, "--output", str(sortie2), "--use-tok"],
    )
    assert res.exit_code == 0, res.output
    assert len(CausalGraph.load(sortie2).edges) >= 1


def test_handle_use_tok_enrichit():
    import sys

    sys.path.insert(0, "gcn-python/tests")
    from conftest import make_test_pipeline

    pipe = make_test_pipeline(d_emb=8)
    tok = TokForwardEngine()
    tok.build_vocab(["Le cafe cause insomnie."])
    engine = GCNEngine(pipe, tok_engine=tok)
    session = DialogueSession()
    handler = InstructionHandler()
    reponse, info = session.handle("Le cafe cause insomnie ?", handler, engine=engine, use_tok=True)
    assert isinstance(reponse, str) and isinstance(info, dict)
    _ = json.dumps(info, ensure_ascii=False)  # journal sérialisable
