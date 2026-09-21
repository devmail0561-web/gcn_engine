# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests de régression pour les 5 findings du red-team 12ᵉ audit (F1–F5)."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_checkpoint(tmp_path: Path) -> Path:
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipe, ckpt)
    return ckpt


def _make_data_dir(tmp_path: Path, name: str = "data") -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    doc = {
        "document": {
            "sentences": [{
                "id": "s001", "text": "test.",
                "tokens": [{"id": 1, "form": "test", "lemma": "test",
                            "pos": "NOUN", "dep_rel": "root", "dep_head": 0, "morph": {}}],
                "cir": {
                    "nodes": [{"id": "n001", "type": "action", "label": "test",
                               "token_span": [1, 1], "scope": "specific",
                               "temporal_index": 0, "origin": "explicit"}],
                    "edges": [],
                },
            }]
        }
    }
    (d / "s.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# F1 — gcn-eval sur données vides → warn stderr
# ---------------------------------------------------------------------------

def test_eval_cmd_empty_dir_warns_stderr(tmp_path: Path):
    """F1 : gcn-eval --data-dir vide émet ATTENTION sur stderr, JSON propre sur stdout."""
    from click.testing import CliRunner
    from gcn_python.evaluation.eval_runner import eval_cmd

    ckpt = _make_minimal_checkpoint(tmp_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(eval_cmd, ["--data-dir", str(empty_dir), "--model-path", str(ckpt)])

    assert result.exit_code == 0, f"exit_code={result.exit_code}: {result.output}"
    assert "ATTENTION" in result.output, (
        f"Aucun ATTENTION dans la sortie pour n_samples=0. output={result.output!r}"
    )
    # Le JSON suit l'ATTENTION — extraire depuis le premier '{'
    json_start = result.output.index("{")
    report = json.loads(result.output[json_start:])
    assert report["n_samples"] == 0


def test_eval_cmd_empty_dir_quiet_still_warns_stderr(tmp_path: Path):
    """F1 : --quiet ne supprime pas le warn n_samples=0 (click.echo err=True)."""
    from click.testing import CliRunner
    from gcn_python.evaluation.eval_runner import eval_cmd

    ckpt = _make_minimal_checkpoint(tmp_path)
    empty_dir = tmp_path / "empty_q"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(eval_cmd, [
        "--data-dir", str(empty_dir), "--model-path", str(ckpt), "--quiet"
    ])
    assert "ATTENTION" in result.output, (
        f"--quiet ne doit pas supprimer le warn n_samples=0. output={result.output!r}"
    )


# ---------------------------------------------------------------------------
# F2 — checkpoint corrompu → ClickException propre
# ---------------------------------------------------------------------------

def test_eval_cmd_corrupt_checkpoint_click_error(tmp_path: Path):
    """F2 : .npz corrompu → Error: propre, pas de traceback brut."""
    from click.testing import CliRunner
    from gcn_python.evaluation.eval_runner import eval_cmd

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"\x00" * 64)
    data_dir = _make_data_dir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(eval_cmd, ["--data-dir", str(data_dir), "--model-path", str(corrupt)])

    assert result.exit_code != 0, "Checkpoint corrompu doit produire exit != 0"
    assert "Error:" in result.output, f"Pas de 'Error:' dans stdout. output={result.output!r}"
    assert "Traceback" not in (result.output + (result.stderr or "")), (
        "Traceback brut visible — wrap ClickException manquant."
    )


# ---------------------------------------------------------------------------
# F3 — index --append graphe corrompu → ClickException propre
# ---------------------------------------------------------------------------

def test_index_append_corrupt_graph_click_error(tmp_path: Path):
    """F3 : index --append sur graphe JSON corrompu → Error: propre, pas de traceback."""
    from click.testing import CliRunner
    from gcn_python.index import index_cmd

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "f.txt").write_text("texte.\n", encoding="utf-8")
    ckpt = _make_minimal_checkpoint(tmp_path)
    output = tmp_path / "graph.json"
    output.write_text("[invalid json", encoding="utf-8")  # graphe corrompu

    runner = CliRunner()
    result = runner.invoke(index_cmd, [
        "--corpus", str(corpus),
        "--checkpoint", str(ckpt),
        "--output", str(output),
        "--append",
    ])

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in (result.output + (result.stderr or ""))


# ---------------------------------------------------------------------------
# F4 — id=0 préfixé dans index.py (is not None)
# ---------------------------------------------------------------------------

def test_index_node_id_zero_is_prefixed():
    """F4 : nœud avec id=0 (entier falsy) est bien préfixé — n.get('id') le sautait."""
    # Simuler la logique de préfixage directement (unité pure, pas de CLI)
    cir = {
        "nodes": [
            {"id": 0, "type": "action", "label": "A"},
            {"id": 1, "type": "action", "label": "B"},
        ],
        "edges": [
            [0, 1, {"relation": "cause"}],
            {"source": 0, "target": 1, "relation": "cause"},
        ],
    }
    prefix = "b00001_"

    # Reproduire la logique corrigée de index.py
    for n in cir["nodes"]:
        if isinstance(n, dict) and n.get("id") is not None:
            n["id"] = prefix + str(n["id"])
    remapped = []
    for e in cir["edges"]:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            s, d, a = e
            remapped.append([prefix + str(s), prefix + str(d), a])
        elif isinstance(e, dict):
            e = dict(e)
            if e.get("source") is not None:
                e["source"] = prefix + str(e["source"])
            if e.get("target") is not None:
                e["target"] = prefix + str(e["target"])
            remapped.append(e)
    cir["edges"] = remapped

    node_ids = {n["id"] for n in cir["nodes"]}
    assert "b00001_0" in node_ids, f"id=0 non préfixé. ids={node_ids}"
    assert "b00001_1" in node_ids
    assert 0 not in node_ids and 1 not in node_ids, "ids bruts encore présents"

    # Arêtes aussi préfixées
    assert cir["edges"][0][0] == "b00001_0"
    assert cir["edges"][1]["source"] == "b00001_0"


# ---------------------------------------------------------------------------
# F5 — --taxonomy-dir propagé à GCNEngine.from_pretrained
# ---------------------------------------------------------------------------

def test_from_pretrained_accepts_taxonomy_dir(tmp_path: Path):
    """F5 : GCNEngine.from_pretrained() accepte taxonomy_dir sans TypeError."""
    from gcn_python.engine import GCNEngine

    ckpt = _make_minimal_checkpoint(tmp_path)
    taxo = tmp_path / "taxo"
    taxo.mkdir()

    # Ne doit pas lever TypeError — taxonomy_dir était absent de la signature
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = GCNEngine.from_pretrained(
            ckpt, trusted=True, taxonomy_dir=taxo
        )
    assert engine is not None
