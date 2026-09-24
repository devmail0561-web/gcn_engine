# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests des patches figés today.txt — silences Cat.2 puis anti-crash Cat.1."""
from __future__ import annotations

import csv
import inspect
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ── helpers dataset minimal ──────────────────────────────────────────────────

def _minimal_document(methode: str | None = None) -> dict:
    sent: dict = {
        "id": "s0001",
        "text": "Le chat dort car il est fatigué.",
        "tokens": [
            {"id": 1, "form": "Le", "lemma": "le", "pos": "DET", "dep_rel": "det", "dep_head": 2},
            {"id": 2, "form": "chat", "lemma": "chat", "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 3},
            {"id": 3, "form": "dort", "lemma": "dormir", "pos": "VERB", "dep_rel": "root", "dep_head": 0},
            {"id": 4, "form": "car", "lemma": "car", "pos": "SCONJ", "dep_rel": "mark", "dep_head": 6},
            {"id": 5, "form": "il", "lemma": "il", "pos": "PRON", "dep_rel": "nsubj", "dep_head": 6},
            {"id": 6, "form": "fatigué", "lemma": "fatigué", "pos": "ADJ", "dep_rel": "advcl", "dep_head": 3},
        ],
        "cir": {
            "nodes": [
                {"id": "n001", "type": "entite", "label": "chat", "token_span": [1, 3],
                 "scope": "specific", "temporal_index": 0, "origin": "explicit"},
                {"id": "n002", "type": "processus", "label": "fatigué", "token_span": [5, 6],
                 "scope": "specific", "temporal_index": 0, "origin": "explicit"},
            ],
            "edges": [
                {"sources": ["n001"], "target": "n002", "relation": "cause"},
            ],
        },
    }
    if methode is not None:
        sent["_methode"] = methode
    return {"document": {"lang": "fr", "sentences": [sent]}}


def _write_dataset(tmp_path: Path, methode: str | None = None) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    (d / "train.json").write_text(
        json.dumps(_minimal_document(methode), ensure_ascii=False), encoding="utf-8"
    )
    return d


def _make_pipeline(mlp_hidden: int = 128):
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    enc = MLPEncoder(
        d_clause=vocab.d_clause,
        d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7),
        mlp_hidden=mlp_hidden,
    )
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)


# ── C2.2 : val_loader reçoit silver_weight ───────────────────────────────────

def test_val_loader_receives_silver_weight(tmp_path: Path):
    """train.py doit passer silver_weight=… au GCNDataLoader du val_dir (C2.2)."""
    from click.testing import CliRunner
    from gcn_python.training.train import GCNDataLoader, train_cmd

    train_dir = _write_dataset(tmp_path, methode=None)
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val.json").write_text(
        json.dumps(_minimal_document("silver-auto-v2"), ensure_ascii=False),
        encoding="utf-8",
    )

    captured: list[dict] = []
    orig = GCNDataLoader.__init__

    def spy(self, data_dir, *a, **kw):
        captured.append({"data_dir": Path(data_dir), **kw})
        orig(self, data_dir, *a, **kw)

    out = tmp_path / "model.npz"
    with patch.object(GCNDataLoader, "__init__", spy):
        result = CliRunner().invoke(train_cmd, [
            "--data-dir", str(train_dir),
            "--val-dir", str(val_dir),
            "--epochs", "1",
            "--silver-weight", "0.5",
            "--output", str(out),
        ])
    assert result.exit_code == 0, f"gcn-train a échoué :\n{result.output}\n{result.exception}"

    val_calls = [c for c in captured if c["data_dir"] == val_dir]
    assert val_calls, "aucun GCNDataLoader instancié pour --val-dir"
    assert val_calls[0].get("silver_weight") == 0.5, (
        f"val_loader sans silver_weight (reçu {val_calls[0].get('silver_weight')!r})"
    )
    # cohérence comportementale : poids appliqué
    from gcn_python.data.loader import GCNDataLoader as RealLoader
    samples = list(RealLoader(val_dir, silver_weight=0.5))
    assert samples[0].sentence.weight == 0.5


# ── C2.4 : assembler_avg_loss séparé + headers CSV ───────────────────────────

def test_csv_writer_extrasaction_ignore_no_crash(tmp_path: Path):
    """C1.5 : DictWriter(extrasaction='ignore') ne lève pas sur clé inconnue."""
    path = tmp_path / "log.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"], extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"epoch": 1, "loss": 0.5, "decoder_loss": 9.9})  # ignorée
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["epoch"] == "1"
    assert "decoder_loss" not in rows[0]


def test_train_cmd_logs_assembler_avg_loss(tmp_path: Path):
    """C2.4 : metrics/CSV contiennent assembler_avg_loss quand l'assembleur est actif.

    source_text volontairement différent de la phrase train pour éviter gold_surface
    (teacher-forcing decodeur avec enriched d_eff=79 vs one-hot d=7 → lock d_in).
    """
    from click.testing import CliRunner
    from gcn_python.training.train import train_cmd

    data_dir = _write_dataset(tmp_path, methode="silver-auto-v2")
    verb_dir = tmp_path / "verb"
    verb_dir.mkdir()
    verbalize_doc = {
        "schema_version": "2.0",
        "examples": [{
            "id": "s0001",
            "causal_ir": {
                "source_text": "texte verbalize hors train pour gold_surface=None",
                "nodes": [
                    {"id": 0, "node_type": "entite", "label": "chat"},
                    {"id": 1, "node_type": "processus", "label": "fatigué"},
                ],
                "edges": [[0, 1, {"relation": "cause"}]],
            },
            "surfaces": [{"text": "le chat dort car il est fatigué", "quality": "gold"}],
        }],
    }
    (verb_dir / "verbalize_s0001.json").write_text(
        json.dumps(verbalize_doc, ensure_ascii=False), encoding="utf-8"
    )
    connectors = tmp_path / "connectors.json"
    connectors.write_text(json.dumps({"connectors": ["car", "parce que", "puisque"]}), encoding="utf-8")
    log_csv = tmp_path / "log.csv"
    out = tmp_path / "model.npz"

    result = CliRunner().invoke(train_cmd, [
        "--data-dir", str(data_dir),
        "--verbalize-dir", str(verb_dir),
        "--connectors-file", str(connectors),
        "--epochs", "1",
        "--log-csv", str(log_csv),
        "--output", str(out),
    ])
    assert result.exit_code == 0, f"gcn-train a échoué :\n{result.output}\n{result.exception}"

    header = log_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "assembler_avg_loss" in header, (
        f"header CSV sans assembler_avg_loss : {header}"
    )
    # l'assembleur est bien actif → la colonne n'est pas vide
    rows = list(csv.DictReader(log_csv.open(encoding="utf-8")))
    assert rows, "CSV vide"
    assert "assembler_avg_loss" in rows[0]


def test_assembler_loss_not_folded_into_epoch_loss(tmp_path: Path):
    """C2.4 : epoch_asm_loss / n_asm_samples ne pollue pas avg_loss (source + run)."""
    from gcn_python.training import train as train_mod

    src = inspect.getsource(train_mod)
    # plus d'accumulation de _a_loss dans epoch_loss / n_samples
    assert "epoch_asm_loss += _a_loss" in src, "C2.4 : accumulation epoch_asm_loss absente"
    assert "n_asm_samples += 1" in src, "C2.4 : compteur n_asm_samples absent"
    # le bloc assembler ne doit plus faire epoch_loss += _a_loss
    # (au moins un site de metrics assembler_avg_loss)
    assert 'metrics["assembler_avg_loss"]' in src


# ── C1.2 : RGCNLayerPT.load_state garde W_0 / count ──────────────────────────

def test_load_state_too_few_arrays_raises():
    pytest.importorskip("torch")
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT

    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    with pytest.raises(ValueError, match="2 arrays attendus"):
        layer.load_state([np.zeros((3, 16, 8), dtype=np.float32)])


def test_load_state_w0_wrong_shape_raises():
    pytest.importorskip("torch")
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT

    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    W_r = np.zeros((3, 16, 8), dtype=np.float32)
    W_0_bad = np.zeros((8, 16), dtype=np.float32)  # transposé
    with pytest.raises(ValueError, match="W_0 shape incompatible"):
        layer.load_state([W_r, W_0_bad])


# ── C1.1 : from_pretrained sans _arch_json → UserWarning, pas ValueError ─────

def test_from_pretrained_without_arch_json_warns_not_raises(tmp_path: Path):
    from gcn_python.engine import GCNEngine

    vocab_pipe = _make_pipeline()
    from gcn_python.training.checkpoint import save_checkpoint

    ckpt = tmp_path / "model.npz"
    save_checkpoint(vocab_pipe, ckpt)

    raw = np.load(ckpt, allow_pickle=True)
    arrays = {k: raw[k] for k in raw.files if k != "_arch_json"}
    np.savez_compressed(ckpt, **arrays)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = GCNEngine.from_pretrained(ckpt, trusted=True)

    assert engine is not None, "from_pretrained a renvoyé None"
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("_arch_json" in m for m in messages), (
        f"Aucun UserWarning '_arch_json' — silencieux. Warnings : {messages}"
    )


# ── C1.3 : mlp_hidden mismatch → warn, pas overwrite ─────────────────────────

def test_load_checkpoint_mlp_hidden_mismatch_warns_and_does_not_overwrite(tmp_path: Path):
    """C1.3 : arch.mlp_hidden ≠ pipeline.encoder.mlp_hidden → warn, pas d'overwrite.

    Les shapes encoder_* sont validées avant ; on force le mismatch en éditant
    uniquement _arch_json (shapes restent compatibles).
    """
    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint

    pipe_src = _make_pipeline(mlp_hidden=256)
    ckpt = tmp_path / "mh256.npz"
    save_checkpoint(pipe_src, ckpt)

    raw = np.load(ckpt, allow_pickle=True)
    arch = json.loads(str(raw["_arch_json"][0]))
    arch["mlp_hidden"] = 999
    arrays = {k: raw[k] for k in raw.files}
    arrays["_arch_json"] = np.array([json.dumps(arch)], dtype=object)
    np.savez_compressed(ckpt, **arrays)

    pipe_dst = _make_pipeline(mlp_hidden=256)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_checkpoint(pipe_dst, ckpt, trusted=True)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("mlp_hidden" in m for m in messages), (
        f"Aucun UserWarning mlp_hidden — silencieux. Warnings : {messages}"
    )
    assert pipe_dst.encoder.mlp_hidden == 256, (
        f"mlp_hidden écrasé silencieusement : {pipe_dst.encoder.mlp_hidden}"
    )


def test_load_checkpoint_mlp_hidden_match_assigns(tmp_path: Path):
    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint

    pipe_src = _make_pipeline(mlp_hidden=256)
    ckpt = tmp_path / "mh256.npz"
    save_checkpoint(pipe_src, ckpt)

    pipe_dst = _make_pipeline(mlp_hidden=256)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_checkpoint(pipe_dst, ckpt, trusted=True)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)
                if "mlp_hidden" in str(w.message)]
    assert not messages, f"Warn inutile quand shapes match : {messages}"
    assert pipe_dst.encoder.mlp_hidden == 256


# ── C1.4 : WordEmbedding.from_json format liste → UserWarning ────────────────

def test_from_json_old_format_warns_pretrained_boundaries():
    from gcn_python.layer1.embedding import WordEmbedding

    old_json = json.dumps(["_unk", "chat", "chien"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = WordEmbedding.from_json(old_json, d_emb=10)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("_pretrained_start" in m for m in messages), (
        f"Aucun UserWarning _pretrained_start — silencieux. Warnings : {messages}"
    )
    assert obj._pretrained_start is None
    assert obj._pretrained_end is None


def test_from_json_new_format_restores_without_warn():
    from gcn_python.layer1.embedding import WordEmbedding

    we = WordEmbedding(d_emb=8, seed=0)
    we.build_vocab(["le", "chat", "dormir"])
    we._pretrained_start = 1
    we._pretrained_end = 3
    s = we.to_json()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = WordEmbedding.from_json(s, d_emb=8)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)
                if "_pretrained_start" in str(w.message)]
    assert not messages, f"Warn faux sur format dict : {messages}"
    assert restored._pretrained_start == 1
    assert restored._pretrained_end == 3


# ── C2.6 / C2.3-doc : docstrings & commentaires de site ──────────────────────

def test_cgnp_sample_weight_docstring_mentions_both_gradients():
    from gcn_python.pipeline import cgnp

    doc = inspect.getsource(cgnp.CGNPipeline.loss)
    assert "nœuds ET arêtes" in doc, (
        "C2.6 : docstring loss() doit dire que sample_weight multiplie nœuds ET arêtes"
    )
    assert "nœuds intacts" not in doc


def test_teacher_forcing_asymmetry_documented():
    from gcn_python.training import train as train_mod

    src = inspect.getsource(train_mod)
    assert "gold_edge_map intentionnellement absent" in src, (
        "C2.3-doc : commentaire d'intention manquant sur le pass val"
    )
    assert "teacher-forcing" in src
