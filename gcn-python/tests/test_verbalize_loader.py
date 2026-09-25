# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcn_python.data.verbalize_loader import VerbalizerDataLoader, VerbalizeSample
from gcn_python.verbalizer.trainable import SurfaceVocabulary

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "gcn-datasets" / "examples"


@pytest.fixture
def examples_dir() -> Path:
    if not EXAMPLES_DIR.exists():
        pytest.skip("gcn-datasets/examples/ not found")
    return EXAMPLES_DIR


def test_load_5_examples(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    # verbalize_cross_modal.json has 5 examples × 2 surfaces = 10 samples
    assert len(loader) == 10


def test_samples_are_verbalize_sample(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    for s in loader:
        assert isinstance(s, VerbalizeSample)
        assert isinstance(s.ir_json, str)
        assert s.node_type_embeddings.ndim == 2
        assert s.node_type_embeddings.shape[1] == 7  # len(NODE_TYPES)
        assert s.gold_tokens.ndim == 1
        assert len(s.gold_tokens) > 0


def test_vocab_built(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    assert len(loader.vocab) > 2  # more than PAD + UNK


def test_gold_tokens_in_vocab_range(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    V = len(loader.vocab)
    for s in loader:
        assert all(0 <= int(t) < V for t in s.gold_tokens)


def test_source_text_map(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    m = loader.source_text_map()
    assert len(m) > 0
    for text, token_lists in m.items():
        assert isinstance(text, str)
        assert len(token_lists) > 0
        for tl in token_lists:
            assert isinstance(tl, np.ndarray)


def test_external_vocab_respected(examples_dir: Path):
    # Provide pre-built vocab — should be used as-is
    external_vocab = SurfaceVocabulary()
    external_vocab.build(["si les ventes baissent"])
    loader = VerbalizerDataLoader(examples_dir, vocab=external_vocab)
    assert loader.vocab is external_vocab


def test_node_type_embeddings_onehot(examples_dir: Path):
    loader = VerbalizerDataLoader(examples_dir)
    for s in loader:
        embs = s.node_type_embeddings
        assert embs.shape[1] == 7
        # Each row is a valid one-hot (sum = 1)
        row_sums = embs.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


def test_no_yaml_loaded(examples_dir: Path, monkeypatch):
    """Verify that no YAML file is opened during loading."""
    import builtins
    original_open = builtins.open
    yaml_opens = []

    def patched_open(file, *args, **kwargs):
        if str(file).endswith(".yaml") or str(file).endswith(".yml"):
            yaml_opens.append(str(file))
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", patched_open)
    VerbalizerDataLoader(examples_dir)
    assert yaml_opens == [], f"YAML files read by loader: {yaml_opens}"


def test_node_labels_aligned(examples_dir):
    """node_labels doit avoir le même nombre d'éléments que les lignes de node_type_embeddings."""
    loader = VerbalizerDataLoader(examples_dir)
    for sample in loader:
        assert isinstance(sample.node_labels, list), "node_labels doit être une list"
        assert all(isinstance(s, str) for s in sample.node_labels), \
            "tous les éléments de node_labels doivent être des str"
        assert len(sample.node_labels) == sample.node_type_embeddings.shape[0], \
            (f"len(node_labels)={len(sample.node_labels)} != "
             f"N={sample.node_type_embeddings.shape[0]}")


# ── Améliorations G1/G2 — clause_texts + connector_gold_idx ───────────────────

def _write_source_dataset(d: Path) -> None:
    doc = {"document": {"lang": "fr", "sentences": [{
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
            "edges": [{"sources": ["n001"], "target": "n002", "relation": "cause"}],
        },
    }]}}
    (d / "train.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _write_verbalize_pair(d: Path) -> None:
    doc = {"schema_version": "2.0", "examples": [{
        "id": "s0001",
        "causal_ir": {
            "source_text": "Le chat dort car il est fatigué.",
            "nodes": [
                {"id": 0, "node_type": "entite", "label": "chat"},
                {"id": 1, "node_type": "processus", "label": "fatigué"},
            ],
            "edges": [[0, 1, {"relation": "cause"}]],
        },
        "surfaces": [{"text": "le chat dort car il est fatigué", "quality": "gold"}],
    }]}
    (d / "verbalize_test.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def test_verbalize_loader_extracts_clause_texts(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write_source_dataset(src)
    vdir = tmp_path / "verb"
    vdir.mkdir()
    _write_verbalize_pair(vdir)
    loader = VerbalizerDataLoader(vdir, source_json_dir=src)
    assert len(loader) == 1
    s = next(iter(loader))
    assert s.clause_texts is not None
    assert s.clause_texts[0] == "Le chat dort", s.clause_texts
    assert s.clause_texts[1] == "il fatigué", s.clause_texts


def test_verbalize_loader_extracts_connector_gold_idx(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write_source_dataset(src)
    vdir = tmp_path / "verb"
    vdir.mkdir()
    _write_verbalize_pair(vdir)
    loader = VerbalizerDataLoader(vdir, source_json_dir=src,
                                  connector_vocab=["car", "parce que"])
    s = next(iter(loader))
    assert s.edge_triples == [(0, 1, 0)], s.edge_triples  # cause = index 0
    assert s.connector_gold_idx == [0], s.connector_gold_idx  # "car" matché


def test_verbalize_loader_no_source_dir_gives_none(tmp_path: Path):
    vdir = tmp_path / "verb"
    vdir.mkdir()
    _write_verbalize_pair(vdir)
    loader = VerbalizerDataLoader(vdir)
    s = next(iter(loader))
    assert s.clause_texts is None
    assert s.connector_gold_idx is None
    assert s.edge_triples == [(0, 1, 0)]
