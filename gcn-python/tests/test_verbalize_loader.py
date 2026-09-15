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
