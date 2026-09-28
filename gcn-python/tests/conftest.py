# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers pour la création de pipelines de test (D10 ETUDE : word_embedding obligatoire)
# ---------------------------------------------------------------------------

def make_word_embedding(d_emb: int = 4, seed: int = 42):
    """Crée un WordEmbedding minimal pour les tests (d_emb petit = rapide)."""
    from gcn_python.layer1.embedding import WordEmbedding
    return WordEmbedding(d_emb=d_emb, seed=seed)


def make_test_pipeline(d_emb: int = 4, **kwargs):
    """Crée un CGNPipeline minimal valide pour les tests.

    Calcule d_clause_effective(d_emb) et passe word_embedding obligatoire.
    Remplace tous les patterns encoder+graph+CGNPipeline dans les tests.
    """
    from gcn_python.constants import NODE_TYPES
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    we = make_word_embedding(d_emb=d_emb)
    d_eff = vocab.d_clause_effective(we.d_emb)
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), we.d_emb)
    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                       word_embedding=we, **kwargs)


@pytest.fixture
def minimal_word_embedding():
    """Fixture — WordEmbedding minimal (d_emb=4)."""
    return make_word_embedding(d_emb=4)


@pytest.fixture
def minimal_pipeline():
    """Fixture — CGNPipeline minimal valide."""
    return make_test_pipeline()


@pytest.fixture
def taxonomy_dir() -> Path:
    candidates = [
        Path(__file__).parents[2] / "gcn-references" / "taxonomies",
        Path(__file__).parents[2] / "gcn-core" / "data" / "taxonomies",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    pytest.skip(f"Taxonomy dir not found (tried: {candidates})")


@pytest.fixture
def paper_examples_json() -> Path:  # L5 : renommé (yaml→json)
    p = Path(__file__).parents[2] / "gcn-core" / "tests" / "fixtures" / "paper_examples.json"
    if not p.exists():
        pytest.skip(f"paper_examples.json not found: {p}")
    return p


@pytest.fixture
def datasets_dir() -> Path:
    d = Path(__file__).parents[2] / "gcn-datasets" / "examples"
    if not d.is_dir():
        pytest.skip(f"datasets_dir not found: {d}")
    return d
