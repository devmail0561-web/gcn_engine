# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests de conformité au Protocol CausalEncoder (gcn-python).
"""
from pathlib import Path

import numpy as np
import pytest

from gcn_transformers import CamembertEncoder, CodeBERTEncoder, XLMRobertaEncoder


# Helper pour vérifier si un modèle HF est en cache
def model_in_cache(model_name: str) -> bool:
    """Vérifie si un modèle HuggingFace est dans le cache local."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = "models--" + model_name.replace("/", "--")
    return (cache_dir / model_slug).exists()


# Skip markers pour modèles non cachés
skip_camembert = pytest.mark.skipif(
    not model_in_cache("camembert-base"),
    reason="CamemBERT not in HF cache"
)
skip_codebert = pytest.mark.skipif(
    not model_in_cache("microsoft/codebert-base"),
    reason="CodeBERT not in HF cache"
)


@pytest.mark.parametrize("encoder_class", [
    XLMRobertaEncoder,
    pytest.param(CamembertEncoder, marks=skip_camembert),
    pytest.param(CodeBERTEncoder, marks=skip_codebert),
])
def test_implements_causal_encoder_protocol(encoder_class, dimensions):
    """Vérifie compliance avec CausalEncoder Protocol."""
    encoder = encoder_class(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Méthodes obligatoires
    assert hasattr(encoder, 'forward_node')
    assert hasattr(encoder, 'forward_edge')
    assert hasattr(encoder, 'parameters')
    assert hasattr(encoder, 'update_node')
    assert hasattr(encoder, 'update_edge')
    assert hasattr(encoder, 'update')

    # Méthodes optionnelles critiques
    assert hasattr(encoder, 'backward_node_dx')
    assert hasattr(encoder, 'backward_edge_dx')
    assert hasattr(encoder, 'forward_batch')
    assert hasattr(encoder, 'snapshot_node_cache')
    assert hasattr(encoder, 'snapshot_edge_cache')
    assert hasattr(encoder, 'restore_node_cache')
    assert hasattr(encoder, 'restore_edge_cache')
    assert hasattr(encoder, 'training')
    assert hasattr(encoder, 'n_node_types')
    assert hasattr(encoder, 'n_relation_types')

    # Attributs requis pour --weighted-loss
    assert encoder.n_node_types == 7
    assert encoder.n_relation_types == 11


@pytest.mark.parametrize("encoder_class", [
    XLMRobertaEncoder,
    pytest.param(CamembertEncoder, marks=skip_camembert),
    pytest.param(CodeBERTEncoder, marks=skip_codebert),
])
def test_forward_node_shape(encoder_class, dimensions, vocab):
    """Vérifie shapes forward_node."""
    encoder = encoder_class(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Générer un vecteur UD
    from gcn_python.layer1.features import vectorize_clause
    from gcn_python.layer1.representation import UDRepresentation
    rep = UDRepresentation(
        tokens=[{"lemma": "test", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="test", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(0, 1),
    )
    vec = vectorize_clause(rep, vocab, word_embedding=None, subject_object_emb=False)

    # Forward
    logits = encoder.forward_node(vec)

    # Vérifier shape
    assert logits.shape == (7,), f"Expected (7,), got {logits.shape}"
    assert logits.dtype == np.float32 or logits.dtype == np.float64


@pytest.mark.parametrize("encoder_class", [
    XLMRobertaEncoder,
    pytest.param(CamembertEncoder, marks=skip_camembert),
    pytest.param(CodeBERTEncoder, marks=skip_codebert),
])
def test_forward_edge_shape(encoder_class, dimensions):
    """Vérifie shapes forward_edge."""
    encoder = encoder_class(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Vecteur edge aléatoire (d_edge=365)
    edge_vec = np.random.randn(dimensions["d_edge"]).astype(np.float32)

    # Forward
    logits = encoder.forward_edge(edge_vec)

    # Vérifier shape
    assert logits.shape == (11,), f"Expected (11,), got {logits.shape}"


@pytest.mark.parametrize("encoder_class", [
    XLMRobertaEncoder,
    pytest.param(CamembertEncoder, marks=skip_camembert),
    pytest.param(CodeBERTEncoder, marks=skip_codebert),
])
def test_parameters_returns_numpy(encoder_class, dimensions):
    """Vérifie que parameters() retourne des np.ndarray."""
    encoder = encoder_class(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    params = encoder.parameters()

    assert isinstance(params, list)
    assert len(params) > 0
    for p in params:
        assert isinstance(p, np.ndarray), f"Expected np.ndarray, got {type(p)}"


@pytest.mark.parametrize("encoder_class", [
    XLMRobertaEncoder,
    pytest.param(CamembertEncoder, marks=skip_camembert),
    pytest.param(CodeBERTEncoder, marks=skip_codebert),
])
def test_eval_train_modes(encoder_class, dimensions):
    """Vérifie basculement eval/train."""
    encoder = encoder_class(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Mode train par défaut
    assert encoder.training is True

    # Basculer en eval
    encoder.eval()
    assert encoder.training is False

    # Retour en train
    encoder.train()
    assert encoder.training is True
