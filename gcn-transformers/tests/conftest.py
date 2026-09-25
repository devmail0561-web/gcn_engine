# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Fixtures pytest pour tests gcn-transformers.
"""
import numpy as np
import pytest

# Skip tous les tests si transformers ou torch non disponibles
transformers = pytest.importorskip("transformers", minversion="4.30.0")
torch = pytest.importorskip("torch", minversion="2.0.0")


@pytest.fixture
def vocab():
    """FeatureVocabulary par défaut."""
    from gcn_python.layer1.features import FeatureVocabulary
    return FeatureVocabulary()


@pytest.fixture
def dimensions(vocab):
    """Dimensions d_clause et d_edge correctes."""
    d_eff = vocab.d_clause_effective(d_emb=0, subject_object_emb=False)  # 79
    d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types=7, d_emb=0,
                                      subject_object_emb=False)  # 365
    return {"d_clause": d_eff, "d_edge": d_edge}


@pytest.fixture
def sample_ud_reps():
    """Sample UDRepresentations pour tests."""
    from gcn_python.layer1.representation import UDRepresentation

    reps = [
        UDRepresentation(
            tokens=[{"lemma": "pleuvoir", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="pleuvoir", root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None,
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, 1),
        ),
        UDRepresentation(
            tokens=[{"lemma": "inondation", "pos": "NOUN", "dep_rel": "root", "morph": {}}],
            root_lemma="inondation", root_pos="NOUN", root_dep_rel="root",
            root_morph={}, subject_pos=None,
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(1, 2),
        ),
    ]
    return reps


@pytest.fixture
def sample_gold_labels():
    """Gold labels pour tests backward."""
    gold_node_labels = np.array([0, 1], dtype=np.int64)  # action, condition
    gold_edge_map = {(0, 1): 0}  # cause
    return {"nodes": gold_node_labels, "edges": gold_edge_map}
