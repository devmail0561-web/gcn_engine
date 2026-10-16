# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
gcn-transformers : Encodeurs Transformer pour GCN Causal Engine.

Ce package fournit des encodeurs basés sur des modèles Transformer pré-entraînés
(CamemBERT, XLM-RoBERTa, CodeBERT) comme alternatives au MLPEncoder de gcn-python.

Utilisation :
    from gcn_transformers import XLMRobertaEncoder
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause_effective(0, False)  # 106
    d_edge = vocab.d_edge_closed_loop(d_eff, 8, 0, False)  # 475

    encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=19)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

    # Entraînement via script Python (pas de CLI)
    # Voir README pour exemples complets
"""
from .camembert import CamembertEncoder
from .codebert import CodeBERTEncoder
from .xlm_roberta import XLMRobertaEncoder

__version__ = "1.1.0"
__all__ = [
    "XLMRobertaEncoder",
    "CamembertEncoder",
    "CodeBERTEncoder",
]
