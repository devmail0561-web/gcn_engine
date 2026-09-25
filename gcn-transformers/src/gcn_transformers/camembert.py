# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Encodeur CamemBERT français pour GCN Causal Engine.

CamemBERT : 110M paramètres, pré-entraîné sur corpus français (OSCAR).
"""
from __future__ import annotations

from .base import TransformerEncoderBase


class CamembertEncoder(TransformerEncoderBase):
    """
    Encodeur CamemBERT français (110M params).

    Pré-entraîné sur 138GB texte français (OSCAR).
    Recommandé pour applications français uniquement (meilleure performance que XLM-R sur FR).
    """

    def __init__(
        self,
        d_clause: int = 79,
        d_edge: int = 365,
        freeze_layers: int = 10,
        learning_rate: float = 1e-5,
        device: str | None = None,
        model_name: str = "camembert-base",
        n_node_types: int = 7,
        n_relation_types: int = 11,
    ):
        """
        Initialise CamemBERT encoder.

        Args:
            d_clause: Dimension features UD (79 par défaut)
            d_edge: Dimension edge vectors (365 par défaut)
            freeze_layers: Nombre de couches Transformer à geler (10/12 par défaut)
            learning_rate: Learning rate AdamW (1e-5 recommandé)
            device: "cuda", "cpu", ou None (auto-détection)
            model_name: Modèle HuggingFace ("camembert-base" par défaut, 440MB download)
            n_node_types: Nombre de types de nœuds (7 par défaut)
            n_relation_types: Nombre de relations causales (11 par défaut)

        Raises:
            ImportError: Si transformers n'est pas installé
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Le package 'transformers' est requis pour CamembertEncoder.\n"
                "Installation : pip install transformers>=4.30.0"
            ) from e

        # Instanciation du modèle HuggingFace AVANT super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Init classe de base
        super().__init__(
            d_clause=d_clause,
            d_edge=d_edge,
            freeze_layers=freeze_layers,
            learning_rate=learning_rate,
            device=device,
            n_node_types=n_node_types,
            n_relation_types=n_relation_types,
        )
