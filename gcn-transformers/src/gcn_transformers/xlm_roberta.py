# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Encodeur XLM-RoBERTa multilingue pour GCN Causal Engine.

XLM-RoBERTa : 280M paramètres, pré-entraîné sur 100 langues (FR/EN/Code).
"""
from __future__ import annotations

from .base import TransformerEncoderBase


class XLMRobertaEncoder(TransformerEncoderBase):
    """
    Encodeur XLM-RoBERTa multilingue (FR/EN/Code, 280M params, 100 langues).

    Pré-entraîné sur 2.5TB texte multilingue (CommonCrawl).
    Recommandé pour applications multilingues ou mixtes FR/EN.
    """

    def __init__(
        self,
        d_clause: int = 79,
        d_edge: int = 365,
        freeze_layers: int = 10,
        learning_rate: float = 1e-5,
        device: str | None = None,
        model_name: str = "xlm-roberta-base",
        n_node_types: int = 7,
        n_relation_types: int = 11,
    ):
        """
        Initialise XLM-RoBERTa encoder.

        IMPORTANT : Instancie self.model AVANT super().__init__().

        Args:
            d_clause: Dimension features UD (79 par défaut = vocab.d_clause_effective(0, False))
            d_edge: Dimension edge vectors (365 par défaut = vocab.d_edge_closed_loop(79, 7, 0, False))
            freeze_layers: Nombre de couches Transformer à geler (10/12 par défaut)
            learning_rate: Learning rate AdamW (1e-5 recommandé)
            device: "cuda", "cpu", ou None (auto-détection)
            model_name: Modèle HuggingFace ("xlm-roberta-base" par défaut, 270MB download)
            n_node_types: Nombre de types de nœuds (7 par défaut)
            n_relation_types: Nombre de relations causales (11 par défaut)

        Raises:
            ImportError: Si transformers n'est pas installé
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Le package 'transformers' est requis pour XLMRobertaEncoder.\n"
                "Installation : pip install transformers>=4.30.0"
            ) from e

        # Instanciation du modèle HuggingFace AVANT super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Init classe de base (configure heads, optimizer)
        super().__init__(
            d_clause=d_clause,
            d_edge=d_edge,
            freeze_layers=freeze_layers,
            learning_rate=learning_rate,
            device=device,
            n_node_types=n_node_types,
            n_relation_types=n_relation_types,
        )

    # forward_batch, backward_node_dx, etc. hérités de TransformerEncoderBase
