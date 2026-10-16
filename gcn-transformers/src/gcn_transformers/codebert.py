# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Encodeur CodeBERT pour GCN Causal Engine.

CodeBERT : 125M paramètres, pré-entraîné sur code Python/Java/JavaScript/PHP/Ruby/Go.
"""
from __future__ import annotations

from .base import TransformerEncoderBase, load_hf_model


class CodeBERTEncoder(TransformerEncoderBase):
    """
    Encodeur CodeBERT (code Python/Rust, 125M params).

    Pré-entraîné sur 6.4M fonctions GitHub (6 langages).
    Recommandé pour analyse causale dans le code (docstrings, commentaires techniques).
    """

    def __init__(
        self,
        d_clause: int = 106,
        d_edge: int = 475,
        freeze_layers: int = 10,
        learning_rate: float = 1e-5,
        device: str | None = None,
        model_name: str = "microsoft/codebert-base",
        revision: str | None = None,
        n_node_types: int = 8,
        n_relation_types: int = 19,
    ):
        """
        Initialise CodeBERT encoder.

        Args:
            d_clause: Dimension features UD (106 v3.0)
            d_edge: Dimension edge vectors (475 v3.0)
            freeze_layers: Nombre de couches Transformer à geler (10/12 par défaut)
            learning_rate: Learning rate AdamW (1e-5 recommandé)
            device: "cuda", "cpu", ou None (auto-détection)
            model_name: Modèle HuggingFace ("microsoft/codebert-base" par défaut, 500MB download)
            n_node_types: Nombre de types de nœuds (8 v3.0)
            n_relation_types: Nombre de relations causales (19 v3.0)
            revision: Révision HF pinnée (sha) — None = ref flottante (repro
                non garantie). trust_remote_code=False toujours.

        Raises:
            ImportError: Si transformers n'est pas installé
        """
        # Instanciation du modèle HuggingFace AVANT super().__init__()
        # (garde supply-chain : revision pinnée, trust_remote_code=False,
        # erreur hors-ligne claire — cf. load_hf_model).
        self.model, self.tokenizer = load_hf_model(model_name, revision)

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
