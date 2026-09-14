from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class CausalEncoder(Protocol):
    """
    Contrat d'interface de la Couche 2 — Encodage causal.

    Le data scientist implémente ce protocol avec le framework de son choix
    (NumPy, PyTorch, JAX, sklearn, …).

    Entrée  : vecteur de features Couche 1 (UDRepresentation vectorisé)
    Sortie  : logits sur les types causaux GCN
    """

    def forward_node(self, x: np.ndarray) -> np.ndarray:
        """
        x      : shape (D_clause,)  — features d'une clause
        retour : shape (7,)         — logits sur NODE_TYPES (non normalisés)
        """
        ...

    def forward_edge(self, x: np.ndarray) -> np.ndarray:
        """
        x      : shape (D_edge,)    — features d'une paire de clauses
        retour : shape (11,)        — logits sur RELATION_TYPES (non normalisés)
        """
        ...

    def parameters(self) -> list[np.ndarray]:
        """Retourne tous les paramètres apprenables."""
        ...

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Applique les gradients. Le DS appelle cette méthode dans sa boucle."""
        ...
