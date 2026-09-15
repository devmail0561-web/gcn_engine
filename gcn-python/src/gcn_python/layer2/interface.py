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
        """Retourne tous les paramètres apprenables (utilisé par le checkpoint)."""
        ...

    def update_node(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        """
        Applique les gradients du MLP nœud.
        grads : liste de (dW, db) par couche, dans l'ordre des couches du node MLP.
        """
        ...

    def update_edge(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        """
        Applique les gradients du MLP arête.
        grads : liste de (dW, db) par couche, dans l'ordre des couches du edge MLP.
        """
        ...

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Applique une liste plate de gradients (conservé pour compatibilité externe)."""
        ...
