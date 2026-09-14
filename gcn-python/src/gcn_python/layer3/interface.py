from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class CausalGraph(Protocol):
    """
    Contrat d'interface de la Couche 3 — Graphe causal R-GCN.

    Implémente le message passing relationnel sur le graphe causal.
    Supporte les cycles (pas de restriction DAG).

    Formule R-GCN :
      h_i^(l+1) = σ( Σ_r Σ_{j∈N_r(i)} (1/c_{i,r}) W_r^(l) h_j^(l) + W_0^(l) h_i^(l) )
    """

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_node)
        edge_index: np.ndarray,     # (2, E) — [sources, targets]
        edge_types: np.ndarray,     # (E,) int — index dans RELATION_TYPES
    ) -> np.ndarray:                # (N, D_out) — représentations enrichies
        ...

    def backward_message_pass(
        self,
        d_output: np.ndarray,       # (N, D_out) — gradient depuis la couche suivante
    ) -> tuple[np.ndarray, list[np.ndarray]]:  # (d_input (N, D_in), [dW_r, dW_0])
        ...

    def parameters(self) -> list[np.ndarray]:
        ...

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        ...
