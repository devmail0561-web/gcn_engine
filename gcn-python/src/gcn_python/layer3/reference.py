from __future__ import annotations
import numpy as np
from ..constants import RELATION_TYPES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class RGCNLayer:
    """
    Implémentation NumPy de référence d'une couche R-GCN.

    h_i^(out) = σ( Σ_r (1/c_{i,r}) Σ_{j∈N_r(i)} W_r h_j + W_0 h_i )

    Supporte les cycles — aucune restriction DAG.
    Le DS empile plusieurs couches ou substitue par PyTorch Geometric.
    """

    def __init__(self, d_in: int, d_out: int, n_relations: int | None = None,
                 seed: int = 42, dropout: float = 0.0):
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)
        self.dropout = dropout
        self.training = True
        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_in)
        self.W_r = rng.normal(0, scale, (self.n_relations, d_out, d_in)).astype(np.float32)
        self.W_0 = rng.normal(0, scale, (d_out, d_in)).astype(np.float32)
        self._fwd_inputs: tuple | None = None
        self._fwd_output: np.ndarray | None = None

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_in)
        edge_index: np.ndarray,     # (2, E)
        edge_types: np.ndarray,     # (E,) int
    ) -> np.ndarray:                # (N, D_out)
        N = node_features.shape[0]

        # Dropout sur les features d'entrée
        if self.dropout > 0.0 and self.training:
            mask = (self._rng.random(node_features.shape) > self.dropout).astype(np.float32)
            node_features = node_features * mask / (1.0 - self.dropout)

        out = node_features @ self.W_0.T  # self-loop

        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            for r in range(self.n_relations):
                mask = edge_types == r
                if not np.any(mask):
                    continue
                src_r, dst_r = src[mask], dst[mask]
                counts = np.maximum(np.bincount(dst_r, minlength=N).astype(np.float32), 1.0)
                msgs = node_features[src_r] @ self.W_r[r].T  # (|E_r|, D_out)
                np.add.at(out, dst_r, msgs / counts[dst_r, np.newaxis])

        h = _sigmoid(out)
        # Copier les tableaux : évite que des mutations externes corrompent le backward
        self._fwd_inputs = (node_features.copy(), edge_index.copy(), edge_types.copy())
        self._fwd_output = h
        return h

    def backward_message_pass(
        self,
        d_output: np.ndarray,       # (N, D_out)
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Rétropropagation à travers le message passing R-GCN.

        Retourne (d_input, [dW_r, dW_0]) où d_input est le gradient vers
        les features d'entrée (ignoré — pas de paramètres apprenables en amont).
        """
        assert self._fwd_inputs is not None, "backward_message_pass appelé avant message_pass"
        node_features, edge_index, edge_types = self._fwd_inputs
        h = self._fwd_output
        N = node_features.shape[0]

        # Gradient à travers sigmoid : d_pre_act = d_output * h * (1 - h)
        d_pre_act = d_output * h * (1.0 - h)  # (N, D_out)

        # Self-loop : out_self = node_features @ W_0.T
        dW_0 = d_pre_act.T @ node_features           # (D_out, D_in)
        d_input = d_pre_act @ self.W_0               # (N, D_in)

        dW_r = np.zeros_like(self.W_r)               # (n_relations, D_out, D_in)

        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            for r in range(self.n_relations):
                mask = edge_types == r
                if not np.any(mask):
                    continue
                src_r, dst_r = src[mask], dst[mask]
                counts = np.maximum(np.bincount(dst_r, minlength=N).astype(np.float32), 1.0)

                # Gradient des messages normalisés vers W_r et noeuds sources
                d_msgs = d_pre_act[dst_r] / counts[dst_r, np.newaxis]  # (|E_r|, D_out)
                dW_r[r] = d_msgs.T @ node_features[src_r]               # (D_out, D_in)
                np.add.at(d_input, src_r, d_msgs @ self.W_r[r])         # (|E_r|, D_in)

        return d_input, [dW_r, dW_0]

    def parameters(self) -> list[np.ndarray]:
        return [self.W_r, self.W_0]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        self.W_r -= lr * grads[0]
        self.W_0 -= lr * grads[1]
