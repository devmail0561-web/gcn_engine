# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from ..constants import RELATION_TYPES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LayerNormNumPy:
    """NumPy LayerNorm : y = gamma * (x - mean) / sqrt(var + eps) + beta."""

    def __init__(self, d: int, eps: float = 1e-5):
        self.d = d
        self.eps = eps
        self.gamma = np.ones(d, dtype=np.float32)
        self.beta = np.zeros(d, dtype=np.float32)
        self._cache: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x - mean) * std_inv
        self._cache = (x_norm, std_inv)
        return self.gamma * x_norm + self.beta

    def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_norm, std_inv = self._cache
        d = x_norm.shape[-1]
        d_gamma = (d_out * x_norm).sum(axis=0)
        d_beta = d_out.sum(axis=0)
        dx_norm = d_out * self.gamma
        dx = (1.0 / d) * std_inv * (
            d * dx_norm
            - dx_norm.sum(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        )
        return dx, d_gamma, d_beta

    def parameters(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def update(self, d_gamma: np.ndarray, d_beta: np.ndarray, lr: float) -> None:
        self.gamma -= lr * d_gamma
        self.beta -= lr * d_beta


class RGCNLayer:
    """
    Implémentation NumPy de référence d'une couche R-GCN.

    h_i^(out) = σ( Σ_r (1/c_{i,r}) Σ_{j∈N_r(i)} W_r h_j + W_0 h_i )

    Supporte les cycles — aucune restriction DAG.
    Le DS empile plusieurs couches ou substitue par PyTorch Geometric.
    """

    def __init__(self, d_in: int, d_out: int, n_relations: int | None = None,
                 seed: int = 42, dropout: float = 0.0,
                 output_activation: str = "sigmoid",
                 use_layernorm: bool = False):
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)
        self.dropout = dropout
        self.output_activation = output_activation
        self.use_layernorm = use_layernorm
        self.norm: LayerNormNumPy | None = LayerNormNumPy(d_out) if use_layernorm else None
        self.training = True
        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_in)
        self.W_r = rng.normal(0, scale, (self.n_relations, d_out, d_in)).astype(np.float32)
        self.W_0 = rng.normal(0, scale, (d_out, d_in)).astype(np.float32)
        self._fwd_inputs: tuple | None = None
        self._fwd_output: np.ndarray | None = None
        self._fwd_pre_norm: np.ndarray | None = None

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_in)
        edge_index: np.ndarray,     # (2, E)
        edge_types: np.ndarray,     # (E,) int
    ) -> np.ndarray:                # (N, D_out)
        if node_features.ndim != 2 or node_features.shape[1] != self.d_in:
            raise ValueError(
                f"RGCNLayer.message_pass : node_features.shape={node_features.shape} "
                f"incompatible avec d_in={self.d_in}."
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"RGCNLayer.message_pass : edge_index.shape={edge_index.shape} "
                "(attendu (2, E))."
            )
        if edge_index.shape[1] > 0 and (
            edge_types.min() < 0 or edge_types.max() >= self.n_relations
        ):
            raise ValueError(
                f"RGCNLayer.message_pass : edge_types hors bornes [0, {self.n_relations}[ "
                f"(min={edge_types.min()}, max={edge_types.max()})."
            )
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

        if self.output_activation == "relu":
            h = np.maximum(0.0, out)
        elif self.output_activation == "none":
            h = out
        else:
            h = _sigmoid(out)
        self._fwd_pre_norm = h
        if self.norm is not None:
            h = self.norm.forward(h)
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
        if self._fwd_inputs is None:
            raise RuntimeError("backward_message_pass appelé avant message_pass")
        node_features, edge_index, edge_types = self._fwd_inputs
        N = node_features.shape[0]

        d_gamma, d_beta = None, None
        if self.norm is not None:
            d_act, d_gamma, d_beta = self.norm.backward(d_output)
        else:
            d_act = d_output

        h = self._fwd_pre_norm
        if self.output_activation == "relu":
            d_pre_act = d_act * (h > 0).astype(np.float32)
        elif self.output_activation == "none":
            d_pre_act = d_act
        else:
            d_pre_act = d_act * h * (1.0 - h)

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

                d_msgs = d_pre_act[dst_r] / counts[dst_r, np.newaxis]  # (|E_r|, D_out)
                dW_r[r] = d_msgs.T @ node_features[src_r]               # (D_out, D_in)
                np.add.at(d_input, src_r, d_msgs @ self.W_r[r])         # (|E_r|, D_in)

        grads = [dW_r, dW_0]
        if self.norm is not None:
            grads.extend([d_gamma, d_beta])
        return d_input, grads

    def parameters(self) -> list[np.ndarray]:
        params = [self.W_r, self.W_0]
        if self.norm is not None:
            params.extend([self.norm.gamma, self.norm.beta])
        return params

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        self.W_r -= lr * grads[0]
        self.W_0 -= lr * grads[1]
        if self.norm is not None and len(grads) >= 4:
            self.norm.update(grads[2], grads[3], lr)
