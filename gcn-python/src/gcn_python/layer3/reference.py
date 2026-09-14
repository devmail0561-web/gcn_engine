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

    def __init__(self, d_in: int, d_out: int, n_relations: int | None = None, seed: int = 42):
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_in)
        self.W_r = rng.normal(0, scale, (self.n_relations, d_out, d_in)).astype(np.float32)
        self.W_0 = rng.normal(0, scale, (d_out, d_in)).astype(np.float32)

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_in)
        edge_index: np.ndarray,     # (2, E)
        edge_types: np.ndarray,     # (E,) int
    ) -> np.ndarray:                # (N, D_out)
        N = node_features.shape[0]
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
                for e_idx, d in enumerate(dst_r):
                    out[d] += msgs[e_idx] / counts[d]

        return _sigmoid(out)

    def parameters(self) -> list[np.ndarray]:
        return [self.W_r, self.W_0]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        self.W_r -= lr * grads[0]
        self.W_0 -= lr * grads[1]
