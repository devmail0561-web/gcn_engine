from __future__ import annotations
import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


class _LinearLayer:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim)  # He init
        self.W = rng.normal(0, scale, (out_dim, in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W.T + self.b
        self._cache["x"] = x
        return out

    def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._cache["x"]
        dW = np.outer(d_out, x)
        db = d_out.copy()
        dx = d_out @ self.W
        return dx, dW, db


class MLPEncoder:
    """
    Implémentation de référence de la Couche 2 en NumPy pur.
    Architecture : fc(D_in→128)+ReLU → fc(128→64)+ReLU → fc(64→out)

    Le data scientist substitue par son propre CausalEncoder
    (PyTorch, JAX, etc.) sans modifier le pipeline.
    """

    def __init__(self, d_clause: int, d_edge: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Node MLP : d_clause → 128 → 64 → 7
        self._node_layers = [
            _LinearLayer(d_clause, 128, rng),
            _LinearLayer(128, 64, rng),
            _LinearLayer(64, 7, rng),
        ]

        # Edge MLP : d_edge → 256 → 128 → 11
        self._edge_layers = [
            _LinearLayer(d_edge, 256, rng),
            _LinearLayer(256, 128, rng),
            _LinearLayer(128, 11, rng),
        ]

        self._node_cache: list = []
        self._edge_cache: list = []

    def _forward_mlp(
        self, x: np.ndarray, layers: list[_LinearLayer], cache_out: list
    ) -> np.ndarray:
        cache_out.clear()
        h = x
        for i, layer in enumerate(layers):
            z = layer.forward(h)
            if i < len(layers) - 1:
                h = _relu(z)
                cache_out.append((z, h))
            else:
                h = z
                cache_out.append((z, z))
        return h

    def forward_node(self, x: np.ndarray) -> np.ndarray:
        self._node_cache = []
        return self._forward_mlp(x, self._node_layers, self._node_cache)

    def forward_edge(self, x: np.ndarray) -> np.ndarray:
        self._edge_cache = []
        return self._forward_mlp(x, self._edge_layers, self._edge_cache)

    def _backward_mlp(
        self, d_logits: np.ndarray, layers: list[_LinearLayer], cache: list
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Retourne (grads, d_input) où d_input est le gradient vers l'entrée."""
        grads = []
        d = d_logits
        for i in reversed(range(len(layers))):
            z, _ = cache[i]
            if i < len(layers) - 1:
                d = d * _relu_grad(z)
            dx, dW, db = layers[i].backward(d)
            grads.insert(0, (dW, db))
            d = dx
        return grads, d

    def backward_node(self, d_logits: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        grads, _ = self._backward_mlp(d_logits, self._node_layers, self._node_cache)
        return grads

    def backward_node_dx(self, d_logits: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Comme backward_node mais retourne aussi le gradient vers l'entrée (pour R-GCN)."""
        return self._backward_mlp(d_logits, self._node_layers, self._node_cache)

    def backward_edge(self, d_logits: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        grads, _ = self._backward_mlp(d_logits, self._edge_layers, self._edge_cache)
        return grads

    def snapshot_node_cache(self) -> list:
        """Snapshot du cache node + inputs des couches (pour backward par nœud)."""
        return [
            ((z.copy(), h.copy()), l._cache.get("x", np.zeros(0)).copy())
            for (z, h), l in zip(self._node_cache, self._node_layers)
        ]

    def restore_node_cache(self, snapshot: list) -> None:
        self._node_cache = [(z.copy(), h.copy()) for (z, h), _ in snapshot]
        for ((_, _), x), layer in zip(snapshot, self._node_layers):
            layer._cache["x"] = x.copy()

    def snapshot_edge_cache(self) -> list:
        return [
            ((z.copy(), h.copy()), l._cache.get("x", np.zeros(0)).copy())
            for (z, h), l in zip(self._edge_cache, self._edge_layers)
        ]

    def restore_edge_cache(self, snapshot: list) -> None:
        self._edge_cache = [(z.copy(), h.copy()) for (z, h), _ in snapshot]
        for ((_, _), x), layer in zip(snapshot, self._edge_layers):
            layer._cache["x"] = x.copy()

    def parameters(self) -> list[np.ndarray]:
        params = []
        for layer in self._node_layers + self._edge_layers:
            params.extend([layer.W, layer.b])
        return params

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        for p, g in zip(self.parameters(), grads):
            p -= lr * g
