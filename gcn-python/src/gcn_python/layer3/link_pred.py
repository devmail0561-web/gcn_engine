# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""LinkPredHead — tête de prédiction de liens (Étape différée du plan v5, livrée).

Le moteur prédit les relations des arêtes connues ; cette tête répond à la
question complémentaire : « manque-t-il une flèche entre A et B ? ».

- Protocol LinkPredictor SÉPARÉ de CausalGraph (ne casse pas
  layer3/interface.py:9-41, vérifié via hasattr dans CGNPipeline).
- Score bilinéaire + sigmoïde : s(u,v) = sigmoid(u^T W v + b).
- Hyperparamètres : src_aggregation (mean|max), bfs_depth (candidats),
  neg_ratio (échantillonnage négatif).
- Gate : gcn-eval --gate <seuil> --on-fail=warn (jamais fail-closed par défaut).
"""
from __future__ import annotations

import json
import warnings
from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LinkPredictor(Protocol):
    """Contrat de la tête de prédiction de liens (séparé de CausalGraph)."""

    def score(self, src_vec: np.ndarray, dst_vec: np.ndarray) -> float:
        ...

    def parameters(self) -> list[np.ndarray]:
        ...

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LinkPredHead:
    """Tête bilinéaire NumPy : s(u,v) = sigmoid(u^T W v + b).

    src_aggregation : 'mean' | 'max' — agrégation multi-sources pour les
      paires N-aires (frozenset de sources -> un vecteur).
    """

    def __init__(self, d_in: int, seed: int = 0,
                 src_aggregation: str = "mean") -> None:
        if src_aggregation not in ("mean", "max"):
            raise ValueError(f"src_aggregation inconnu : {src_aggregation!r} (mean|max).")
        self.d_in = int(d_in)
        self.src_aggregation = src_aggregation
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / self.d_in)
        self.W = (rng.normal(0, scale, (self.d_in, self.d_in))).astype(np.float32)
        self.b = np.zeros((), dtype=np.float32)
        self._cache: dict | None = None

    # ------------------------------------------------------------------
    def aggregate_sources(self, vecs: np.ndarray) -> np.ndarray:
        """(K, D) -> (D,) selon src_aggregation."""
        if len(vecs) == 1:
            return np.asarray(vecs[0], dtype=np.float32)
        if self.src_aggregation == "max":
            return np.asarray(vecs, dtype=np.float32).max(axis=0).astype(np.float32)
        return np.asarray(vecs, dtype=np.float32).mean(axis=0).astype(np.float32)

    def score(self, src_vec: np.ndarray, dst_vec: np.ndarray) -> float:
        u = np.asarray(src_vec, dtype=np.float32).reshape(-1)
        v = np.asarray(dst_vec, dtype=np.float32).reshape(-1)
        if u.shape[0] != self.d_in or v.shape[0] != self.d_in:
            raise ValueError(
                f"LinkPredHead.score : dims {u.shape}/{v.shape} ≠ d_in={self.d_in}."
            )
        logit = float(u @ self.W @ v + self.b)
        self._cache = {"u": u, "v": v, "logit": logit}
        return float(_sigmoid(np.array(logit)))

    def loss_and_grad(self, src_vec: np.ndarray, dst_vec: np.ndarray,
                      label: int) -> tuple[float, list[np.ndarray]]:
        """BCE + gradients (dW, db). label ∈ {0, 1}."""
        p = self.score(src_vec, dst_vec)
        p = min(1.0 - 1e-9, max(1e-9, p))
        y = float(label)
        loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        d_logit = p - y
        u, v = self._cache["u"], self._cache["v"]
        dW = (d_logit * np.outer(u, v)).astype(np.float32)
        db = np.array(d_logit, dtype=np.float32).reshape(())
        return float(loss), [dW, db]

    # ------------------------------------------------------------------
    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        if len(grads) != 2:
            raise ValueError(f"update() : {len(grads)} gradients pour 2 paramètres.")
        self.W -= lr * grads[0]
        self.b -= lr * grads[1]

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"d_in": self.d_in, "src_aggregation": self.src_aggregation})

    @classmethod
    def from_json(cls, s: str) -> "LinkPredHead":
        data = json.loads(s)
        return cls(d_in=int(data["d_in"]),
                   src_aggregation=data.get("src_aggregation", "mean"))


def sample_negatives(edge_pairs: list[tuple[int, int]], n_nodes: int,
                     neg_ratio: float = 1.0, seed: int = 0) -> list[tuple[int, int]]:
    """Échantillonne des paires non-arêtes (négatifs) : n = len(positifs) * neg_ratio.

    Exclut les boucles (i, i) et les paires positives. Warn si le graphe est
    trop dense pour fournir assez de négatifs (retourne ce qui existe).
    """
    rng = np.random.default_rng(seed)
    positive = set(edge_pairs)
    n_want = int(len(edge_pairs) * neg_ratio)
    negatives: list[tuple[int, int]] = []
    candidates = [(i, j) for i in range(n_nodes) for j in range(n_nodes)
                  if i != j and (i, j) not in positive]
    if not candidates:
        warnings.warn("sample_negatives : aucun négatif disponible (graphe complet).",
                      UserWarning, stacklevel=2)
        return []
    if len(candidates) < n_want:
        warnings.warn(
            f"sample_negatives : {len(candidates)} négatifs dispo < {n_want} demandés.",
            UserWarning, stacklevel=2,
        )
        return candidates
    idxs = rng.choice(len(candidates), size=n_want, replace=False)
    return [candidates[int(i)] for i in idxs]


def candidates_within_depth(adjacency: dict, src: int, depth: int) -> set[int]:
    """Nœuds atteignables depuis src en ≤ depth sauts (BFS). depth ≥ 1."""
    if depth < 1:
        raise ValueError(f"bfs_depth doit être ≥ 1 (reçu {depth!r}).")
    seen = {src}
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        for nb in adjacency.get(node, []):
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, d + 1))
    seen.discard(src)
    return seen
