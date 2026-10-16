# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Encodeur contextuel CPU (O1) : CNN dilaté 2 couches + résidus (NumPy).

Donne à chaque token son contexte (connecteurs et bornes inclus) sans
lexique ni langue : `reps (N,d)` → `états (N,d)`. ≤1M params (d=128 :
~98k). Zéro UD, zéro table. Backward complet pour le train offline ;
l'inférence reste une passe avant légère.
"""
from __future__ import annotations

import numpy as np

from ..param_epoch import bump as _bump_param_epoch


class DilatedConvBlock:
    """Conv1D causale-centrée k=3, dilation d, résidu + tanh. NumPy pur."""

    def __init__(self, d_model: int, dilation: int = 1, seed: int = 42) -> None:
        if d_model < 1:
            raise ValueError(f"d_model doit être ≥ 1 (reçu {d_model!r}).")
        if dilation < 1:
            raise ValueError(f"dilation doit être ≥ 1 (reçu {dilation!r}).")
        rng = np.random.default_rng(int(seed))
        s = float(np.sqrt(2.0 / (3 * d_model)))
        self.W = rng.normal(0, s, (3, d_model, d_model)).astype(np.float32)
        self.b = np.zeros(d_model, dtype=np.float32)
        self.dilation = int(dilation)
        self._cache: tuple = ()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(N,d) → (N,d) avec padding zéros bords (copies, jamais d'alias)."""
        X = np.asarray(x, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"(N,d) requises, reçu {X.shape}.")
        n, d = X.shape
        pad = self.dilation
        Xp = np.vstack([np.zeros((pad, d), dtype=np.float32), X, np.zeros((pad, d), dtype=np.float32)])
        idx = np.arange(n)[:, None] + np.array([0, pad, 2 * pad])[None, :]
        fen = Xp[idx]
        out = np.einsum("nki,kij->nj", fen, self.W) + self.b
        res = np.tanh(out) + X
        self._cache = (X, fen)
        return res.astype(np.float32)

    def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """d_out (N,d) → (d_input (N,d), [dW, db]). Dérivée tanh incluse."""
        X, fen = self._cache
        d = np.asarray(d_out, dtype=np.float32)
        n = X.shape[0]
        # Reproduit la passe avant (tanh) pour la dérivée — déterministe.
        idx = np.arange(n)[:, None] + np.array([0, self.dilation, 2 * self.dilation])[None, :]
        Xp = np.vstack(
            [np.zeros((self.dilation, X.shape[1]), dtype=np.float32), X,
             np.zeros((self.dilation, X.shape[1]), dtype=np.float32)]
        )
        pre = np.einsum("nki,kij->nj", Xp[idx], self.W) + self.b
        dt = d * (1.0 - np.tanh(pre) ** 2).astype(np.float32)
        dW = np.einsum("nki,nj->kij", fen, dt).astype(np.float32)
        db = dt.sum(axis=0).astype(np.float32)
        # Retour vers l'entrée via les 3 positions du noyau.
        d_in = np.zeros_like(X)
        for k, decal in enumerate((-self.dilation, 0, self.dilation)):
            contrib = dt @ self.W[k].T
            src = np.arange(n) + decal
            ok = (src >= 0) & (src < n)
            d_in[ok] += contrib[ok]
        d_in += d  # résidu
        return d_in.astype(np.float32), [dW, db]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """SGD sur [dW, db] (ignore non-finis, poids préservés)."""
        _bump_param_epoch()  # M1a : invalide le cache analyze (sens sûr).
        if not (np.isfinite(lr) and lr > 0):
            raise ValueError(f"lr doit être > 0 et fini (reçu {lr!r}).")
        dW, db = grads
        if np.all(np.isfinite(dW)) and np.all(np.isfinite(db)):
            self.W -= np.float32(lr) * dW.astype(np.float32)
            self.b -= np.float32(lr) * db.astype(np.float32)

    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b]


class ContextEncoder:
    """2 blocs dilatés (d=1 puis d=2, champ réceptif ±3) + résidus."""

    def __init__(self, d_model: int = 128, seed: int = 42) -> None:
        self.bloc1 = DilatedConvBlock(d_model, dilation=1, seed=seed)
        self.bloc2 = DilatedConvBlock(d_model, dilation=2, seed=seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(N,d) → (N,d) états contextuels."""
        return self.bloc2.forward(self.bloc1.forward(x))

    def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, list]:
        """d_out → (d_input, grads [dW1,db1,dW2,db2])."""
        d_mid, g2 = self.bloc2.backward(d_out)
        d_in, g1 = self.bloc1.backward(d_mid)
        return d_in, g1 + g2

    def update(self, grads: list, lr: float) -> None:
        """SGD (4 tenseurs)."""
        self.bloc1.update(grads[:2], lr)
        self.bloc2.update(grads[2:], lr)

    def parameters(self) -> list[np.ndarray]:
        return self.bloc1.parameters() + self.bloc2.parameters()

    @property
    def n_params(self) -> int:
        """Nombre de paramètres (garde ≤1M du plan)."""
        return sum(int(a.size) for a in self.parameters())


__all__ = ["ContextEncoder", "DilatedConvBlock"]
