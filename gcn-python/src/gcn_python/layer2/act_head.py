# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tête act (1, 3) — MLP NumPy produisant ``cir.intent`` (refacto tête act).

Références : ``plan-remodelage-tout-appris.md`` (étape 3, [T4], [R9]),
``bascule-intents.md`` (``layer2/reference.py`` REMPLACÉ par tête act ;
option (b) : les 21 intents survivent SOUS ``question`` — ici on ne touche
pas à la tête 21 legacy), [R4] (amendement s3 = +``act_types``).

``ActHead`` : ``Linear(d_model→64) + ReLU → Linear(64→3)`` + softmax ;
entrée = rep de clause ``(d_model,)``, sortie = distribution sur
``ACT_TYPES``. Entraînement plus tard (gold-3000, weight silver 0.0 par
défaut [R9]) ; ``cir.intent`` v0 vient des règles (``act.py``).
Format s3 : ``{"format_refacto", "act_types", "d_model", "params"}``.
"""

from __future__ import annotations

import json

import numpy as np

from .act import ACT_TYPES


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits - logits.max()
    e = np.exp(m)
    return e / e.sum()


class ActHead:
    """Tête d'actes apprenable (NumPy pur, style ``reference.py``)."""

    def __init__(self, d_model: int = 128, hidden: int = 64, seed: int = 42) -> None:
        if d_model < 1 or hidden < 1:
            raise ValueError(f"d_model/hidden ≥ 1 (reçus {d_model!r}, {hidden!r}).")
        self.d_model = d_model
        self.hidden = hidden
        self.act_types = list(ACT_TYPES)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_model), (hidden, d_model)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (3, hidden)).astype(np.float32)
        self.b2 = np.zeros(3, dtype=np.float32)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``(d_model,) → (3,)`` distribution (copie, cache forward pour backward)."""
        vec = np.asarray(x, dtype=np.float32).ravel()
        if vec.shape[0] != self.d_model:
            raise ValueError(f"entrée ({vec.shape[0]},) ≠ d_model ({self.d_model}).")
        h = np.maximum(0.0, vec @ self.W1.T + self.b1)
        logits = h @ self.W2.T + self.b2
        probs = _softmax(logits)
        self._cache = {"x": vec.copy(), "h": h.copy(), "probs": probs.copy()}
        return probs.copy()

    def predict(self, x: np.ndarray) -> str:
        """Acte gagnant (argmax déterministe — inférence seule)."""
        return self.act_types[int(np.argmax(self.forward(x)))]

    def loss_and_grad(self, x: np.ndarray, gold_idx: int) -> tuple[float, dict]:
        """Cross-entropie + grads (``gold_idx`` ∈ {0, 1, 2})."""
        if gold_idx not in (0, 1, 2):
            raise ValueError(f"gold_idx ∈ {{0,1,2}} (reçu {gold_idx!r}).")
        probs = self.forward(x)
        grad_logits = probs.copy()
        grad_logits[gold_idx] -= 1.0
        h = self._cache["h"]
        vec = self._cache["x"]
        dW2 = np.outer(grad_logits, h)
        db2 = grad_logits.copy()
        dh = grad_logits @ self.W2 * (h > 0).astype(np.float32)
        dW1 = np.outer(dh, vec)
        db1 = dh.copy()
        return float(-np.log(max(probs[gold_idx], 1e-12))), {
            "W1": dW1, "b1": db1, "W2": dW2, "db2": db2,
        }

    def apply_grads(self, grads: dict, lr: float) -> None:
        """Descente SGD (grads de ``loss_and_grad``)."""
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["db2"]

    def parameters(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    # ── format s3 ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Dictionnaire s3 : format + actes + d_model + params."""
        return {
            "format_refacto": 20261007,
            "act_types": list(self.act_types),
            "d_model": self.d_model,
            "hidden": self.hidden,
            "params": {k: v.tolist() for k, v in self.parameters().items()},
        }

    def to_json(self) -> str:
        """Sérialise s3."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> ActHead:
        """Reconstruit depuis s3 (vérifie format + actes + dims, fail-fast)."""
        if data.get("format_refacto") != 20261007:
            raise ValueError("format legacy-UD : convertir ou ré-entraîner.")
        if list(data.get("act_types", [])) != list(ACT_TYPES):
            raise ValueError(f"act_types ≠ {list(ACT_TYPES)} : {data.get('act_types')!r}.")
        params = data.get("params", {})
        d_model = int(data.get("d_model", 0))
        hidden = int(data.get("hidden", 0))
        obj = cls(d_model=d_model, hidden=hidden)
        obj.W1 = np.asarray(params["W1"], dtype=np.float32)
        obj.b1 = np.asarray(params["b1"], dtype=np.float32)
        obj.W2 = np.asarray(params["W2"], dtype=np.float32)
        obj.b2 = np.asarray(params["b2"], dtype=np.float32)
        if obj.W1.shape != (hidden, d_model) or obj.W2.shape != (3, hidden):
            raise ValueError("dims params incohérentes avec d_model/hidden.")
        return obj

    @classmethod
    def from_json(cls, s: str) -> ActHead:
        """Recrée depuis ``to_json()`` (mêmes vérifications)."""
        return cls.from_dict(json.loads(s))


__all__ = ["ActHead"]
