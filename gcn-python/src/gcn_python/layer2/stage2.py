# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Étage-2 deux-étages : classifieur 12 classes (11 relations + REJECT).

Entrée : vecteur 36 dims AGNOSTIQUE produit par `scripts/build_stage2.py`
(prior proposeur one-hot + confiance + direction + structure des spans).
AUCUN lemme, AUCUNE langue, AUCUN lexique ici — la langue vit dans
`gcn-tools` (étage-1) et les données. Voir `tests/test_stage2.py`
(garde anti-lexique).

MLP NumPy : fc(36→32)+ReLU → fc(32→12), softmax cross-entropie,
SGD mini-batch + class weights (rejet ~2× les classes).
"""
from __future__ import annotations

import numpy as np

from ..security import guarded_np_load

N_FEATURES = 36
N_CLASSES = 12
REJECT_LABEL = 11


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=-1, keepdims=True)


class Stage2Classifier:
    """MLP 12 classes pour l'étage-2 (dims fixées à la construction)."""

    def __init__(self, hidden: int = 32, seed: int = 42,
                 weight_decay: float = 1e-4):
        if hidden < 1:
            raise ValueError(f"hidden doit être ≥ 1 (reçu {hidden!r}).")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                f"weight_decay doit être fini ≥ 0 (reçu {weight_decay!r}).")
        rng = np.random.default_rng(seed)
        self.W1 = (rng.normal(0, np.sqrt(2.0 / N_FEATURES),
                              (hidden, N_FEATURES))).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.normal(0, np.sqrt(2.0 / hidden),
                              (N_CLASSES, hidden))).astype(np.float32)
        self.b2 = np.zeros(N_CLASSES, dtype=np.float32)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.weight_decay = float(weight_decay)
        # Scaler z-score (optionnel, fitté sur train uniquement) :
        #MU0/SIG0 = identité. Voir fit_scaler/transform.
        self.mu = np.zeros(N_FEATURES, dtype=np.float32)
        self.sigma = np.ones(N_FEATURES, dtype=np.float32)

    def fit_scaler(self, X: np.ndarray) -> None:
        """Calcule mu/sigma sur X (train uniquement — pas de fuite)."""
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sigma = X.std(axis=0).astype(np.float32)
        self.sigma[self.sigma < 1e-6] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"forward attend (N,{N_FEATURES}), reçu {X.shape}.")
        h = _relu(X @ self.W1.T + self.b1)
        return h @ self.W2.T + self.b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1).astype(np.int64)

    @staticmethod
    def class_weights(y: np.ndarray) -> np.ndarray:
        """Poids inverses fréquences (moyenne 1.0), comme --weighted-loss."""
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        if (counts == 0).any():
            raise ValueError(
                f"class_weights : classes absentes {np.where(counts == 0)[0]}.")
        w = len(y) / (N_CLASSES * counts)
        return (w / w.mean()).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, val: tuple | None = None,
            epochs: int = 200, lr: float = 0.05, batch: int = 64,
            patience: int = 30, verbose: bool = False) -> dict:
        """Mini-batch SGD + early stopping sur acc val (restore best).

        Retourne l'historique (train_loss, val_acc, best_epoch, epochs_run).
        Déterministe à seed fixé (shuffle seedé une fois par epoch).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if len(X) != len(y):
            raise ValueError(f"X/y désalignés : {len(X)} vs {len(y)}.")
        if set(np.unique(y)) - set(range(N_CLASSES)):
            raise ValueError(f"labels hors [0,{N_CLASSES}[.")
        w = self.class_weights(y)
        rng = np.random.default_rng(self.seed)
        best = {"acc": -1.0, "params": None, "epoch": 0}
        hist = {"train_loss": [], "val_acc": [], "best_epoch": 0,
                "epochs_run": 0}
        no_improve = 0
        for ep in range(1, epochs + 1):
            perm = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                idx = perm[s:s + batch]
                xb, yb = X[idx], y[idx]
                h = _relu(xb @ self.W1.T + self.b1)
                logits = h @ self.W2.T + self.b2
                probs = _softmax(logits)
                n = len(xb)
                dlog = probs
                dlog[np.arange(n), yb] -= 1.0
                dlog *= (w[yb] / n)[:, None]
                dW2 = dlog.T @ h + self.weight_decay * self.W2
                db2 = dlog.sum(axis=0)
                dh = (dlog @ self.W2) * (h > 0)
                dW1 = dh.T @ xb + self.weight_decay * self.W1
                db1 = dh.sum(axis=0)
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
            with np.errstate(divide="ignore"):
                pl = _softmax(self.forward(X))
                loss = float(-np.log(pl[np.arange(len(X)), y] + 1e-12)
                             .mean())
            hist["train_loss"].append(loss)
            if val is not None:
                acc = float((self.predict(val[0]) == val[1]).mean())
                hist["val_acc"].append(acc)
                if acc > best["acc"]:
                    best = {"acc": acc,
                            "params": (self.W1.copy(), self.b1.copy(),
                                       self.W2.copy(), self.b2.copy()),
                            "epoch": ep}
                    hist["best_epoch"] = ep
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            hist["epochs_run"] = ep
            if verbose and ep % 50 == 0:
                print(f"ep {ep}: loss={loss:.4f} "
                      f"val_acc={hist['val_acc'][-1] if val else '-'}")
        if best["params"] is not None:
            self.W1, self.b1, self.W2, self.b2 = best["params"]
        return hist

    def save(self, path: str, provenance: dict | None = None) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 mu=self.mu, sigma=self.sigma,
                 hidden=np.array([self.hidden]), seed=np.array([self.seed]),
                 prov=np.array([json_dumps(provenance or {})], dtype=object))

    @classmethod
    def load(cls, path: str) -> tuple["Stage2Classifier", dict]:
        import json as _json
        # F1 audit : pickle audité (prov = object array) via la garde centrale.
        data = guarded_np_load(path)
        obj = cls(hidden=int(data["hidden"][0]), seed=int(data["seed"][0]))
        obj.W1 = data["W1"].astype(np.float32)
        obj.b1 = data["b1"].astype(np.float32)
        obj.W2 = data["W2"].astype(np.float32)
        obj.b2 = data["b2"].astype(np.float32)
        # Anciens checkpoints sans scaler : identité (comportement historique).
        obj.mu = (data["mu"].astype(np.float32) if "mu" in data
                  else np.zeros(N_FEATURES, dtype=np.float32))
        obj.sigma = (data["sigma"].astype(np.float32) if "sigma" in data
                     else np.ones(N_FEATURES, dtype=np.float32))
        if obj.W1.shape != (obj.hidden, N_FEATURES) or \
                obj.W2.shape != (N_CLASSES, obj.hidden):
            raise ValueError(f"checkpoint {path} : dims incompatibles "
                             f"({obj.W1.shape}, {obj.W2.shape}).")
        return obj, _json.loads(str(data["prov"][0]))


def json_dumps(d: dict) -> str:
    import json as _json
    return _json.dumps(d, ensure_ascii=False, sort_keys=True)
