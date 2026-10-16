# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Têtes d'arêtes factorisées (R2) : existence × base × flags → 19 relations.

Remplace la classification à plat 19 sorties (qui produit des classes vides :
8 relations v3 à 0 occurrence → gel R8/R10) par des facteurs partagés :
- base : 11 relations directes (toutes >= 30 en v4/final) ;
- ternary : none | conditional | mediated | joint ;
- special : normal | analogy | counterfactual.

`compose()` reconstruit les 19 relations de façon déterministe et documentée
(règles de reconstruction, pas de lexique de langue) :

- special=analogy → analogy ; special=counterfactual → counterfactual ;
- ternary=none → base ;
- base=cause + ternary → conditional/mediated/joint_cause ;
- base=prevent + ternary → conditional/mediated/joint_prevent ;
- autre base + ternary≠none → base (ternaire ignoré, compté).

Zéro UD/lemme/table de langue — indices RELATION_TYPES seuls.
"""
from __future__ import annotations

import numpy as np

BASE_RELATIONS = (
    "cause",
    "enable",
    "prevent",
    "condition",
    "concession",
    "sequence",
    "motivation",
    "filter",
    "opposition",
    "data_dependency",
    "control_dependency",
)

TERNARY = ("none", "conditional", "mediated", "joint")

SPECIAL = ("normal", "analogy", "counterfactual")

# 19 relations cibles (ordre RELATION_TYPES) reconstruites par compose().
RELATION_19 = (
    "cause",
    "enable",
    "prevent",
    "condition",
    "concession",
    "sequence",
    "motivation",
    "filter",
    "opposition",
    "data_dependency",
    "control_dependency",
    "analogy",
    "counterfactual",
    "conditional_cause",
    "mediated_cause",
    "joint_cause",
    "conditional_prevent",
    "mediated_prevent",
    "joint_prevent",
)


def decompose(relation: str) -> tuple[str, str, str]:
    """19 → (base, ternary, special). Inverse total de compose()."""
    if relation == "analogy":
        return ("cause", "none", "analogy")
    if relation == "counterfactual":
        return ("cause", "none", "counterfactual")
    for kind in ("conditional", "mediated", "joint"):
        if relation == f"{kind}_cause":
            return ("cause", kind, "normal")
        if relation == f"{kind}_prevent":
            return ("prevent", kind, "normal")
    if relation in BASE_RELATIONS:
        return (relation, "none", "normal")
    raise ValueError(f"relation inconnue : {relation!r}.")


def compose(base: str, ternary: str, special: str) -> str:
    """(base, ternary, special) → 1 des 19. Total et déterministe."""
    if special == "analogy":
        return "analogy"
    if special == "counterfactual":
        return "counterfactual"
    if ternary == "none":
        if base not in BASE_RELATIONS:
            raise ValueError(f"base inconnue : {base!r}.")
        return base
    if base == "cause":
        return f"{ternary}_cause"
    if base == "prevent":
        return f"{ternary}_prevent"
    if base not in BASE_RELATIONS:
        raise ValueError(f"base inconnue : {base!r}.")
    # Base non causale + ternaire : ternaire ignoré (documenté, compté en eval).
    return base


class FactorizedEdgeHead:
    """3 petites têtes linéaires sur vecteur d'arête (NumPy, CPU).

    Remplace 1 tête (d_edge → 19). Params : (11 + 4 + 3) × (d+1) au lieu de
    19 × (d+1) — partage statistique entre variantes d'une même base.
    """

    def __init__(self, d_edge: int, seed: int = 42) -> None:
        if d_edge < 1:
            raise ValueError(f"d_edge doit être ≥ 1 (reçu {d_edge!r}).")
        rng = np.random.default_rng(int(seed))
        s = float(np.sqrt(1.0 / d_edge))
        self.W_base = rng.normal(0, s, (len(BASE_RELATIONS), d_edge)).astype(np.float32)
        self.b_base = np.zeros(len(BASE_RELATIONS), dtype=np.float32)
        self.W_tern = rng.normal(0, s, (len(TERNARY), d_edge)).astype(np.float32)
        self.b_tern = np.zeros(len(TERNARY), dtype=np.float32)
        self.W_spec = rng.normal(0, s, (len(SPECIAL), d_edge)).astype(np.float32)
        self.b_spec = np.zeros(len(SPECIAL), dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vecteur d'arête → (logits_base(11,), logits_tern(4,), logits_spec(3,))."""
        v = np.asarray(x, dtype=np.float32).ravel()
        return (
            (self.W_base @ v + self.b_base).astype(np.float32),
            (self.W_tern @ v + self.b_tern).astype(np.float32),
            (self.W_spec @ v + self.b_spec).astype(np.float32),
        )

    def predict(self, x: np.ndarray) -> str:
        """Prédiction 19 via argmax facteurs + compose()."""
        lb, lt, ls = self.forward(x)
        return compose(
            BASE_RELATIONS[int(np.argmax(lb))],
            TERNARY[int(np.argmax(lt))],
            SPECIAL[int(np.argmax(ls))],
        )

    def parameters(self) -> list[np.ndarray]:
        return [self.W_base, self.b_base, self.W_tern, self.b_tern, self.W_spec, self.b_spec]

    def loss_and_update(
        self,
        xs: np.ndarray,
        base_gold: np.ndarray,
        tern_gold: np.ndarray,
        spec_gold: np.ndarray,
        lr: float,
    ) -> float:
        """Un pas SGD multi-tâche sur les 3 facteurs (copies, jamais d'alias).

        Retourne la CE totale (base + ternary + special). Chaque tête est une
        softmax linéaire : dW = outer(p - y, x) / n. Zéro lexique, CPU pur.
        Sans reflux vers le tronc (auxiliaire) — voir ``loss_update_dx``.
        """
        loss, _ = self.loss_update_dx(xs, base_gold, tern_gold, spec_gold, None, lr)
        return loss

    def loss_update_dx(
        self,
        xs: np.ndarray,
        base_gold: np.ndarray,
        tern_gold: np.ndarray,
        spec_gold: np.ndarray,
        masks: tuple | None,
        lr: float,
    ) -> tuple[float, np.ndarray]:
        """Loss + update + reflux dX (tête primaire, espace facteur D-R8F).

        ``masks`` : (base_mask, tern_mask, spec_mask) booléens (classes
        gardées) ou None (= tout observé). Retourne (loss_totale, dX (n,d)).
        dX = Σ_têtes dlogits_t @ W_t (déjà /n). Mise à jour SGD immédiate.
        """
        if not (np.isfinite(lr) and lr > 0):
            raise ValueError(f"lr doit être > 0 et fini (reçu {lr!r}).")
        X = np.asarray(xs, dtype=np.float32)
        gb = np.asarray(base_gold, dtype=np.int64).ravel()
        gt = np.asarray(tern_gold, dtype=np.int64).ravel()
        gs = np.asarray(spec_gold, dtype=np.int64).ravel()
        n = X.shape[0]
        if n == 0:
            return 0.0, np.zeros_like(X)
        mb, mt, ms = (None, None, None) if masks is None else masks
        total = 0.0
        dX = np.zeros_like(X)
        for W, b, g, mask in (
            (self.W_base, self.b_base, gb, mb),
            (self.W_tern, self.b_tern, gt, mt),
            (self.W_spec, self.b_spec, gs, ms),
        ):
            logits = X @ W.T + b
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            probs = e / e.sum(axis=1, keepdims=True)
            garde = np.ones(n, dtype=bool)
            if mask is not None:
                mk = np.asarray(mask, dtype=bool).ravel()
                garde &= mk[g]
            if not garde.any():
                continue
            total += float(-np.log(np.maximum(probs[garde][np.arange(garde.sum()), g[garde]], 1e-12)).mean())
            dlog = np.zeros_like(probs)
            dlog[garde] = probs[garde]
            dlog[garde, g[garde]] -= 1.0
            dlog[garde] /= garde.sum()
            dW = dlog.T @ X
            db = dlog.sum(axis=0)
            if np.all(np.isfinite(dW)) and np.all(np.isfinite(db)):
                W -= np.float32(lr) * dW.astype(np.float32)
                b -= np.float32(lr) * db.astype(np.float32)
            dX += (dlog @ W).astype(np.float32)
        return float(total), dX


__all__ = [
    "BASE_RELATIONS",
    "RELATION_19",
    "SPECIAL",
    "TERNARY",
    "FactorizedEdgeHead",
    "compose",
    "decompose",
]
