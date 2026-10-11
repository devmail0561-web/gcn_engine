# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""R1 — étapes typées du raisonnement (codes D-AGNOSTIC, contenus libres).

Aucun nouveau code réponse : `abandoned:` est réutilisé depuis le dialogue ;
`multigoal:` / `oversize:` sont des motifs de `Degrade`, pas des codes.
"""
from __future__ import annotations

REASON_VERSION = "reason/1.0"

#: Codes d'étape fermés (même modèle que le DSL Pearl : fixes, contenu libre).
STEP_CODES = frozenset({
    "goal:", "subgoal:", "query:", "observe:", "infer:",
    "verify:", "answer:", "abandoned:",
})

STATUS_OK = "ok"
STATUS_DEGRADE = "degrade"


def make_step(code: str, tour: int, premises: tuple = (),
              conf: float | None = None, motif: str | None = None,
              extra: dict | None = None) -> dict:
    """Construit une étape de trace. Lève ValueError sur code inconnu."""
    if code not in STEP_CODES:
        raise ValueError(f"étape inconnue {code!r} (codes: {sorted(STEP_CODES)})")
    step = {"code": code, "tour": int(tour), "premises": list(premises)}
    if conf is not None:
        step["conf"] = float(conf)
    if motif is not None:
        step["motif"] = motif
    if extra:
        step.update(extra)
    return step


def reason_response(status: str, answer: dict, trace: list) -> dict:
    """Enveloppe versionnée (modèle EXCHANGE_SCHEMA_VERSION : clé future ignorée)."""
    if status not in (STATUS_OK, STATUS_DEGRADE):
        raise ValueError(f"status inconnu {status!r}")
    return {"version": REASON_VERSION, "status": status,
            "answer": answer, "trace": trace}
