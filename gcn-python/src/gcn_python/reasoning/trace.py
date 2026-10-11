# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""R1 — journal rejouable des traces de raisonnement (JSONL, esprit graph_log)."""
from __future__ import annotations

import json
from pathlib import Path


def dump_trace(trace: list, path: str | Path) -> Path:
    """Écrit la trace (une étape par ligne). Retourne le chemin."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for step in trace:
            f.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")
    return path


def load_trace(path: str | Path) -> list:
    """Relit une trace. Lève ValueError si une étape est malformée."""
    from .steps import STEP_CODES
    out = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            if step.get("code") not in STEP_CODES:
                raise ValueError(f"trace : code inconnu {step.get('code')!r}")
            out.append(step)
    return out
