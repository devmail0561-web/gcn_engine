# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Gate grid F1 sur jouet : GEL par défaut, grid sous --force ([R8]/[R10]).

Données locales (gold_mini lot A, DATA hors git) : ignoré en CI, actif en local.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SCRIPT = RACINE / "scripts" / "grid_f1_thresholds.py"
GOLD_MINI = RACINE / "gcn-datasets" / "DATA" / "supervision" / "gold_mini" / "gold_mini.jsonl"


def lance(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, check=False, cwd=RACINE)


@pytest.mark.skipif(not GOLD_MINI.exists(), reason="DATA hors git (local uniquement)")
def test_grid_gele_sans_force():
    proc = lance()
    assert proc.returncode == 2, proc.stderr
    rapport = json.loads(proc.stdout)
    assert rapport["verdict"] == "GEL"
    assert "gelée" in rapport["motif"]


@pytest.mark.skipif(not GOLD_MINI.exists(), reason="DATA hors git (local uniquement)")
def test_grid_force_demo_non_adopte():
    proc = lance("--force")
    assert proc.returncode == 0, proc.stderr
    rapport = json.loads(proc.stdout)
    assert rapport["verdict"].startswith("GO-DÉMO")
    assert rapport["gele"] is not None, "seuils non adoptés sous gel"
    m = rapport["meilleur"]
    assert (m["edge_threshold"], m["theta_ambiguity"], m["temperature"]) != (0.0, 0.65, 1.0)
    assert m["n_tries"] == 250
