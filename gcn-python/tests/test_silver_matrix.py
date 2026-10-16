# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Gate matrice silver 30 ex AVANT tête (R9) : 30/30 exigés.

Donnees locales (gold_mini lot A, DATA hors git) : ignore en CI, actif en local.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SCRIPT = RACINE / "scripts" / "silver_matrix_30.py"
GOLD_MINI = RACINE / "gcn-datasets" / "DATA" / "supervision" / "gold_mini" / "gold_mini.jsonl"


@pytest.mark.skipif(
    not GOLD_MINI.exists(), reason="DATA hors git (local uniquement)"
)
def test_matrice_silver_30_avant_tete():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        cwd=RACINE,
    )
    assert proc.returncode == 0, proc.stderr
    rapport = json.loads(proc.stdout)
    assert rapport["n"] == 30
    assert rapport["exacts"] == 30, f"matrice R9 : {rapport['erreurs']}"
    assert rapport["precision"] == 1.0
