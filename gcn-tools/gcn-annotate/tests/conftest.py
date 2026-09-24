# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "src"))
# Tester le gcn-python du workspace (pas l'editable installé ailleurs)
_gcn_python_src = _here.parents[2] / "gcn-python" / "src"
if _gcn_python_src.is_dir():
    sys.path.insert(0, str(_gcn_python_src))
