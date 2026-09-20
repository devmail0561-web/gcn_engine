# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""C4 : shuffled()/page_offset() doivent être stables entre processus (même seed)."""
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).parent.parent / "src")
SNIPPET = (
    "from gcn_scraper.diversity import shuffled, page_offset;"
    "print(shuffled(list(range(50)), 42, 'wiki-fr'));"
    "print(page_offset(42, 'hal', 100))"
)


def _run_with_hashseed(hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True, text=True, env={**env, "PYTHONPATH": SRC},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_shuffled_stable_across_processes():
    assert _run_with_hashseed("1") == _run_with_hashseed("2")


def test_seed_zero_is_valid_seed():
    from gcn_scraper.diversity import page_offset, shuffled
    # seed=0 doit diversifier comme une vraie graine (convention : seul None = fixe)
    a = shuffled(list(range(20)), 0, "s")
    b = shuffled(list(range(20)), 0, "s")
    assert a == b  # déterministe
    assert a == shuffled(list(range(20)), None, "s")  # None documenté = ordre fixe seed 0
    assert page_offset(1, "hal", 100) in (0, 100, 200, 300)
