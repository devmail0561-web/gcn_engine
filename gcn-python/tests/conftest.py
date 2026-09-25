# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest


@pytest.fixture
def taxonomy_dir() -> Path:
    candidates = [
        Path(__file__).parents[2] / "gcn-references" / "taxonomies",
        Path(__file__).parents[2] / "gcn-core" / "data" / "taxonomies",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    pytest.skip(f"Taxonomy dir not found (tried: {candidates})")


@pytest.fixture
def paper_examples_json() -> Path:  # L5 : renommé (yaml→json)
    p = Path(__file__).parents[2] / "gcn-core" / "tests" / "fixtures" / "paper_examples.json"
    if not p.exists():
        pytest.skip(f"paper_examples.json not found: {p}")
    return p


@pytest.fixture
def datasets_dir() -> Path:
    d = Path(__file__).parents[2] / "gcn-datasets" / "examples"
    if not d.is_dir():
        pytest.skip(f"Datasets dir not found: {d}")
    return d
