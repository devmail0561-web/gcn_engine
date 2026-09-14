from pathlib import Path
import pytest


@pytest.fixture
def taxonomy_dir() -> Path:
    d = Path(__file__).parents[2] / "gcn-core" / "data" / "taxonomies"
    if not d.is_dir():
        pytest.skip(f"Taxonomy dir not found: {d}")
    return d


@pytest.fixture
def paper_examples_yaml() -> Path:
    p = Path(__file__).parents[2] / "gcn-core" / "tests" / "fixtures" / "paper_examples.yaml"
    if not p.exists():
        pytest.skip(f"paper_examples.yaml not found: {p}")
    return p


@pytest.fixture
def datasets_dir() -> Path:
    d = Path(__file__).parents[2] / "gcn-core" / "datasets" / "examples"
    if not d.is_dir():
        pytest.skip(f"Datasets dir not found: {d}")
    return d
