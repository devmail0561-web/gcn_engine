# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests routage bucket code_* — Fix C1 2026-10-11 (rejet 100 % du code)."""
import io
from pathlib import Path
from gcn_scraper.balance_tracker import BalanceTracker
from gcn_scraper.filters.deduplicator import Deduplicator
from gcn_scraper.filters.quality_scorer import QualityScorer
from gcn_scraper.pipeline import ScrapingPipeline


def _run(items, budget, **kw):
    pipe = ScrapingPipeline(Path("/tmp/nonexistent-probe"))
    out, src = io.StringIO(), io.StringIO()
    n = pipe._process_texts(
        items, out, QualityScorer(), Deduplicator(),
        BalanceTracker(budget), 0.0,
        skip_split=True, default_lang="code", source_out=src, **kw)
    return n, out.getvalue()


def test_github_routed_to_code_bucket():
    items = [{"text": "def f():\n    # parce que x\n    return 1",
              "lang": "code", "source": "github_python", "url": "https://u/1"}]
    n, val = _run(items, {"code_python": 10})
    assert n == 1, val
    assert '"code_python"' in val


def test_retro_global_code_bucket_untouched():
    items = [{"text": "def f():\n    return 1",
              "lang": "code", "source": "github_python", "url": "https://u/2"}]
    n, val = _run(items, {"code": 10})
    assert n == 1, val
    assert '"code"' in val


def test_flush_periodique_durabilite(tmp_path):
    import os
    from gcn_scraper.pipeline import ScrapingPipeline
    from gcn_scraper.balance_tracker import BalanceTracker
    from gcn_scraper.filters.deduplicator import Deduplicator
    from gcn_scraper.filters.quality_scorer import QualityScorer
    pipe = ScrapingPipeline(tmp_path)
    f = open(tmp_path / "out.jsonl", "w", encoding="utf-8")
    items = [{"text": f"phrase code numero {i} avec du contenu supplementaire x",
              "lang": "fr", "source": "t", "url": ""} for i in range(150)]
    n = pipe._process_texts(items, f, QualityScorer(), Deduplicator(),
                            BalanceTracker({"fr": 1000}), 0.0, skip_split=True)
    assert n == 150
    assert os.fstat(f.fileno()).st_size > 0  # visible avant close = flush fait
    f.close()
