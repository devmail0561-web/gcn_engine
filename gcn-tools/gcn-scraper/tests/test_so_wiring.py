# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests câblage source so — pipeline (stub, 100 % offline)."""
import json
import gcn_scraper.sources.so_qa as so_mod
from gcn_scraper.pipeline import ScrapingPipeline


class _StubSO:
    def __init__(self, *a, **k):
        pass

    def scrape(self, languages=None, max_per_query=None, seed=None, tracker=None):
        return [{"text": "Why X fails in production environments widely? Because Y happens "
                         "under load conditions every day here.",
                "lang": "en", "source": "stackoverflow",
                "url": "https://stackoverflow.com/q/9",
                "licence": "CC-BY-SA-4.0", "auteur": "Ada"}]


def test_so_wired_with_attribution(tmp_path, monkeypatch):
    monkeypatch.setattr(so_mod, "StackOverflowScraper", _StubSO)
    out = tmp_path / "c1doc"
    out.mkdir()
    pipe = ScrapingPipeline(out, registry_db=None)
    res = pipe.run({"so": {"max_per_query": 10}, "wikipedia": {"enabled": False},
                    "langs": ["en"], "prog_langs": ["python"],
                    "target_total": 100, "seed": 42, "min_quality": 0.0})
    assert res["total_written"] == 2, res  # 1 notice → 2 phrases après split
    lines = (out / res["output"]).read_text(encoding="utf-8").strip().split("\n")
    rec = json.loads(lines[0])
    assert rec["source"] == "stackoverflow"
    assert rec["licence"] == "CC-BY-SA-4.0" and rec["auteur"] == "Ada"
