# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests GitHub séquentiel — écriture incrémentale par langage (Fix C1 2026-10-12)."""
import json
from pathlib import Path
import gcn_scraper.sources.github_code as gh_mod
from gcn_scraper.pipeline import ScrapingPipeline


class _StubGH:
    def __init__(self, *a, **k):
        pass

    def scrape(self, languages=None, max_per_query=None, seed=None, tracker=None):
        lang = (languages or ["python"])[0]
        return [{"text": f"def f_{lang}():\n    # parce que test\n    return 1",
                 "lang": "code", "source": f"github_{lang}",
                 "url": f"https://example.test/{lang}"}]


def test_github_sequential_writes_per_lang(tmp_path, monkeypatch):
    monkeypatch.setattr(gh_mod, "GitHubCodeScraper", _StubGH)
    out = tmp_path / "c1"
    out.mkdir()
    pipe = ScrapingPipeline(out, registry_db=None)
    res = pipe.run({"github": {"token": None, "max_per_query": 1},
                    "langs": [], "prog_langs": ["python", "rust"],
                    "target_total": 10, "seed": 42, "min_quality": 0.0})
    assert res["total_written"] == 2, res
    lines = (out / res["output"]).read_text(encoding="utf-8").strip().split("\n")
    langs = sorted(json.loads(l)["lang"] for l in lines)
    assert langs == ["code_python", "code_rust"], langs


class _StubFlap:
    last_failures = 3

    def __init__(self, *a, **k):
        pass

    def scrape(self, languages=None, max_per_query=None, seed=None, tracker=None):
        type(self).last_failures = 3
        return []


def test_flapping_not_marked_done(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(gh_mod, "GitHubCodeScraper", _StubFlap)
    out = tmp_path / "c1"
    out.mkdir()
    pipe = ScrapingPipeline(out, registry_db=None)
    res = pipe.run({"github": {"token": None, "max_per_query": 1},
                    "langs": [], "prog_langs": ["python"],
                    "target_total": 10, "seed": 42, "min_quality": 0.0})
    assert res["total_written"] == 0
    ckpt = _json.loads((out / ".checkpoint.json").read_text(encoding="utf-8"))
    assert "github_python" not in ckpt, ckpt
