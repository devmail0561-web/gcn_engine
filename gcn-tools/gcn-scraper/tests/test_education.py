# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests câblage source education — C6 (stub, 100 % offline)."""
import json
import gcn_scraper.sources.education_web as edu_mod
from gcn_scraper.pipeline import ScrapingPipeline


class _StubEdu:
    def __init__(self, *a, **k):
        pass

    def scrape_all(self):
        return [
            {"title": "Cause", "text": "La photosynthèse produit du glucose car la plante capte la lumière.",
             "url": "https://fr.vikidia.test/Cause", "source": "education"},
            {"title": "Causality", "text": "Plants grow because they capture sunlight every day widely.",
             "url": "https://simple.test/Cause", "source": "education"},
        ]


def test_education_wired_and_lang_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(edu_mod, "EducationScraper", _StubEdu)
    out = tmp_path / "c6"
    out.mkdir()
    pipe = ScrapingPipeline(out, registry_db=None)
    res = pipe.run({"education": {}, "wikipedia": {"enabled": False},
                    "langs": ["fr", "en"], "prog_langs": [],
                    "target_total": 100, "seed": 42, "min_quality": 0.0})
    assert res["total_written"] == 2, res
    lines = (out / res["output"]).read_text(encoding="utf-8").strip().split("\n")
    by_lang = sorted(json.loads(l)["lang"] for l in lines)
    assert by_lang == ["en", "fr"], by_lang
