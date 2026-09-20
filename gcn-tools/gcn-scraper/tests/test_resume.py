# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""C2/C3 : run fresh ne skippe rien ; run resume respecte les budgets remplis."""
import json

import gcn_scraper.sources.wikipedia as wiki_mod
from gcn_scraper.pipeline import ScrapingPipeline

ITEMS = [
    {
        "text": f"Phrase de test numéro {i} avec assez de mots pour passer le filtre qualité.",
        "lang": "fr",
        "source": "wikipedia_fr",
        "url": f"https://fr.wikipedia.org/wiki/Article_{i}",
    }
    for i in range(150)
]
# Budget fr pour target_total=100 (plancher max(100,…)) → 100 phrases.


def _mock_scrape(self, tracker=None, seed=None):
    return [dict(it) for it in ITEMS]


def _base_config(**over):
    cfg = {
        "langs": ["fr"],
        "prog_langs": [],
        "target_total": 100,
        "min_quality": 0.0,
        "seed": 7,
    }
    cfg.update(over)
    return cfg


def test_fresh_run_does_not_skip_previous_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_mod.WikipediaLangScraper, "scrape", _mock_scrape)
    pipe = ScrapingPipeline(tmp_path, "test-agent")
    r1 = pipe.run(_base_config())
    assert r1["total_written"] == 100
    # 2e run SANS --resume : doit réécrire autant (fresh = fresh, cf C2)
    r2 = pipe.run(_base_config())
    assert r2["total_written"] == 100


def test_registry_skips_seen_urls_cross_run(tmp_path, monkeypatch):
    """Run 1 enregistre les URLs dans le registre ; run 2 (fresh) les skippe toutes."""
    monkeypatch.setattr(wiki_mod.WikipediaLangScraper, "scrape", _mock_scrape)
    registry_db = tmp_path / "url_registry.db"
    pipe = ScrapingPipeline(tmp_path, "test-agent", registry_db=registry_db)
    # target_total=200 : force run 1 à traiter les 150 items en entier.
    r1 = pipe.run(_base_config(target_total=200))
    assert r1["total_written"] == 150

    # Run 2 (fresh, même registre) : toutes les 150 URLs sont déjà connues → 0 phrase.
    r2 = pipe.run(_base_config(target_total=200))
    assert r2["total_written"] == 0, (
        "Run 2 devrait skipper toutes les URLs déjà dans le registre"
    )


def test_no_registry_does_not_skip_cross_run(tmp_path, monkeypatch):
    """Sans registre (--no-registry), chaque run repart de zéro."""
    monkeypatch.setattr(wiki_mod.WikipediaLangScraper, "scrape", _mock_scrape)
    # registry_db=None → pas de registre, comportement original
    pipe = ScrapingPipeline(tmp_path, "test-agent", registry_db=None)
    r1 = pipe.run(_base_config())
    r2 = pipe.run(_base_config())
    assert r1["total_written"] == 100
    assert r2["total_written"] == 100


def test_resume_honours_filled_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_mod.WikipediaLangScraper, "scrape", _mock_scrape)
    pipe = ScrapingPipeline(tmp_path, "test-agent")
    r1 = pipe.run(_base_config(target_total=100))
    assert r1["total_written"] == 100
    # Simule une source non terminée alors que le budget est plein (cf C3) :
    # on retire wikipedia_fr des "done" mais le budget fr (100) est atteint.
    ckpt = tmp_path / ".checkpoint.json"
    state = json.loads(ckpt.read_text(encoding="utf-8"))
    state.pop("wikipedia_fr", None)
    ckpt.write_text(json.dumps(state), encoding="utf-8")
    r2 = pipe.run(_base_config(target_total=100, resume=True))
    assert r2["total_written"] == 0
