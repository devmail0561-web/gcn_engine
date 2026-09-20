# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests WebSearchScraper — mocks réseau, self.user_agent, scrape() de base."""
import pytest
from unittest.mock import MagicMock, patch
from gcn_scraper.sources.web_search import WebSearchScraper
from gcn_scraper.sources._net import DEFAULT_USER_AGENT


def test_user_agent_attribute_exists():
    """Régression : self.user_agent manquant causait AttributeError sur _search_openalex."""
    scraper = WebSearchScraper(user_agent="TestAgent/1.0")
    assert scraper.user_agent == "TestAgent/1.0"


def test_user_agent_passed_to_session():
    scraper = WebSearchScraper(user_agent="TestAgent/1.0")
    assert scraper.session.headers.get("User-Agent") == "TestAgent/1.0"


def test_search_openalex_uses_user_agent(monkeypatch):
    """_search_openalex ne doit pas lever AttributeError même si retry_get retourne None."""
    scraper = WebSearchScraper(user_agent="TestAgent/1.0")
    monkeypatch.setattr(
        "gcn_scraper.sources.web_search._retry_get",
        lambda *a, **k: (None, scraper.session),
    )
    result = scraper._search_openalex("test query", max_results=5)
    assert result == []


def test_search_pubmed_uses_user_agent(monkeypatch):
    scraper = WebSearchScraper(user_agent="TestAgent/1.0")
    monkeypatch.setattr(
        "gcn_scraper.sources.web_search._retry_get",
        lambda *a, **k: (None, scraper.session),
    )
    result = scraper._search_pubmed("test query", max_results=5)
    assert result == []


def test_scrape_returns_list_on_empty_queries(monkeypatch):
    """scrape() sans requêtes configurées retourne une liste vide (pas d'erreur)."""
    monkeypatch.setattr(
        "gcn_scraper.sources.web_search._build_queries",
        lambda lang: [],
    )
    scraper = WebSearchScraper()
    result = scraper.scrape(langs=["fr", "en"])
    assert isinstance(result, list)
    assert result == []


def test_scrape_deduplicates_urls(monkeypatch):
    """Deux hits avec la même URL ne produisent qu'un seul résultat."""
    scraper = WebSearchScraper()
    monkeypatch.setattr(
        "gcn_scraper.sources.web_search._build_queries",
        lambda lang: ["test query"],
    )
    fake_hit = {
        "url": "https://example.com/page",
        "text": "Ceci est un texte suffisamment long pour passer les filtres qualité.",
        "lang": "fr",
        "title": "Test",
        "relevance_score": 1,
        "found_by": ["duckduckgo"],
    }
    monkeypatch.setattr(scraper, "search_multi", lambda q, lang="auto": [fake_hit, fake_hit])
    result = scraper.scrape(langs=["fr"])
    urls = [r["url"] for r in result]
    assert len(urls) == len(set(urls)), "URLs dupliquées dans les résultats"
