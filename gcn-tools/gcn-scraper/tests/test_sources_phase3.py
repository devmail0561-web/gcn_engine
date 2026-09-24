# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Phase 3 (M5-M13, M19) : pagination réelle, parsing news, srlimit recalculé."""
import xml.etree.ElementTree as ET

import gcn_scraper.sources.arxiv as arxiv_mod
import gcn_scraper.sources.github_code as gh_mod
import gcn_scraper.sources.hal_scientific as hal_mod
import gcn_scraper.sources.news_rss as news_mod
import gcn_scraper.sources.wikipedia as wiki_mod
from gcn_scraper.sources.arxiv import ArXivScraper
from gcn_scraper.sources.github_code import GitHubCodeScraper
from gcn_scraper.sources.hal_scientific import HALScraper
from gcn_scraper.sources.news_rss import NewsRSSScraper
from gcn_scraper.sources.wikipedia import WikipediaLangScraper


class _JsonResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _TextResp:
    def __init__(self, text):
        self.text = text

    def json(self):
        raise ValueError("no json")


# --- HAL (M5, M6) ---

def _hal_docs(n, start_id=0):
    return [
        {
            "docid": str(start_id + i),
            "uri_s": f"https://hal.science/hal-{start_id + i:08d}",
            "language_s": ["fr"],
            "label_s": [f"Titre {start_id + i}"],
            "abstract_s": [f"Résumé détaillé numéro {start_id + i} " * 10],
        }
        for i in range(n)
    ]


def test_hal_paginates_beyond_100(monkeypatch):
    calls = []

    def fake_retry(session, url, params, **kw):
        calls.append(dict(params))
        start, rows = params["start"], params["rows"]
        total = 130
        docs = _hal_docs(min(rows, max(0, total - start)), start)
        return _JsonResp({"response": {"docs": docs}}), session

    monkeypatch.setattr(hal_mod, "_retry_get", fake_retry)
    scraper = HALScraper("test-agent")
    scraper._delay = 0
    docs = scraper.search_papers("causalité", max_results=150)
    assert len(docs) == 130
    assert len(calls) == 2  # 100 + 30, pas une seule page
    assert calls[1]["start"] == 100 and calls[1]["rows"] == 50


def test_hal_scalar_fields_accepted(monkeypatch):
    monkeypatch.setattr(hal_mod, "_retry_get", lambda *a, **k: (None, a[0]))
    scraper = HALScraper("test-agent")
    doc = {
        "docid": "1",
        "uri_s": "https://hal.science/hal-1",
        "language_s": "en",  # scalaire, pas liste
        "label_s": "A Full Title Here",  # scalaire
        "abstract_s": "This abstract is long enough to pass the fifty character threshold easily.",
    }
    item = scraper._doc_to_item(doc, "q")
    assert item is not None
    assert item["lang"] == "en"  # pas "e" (1er caractère)
    assert item["title"] == "A Full Title Here"  # pas "A"


# --- arXiv (M7) ---

def _atom_xml(n):
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/2401.{i:05d}</id>"
        f"<title>Paper title number {i}</title>"
        f"<summary>{'Abstract content here. ' * 10}</summary></entry>"
        for i in range(n)
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_arxiv_paginates(monkeypatch):
    calls = []

    def fake_retry(session, url, params, **kw):
        calls.append(dict(params))
        if len(calls) == 1:
            return _TextResp(_atom_xml(params["max_results"])), session
        return _TextResp(_atom_xml(0)), session

    monkeypatch.setattr(arxiv_mod, "_retry_get", fake_retry)
    monkeypatch.setattr(ArXivScraper, "_fetch_abs_text", lambda self, url: None)
    scraper = ArXivScraper("test-agent")
    scraper._delay = 0  # pas de sleep 3s du YAML en test
    papers = scraper.search("causal inference", max_results=250)
    assert len(calls) == 2  # page pleine → 2e page demandée (M7)
    assert len(papers) == 200  # 200 + 0
    assert calls[1]["start"] == 200


# --- News (M9, M10) ---

RSS_FULL = """<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel>
<item><title>Short</title>
<description>Court.</description>
<content:encoded><![CDATA[<p>{full}</p>]]></content:encoded>
</item>
<item><title>{long_title}</title>
<description>{long_desc}</description>
</item>
</channel></rss>""".format(
    full="Contenu complet de l'article avec beaucoup de détails importants. " * 5,
    long_title="Un titre assez long pour dépasser le seuil de cinquante caractères requis",
    long_desc="Description longue de repli qui dépasse elle aussi le seuil minimal requis ici.",
)

ATOM_FULL = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Titre article atom assez long pour passer le seuil minimal requis</title>
<link rel="self" href="http://flux.example.com/self"/>
<link rel="alternate" href="http://example.com/article-1"/>
<summary>Résumé court.</summary>
<content type="html"><![CDATA[<p>{full}</p>]]></content>
</entry></feed>""".format(
    full="Texte intégral atom avec tous les détails nécessaires à l'article. " * 5
)


def test_news_prefers_full_content_and_real_urls():
    scraper = NewsRSSScraper.__new__(NewsRSSScraper)  # pas de session réseau
    rss_items = scraper._parse_feed(RSS_FULL, "fr", "test", "http://flux.example.com/rss")
    assert len(rss_items) == 2
    assert "Contenu complet" in rss_items[0]["text"]  # content:encoded > description
    assert rss_items[1]["url"] == "http://flux.example.com/rss#item-1"  # pas d'URL dégénérée partagée
    atom_items = scraper._parse_feed(ATOM_FULL, "fr", "test", "http://flux.example.com/atom")
    assert len(atom_items) == 1
    assert "Texte intégral" in atom_items[0]["text"]  # <content> > summary
    assert atom_items[0]["url"] == "http://example.com/article-1"  # alternate, pas self


def test_news_failures_are_per_language(monkeypatch):
    monkeypatch.setattr(
        news_mod, "_get_rss_sources",
        lambda: {"fr": [(f"fr{i}", f"http://fr{i}.example.com") for i in range(6)],
                 "en": [("en0", "http://en0.example.com")]},
    )

    def fake_feed(self, name, url, lang, tracker=None):
        if lang == "fr":
            return []
        return [{"text": "x" * 60, "url": url, "source": "news:en0", "lang": "en"}]

    monkeypatch.setattr(NewsRSSScraper, "scrape_feed", fake_feed)
    scraper = NewsRSSScraper.__new__(NewsRSSScraper)
    scraper._delay = 0
    results = scraper.scrape(langs=["fr", "en"])
    assert len(results) == 1  # 6 échecs FR n'annulent plus EN (M10)


# --- GitHub (M13) ---

def test_github_search_paginates(monkeypatch):
    calls = []

    def fake_retry(session, url, params, **kw):
        calls.append(dict(params))
        n = params["per_page"] if params["page"] == 1 else 10
        return _JsonResp({"items": [{"url": f"u{i}"} for i in range(n)]}), session

    monkeypatch.setattr(gh_mod, "retry_get", fake_retry)
    scraper = GitHubCodeScraper("test-agent")
    items = scraper.search_code("q", "python", max_results=120)
    assert len(items) == 110  # 100 + 10
    assert [c["page"] for c in calls] == [1, 2]


# --- Wikipedia srlimit (M19) ---

def test_wikipedia_srlimit_recalculated(monkeypatch):
    seen = []

    def fake_retry(session, url, params, **kw):
        seen.append(dict(params))
        hits = [{"title": f"Article {seen.__len__()}_{i}"} for i in range(50)]
        payload = {"query": {"search": hits}, "continue": {"sroffset": 50}}
        return _JsonResp(payload), session

    monkeypatch.setattr(wiki_mod, "retry_get", fake_retry)
    scraper = WikipediaLangScraper("fr", "test-agent")
    scraper._delay = 0
    titles = scraper.search_articles_paginated("science", max_total=60)
    assert len(titles) == 60
    assert seen[0]["srlimit"] == 50
    assert seen[1]["srlimit"] == 10  # reliquat recalculé, pas sur-fetch (M19)
    assert seen[1]["sroffset"] == 50  # offset officiel API
