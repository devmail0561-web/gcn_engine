"""C1 : formatversion=2 renvoie query.pages en LISTE — l'extraction doit marcher."""
import gcn_scraper.sources.wikipedia as wiki_mod
from gcn_scraper.sources.wikipedia import WikipediaLangScraper


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_retry_get(session, url, params, **kwargs):
    prop = params.get("prop")
    if prop == "extracts":
        payload = {"query": {"pages": [{"pageid": 1, "extract": "x" * 200}]}}
    elif prop == "categories":
        payload = {"query": {"pages": [{"pageid": 1, "categories": [{"title": "Catégorie:Science"}]}]}}
    else:
        payload = {"query": {"search": []}}
    return _FakeResp(payload), session


def test_get_article_text_format_v2_list(monkeypatch):
    monkeypatch.setattr(wiki_mod, "retry_get", _fake_retry_get)
    scraper = WikipediaLangScraper("fr", "test-agent")
    assert scraper.get_article_text("Science") == "x" * 200


def test_get_article_categories_format_v2_list(monkeypatch):
    monkeypatch.setattr(wiki_mod, "retry_get", _fake_retry_get)
    scraper = WikipediaLangScraper("fr", "test-agent")
    assert scraper.get_article_categories("Science") == ["Science"]
