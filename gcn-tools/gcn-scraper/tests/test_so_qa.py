# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests so_qa — HTTP stubbé (aucun appel réseau)."""
from gcn_scraper.sources import so_qa
from gcn_scraper.sources.so_qa import StackOverflowScraper, nettoie


def test_nettoie_opacifie_code():
    h = "<p>Use X because Y</p><pre>def f(): pass</pre><p>done &amp; ok</p>"
    t = nettoie(h)
    assert "[CODE]" in t and "def f" not in t and "&amp;" not in t
    assert "Use X because Y" in t and "done & ok" in t


Q = {"question_id": 1, "title": "Why does X fail under load conditions here?",
     "body": "<p>X fails because Y happens here and there under heavy concurrent load situations</p>",
     "link": "https://stackoverflow.com/q/1",
     "content_license": "CC-BY-SA-4.0",
     "owner": {"display_name": "Ada"}}
A = {"question_id": 1, "body": "<p>Use Z instead because it avoids the issue widely in all production environments</p>",
     "owner": {"display_name": "Bob"}}


class _StubSO(StackOverflowScraper):
    def __init__(self, *a, **k):
        pass

    def search(self, tag, query, pagesize=50, pages=1):
        assert tag == "python"
        return [dict(Q)]

    def top_answers(self, qids):
        assert qids == [1]
        return {1: dict(A)}


def test_scrape_notice_attribution():
    from gcn_scraper.sources.so_qa import QUERIES
    s = _StubSO()
    recs = s.scrape(languages=["python"])
    assert len(recs) == len(QUERIES)  # stub répond à chaque requête
    r = recs[0]
    assert r["lang"] == "en" and r["source"] == "stackoverflow"
    assert r["url"] == "https://stackoverflow.com/q/1"
    assert r["auteur"] == "Ada" and r["licence"] == "CC-BY-SA-4.0"
    assert "Why does X fail" in r["text"] and "Use Z instead" in r["text"]


def test_reponse_courte_rejetee(monkeypatch):
    s = _StubSO()
    monkeypatch.setattr(_StubSO, "top_answers",
                        lambda self, qids: {1: {"question_id": 1, "body": "ok",
                                                "owner": {}}})
    assert s.scrape(languages=["python"]) == []


def test_tag_inconnu_skip(capsys):
    s = _StubSO()
    assert s.scrape(languages=["cobol"]) == []
    assert "pas de tag" in capsys.readouterr().out
