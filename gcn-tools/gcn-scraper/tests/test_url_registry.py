# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests URLRegistry — persistance, is_known, idempotence, context manager."""
import pathlib
import pytest
from gcn_scraper.url_registry import URLRegistry


@pytest.fixture
def db(tmp_path) -> URLRegistry:
    r = URLRegistry(tmp_path / "test.db")
    yield r
    r.close()


def test_is_known_before_register(db):
    assert not db.is_known("https://example.com/a")


def test_register_makes_known(db):
    cid = db.new_campaign("/out", "20260919_120000")
    db.register("https://example.com/a", cid, "wikipedia_fr", "wiki_fr.jsonl")
    assert db.is_known("https://example.com/a")
    assert not db.is_known("https://example.com/b")


def test_register_idempotent(db):
    cid = db.new_campaign("/out", "20260919_120000")
    db.register("https://example.com/a", cid, "wikipedia_fr", "wiki_fr.jsonl")
    db.register("https://example.com/a", cid, "wikipedia_fr", "wiki_fr.jsonl")
    assert db.stats()["urls"] == 1


def test_empty_url_ignored(db):
    cid = db.new_campaign("/out", "20260919_120000")
    db.register("", cid, "wikipedia_fr", "wiki_fr.jsonl")
    assert db.stats()["urls"] == 0
    assert not db.is_known("")


def test_persistence_across_instances(tmp_path):
    db_path = tmp_path / "reg.db"
    r1 = URLRegistry(db_path)
    cid = r1.new_campaign("/out", "ts1")
    r1.register("https://example.com/page1", cid, "hal", "hal.jsonl")
    r1.close()

    r2 = URLRegistry(db_path)
    assert r2.is_known("https://example.com/page1")
    assert not r2.is_known("https://example.com/page2")
    r2.close()


def test_get_all_urls(db):
    cid = db.new_campaign("/out", "ts1")
    db.register("https://example.com/a", cid, "src", "out.jsonl")
    db.register("https://example.com/b", cid, "src", "out.jsonl")
    urls = db.get_all_urls()
    assert urls == {"https://example.com/a", "https://example.com/b"}


def test_stats(db):
    s0 = db.stats()
    assert s0["campaigns"] == 0 and s0["urls"] == 0
    cid = db.new_campaign("/out", "ts1")
    db.register("https://example.com/x", cid, "s", "f.jsonl")
    s1 = db.stats()
    assert s1["campaigns"] == 1 and s1["urls"] == 1


def test_update_campaign_total(db):
    cid = db.new_campaign("/out", "ts1")
    db.update_campaign_total(cid, 42)
    row = db._conn.execute("SELECT total_scraped FROM campaigns WHERE id=?", (cid,)).fetchone()
    assert row[0] == 42


def test_context_manager(tmp_path):
    with URLRegistry(tmp_path / "cm.db") as r:
        cid = r.new_campaign("/out", "ts1")
        r.register("https://example.com/cm", cid, "s", "f.jsonl")
    # Après le with, la connexion est fermée — ré-ouvrir pour vérifier.
    r2 = URLRegistry(tmp_path / "cm.db")
    assert r2.is_known("https://example.com/cm")
    r2.close()


def test_cross_campaign_dedup(tmp_path):
    """Une URL scrapée en campagne 1 est connue en campagne 2."""
    db_path = tmp_path / "reg.db"
    r = URLRegistry(db_path)
    cid1 = r.new_campaign("/out/c1", "ts1")
    r.register("https://example.com/shared", cid1, "wiki", "c1.jsonl")
    r.close()

    r2 = URLRegistry(db_path)
    cid2 = r2.new_campaign("/out/c2", "ts2")
    assert r2.is_known("https://example.com/shared")
    assert not r2.is_known("https://example.com/new")
    r2.close()
