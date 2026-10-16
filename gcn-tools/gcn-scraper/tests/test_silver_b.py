# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests silver_b — records, quarantaine, poids 0.0, jamais d'UD legacy."""
import json
import pytest
from gcn_scraper import silver_b
from gcn_scraper.silver_b import SILVER_WEIGHT, silver_annotate_texts, silver_annotate_file


def test_weight_zero_and_format():
    res = silver_annotate_texts([("s1", "Il pleut donc je reste.")], "fr")
    assert res["stats"]["silver"] == 1
    rec = res["records"][0]
    assert rec["weight"] == SILVER_WEIGHT == 0.0
    assert rec["format_refacto"] == 20261007
    assert rec["source"] in ("silver-B", "fallback")
    assert rec["outil"] == "moteur-B"
    assert len(rec["text_sha256"]) == 64


def test_fallback_path_deterministic():
    res = silver_annotate_texts([("s1", "Il pleut donc je reste.")], "fr", force_fallback=True)
    assert res["records"][0]["source"] == "fallback"
    assert res["stats"]["sources"] == {"fallback": 1}


def test_live_silver_b_fr():
    res = silver_annotate_texts([("s1", "Il pleut donc je reste.")], "fr")
    assert res["records"][0]["source"] == "silver-B"


def test_quarantine_sans_connecteur():
    res = silver_annotate_texts([("s1", "Le chat dort sur le tapis.")], "fr")
    assert res["stats"]["quarantaine"] == 1
    assert res["records"] == []
    assert res["quarantine"][0]["motif"] == "causal_none"


def test_lang_hors_budget_leve():
    with pytest.raises(ValueError):
        silver_annotate_texts([("s1", "hello")], "xx")


def test_routage_par_phrase_mixte():
    res = silver_annotate_texts(
        [("s1", "Il pleut donc je reste.", "fr"),
         ("s2", "It rains therefore I stay.", "en")], "fr")
    assert res["stats"]["silver"] == 2
    langs = {r["sid"]: r["lang"] for r in res["records"]}
    assert langs == {"s1": "fr", "s2": "en"}


def test_file_jsonl_end_to_end(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": "a", "text": "Il pleut donc je reste.", "lang": "fr"},
                              ensure_ascii=False) + "\n"
                   + json.dumps({"id": "b", "text": "Le chat dort.", "lang": "fr"},
                                ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "silver.jsonl"
    man = silver_annotate_file(src, out, "fr")
    assert man["stats"]["silver"] == 1 and man["stats"]["quarantaine"] == 1
    assert (tmp_path / "silver_quarantaine.jsonl").exists()
    assert (tmp_path / "silver_manifest.json").exists()


def test_no_legacy_ud_import():
    import sys
    assert "ud_annotator" not in sys.modules
    assert "stanza" not in sys.modules
