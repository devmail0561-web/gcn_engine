# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Tests pdf_latex — 100 % offline (PDF synthétiques pymupdf, pas d'Exo7)."""
import json
import pymupdf
import pytest
from gcn_scraper.sources.pdf_latex import (
    FORMULE, agglutiner, extract_file, extract_sections, opacify_spans)


def _pdf_2_sections(path):
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Logique. Si P alors Q, donc R. La negation echange les quantificateurs.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Ensembles. Soit E un ensemble fini de cardinal n bien defini.")
    doc.set_toc([[1, "Logique", 1], [1, "Ensembles", 2]])
    doc.save(path)
    doc.close()


def test_opacify_par_fonte():
    spans = [{"text": "Soit ", "font": "CharterBT-Roman"},
             {"text": "x", "font": "MathDesign-CH-Regular-Ma"},
             {"text": " reel.", "font": "CharterBT-Roman"}]
    assert opacify_spans(spans).strip() == f"Soit {FORMULE} reel."


def test_agglutiner_variables_isolees():
    t = f"« {FORMULE} x {FORMULE} f {FORMULE} »"
    assert agglutiner(t) == f"« {FORMULE} »"


def test_agglutiner_ne_touche_pas_le_texte():
    t = "Pour la négation, on change tout en il existe."
    assert agglutiner(t) == t


def test_sections_depuis_toc(tmp_path):
    pdf = tmp_path / "mini.pdf"
    _pdf_2_sections(pdf)
    recs = extract_sections(pdf, url="https://exo7.test/livre.pdf",
                            source="exo7", lang="fr", licence="CC-BY-NC-SA")
    assert [r["section"] for r in recs] == ["Logique", "Ensembles"]
    assert all(r["licence"] == "CC-BY-NC-SA" and r["lang"] == "fr" for r in recs)
    assert "Si P alors Q" in recs[0]["text"]


def test_sans_toc_repli_doc_entier(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Un seul bloc de texte assez long " * 10)
    pdf = tmp_path / "notoc.pdf"
    doc.save(pdf)
    doc.close()
    recs = extract_sections(pdf, url="u", source="s", lang="fr", licence="L")
    assert len(recs) == 1


def test_extract_file_jsonl(tmp_path):
    pdf = tmp_path / "mini.pdf"
    _pdf_2_sections(pdf)
    out = tmp_path / "sections.jsonl"
    stats = extract_file(pdf, out, url="u", source="exo7", lang="fr", licence="CC-BY-NC-SA")
    assert stats["sections"] == 2
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["source"] == "exo7"
