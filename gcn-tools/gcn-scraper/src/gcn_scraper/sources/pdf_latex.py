# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Extraction PDF maths/physique par sections — module C2 (phase 0, P0-6).

Unités = SECTIONS (énoncé + preuve gardés ensemble), jamais de phrases isolées.
Formules → blocs opaques `[FORMULE]` (détection par fontes maths + agglutination
des variables isolées). Sortie : `{text, lang, source, url, section, licence,
quality}` — sans split phrastique (le pipeline C2 utilisera `skip_split`).

Licences : le module n'a PAS d'allowlist — l'appelant fournit `licence` par
source (audit AVANT extraction : Exo7 CC-BY-NC-SA, Sésamath GNU-FDL/CC-BY-SA,
OpenStax CC-BY, LibreTexts/MIT-OCW à auditer page par page). Exclus : Khan
Academy, Annabac/Hatier/manuels sous droits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FORMULE = "[FORMULE]"

#: Préfixes de fontes mathématiques (pymupdf) → opacifiées.
MATH_FONT_PREFIXES = ("MathDesign", "MSAM", "MSBM", "CMR", "CMMI", "CMSY",
                      "CMEX", "DSHOFR", "ESSTIX", "STIX", "Latin-Modern-Math",
                      "Asana-Math")

#: Variables isolées (1-2 lettres restées en fonte texte entre formules).
_VAR_RUN = re.compile(
    r"\[FORMULE\](?:\s+[A-Za-zα-ωΑ-Ω]\w{0,2})?(?:\s*\[FORMULE\](?:\s+[A-Za-zα-ωΑ-Ω]\w{0,2})?)+")


def _is_math_font(font: str) -> bool:
    return font.startswith(MATH_FONT_PREFIXES)


def opacify_spans(spans: list[dict]) -> str:
    """Spans pymupdf `{'text', 'font'}` → texte, fontes maths → `[FORMULE]`."""
    parts = []
    for s in spans:
        parts.append(FORMULE if _is_math_font(s.get("font", "")) else s.get("text", ""))
        parts.append(" ")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def agglutiner(text: str) -> str:
    """Fusionne `[FORMULE] x [FORMULE]` → `[FORMULE]` (variables isolées)."""
    return _VAR_RUN.sub(FORMULE, text)


def page_text(page) -> str:
    """Texte d'une page pymupdf, formules opacifiées + agglutinées."""
    d = page.get_text("dict")
    chunks = []
    for b in d.get("blocks", []):
        for line in b.get("lines", []):
            chunks.append(opacify_spans(line.get("spans", [])))
    text = re.sub(r"\s+", " ", "".join(chunks)).strip()
    return agglutiner(text)


def extract_sections(pdf_path: Path, *, url: str, source: str, lang: str,
                     licence: str) -> list[dict]:
    """PDF → une notice par section TOC (repli : doc entier si pas de TOC)."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()  # [niveau, titre, page(1-based)]
    if not toc:
        full = " ".join(page_text(doc[i]) for i in range(doc.page_count))
        return [{"text": full, "lang": lang, "source": source, "url": str(url),
                 "section": doc.metadata.get("title") or Path(pdf_path).stem,
                 "licence": licence, "quality": 1.0 if len(full) > 200 else 0.5}]
    # Plages : page de la section → page suivante (même niveau ou supérieur).
    bounds = []
    for i, (lvl, titre, pg) in enumerate(toc):
        nxt = next((q[2] for q in toc[i + 1:] if q[0] <= lvl), doc.page_count + 1)
        bounds.append((titre, pg - 1, nxt - 1))
    records = []
    for titre, start, end in bounds:
        txt = " ".join(page_text(doc[p]) for p in range(start, min(end, doc.page_count)))
        if len(txt) < 50:
            continue
        records.append({"text": txt, "lang": lang, "source": source,
                        "url": str(url), "section": titre,
                        "licence": licence,
                        "quality": 1.0 if len(txt) > 200 else 0.5})
    return records


def extract_file(pdf: Path, out: Path, *, url: str, source: str,
                 lang: str, licence: str) -> dict:
    """Un PDF → JSONL sections. Retourne {"sections", "formules"}."""
    recs = extract_sections(pdf, url=url, source=source, lang=lang, licence=licence)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_form = sum(r["text"].count(FORMULE) for r in recs)
    print(f"PDF : {len(recs)} sections, {n_form} blocs [FORMULE] → {out}")
    return {"sections": len(recs), "formules": n_form}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PDF maths/phys → sections JSONL.")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--source", required=True, help="ex. exo7 (licence auditée)")
    ap.add_argument("--lang", default="fr")
    ap.add_argument("--licence", required=True, help="ex. CC-BY-NC-SA (audit préalable)")
    a = ap.parse_args()
    extract_file(a.pdf, a.output, url=a.url, source=a.source,
                 lang=a.lang, licence=a.licence)
