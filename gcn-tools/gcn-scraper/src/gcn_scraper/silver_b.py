# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Annotation silver par le moteur B — tranche T2/T3 (phase 0, P0-5).

Chaîne : phrases brutes (JSONL `sentences_*.jsonl` ou doc-JSON
`{document: {sentences}}`) → filtre connecteur (quarantaine `causal_none`,
jamais suppression) → `moteur B` (spaCy + fallback regex, T1) →
enregistrements silver JSONL (`source=silver-B|fallback`, `weight=0.0`).

Interdits : ancien moteur UD (`ud_annotator`, stanza) — le silver vient de B
uniquement ; aucun appel réseau/LLM (100 % local) ; aucune écriture dans les
fichiers sources (sortie + manifest séparés).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SILVER_WEIGHT = 0.0


def _sha256(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def iter_input_texts(path: Path):
    """Yield (sid, text, lang_hint) depuis JSONL brut ou doc-JSON."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if isinstance(o, dict) and o.get("text"):
                    yield o.get("id") or f"l{i:06d}", o["text"], o.get("lang")
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs = doc.get("document", doc) if isinstance(doc, dict) else doc
        sents = docs.get("sentences", []) if isinstance(docs, dict) else []
        for i, s in enumerate(sents):
            if isinstance(s, dict) and s.get("text"):
                yield s.get("id") or f"s{i:06d}", s["text"], s.get("lang")


def silver_annotate_texts(items, lang: str, *, force_fallback: bool = False) -> dict:
    """Annote `[(sid, text)]` ou `[(sid, text, lang|None)]` via le moteur B.

    `lang` = langue budgétée fr/en/es/de/pt/it par défaut (validée par B,
    ValueError sinon) ; si un item porte sa propre langue budgétée, elle
    prime (fichiers mixtes : routage par phrase, pas de forçage).
    Chaque record porte `weight=SILVER_WEIGHT` (0.0 : le silver n'apprend rien
    tant que sa qualité n'est pas prouvée, R9) et `format_refacto` s0.
    """
    from gcn_python.layer0 import engine_b
    from gcn_python.layer0.fallback_regex import SUPPORTED_LANGS

    records, quarantine = [], []
    sources: dict[str, int] = {}
    for item in items:
        sid, text = item[0], item[1]
        want = item[2] if len(item) > 2 and item[2] in SUPPORTED_LANGS else lang
        out = engine_b.analyze(text, want, force_fallback=force_fallback)
        sources[out.source] = sources.get(out.source, 0) + 1
        if not out.connecteurs:
            quarantine.append({"sid": sid, "text": text[:200], "motif": "causal_none"})
            continue
        records.append({
            "text_sha256": _sha256(text), "lang": want, "sid": sid,
            "source": out.source, "weight": SILVER_WEIGHT,
            "clauses": list(out.clauses),
            "connecteurs": list(out.connecteurs),
            "published_at": None, "outil": engine_b.OUTIL_B,
            "format_refacto": engine_b.FORMAT_REFACTO_S0,
        })
    return {"records": records, "quarantine": quarantine,
            "stats": {"n": len(items), "silver": len(records),
                      "quarantaine": len(quarantine), "sources": sources}}


def silver_annotate_file(input_path: Path, output_path: Path, lang: str,
                         *, force_fallback: bool = False) -> dict:
    """Fichier → silver JSONL + quarantaine + manifest. Retourne le manifest."""
    input_path = Path(input_path)
    items = [(sid, text, lh) for sid, text, lh in iter_input_texts(input_path)]
    res = silver_annotate_texts(items, lang, force_fallback=force_fallback)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in res["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    qpath = output_path.with_name(output_path.stem + "_quarantaine.jsonl")
    with open(qpath, "w", encoding="utf-8") as f:
        for q in res["quarantine"]:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifest = {
        "id": "silver-b", "outil": "moteur-B",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(input_path), "input_sha256": _sha256(input_path.read_bytes().decode("utf-8", "replace")),
        "lang": lang, "weight": SILVER_WEIGHT, "stats": res["stats"],
        "outputs": {p.name: _sha256(p.read_bytes().decode("utf-8", "replace"))
                    for p in (output_path, qpath)},
    }
    mpath = output_path.with_name(output_path.stem + "_manifest.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"silver : {res['stats']['silver']}/{res['stats']['n']} "
          f"(quarantaine {res['stats']['quarantaine']}, {res['stats']['sources']})")
    print(f"Écrit : {output_path} + {qpath.name} + {mpath.name}")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Silver-B local : brut → silver JSONL.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang", required=True, help="fr/en/es/de/pt/it")
    ap.add_argument("--force-fallback", action="store_true")
    a = ap.parse_args()
    silver_annotate_file(a.input, a.output, a.lang, force_fallback=a.force_fallback)
