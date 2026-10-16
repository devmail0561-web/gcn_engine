#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Chaînage v5 au volume : jsonl brut -> propose/quar/litig (réutilise build_lot).

Usage :
    python3 scripts/build_v5.py <input.jsonl> <tag> <registre> <lang> [--strict-auto]
    -> gcn-datasets/DATA/supervision/v5/<registre>/<tag>_propose.json|quar|litig

Déduplique contre v4 (textes annotés + quarantaines).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "gcn-tools" / "gcn-annotate" / "src"))
import build_lot as B  # noqa: E402
from gcn_annotate.balanced_auto import annotate, tokenize  # noqa: E402
from gcn_annotate.normalize import normalize_annotation  # noqa: E402

V5 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v5"
V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"


def load_v4_texts():
    done = set()
    for f in V4.glob("annotated/lot*.json"):
        try:
            for s in json.load(open(f))["document"]["sentences"]:
                done.add(s["text"].strip().lower())
        except Exception:  # noqa: BLE001
            pass
    for f in V4.glob("annotated/quarantaine_*.json"):
        try:
            for q in json.load(open(f)):
                if isinstance(q, dict) and "text" in q:
                    done.add(q["text"].strip().lower())
        except Exception:  # noqa: BLE001
            pass
    return done


def main() -> None:
    src, tag, registre, lang = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    strict_auto = len(sys.argv) > 5 and sys.argv[5] == "--strict-auto"
    seen = load_v4_texts()
    print(f"v4 exclus: {len(seen)}", flush=True)
    lot, quar, litig = [], [], []
    n_in = 0
    for line in open(src, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        text = (r.get("text") or "").strip()
        n_in += 1
        if len(text) < 40 or text.strip().lower() in seen:
            continue
        seen.add(text.strip().lower())
        sid = f"{tag}-{n_in:06d}"
        try:
            toks = B.lattice(text, lang)
        except Exception as e:  # noqa: BLE001
            quar.append({"id": sid, "text": text, "motif": f"lattice: {e}"[:200], "tombstone": True})
            continue
        p = annotate(text, lang)
        if not p or not p.get("edges"):
            quar.append({"id": sid, "text": text, "motif": "sans proposition", "tombstone": True})
            continue
        Psp = [(m.start(), m.end()) for m in B.TOK.finditer(text)]
        Lsp = B.char_spans([t["form"] for t in toks], text)
        if all(s is None for s in Lsp):
            quar.append({"id": sid, "text": text, "motif": "formes non localisables", "tombstone": True})
            continue
        Lsp = [s if s else (-1, -1) for s in Lsp]
        pn, pe = p["nodes"], p["edges"][0]
        _stripped = B._strip_lead(text.strip())
        _prefix = len(tokenize(text)) - len(tokenize(_stripped))
        flags = [f"recalage strip +{_prefix}"] if _prefix else []
        ok, nrecs = True, []
        for i, nd in enumerate(pn):
            a, b = nd["token_span"][0] + _prefix, nd["token_span"][1] + _prefix
            if a < 1 or b > len(Psp) or a > b:
                ok = False
                flags.append(f"noeud {nd['id']} span hors bornes")
                break
            m = B.to_lattice_span(Psp, Lsp, a, b)
            if not m:
                ok = False
                flags.append(f"noeud {nd['id']} inalignable")
                break
            nrecs.append({"id": f"n{i + 1}", "type": nd["type"], "label": nd["label"],
                          "token_span": list(m), "scope": nd.get("scope", "specific"),
                          "temporal_index": nd.get("temporal_index", i),
                          "origin": nd.get("origin", "explicit"),
                          "span_source": "auto"})
        if not ok:
            quar.append({"id": sid, "text": text, "motif": "; ".join(flags), "tombstone": True})
            continue
        if strict_auto and any(n["token_span"][1] - n["token_span"][0] + 1 <= 1 for n in nrecs):
            quar.append({"id": sid, "text": text, "motif": "noeud degenere", "tombstone": True})
            continue
        mid = pe.get("attributes", {}).get("marker_token")
        if mid:
            mid += _prefix
        Lm = None
        if mid and 1 <= mid <= len(Psp):
            cs, ce = Psp[mid - 1]
            ids = [i + 1 for i, s in enumerate(Lsp) if s[0] < ce and s[1] > cs]
            Lm = ids[0] if ids else None
        if Lm:
            marker, explicit = Lm, True
            conf = float(pe.get("attributes", {}).get("confidence", 0.9))
        else:
            marker, explicit, conf = None, False, 0.6
            flags.append("marqueur non mappe")
        if strict_auto and (conf < 0.8 or not explicit):
            quar.append({"id": sid, "text": text, "motif": f"conf={conf} explicit={explicit}", "tombstone": True})
            continue
        e0 = {"sources": [pe["source"].replace("n001", "n1").replace("n002", "n2")],
              "target": pe["target"].replace("n001", "n1").replace("n002", "n2"),
              "relation": pe["relation"], "explicit": explicit,
              "confidence": conf, "third": None}
        if marker:
            e0["marker_token"] = marker
        rec = {"id": sid, "text": text, "tokens": toks, "sentence_type": "declarative",
               "registre": registre, "cir": {"nodes": nrecs, "edges": [e0]},
               "_methode": "silver-auto"}
        if not explicit or conf < 0.9:
            flags.append(f"conf={conf} explicit={explicit}")
        for n in nrecs:
            if n["token_span"][1] - n["token_span"][0] + 1 > 15:
                flags.append(f"span {n['id']} >15tok")
        if flags:
            litig.append({"id": sid, "rel": pe["relation"], "flags": flags})
        lot.append(rec)
    doc = {"schema_version": "4.0", "document": {"id": f"{tag}-propose", "sentences": lot}}
    normalize_annotation(doc, span_base="1")
    for s in doc["document"]["sentences"]:
        for e in s["cir"]["edges"]:
            for k in ("polarity", "voice", "modality", "has_restriction", "condition_prominence"):
                e.pop(k, None)
    out = V5 / registre
    out.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out / f"{tag}_propose.json", "w"), ensure_ascii=False)
    json.dump(quar, open(out / f"{tag}_quar.json", "w"), ensure_ascii=False)
    json.dump(litig, open(out / f"{tag}_litig.json", "w"), ensure_ascii=False)
    print(f"IN: {n_in} | LOT: {len(lot)} | QUAR: {len(quar)} | LITIG: {len(litig)}", flush=True)
    print("REL:", dict(Counter(e["relation"] for s in lot for e in s["cir"]["edges"])), flush=True)


main()
