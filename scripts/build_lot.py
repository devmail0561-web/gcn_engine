#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Construit un lot proposé : lattice + alignement char + normalize D5.

Chaînage validé lots 08-10 : tokens via `gcn analyze` (référence),
spans balanced_auto (whitespace) → lattice par recouvrement char-offsets
(apostrophes normalisées 1:1), recalage `_strip_lead`, normalize
(span_base 1), schéma v4 strict (clés surnuméraires retirées en aval).

Usage :
    python3 scripts/build_lot.py /tmp/opencode/lot11_sel.json lot11
    → /tmp/opencode/lot11_propose.json|quar|litig
"""
import json
import subprocess
import sys
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gcn-tools" / "gcn-annotate" / "src"))
from gcn_annotate.balanced_auto import annotate, TOK, tokenize, _strip_lead  # noqa: E402
from gcn_annotate.normalize import normalize_annotation  # noqa: E402

GCN = str(ROOT / "gcn-core" / "target" / "debug" / "gcn")
V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"


def load_pool():
    pool = {}
    for rel, reg in (("candidates/candidates.json", "encyclo"),
                     ("candidates/candidates_fresh.json", "science"),
                     ("candidates/candidates_rares.json", None),
                     ("candidates/candidates_rares2.json", None),
                     ("candidates/candidates_divers.json", None)):
        p = V4 / rel
        if p.exists():
            for s in json.load(open(p))["document"]["sentences"]:
                pool.setdefault(s["id"], (s["text"], s.get("registre", reg)))
    return pool


def lattice(text, lang="fr"):
    cmd = [GCN, "analyze-en" if lang == "en" else "analyze", "--", text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    toks = json.loads(r.stdout)["tokens"]
    return [{"id": t["index"], "form": t["form"], "lemma": t.get("lemma", ""),
             "pos": t.get("pos", ""), "dep_rel": t.get("dep_rel", ""),
             "dep_head": t.get("dep_head", -1), "morph": {}} for t in toks]


_QNORM = str.maketrans({chr(c): "'" for c in (0x2019, 0x2018, 0x0060, 0x00B4)})


def char_spans(forms, text):
    """Localise chaque forme séquentiellement (apostrophes normalisées 1:1)."""
    tn = text.translate(_QNORM)
    out, cur = [], 0
    for f in forms:
        fn = f.translate(_QNORM)
        i = tn.find(fn, cur)
        if i < 0:
            i = tn.lower().find(fn.lower(), cur)
        if i < 0:
            out.append(None)
        else:
            out.append((i, i + len(fn)))
            cur = i + len(fn)
    return out


def to_lattice_span(Pspans, Lspans, a, b):
    """Span P 1-based [a,b] -> [min,max] ids lattice par recouvrement char."""
    cs, ce = Pspans[a - 1][0], Pspans[b - 1][1]
    ids = [i + 1 for i, s in enumerate(Lspans) if s and s[0] < ce and s[1] > cs]
    return (min(ids), max(ids)) if ids else None


def main() -> None:
    sel_path, tag = sys.argv[1], sys.argv[2]
    strict_auto = len(sys.argv) > 3 and sys.argv[3] == "--strict-auto"
    cand = load_pool()
    sel = json.load(open(sel_path))
    lot, quar, litig = [], [], []
    for sid, _rel_exp, lang in sel:
        text, registre = cand[sid]
        try:
            toks = lattice(text, lang)
        except Exception as e:  # noqa: BLE001
            quar.append({"id": sid, "text": text,
                         "motif": f"lattice indisponible: {e}", "tombstone": True})
            continue
        p = annotate(text, lang)
        if not p or not p.get("edges"):
            quar.append({"id": sid, "text": text,
                         "motif": "sans proposition au rebuild", "tombstone": True})
            continue
        Psp = [(m.start(), m.end()) for m in TOK.finditer(text)]
        Lsp = char_spans([t["form"] for t in toks], text)
        if all(s is None for s in Lsp):
            quar.append({"id": sid, "text": text,
                         "motif": "formes lattice non localisables", "tombstone": True})
            continue
        Lsp = [s if s else (-1, -1) for s in Lsp]
        pn, pe = p["nodes"], p["edges"][0]
        # _strip_lead : spans relatives au texte amputé → recalage +préfixe.
        _stripped = _strip_lead(text.strip())
        _prefix = len(tokenize(text)) - len(tokenize(_stripped))
        flags = [f"recalage strip +{_prefix}"] if _prefix else []
        ok, nrecs = True, []
        for i, nd in enumerate(pn):
            a, b = nd["token_span"][0] + _prefix, nd["token_span"][1] + _prefix
            if a < 1 or b > len(Psp) or a > b:
                ok = False
                flags.append(f"noeud {nd['id']} span hors bornes")
                break
            m = to_lattice_span(Psp, Lsp, a, b)
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
            quar.append({"id": sid, "text": text, "motif": "; ".join(flags),
                         "tombstone": True})
            continue
        if strict_auto and any(
                n["token_span"][1] - n["token_span"][0] + 1 <= 1 for n in nrecs):
            quar.append({"id": sid, "text": text,
                         "motif": "auto-quarantaine vague (noeud dégénéré)",
                         "tombstone": True})
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
            flags.append("marqueur non mappé → explicit:false")
        if strict_auto and (conf < 0.8 or not explicit):
            quar.append({"id": sid, "text": text,
                         "motif": f"auto-quarantaine vague (conf={conf} explicit={explicit})",
                         "tombstone": True})
            continue
        e0 = {"sources": [pe["source"].replace("n001", "n1").replace("n002", "n2")],
              "target": pe["target"].replace("n001", "n1").replace("n002", "n2"),
              "relation": pe["relation"], "explicit": explicit,
              "confidence": conf, "third": None}
        if marker:
            e0["marker_token"] = marker
        st, intent = "declarative", None
        if text.rstrip().endswith("?"):
            st = "interrogative"
            intent = "chain" if re.search(r"comment|pourquoi|how", text, re.I) else "explain"
        elif text.rstrip().endswith("!"):
            st = "exclamative"
        rec = {"id": sid, "text": text, "tokens": toks, "sentence_type": st,
               "registre": registre or "encyclo",
               "cir": {"nodes": nrecs, "edges": [e0]}}
        if intent:
            rec["intent"] = intent
        if not explicit or conf < 0.9:
            flags.append(f"conf={conf} explicit={explicit}")
        for n in nrecs:
            if n["token_span"][1] - n["token_span"][0] + 1 > 15:
                flags.append(f"span {n['id']} >15tok {n['token_span']}")
        if flags:
            litig.append({"id": sid, "rel": pe["relation"], "flags": flags})
        lot.append(rec)

    doc = {"schema_version": "4.0",
           "document": {"id": f"{tag}-propose", "sentences": lot}}
    normalize_annotation(doc, span_base="1")
    for s in doc["document"]["sentences"]:
        for e in s["cir"]["edges"]:
            for k in ("polarity", "voice", "modality", "has_restriction",
                      "condition_prominence"):
                e.pop(k, None)
    json.dump(doc, open(f"/tmp/opencode/{tag}_propose.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump(quar, open(f"/tmp/opencode/{tag}_quar.json", "w"), ensure_ascii=False, indent=1)
    json.dump(litig, open(f"/tmp/opencode/{tag}_litig.json", "w"),
              ensure_ascii=False, indent=1)
    print("LOT:", len(lot), "| QUAR:", len(quar), "| LITIG:", len(litig))
    print("REL:", dict(Counter(e["relation"] for s in lot for e in s["cir"]["edges"])))


if __name__ == "__main__":
    main()
