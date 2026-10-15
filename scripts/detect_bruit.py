#!/usr/bin/env python3
"""Detecteur de bruit dataset v4 (P0 plan-remediation-donnees.md). Lecture seule (stdlib).

Categories :
  junk certain  : latex (residus LaTeX), en_texte (EN avéré, >=4 mots)
  bruit graduel : label_sujet_hors_span, label_sans_parentheses, label_coquille,
                  span_gt15, span_hors_bornes, conf1, sans_marqueur, back_<relation>
Seuils (D2) : junk certain -> quarantaine immediate, sans seuil.
Bruit graduel -> quarantaine si >5 % des phrases du lot (coherent audit_spans_tok.py).

Sortie : <out> JSON {lots: {...}, totaux: {...}, quarantaine_proposee: [...]} + tableau console.
Exit 0 toujours (constat, pas gate) ; la revue humaine decide (zero suppression auto).
"""
import argparse
import glob
import json
import os
import re
import sys

LATEX_RE = re.compile(r"\\\\|displaystyle|\$[^$]{1,60}\$|\^[{}]|\{\\")
LABEL_RE = re.compile(r"^([^(]+)\(([^)]+)\)$")
# Verbes (partie avant parenthèse) invalides : égalité stricte, pas sous-chaîne
# (ex : "décréter" est valide et ne doit PAS matcher "éter").
COQUILLES = {"éter", "sociéter", "hypothesi", "décriver"}
EN_STOP = {"the", "and", "of", "to", "in", "is", "was", "were", "with", "for",
           "that", "this", "from", "have", "has", "been", "are", "which", "their"}
# Sortie P4 (arbitrage humain, pas quarantaine) :
P4_CATS = {"back_cause", "back_concession", "back_condition", "back_data_dependency",
           "back_enable", "back_motivation", "back_opposition", "back_sequence",
           "label_sujet_hors_span", "label_coquille", "conf1", "span_gt15"}
# Conventions de generation (info : a corriger avant promotion gold, pas quarantaine) :
CONV_CATS = {"label_sans_parentheses", "sans_marqueur"}


def detect_lang(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        toks = re.findall(r"[A-Za-z]+", text.lower())
        return "en" if sum(1 for t in toks if t in EN_STOP) >= 2 else "?"


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def audit_file(fp):
    d = json.load(open(fp, encoding="utf-8"))
    sents = d.get("document", {}).get("sentences", [])
    st = {"n": len(sents), "flags": []}
    for s in sents:
        sid = s.get("id", "?")
        t = s.get("text", "")
        toks = s.get("tokens", []) or []
        forms = [x.get("form", "") for x in toks]
        if LATEX_RE.search(t):
            st["flags"].append({"id": sid, "cat": "latex", "detail": t[:80]})
        if len(t.split()) >= 4 and detect_lang(t) == "en":
            st["flags"].append({"id": sid, "cat": "en_texte", "detail": t[:80]})
        cir = s.get("cir") or {}
        nodes = {n.get("id"): n for n in cir.get("nodes", [])}
        for nid, n in nodes.items():
            lab = n.get("label", "")
            span = n.get("token_span")
            if not span or span[1] <= span[0]:
                # Convention lots propose : span ponctuel [k,k] = token k seul.
                if span and span[1] == span[0] and 1 <= span[0] <= len(forms):
                    seg = norm(forms[span[0] - 1])
                    tok_len = 1
                else:
                    st["flags"].append({"id": sid, "cat": "span_hors_bornes",
                                        "detail": f"{nid} span={span}"})
                    continue
            # Unite du span : lots 01-07 = offsets chars, lots 08+ = indices tokens (1-based).
            elif span[1] <= len(forms):
                seg = norm(" ".join(forms[span[0] - 1:span[1]]))
                tok_len = span[1] - span[0]
            elif span[1] <= len(t):
                seg = norm(t[span[0]:span[1]])
                tok_len = len(seg.split())
            else:
                st["flags"].append({"id": sid, "cat": "span_hors_bornes",
                                    "detail": f"{nid} span={span} ntok={len(forms)} nchar={len(t)}"})
                continue
            if tok_len > 15:
                st["flags"].append({"id": sid, "cat": "span_gt15",
                                    "detail": f"{nid} ~{tok_len}tok"})
            m = LABEL_RE.match(lab)
            if m:
                verbe, sujet = m.group(1), m.group(2)
                if norm(sujet) not in seg:
                    st["flags"].append({"id": sid, "cat": "label_sujet_hors_span",
                                        "detail": f"{nid} {lab} | seg={seg[:60]}"})
                if verbe in COQUILLES:
                    st["flags"].append({"id": sid, "cat": "label_coquille",
                                        "detail": f"{nid} {lab}"})
            else:
                st["flags"].append({"id": sid, "cat": "label_sans_parentheses",
                                    "detail": f"{nid} {lab}"})
        for e in cir.get("edges", []):
            rel = e.get("relation", "?")
            if e.get("confidence") == 1.0:
                st["flags"].append({"id": sid, "cat": "conf1",
                                    "detail": f"{rel}"})
            if not e.get("marker_token"):
                st["flags"].append({"id": sid, "cat": "sans_marqueur",
                                    "detail": f"{rel}"})
            src = e.get("sources", [None])[0]
            s0 = (nodes.get(src, {}).get("token_span") or [0, 0])[0]
            t0 = (nodes.get(e.get("target"), {}).get("token_span") or [0, 0])[0]
            if s0 > t0:
                st["flags"].append({"id": sid, "cat": f"back_{rel}",
                                    "detail": f"src@{s0} tgt@{t0}"})
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dossier annotated/")
    ap.add_argument("--out", required=True, help="rapport JSON")
    ap.add_argument("--seuil", type=float, default=0.05)
    ap.add_argument("--exclude-tombstones", action="store_true",
                    help="ignorer les phrases tombstonees (quarantaine*.json)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 si junk (latex) ou en_texte sur phrases actives (gate CI)")
    ap.add_argument("--exceptions", default=None,
                    help="JSON {id: motif} : faux positifs avérés, ignorés avec trace "
                         "(ex : titres EN dans phrase FR). Défaut : exceptions_bruit.json du dossier.")
    a = ap.parse_args()
    exc = {}
    exc_path = a.exceptions or os.path.join(a.dir, "exceptions_bruit.json")
    if os.path.exists(exc_path):
        exc = json.load(open(exc_path, encoding="utf-8"))
        print(f"(exceptions connues: {len(exc)})")
    tomb = set()
    if a.exclude_tombstones:
        for fp in sorted(glob.glob(os.path.join(a.dir, "quarantaine*.json"))):
            try:
                for q in json.load(open(fp, encoding="utf-8")):
                    if isinstance(q, dict) and q.get("id"):
                        tomb.add(q["id"])
            except Exception:
                pass
    lots, totaux = {}, {}
    quar_phrases, en_phrases, p4_ids = [], [], []
    for fp in sorted(glob.glob(os.path.join(a.dir, "lot*.json"))):
        name = os.path.basename(fp)
        st = audit_file(fp)
        cats = {}
        with_flag = set()
        for f in st["flags"]:
            if f["id"] in tomb:
                continue
            if f["id"] in exc:
                continue
            cats[f["cat"]] = cats.get(f["cat"], 0) + 1
            totaux[f["cat"]] = totaux.get(f["cat"], 0) + 1
            with_flag.add(f["id"])
            if f["cat"] == "latex":
                quar_phrases.append(f"{name}#{f['id']}")
            elif f["cat"] == "en_texte":
                en_phrases.append(f"{name}#{f['id']}")
            elif f["cat"] in P4_CATS:
                p4_ids.append(f"{name}#{f['id']}:{f['cat']}")
        conv = {k: v for k, v in cats.items() if k in CONV_CATS}
        lots[name] = {"n": st["n"], "cats": cats,
                      "phrases_avec_flag": len(with_flag),
                      "conventions": conv,
                      "flags": st["flags"]}
    rep = {"lots": lots, "totaux": totaux,
           "quarantaine_junk": quar_phrases,
           "split_en": en_phrases,
           "p4_arbitrage": sorted(set(p4_ids)),
           "seuil": a.seuil}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(f"{'lot':28s} {'n':>4s} {'latex':>5s} {'en':>4s} {'p4':>4s} conv(sans_par/sans_marq)")
    for name, l in lots.items():
        c = l["cats"]
        p4n = sum(v for k, v in c.items() if k in P4_CATS)
        conv = l["conventions"]
        print(f"{name:28s} {l['n']:4d} {c.get('latex',0):5d} {c.get('en_texte',0):4d} "
              f"{p4n:4d} sans_par={conv.get('label_sans_parentheses',0)} "
              f"sans_marq={conv.get('sans_marqueur',0)}")
    print("--- totaux ---")
    for k in sorted(totaux):
        print(f"{k}: {totaux[k]}")
    print(f"junk quarantaine: {len(quar_phrases)} phrases | EN split: {len(en_phrases)} phrases "
          f"| P4 entrees: {len(set(p4_ids))} -> {a.out}")
    if a.exclude_tombstones:
        print(f"(tombstones ignorés: {len(tomb)} ids)")
    if a.strict:
        junk = totaux.get("latex", 0) + totaux.get("en_texte", 0)
        if junk:
            print(f"STRICT FAIL: junk actif latex={totaux.get('latex', 0)} "
                  f"en={totaux.get('en_texte', 0)} -> quarantaine requise")
            raise SystemExit(1)
        print("STRICT OK: 0 junk actif (latex/EN)")


if __name__ == "__main__":
    main()
