#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Arbitrage lot24 (103 : ~12 strict + 91 litigieuses) — doctrine lots 01-11 + lot20/21/22/23.

Produit (DATA, hors git) :
  annotated/lot24_87.json      — gold (strict + gardées corrigées)
  annotated/quarantaine_lot24.json — 16 tombstones + motifs
  ARBITRAGE_lot24.md           — sections + verdicts (assistant délégué)
Ne modifie aucun autre fichier.
"""
import json
import sys

V4 = "gcn-datasets/DATA/supervision/v4/annotated"
PROP = "/tmp/opencode/lot24_propose.json"
LITIG = "/tmp/opencode/lot24_litig.json"

QNORM = str.maketrans({chr(c): "'" for c in (0x2019, 0x2018, 0x0060, 0x00B4)})

def norm(w): return w.translate(QNORM).lower()

def tok_idx(toks, a, b, label):
    for i in range(a - 1, b):
        t = toks[i]
        if norm(t["form"]) == norm(label) or norm(t.get("lemma", "")) == norm(label):
            return i + 1
    return None

def head_verb(toks, a, b, anchor):
    best, bd = None, 10 ** 9
    for i in range(a - 1, b):
        if toks[i].get("pos") == "verb" and abs((i + 1) - anchor) < bd:
            bd, best = abs((i + 1) - anchor), i + 1
    return best if best else anchor

def window(a, b, head, marker):
    na, nb = max(a, head - 7), min(b, head + 7)
    if marker and na <= marker <= nb:
        if marker <= head: na = marker + 1
        else: nb = marker - 1
    return [na, nb]

PUNCT = set(",;:()\"«»")

import re as _re
_PAREN = _re.compile(r"^\(+[a-zàâäéèêëîïôöùûüç]{1,4}\)?$")
def weak(tok):
    w = norm(tok.strip().rstrip(",;:"))
    return w in TRIM or bool(_PAREN.match(w))
def strip_punct(toks, a, b):
    while a < b and toks[a - 1]["form"].strip() in PUNCT: a += 1
    while b > a and toks[b - 1]["form"] in PUNCT: b -= 1
    return [a, b]

TRIM = {"et", "ou", "ne", "se", "me", "te", "nous", "vous", "le", "la", "les",
        "l'", "de", "des", "du", "au", "aux", "à", "en", "y", "que", "qui",
        "dont", "où", "pour", "dans", "sur", "sous", "avec", "par", "vers",
        "contre", "entre", "comme", "est", "sont", "être", "avoir", "faire",
        "plus", "moins", "très", "mais", "davantage", "to", "the", "a", "an", "that", "this", "these", "those",
        "il", "elle", "ils", "elles", "on", "ce", "cela", "ça", "ceci", "other"}

# --- plan par phrase -------------------------------------------------------
# Q = quarantaine(motif) ; G = garder + options :
#   lbl {nid: label} | n1span | spans {(nid): [a,b]} | anchor (nid, tok|label)
#   exp0 (explicit:false) | rel (retype) | mark (nouveau marker) | note
Q = {
 "r2-00300": "'calls' nominal (relevés d'appels), pas de contrôle (cf. aveugle 024)",
 "r2-00598": "'needs' modal (needs to be brought), pas de dépendance aux données",
 "r2-01394": "'contrôle' mentionnel (système décrit), pas d'asymétrie (cohérence lot23, veto pending)",
 "r2-01529": "'sans' circonstanciel d'absence (sans problème), pas d'empêchement",
 "r2-01578": "'sans' circonstanciel (sans avoir obtenu), pas d'empêchement",
 "r2-01579": "'contrôle' nominal (rôle dans le contrôle), mention (cohérence lot23, veto pending)",
 "r2-01905": "'sans pour autant' concessif, pas d'empêchement",
 "r2-02124": "titre accolé ('Contenu Dans ce cadre :') + 'contrôle' mentionnel",
 "r2-02160": "'contrôle' nominal (contrôle génétique), mention (cohérence lot23, veto pending)",
 "r2-02307": "'contre-indications' nominal, pas d'événement d'empêchement",
 "r2-02466": "titre ('Histoire') + 'contrôle' mentionnel ×2",
 "s0470": "'si oui' interrogatif indirect (cf. s1679 sorti du gold)",
 "s0639": "'si' intensif-consécutif (si perturbantes que), marqueur invalide (cf. s0377)",
 "s2310": "formule métadiscursive 'pour commencer', pas de but (piste garde)",
 "s2396": "'sans' circonstanciel d'absence (sans influencer), pas d'empêchement",
 "s2642": "titre + parenthèse conditionnelle, marqueur 'en' erroné (pas 'en cas de')",
}
G = {
 "f0097": {},
 "f0164": {},
 "f0197": {},
 "r2-00195": {},
 "r2-00212": {},
 "r2-00393": {"rel": "condition", "note": "'if and only if' conditionnel, pas restrictif (cf. r2-00124)"},
 "r2-00475": {"note": "marqueur 'animals—based' (trait d'union, lattice)"},
 "r2-00507": {},
 "r2-00529": {"lbl": {"n1": "factors"}},
 "r2-00533": {},
 "r2-00618": {"n1span": [3, 7]},
 "r2-00781": {},
 "r2-00855": {},
 "r2-00909": {"lbl": {"n1": "relation"}, "rel": "condition", "note": "'if and only if' conditionnel"},
 "r2-00913": {},
 "r2-00976": {},
 "r2-01039": {"rel": "prevent", "note": "activation niée (activates no pathways) + preventing"},
 "r2-01131": {"lbl": {"n1": "verification", "n2": "System"}},
 "r2-01132": {},
 "r2-01309": {},
 "r2-01315": {},
 "r2-01356": {},
 "r2-01362": {"n1span": [5, 7], "note": "marqueur partiel 'À' (de 'À l'exception de') ; nœud [1,1] réparé"},
 "r2-01565": {"note": "marqueur '(lutte' (parenthèse, énumération d'objectifs)"},
 "r2-01745": {"lbl": {"n1": "limite"}},
 "r2-01770": {"lbl": {"n1": "attribué"}},
 "r2-01773": {"lbl": {"n1": "lancement"}},
 "r2-01958": {"lbl": {"n1": "réservé"}, "note": "marqueur partiel 'alors' (de 'alors que')"},
 "r2-01960": {},
 "r2-02007": {"lbl": {"n1": "établi"}},
 "r2-02045": {},
 "r2-02103": {"note": "marqueur partiel 'alors' (de 'alors que')"},
 "r2-02195": {},
 "r2-02274": {},
 "r2-02395": {"n1span": [2, 6]},
 "r2-02403": {},
 "r2-02427": {},
 "r2-02488": {},
 "r2-02538": {},
 "r2-02578": {},
 "r2-02579": {"note": "marqueur partiel 'alors' (de 'alors que')"},
 "r2-02585": {},
 "s0050": {},
 "s0286": {"exp0": True, "note": "'ainsi que' énumératif"},
 "s0307": {"lbl": {"n1": "abondantes"}, "note": "marqueur '(mais' (parenthèse)"},
 "s0411": {"note": "marqueur partiel 'en' (de 'en raison de')"},
 "s0483": {},
 "s0618": {},
 "s0944": {"lbl": {"n1": "réaliser"}},
 "s1050": {},
 "s1051": {},
 "s1064": {},
 "s1118": {"exp0": True, "note": "'ainsi que' énumératif"},
 "s1158": {"exp0": True, "note": "'ainsi que' énumératif"},
 "s1162": {},
 "s1210": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1211": {},
 "s1224": {"exp0": True, "note": "'ainsi que' comparatif, pas causal"},
 "s1277": {},
 "s1472": {},
 "s1531": {},
 "s1548": {},
 "s1569": {},
 "s1591": {"rel": "cause", "note": "'si bien que' consécutif (cf. s1196)"},
 "s1606": {},
 "s1669": {},
 "s1751": {},
 "s1761": {},
 "s1798": {},
 "s1886": {},
 "s1924": {},
 "s1985": {"exp0": True, "note": "'ainsi' conséquence"},
 "s2043": {"lbl": {"n1": "batterie"}, "note": "marqueur partiel 'en' (de 'en cas de')"},
 "s2096": {},
 "s2375": {"lbl": {"n1": "certitude", "n2": "définit"}},
 "s2393": {},
 "s2522": {},
 "s2640": {},
 "s2661": {},
 "s2676": {},
 "s2856": {},
 "s2931": {},
 "s2979": {},
 "s3016": {},
 "s3022": {},
 "s3086": {"lbl": {"n1": "gagner"}},
 "s3095": {"lbl": {"n1": "documents"}},
}
JUSTIF_DEFAUT = "spans ±7 (marqueur exclu), conf revalidée (doctrine lots 08-10)"

def main():
    prop = {s["id"]: s for s in json.load(open(PROP))["document"]["sentences"]}
    litig = {l["id"]: l["flags"] for l in json.load(open(LITIG))}
    gold, quar, sections = [], [], []
    n_strict = 0
    for sid in sorted(prop):
        s = prop[sid]
        e = s["cir"]["edges"][0]; ns = s["cir"]["nodes"]; toks = s["tokens"]
        mk = e.get("marker_token")
        mkf = toks[mk - 1]["form"] if mk and 1 <= mk <= len(toks) else "—"
        avant = " | ".join(f"{n['id']}='{n['label']}'{n['token_span']}" for n in ns)
        avant += f" — {e['relation']} (conf={e['confidence']}, expl={e['explicit']}, mk={mk} '{mkf}')"
        flags = [f for f in litig.get(sid, []) if "strip" not in f]
        litige = "; ".join(flags) if flags else "aucun (strict)"
        if sid in Q:
            quar.append({"id": sid, "text": s["text"],
                         "motif": Q[sid] + " (arbitrage 2026-10-04)", "tombstone": True})
            sections.append((sid, "QUARANTAINE", Q[sid], None))
            continue
        p = G.get(sid, {})
        nodes = [dict(n, token_span=list(n["token_span"])) for n in ns]
        orig = {n["id"]: list(n["token_span"]) for n in ns}
        if "n1span" in p:
            nodes[0]["token_span"] = list(p["n1span"])
        # imposed spans d'abord, labels ensuite (sur nouveaux spans, repli orig)
        for nid, sp in p.get("spans", {}).items():
            for n in nodes:
                if n["id"] == nid: n["token_span"] = list(sp)
        for nid, lb in p.get("lbl", {}).items():
            for n in nodes:
                if n["id"] == nid:
                    a, b = n["token_span"]
                    if tok_idx(toks, a, b, lb): n["label"] = lb
                    else:
                        oa, ob = orig[nid]
                        if tok_idx(toks, oa, ob, lb):
                            print(f"WARN {sid} {nid}: label '{lb}' hors nouveau span, gardé ancien")
                        else: print(f"WARN {sid} {nid}: label '{lb}' absent")
        edge = dict(e)
        if "rel" in p: edge["relation"] = p["rel"]
        if p.get("exp0"):
            edge["explicit"] = False; edge.pop("marker_token", None); mk = None
        if "mark" in p:
            assert 1 <= p["mark"] <= len(toks), f"{sid} mark"
            edge["marker_token"] = p["mark"]; mk = p["mark"]
        # imposed spans déjà appliquées plus haut (skip windows)
        imposed = set(p.get("spans", {}))
        # windows ±7
        imposed_here = {nid: sp for nid, sp in p.get("spans", {}).items()}
        if "n1span" in p:
            imposed_here["n1"] = list(p["n1span"])
        for i, n in enumerate(nodes):
            if n["id"] in imposed or (i == 0 and "n1span" in p):
                continue
            a, b = n["token_span"]
            oa, ob = next(nn["token_span"] for nn in ns if nn["id"] == n["id"])
            for oid, (sa, sb) in imposed_here.items():
                if oid != n["id"] and oa <= sa and sb <= ob and sa <= a + 2:
                    a = max(a, sb + 1)
            if a > b:
                print(f"WARN {sid} {n['id']}: fenêtre vide après exclusion")
                continue
            if (b - a + 1) <= 15:
                if (b - a + 1) <= 1:
                    continue
                mkform = toks[mk - 1]["form"] if mk and 1 <= mk <= len(toks) else ""
                if (mk and a <= mk <= b and tok_idx(toks, a, b, n["label"]) != mk
                        and norm(n["label"]) not in norm(mkform)):
                    if mk == b: b -= 1
                    elif mk == a: a += 1
                    n["token_span"] = [a, b]
                continue
            anchor = tok_idx(toks, a, b, n["label"]) or (a + b) // 2
            force = None
            if "anchor" in p and p["anchor"][0] == n["id"]:
                ov = p["anchor"][1]
                if isinstance(ov, int): anchor = ov; force = ov
                else: anchor = tok_idx(toks, a, b, ov) or anchor; force = anchor
            head = force if force else head_verb(toks, a, b, anchor)
            nb = window(a, b, head, mk)
            nb = strip_punct(toks, nb[0], nb[1])
            n["token_span"] = nb
        for k in range(1, len(nodes)):
            order = sorted(range(len(nodes)), key=lambda i: nodes[i]["token_span"][0])
            pv, cu = nodes[order[k - 1]], nodes[order[k]]
            if cu["token_span"][0] <= pv["token_span"][1]:
                cu["token_span"][0] = pv["token_span"][1] + 1
                if cu["token_span"][0] > cu["token_span"][1]:
                    print(f"WARN {sid} {cu['id']}: span vidé par clamp")
        # finition
        for n in nodes:
            a, b = n["token_span"]
            oa, ob = next(nn["token_span"] for nn in ns if nn["id"] == n["id"])
            if n["id"] in imposed or (nodes.index(n) == 0 and "n1span" in p):
                oa, ob = a, b
            while b > a and toks[b - 1]["form"] in (",", ";", ":"): b -= 1
            while b > a and weak(toks[b - 1]["form"]):
                if (b < ob and (b + 1 - a + 1) <= 15
                        and toks[b]["form"] not in (",", ";", ":")
                        and not weak(toks[b]["form"])):
                    b += 1
                    break
                b -= 1
            while b > a and toks[b - 1]["form"] in (",", ";", ":"): b -= 1
            n["token_span"] = [a, b]
        # disjonction en ordre listé (n1 puis n2) AVANT finition
        # justif
        bits = []
        if not flags:
            bits.append("strict-auto (explicit, conf≥0,9, spans≤15, 0 flag)"); n_strict += 1
        else: bits.append(JUSTIF_DEFAUT)
        if p.get("note"): bits.append(p["note"])
        if "rel" in p: bits.append(f"retype {e['relation']}→{p['rel']}")
        if p.get("exp0"): bits.append("explicit:false")
        for nid in p.get("lbl", {}): bits.append(f"label {nid} re-ancré '{p['lbl'][nid]}'")
        if "mark" in p: bits.append(f"marqueur → {p['mark']}")
        just = " ; ".join(bits)
        newspans = [list(n["token_span"]) for n in nodes]
        sections.append((sid, "GARDER", just, newspans))
        rec = dict(s); rec["cir"] = {"nodes": nodes, "edges": [edge]}
        gold.append(rec)
    # écritures
    json.dump({"schema_version": "4.0",
               "document": {"id": "lot24-gold", "sentences": sorted(gold, key=lambda r: r["id"])}},
              open(f"{V4}/lot24_{len(gold)}.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sorted(quar, key=lambda q: q["id"]),
              open(f"{V4}/quarantaine_lot24.json", "w"), ensure_ascii=False, indent=1)
    prop_all = {s["id"]: s for s in json.load(open(PROP))["document"]["sentences"]}
    md = ["# Arbitrage lot 24 — 103 proposées (seed 42, caps rares-first)",
          "", f"Gold : {len(gold)} (dont {n_strict} strict-auto) + {len(quar)} quarantaines (assistant délégué 2026-10-04).",
          "Tombstones conservés, jamais de suppression.", ""]
    for sid, verdict, just, _ns in sections:
        s = prop_all[sid]; toks = s["tokens"]
        e = s["cir"]["edges"][0]
        md.append(f"## {sid}"); md.append(s["text"])
        md.append("- noeuds: " + " | ".join(
            f"{n['id']}='{n['label']}'{n['token_span']}" for n in s["cir"]["nodes"]))
        md.append(f"- relation: {e['relation']} (conf={e['confidence']}, "
                  f"explicit={e['explicit']}, marker={e.get('marker_token')})")
        fl = [f for f in litig.get(sid, []) if "strip" not in f]
        if fl: md.append("- LITIGE: " + "; ".join(fl))
        if verdict == "GARDER":
            g = next(r for r in gold if r["id"] == sid); det = []
            for n in g["cir"]["nodes"]:
                a, b = n["token_span"]
                det.append(f"{n['id']}='{n['label']}'[{a},{b}]")
            pe = g["cir"]["edges"][0]
            md.append(f"- après: {' | '.join(det)} — {pe['relation']} "
                      f"(conf={pe['confidence']}, expl={pe['explicit']}, mk={pe.get('marker_token')})")
            md.append(f"- verdict: GARDER — {just} (assistant délégué 2026-10-04)")
        else:
            md.append(f"- verdict: QUARANTAINE : {just} (assistant délégué 2026-10-04)")
        md.append("")
    open("gcn-datasets/DATA/supervision/v4/ARBITRAGE_lot24.md", "w").write("\n".join(md))
    print(f"GOLD: {len(gold)} (strict {n_strict}) | QUAR: {len(quar)}")

if __name__ == "__main__":
    sys.exit(main())
