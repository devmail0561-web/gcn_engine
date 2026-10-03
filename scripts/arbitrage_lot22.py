#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Arbitrage lot22 (93 : 22 strict + 71 litigieuses) — doctrine lots 01-11 + lot20/21.

Produit (DATA, hors git) :
  annotated/lot21_83.json      — gold (strict + gardées corrigées)
  annotated/quarantaine_lot21.json — 10 tombstones + motifs
  ARBITRAGE_lot22.md           — sections + verdicts (assistant délégué)
Ne modifie aucun autre fichier.
"""
import json
import sys

V4 = "gcn-datasets/DATA/supervision/v4/annotated"
PROP = "/tmp/opencode/lot22_propose.json"
LITIG = "/tmp/opencode/lot22_litig.json"

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
 "r2-02604": "fragment ('nécessaire]...', marqueur bruité crochet, n1/n2 chevauchés (cf. r2-02607)",
 "s0653": "marqueur 'Froide' dans nom propre (Guerre Froide), relation sequence non fondée",
 "s0883": "double 'si' coordonnés + conséquent 66 toks, indécoupable proprement (cf. f0073)",
 "s2782": "notice bibliographique (Laudato Si'), 'Si' dans titre, pas conditionnel",
 "r2-02441": "'contrôle' objet d'étude définitionnel, pas de relation de contrôle",
 "r2-01899": "'contre' = collision physique (choc), pas empêchement",
 "r2-02447": "'sans nécessairement être' circonstanciel, pas d'empêchement",
 "s2874": "'sans effacer' circonstanciel (manière), pas d'empêchement",
}
G = {
 # strict-auto : verdict mécanique, spans/labels/edges inchangés (détecté auto)
 "f0007": {"lbl": {"n1": "engendre", "n2": "fission"}},
 "q0030": {},
 "q0037": {"spans": {"n2": [3, 12]}},
 "r2-00170": {"n1span": [3, 3]},
 "r2-00405": {"n1span": [3, 4]},
 "r2-00248": {"lbl": {"n1": "interests"}},
 "r2-00315": {"lbl": {"n2": "inhibited"}},
 "r2-01008": {"lbl": {"n2": "mistress"}, "rel": "concession",
              "note": "'against Pizarro\u2019s better judgment' = malgré (concessif)"},
 "r2-01077": {"lbl": {"n2": "ways"}},
 "r2-01109": {"lbl": {"n2": "provide"}},
 "r2-01204": {"lbl": {"n1": "compléments"}},
 "r2-01326": {"lbl": {"n2": "prolifération"}},
 "r2-01797": {"lbl": {"n2": "briques"}, "anchor": ("n2", "hypothèses")},
 "r2-01909": {"anchor": ("n2", "trampoline")},
 "r2-02221": {"lbl": {"n1": "établit"}},
 "r2-02331": {"lbl": {"n1": "action"}},
 "r2-02406": {"rel": "opposition",
              "note": "'aller contre' = violation des normes (opposition), pas empêchement"},
 "r2-02586": {"rel": "opposition",
              "note": "'contre' sportif (adversaire) comme s1713, pas préventif"},
 "r2-02677": {"lbl": {"n1": "surfaces"}},
 "s0361": {"lbl": {"n1": "Amérique"}, "exp0": True, "note": "'ainsi' conséquence"},
 "s0414": {"lbl": {"n2": "mois"}},
 "s1916": {"lbl": {"n1": "proximité"}, "note": "règle A2 (sans verbe)"},
 "s1388": {"lbl": {"n1": "respecté"}, "rel": "concession",
           "note": "'quand bien même' = concessif"},
 "r2-01625": {"mark": 32, "note": "marqueur 'à' → 'exception' (cf. lot08 s0415)"},
 "s0047": {"note": "marqueur partiel 'du' (de 'du fait que')"},
 "s2591": {"note": "marqueur 'cause' nominal"},
 "s2955": {"note": "marqueur partiel 'parce' (de 'parce que')"},
 "s2437": {"note": "marqueur 'cause' nominal"},
 "r2-00545": {"lbl": {"n2": "safeguards"}, "note": "marqueur 'needs' avec guillemet (bruit pool)"},
 "r2-00759": {"note": "marqueur 'controls:' avec deux-points (bruit pool)"},
 "r2-01968": {"note": "marqueur 'uniquement)' avec parenthèse (bruit pool)"},
 "r2-01971": {"note": "marqueur composé 'contrôle-commande' lexical (cf. s1327 contre-maquis)"},
 # --- 'ainsi' manière/conséquence/énumératif -> explicit:false (lot05 s1763) ---
 "s0230": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1070": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1785": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1979": {"exp0": True, "note": "'ainsi' conséquence"},
 "s2346": {"exp0": True, "note": "'ainsi' manière"},
 "s2565": {"exp0": True, "note": "'ainsi' conséquence"},
 "s2772": {"exp0": True, "note": "'ainsi' conséquence"},
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
            assert norm(toks[p["mark"] - 1]["form"]) == "exception", f"{sid} mark"
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
               "document": {"id": "lot22-gold", "sentences": sorted(gold, key=lambda r: r["id"])}},
              open(f"{V4}/lot22_{len(gold)}.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sorted(quar, key=lambda q: q["id"]),
              open(f"{V4}/quarantaine_lot22.json", "w"), ensure_ascii=False, indent=1)
    prop_all = {s["id"]: s for s in json.load(open(PROP))["document"]["sentences"]}
    md = ["# Arbitrage lot 22 — 93 proposées (seed 42, caps rares-first)",
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
    open("gcn-datasets/DATA/supervision/v4/ARBITRAGE_lot22.md", "w").write("\n".join(md))
    print(f"GOLD: {len(gold)} (strict {n_strict}) | QUAR: {len(quar)}")

if __name__ == "__main__":
    sys.exit(main())
