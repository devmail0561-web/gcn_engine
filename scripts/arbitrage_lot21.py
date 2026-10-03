#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Arbitrage lot21 (93 : 14 strict + 79 litigieuses) — doctrine lots 01-11 + lot20.

Produit (DATA, hors git) :
  annotated/lot21_83.json      — gold (strict + gardées corrigées)
  annotated/quarantaine_lot21.json — 10 tombstones + motifs
  ARBITRAGE_lot21.md           — sections + verdicts (assistant délégué)
Ne modifie aucun autre fichier.
"""
import json
import sys

V4 = "gcn-datasets/DATA/supervision/v4/annotated"
PROP = "/tmp/opencode/lot21_propose.json"
LITIG = "/tmp/opencode/lot21_litig.json"

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

def strip_punct(toks, a, b):
    while a < b and toks[a - 1]["form"].strip() in PUNCT: a += 1
    while b > a and toks[b - 1]["form"] in PUNCT: b -= 1
    return [a, b]

TRIM = {"et", "ou", "ne", "se", "me", "te", "nous", "vous", "le", "la", "les",
        "l'", "de", "des", "du", "au", "aux", "à", "en", "y", "que", "qui",
        "dont", "où", "pour", "dans", "sur", "sous", "avec", "par", "vers",
        "contre", "entre", "comme", "est", "sont", "être", "avoir", "faire",
        "plus", "moins", "très", "mais", "davantage", "to", "the", "a", "an"}

# --- plan par phrase -------------------------------------------------------
# Q = quarantaine(motif) ; G = garder + options :
#   lbl {nid: label} | n1span | spans {(nid): [a,b]} | anchor (nid, tok|label)
#   exp0 (explicit:false) | rel (retype) | mark (nouveau marker) | note
Q = {
 "r2-00412": "marqueur 'Without' dans nom propre (Reporters Without Borders), prevent non fondée (cf. s1690)",
 "r2-01749": "'sans sacrifier' circonstanciel (manière), pas d'empêchement (cf. s2758)",
 "r2-02097": "'sans s'intéresser' circonstanciel, pas d'empêchement (cf. s2758)",
 "r2-01857": "titre accolé ('Contrôle par voie d\\u2019action' + corps), spans à cheval (cf. s2540/s2541)",
 "r2-02532": "'active' adjectival ×2 (être actif, sérine active), control non fondé (cf. s1111)",
 "r2-02607": "fragment ('nécessaire])...', marqueur bruité double parenthèse, n1/n2 chevauchés",
 "s1567": "'contre' = ennemi désigné (campagnes contre les Vascons), pas d'empêchement ; lecture enable sans déclencheur",
 "s2195": "marqueur 'Against' dans nom propre (Race Against The Machine, titre), prevent non fondée",
 "s0384": "'si' interrogatif indirect (déterminer si), condition non fondée (cf. s0475/s0107/s1679)",
 "s1538": "'si' intensif (un si grave outrage), pas conditionnel",
}
G = {
 # strict-auto : verdict mécanique, spans/labels/edges inchangés (détecté auto)
 # --- re-ancrages labels faibles / marqueur-dans-nœud ---
 "r2-01547": {"lbl": {"n1": "considérer"}},
 "r2-02485": {"lbl": {"n1": "intégrée"}},
 "r2-00956": {"lbl": {"n2": "method"}, "spans": {"n2": [4, 13]}},
 "r2-01269": {"lbl": {"n1": "équipés"}},
 "r2-02408": {"lbl": {"n1": "définit"}},
 "r2-01381": {"lbl": {"n1": "implémente"}},
 "r2-00329": {"spans": {"n1": [6, 16]}},
 "s0577": {"lbl": {"n2": "adopte"}, "n1span": [1, 4], "spans": {"n2": [6, 16]}},
 "s2287": {"lbl": {"n1": "inventions"}, "spans": {"n1": [2, 11]}},
 "s2148": {"lbl": {"n2": "domine"}},
 "s0851": {"n1span": [2, 3]},
 "s0866": {"n1span": [1, 6]},
 "s0936": {"lbl": {"n1": "désintéressent", "n2": "craignent"},
           "spans": {"n1": [2, 14], "n2": [16, 28]}},
 "s0938": {"spans": {"n1": [2, 5], "n2": [7, 16]}},
 "s2822": {"lbl": {"n1": "dote"}, "rel": "cause",
           "note": "'si bien que' = consécutive (cf. s0649)"},
 "r2-00203": {"anchor": ("n1", 29), "rel": "concession",
              "note": "'against token opposition' = malgré (concessif), pas préventif"},
 # --- 'ainsi' manière/conséquence/énumératif -> explicit:false (lot05 s1763) ---
 "s0173": {"exp0": True, "note": "'ainsi' manière"},
 "s0420": {"exp0": True, "note": "'ainsi que' énumératif"},
 "s0456": {"exp0": True, "note": "'ainsi' manière (ainsi entretenue)"},
 "s1201": {"exp0": True, "note": "'ainsi' manière/conséquence"},
 "s2358": {"exp0": True, "note": "'ainsi que' énumératif"},
 "s3122": {"exp0": True, "note": "'ainsi' conséquence factuelle"},
 # --- retypes / marqueurs ---
 "r2-00124": {"rel": "condition", "note": "'only if' = conditionnel (contingence), pas restrictif nominal"},
 "r2-01117": {"rel": "condition", "note": "idem 'only if'"},
 "r2-01625": {"mark": 32, "note": "marqueur 'à' (grâce à) → 'exception' (à l'exception de, cf. lot08 s0415)"},
 "s0047": {"note": "marqueur partiel 'du' (de 'du fait que')"},
 "s2591": {"note": "marqueur 'cause' nominal (la principale cause de)"},
 "s2955": {"note": "marqueur partiel 'parce' (de 'parce que')"},
 "s0147": {"anchor": ("n2", 28), "n1span": [2, 5]},
 "r2-00249": {"lbl": {"n1": "planning"}},
 "r2-00275": {"lbl": {"n1": "argued"}},
 "r2-00374": {"lbl": {"n2": "success"}},
 "r2-00588": {"lbl": {"n1": "confirmed"}},
 "r2-02034": {"spans": {"n2": [24, 29]}},
 "s0420": {"lbl": {"n1": "regroupent", "n2": "définis"}},
 "s0722": {"lbl": {"n2": "École"}},
 "s2102": {"lbl": {"n1": "onéreux", "n2": "offre"}, "anchor": ("n2", 17)},
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
        for i, n in enumerate(nodes):
            if n["id"] in imposed or (i == 0 and "n1span" in p):
                continue
            a, b = n["token_span"]
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
        # finition
        for n in nodes:
            a, b = n["token_span"]
            oa, ob = next(nn["token_span"] for nn in ns if nn["id"] == n["id"])
            if n["id"] in imposed or (nodes.index(n) == 0 and "n1span" in p):
                oa, ob = a, b
            while b > a and toks[b - 1]["form"] in (",", ";", ":"): b -= 1
            while b > a and norm(toks[b - 1]["form"]) in TRIM:
                if (b < ob and (b + 1 - a + 1) <= 15
                        and toks[b]["form"] not in (",", ";", ":")
                        and norm(toks[b]["form"]) not in TRIM):
                    b += 1
                    break
                b -= 1
            while b > a and toks[b - 1]["form"] in (",", ";", ":"): b -= 1
            n["token_span"] = [a, b]
        # disjonction
        order = sorted(range(len(nodes)), key=lambda i: nodes[i]["token_span"][0])
        for k in range(1, len(order)):
            pv, cu = nodes[order[k - 1]], nodes[order[k]]
            if cu["token_span"][0] <= pv["token_span"][1]:
                cu["token_span"][0] = pv["token_span"][1] + 1
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
               "document": {"id": "lot21-gold", "sentences": sorted(gold, key=lambda r: r["id"])}},
              open(f"{V4}/lot21_{len(gold)}.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sorted(quar, key=lambda q: q["id"]),
              open(f"{V4}/quarantaine_lot21.json", "w"), ensure_ascii=False, indent=1)
    prop_all = {s["id"]: s for s in json.load(open(PROP))["document"]["sentences"]}
    md = ["# Arbitrage lot 21 — 93 proposées (seed 42, caps rares-first)",
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
    open("gcn-datasets/DATA/supervision/v4/ARBITRAGE_lot21.md", "w").write("\n".join(md))
    print(f"GOLD: {len(gold)} (strict {n_strict}) | QUAR: {len(quar)}")

if __name__ == "__main__":
    sys.exit(main())
