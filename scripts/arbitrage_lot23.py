#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Arbitrage lot23 (93 : ~15 strict + 78 litigieuses) — doctrine lots 01-11 + lot20/21/22.

Produit (DATA, hors git) :
  annotated/lot23_82.json      — gold (strict + gardées corrigées)
  annotated/quarantaine_lot23.json — 11 tombstones + motifs
  ARBITRAGE_lot23.md           — sections + verdicts (assistant délégué)
Ne modifie aucun autre fichier.
"""
import json
import sys

V4 = "gcn-datasets/DATA/supervision/v4/annotated"
PROP = "/tmp/opencode/lot23_propose.json"
LITIG = "/tmp/opencode/lot23_litig.json"

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
 "r2-00573": "fragment tronqué (fin 'soured, the', parenthèse ouvrante), spans à cheval",
 "r2-01359": "catalogue lexicographique (3 définitions accolées : agroforesterie/aleurodicide/algicide), indécoupable en 2 nœuds (cf. f0073)",
 "r2-01382": "'contrôle' mentionnel définitionnel (systèmes de contrôle), pas d'asymétrie contrôleur/contrôlé (cf. lot22 'contrôle' mentionné)",
 "r2-01480": "titre accolé ('Évaluation et analyse de risques') + 'contre-mesures' mentionné, pas d'empêchement (cf. s01857)",
 "r2-01653": "'contre' = ennemi désigné (vengeance contre le village), pas d'empêchement (cf. s1567)",
 "r2-02149": "'contrôle' mentionné (régulation biologique), pas de relation de contrôle (cf. lot22)",
 "r2-02366": "'active' adjectival (est active de facto), control non fondé (cf. s1111)",
 "r2-02389": "'active' adjectival (non conformité active), control non fondé (cf. s1111)",
 "r2-02495": "'contrôle' mentionné en parenthèse comparative (par opposition au contrôle optimal), pas de relation",
 "s0377": "'si' intensif-consécutif (si rapide que), marqueur conditionnel invalide (cf. s1538)",
 "s0554": "fragment (sujet manquant, minuscule initiale), 'quand' hors portée",
}
G = {
 "f0005": {"lbl": {"n1": "provoque"}},
 "f0065": {},
 "r2-00035": {"lbl": {"n1": "proposals"}},
 "r2-00183": {},
 "r2-00256": {"lbl": {"n1": "reveal"}},
 "r2-00259": {},
 "r2-00271": {"lbl": {"n1": "charter"}},
 "r2-00327": {},
 "r2-00408": {"spans": {"n1": [1, 9]}, "note": "fenêtre auto égarée sur 'commissioners' (nom taggé verbe par lattice), span imposé"},
 "r2-00438": {"lbl": {"n1": "believed", "n2": "use"}},
 "r2-00461": {},
 "r2-00476": {"lbl": {"n1": "DDR", "n2": "kinases"}},
 "r2-00575": {"lbl": {"n1": "paper"}},
 "r2-00736": {"note": "nœud n1 [1,1] assumé (entité nominale)"},
 "r2-00754": {},
 "r2-00819": {"lbl": {"n2": "65s"}},
 "r2-00878": {"lbl": {"n1": "findings"}, "note": "rubrique 'Egypt' exclue par fenêtre"},
 "r2-00881": {},
 "r2-00919": {"lbl": {"n1": "plays", "n2": "minority"}},
 "r2-00948": {"lbl": {"n1": "associated"}},
 "r2-00986": {},
 "r2-01101": {"lbl": {"n1": "states"}},
 "r2-01110": {"lbl": {"n2": "association"}},
 "r2-01127": {},
 "r2-01282": {},
 "r2-01354": {},
 "r2-01393": {"lbl": {"n1": "appuie"}, "anchor": ("n2", "contrôle")},
 "r2-01421": {"lbl": {"n1": "conversationnels"}},
 "r2-01452": {},
 "r2-01483": {"lbl": {"n1": "stratégies"}},
 "r2-01783": {},
 "r2-01887": {},
 "r2-01919": {},
 "r2-02027": {"n1span": [5, 10], "note": "marqueur partiel 'À' (de 'À l'exception de') ; nœud [1,1] réparé"},
 "r2-02453": {},
 "r2-02486": {},
 "r2-02491": {},
 "r2-02535": {},
 "r2-02652": {"lbl": {"n2": "crée"}},
 "s0340": {"n1span": [4, 10], "note": "label 'régions' hors span proposeur [3,3] ('si'), réparé"},
 "s0409": {},
 "s0640": {"lbl": {"n1": "portant"}},
 "s0744": {},
 "s0881": {},
 "s0955": {},
 "s0976": {},
 "s1010": {"lbl": {"n2": "atteint"}},
 "s1157": {},
 "s1166": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1183": {"n1span": [2, 10]},
 "s1189": {},
 "s1196": {"rel": "cause", "note": "'si bien que' consécutif, pas conditionnel"},
 "s1271": {},
 "s1286": {},
 "s1354": {"lbl": {"n2": "soutenu"}},
 "s1422": {"exp0": True, "note": "'ainsi' conséquence"},
 "s1461": {},
 "s1487": {},
 "s1508": {},
 "s1512": {"lbl": {"n2": "migration"}},
 "s1540": {"lbl": {"n2": "Gallo-Romains"}},
 "s1555": {},
 "s1771": {},
 "s1958": {"spans": {"n2": [13, 22]}},
 "s2092": {"mark": 4, "note": "marqueur corrigé 'vie'→'après' (séquence)"},
 "s2125": {},
 "s2155": {},
 "s2163": {"n1span": [3, 13], "note": "marqueur partiel 'En' (de 'En revanche') ; nœud [1,1] réparé"},
 "s2169": {},
 "s2306": {"lbl": {"n1": "propriétaire"}, "rel": "concession",
           "note": "'indépendamment du fait que' concessif ; marqueur partiel 'du'"},
 "s2309": {"spans": {"n1": [10, 21]}, "lbl": {"n1": "idées"}},
 "s2333": {"spans": {"n1": [16, 25]}, "lbl": {"n1": "relève"}},
 "s2335": {},
 "s2469": {},
 "s2549": {"lbl": {"n1": "non-humains"}, "exp0": True, "note": "'ainsi' conséquence"},
 "s2885": {},
 "s2902": {"exp0": True, "note": "'ainsi que' énumératif, pas causal (cf. f0127)"},
 "s2938": {},
 "s3048": {},
 "s3066": {},
 "s3093": {},
 "s3154": {},
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
               "document": {"id": "lot23-gold", "sentences": sorted(gold, key=lambda r: r["id"])}},
              open(f"{V4}/lot23_{len(gold)}.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sorted(quar, key=lambda q: q["id"]),
              open(f"{V4}/quarantaine_lot23.json", "w"), ensure_ascii=False, indent=1)
    prop_all = {s["id"]: s for s in json.load(open(PROP))["document"]["sentences"]}
    md = ["# Arbitrage lot 23 — 93 proposées (seed 42, caps rares-first)",
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
    open("gcn-datasets/DATA/supervision/v4/ARBITRAGE_lot23.md", "w").write("\n".join(md))
    print(f"GOLD: {len(gold)} (strict {n_strict}) | QUAR: {len(quar)}")

if __name__ == "__main__":
    sys.exit(main())
