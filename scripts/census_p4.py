#!/usr/bin/env python3
"""P4 plan-remediation-donnees.md : census des arêtes à sens douteux + ARBITRAGE_P4 pré-remplis.

Lecture seule sur les lots. Ecrit :
  <v4>/ARBITRAGE_P4_<lot>.md : un fichier par lot avec cas back_*, proposition + cases à cocher.
  <v4>/P4_propositions.json   : même contenu, exploitable pour application après validation.
Ne modifie AUCUNE donnée. L'arbitre humain tranche (doctrine : juge=humain).
"""
import argparse
import glob
import json
import os
import re

AP = argparse.ArgumentParser()
AP.add_argument("--rapport", required=True, help="bruit_*.json de detect_bruit.py")
AP.add_argument("--annotated", required=True)
AP.add_argument("--outdir", required=True, help="dossier v4/ recevant ARBITRAGE_P4_*")
AP.add_argument("--actifs-seuls", action="store_true",
                help="exclure les phrases tombstonees (quarantaine*.json) : arbitrage inutile")
A = AP.parse_args()

tomb = set()
if A.actifs_seuls:
    for fp in sorted(glob.glob(os.path.join(A.annotated, "quarantaine*.json"))):
        try:
            for q in json.load(open(fp, encoding="utf-8")):
                if isinstance(q, dict) and q.get("id"):
                    tomb.add(q["id"])
        except Exception:
            pass
    print(f"tombstones charges: {len(tomb)}")

CAUSALES = {"cause", "enable", "prevent", "motivation", "condition", "sequence",
            "data_dependency", "control_dependency", "filter"}
DISCURSIVES = {"concession", "opposition"}


def span_text(s, span):
    forms = [x.get("form", "") for x in (s.get("tokens", []) or [])]
    t = s.get("text", "")
    if not span or span[1] < span[0]:
        return "?"
    if span[1] == span[0]:
        return forms[span[0] - 1] if 1 <= span[0] <= len(forms) else "?"
    if span[1] <= len(forms):
        return " ".join(forms[span[0] - 1:span[1]])
    if span[1] <= len(t):
        return t[span[0]:span[1]]
    return "?"


def proposer(rel, tsrc, ttgt, s0, t0):
    """Retourne (proposition, regle, confiance). Convention proposee (à valider) :
    source = antécédent textuel. Toute proposition reste à cocher par l'humain."""
    if rel in DISCURSIVES and tsrc == "entite" and ttgt == "entite":
        return ("QUARANTAINE",
                "relation discursive entre deux entités, aucun fait concédé/opposé",
                "moyenne")
    if rel in CAUSALES:
        return ("INVERSER (src<->tgt)",
                "relation causale : source attendue = antécédent textuel",
                "moyenne")
    return ("INVERSER (src<->tgt)",
            "convention proposée : source = antécédent textuel (à trancher, préalable P4)",
            "basse")


by_lot = {}
for fp in sorted(glob.glob(os.path.join(A.annotated, "lot*.json"))):
    d = json.load(open(fp, encoding="utf-8"))
    by_lot[os.path.basename(fp)] = {s["id"]: s for s in d["document"]["sentences"]}

rep = json.load(open(A.rapport, encoding="utf-8"))
cas = [f for lot, l in rep["lots"].items() for f in l["flags"]
       if f["cat"].startswith("back_")]
par_lot = {}
for lot, l in rep["lots"].items():
    backs = [f for f in l["flags"] if f["cat"].startswith("back_")
             and f["id"] not in tomb]
    if backs:
        par_lot[lot] = backs
print(f"cas back_*: {len(cas)} dans {len(par_lot)} lots")

prop_json = []
os.makedirs(A.outdir, exist_ok=True)
for lot, backs in sorted(par_lot.items()):
    lines = [f"# Arbitrage P4 — {lot} (PROPOSITION, à valider)",
             "",
             f"Fichier : `annotated/{lot}`. {len(backs)} arête(s) à sens douteux "
             "(source après cible en ordre textuel).",
             "Préalables à trancher (plan §P4) : convention de sens "
             "(proposé : source = antécédent textuel) ; F2 sens `motivation` (guide §4).",
             "Cases : cocher UNE seule (oui = appliquer la proposition).",
             ""]
    for i, f in enumerate(backs, 1):
        s = by_lot[lot][f["id"]]
        rel = f["cat"][5:]
        cir = s.get("cir", {})
        nodes = {n.get("id"): n for n in cir.get("nodes", [])}
        e = next((x for x in cir.get("edges", [])
                  if x.get("relation") == rel), {})
        src, tgt = (e.get("sources") or ["?"])[0], e.get("target", "?")
        ns, nt = nodes.get(src, {}), nodes.get(tgt, {})
        s0 = (ns.get("token_span") or [0, 0])[0]
        t0 = (nt.get("token_span") or [0, 0])[0]
        m = e.get("marker_token")
        forms = [x.get("form", "") for x in (s.get("tokens", []) or [])]
        mform = forms[m - 1] if isinstance(m, int) and 1 <= m <= len(forms) else "?"
        prop, regle, conf = proposer(rel, ns.get("type"), nt.get("type"), s0, t0)
        prop_json.append({"lot": lot, "id": f["id"], "relation": rel, "src": src,
                          "tgt": tgt, "proposition": prop, "regle": regle,
                          "confiance": conf, "verdict": None})
        lines += [f"## {i}. {f['id']} — `{rel}`",
                  f"Texte : {s.get('text', '')}",
                  f"- SRC {src} [{ns.get('type')}] « {span_text(s, ns.get('token_span'))} » "
                  f"(label `{ns.get('label')}`)",
                  f"- TGT {tgt} [{nt.get('type')}] « {span_text(s, nt.get('token_span'))} » "
                  f"(label `{nt.get('label')}`)",
                  f"- marqueur : `{mform}` | ordre textuel : src@{s0} tgt@{t0}",
                  f"- **Proposition : {prop}** — {regle} (confiance {conf})",
                  "- [ ] oui  [ ] non (GARDER)  [ ] corriger : ___",
                  ""]
    lines += ["Après validation : reporter verdicts dans `P4_propositions.json` "
              "(`verdict`: garder|inverser|quarantaine|retyper:REL), puis application.",
              "Tombstones/quarantaines, jamais de suppression."]
    open(os.path.join(A.outdir, f"ARBITRAGE_P4_{lot[:-5]}.md"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(prop_json, open(os.path.join(A.outdir, "P4_propositions.json"), "w",
                                        encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"écrit: {len(par_lot)} ARBITRAGE_P4_*.md + P4_propositions.json ({len(prop_json)} cas)")
