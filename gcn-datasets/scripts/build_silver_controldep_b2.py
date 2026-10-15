#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""P1b-batch2 : 50 control_dependency inédits (classe dure : 68 proposées minage cumulé).

Même contrat que build_silver_rares_p1b.py : spans calculés, marqueurs token exact,
poids 0.0, hors DATA/train. Dédup vs pool + batch1 vérifiée après build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

# (texte, [(span, label, type)], [(src, dst, relation, marqueur)])
CD2 = [
("Le berger a sifflé le rassemblement sous les premiers flocons.",
 [("Le berger", "le berger", "entite"), ("le rassemblement", "rassemblement sifflé", "processus")],
 [(0, 1, "control_dependency", "sifflé")]),
("La directrice a avancé la réunion sous la visite de l'inspecteur.",
 [("La directrice", "la directrice", "entite"), ("la réunion", "réunion avancée", "evenement")],
 [(0, 1, "control_dependency", "avancé")]),
("Le moniteur a interrompu la descente sous le brouillard givrant.",
 [("Le moniteur", "le moniteur", "entite"), ("la descente", "descente interrompue", "processus")],
 [(0, 1, "control_dependency", "interrompu")]),
("L'huissier a scellé les cartons sous l'ordonnance du juge.",
 [("L'huissier", "huissier", "entite"), ("les cartons", "cartons scellés", "entite")],
 [(0, 1, "control_dependency", "scellé")]),
("La cheffe de chœur a fait taire les altos sous la fausse note.",
 [("La cheffe de chœur", "cheffe de chœur", "entite"), ("les altos", "altos tus", "entite")],
 [(0, 1, "control_dependency", "taire")]),
("Le régisseur a coupé les micros sous le larsen strident.",
 [("Le régisseur", "le régisseur", "entite"), ("les micros", "micros coupés", "entite")],
 [(0, 1, "control_dependency", "coupé")]),
("La surveillante a ramassé les copies sous la sonnerie.",
 [("La surveillante", "la surveillante", "entite"), ("les copies", "copies ramassées", "entite")],
 [(0, 1, "control_dependency", "ramassé")]),
("Le skipper a affalé la grand-voile sous la rafale.",
 [("Le skipper", "le skipper", "entite"), ("la grand-voile", "grand-voile affalée", "etat_local")],
 [(0, 1, "control_dependency", "affalé")]),
("L'aiguilleur a dérouté le convoi sous les travaux de nuit.",
 [("L'aiguilleur", "aiguilleur", "entite"), ("le convoi", "convoi dérouté", "processus")],
 [(0, 1, "control_dependency", "dérouté")]),
("La bibliothécaire a chuchoté le silence sous les regards.",
 [("La bibliothécaire", "bibliothécaire", "entite"), ("le silence", "silence imposé", "etat_local")],
 [(0, 1, "control_dependency", "chuchoté")]),
("Le chef de chantier a stoppé la grue sous les vents violents.",
 [("Le chef de chantier", "chef de chantier", "entite"), ("la grue", "grue stoppée", "etat_local")],
 [(0, 1, "control_dependency", "stoppé")]),
("La présidente a levé la séance sous le tumulte.",
 [("La présidente", "la présidente", "entite"), ("la séance", "séance levée", "evenement")],
 [(0, 1, "control_dependency", "levé")]),
("Le sommelier a débouché le bordeaux sous l'œil du client.",
 [("Le sommelier", "le sommelier", "entite"), ("le bordeaux", "bordeaux débouché", "processus")],
 [(0, 1, "control_dependency", "débouché")]),
("L'ambulancier a sanglé le brancard sous les secousses.",
 [("L'ambulancier", "ambulancier", "entite"), ("le brancard", "brancard sanglé", "etat_local")],
 [(0, 1, "control_dependency", "sanglé")]),
("Le jardinier a bâché les rosiers sous la grêle.",
 [("Le jardinier", "le jardinier", "entite"), ("les rosiers", "rosiers bâchés", "etat_local")],
 [(0, 1, "control_dependency", "bâché")]),
("La caissière a appelé le responsable sous le billet déchiré.",
 [("La caissière", "la caissière", "entite"), ("le responsable", "responsable appelé", "entite")],
 [(0, 1, "control_dependency", "appelé")]),
("Le maçon a étayé le linteau sous les fissures.",
 [("Le maçon", "le maçon", "entite"), ("le linteau", "linteau étayé", "etat_local")],
 [(0, 1, "control_dependency", "étayé")]),
("L'apiculteur a enfumé la ruche sous l'agitation des abeilles.",
 [("L'apiculteur", "apiculteur", "entite"), ("la ruche", "ruche enfumée", "etat_local")],
 [(0, 1, "control_dependency", "enfumé")]),
("Le projectionniste a relancé la bobine sous les applaudissements.",
 [("Le projectionniste", "projectionniste", "entite"), ("la bobine", "bobine relancée", "processus")],
 [(0, 1, "control_dependency", "relancé")]),
("La fleuriste a arrosé les orchidées sous la canicule.",
 [("La fleuriste", "la fleuriste", "entite"), ("les orchidées", "orchidées arrosées", "etat_local")],
 [(0, 1, "control_dependency", "arrosé")]),
("Le poissonnier a glacé l'étal sous la chaleur.",
 [("Le poissonnier", "le poissonnier", "entite"), ("l'étal", "étal glacé", "etat_local")],
 [(0, 1, "control_dependency", "glacé")]),
("L'électricien a disjoncté l'atelier sous le court-circuit.",
 [("L'électricien", "électricien", "entite"), ("l'atelier", "atelier disjoncté", "etat_local")],
 [(0, 1, "control_dependency", "disjoncté")]),
("Le caviste a carafer le pomerol sous la commande du client.",
 [("Le caviste", "le caviste", "entite"), ("le pomerol", "pomerol carafé", "processus")],
 [(0, 1, "control_dependency", "carafer")]),
("La lingère a amidonné les nappes sous la réception du soir.",
 [("La lingère", "la lingère", "entite"), ("les nappes", "nappes amidonnées", "etat_local")],
 [(0, 1, "control_dependency", "amidonné")]),
("Le forgeron a trempé la lame sous la braise.",
 [("Le forgeron", "le forgeron", "entite"), ("la lame", "lame trempée", "processus")],
 [(0, 1, "control_dependency", "trempé")]),
("La relieuse a cousu les cahiers sous la commande urgente.",
 [("La relieuse", "la relieuse", "entite"), ("les cahiers", "cahiers cousus", "processus")],
 [(0, 1, "control_dependency", "cousu")]),
("Le cireur a lustré les chaussures sous le regard du client.",
 [("Le cireur", "le cireur", "entite"), ("les chaussures", "chaussures lustrées", "etat_local")],
 [(0, 1, "control_dependency", "lustré")]),
("L'horloger a remonté la pendule sous la panne.",
 [("L'horloger", "horloger", "entite"), ("la pendule", "pendule remontée", "processus")],
 [(0, 1, "control_dependency", "remonté")]),
("Le vitrier a remplacé la vitrine sous le vandalisme.",
 [("Le vitrier", "le vitrier", "entite"), ("la vitrine", "vitrine remplacée", "processus")],
 [(0, 1, "control_dependency", "remplacé")]),
("La modiste a repassé les voilettes sous la commande de la mariée.",
 [("La modiste", "la modiste", "entite"), ("les voilettes", "voilettes repassées", "etat_local")],
 [(0, 1, "control_dependency", "repassé")]),
("Le tonnelier a cerclé les fûts sous la vendange.",
 [("Le tonnelier", "le tonnelier", "entite"), ("les fûts", "fûts cerclés", "processus")],
 [(0, 1, "control_dependency", "cerclé")]),
("L'émouleur a affûté les couteaux sous la demande du boucher.",
 [("L'émouleur", "émouleur", "entite"), ("les couteaux", "couteaux affûtés", "etat_local")],
 [(0, 1, "control_dependency", "affûté")]),
("Le sellier a graissé les harnais sous la pluie.",
 [("Le sellier", "le sellier", "entite"), ("les harnais", "harnais graissés", "etat_local")],
 [(0, 1, "control_dependency", "graissé")]),
("La dentellière a tendu le fuseau sous la lumière rasante.",
 [("La dentellière", "la dentellière", "entite"), ("le fuseau", "fuseau tendu", "processus")],
 [(0, 1, "control_dependency", "tendu")]),
("Le maréchal-ferrant a ferré le percheron sous la chaleur de la forge.",
 [("Le maréchal-ferrant", "maréchal-ferrant", "entite"), ("le percheron", "percheron ferré", "processus")],
 [(0, 1, "control_dependency", "ferré")]),
("Le sabotier a creusé les galoches sous la commande du fermier.",
 [("Le sabotier", "le sabotier", "entite"), ("les galoches", "galoches creusées", "processus")],
 [(0, 1, "control_dependency", "creusé")]),
("La fileuse a cardé la laine sous la veillée.",
 [("La fileuse", "la fileuse", "entite"), ("la laine", "laine cardée", "processus")],
 [(0, 1, "control_dependency", "cardé")]),
("Le vannier a tressé les paniers sous la commande du marché.",
 [("Le vannier", "le vannier", "entite"), ("les paniers", "paniers tressés", "processus")],
 [(0, 1, "control_dependency", "tressé")]),
("Le chaudronnier a riveté la cuve sous la pression du délai.",
 [("Le chaudronnier", "le chaudronnier", "entite"), ("la cuve", "cuve rivetée", "etat_local")],
 [(0, 1, "control_dependency", "riveté")]),
("La brodeuse a ourlé les mouchoirs sous la commande du trousseau.",
 [("La brodeuse", "la brodeuse", "entite"), ("les mouchoirs", "mouchoirs ourlés", "etat_local")],
 [(0, 1, "control_dependency", "ourlé")]),
("Le doreur a patiné les cadres sous la poussière d'or.",
 [("Le doreur", "le doreur", "entite"), ("les cadres", "cadres patinés", "processus")],
 [(0, 1, "control_dependency", "patiné")]),
("Le relieur a doré les tranches sous la lampe.",
 [("Le relieur", "le relieur", "entite"), ("les tranches", "tranches dorées", "etat_local")],
 [(0, 1, "control_dependency", "doré")]),
("La passementière a galonné les rideaux sous la commande du théâtre.",
 [("La passementière", "la passementière", "entite"), ("les rideaux", "rideaux galonnés", "etat_local")],
 [(0, 1, "control_dependency", "galonné")]),
("Le tabletier a incrusté les coffrets sous la nacre.",
 [("Le tabletier", "le tabletier", "entite"), ("les coffrets", "coffrets incrustés", "processus")],
 [(0, 1, "control_dependency", "incrusté")]),
("L'enlumineur a rehaussé les lettrines sous la feuille d'or.",
 [("L'enlumineur", "enlumineur", "entite"), ("les lettrines", "lettrines rehaussées", "processus")],
 [(0, 1, "control_dependency", "rehaussé")]),
("Le santonnier a moulé les santons sous la commande de Noël.",
 [("Le santonnier", "le santonnier", "entite"), ("les santons", "santons moulés", "processus")],
 [(0, 1, "control_dependency", "moulé")]),
("La plumassière a monté les aigrettes sous la commande du cabaret.",
 [("La plumassière", "la plumassière", "entite"), ("les aigrettes", "aigrettes montées", "etat_local")],
 [(0, 1, "control_dependency", "monté")]),
("Le feutrier a foulé les chapeaux sous la vapeur.",
 [("Le feutrier", "le feutrier", "entite"), ("les chapeaux", "chapeaux foulés", "processus")],
 [(0, 1, "control_dependency", "foulé")]),
("La gantière a piqué les gants sous la commande du régiment.",
 [("La gantière", "la gantière", "entite"), ("les gants", "gants piqués", "etat_local")],
 [(0, 1, "control_dependency", "piqué")]),
("Le layetier a chevillé les tiroirs sous la commande de l'ébéniste.",
 [("Le layetier", "le layetier", "entite"), ("les tiroirs", "tiroirs chevillés", "processus")],
 [(0, 1, "control_dependency", "chevillé")]),
]


def _tokens(text: str) -> list[dict]:
    toks, pos = [], 0
    for k, forme in enumerate(text.split(), 1):
        i = text.find(forme, pos)
        assert i >= 0, forme
        toks.append({"id": k, "form": forme, "lemma": forme.lower(), "pos": "other",
                     "dep_rel": "other", "dep_head": -1, "morph": {}})
        pos = i + len(forme)
    return toks


def _span_tokens(text: str, toks: list[dict], clause: str) -> list[int]:
    s = text.find(clause)
    assert s >= 0, clause
    e = s + len(clause)
    bounds, p = [], 0
    for t in toks:
        i = text.find(t["form"], p)
        bounds.append((i, i + len(t["form"])))
        p = i + len(t["form"])
    out = [k for k, t in enumerate(toks, 1) if s <= bounds[k - 1][0] < e]
    assert out, clause
    while len(out) > 1 and re.fullmatch(r"[.,;:!?]+", toks[out[-1] - 1]["form"]):
        out.pop()
    return [out[0], out[-1]]


def _marker_id(toks: list[dict], marqueur: str) -> int:
    for t in toks:
        if t["form"] == marqueur or t["form"].rstrip(",.") == marqueur:
            return t["id"]
    raise AssertionError(marqueur)


def construire() -> dict:
    assert len(CD2) == 50, len(CD2)
    sentences = []
    for n, (text, clauses, arretes) in enumerate(CD2, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-p1b2"})
        edges = []
        for src, dst, rel, marqueur in arretes:
            edges.append({"sources": [f"n{src:03d}"], "target": f"n{dst:03d}",
                          "relation": rel, "explicit": True, "confidence": 0.0,
                          "negated": False, "third": None,
                          "marker_token": _marker_id(toks, marqueur)})
        sentences.append({"id": f"p1b2{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": edges}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-control-dep-batch2", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver control_dep batch2.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = construire()
    blob = json.dumps(doc, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(blob)
    print(json.dumps({"phrases": len(doc["document"]["sentences"]),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                      "poids": 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
