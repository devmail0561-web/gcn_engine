#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Mini-campagne ciblée (décision gold-3000 partielle) : ternaire batch3 (42,
joint_prevent EXCLU : GELÉ table) + contrainte (17 : 23→40) + actes (15).
Même contrat que tb2 (spans calculés, thirds vérifiés, poids 0.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

# (texte, [(span, label, type)], (src, dst, relation, third|None))
TERN3 = [
("Si les abeilles disparaissent, un tiers des cultures dépérira.",
 [("les abeilles disparaissent", "abeilles disparues", "condition"), ("un tiers des cultures dépérira", "dépérissement asserté", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le pain lèvera si la pâte repose deux heures au chaud.",
 [("Le pain lèvera", "levée assertée", "processus"), ("la pâte repose deux heures au chaud", "repos de la pâte", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la neige tient, les pistes ouvriront samedi.",
 [("la neige tient", "neige tenue", "condition"), ("les pistes ouvriront samedi", "ouverture assertée", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les cloches sonneront si le mariage est célébré.",
 [("Les cloches sonneront", "sonnerie assertée", "evenement"), ("le mariage est célébré", "mariage célébré", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la rivière déborde, le moulin s'arrêtera.",
 [("la rivière déborde", "débordement", "condition"), ("le moulin s'arrêtera", "arrêt asserté", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le cidre sera bon si les pommes sont pressées à froid.",
 [("Le cidre sera bon", "cidre bon", "etat_local"), ("les pommes sont pressées à froid", "pressée à froid", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si les renforts tardent, la place tombera avant minuit.",
 [("les renforts tardent", "retard des renforts", "condition"), ("la place tombera avant minuit", "chute assertée", "evenement")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le feu couve si les braises restent chaudes.",
 [("Le feu couve", "feu couvant", "etat_local"), ("les braises restent chaudes", "braises chaudes", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si la garde veille, aucun intrus ne passera le fossé.",
 [("la garde veille", "garde vigilante", "condition"), ("aucun intrus ne passera le fossé", "fossé tenu", "etat_local")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La cargaison pourrira si la cale prend l'eau.",
 [("La cargaison pourrira", "pourrissement asserté", "processus"), ("la cale prend l'eau", "voie d'eau", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le serrurier tarde, la porte restera close.",
 [("le serrurier tarde", "retard du serrurier", "condition"), ("la porte restera close", "porte close", "etat_local")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le blé versera si l'orage éclate avant la moisson.",
 [("Le blé versera", "verse assertée", "processus"), ("l'orage éclate avant la moisson", "orage précoce", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("La forge chauffe le fer via le soufflet actionné au pied.",
 [("La forge chauffe", "forge chaude", "etat_local"), ("le soufflet actionné au pied", "soufflet (moyen)", "entite"), ("le fer", "fer", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le moulin tourne et moud la farine grâce à la roue immergée dans le bief.",
 [("Le moulin tourne", "moulin en marche", "processus"), ("la roue immergée dans le bief", "roue (moyen)", "entite"), ("la farine", "farine", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La cloche sonne l'angélus via la corde tirée par le bedeau.",
 [("La cloche sonne", "cloche qui sonne", "processus"), ("la corde tirée par le bedeau", "corde (moyen)", "entite"), ("l'angélus", "angélus", "evenement")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le pressoir écrase le raisin grâce au vérin actionné au levier.",
 [("Le pressoir écrase", "pressoir qui écrase", "processus"), ("au vérin actionné au levier", "vérin (moyen)", "entite"), ("le raisin", "raisin", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le lavoir blanchit le linge au moyen de la cendre dissoute.",
 [("Le lavoir blanchit", "linge blanchi", "processus"), ("la cendre dissoute", "cendre (moyen)", "entite"), ("le linge", "linge", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le four cuit le pain par l'intermédiaire des braises étalées.",
 [("Le four cuit", "four qui cuit", "processus"), ("des braises étalées", "braises (moyen)", "entite"), ("le pain", "pain", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La douve protège le château via un pont-levis relevé chaque nuit.",
 [("La douve protège", "douve qui protège", "processus"), ("un pont-levis relevé chaque nuit", "pont-levis (moyen)", "entite"), ("le château", "château", "etat_local")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le grenier reste sec contre la pluie grâce au chaume refait chaque automne.",
 [("Le grenier reste sec", "grenier sec", "etat_local"), ("au chaume refait chaque automne", "chaume (moyen)", "entite"), ("la pluie", "pluie", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La bergerie tient chaud contre le froid au moyen des murs en pierre sèche.",
 [("La bergerie tient chaud", "bergerie chaude", "etat_local"), ("des murs en pierre sèche", "murs (moyen)", "entite"), ("le froid", "froid", "etat_global")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le puits garde son eau contre la soif via une margelle haute.",
 [("Le puits garde son eau", "eau gardée", "etat_local"), ("une margelle haute", "margelle (moyen)", "entite"), ("la soif", "soif", "concept")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La grange résiste à l'effondrement par ses contreforts en chêne.",
 [("La grange résiste", "grange debout", "etat_local"), ("ses contreforts en chêne", "contreforts (moyen)", "entite"), ("l'effondrement", "effondrement", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La mare gèle et la glace porte les patineurs.",
 [("La mare gèle", "gel de la mare", "processus"), ("la glace", "glace (moyen)", "etat_local"), ("les patineurs", "patineurs portés", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le gel et la bise ont durci la terre.",
 [("Le gel", "gel", "etat_global"), ("la bise", "bise", "etat_global"), ("la terre", "terre durcie", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("La pluie et le vent ont couché les blés.",
 [("La pluie", "pluie", "evenement"), ("le vent", "vent", "etat_global"), ("les blés", "blés couchés", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("Le froid et la faim ont décimé le régiment.",
 [("Le froid", "froid", "etat_global"), ("la faim", "faim", "etat_global"), ("le régiment", "régiment décimé", "entite")],
 ([0, 1], 2, "joint_cause", None)),
("La rouille et l'usure ont rongé la faux.",
 [("La rouille", "rouille", "processus"), ("l'usure", "usure", "processus"), ("la faux", "faux rongée", "entite")],
 ([0, 1], 2, "joint_cause", None)),
("L'âge et la maladie ont affaibli le forgeron.",
 [("L'âge", "âge", "concept"), ("la maladie", "maladie", "etat_local"), ("le forgeron", "forgeron affaibli", "entite")],
 ([0, 1], 2, "joint_cause", None)),
("La poussière et l'humidité ont grippé le mécanisme.",
 [("La poussière", "poussière", "entite"), ("l'humidité", "humidité", "etat_global"), ("le mécanisme", "mécanisme grippé", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("Tout comme le meunier règle son moulin, le marin règle ses voiles.",
 [("le meunier règle son moulin", "meunier→moulin", "concept"), ("le marin règle ses voiles", "marin→voiles", "concept")],
 (0, 1, "analogy", None)),
("De même que la sage-femme veille l'accouchée, le berger veille l'agnelage.",
 [("la sage-femme veille l'accouchée", "sage-femme→accouchée", "concept"), ("le berger veille l'agnelage", "berger→agnelage", "concept")],
 (0, 1, "analogy", None)),
("Tout comme le boulanger pétrit la pâte, le maçon gâche le mortier.",
 [("le boulanger pétrit la pâte", "boulanger→pâte", "concept"), ("le maçon gâche le mortier", "maçon→mortier", "concept")],
 (0, 1, "analogy", None)),
("De même que le faucon fond sur sa proie, le brochet fond sur le gardon.",
 [("le faucon fond sur sa proie", "faucon→proie", "concept"), ("le brochet fond sur le gardon", "brochet→gardon", "concept")],
 (0, 1, "analogy", None)),
("Tout comme le laboureur trace son sillon, le scribe trace sa ligne.",
 [("le laboureur trace son sillon", "laboureur→sillon", "concept"), ("le scribe trace sa ligne", "scribe→ligne", "concept")],
 (0, 1, "analogy", None)),
("De même que l'abeille butine les fleurs, le collecteur glane les nouvelles.",
 [("l'abeille butine les fleurs", "abeille→fleurs", "concept"), ("le collecteur glane les nouvelles", "collecteur→nouvelles", "concept")],
 (0, 1, "analogy", None)),
("Sans le coq, la basse-cour serait restée endormie.",
 [("Sans le coq", "coq absent (irréel)", "condition"), ("la basse-cour serait restée endormie", "basse-cour endormie (irréel)", "etat_local")],
 (0, 1, "counterfactual", None)),
("Si la cloche avait sonné, le village se serait réveillé.",
 [("la cloche avait sonné", "cloche non sonnée (irréel)", "evenement"), ("le village se serait réveillé", "village endormi (irréel)", "etat_local")],
 (0, 1, "counterfactual", None)),
("Sans la lune, les braconniers auraient marché dans le noir.",
 [("Sans la lune", "lune absente (irréel)", "condition"), ("les braconniers auraient marché dans le noir", "marche aveugle (irréel)", "processus")],
 (0, 1, "counterfactual", None)),
("Si le meunier avait veillé, la roue n'aurait pas cassé.",
 [("le meunier avait veillé", "veille non faite (irréel)", "evenement"), ("la roue n'aurait pas cassé", "roue cassée (irréel)", "etat_local")],
 (0, 1, "counterfactual", None)),
("Sans le chien, le troupeau se serait dispersé dans la lande.",
 [("Sans le chien", "chien absent (irréel)", "condition"), ("le troupeau se serait dispersé", "dispersion hypothétique", "processus")],
 (0, 1, "counterfactual", None)),
("Si l'averse avait tenu, les foins auraient pourri sur pied.",
 [("l'averse avait tenu", "averse durable (irréel)", "evenement"), ("les foins auraient pourri", "foins pourris (irréel)", "etat_local")],
 (0, 1, "counterfactual", None)),
]

# (texte, [(span, label, type)], [(src, dst, relation, marqueur)])
CONTR2 = [
("Le ban seigneurial réserve la chasse au cerf.",
 [("Le ban seigneurial", "ban (règle)", "contrainte"), ("la chasse au cerf", "chasse réservée", "processus")],
 [(0, 1, "filter", "réserve")]),
("La dîme prélève un dixième des récoltes.",
 [("La dîme", "dîme (règle)", "contrainte"), ("un dixième des récoltes", "prélèvement", "processus")],
 [(0, 1, "control_dependency", "prélève")]),
("La corvée mobilise les villageois deux jours par mois.",
 [("La corvée", "corvée (règle)", "contrainte"), ("les villageois deux jours par mois", "villageois mobilisés", "entite")],
 [(0, 1, "control_dependency", "mobilise")]),
("Le droit de péage taxe les marchands au pont.",
 [("Le droit de péage", "péage (règle)", "contrainte"), ("les marchands au pont", "marchands taxés", "entite")],
 [(0, 1, "control_dependency", "taxe")]),
("L'interdit jeté par l'évêque frappe toute la paroisse.",
 [("L'interdit jeté par l'évêque", "interdit (règle)", "contrainte"), ("toute la paroisse", "paroisse frappée", "entite")],
 [(0, 1, "cause", "frappe")]),
("L'excommunication retranche le fidèle de la communauté.",
 [("L'excommunication", "excommunication (règle)", "contrainte"), ("le fidèle de la communauté", "fidèle retranché", "entite")],
 [(0, 1, "prevent", "retranche")]),
("Le jubilé efface les dettes tous les cinquante ans.",
 [("Le jubilé", "jubilé (règle)", "contrainte"), ("les dettes", "dettes effacées", "concept")],
 [(0, 1, "prevent", "efface")]),
("L'amnistie libère les prisonniers politiques.",
 [("L'amnistie", "amnistie (règle)", "contrainte"), ("les prisonniers politiques", "prisonniers libérés", "entite")],
 [(0, 1, "enable", "libère")]),
("Le sauf-conduit protège le messager des barrages.",
 [("Le sauf-conduit", "sauf-conduit (règle)", "contrainte"), ("le messager des barrages", "messager protégé", "entite")],
 [(0, 1, "prevent", "protège")]),
("La trêve suspend les hostilités jusqu'à Pâques.",
 [("La trêve", "trêve (règle)", "contrainte"), ("les hostilités jusqu'à Pâques", "hostilités suspendues", "processus")],
 [(0, 1, "prevent", "suspend")]),
("Le guet impose des rondes nocturnes aux bourgeois.",
 [("Le guet", "guet (règle)", "contrainte"), ("des rondes nocturnes aux bourgeois", "rondes imposées", "processus")],
 [(0, 1, "control_dependency", "impose")]),
("La taille frappe les roturiers chaque année.",
 [("La taille", "taille (règle)", "contrainte"), ("les roturiers chaque année", "roturiers frappés", "entite")],
 [(0, 1, "cause", "frappe")]),
("Le cens réserve le vote aux plus imposés.",
 [("Le cens", "cens (règle)", "contrainte"), ("le vote aux plus imposés", "vote réservé", "processus")],
 [(0, 1, "filter", "réserve")]),
("La gabelle renchérit le sel du royaume.",
 [("La gabelle", "gabelle (règle)", "contrainte"), ("le sel du royaume", "sel renchéri", "concept")],
 [(0, 1, "cause", "renchérit")]),
("Le monopole interdit toute concurrence étrangère.",
 [("Le monopole", "monopole (règle)", "contrainte"), ("toute concurrence étrangère", "concurrence interdite", "processus")],
 [(0, 1, "prevent", "interdit")]),
("L'octroi taxe les denrées à l'entrée de la ville.",
 [("L'octroi", "octroi (règle)", "contrainte"), ("les denrées à l'entrée de la ville", "denrées taxées", "entite")],
 [(0, 1, "control_dependency", "taxe")]),
("Le blocus continental affame les ports anglais.",
 [("Le blocus continental", "blocus (règle)", "contrainte"), ("les ports anglais", "ports affamés", "entite")],
 [(0, 1, "cause", "affame")]),
]

# Actes non-déclaratifs ciblés (impératifs/exclamatifs à claim causal asserté).
ACTES = [
("Fermez les vannes avant que la cuve ne déborde !",
 [("Fermez les vannes", "fermeture ordonnée", "processus"), ("la cuve ne déborde", "débordement évité", "evenement")],
 [(0, 1, "prevent", "Fermez")]),
("Ne laissez aucun outil traîner sur le chantier !",
 [("Ne laissez aucun outil traîner", "interdiction", "contrainte"), ("sur le chantier", "chantier", "etat_local")],
 [(0, 1, "prevent", "laissez")]),
("Sonnez l'alerte dès que la cloche tinte !",
 [("Sonnez l'alerte", "alerte ordonnée", "processus"), ("la cloche tinte", "cloche qui tinte", "processus")],
 [(0, 1, "sequence", "Sonnez")]),
("Quel Carnot ! Ce Carnot foudroie les blés et ruine les moissons !",
 [("Ce Carnot foudroie les blés", "foudre qui frappe", "evenement"), ("les moissons", "moissons ruinées", "etat_local")],
 [(0, 1, "cause", "foudroie")]),
("Quelle tempête ! Le vent arrache les tuiles et inonde les greniers !",
 [("Le vent arrache les tuiles", "vent qui arrache", "processus"), ("les greniers", "greniers inondés", "etat_local")],
 [(0, 1, "cause", "inonde")]),
("Tenez la corde, sinon la barque dérive vers les rochers !",
 [("Tenez la corde", "corde tenue", "processus"), ("la barque dérive vers les rochers", "dérive évitée", "evenement")],
 [(0, 1, "prevent", "Tenez")]),
("Poussez la charrue droit, sinon le sillon dévie vers le fossé !",
 [("Poussez la charrue droit", "ordre de labour", "processus"), ("le sillon dévie vers le fossé", "déviation évitée", "processus")],
 [(0, 1, "prevent", "Poussez")]),
("Restez groupés, sinon les loups disperseront le troupeau !",
 [("Restez groupés", "ordre de grouper", "processus"), ("les loups disperseront le troupeau", "dispersion évitée", "processus")],
 [(0, 1, "prevent", "Restez")]),
("Gardez les chiens attachés, sinon ils effraieront les brebis !",
 [("Gardez les chiens attachés", "ordre d'attacher", "processus"), ("ils effraieront les brebis", "effroi évité", "etat_local")],
 [(0, 1, "prevent", "Gardez")]),
("Chantez plus fort, que la vallée entière vous entende !",
 [("Chantez plus fort", "chant ordonné", "processus"), ("la vallée entière vous entende", "vallée qui entend", "etat_local")],
 [(0, 1, "enable", "entende")]),
("Frappez trois coups avant d'entrer dans la forge !",
 [("Frappez trois coups", "coups ordonnés", "processus"), ("d'entrer dans la forge", "entrée", "processus")],
 [(0, 1, "sequence", "avant")]),
("Méfiez-vous des eaux dormantes qui cachent des sables mouvants !",
 [("Méfiez-vous des eaux dormantes", "méfiance ordonnée", "processus"), ("des sables mouvants", "sables cachés", "etat_local")],
 [(0, 1, "prevent", "cachent")]),
("Courez annoncer la nouvelle au village avant la tombée du jour !",
 [("Courez annoncer la nouvelle au village", "ordre de courir", "processus"), ("la tombée du jour", "jour qui tombe", "etat_global")],
 [(0, 1, "sequence", "avant")]),
("Ne traversez jamais la lande sans lanterne après minuit !",
 [("Ne traversez jamais la lande sans lanterne", "interdiction", "contrainte"), ("après minuit", "minuit", "concept")],
 [(0, 1, "prevent", "traversez")]),
("Videz les seaux avant que l'orage n'éclate sur les toits !",
 [("Videz les seaux", "ordre de vider", "processus"), ("l'orage n'éclate sur les toits", "orage", "evenement")],
 [(0, 1, "sequence", "avant")]),
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
    assert len(TERN3) == 42 and len(CONTR2) == 17 and len(ACTES) == 15, (
        len(TERN3), len(CONTR2), len(ACTES))
    sentences = []
    n = 0
    for text, clauses, (src, dst, rel, third) in TERN3:
        n += 1
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-mc3"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src] if isinstance(src, list) else [f"n{src:03d}"]
        e["target"] = f"n{dst:03d}"
        e["third"] = {"role": third[0], "node": f"n{third[1]:03d}"} if third else None
        edges = [e]
        st = "declarative"
        sentences.append({"id": f"mc3{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": st, "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": edges}})
    for text, clauses, arretes in CONTR2 + ACTES:
        n += 1
        toks = _tokens(text)
        st = ("interrogative" if text.rstrip().endswith("?") else
              "exclamative" if text.rstrip().endswith("!") else "declarative")
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-mc3"})
        edges = []
        for src, dst, rel, marqueur in arretes:
            edges.append({"sources": [f"n{src:03d}"], "target": f"n{dst:03d}",
                          "relation": rel, "explicit": True, "confidence": 0.0,
                          "negated": False, "third": None,
                          "marker_token": _marker_id(toks, marqueur)})
        sentences.append({"id": f"mc3{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": st, "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": edges}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-minicampagne-3", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver mini-campagne 3.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = construire()
    blob = json.dumps(doc, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(blob)
    from collections import Counter
    rels = Counter(e["relation"]
                   for s in doc["document"]["sentences"] for e in s["cir"]["edges"])
    print(json.dumps({"phrases": len(doc["document"]["sentences"]),
                      "relations": dict(sorted(rels.items())),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                      "poids": 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
