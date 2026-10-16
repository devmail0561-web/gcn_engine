#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Silver ternaire batch2 + contrainte (P1b-pattern, auteur assistant).

39 ternaires (conditional_cause 6, conditional_prevent 5, mediated_cause 6,
mediated_prevent 5, joint_cause 6, analogy 5, counterfactual 6 ; joint_prevent
EXCLU : GELÉ table-11-19, précision 1-2/5) + 20 contrainte (nœuds `contrainte`,
classe à 3 occurrences → N_min 20 visé).
Déclencheurs : table-11-19.md v1 validée (si+causale, grâce à/via, combinaison/
à la fois, de même que/tout comme, sans+aurait, si+PQP+aurait). Exclusions
appliquées : tel que énumératif, conditionnel journalistique, code-buckets.
Statut : SILVER poids 0.0, hors DATA/train. Promotion = validation + IC95.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

# (texte, [(span, label, type)], (src, dst, relation, third|None))
# third = (role, clause_idx) ; src/dst = indices clauses (ou listes pour joint).
TERN = [
("Si la digue cède, le village sera inondé.",
 [("la digue cède", "digue qui cède", "condition"), ("le village sera inondé", "inondation assertée", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le blé lève si la pluie d'automne est abondante.",
 [("Le blé lève", "levée du blé", "processus"), ("la pluie d'automne est abondante", "pluie abondante", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si le vent tourne au nord, les températures chuteront sous zéro.",
 [("le vent tourne au nord", "vent au nord", "condition"), ("les températures chuteront sous zéro", "chute assertée", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("La récolte pourrit si l'humidité persiste dix jours.",
 [("La récolte pourrit", "pourrissement", "processus"), ("l'humidité persiste dix jours", "humidité persistante", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la banque relève ses taux, le crédit immobilier ralentira.",
 [("la banque relève ses taux", "hausse des taux", "condition"), ("le crédit immobilier ralentira", "ralentissement asserté", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les fraises mûrissent si les gelées d'avril les épargnent.",
 [("Les fraises mûrissent", "maturation", "processus"), ("les gelées d'avril les épargnent", "absence de gel", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si le gardien dort, les cambrioleurs entreront sans bruit.",
 [("le gardien dort", "gardien endormi", "condition"), ("les cambrioleurs entreront sans bruit", "intrusion assertée", "evenement")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le barrage tiendra si les vannes restent ouvertes.",
 [("Le barrage tiendra", "barrage tenu", "etat_local"), ("les vannes restent ouvertes", "vannes ouvertes", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le témoin se tait, l'enquête s'enlisera.",
 [("le témoin se tait", "témoin muet", "condition"), ("l'enquête s'enlisera", "enlisement asserté", "processus")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La forêt survivra si les pompiers contiennent le flanc ouest.",
 [("La forêt survivra", "forêt sauvée", "etat_local"), ("les pompiers contiennent le flanc ouest", "flanc contenu", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si la digue est surveillée, la crue sera contenue.",
 [("la digue est surveillée", "surveillance", "condition"), ("la crue sera contenue", "crue contenue", "evenement")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La ville a tenu contre la crue grâce aux barrages de sacs de sable.",
 [("La ville a tenu", "ville tenue", "etat_local"), ("aux barrages de sacs de sable", "barrages (moyen)", "entite"), ("la crue", "crue", "evenement")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le malade guérit de l'infection via un protocole d'antibiotiques ciblés.",
 [("Le malade guérit", "guérison", "processus"), ("un protocole d'antibiotiques ciblés", "protocole (moyen)", "concept"), ("l'infection", "infection", "etat_local")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("L'usine tourne grâce à sa turbine et assure la production.",
 [("L'usine tourne", "usine en marche", "processus"), ("sa turbine", "turbine (moyen)", "entite"), ("la production", "production", "processus")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le village survit à la famine par l'intermédiaire des convois humanitaires.",
 [("Le village survit", "survie du village", "etat_local"), ("des convois humanitaires", "convois (moyen)", "entite"), ("la famine", "famine", "evenement")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La thèse dissipe l'ignorance grâce aux archives municipales inédites.",
 [("La thèse dissipe l'ignorance", "ignorance dissipée", "processus"), ("aux archives municipales inédites", "archives (moyen)", "entite"), ("l'ignorance", "ignorance", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le chantier résorbe le retard via la préfabrication des modules.",
 [("Le chantier résorbe le retard", "retard résorbé", "processus"), ("la préfabrication des modules", "préfabrication (moyen)", "processus"), ("le retard", "retard", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("L'épidémie recule et stoppe la contagion grâce à la vaccination des soignants.",
 [("L'épidémie recule", "épidémie en recul", "processus"), ("la vaccination des soignants", "vaccination (moyen)", "processus"), ("la contagion", "contagion stoppée", "processus")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La cave reste sèche et repousse l'inondation via un drainage périphérique continu.",
 [("La cave reste sèche", "cave sèche", "etat_local"), ("un drainage périphérique continu", "drainage (moyen)", "processus"), ("l'inondation", "inondation repoussée", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La récolte est sauvée de la sécheresse grâce à l'irrigation nocturne.",
 [("La récolte est sauvée", "récolte sauvée", "etat_local"), ("l'irrigation nocturne", "irrigation (moyen)", "processus"), ("la sécheresse", "sécheresse", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le réseau tient contre la panne par l'intermédiaire des serveurs de secours.",
 [("Le réseau tient", "réseau tenu", "etat_local"), ("des serveurs de secours", "serveurs (moyen)", "entite"), ("la panne", "panne", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le verger survit au gel grâce aux voiles d'hivernage.",
 [("Le verger survit", "verger sauf", "etat_local"), ("aux voiles d'hivernage", "voiles (moyen)", "entite"), ("au gel", "gel", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La sécheresse et le vent ont attisé l'incendie.",
 [("La sécheresse", "sécheresse", "etat_global"), ("le vent", "vent", "etat_global"), ("l'incendie", "incendie attisé", "evenement")],
 ([0, 1], 2, "joint_cause", None)),
("Le froid et l'humidité ont rouillé les outils.",
 [("Le froid", "froid", "etat_global"), ("l'humidité", "humidité", "etat_global"), ("les outils", "outils rouillés", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("La grève et la panne ont paralysé la gare.",
 [("La grève", "grève", "evenement"), ("la panne", "panne", "evenement"), ("la gare", "gare paralysée", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("Le sel et le gel ont fissuré la chaussée.",
 [("Le sel", "sel", "entite"), ("le gel", "gel", "etat_global"), ("la chaussée", "chaussée fissurée", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("La nuit et le brouillard ont égaré les randonneurs.",
 [("La nuit", "nuit", "etat_global"), ("le brouillard", "brouillard", "etat_global"), ("les randonneurs", "randonneurs égarés", "entite")],
 ([0, 1], 2, "joint_cause", None)),
("L'orage et la grêle ont haché les vignes.",
 [("L'orage", "orage", "evenement"), ("la grêle", "grêle", "evenement"), ("les vignes", "vignes hachées", "etat_local")],
 ([0, 1], 2, "joint_cause", None)),
("De même que le loup suit le troupeau, le fisc suit les fraudeurs.",
 [("le loup suit le troupeau", "loup→troupeau", "concept"), ("le fisc suit les fraudeurs", "fisc→fraudeurs", "concept")],
 (0, 1, "analogy", None)),
("Tout comme la marée use la falaise, le temps use les souvenirs.",
 [("la marée use la falaise", "marée→falaise", "concept"), ("le temps use les souvenirs", "temps→souvenirs", "concept")],
 (0, 1, "analogy", None)),
("De même que la forge durcit le fer, l'effort durcit le caractère.",
 [("la forge durcit le fer", "forge→fer", "concept"), ("l'effort durcit le caractère", "effort→caractère", "concept")],
 (0, 1, "analogy", None)),
("Tout comme le jardinier taille les rosiers, le professeur corrige les copies.",
 [("le jardinier taille les rosiers", "jardinier→rosiers", "concept"), ("le professeur corrige les copies", "professeur→copies", "concept")],
 (0, 1, "analogy", None)),
("De même que la chouette voit la nuit, le sonar voit sous l'eau.",
 [("la chouette voit la nuit", "chouette→nuit", "concept"), ("le sonar voit sous l'eau", "sonar→eau", "concept")],
 (0, 1, "analogy", None)),
("Sans le phare, les navires se seraient brisés sur les récifs.",
 [("Sans le phare", "phare absent (irréel)", "condition"), ("les navires se seraient brisés", "naufrage hypothétique", "evenement")],
 (0, 1, "counterfactual", None)),
("Si le pont avait tenu, les renforts seraient arrivés avant l'aube.",
 [("le pont avait tenu", "pont tenu (irréel)", "evenement"), ("les renforts seraient arrivés", "renforts non arrivés (irréel)", "evenement")],
 (0, 1, "counterfactual", None)),
("Sans les vaccins, la variole ravagerait encore des continents.",
 [("Sans les vaccins", "vaccins absents (irréel)", "condition"), ("la variole ravagerait", "ravage hypothétique", "evenement")],
 (0, 1, "counterfactual", None)),
("Si la sentinelle avait veillé, l'attaque aurait échoué.",
 [("la sentinelle avait veillé", "veille non faite (irréel)", "evenement"), ("l'attaque aurait échoué", "attaque réussie (irréel)", "evenement")],
 (0, 1, "counterfactual", None)),
("Sans les digues, la mer aurait repris les polders.",
 [("Sans les digues", "digues absentes (irréel)", "condition"), ("la mer aurait repris", "submersion hypothétique", "evenement")],
 (0, 1, "counterfactual", None)),
("Si l'orage avait éclaté plus tôt, la moisson aurait été perdue.",
 [("l'orage avait éclaté plus tôt", "orage précoce (irréel)", "evenement"), ("la moisson aurait été perdue", "moisson perdue (irréel)", "evenement")],
 (0, 1, "counterfactual", None)),
]

# (texte, [(span, label, type)], [(src, dst, relation, marqueur)]) — nœud contrainte présent.
CONTR = [
("Le couvre-feu interdit toute sortie après vingt et une heures.",
 [("Le couvre-feu", "couvre-feu (règle)", "contrainte"), ("toute sortie après vingt et une heures", "sorties interdites", "processus")],
 [(0, 1, "control_dependency", "interdit")]),
("Le règlement intérieur impose le port du badge.",
 [("Le règlement intérieur", "règlement (règle)", "contrainte"), ("le port du badge", "port imposé", "processus")],
 [(0, 1, "control_dependency", "impose")]),
("La loi oblige les commerces à fermer le dimanche.",
 [("La loi", "loi (règle)", "contrainte"), ("les commerces à fermer le dimanche", "fermeture obligée", "processus")],
 [(0, 1, "control_dependency", "oblige")]),
("L'interdiction de fumer protège les non-fumeurs.",
 [("L'interdiction de fumer", "interdiction (règle)", "contrainte"), ("les non-fumeurs", "non-fumeurs protégés", "entite")],
 [(0, 1, "prevent", "protège")]),
("Le quota de pêche préserve les stocks de cabillaud.",
 [("Le quota de pêche", "quota (règle)", "contrainte"), ("les stocks de cabillaud", "stocks préservés", "etat_local")],
 [(0, 1, "prevent", "préserve")]),
("L'obligation du masque freinait la contagion.",
 [("L'obligation du masque", "obligation (règle)", "contrainte"), ("la contagion", "contagion freinée", "processus")],
 [(0, 1, "prevent", "freinait")]),
("La consigne de tri réduit les déchets incinérés.",
 [("La consigne de tri", "consigne (règle)", "contrainte"), ("les déchets incinérés", "déchets réduits", "etat_local")],
 [(0, 1, "prevent", "réduit")]),
("Le permis à points sanctionne les chauffards.",
 [("Le permis à points", "permis (règle)", "contrainte"), ("les chauffards", "chauffards sanctionnés", "entite")],
 [(0, 1, "control_dependency", "sanctionne")]),
("La limitation à quatre-vingts punit les excès de vitesse.",
 [("La limitation à quatre-vingts", "limitation (règle)", "contrainte"), ("les excès de vitesse", "excès punis", "processus")],
 [(0, 1, "control_dependency", "punit")]),
("L'astreinte judiciaire force le débiteur à payer.",
 [("L'astreinte judiciaire", "astreinte (règle)", "contrainte"), ("le débiteur à payer", "paiement forcé", "processus")],
 [(0, 1, "control_dependency", "force")]),
("Le couvre-feu dépend du préfet et de la situation sanitaire.",
 [("Le couvre-feu", "couvre-feu (règle)", "contrainte"), ("du préfet et de la situation sanitaire", "préfet et situation", "entite")],
 [(0, 1, "data_dependency", "dépend")]),
("La validité du pass dépend du décret en vigueur.",
 [("La validité du pass", "validité du pass", "concept"), ("du décret en vigueur", "décret (règle)", "contrainte")],
 [(0, 1, "data_dependency", "dépend")]),
("Le confinement n'autorisait que les sorties essentielles.",
 [("Le confinement", "confinement (règle)", "contrainte"), ("les sorties essentielles", "sorties admises", "processus")],
 [(0, 1, "filter", "que")]),
("Seuls les véhicules autorisés franchissent le barrage.",
 [("les véhicules autorisés", "véhicules autorisés", "entite"), ("le barrage", "barrage (règle)", "contrainte")],
 [(0, 1, "filter", "Seuls")]),
("La zone interdite exclut tout promeneur.",
 [("La zone interdite", "zone (règle)", "contrainte"), ("tout promeneur", "promeneurs exclus", "entite")],
 [(0, 1, "filter", "exclut")]),
("Le péage conditionne l'accès à l'autoroute.",
 [("Le péage", "péage (règle)", "contrainte"), ("l'accès à l'autoroute", "accès conditionné", "processus")],
 [(0, 1, "condition", "conditionne")]),
("Le visa conditionne l'entrée sur le territoire.",
 [("Le visa", "visa (règle)", "contrainte"), ("l'entrée sur le territoire", "entrée conditionnée", "processus")],
 [(0, 1, "condition", "conditionne")]),
("L'embargo affame la population assiégée.",
 [("L'embargo", "embargo (règle)", "contrainte"), ("la population assiégée", "population affamée", "entite")],
 [(0, 1, "cause", "affame")]),
("Le blocus ruine le commerce du port.",
 [("Le blocus", "blocus (règle)", "contrainte"), ("le commerce du port", "commerce ruiné", "processus")],
 [(0, 1, "cause", "ruine")]),
("La grève bloque les trains en gare.",
 [("La grève", "grève (contrainte collective)", "contrainte"), ("les trains en gare", "trains bloqués", "etat_local")],
 [(0, 1, "prevent", "bloque")]),
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
    assert len(TERN) == 39 and len(CONTR) == 20, (len(TERN), len(CONTR))
    sentences = []
    n = 0
    for text, clauses, (src, dst, rel, third) in TERN:
        n += 1
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-tb2"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src] if isinstance(src, list) else [f"n{src:03d}"]
        e["target"] = f"n{dst:03d}"
        e["third"] = {"role": third[0], "node": f"n{third[1]:03d}"} if third else None
        # Pas de marker_token (comme batch1) : thirds portent la relation.
        sentences.append({"id": f"tb2{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": [e]}})
    for text, clauses, arretes in CONTR:
        n += 1
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-tb2"})
        edges = []
        for src, dst, rel, marqueur in arretes:
            edges.append({"sources": [f"n{src:03d}"], "target": f"n{dst:03d}",
                          "relation": rel, "explicit": True, "confidence": 0.0,
                          "negated": False, "third": None,
                          "marker_token": _marker_id(toks, marqueur)})
        sentences.append({"id": f"tb2{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": [edges[0]]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-ternaire-contrainte-b2", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver ternaire batch2 + contrainte.")
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
