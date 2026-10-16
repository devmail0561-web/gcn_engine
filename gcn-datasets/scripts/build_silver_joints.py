# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Silver joints : 20 joint_prevent + 20 joint_cause (vague J, gold-3000).

Même contrat que mc3/mc4a (registre pastoral, tokens whitespace, spans
calculés depuis les spans caractères, thirds None, poids 0.0).
Joint = 2 sources distinctes, même (cible, relation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

# (texte, [(span_clause, label, type)], (sources, cible, relation))
J: list = []
_P = "processus"
_E = "evenement"
_G = "etat_global"
_L = "etat_local"
_N = "entite"


def _j(text, c0, l0, t0, c1, l1, t1, c2, l2, t2, rel):
    J.append((text, [(c0, l0, t0), (c1, l1, t1), (c2, l2, t2)], ([0, 1], 2, rel, None)))


# --- joint_prevent × 20 ---
_j("La digue retient les eaux et la vanne ferme le bief : le pré ne sera pas inondé.",
   "La digue retient les eaux", "digue qui retient", _P,
   "la vanne ferme le bief", "vanne fermée", _P,
   "le pré ne sera pas inondé", "pré épargné", _L, "joint_prevent")
_j("Le gel scelle les sillons et la neige comble les ornières : la herse ne sortira pas.",
   "Le gel scelle les sillons", "gel qui scelle", _G,
   "la neige comble les ornières", "neige qui comble", _G,
   "la herse ne sortira pas", "herse au hangar", _L, "joint_prevent")
_j("La haie arrête le vent et le mur coupe la bise : le potager ne gèlera point.",
   "La haie arrête le vent", "haie brise-vent", _P,
   "le mur coupe la bise", "mur coupe-bise", _P,
   "le potager ne gèlera point", "potager épargné", _L, "joint_prevent")
_j("Le garde barre le pont et le meunier cadenasse l'écluse : les maraudeurs ne passeront pas.",
   "Le garde barre le pont", "garde qui barre", _P,
   "le meunier cadenasse l'écluse", "meunier qui cadenasse", _P,
   "les maraudeurs ne passeront pas", "maraudeurs bloqués", _L, "joint_prevent")
_j("La sécheresse tarit le puits et le maire rationne la fontaine : le lavoir restera vide.",
   "La sécheresse tarit le puits", "sécheresse qui tarit", _G,
   "le maire rationne la fontaine", "maire qui rationne", _P,
   "le lavoir restera vide", "lavoir vide", _L, "joint_prevent")
_j("La grêle a ravagé les blés et la rouille a gâté les avoines : le grenier restera vide.",
   "La grêle a ravagé les blés", "grêle ravageuse", _E,
   "la rouille a gâté les avoines", "rouille qui gâte", _P,
   "le grenier restera vide", "grenier vide", _L, "joint_prevent")
_j("Le loup décime le troupeau et la gale affaiblit les bêtes : la foire n'aura point lieu.",
   "Le loup décime le troupeau", "loup qui décime", _P,
   "la gale affaiblit les bêtes", "gale qui affaiblit", _G,
   "la foire n'aura point lieu", "foire annulée", _L, "joint_prevent")
_j("La rivière a emporté le pont et la boue a comblé le gué : le convoi ne traversera pas.",
   "La rivière a emporté le pont", "rivière qui emporte", _E,
   "la boue a comblé le gué", "boue qui comble", _G,
   "le convoi ne traversera pas", "convoi bloqué", _L, "joint_prevent")
_j("Le froid a gelé l'encre et l'obscurité voile la page : le greffier n'écrira point.",
   "Le froid a gelé l'encre", "froid qui gèle", _G,
   "l'obscurité voile la page", "obscurité qui voile", _G,
   "le greffier n'écrira point", "greffier empêché", _L, "joint_prevent")
_j("La disette vide les greniers et l'épizootie décime les étables : le marché restera désert.",
   "La disette vide les greniers", "disette qui vide", _G,
   "l'épizootie décime les étables", "épizootie qui décime", _E,
   "le marché restera désert", "marché désert", _L, "joint_prevent")
_j("Le vent a éteint la lanterne et la nuit aveugle le sentier : le voyageur ne partira pas.",
   "Le vent a éteint la lanterne", "vent qui éteint", _E,
   "la nuit aveugle le sentier", "nuit qui aveugle", _G,
   "le voyageur ne partira pas", "voyageur cloué", _L, "joint_prevent")
_j("La crue submerge les prés et le limon colmate les fossés : les bêtes ne pâtureront point.",
   "La crue submerge les prés", "crue qui submerge", _E,
   "le limon colmate les fossés", "limon qui colmate", _P,
   "les bêtes ne pâtureront point", "bêtes sans pâture", _L, "joint_prevent")
_j("Le fisc saisit la récolte et l'usurier prend le bétail : le fermier ne sèmera plus.",
   "Le fisc saisit la récolte", "fisc qui saisit", _P,
   "l'usurier prend le bétail", "usurier qui prend", _P,
   "le fermier ne sèmera plus", "fermier ruiné", _L, "joint_prevent")
_j("La neige ensevelit les toits et le verglas verrouille les portes : le village restera isolé.",
   "La neige ensevelit les toits", "neige qui ensevelit", _G,
   "le verglas verrouille les portes", "verglas qui verrouille", _G,
   "le village restera isolé", "village isolé", _L, "joint_prevent")
_j("Le charançon perce les sacs et l'humidité pourrit le grain : la réserve ne tiendra pas l'hiver.",
   "Le charançon perce les sacs", "charançon qui perce", _P,
   "l'humidité pourrit le grain", "humidité qui pourrit", _G,
   "la réserve ne tiendra pas l'hiver", "réserve perdue", _L, "joint_prevent")
_j("L'incendie dévore la grange et la foudre fend le clocher : la fête est annulée.",
   "L'incendie dévore la grange", "incendie qui dévore", _E,
   "la foudre fend le clocher", "foudre qui fend", _E,
   "la fête est annulée", "fête annulée", _L, "joint_prevent")
_j("Le renard pille le poulailler et la fouine saigne les poussins : le fermier ne vendra point d'œufs.",
   "Le renard pille le poulailler", "renard qui pille", _P,
   "la fouine saigne les poussins", "fouine qui saigne", _P,
   "le fermier ne vendra point d'œufs", "fermier sans œufs", _L, "joint_prevent")
_j("La sécheresse fend les labours et les criquets rasent les chaumes : le pain manquera à Noël.",
   "La sécheresse fend les labours", "sécheresse qui fend", _G,
   "les criquets rasent les chaumes", "criquets qui rasent", _P,
   "le pain manquera à Noël", "pain qui manque", _L, "joint_prevent")
_j("Le notaire scelle le testament et le juge gèle la succession : les héritiers n'auront rien.",
   "Le notaire scelle le testament", "notaire qui scelle", _P,
   "le juge gèle la succession", "juge qui gèle", _P,
   "les héritiers n'auront rien", "héritiers bredouilles", _L, "joint_prevent")
_j("La guerre réquisitionne les chevaux et le blocus affame les villes : le moulin ne tournera plus.",
   "La guerre réquisitionne les chevaux", "guerre qui réquisitionne", _E,
   "le blocus affame les villes", "blocus qui affame", _G,
   "le moulin ne tournera plus", "moulin arrêté", _L, "joint_prevent")

# --- joint_cause × 20 ---
_j("La pluie gonfle le ruisseau et la fonte des neiges grossit la rivière : le pont sera emporté.",
   "La pluie gonfle le ruisseau", "pluie qui gonfle", _E,
   "la fonte des neiges grossit la rivière", "fonte qui grossit", _G,
   "le pont sera emporté", "pont emporté", _L, "joint_cause")
_j("Le soleil mûrit les blés et la rosée engraisse les prés : la moisson sera bonne.",
   "Le soleil mûrit les blés", "soleil qui mûrit", _G,
   "la rosée engraisse les prés", "rosée qui engraisse", _G,
   "la moisson sera bonne", "bonne moisson", _L, "joint_cause")
_j("Le forgeron bat le fer et le meunier moud le grain : le village prospère.",
   "Le forgeron bat le fer", "forgeron qui bat", _P,
   "le meunier moud le grain", "meunier qui moud", _P,
   "le village prospère", "village prospère", _L, "joint_cause")
_j("La cloche sonne l'angélus et le coq chante l'aube : les champs s'éveillent.",
   "La cloche sonne l'angélus", "cloche qui sonne", _P,
   "le coq chante l'aube", "coq qui chante", _P,
   "les champs s'éveillent", "champs qui s'éveillent", _L, "joint_cause")
_j("Le berger rassemble le troupeau et le chien rabat les brebis : le parc se remplit.",
   "Le berger rassemble le troupeau", "berger qui rassemble", _P,
   "le chien rabat les brebis", "chien qui rabat", _P,
   "le parc se remplit", "parc qui se remplit", _L, "joint_cause")
_j("La sage-femme veille l'accouchée et le médecin surveille la fièvre : l'enfant vivra.",
   "La sage-femme veille l'accouchée", "sage-femme qui veille", _P,
   "le médecin surveille la fièvre", "médecin qui surveille", _P,
   "l'enfant vivra", "enfant qui vivra", _L, "joint_cause")
_j("Le maçon scelle les pierres et le charpentier lève la charpente : la maison s'achève.",
   "Le maçon scelle les pierres", "maçon qui scelle", _P,
   "le charpentier lève la charpente", "charpentier qui lève", _P,
   "la maison s'achève", "maison qui s'achève", _L, "joint_cause")
_j("L'instituteur instruit les enfants et le curé prêche la morale : le village s'assagit.",
   "L'instituteur instruit les enfants", "instituteur qui instruit", _P,
   "le curé prêche la morale", "curé qui prêche", _P,
   "le village s'assagit", "village qui s'assagit", _L, "joint_cause")
_j("Le vent pousse les nuages et la pluie arrose les semis : les vergers fleurissent.",
   "Le vent pousse les nuages", "vent qui pousse", _G,
   "la pluie arrose les semis", "pluie qui arrose", _E,
   "les vergers fleurissent", "vergers qui fleurissent", _L, "joint_cause")
_j("La mère pétrit la pâte et la fille attise le four : le pain lèvera.",
   "La mère pétrit la pâte", "mère qui pétrit", _P,
   "la fille attise le four", "fille qui attise", _P,
   "le pain lèvera", "pain qui lèvera", _L, "joint_cause")
_j("Le jardinier taille les rosiers et l'apprenti arrose les plates-bandes : le jardin s'embellit.",
   "Le jardinier taille les rosiers", "jardinier qui taille", _P,
   "l'apprenti arrose les plates-bandes", "apprenti qui arrose", _P,
   "le jardin s'embellit", "jardin qui s'embellit", _L, "joint_cause")
_j("Le maréchal ferre les chevaux et le bourrelier coud les harnais : la diligence partira.",
   "Le maréchal ferre les chevaux", "maréchal qui ferre", _P,
   "le bourrelier coud les harnais", "bourrelier qui coud", _P,
   "la diligence partira", "diligence qui partira", _L, "joint_cause")
_j("L'abeille butine les fleurs et le papillon porte le pollen : le verger donnera.",
   "L'abeille butine les fleurs", "abeille qui butine", _P,
   "le papillon porte le pollen", "papillon qui porte", _P,
   "le verger donnera", "verger qui donnera", _L, "joint_cause")
_j("Le cantonnier empierre le chemin et le garde pose les bornes : la route s'ouvre.",
   "Le cantonnier empierre le chemin", "cantonnier qui empierre", _P,
   "le garde pose les bornes", "garde qui pose", _P,
   "la route s'ouvre", "route qui s'ouvre", _L, "joint_cause")
_j("La cuisinière sale le pot et la servante écume le bouillon : la soupe est prête.",
   "La cuisinière sale le pot", "cuisinière qui sale", _P,
   "la servante écume le bouillon", "servante qui écume", _P,
   "la soupe est prête", "soupe prête", _L, "joint_cause")
_j("Le pêcheur raccommode les filets et le mousse amorce les lignes : la pêche sera bonne.",
   "Le pêcheur raccommode les filets", "pêcheur qui raccommode", _P,
   "le mousse amorce les lignes", "mousse qui amorce", _P,
   "la pêche sera bonne", "bonne pêche", _L, "joint_cause")
_j("Le tisserand tend la chaîne et la fileuse tourne le fuseau : la toile s'allonge.",
   "Le tisserand tend la chaîne", "tisserand qui tend", _P,
   "la fileuse tourne le fuseau", "fileuse qui tourne", _P,
   "la toile s'allonge", "toile qui s'allonge", _L, "joint_cause")
_j("Le vigneron taille la vigne et le tonnelier cerclera les fûts : le vin vieillira.",
   "Le vigneron taille la vigne", "vigneron qui taille", _P,
   "le tonnelier cerclera les fûts", "tonnelier qui cerclera", _P,
   "le vin vieillira", "vin qui vieillira", _L, "joint_cause")
_j("Le sonneur tire les cordes et les enfants agitent les rameaux : la procession s'ébranle.",
   "Le sonneur tire les cordes", "sonneur qui tire", _P,
   "les enfants agitent les rameaux", "enfants qui agitent", _P,
   "la procession s'ébranle", "procession qui s'ébranle", _L, "joint_cause")
_j("Le laboureur ouvre les sillons et le semeur répand le grain : la terre promet.",
   "Le laboureur ouvre les sillons", "laboureur qui ouvre", _P,
   "le semeur répand le grain", "semeur qui répand", _P,
   "la terre promet", "terre qui promet", _L, "joint_cause")


def _tokens(text: str) -> list[dict]:
    toks, cur = [], 0
    for mot in text.split(" "):
        i = text.find(mot, cur)
        toks.append({"id": len(toks) + 1, "form": mot})
        cur = i + len(mot)
    return toks


def _span_tokens(text: str, toks: list[dict], clause: str) -> list[int]:
    bounds, p = [], 0
    for t in toks:
        i = text.find(t["form"], p)
        bounds.append((i, i + len(t["form"])))
        p = i + len(t["form"])
    s = text.find(clause)
    assert s >= 0, clause
    e = s + len(clause)
    out = [k for k, t in enumerate(toks, 1) if s <= bounds[k - 1][0] < e]
    assert out, clause
    while len(out) > 1 and re.fullmatch(r"[.,;:!?]+", toks[out[-1] - 1]["form"]):
        out.pop()
    return [out[0], out[-1]]


def construire() -> dict:
    assert len(J) == 40, len(J)
    sentences = []
    for n, (text, clauses, (src, dst, rel, third)) in enumerate(J, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-joints"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src]
        e["target"] = f"n{dst:03d}"
        e["third"] = None
        sentences.append({"id": f"j{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "pastoral",
                          "cir": {"nodes": nodes, "edges": [e]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-joints", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver joints (vague J).")
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
