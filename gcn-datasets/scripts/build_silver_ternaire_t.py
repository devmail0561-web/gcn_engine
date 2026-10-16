# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Silver ternaire vague T : 10 conditional_prevent + 10 mediated_prevent
+ 10 mediated_cause + 10 conditional_cause (gold-3000).

Même contrat que mc4a (registre pastoral, spans calculés, thirds vérifiés,
poids 0.0). Conditional : Si-clause (Condition) + effet. Mediated :
source → cible via nœud Mediator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

T: list = []


def _cprev(text, c0, l0, c1, l1):
    T.append((text, [(c0, l0, "condition"), (c1, l1, "etat_local")],
              (0, 1, "conditional_prevent", ("Condition", 0))))


def _ccaus(text, c0, l0, c1, l1, t1):
    T.append((text, [(c0, l0, "condition"), (c1, l1, t1)],
              (0, 1, "conditional_cause", ("Condition", 0))))


def _mprev(text, c0, l0, t0, c1, l1, t1, c2, l2, t2):
    T.append((text, [(c0, l0, t0), (c1, l1, t1), (c2, l2, t2)],
              (0, 2, "mediated_prevent", ("Mediator", 1))))


def _mcaus(text, c0, l0, t0, c1, l1, t1, c2, l2, t2):
    T.append((text, [(c0, l0, t0), (c1, l1, t1), (c2, l2, t2)],
              (0, 2, "mediated_cause", ("Mediator", 1))))


# --- conditional_prevent × 10 ---
_cprev("Si la meule tourne trop vite, la farine chauffera et s'abîmera.",
       "Si la meule tourne trop vite", "meule trop vive",
       "la farine chauffera et s'abîmera", "farine abîmée")
_cprev("Si le gel surprend les bourgeons, la récolte sera perdue.",
       "Si le gel surprend les bourgeons", "gel sur bourgeons",
       "la récolte sera perdue", "récolte perdue")
_cprev("Si la digue cède pendant la crue, le village sera évacué.",
       "Si la digue cède pendant la crue", "digue qui cède",
       "le village sera évacué", "village évacué")
_cprev("Si le loup rôde autour du parc, les agneaux resteront à l'étable.",
       "Si le loup rôde autour du parc", "loup qui rôde",
       "les agneaux resteront à l'étable", "agneaux à l'étable")
_cprev("Si l'orage éclate avant la fenaison, le foin pourrira sur pied.",
       "Si l'orage éclate avant la fenaison", "orage avant fenaison",
       "le foin pourrira sur pied", "foin pourri")
_cprev("Si la rivière déborde en mars, les semailles seront reportées.",
       "Si la rivière déborde en mars", "rivière qui déborde",
       "les semailles seront reportées", "semailles reportées")
_cprev("Si le feu prend à la grange, les bêtes seront lâchées dans le pré.",
       "Si le feu prend à la grange", "feu à la grange",
       "les bêtes seront lâchées dans le pré", "bêtes lâchées")
_cprev("Si la neige bloque le col, le courrier passera par la vallée.",
       "Si la neige bloque le col", "neige qui bloque",
       "le courrier passera par la vallée", "courrier dévié")
_cprev("Si le puits tarit en août, l'eau sera rationnée au village.",
       "Si le puits tarit en août", "puits qui tarit",
       "l'eau sera rationnée au village", "eau rationnée")
_cprev("Si le vent tourne au nord, les marins cargueront les voiles.",
       "Si le vent tourne au nord", "vent qui tourne",
       "les marins cargueront les voiles", "voiles carguées")

# --- conditional_cause × 10 ---
_ccaus("Si la pluie revient avant mai, les prés reverdiront.",
       "Si la pluie revient avant mai", "pluie avant mai",
       "les prés reverdiront", "prés reverdis", "etat_local")
_ccaus("Si le soleil chauffe la serre, les semis lèveront plus tôt.",
       "Si le soleil chauffe la serre", "soleil sur serre",
       "les semis lèveront plus tôt", "semis précoces", "etat_local")
_ccaus("Si le forgeron reçoit du charbon, la forge ronflera tout l'hiver.",
       "Si le forgeron reçoit du charbon", "charbon reçu",
       "la forge ronflera tout l'hiver", "forge qui ronfle", "processus")
_ccaus("Si la foire attire les maquignons, les prix monteront.",
       "Si la foire attire les maquignons", "foire attirante",
       "les prix monteront", "prix qui montent", "etat_local")
_ccaus("Si le ruisseau grossit au printemps, le moulin tournera jour et nuit.",
       "Si le ruisseau grossit au printemps", "ruisseau grossi",
       "le moulin tournera jour et nuit", "moulin qui tourne", "processus")
_ccaus("Si la terre repose un an, elle rendra le double.",
       "Si la terre repose un an", "terre au repos",
       "elle rendra le double", "double rendement", "etat_local")
_ccaus("Si les abeilles reviennent au verger, les pommiers donneront.",
       "Si les abeilles reviennent au verger", "abeilles revenues",
       "les pommiers donneront", "pommiers qui donnent", "processus")
_ccaus("Si le sel ne manque point, le beurre se gardera jusqu'au printemps.",
       "Si le sel ne manque point", "sel présent",
       "le beurre se gardera jusqu'au printemps", "beurre gardé", "etat_local")
_ccaus("Si le chemin est empierré, les charrettes passeront même en hiver.",
       "Si le chemin est empierré", "chemin empierré",
       "les charrettes passeront même en hiver", "charrettes qui passent", "processus")
_ccaus("Si la cloche sonne clair, les enfants accourront au catéchisme.",
       "Si la cloche sonne clair", "cloche claire",
       "les enfants accourront au catéchisme", "enfants qui accourent", "processus")

# --- mediated_prevent × 10 ---
_mprev("La muraille arrête les pillards grâce au guet posté sur la tour : les pillards rebroussent chemin.",
       "La muraille arrête les pillards", "muraille qui arrête", "processus",
       "grâce au guet posté sur la tour", "guet sur la tour", "processus",
       "les pillards rebroussent chemin", "pillards repoussés", "etat_local")
_mprev("Le vaccin protège les enfants par l'immunité transmise dans le lait : les enfants ne tomberont point malades.",
       "Le vaccin protège les enfants", "vaccin qui protège", "processus",
       "par l'immunité transmise dans le lait", "immunité transmise", "processus",
       "les enfants ne tomberont point malades", "enfants sains", "etat_local")
_mprev("Le parapet retient la terre au moyen de pieux enfoncés dans la pente : la terre ne glissera pas.",
       "Le parapet retient la terre", "parapet qui retient", "processus",
       "au moyen de pieux enfoncés dans la pente", "pieux enfoncés", "entite",
       "la terre ne glissera pas", "terre tenue", "etat_local")
_mprev("Le berger défend le troupeau avec son bâton ferré et ses chiens : le loup n'approchera point.",
       "Le berger défend le troupeau", "berger qui défend", "processus",
       "avec son bâton ferré et ses chiens", "bâton et chiens", "entite",
       "le loup n'approchera point", "loup tenu à distance", "etat_local")
_mprev("La haie vive freine le vent par ses épines entrelacées et ses ronces : le jardin ne desséchera point.",
       "La haie vive freine le vent", "haie qui freine", "processus",
       "par ses épines entrelacées et ses ronces", "épines et ronces", "entite",
       "le jardin ne desséchera point", "jardin préservé", "etat_local")
_mprev("Le couvercle étouffe le feu au moyen d'un linge mouillé jeté dessus : le feu ne reprendra point.",
       "Le couvercle étouffe le feu", "couvercle qui étouffe", "processus",
       "au moyen d'un linge mouillé jeté dessus", "linge mouillé", "entite",
       "le feu ne reprendra point", "feu éteint", "etat_local")
_mprev("Le serment lie les contractants par la parole donnée devant témoins : nul ne se dérobera.",
       "Le serment lie les contractants", "serment qui lie", "processus",
       "par la parole donnée devant témoins", "parole donnée", "processus",
       "nul ne se dérobera", "contractants tenus", "etat_local")
_mprev("Le fossé stoppe les sangliers grâce à des pieux aiguisés au fond : les sangliers ne passeront point.",
       "Le fossé stoppe les sangliers", "fossé qui stoppe", "processus",
       "grâce à des pieux aiguisés au fond", "pieux aiguisés", "entite",
       "les sangliers ne passeront point", "sangliers stoppés", "etat_local")
_mprev("La consigne retient les curieux au moyen d'une corde tendue à l'entrée : les curieux n'entreront point.",
       "La consigne retient les curieux", "consigne qui retient", "processus",
       "au moyen d'une corde tendue à l'entrée", "corde tendue", "entite",
       "les curieux n'entreront point", "curieux tenus dehors", "etat_local")
_mprev("Le remède coupe la fièvre par une saignée pratiquée à l'aube : le malade ne délirera plus.",
       "Le remède coupe la fièvre", "remède qui coupe", "processus",
       "par une saignée pratiquée à l'aube", "saignée à l'aube", "processus",
       "le malade ne délirera plus", "malade apaisé", "etat_local")

# --- mediated_cause × 10 ---
_mcaus("Le soleil mûrit les blés par la chaleur versée sur les sillons : les blés jaunissent.",
       "Le soleil mûrit les blés", "soleil qui mûrit", "processus",
       "par la chaleur versée sur les sillons", "chaleur versée", "processus",
       "les blés jaunissent", "blés qui jaunissent", "etat_local")
_mcaus("Le maître instruit les enfants au moyen de fables racontées le soir : les enfants s'assagissent.",
       "Le maître instruit les enfants", "maître qui instruit", "processus",
       "au moyen de fables racontées le soir", "fables du soir", "entite",
       "les enfants s'assagissent", "enfants qui s'assagissent", "etat_local")
_mcaus("La pluie ranime les semis grâce à l'eau infiltrée jusqu'aux racines : les semis reverdissent.",
       "La pluie ranime les semis", "pluie qui ranime", "processus",
       "grâce à l'eau infiltrée jusqu'aux racines", "eau infiltrée", "processus",
       "les semis reverdissent", "semis qui reverdissent", "etat_local")
_mcaus("Le forgeron soude le fer par un martelage répété sur l'enclume : le fer tient bon.",
       "Le forgeron soude le fer", "forgeron qui soude", "processus",
       "par un martelage répété sur l'enclume", "martelage répété", "processus",
       "le fer tient bon", "fer qui tient", "etat_local")
_mcaus("La mère console l'enfant au moyen d'une chanson fredonnée tout bas : l'enfant s'endort.",
       "La mère console l'enfant", "mère qui console", "processus",
       "au moyen d'une chanson fredonnée tout bas", "chanson fredonnée", "processus",
       "l'enfant s'endort", "enfant qui s'endort", "etat_local")
_mcaus("Le vent disperse les graines par des rafales venues de l'ouest : les friches fleurissent.",
       "Le vent disperse les graines", "vent qui disperse", "processus",
       "par des rafales venues de l'ouest", "rafales d'ouest", "processus",
       "les friches fleurissent", "friches qui fleurissent", "etat_local")
_mcaus("Le meunier enrichit le village grâce à la farine moulue sur pierre : le village prospère.",
       "Le meunier enrichit le village", "meunier qui enrichit", "processus",
       "grâce à la farine moulue sur pierre", "farine sur pierre", "entite",
       "le village prospère", "village qui prospère", "etat_local")
_mcaus("Le curé apaise les querelles par des paroles dites en chaire : le village se réconcilie.",
       "Le curé apaise les querelles", "curé qui apaise", "processus",
       "par des paroles dites en chaire", "paroles en chaire", "processus",
       "le village se réconcilie", "village réconcilié", "etat_local")
_mcaus("La lampe éclaire la veillée au moyen d'huile versée sans compter : la veillée se prolonge.",
       "La lampe éclaire la veillée", "lampe qui éclaire", "processus",
       "au moyen d'huile versée sans compter", "huile versée", "entite",
       "la veillée se prolonge", "veillée prolongée", "etat_local")
_mcaus("Le jardinier attire les abeilles par des rangées de lavande en fleur : le verger bourdonne.",
       "Le jardinier attire les abeilles", "jardinier qui attire", "processus",
       "par des rangées de lavande en fleur", "lavande en fleur", "entite",
       "le verger bourdonne", "verger qui bourdonne", "etat_local")


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
    assert len(T) == 40, len(T)
    sentences = []
    for n, (text, clauses, (src, dst, rel, third)) in enumerate(T, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-ternaire-t"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src] if isinstance(src, list) else [f"n{src:03d}"]
        e["target"] = f"n{dst:03d}"
        e["third"] = {"role": third[0], "node": f"n{third[1]:03d}"} if third else None
        sentences.append({"id": f"t{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "pastoral",
                          "cir": {"nodes": nodes, "edges": [e]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-ternaire-t", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver ternaire vague T.")
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
