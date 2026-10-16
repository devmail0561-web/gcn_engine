#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Mini-campagne ternaire 4a : conditional (20+20) + mediated (20+20).
Même contrat que tb2/mc3 (spans calculés, thirds vérifiés, poids 0.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

TERN4A = [
("Si la mare gèle, les canards partiront vers le sud.",
 [("la mare gèle", "mare gelée", "condition"), ("les canards partiront vers le sud", "départ asserté", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le foin moisira si la grange prend l'eau.",
 [("Le foin moisira", "moisissure assertée", "processus"), ("la grange prend l'eau", "grange mouillée", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si le meunier dort, la meule s'arrêtera avant l'aube.",
 [("le meunier dort", "meunier endormi", "condition"), ("la meule s'arrêtera avant l'aube", "arrêt asserté", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les poules picoreront si le grain est répandu dans la cour.",
 [("Les poules picoreront", "picorage asserté", "processus"), ("le grain est répandu dans la cour", "grain répandu", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la cloche sonne faux, les fidèles lèveront la tête.",
 [("la cloche sonne faux", "fausse note", "condition"), ("les fidèles lèveront la tête", "têtes levées", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le lait tournera si l'orage éclate avant la traite.",
 [("Le lait tournera", "lait tourné", "etat_local"), ("l'orage éclate avant la traite", "orage précoce", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si le chien aboie, le renard contournera le poulailler.",
 [("le chien aboie", "aboiements", "condition"), ("le renard contournera le poulailler", "contournement asserté", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les cravates des bourgeons éclateront si le soleil chauffe la vigne.",
 [("Les cravates des bourgeons éclateront", "éclatement asserté", "processus"), ("le soleil chauffe la vigne", "soleil chauffant", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la rivière baisse, le passeur gardera sa barque au sec.",
 [("la rivière baisse", "baisse des eaux", "condition"), ("le passeur gardera sa barque au sec", "barque gardée", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le coq chantera si l'aube pointe derrière les collines.",
 [("Le coq chantera", "chant asserté", "processus"), ("l'aube pointe derrière les collines", "aube pointante", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si la barrière cède, les vaches paîtront le jardin du voisin.",
 [("la barrière cède", "barrière cédée", "condition"), ("les vaches paîtront le jardin du voisin", "jardin pâturé", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les oignons germeront si la cave reste humide.",
 [("Les oignons germeront", "germination assertée", "processus"), ("la cave reste humide", "cave humide", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si le forgeron trempe la lame, l'acier durcira.",
 [("le forgeron trempe la lame", "trempe", "condition"), ("l'acier durcira", "durcissement asserté", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("La soupe accrochera si le feu est trop vif sous la marmite.",
 [("La soupe accrochera", "accrochage asserté", "processus"), ("le feu est trop vif sous la marmite", "feu vif", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si les taupes labourent, les prés seront bosselés au printemps.",
 [("les taupes labourent", "taupes actives", "condition"), ("les prés seront bosselés au printemps", "prés bosselés", "etat_local")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le beurre rancira si la laiterie manque de glace.",
 [("Le beurre rancira", "rancissement asserté", "processus"), ("la laiterie manque de glace", "manque de glace", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si les corbeaux croassent, la pluie tombera avant vêpres.",
 [("les corbeaux croassent", "croassements", "condition"), ("la pluie tombera avant vêpres", "pluie assertée", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Les pommes blettiront si la grêle les a meurtries.",
 [("Les pommes blettiront", "blettissement asserté", "processus"), ("la grêle les a meurtries", "grêle meurtrière", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si les loups hurlent, les moutons se serreront dans l'enclos.",
 [("les loups hurlent", "hurlements", "condition"), ("les moutons se serreront dans l'enclos", "troupeau serré", "processus")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("Le fumier chauffera si la paille est bien tassée.",
 [("Le fumier chauffera", "chauffe assertée", "processus"), ("la paille est bien tassée", "paille tassée", "condition")],
 (0, 1, "conditional_cause", ("Condition", 1))),
("Si les grenouilles coassent, l'étang débordera après l'averse.",
 [("les grenouilles coassent", "coassements", "condition"), ("l'étang débordera après l'averse", "débordement asserté", "evenement")],
 (0, 1, "conditional_cause", ("Condition", 0))),
("La huche gardera le pain si le couvercle joint bien.",
 [("La huche gardera le pain", "pain gardé", "etat_local"), ("le couvercle joint bien", "couvercle jointif", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le berger veille, aucun agneau ne manquera à l'appel.",
 [("le berger veille", "berger vigilant", "condition"), ("aucun agneau ne manquera à l'appel", "troupeau complet", "etat_local")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le cellier restera frais si la porte demeure close.",
 [("Le cellier restera frais", "cellier frais", "etat_local"), ("la porte demeure close", "porte close", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si les chiens gardent, les maraudeurs contourneront la ferme.",
 [("les chiens gardent", "chiens de garde", "condition"), ("les maraudeurs contourneront la ferme", "ferme évitée", "evenement")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La meule tiendra si le cerclage est resserré.",
 [("La meule tiendra", "meule tenue", "etat_local"), ("le cerclage est resserré", "cerclage resserré", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le guet ronde, les rôdeurs éviteront le village.",
 [("le guet ronde", "ronde du guet", "condition"), ("les rôdeurs éviteront le village", "village évité", "evenement")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le toit tiendra si les tuiles sont bien scellées.",
 [("Le toit tiendra", "toit tenu", "etat_local"), ("les tuiles sont bien scellées", "tuiles scellées", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si la lampe reste allumée, les mites n'attaqueront pas les lainages.",
 [("la lampe reste allumée", "lampe allumée", "condition"), ("les mites n'attaqueront pas les lainages", "lainages saufs", "etat_local")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La bergerie restera close si le verrou tient bon.",
 [("La bergerie restera close", "bergerie close", "etat_local"), ("le verrou tient bon", "verrou solide", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le coq chante, le renard n'approchera pas du poulailler.",
 [("le coq chante", "chant du coq", "condition"), ("le renard n'approchera pas du poulailler", "poulailler sauf", "etat_local")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le puits restera clair si la margelle est intacte.",
 [("Le puits restera clair", "eau claire", "etat_local"), ("la margelle est intacte", "margelle intacte", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si les oies crient, le blaireau rebroussera chemin.",
 [("les oies crient", "oies qui crient", "condition"), ("le blaireau rebroussera chemin", "blaireau repoussé", "processus")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La grange tiendra si les poutres sont étayées.",
 [("La grange tiendra", "grange tenue", "etat_local"), ("les poutres sont étayées", "poutres étayées", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si le chien dort, le facteur glissera la lettre sans bruit.",
 [("le chien dort", "chien endormi", "condition"), ("le facteur glissera la lettre sans bruit", "lettre glissée", "processus")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("Le grenier restera sec si le chaume est refait.",
 [("Le grenier restera sec", "grenier sec", "etat_local"), ("le chaume est refait", "chaume refait", "condition")],
 (0, 1, "conditional_prevent", ("Condition", 1))),
("Si la cloche tinte, les Levantins ouvriront leurs étals.",
 [("la cloche tinte", "cloche qui tinte", "condition"), ("les Levantins ouvriront leurs étals", "étals ouverts", "processus")],
 (0, 1, "conditional_prevent", ("Condition", 0))),
("La tisane exhale son parfum grâce aux herbes cueillies à l'aube.",
 [("La tisane exhale son parfum", "tisane qui exhale", "processus"), ("herbes cueillies à l'aube", "herbes (moyen)", "entite"), ("son parfum", "parfum exhalé", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le cidre produit son alcool via les levures sauvages de la cave.",
 [("Le cidre produit son alcool", "cidre qui produit", "processus"), ("les levures sauvages de la cave", "levures (moyen)", "entite"), ("son alcool", "alcool produit", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La soupe épaissit le bouillon au moyen du pain rassis émietté.",
 [("La soupe épaissit le bouillon", "soupe qui épaissit", "processus"), ("pain rassis émietté", "pain (moyen)", "entite"), ("le bouillon", "bouillon", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le cuir chausse le pied par l'intermédiaire de la graisse de phoque.",
 [("Le cuir chausse le pied", "cuir qui chausse", "processus"), ("la graisse de phoque", "graisse (moyen)", "entite"), ("le pied", "pied chaussé", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La laine donne son drap via le foulon actionné au moulin.",
 [("La laine donne son drap", "laine qui donne", "processus"), ("foulon actionné au moulin", "foulon (moyen)", "entite"), ("son drap", "drap donné", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le beurre naît de la crème via la baratte tournée chaque matin.",
 [("Le beurre naît de la crème", "beurre qui naît", "processus"), ("la baratte tournée chaque matin", "baratte (moyen)", "entite"), ("la crème", "crème", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La chaux couvre le mur au moyen du badigeon passé au pinceau.",
 [("La chaux couvre le mur", "chaux qui couvre", "processus"), ("badigeon passé au pinceau", "badigeon (moyen)", "entite"), ("le mur", "mur", "etat_local")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le fil casse et gâche la quenouillée par l'intermédiaire du rouet trop tendu.",
 [("Le fil casse et gâche la quenouillée", "fil qui casse", "processus"), ("rouet trop tendu", "rouet (moyen)", "entite"), ("la quenouillée", "quenouillée", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La bière remplit la pinte grâce au houblon ajouté en fin d'ébullition.",
 [("La bière remplit la pinte", "bière qui remplit", "processus"), ("houblon ajouté en fin d'ébullition", "houblon (moyen)", "entite"), ("la pinte", "pinte", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le papier porte la lettre via l'encre diluée à l'eau de pluie.",
 [("Le papier porte la lettre", "papier qui porte", "processus"), ("l'encre diluée à l'eau de pluie", "encre (moyen)", "entite"), ("la lettre", "lettre", "concept")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La braise rallume le feu sous la cendre amassée chaque soir.",
 [("La braise rallume le feu", "braise qui rallume", "processus"), ("la cendre amassée chaque soir", "cendre (moyen)", "entite"), ("le feu", "feu", "evenement")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le toit laisse passer la goutte par l'intermédiaire des tuiles fendues par le gel.",
 [("Le toit laisse passer la goutte", "toit qui laisse passer", "processus"), ("tuiles fendues par le gel", "tuiles (moyen)", "entite"), ("la goutte", "goutte", "evenement")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La meule moud le grain grâce à l'eau détournée par la vanne.",
 [("La meule moud le grain", "meule qui moud", "processus"), ("l'eau détournée par la vanne", "eau (moyen)", "entite"), ("le grain", "grain", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La lampe dégage sa suie au moyen de la mèche trop longue.",
 [("La lampe dégage sa suie", "lampe qui dégage", "processus"), ("la mèche trop longue", "mèche (moyen)", "entite"), ("sa suie", "suie dégagée", "etat_local")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le seau perd son eau par l'intermédiaire des douves disjointes.",
 [("Le seau perd son eau", "seau qui perd", "processus"), ("douves disjointes", "douves (moyen)", "entite"), ("son eau", "eau perdue", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La herse ouvre le sol via les dents forgées par le maréchal.",
 [("La herse ouvre le sol", "herse qui ouvre", "processus"), ("les dents forgées par le maréchal", "dents (moyen)", "entite"), ("le sol", "sol", "etat_local")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La faux fauche l'herbe grâce à la lame affûtée à la pierre.",
 [("La faux fauche l'herbe", "faux qui fauche", "processus"), ("la lame affûtée à la pierre", "lame (moyen)", "entite"), ("l'herbe", "herbe", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La nasse prend les poissons via les joncs tressés par le vannier.",
 [("La nasse prend les poissons", "nasse qui prend", "processus"), ("les joncs tressés par le vannier", "joncs (moyen)", "entite"), ("les poissons", "poissons", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La ruche nourrit le miel au moyen des abeilles rentrant des champs.",
 [("La ruche nourrit le miel", "ruche qui nourrit", "processus"), ("abeilles rentrant des champs", "abeilles (moyen)", "entite"), ("le miel", "miel", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("Le pressoir verse son jus par l'intermédiaire de la vis serrée au cabestan.",
 [("Le pressoir verse son jus", "pressoir qui verse", "processus"), ("la vis serrée au cabestan", "vis (moyen)", "entite"), ("son jus", "jus versé", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
("La bergerie repousse le loup grâce au loquet forgé par le maréchal.",
 [("La bergerie repousse le loup", "bergerie qui repousse", "processus"), ("loquet forgé par le maréchal", "loquet (moyen)", "entite"), ("le loup", "loup", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La huche prive les souris de grain au moyen d'un couvercle lesté de pierres.",
 [("La huche prive les souris de grain", "huche qui prive", "processus"), ("un couvercle lesté de pierres", "couvercle (moyen)", "entite"), ("les souris", "souris", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La basse-cour tient le renard à distance via le verrou tiré chaque soir.",
 [("La basse-cour tient le renard à distance", "basse-cour qui tient", "processus"), ("le verrou tiré chaque soir", "verrou (moyen)", "entite"), ("le renard", "renard", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le poulailler écarte la fouine par l'intermédiaire du grillage enterré profond.",
 [("Le poulailler écarte la fouine", "poulailler qui écarte", "processus"), ("grillage enterré profond", "grillage (moyen)", "entite"), ("la fouine", "fouine", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La malle décourage le voleur grâce au cadenas forgé à la ville.",
 [("La malle décourage le voleur", "malle qui décourage", "processus"), ("cadenas forgé à la ville", "cadenas (moyen)", "entite"), ("le voleur", "voleur", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le coffre rebute les curieux au moyen de la serrure à secret du grand-père.",
 [("Le coffre rebute les curieux", "coffre qui rebute", "processus"), ("la serrure à secret du grand-père", "serrure (moyen)", "entite"), ("les curieux", "curieux", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La porte arrête le vent par l'intermédiaire de la barre posée en travers.",
 [("La porte arrête le vent", "porte qui arrête", "processus"), ("la barre posée en travers", "barre (moyen)", "entite"), ("le vent", "vent", "etat_global")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le tonneau retient l'évaporation grâce à la bonde enfoncée au maillet.",
 [("Le tonneau retient l'évaporation", "tonneau qui retient", "processus"), ("la bonde enfoncée au maillet", "bonde (moyen)", "entite"), ("l'évaporation", "évaporation", "processus")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La palissade repousse les sangliers via les pieux appointés plantés serré.",
 [("La palissade repousse les sangliers", "palissade qui repousse", "processus"), ("les pieux appointés plantés serré", "pieux (moyen)", "entite"), ("les sangliers", "sangliers", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le rucher brave le froid au moyen des toiles cirées tendues l'hiver.",
 [("Le rucher brave le froid", "rucher qui brave", "processus"), ("toiles cirées tendues l'hiver", "toiles (moyen)", "entite"), ("le froid", "froid", "etat_global")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La serre défie la gelée par l'intermédiaire des vitres closes hermétiquement.",
 [("La serre défie la gelée", "serre qui défie", "processus"), ("vitres closes hermétiquement", "vitres (moyen)", "entite"), ("la gelée", "gelée", "evenement")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le fossé arrête les maraudeurs au moyen des ronces laissées en bordure.",
 [("Le fossé arrête les maraudeurs", "fossé qui arrête", "processus"), ("ronces laissées en bordure", "ronces (moyen)", "entite"), ("les maraudeurs", "maraudeurs", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La tour repère l'ennemi par l'intermédiaire du guetteur posté au créneau.",
 [("La tour repère l'ennemi", "tour qui repère", "processus"), ("guetteur posté au créneau", "guetteur (moyen)", "entite"), ("l'ennemi", "ennemi", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le grenier tient les rongeurs à distance via le chat posté contre les souris.",
 [("Le grenier tient les rongeurs à distance", "grenier qui tient", "processus"), ("le chat posté contre les souris", "chat (moyen)", "entite"), ("les rongeurs", "rongeurs", "entite")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La digue freine le courant grâce aux pieux enfoncés dans la vase.",
 [("La digue freine le courant", "digue qui freine", "processus"), ("pieux enfoncés dans la vase", "pieux (moyen)", "entite"), ("le courant", "courant", "processus")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le lavoir chasse la vase au moyen du rinçage à grande eau.",
 [("Le lavoir chasse la vase", "lavoir qui chasse", "processus"), ("rinçage à grande eau", "rinçage (moyen)", "processus"), ("la vase", "vase", "etat_local")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La mare étouffe la croupissure par l'intermédiaire des roseaux filtrant l'eau.",
 [("La mare étouffe la croupissure", "mare qui étouffe", "processus"), ("roseaux filtrant l'eau", "roseaux (moyen)", "entite"), ("la croupissure", "croupissure", "processus")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("Le cellier repousse la chaleur au moyen de la terre battue épaisse.",
 [("Le cellier repousse la chaleur", "cellier qui repousse", "processus"), ("la terre battue épaisse", "terre (moyen)", "entite"), ("la chaleur", "chaleur", "etat_global")],
 (0, 2, "mediated_prevent", ("Mediator", 1))),
("La ruche disperse ses abeilles via la nouvelle reine née au printemps.",
 [("La ruche disperse ses abeilles", "ruche qui disperse", "processus"), ("la nouvelle reine née au printemps", "reine (moyen)", "entite"), ("ses abeilles", "abeilles dispersées", "entite")],
 (0, 2, "mediated_cause", ("Mediator", 1))),
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
    raise AssertionError("pas de marqueurs : ternaires purs (tiers porte la relation)")


def construire() -> dict:
    assert len(TERN4A) == 76, len(TERN4A)
    sentences = []
    for n, (text, clauses, (src, dst, rel, third)) in enumerate(TERN4A, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-mc4a"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src] if isinstance(src, list) else [f"n{src:03d}"]
        e["target"] = f"n{dst:03d}"
        e["third"] = {"role": third[0], "node": f"n{third[1]:03d}"} if third else None
        sentences.append({"id": f"mc4a{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": [e]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-minicampagne-4a", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver mini-campagne 4a.")
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
