#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""P1b plan-remediation-donnees.md : 129 phrases FR candidates pour classes rares
(control_dependency 45, prevent 33, filter 27, data_dependency 24).

Convention : source = élément déclencheur / contrôleur / exclu ; cible = élément
contrôlé / effet / ensemble filtré. À VALIDER par l'humain (doctrine : juge=humain).

Statut : SILVER non validé — poids 0.0, hors DATA/train, jamais mergé sans validation
+ gate IC95 (GATES:15). Tokens = segmentation whitespace, token_span CALCULÉS
depuis les clauses (même contrat que build_ternary_proposal.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

# (texte, [(span, label, type)], [(src, dst, relation, marqueur)])
CD = [
("Le général a déclenché l'offensive sous son commandement direct.",
 [("Le général", "le général", "entite"), ("l'offensive", "offensive déclenchée", "evenement")],
 [(0, 1, "control_dependency", "déclenché")]),
("L'éruption a déclenché l'alerte sous la supervision des volcanologues.",
 [("L'éruption", "éruption", "evenement"), ("l'alerte", "alerte déclenchée", "processus")],
 [(0, 1, "control_dependency", "déclenché")]),
("Le ministre a limité la vitesse sous la pression des associations.",
 [("Le ministre", "le ministre", "entite"), ("la vitesse", "vitesse limitée", "concept")],
 [(0, 1, "control_dependency", "limité")]),
("Le préfet a interdit la manifestation sous l'état d'urgence.",
 [("Le préfet", "le préfet", "entite"), ("la manifestation", "manifestation interdite", "evenement")],
 [(0, 1, "control_dependency", "interdit")]),
("La banque centrale régule le crédit sous mandat du gouvernement.",
 [("La banque centrale", "banque centrale", "entite"), ("le crédit", "crédit régulé", "concept")],
 [(0, 1, "control_dependency", "régule")]),
("Le pilote a déclenché l'atterrissage d'urgence sous les ordres de la tour.",
 [("Le pilote", "le pilote", "entite"), ("l'atterrissage d'urgence", "atterrissage déclenché", "evenement")],
 [(0, 1, "control_dependency", "déclenché")]),
("L'arbitre a interrompu le match sous la menace des supporters.",
 [("L'arbitre", "arbitre", "entite"), ("le match", "match interrompu", "evenement")],
 [(0, 1, "control_dependency", "interrompu")]),
("Le directeur a suspendu les licenciements sous la contrainte du tribunal.",
 [("Le directeur", "le directeur", "entite"), ("les licenciements", "licenciements suspendus", "processus")],
 [(0, 1, "control_dependency", "suspendu")]),
("La mairie a restreint la circulation sous pic de pollution.",
 [("La mairie", "la mairie", "entite"), ("la circulation", "circulation restreinte", "processus")],
 [(0, 1, "control_dependency", "restreint")]),
("Le chirurgien a déclenché l'opération sous anesthésie générale.",
 [("Le chirurgien", "le chirurgien", "entite"), ("l'opération", "opération déclenchée", "processus")],
 [(0, 1, "control_dependency", "déclenché")]),
("Le capitaine a ordonné l'évacuation sous la menace de l'incendie.",
 [("Le capitaine", "le capitaine", "entite"), ("l'évacuation", "évacuation ordonnée", "processus")],
 [(0, 1, "control_dependency", "ordonné")]),
("L'éditeur a censuré le passage sous la pression du pouvoir.",
 [("L'éditeur", "éditeur", "entite"), ("le passage", "passage censuré", "concept")],
 [(0, 1, "control_dependency", "censuré")]),
("Le juge a placé le suspect sous contrôle judiciaire strict.",
 [("Le juge", "le juge", "entite"), ("le suspect", "suspect contrôlé", "entite")],
 [(0, 1, "control_dependency", "contrôle")]),
("L'entraîneur a remplacé le gardien sous les sifflets du public.",
 [("L'entraîneur", "entraîneur", "entite"), ("le gardien", "gardien remplacé", "entite")],
 [(0, 1, "control_dependency", "remplacé")]),
("La direction a gelé les embauches sous la crise budgétaire.",
 [("La direction", "la direction", "entite"), ("les embauches", "embauches gelées", "processus")],
 [(0, 1, "control_dependency", "gelé")]),
("Le douanier a confisqué la marchandise sous soupçon de fraude.",
 [("Le douanier", "le douanier", "entite"), ("la marchandise", "marchandise confisquée", "entite")],
 [(0, 1, "control_dependency", "confisqué")]),
("L'ingénieur a coupé l'alimentation sous le risque de surchauffe.",
 [("L'ingénieur", "ingénieur", "entite"), ("l'alimentation", "alimentation coupée", "processus")],
 [(0, 1, "control_dependency", "coupé")]),
("Le proviseur a exclu l'élève sous le règlement intérieur.",
 [("Le proviseur", "le proviseur", "entite"), ("l'élève", "élève exclu", "entite")],
 [(0, 1, "control_dependency", "exclu")]),
("Le gouvernement a décrété le couvre-feu sous l'épidémie galopante.",
 [("Le gouvernement", "le gouvernement", "entite"), ("le couvre-feu", "couvre-feu décrété", "contrainte")],
 [(0, 1, "control_dependency", "décrété")]),
("Le chef d'orchestre a lancé l'ouverture sous les applaudissements.",
 [("Le chef d'orchestre", "chef d'orchestre", "entite"), ("l'ouverture", "ouverture lancée", "evenement")],
 [(0, 1, "control_dependency", "lancé")]),
("La maîtresse a confisqué le portable sous l'interdiction du collège.",
 [("La maîtresse", "la maîtresse", "entite"), ("le portable", "portable confisqué", "entite")],
 [(0, 1, "control_dependency", "confisqué")]),
("Le maire a fermé les plages sous l'alerte aux méduses.",
 [("Le maire", "le maire", "entite"), ("les plages", "plages fermées", "etat_local")],
 [(0, 1, "control_dependency", "fermé")]),
("Le producteur a annulé le concert sous la tempête annoncée.",
 [("Le producteur", "le producteur", "entite"), ("le concert", "concert annulé", "evenement")],
 [(0, 1, "control_dependency", "annulé")]),
("L'administrateur a bloqué le compte sous suspicion de piratage.",
 [("L'administrateur", "administrateur", "entite"), ("le compte", "compte bloqué", "concept")],
 [(0, 1, "control_dependency", "bloqué")]),
("Le vétérinaire a isolé l'animal sous le protocole sanitaire.",
 [("Le vétérinaire", "vétérinaire", "entite"), ("l'animal", "animal isolé", "entite")],
 [(0, 1, "control_dependency", "isolé")]),
("Le colonel a replié ses troupes sous le bombardement ennemi.",
 [("Le colonel", "le colonel", "entite"), ("ses troupes", "troupes repliées", "entite")],
 [(0, 1, "control_dependency", "replié")]),
("La commission a rejeté le projet sous son irrégularité manifeste.",
 [("La commission", "la commission", "entite"), ("le projet", "projet rejeté", "concept")],
 [(0, 1, "control_dependency", "rejeté")]),
("Le patron a réduit les horaires sous la chute des commandes.",
 [("Le patron", "le patron", "entite"), ("les horaires", "horaires réduits", "concept")],
 [(0, 1, "control_dependency", "réduit")]),
("Le syndic a fait évacuer l'immeuble sous la fissure du mur.",
 [("Le syndic", "le syndic", "entite"), ("l'immeuble", "immeuble évacué", "etat_local")],
 [(0, 1, "control_dependency", "sous")]),
("L'arbitre a expulsé le joueur sous son geste violent.",
 [("L'arbitre", "arbitre", "entite"), ("le joueur", "joueur expulsé", "entite")],
 [(0, 1, "control_dependency", "expulsé")]),
("Le rectorat a fermé le lycée sous la grève des transports.",
 [("Le rectorat", "le rectorat", "entite"), ("le lycée", "lycée fermé", "etat_local")],
 [(0, 1, "control_dependency", "fermé")]),
("Le metteur en scène a écourté la pièce sous les huées.",
 [("Le metteur en scène", "metteur en scène", "entite"), ("la pièce", "pièce écourtée", "evenement")],
 [(0, 1, "control_dependency", "écourté")]),
("Le capitaine a fait demi-tour sous la brume épaisse.",
 [("Le capitaine", "le capitaine", "entite"), ("demi-tour", "demi-tour ordonné", "processus")],
 [(0, 1, "control_dependency", "sous")]),
("La préfecture a suspendu le permis sous l'alcoolémie du conducteur.",
 [("La préfecture", "la préfecture", "entite"), ("le permis", "permis suspendu", "concept")],
 [(0, 1, "control_dependency", "suspendu")]),
("Le berger a rassemblé le troupeau sous l'orage imminent.",
 [("Le berger", "le berger", "entite"), ("le troupeau", "troupeau rassemblé", "entite")],
 [(0, 1, "control_dependency", "rassemblé")]),
("Le chef de gare a immobilisé le train sous le signal défectueux.",
 [("Le chef de gare", "chef de gare", "entite"), ("le train", "train immobilisé", "etat_local")],
 [(0, 1, "control_dependency", "immobilisé")]),
("La douane a retenu le colis sous le contrôle des stupéfiants.",
 [("La douane", "la douane", "entite"), ("le colis", "colis retenu", "entite")],
 [(0, 1, "control_dependency", "retenu")]),
("L'entraîneur a sorti l'attaquant sous sa blessure au genou.",
 [("L'entraîneur", "entraîneur", "entite"), ("l'attaquant", "attaquant sorti", "entite")],
 [(0, 1, "control_dependency", "sorti")]),
("Le curateur a mis le musée sous scellés après le vol.",
 [("Le curateur", "le curateur", "entite"), ("le musée", "musée sous scellés", "etat_local")],
 [(0, 1, "control_dependency", "sous")]),
("Le gendarme a fait souffler le conducteur sous l'éthylotest.",
 [("Le gendarme", "le gendarme", "entite"), ("le conducteur", "conducteur contrôlé", "entite")],
 [(0, 1, "control_dependency", "sous")]),
("La maîtresse-nageuse a évacué le bassin sous l'orage électrique.",
 [("La maîtresse-nageuse", "maîtresse-nageuse", "entite"), ("le bassin", "bassin évacué", "etat_local")],
 [(0, 1, "control_dependency", "évacué")]),
("Le président a dissous l'assemblée sous la crise institutionnelle.",
 [("Le président", "le président", "entite"), ("l'assemblée", "assemblée dissoute", "entite")],
 [(0, 1, "control_dependency", "dissous")]),
("Le régisseur a baissé le rideau sous le malaise de l'acteur.",
 [("Le régisseur", "le régisseur", "entite"), ("le rideau", "rideau baissé", "processus")],
 [(0, 1, "control_dependency", "baissé")]),
("Le chef a renvoyé le plat sous la plainte du client.",
 [("Le chef", "le chef", "entite"), ("le plat", "plat renvoyé", "entite")],
 [(0, 1, "control_dependency", "renvoyé")]),
("Le gardien a verrouillé les grilles sous l'alerte intrusion.",
 [("Le gardien", "le gardien", "entite"), ("les grilles", "grilles verrouillées", "etat_local")],
 [(0, 1, "control_dependency", "verrouillé")]),
]

PREVENT = [
("Le barrage empêche la crue de submerger le village.",
 [("Le barrage", "le barrage", "entite"), ("la crue de submerger le village", "submersion évitée", "evenement")],
 [(0, 1, "prevent", "empêche")]),
("Le vaccin empêche le virus de se propager dans l'organisme.",
 [("Le vaccin", "le vaccin", "entite"), ("le virus de se propager dans l'organisme", "propagation évitée", "processus")],
 [(0, 1, "prevent", "empêche")]),
("Le casque évite au cycliste un traumatisme crânien.",
 [("Le casque", "le casque", "entite"), ("un traumatisme crânien", "traumatisme évité", "etat_local")],
 [(0, 1, "prevent", "évite")]),
("La digue empêche la mer d'envahir les polders.",
 [("La digue", "la digue", "entite"), ("la mer d'envahir les polders", "invasion évitée", "evenement")],
 [(0, 1, "prevent", "empêche")]),
("Le parapluie évite à la mariée de ruiner sa coiffure.",
 [("Le parapluie", "le parapluie", "entite"), ("de ruiner sa coiffure", "coiffure sauvée", "etat_local")],
 [(0, 1, "prevent", "évite")]),
("Le pare-feu bloque l'intrusion avant tout dégât.",
 [("Le pare-feu", "le pare-feu", "entite"), ("l'intrusion", "intrusion bloquée", "evenement")],
 [(0, 1, "prevent", "bloque")]),
("La ceinture empêche le passager d'être éjecté.",
 [("La ceinture", "la ceinture", "entite"), ("d'être éjecté", "éjection évitée", "evenement")],
 [(0, 1, "prevent", "empêche")]),
("L'extincteur stoppe le départ de feu dans la cuisine.",
 [("L'extincteur", "extincteur", "entite"), ("le départ de feu dans la cuisine", "incendie stoppé", "evenement")],
 [(0, 1, "prevent", "stoppe")]),
("Le pansement empêche la plaie de s'infecter.",
 [("Le pansement", "le pansement", "entite"), ("la plaie de s'infecter", "infection évitée", "processus")],
 [(0, 1, "prevent", "empêche")]),
("La moustiquaire évite aux enfants les piqûres nocturnes.",
 [("La moustiquaire", "la moustiquaire", "entite"), ("les piqûres nocturnes", "piqûres évitées", "evenement")],
 [(0, 1, "prevent", "évite")]),
("Le brise-lames protège le port de la houle.",
 [("Le brise-lames", "brise-lames", "entite"), ("le port de la houle", "port protégé", "etat_local")],
 [(0, 1, "prevent", "protège")]),
("Le gilet pare-balles stoppe le projectile avant l'impact.",
 [("Le gilet pare-balles", "gilet pare-balles", "entite"), ("le projectile", "projectile stoppé", "evenement")],
 [(0, 1, "prevent", "stoppe")]),
("La crème solaire prévient les coups de soleil des baigneurs.",
 [("La crème solaire", "crème solaire", "entite"), ("les coups de soleil des baigneurs", "coups de soleil évités", "etat_local")],
 [(0, 1, "prevent", "prévient")]),
("Le cadenas empêche les voleurs d'ouvrir le garage.",
 [("Le cadenas", "le cadenas", "entite"), ("d'ouvrir le garage", "ouverture évitée", "processus")],
 [(0, 1, "prevent", "empêche")]),
("L'antivirus neutralise le rançongiciel avant le chiffrement.",
 [("L'antivirus", "antivirus", "entite"), ("le rançongiciel", "rançongiciel neutralisé", "entite")],
 [(0, 1, "prevent", "neutralise")]),
("Le parapet retient les chutes de pierres sur la route.",
 [("Le parapet", "le parapet", "entite"), ("les chutes de pierres sur la route", "chutes retenues", "evenement")],
 [(0, 1, "prevent", "retient")]),
("Le traitement précoce évite l'aggravation de la maladie.",
 [("Le traitement précoce", "traitement précoce", "processus"), ("l'aggravation de la maladie", "aggravation évitée", "processus")],
 [(0, 1, "prevent", "évite")]),
("Le fusible coupe le circuit avant la surcharge.",
 [("Le fusible", "le fusible", "entite"), ("le circuit", "circuit coupé", "etat_local")],
 [(0, 1, "prevent", "coupe")]),
("La rampe empêche le fauteuil de basculer dans l'escalier.",
 [("La rampe", "la rampe", "entite"), ("de basculer dans l'escalier", "bascule évitée", "evenement")],
 [(0, 1, "prevent", "empêche")]),
("Le masque bloque la projection des gouttelettes.",
 [("Le masque", "le masque", "entite"), ("la projection des gouttelettes", "projection bloquée", "processus")],
 [(0, 1, "prevent", "bloque")]),
("L'arrosage nocturne prévient le dessèchement des semis.",
 [("L'arrosage nocturne", "arrosage nocturne", "processus"), ("le dessèchement des semis", "dessèchement évité", "processus")],
 [(0, 1, "prevent", "prévient")]),
("Le grillage retient les ballons perdus du stade.",
 [("Le grillage", "le grillage", "entite"), ("les ballons perdus du stade", "ballons retenus", "etat_local")],
 [(0, 1, "prevent", "retient")]),
("La révision empêche la panne du moteur en plein vol.",
 [("La révision", "la révision", "processus"), ("la panne du moteur en plein vol", "panne évitée", "evenement")],
 [(0, 1, "prevent", "empêche")]),
("Le sas bloque la contamination entre les deux salles.",
 [("Le sas", "le sas", "entite"), ("la contamination entre les deux salles", "contamination bloquée", "processus")],
 [(0, 1, "prevent", "bloque")]),
("La housse protège le canapé des griffes du chat.",
 [("La housse", "la housse", "entite"), ("le canapé des griffes du chat", "canapé protégé", "etat_local")],
 [(0, 1, "prevent", "protège")]),
("Le détecteur de fumée prévient l'intoxication des dormeurs.",
 [("Le détecteur de fumée", "détecteur de fumée", "entite"), ("l'intoxication des dormeurs", "intoxication évitée", "evenement")],
 [(0, 1, "prevent", "prévient")]),
("La barrière stoppe le troupeau avant la falaise.",
 [("La barrière", "la barrière", "entite"), ("le troupeau", "troupeau stoppé", "processus")],
 [(0, 1, "prevent", "stoppe")]),
("Le tuteur maintient la tige malgré le vent fort.",
 [("Le tuteur", "le tuteur", "entite"), ("la tige", "tige maintenue", "etat_local")],
 [(0, 1, "prevent", "maintient")]),
("L'embargo bloque l'arrivée des armes dans la région.",
 [("L'embargo", "embargo", "contrainte"), ("l'arrivée des armes dans la région", "arrivée bloquée", "processus")],
 [(0, 1, "prevent", "bloque")]),
("Le rappel vaccinal écarte tout risque de récidive.",
 [("Le rappel vaccinal", "rappel vaccinal", "processus"), ("tout risque de récidive", "récidive écartée", "concept")],
 [(0, 1, "prevent", "écarte")]),
("La gouttière détourne la pluie loin des fondations.",
 [("La gouttière", "la gouttière", "entite"), ("la pluie", "pluie détournée", "processus")],
 [(0, 1, "prevent", "détourne")]),
("Le rideau de fer contenait la foule des supporters.",
 [("Le rideau de fer", "rideau de fer", "entite"), ("la foule des supporters", "foule contenue", "etat_local")],
 [(0, 1, "prevent", "contenait")]),
("Le verrouillage automatique interdit l'accès après minuit.",
 [("Le verrouillage automatique", "verrouillage automatique", "contrainte"), ("l'accès après minuit", "accès interdit", "processus")],
 [(0, 1, "prevent", "interdit")]),
]

FILTER = [
("Tous les invités sont venus, à l'exception de la tante malade.",
 [("Tous les invités", "tous les invités", "entite"), ("la tante malade", "tante exclue", "entite")],
 [(0, 1, "filter", "l'exception")]),
("Seuls les abonnés accèdent à la salle de sport.",
 [("les abonnés", "abonnés admis", "entite"), ("la salle de sport", "salle filtrée", "etat_local")],
 [(0, 1, "filter", "Seuls")]),
("Le musée est ouvert tous les jours sauf le lundi.",
 [("Le musée", "le musée", "entite"), ("le lundi", "lundi exclu", "concept")],
 [(0, 1, "filter", "sauf")]),
("Tout le village a voté, hormis les absents excusés.",
 [("Tout le village", "tout le village", "entite"), ("les absents excusés", "absents exclus", "entite")],
 [(0, 1, "filter", "hormis")]),
("Uniquement les urgences passent après vingt-deux heures.",
 [("les urgences", "urgences admises", "evenement"), ("après vingt-deux heures", "créneau filtré", "concept")],
 [(0, 1, "filter", "Uniquement")]),
("La promotion concerne tous les articles, hors électronique.",
 [("tous les articles", "tous les articles", "entite"), ("électronique", "électronique exclue", "concept")],
 [(0, 1, "filter", "hors")]),
("Chaque élève a rendu son devoir, excepté le délégué malade.",
 [("Chaque élève", "chaque élève", "entite"), ("le délégué malade", "délégué exclu", "entite")],
 [(0, 1, "filter", "excepté")]),
("Seules les voitures électriques entrent dans le centre.",
 [("les voitures électriques", "voitures électriques admises", "entite"), ("le centre", "centre filtré", "etat_local")],
 [(0, 1, "filter", "Seules")]),
("Le train dessert toutes les gares sauf la halte fermée.",
 [("toutes les gares", "toutes les gares", "entite"), ("la halte fermée", "halte exclue", "etat_local")],
 [(0, 1, "filter", "sauf")]),
("Tous les dossiers sont traités, à part celui de l'héritier.",
 [("Tous les dossiers", "tous les dossiers", "concept"), ("celui de l'héritier", "dossier exclu", "concept")],
 [(0, 1, "filter", "part")]),
("Le forfait inclut toutes les remontées, hormis le téléphérique.",
 [("toutes les remontées", "toutes les remontées", "entite"), ("le téléphérique", "téléphérique exclu", "entite")],
 [(0, 1, "filter", "hormis")]),
("Seul le capitaine connaît la route du trésor.",
 [("le capitaine", "capitaine informé", "entite"), ("la route du trésor", "route filtrée", "concept")],
 [(0, 1, "filter", "Seul")]),
("La cantine sert tout le monde, excepté les externes.",
 [("tout le monde", "tout le monde", "entite"), ("les externes", "externes exclus", "entite")],
 [(0, 1, "filter", "excepté")]),
("Toutes les chambres sont libres, à l'exception de la suite.",
 [("Toutes les chambres", "toutes les chambres", "entite"), ("la suite", "suite exclue", "etat_local")],
 [(0, 1, "filter", "l'exception")]),
("Uniquement le personnel traverse la zone de sécurité.",
 [("le personnel", "personnel admis", "entite"), ("la zone de sécurité", "zone filtrée", "etat_local")],
 [(0, 1, "filter", "Uniquement")]),
("Le contrat couvre tous les risques sauf le vol.",
 [("tous les risques", "tous les risques", "concept"), ("le vol", "vol exclu", "evenement")],
 [(0, 1, "filter", "sauf")]),
("Chaque commune vote dimanche, hormis celles sous tutelle.",
 [("Chaque commune", "chaque commune", "entite"), ("celles sous tutelle", "communes exclues", "etat_local")],
 [(0, 1, "filter", "hormis")]),
("Seuls les diplômés postulent au poste d'ingénieur.",
 [("les diplômés", "diplômés admis", "entite"), ("au poste d'ingénieur", "poste filtré", "concept")],
 [(0, 1, "filter", "Seuls")]),
("Le spectacle plaît à tous, excepté aux critiques grincheux.",
 [("à tous", "tous conquis", "entite"), ("aux critiques grincheux", "critiques exclus", "entite")],
 [(0, 1, "filter", "excepté")]),
("Toute la classe part en sortie, à part les punis.",
 [("Toute la classe", "toute la classe", "entite"), ("les punis", "punis exclus", "etat_local")],
 [(0, 1, "filter", "part")]),
("Le régime autorise tous les légumes, hors féculents.",
 [("tous les légumes", "tous les légumes", "entite"), ("féculents", "féculents exclus", "concept")],
 [(0, 1, "filter", "hors")]),
("Seule la version payante débloque les statistiques.",
 [("la version payante", "version payante admise", "concept"), ("les statistiques", "statistiques filtrées", "concept")],
 [(0, 1, "filter", "Seule")]),
("Les secours ont fouillé tout l'immeuble sauf la cave inondée.",
 [("tout l'immeuble", "tout l'immeuble", "etat_local"), ("la cave inondée", "cave exclue", "etat_local")],
 [(0, 1, "filter", "sauf")]),
("Tous les coureurs ont fini, à l'exception du favori blessé.",
 [("Tous les coureurs", "tous les coureurs", "entite"), ("du favori blessé", "favori exclu", "entite")],
 [(0, 1, "filter", "l'exception")]),
("La bibliothèque prête tous les livres hormis les usuels.",
 [("tous les livres", "tous les livres", "entite"), ("les usuels", "usuels exclus", "concept")],
 [(0, 1, "filter", "hormis")]),
("Chaque salarié reçoit la prime, excepté les apprentis.",
 [("Chaque salarié", "chaque salarié", "entite"), ("les apprentis", "apprentis exclus", "entite")],
 [(0, 1, "filter", "excepté")]),
("Seuls les riverains stationnent dans la ruelle.",
 [("les riverains", "riverains admis", "entite"), ("la ruelle", "ruelle filtrée", "etat_local")],
 [(0, 1, "filter", "Seuls")]),
]

DATADEP = [
("La récolte dépend de la pluie de printemps.",
 [("La récolte", "la récolte", "processus"), ("la pluie de printemps", "pluie déterminante", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("Le verdict repose sur trois témoignages concordants.",
 [("Le verdict", "le verdict", "concept"), ("trois témoignages concordants", "témoignages fondants", "evenement")],
 [(0, 1, "data_dependency", "repose")]),
("Son humeur dépend de la météo du matin.",
 [("Son humeur", "son humeur", "etat_local"), ("la météo du matin", "météo déterminante", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("La note finale découle de la moyenne des contrôles.",
 [("La note finale", "la note finale", "concept"), ("la moyenne des contrôles", "moyenne fondatrice", "concept")],
 [(0, 1, "data_dependency", "découle")]),
("Le prix du pain dépend du cours du blé.",
 [("Le prix du pain", "le prix du pain", "concept"), ("du cours du blé", "cours déterminant", "concept")],
 [(0, 1, "data_dependency", "dépend")]),
("La victoire repose sur une défense de fer.",
 [("La victoire", "la victoire", "evenement"), ("une défense de fer", "défense fondatrice", "processus")],
 [(0, 1, "data_dependency", "repose")]),
("Sa décision dépend de l'avis du médecin.",
 [("Sa décision", "sa décision", "processus"), ("l'avis du médecin", "avis déterminant", "concept")],
 [(0, 1, "data_dependency", "dépend")]),
("Le diagnostic découle des résultats du scanner.",
 [("Le diagnostic", "le diagnostic", "concept"), ("des résultats du scanner", "résultats fondants", "evenement")],
 [(0, 1, "data_dependency", "découle")]),
("La floraison dépend des gelées tardives.",
 [("La floraison", "la floraison", "processus"), ("des gelées tardives", "gelées déterminantes", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("L'équilibre du budget repose sur les recettes fiscales.",
 [("L'équilibre du budget", "équilibre budgétaire", "etat_global"), ("les recettes fiscales", "recettes fondatrices", "concept")],
 [(0, 1, "data_dependency", "repose")]),
("Le classement dépend de la différence de buts.",
 [("Le classement", "le classement", "concept"), ("la différence de buts", "différence déterminante", "concept")],
 [(0, 1, "data_dependency", "dépend")]),
("Sa réputation repose sur vingt ans de chroniques.",
 [("Sa réputation", "sa réputation", "concept"), ("vingt ans de chroniques", "chroniques fondatrices", "processus")],
 [(0, 1, "data_dependency", "repose")]),
("La cuisson dépend de l'épaisseur de la pièce.",
 [("La cuisson", "la cuisson", "processus"), ("l'épaisseur de la pièce", "épaisseur déterminante", "concept")],
 [(0, 1, "data_dependency", "dépend")]),
("Le succès du film découle du bouche-à-oreille.",
 [("Le succès du film", "le succès du film", "evenement"), ("du bouche-à-oreille", "bouche-à-oreille fondant", "processus")],
 [(0, 1, "data_dependency", "découle")]),
("La traversée dépend de la marée haute.",
 [("La traversée", "la traversée", "processus"), ("la marée haute", "marée déterminante", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("Son autorité repose sur la peur des sanctions.",
 [("Son autorité", "son autorité", "concept"), ("la peur des sanctions", "peur fondatrice", "etat_local")],
 [(0, 1, "data_dependency", "repose")]),
("La guérison dépend de la précocité du traitement.",
 [("La guérison", "la guérison", "processus"), ("la précocité du traitement", "précocité déterminante", "concept")],
 [(0, 1, "data_dependency", "dépend")]),
("Le charme du village découle de ses ruelles pavées.",
 [("Le charme du village", "le charme du village", "concept"), ("ses ruelles pavées", "ruelles fondatrices", "etat_local")],
 [(0, 1, "data_dependency", "découle")]),
("La qualité du vin dépend de l'ensoleillement d'août.",
 [("La qualité du vin", "la qualité du vin", "concept"), ("l'ensoleillement d'août", "ensoleillement déterminant", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("La solidité du pont repose sur ses piles de granit.",
 [("La solidité du pont", "la solidité du pont", "etat_local"), ("ses piles de granit", "piles fondatrices", "entite")],
 [(0, 1, "data_dependency", "repose")]),
("L'ambiance dépend du choix des invités.",
 [("L'ambiance", "ambiance", "etat_local"), ("du choix des invités", "choix déterminant", "processus")],
 [(0, 1, "data_dependency", "dépend")]),
("Sa thèse découle de trois années d'archives.",
 [("Sa thèse", "sa thèse", "concept"), ("trois années d'archives", "archives fondatrices", "processus")],
 [(0, 1, "data_dependency", "découle")]),
("Le moral des troupes dépend des nouvelles du front.",
 [("Le moral des troupes", "le moral des troupes", "etat_global"), ("des nouvelles du front", "nouvelles déterminantes", "evenement")],
 [(0, 1, "data_dependency", "dépend")]),
("La crédibilité du journal repose sur ses sources.",
 [("La crédibilité du journal", "crédibilité du journal", "concept"), ("ses sources", "sources fondatrices", "entite")],
 [(0, 1, "data_dependency", "repose")]),
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
    all_props = ([(t, c, e, "control_dependency") for t, c, e in CD]
                 + [(t, c, e, "prevent") for t, c, e in PREVENT]
                 + [(t, c, e, "filter") for t, c, e in FILTER]
                 + [(t, c, e, "data_dependency") for t, c, e in DATADEP])
    assert len(CD) == 45 and len(PREVENT) == 33 and len(FILTER) == 27 and len(DATADEP) == 24, (
        len(CD), len(PREVENT), len(FILTER), len(DATADEP))
    sentences = []
    for n, (text, clauses, arretes, _fam) in enumerate(all_props, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-p1b"})
        edges = []
        for src, dst, rel, marqueur in arretes:
            edges.append({"sources": [f"n{src:03d}"], "target": f"n{dst:03d}",
                          "relation": rel, "explicit": True, "confidence": 0.0,
                          "negated": False, "third": None,
                          "marker_token": _marker_id(toks, marqueur)})
        sentences.append({"id": f"p1b{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "mixte",
                          "cir": {"nodes": nodes, "edges": edges}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-rares-p1b", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver P1b classes rares.")
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
