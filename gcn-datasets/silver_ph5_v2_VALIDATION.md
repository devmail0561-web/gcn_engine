# Ph5-bis — validation PV1 (pipeline v2 : fusion B-a + subdivision B-b + gate resserré)

Juger CHAQUE paire : [oui]/[non]+motif/[corriger→X].

## v2-003 silver-B
Il postule que pour qu'un effet puisse être attribué à une cause, l'événement déclencheur (la cause) doit se produire avant l'apparition de l'effet.
- n0 « Il postule que »
- n1 « pour qu'un effet puisse être attribué à une »
- n2 « cause, l'événement déclencheur (la cause) doit se produire »
- n3 « avant l'apparition de l'effet. »
- e0: n0 --restrictive--> n1 (conf 0.43, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.39, marge 0.31)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.41, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --causale--> n3 (conf 0.56, marge 0.56)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.54, marge 0.65)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-005 silver-B
DÉCRYPTAGE- Omniprésentes dans les aliments industriels, ces substances, souvent controversées, commencent à faire l’objet d’études pour établir l’impact sur la santé à long terme.
- n0 « DÉCRYPTAGE- Omniprésentes dans les aliments industriels, ces substances, souvent controver »
- n1 « pour établir l’impact sur la santé à long terme. »
- e0: n0 --negative--> n1 (conf 0.31, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-016 silver-B
DÉCRYPTAGE - TotalEnergies et Engie sont en pointe pour fournir aux acteurs des données et de l’intelligence artificielle l’électricité verte dont ils ont besoin.
- n0 « DÉCRYPTAGE - TotalEnergies et Engie sont en pointe »
- n1 « pour fournir aux acteurs des données et de l’intelligence artificielle l’électricité verte »
- e0: n0 --negative--> n1 (conf 0.41, marge 0.49)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-018 silver-B
DÉCRYPTAGE - Des scientifiques australiens montrent que l’allaitement maternel stimule les défenses immunitaires contre le cancer du sein, expliquant un phénomène connu, mais encore mal compris.
- n0 « DÉCRYPTAGE - Des scientifiques australiens montrent que l’allaitement maternel stimule les »
- n1 « contre le cancer du sein, expliquant un phénomène connu, »
- n2 « mais encore mal compris. »
- e0: n0 --negative--> n1 (conf 0.49, marge 1.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.62, marge 1.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.40, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-019 silver-B
L’astronaute française a terminé ce mardi après-midi de remplacer une antenne de télécommunication à l’extérieur de la Station spatiale internationale.
- n0 « L’astronaute française a terminé ce mardi »
- n1 « après-midi de remplacer une antenne de télécommunication à l’extérieur de la Station spati »
- e0: n0 --causale--> n1 (conf 0.83, marge 1.93)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-022 silver-B
Flashez ce QR Code pour accéder à la page de ce numéro sur Cairn.info.
- n0 « Flashez ce QR Code »
- n1 « pour accéder à la page de ce numéro sur Cairn.info. »
- e0: n0 --negative--> n1 (conf 0.50, marge 0.66)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-028 silver-B
Notre hypothèse était qu’avec cette nouvelle version d’Antoid, il serait possible d’accroître massivement le nombre d’entités en interaction (donc d’événements) en ajoutant simplement de nouveaux noeuds à la grappe d’ordinateurs, sans effondrement des performances.
- n0 « Notre hypothèse était qu’avec cette nouvelle version d’Antoid, il serait possible d’accroî »
- n1 « donc d’événements) en ajoutant simplement de nouveaux noeuds à la grappe d’ordinateurs, »
- n2 « sans effondrement des performances. »
- e0: n0 --restrictive--> n1 (conf 0.60, marge 0.73)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.66, marge 1.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.43, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-032 silver-B
Sur le terrain, la situation est catastrophique pour les Palestiniens et peu rassurante pour les Israéliens.
- n0 « Sur le terrain, la situation est catastrophique »
- n1 « pour les Palestiniens et peu rassurante pour les Israéliens. »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.80)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-033 silver-B
« La couverture vaccinale a progressé l’hiver dernier, mais nous sommes encore trop loin des niveaux de protection nécessaires », a insisté jeudi devant la presse Grégory Emery, directeur général de la santé.
- n0 « « La couverture vaccinale a progressé l’hiver dernier, »
- n1 « mais nous sommes encore trop loin des niveaux de protection nécessaires », a insisté jeudi »
- n2 « devant la presse Grégory Emery, directeur général de la santé. »
- e0: n0 --causale--> n1 (conf 0.71, marge 1.48)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.35, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.44, marge 0.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-037 silver-B
Suivant leurs conseils, j’ai donc décidé de subir une opération pour le faire retirer.
- n0 « Suivant leurs conseils, j’ai »
- n1 « donc décidé de subir une opération »
- n2 « pour le faire retirer. »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.45, marge 0.47)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.56, marge 0.89)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-039 silver-B
- et al. - DOURIS, Marie - et ROMAN, Pascal, https://doi.org/10.3917/ep.101.0147 Notes - [1] Loi n°2022-140 du 7 février 2022, relative à la protection des enfants, j.o.
- n0 « - et al. »
- n1 « - DOURIS, Marie - et ROMAN, Pascal, https://doi.org/10.3917/ep.101.0147 Notes - [1] Loi n° »
- e0: n0 --negative--> n1 (conf 0.35, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-046 silver-B
Cela inclut non seulement la gestion des données des utilisateurs, mais également la sécurité des informations et le respect des lois en vigueur.
- n0 « Cela inclut non seulement la gestion des données des utilisateurs, »
- n1 « mais également la sécurité des informations et le respect des lois en vigueur. »
- e0: n0 --negative--> n1 (conf 0.42, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-048 silver-B
Si c’est une victoire de santé publique, il faut néanmoins rester vigilants.
- n0 « Si c’est une victoire de santé publique, il faut »
- n1 « néanmoins rester vigilants. »
- e0: n0 --negative--> n1 (conf 0.50, marge 0.95)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-055 silver-B
« Un coup de tonnerre dans un ciel serein »: beaucoup décrivent ainsi l’annonce d’un cancer.
- n0 « « Un coup de tonnerre dans un ciel serein »: beaucoup décrivent »
- n1 « ainsi l’annonce d’un cancer. »
- e0: n0 --causale--> n1 (conf 0.89, marge 2.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-057 silver-B
«Jamais on n’aurait pu faire ça tout seuls!» Chacun à son rythme.
- n0 « «Jamais on n’aurait pu faire ça tout »
- n1 « seuls! »
- n2 « » Chacun à son rythme. »
- e0: n0 --causale--> n1 (conf 0.84, marge 2.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.46, marge 0.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.38, marge 0.33)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-065 silver-B
J’ai essayé de rentrer chez moi, mais au bout de 100 mètres, j’ai senti que ça n’allait pas le faire.
- n0 « J’ai essayé de rentrer chez moi, »
- n1 « mais au bout de 100 mètres, j’ai senti que ça n’allait pas le faire. »
- e0: n0 --negative--> n1 (conf 0.40, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-067 silver-B
aussi Daniel Thym, Respect for Private and Family Life under article 8 ECHR in Immigration Cases : a Human Right to Regularize Illegal Stay ?, International and Comparative Law Quarterly, 2008, vol. 57, p. 87-112.
- n0 « aussi Daniel Thym, Respect for Private and Family Life under article 8 ECHR in Immigration »
- n1 « , International and Comparative Law Quarterly, 2008, vol. »
- n2 « 57, p. 87-112. »
- e0: n0 --negative--> n1 (conf 0.35, marge 0.44)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.38, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.33, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-069 silver-B
Et puis, vous pourriez devoir vous désister bientôt pour M. Philippe.
- n0 « puis, vous pourriez devoir vous désister bientôt »
- n1 « pour M. Philippe. »
- e0: n0 --systemique--> n1 (conf 0.44, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-071 silver-B
Plusieurs autres symptômes peuvent être présents, sans être pour autant requis pour le diagnostic de l’EHS.
- n0 « Plusieurs autres symptômes peuvent être présents, »
- n1 « sans être »
- n2 « pour autant requis pour le diagnostic de l’EHS. »
- e0: n0 --restrictive--> n1 (conf 0.60, marge 0.92)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.48, marge 0.68)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.40, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-072 silver-B
"La législation est un guide, pas une contrainte." 📜 Protection des Données Personnelles dans les Applications Mobiles La protection des données personnelles est au cœur des préoccupations des utilisateurs d’applications.
- n0 « "La législation est un guide, pas une contrainte. »
- n1 « " 📜 Protection des Données Personnelles dans les Applications Mobiles La protection des do »
- e0: n0 --negative--> n1 (conf 0.34, marge 0.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-073 silver-B
DÉCRYPTAGE - Ce phénomène est d’autant plus étonnant que la fonte des glaces polaires tend au contraire à ralentir sa vitesse de rotation, selon une étude publiée dans la revue Nature.
- n0 « DÉCRYPTAGE - Ce phénomène est d’autant plus étonnant que la fonte des glaces polaires tend »
- n1 « au contraire à ralentir sa vitesse de rotation, selon une étude publiée dans la revue Natu »
- e0: n0 --causale--> n1 (conf 0.54, marge 0.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-076 silver-B
Mais il ne doit pas être négligé pour autant car il s’agit du premier signe visible d’un cancer de la vessie.
- n0 « Mais il ne doit pas être négligé »
- n1 « pour autant »
- n2 « car il s’agit du premier signe visible d’un cancer de la vessie. »
- e0: n0 --restrictive--> n1 (conf 0.53, marge 0.55)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.48, marge 1.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.50, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-078 silver-B
- [3] Cf. CIA World fact book (1er janvier 2012) et Voir tableau 1.
- n0 « - [3] Cf. »
- n1 « CIA World fact book (1er janvier 2012) et Voir tableau 1. »
- e0: n0 --negative--> n1 (conf 0.60, marge 0.81)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-079 silver-B
Par ailleurs, il y a un ralentissement dans la compréhension de cette maladie à cause du manque de modèle animal représentatif de la SSc humaine.
- n0 « Par ailleurs, il y a un ralentissement dans la compréhension de cette maladie »
- n1 « à cause du manque de modèle animal représentatif de la SSc humaine. »
- e0: n0 --restrictive--> n1 (conf 0.65, marge 0.98)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-083 silver-B
« Et enfin, on chasse l’esprit de défaite de nos âmes et de nos cœurs, et c’est ça qui est le principal.
- n0 « « Et »
- n1 « enfin, on chasse l’esprit de défaite de nos âmes et de nos cœurs, et c’est ça qui est le p »
- e0: n0 --restrictive--> n1 (conf 0.81, marge 1.87)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-086 silver-B
Cette chaleur emmagasinée est ensuite redistribuée dans l’océan et l’atmosphère, modifiant… Cet article est réservé aux abonnés.
- n0 « Cette chaleur emmagasinée est »
- n1 « ensuite redistribuée dans l’océan et l’atmosphère, modifiant… »
- n2 « Cet article est »
- n3 « réservé aux abonnés. »
- e0: n0 --negative--> n1 (conf 0.40, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.47, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.74, marge 1.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.42, marge 0.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.80, marge 1.75)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-090 silver-B
cit., p. 662, concernant les problèmes de diffusion de Fécondité.
- n0 « cit. »
- n1 « , p. 662, concernant les problèmes de diffusion de Fécondité. »
- e0: n0 --restrictive--> n1 (conf 0.43, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-091 silver-B
« Ce qui a brûlé n’a pas de valeur patrimoniale en tant que tel, il s’agissait surtout d’étals en structure légère », a toutefois précisé M. Weil.
- n0 « « Ce qui a brûlé n’a pas de valeur patrimoniale en tant que tel, il s’agissait surtout d’é »
- n1 « toutefois précisé M. Weil. »
- e0: n0 --causale--> n1 (conf 0.94, marge 3.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-092 silver-B
La tempête Isaias a fait au moins quatre morts dans le sud des Etats-Unis, ont annoncé les autorités locales, samedi 10 octobre, tandis que des centaines de milliers d’usagers restent sans électricité.
- n0 « La tempête Isaias a fait au moins quatre morts dans le sud des Etats-Unis, ont annoncé les »
- n1 « tandis que des centaines de milliers d’usagers restent »
- n2 « sans électricité. »
- e0: n0 --restrictive--> n1 (conf 0.38, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.62, marge 1.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.43, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-098 silver-B
Trop souvent assimilée à une simple « toux du fumeur », la bronchopneumopathie chronique obstructive entraîne des difficultés respiratoires.
- n0 « Trop souvent assimilée à une simple « toux du fumeur », la bronchopneumopathie chronique o »
- n1 « entraîne des difficultés respiratoires. »
- e0: n0 --restrictive--> n1 (conf 0.60, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-099 silver-B
- [15] Cf. par exemple PALIER B., Gouverner la sécurité sociale.
- n0 « - [15] Cf. »
- n1 « par exemple PALIER B. »
- n2 « , Gouverner la sécurité sociale. »
- e0: n0 --negative--> n1 (conf 0.46, marge 0.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.36, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.45, marge 0.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-100 silver-B
L’utilisation de la voiture est une solution de facilité pour charger tout le monde le matin et faire le circuit de l’école vers mon lieu de travail.
- n0 « L’utilisation de la voiture est une solution de facilité »
- n1 « pour charger tout le monde le matin et faire le circuit de l’école vers mon lieu de travai »
- e0: n0 --negative--> n1 (conf 0.36, marge 0.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-104 silver-B
Excusez moi, je renouvelle l'expérience pour vérifier la justesse de la mesure, à la vôtre!
- n0 « Excusez moi, je renouvelle l'expérience »
- n1 « pour vérifier la justesse de la mesure, à la vôtre! »
- e0: n0 --restrictive--> n1 (conf 0.33, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-106 silver-B
Le scrutin, uniquement électronique, a été perturbé par une cyberattaque, lancée vendredi matin à l’ouverture du scrutin, qui a ralenti les connexions à la plateforme ad hoc et a obligé les organisateurs à prolonger le vote jusqu’à dimanche soir.
- n0 « Le scrutin, »
- n1 « uniquement électronique, a été perturbé par une cyberattaque, lancée vendredi matin à l’ou »
- e0: n0 --restrictive--> n1 (conf 0.69, marge 1.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-107 silver-B
- [23] L’exercice exclusif s’envisage aussi logiquement lorsqu’un parent est décédé ou lorsqu’un des parent
- n0 « - [23] L’exercice exclusif s’envisage aussi logiquement »
- n1 « lorsqu’un parent est décédé ou lorsqu’un des parent »
- e0: n0 --negative--> n1 (conf 0.62, marge 0.85)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-110 silver-B
Ça me procure un tel plaisir que je suis prêt à mettre de l’argent pour avoir ce plaisir.
- n0 « Ça me procure un tel plaisir que je suis prêt à mettre de l’argent »
- n1 « pour avoir ce plaisir. »
- e0: n0 --negative--> n1 (conf 0.67, marge 1.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-111 silver-B
Les données cliniques de cette étude ont été analysées par la CNEDiMTS ce qui a conduit à l’attribution d’un service attendu suffisant et donc à obtenir le remboursement du dispositif aux assurés sociaux.
- n0 « Les données cliniques de cette étude ont été analysées par la CNEDiMTS ce qui a »
- n1 « conduit à l’attribution d’un service attendu suffisant et »
- n2 « donc à obtenir le remboursement du dispositif aux assurés sociaux. »
- e0: n0 --negative--> n1 (conf 0.48, marge 0.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.44, marge 0.49)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.53, marge 0.55)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-114 silver-B
Adobe Stock RÉCIT - Le 13 octobre a lieu la Journée mondiale de lutte contre le cancer du sein métastatique.
- n0 « Adobe Stock RÉCIT - Le 13 octobre a lieu la Journée mondiale de »
- n1 « lutte contre le cancer du sein métastatique. »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.46)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-118 silver-B
Le président de la République négocie avec son homologue allemand, Friedrich Merz, une réglementation pour préserver l’industrie européenne de l’offensive chinoise sur les véhicules électriques et hybrides.
- n0 « Le président de la République négocie avec son homologue allemand, Friedrich Merz, une rég »
- n1 « pour préserver l’industrie européenne de l’offensive chinoise sur les véhicules électrique »
- e0: n0 --negative--> n1 (conf 0.33, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-122 silver-B
Cependant, si la force de l'association est faible, il est possible que d'autres variables non contrôlées (facteurs de confusion) influencent ce lien observé.
- n0 « Cependant, »
- n1 « si la force de l'association est faible, il est possible que d'autres variables non contrô »
- e0: n0 --restrictive--> n1 (conf 0.54, marge 0.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-123 silver-B
- Baudry, P. https://doi.org/10.3917/rfsp.712.0219 Citer cet article - Baudry, P. - Baudry, Pierre.
- n0 « - Baudry, P. https://doi.org/10.3917/rfsp.712.0219 Citer cet article - Baudry, P. »
- n1 « - Baudry, Pierre. »
- e0: n0 --negative--> n1 (conf 0.33, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-124 silver-B
La référence aux ouvrages de stockage d’eau conçus pour puiser dans les nappes souterraines l’hiver, dont la très polémique réserve agricole de Sainte-Soline (Deux-Sèvres), se veut un brin provocatrice.
- n0 « La référence aux ouvrages de stockage d’eau conçus »
- n1 « pour puiser dans les nappes souterraines l’hiver, dont la très polémique réserve agricole  »
- e0: n0 --causale--> n1 (conf 0.47, marge 0.70)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-126 silver-B
De plus, hnRNPM et p54 nrb/NonO se lient au domaine CTD de l'ARN polymérase II, et leur fonction d'activateur de l'IRES nécessite une étape transcriptionnel.
- n0 « De plus, hnRNPM et p54 nrb/NonO se lient au domaine CTD de l'ARN polymérase II, et leur fo »
- n1 « nécessite une étape transcriptionnel. »
- e0: n0 --causale--> n1 (conf 0.52, marge 0.68)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-130 silver-B
En 1968, Per Asbrink, gouverneur de la Banque de Suède, bataille en vain pour arracher au gouvernement social-démocrate de Tage Erlander (1901-1985) l’indépendance de son institution.
- n0 « En 1968, Per Asbrink, gouverneur de la Banque de Suède, bataille en vain »
- n1 « pour arracher au gouvernement social-démocrate de Tage Erlander (1901-1985) l’indépendance »
- e0: n0 --negative--> n1 (conf 0.44, marge 0.46)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-132 silver-B
D’un poids de 100 kilos, elle bénéficie d’une petite mare dans son enclos et d’un box équipé d’un ventilateur pour les fortes chaleurs.
- n0 « D’un poids de 100 kilos, elle bénéficie d’une petite mare dans son enclos et d’un box équi »
- n1 « pour les fortes chaleurs. »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-136 silver-B
Sophie Adenot et ses camarades seront ensuite transportés sur la terre ferme en hélicoptère, puis ils prendront un avion pour Houston au Texas, où se trouve le centre de la NASA chargé des astronautes.
- n0 « Sophie Adenot et ses camarades seront »
- n1 « ensuite transportés sur la terre ferme en hélicoptère, »
- n2 « puis ils prendront un avion »
- n3 « pour Houston au Texas, où se trouve le centre de la NASA chargé des astronautes. »
- e0: n0 --negative--> n1 (conf 0.33, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.41, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.77, marge 1.74)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.43, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.61, marge 1.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-138 silver-B
Ainsi, même si la temporalité est indispensable, elle ne peut être considérée comme suffisante pour établir une causalité de manière définitive.
- n0 « Ainsi, »
- n1 « même si la temporalité est indispensable, elle ne peut être considérée comme suffisante »
- n2 « pour établir une causalité de manière définitive. »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.54, marge 0.96)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.42, marge 0.03)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-139 silver-B
Les mues du papillon, celles de l’embryon en développement ou des plantes à fleurs, saison après saison, en offrent de fameux exemples.
- n0 « Les mues du papillon, celles de l’embryon en développement ou des plantes à fleurs, saison »
- n1 « après saison, en offrent de fameux exemples. »
- e0: n0 --restrictive--> n1 (conf 0.41, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-142 silver-B
Des articles sur le sujet obtenus au moyen de sites spécialisés en oncologie ont également été consultés ainsi que leur liste de références.
- n0 « Des articles sur le sujet obtenus au moyen de sites spécialisés en oncologie ont également »
- n1 « ainsi que leur liste de références. »
- e0: n0 --restrictive--> n1 (conf 0.45, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-150 silver-B
Jugé dangereux en cas de survie, il sera abattu à l’aide d’un poignard par des chasseurs, sous l’œil médusé des propriétaires.
- n0 « Jugé dangereux »
- n1 « en cas de survie, il sera abattu à l’aide d’un poignard par des chasseurs, sous l’œil médu »
- e0: n0 --restrictive--> n1 (conf 0.57, marge 1.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-154 silver-B
Trois siècles plus tard, tout a été bouleversé autour d’elle, mais elle subsiste, intacte, en souvenir de ce jour de 1719 où, par la volonté du souverain, la petite ville est devenue un port franc.
- n0 « Trois siècles plus tard, tout a été bouleversé autour d’elle, »
- n1 « mais elle subsiste, intacte, en souvenir de ce jour de 1719 où, par la volonté du souverai »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-158 silver-B
Des chercheurs de Singapour ont même annoncé fin septembre avoir atteint le seuil de 10−19, soit une seconde… Cet article est réservé aux abonnés.
- n0 « Des chercheurs de Singapour ont même annoncé fin septembre avoir atteint le seuil de 10−19 »
- n1 « Cet article est »
- n2 « réservé aux abonnés. »
- e0: n0 --negative--> n1 (conf 0.34, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.46, marge 0.98)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.84, marge 1.91)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-160 silver-B
L’étude juridique application mobile est cruciale pour assurer la conformité et la pérennité de tout projet numérique.
- n0 « L’étude juridique application mobile est cruciale »
- n1 « pour assurer la conformité et la pérennité de tout projet numérique. »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.44)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-163 silver-B
Cette étude permet de corréler les propriétés texturales de ces matériaux avec celles de la conductivité et de l’adsorption.
- n0 « Cette étude »
- n1 « permet de corréler les propriétés texturales de ces matériaux avec celles de la conductivi »
- e0: n0 --restrictive--> n1 (conf 0.61, marge 0.85)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-164 silver-B
Des cas isolés rapportent une efficacité d’un antidépresseur tricyclique, la clomipramine, d’une benzodiazépine, le clonazépam, d’antiépileptiques, comme le topiramate et la carbamazépine, ainsi que d’inhibiteurs calciques, comme la flunarizine et la nifédipine à libération prolongée.
- n0 « Des cas isolés rapportent une efficacité d’un antidépresseur tricyclique, la clomipramine, »
- n1 « ainsi que d’inhibiteurs calciques, comme la flunarizine et la nifédipine à libération prol »
- e0: n0 --restrictive--> n1 (conf 0.40, marge 0.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-165 silver-B
Pour les spécialistes du vieillissement, il constitue donc une pièce de choix.
- n0 « Pour les spécialistes du vieillissement, il constitue »
- n1 « donc une pièce de choix. »
- e0: n0 --negative--> n1 (conf 0.39, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-169 silver-B
Si certains joueurs présentent des mécanismes pervers, il n’est pas étonnant alors que ce comportement soit en majorité un comportement masculin.
- n0 « Si certains joueurs présentent des mécanismes pervers, il n’est pas étonnant »
- n1 « alors que ce comportement soit en majorité un comportement masculin. »
- e0: n0 --causale--> n1 (conf 0.87, marge 2.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-171 silver-B
Qu’il s’agisse des secours en montagne ou en mer, le cancer d’un fumeur, la cirrhose d’un buveur, il me semble nécessaire de responsabiliser les gens en leur demandant une participation financière significative.
- n0 « Qu’il s’agisse des secours en montagne ou en mer, le cancer d’un fumeur, la cirrhose d’un  »
- n1 « nécessaire de responsabiliser les gens en leur demandant une participation financière sign »
- e0: n0 --causale--> n1 (conf 0.68, marge 1.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-174 silver-B
Télécharger CHRONIQUES CHRONIQUE JURIDIQUE - Rémi Moreau Diffusion numérique : 14 novembre 2023 URI : https://id.erudit.org/iderudit/1107533ar Copier le permalien de l’articleAdresse copiée.Une erreur s’est produite.
- n0 « Télécharger CHRONIQUES CHRONIQUE JURIDIQUE - Rémi Moreau Diffusion numérique : 14 novembre »
- n1 « Une erreur s’est produite. »
- e0: n0 --restrictive--> n1 (conf 0.29, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-177 silver-B
Ils prônent en public un équilibre faune-flore mais ne le définissent jamais et obligent leurs adjudicataires (qu'ils tiennent par les c....) à des pratiques extrêmes.
- n0 « Ils prônent en public un équilibre faune-flore »
- n1 « mais ne le définissent jamais et obligent leurs adjudicataires (qu'ils tiennent par les c. »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-179 silver-B
Dans notre corps, les protéines ne sont composées que d’acides aminés qui sont dans une forme dite « gauche », alors que la forme droite pourrait aussi exister.
- n0 « Dans notre corps, les protéines ne sont composées que d’acides aminés qui sont dans une fo »
- n1 « alors que la forme droite pourrait aussi exister. »
- e0: n0 --negative--> n1 (conf 0.50, marge 0.31)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-182 silver-B
Les médecins comprennent alors que cette septuagénaire ne souffre pas de maux de tête ressentis comme des battements, survenant la nuit et suffisamment intenses pour perturber la vie quotidienne.
- n0 « Les médecins comprennent »
- n1 « alors que cette septuagénaire ne souffre pas de maux de tête ressentis comme des battement »
- n2 « pour perturber la vie quotidienne. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.63, marge 1.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.38, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-185 silver-B
Il faut, en revanche, prendre en compte certains troubles qui peuvent parfois perturber la libido.
- n0 « Il faut, »
- n1 « en revanche, prendre en compte certains troubles qui peuvent parfois perturber la libido. »
- e0: n0 --restrictive--> n1 (conf 0.64, marge 1.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-191 silver-B
La hausse de l’espérance de vie a décéléré sur le continent, et ce bien avant le Covid.
- n0 « La hausse de l’espérance de vie a décéléré sur le continent, et ce bien »
- n1 « avant le Covid. »
- e0: n0 --negative--> n1 (conf 0.34, marge 0.31)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-194 silver-B
Villanueva C, Balanzo J, Novella MT, et al. Nadolol plus isosorbide mononitrate compared with sclerotherapy for the prevention of variceal rebleeding.
- n0 « Villanueva C, Balanzo J, Novella MT, et al. »
- n1 « Nadolol plus isosorbide mononitrate compared with sclerotherapy for the prevention of vari »
- e0: n0 --restrictive--> n1 (conf 0.35, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-197 silver-B
Une procédure de dissolution a été engagée contre l’association Egalité et réconciliation (E&R), fondée par l’essayiste d’extrême droite Alain Soral, a-t-on appris, dimanche 11 octobre, auprès du ministère de l’intérieur, confirmant une information de Libération.
- n0 « Une procédure de dissolution a été engagée »
- n1 « contre l’association Egalité et réconciliation (E&R), fondée par l’essayiste d’extrême dro »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-203 silver-B
L’objectif des deux protocoles est d’obtenir une évaluation standardisée, homogène et neutre des performances des SDA déployés sur les parcs pour réduire les mortalités d’oiseaux par collision.
- n0 « L’objectif des deux protocoles est d’obtenir une évaluation standardisée, homogène et neut »
- n1 « pour réduire les mortalités d’oiseaux par collision. »
- e0: n0 --negative--> n1 (conf 0.40, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-205 silver-B
Face à l’envolée des prix de l’essence et du fioul, l’argument de l’indépendance énergétique est le principal motif de soutien (pour 38% des Français selon le baromètre).
- n0 « Face à l’envolée des prix de l’essence et du fioul, l’argument de l’indépendance énergétiq »
- n1 « pour 38% des Français selon le baromètre). »
- e0: n0 --negative--> n1 (conf 0.41, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-206 silver-B
Il explique comment se délivrer de l’obsession de la productivité pour retrouver le sens des limites, afin de savourer à nouveau la vie.
- n0 « Il explique comment se délivrer de l’obsession de la productivité »
- n1 « pour retrouver le sens des limites, »
- n2 « afin de savourer à nouveau la vie. »
- e0: n0 --systemique--> n1 (conf 0.42, marge 0.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.35, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.46, marge 0.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-208 silver-B
Portée par une association, une anomalie à ce niveau, la course réunionnaise d’ultra-trail, qui s’élance jeudi 15 octobre, défend son indépendance malgré une gouvernance discutée et des propositions de rachat.
- n0 « Portée par une association, une anomalie à ce niveau, la course réunionnaise d’ultra-trail »
- n1 « malgré une gouvernance discutée et des propositions de rachat. »
- e0: n0 --causale--> n1 (conf 0.65, marge 1.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-209 silver-B
Singulièrement barrée, la création qui mérite d’être découverte sans en savoir trop impose sa drôlerie entre théâtre, danse et mime au théâtre de l’Atelier.
- n0 « Singulièrement barrée, la création qui mérite d’être découverte »
- n1 « sans en savoir trop impose sa drôlerie entre théâtre, danse et mime au théâtre de l’Atelie »
- e0: n0 --restrictive--> n1 (conf 0.37, marge 0.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-212 silver-B
Toutefois, une association statistiquement forte ne garantit pas à elle seule une causalité, car il est nécessaire de vérifier que cette relation ne soit pas induite par des facteurs de confusion ou d'autres biais méthodologiques.
- n0 « Toutefois, une association statistiquement forte ne garantit pas à elle seule une causalit »
- n1 « car il est »
- n2 « nécessaire de vérifier que cette relation ne soit pas induite par des facteurs de confusio »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.50, marge 0.48)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.96, marge 3.61)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-214 silver-B
Il s’agit de maintenir et de renforcer la catégorie des normes du jus cogens tout en les assortissant de moyens efficaces en vue de garantir leur respect.
- n0 « Il s’agit de maintenir et de renforcer la catégorie des normes du jus cogens tout en les a »
- n1 « en vue de garantir leur respect. »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-215 silver-B
Une évolution qui, derrière une promesse d’ouverture, continue donc de soulever plusieurs questions sensibles.
- n0 « Une évolution qui, derrière une promesse d’ouverture, continue »
- n1 « donc de soulever plusieurs questions sensibles. »
- e0: n0 --restrictive--> n1 (conf 0.51, marge 0.47)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-216 silver-B
ENTRETIEN - Le gouvernement vient d’annoncer un quatrième plan sur les maladies rares, porteur de progrès majeurs si des moyens financiers permettent de les réaliser.
- n0 « ENTRETIEN - Le gouvernement vient d’annoncer un quatrième plan sur les maladies rares, por »
- n1 « si des moyens financiers permettent de les réaliser. »
- e0: n0 --causale--> n1 (conf 0.36, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-218 silver-B
cit., p. 196-197 : « Mes visions, à moi, d’un avenir meilleur où les hommes vivront dans une solidarité étroite et parfaite, n’ont pas la rigueur d’une doctrine.
- n0 « cit. »
- n1 « , p. 196-197 : « Mes visions, à moi, d’un avenir meilleur où les hommes vivront dans une s »
- e0: n0 --systemique--> n1 (conf 0.37, marge 0.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-221 silver-B
Elle observe un recours massif et transclasse aux opérations, mais qui demeure un secret de Polichinelle.
- n0 « Elle observe un recours massif et transclasse aux opérations, »
- n1 « mais qui demeure un secret de Polichinelle. »
- e0: n0 --causale--> n1 (conf 0.83, marge 2.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-224 silver-B
Panthat Sitdatvirade - stock.adobe.com Les efforts de protection montrent que l’érosion du vivant n’est pas une fatalité, explique le dernier grand rapport global du WWF, mais la fenêtre pour agir se rétrécit.
- n0 « Panthat Sitdatvirade - stock.adobe.com Les efforts de protection montrent que l’érosion du »
- n1 « mais la fenêtre »
- n2 « pour agir se rétrécit. »
- e0: n0 --negative--> n1 (conf 0.42, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.45, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.28, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-225 silver-B
Avant tout parce que l’engagement politique du trentenaire, à l’allure toujours juvénile, a commencé au sein du lycée Roland-Garros, celui des élèves venus manifester devant sa mairie.
- n0 « Avant tout »
- n1 « parce que l’engagement politique du trentenaire, à l’allure toujours juvénile, a commencé  »
- n2 « devant sa mairie. »
- e0: n0 --systemique--> n1 (conf 0.89, marge 2.85)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.85, marge 2.48)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.41, marge 0.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-226 silver-B
En 1988, lorsque l’Initiative mondiale pour l’éradication de la poliomyélite est créée, on estime encore à 350 000 le nombre annuel d’enfants atteints par le poliovirus sauvage.
- n0 « En 1988, »
- n1 « lorsque l’Initiative mondiale »
- n2 « pour l’éradication de la poliomyélite est créée, on estime encore à 350 000 le nombre annu »
- e0: n0 --restrictive--> n1 (conf 0.47, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.48, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.34, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-228 silver-B
- [13] Cf. notamment INGLEHART R., Modernization and Postmodernization : Cultural, Economic, and Political Change in 43 societies, Princeton (NJ), Princeton University Press, 1997.
- n0 « - [13] Cf. »
- n1 « notamment INGLEHART R., Modernization and Postmodernization : Cultural, Economic, and Poli »
- e0: n0 --negative--> n1 (conf 0.61, marge 0.96)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-236 silver-B
DÉCRYPTAGE - Une vaste étude française confirme que l’incidence du cancer du sein avant 50 ans a bel et bien augmenté de façon continue depuis trente ans.
- n0 « DÉCRYPTAGE - Une vaste étude française confirme que l’incidence du cancer du sein »
- n1 « avant 50 ans a bel et bien augmenté de façon continue depuis trente ans. »
- e0: n0 --negative--> n1 (conf 0.52, marge 0.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-239 silver-B
Une bataille juridique s’était ensuivie après la décision de la préfecture de l’Aube de saisir le suidé et de l’euthanasier.
- n0 « Une bataille juridique s’était ensuivie »
- n1 « après la décision de la préfecture de l’Aube de saisir le suidé et de l’euthanasier. »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-245 silver-B
Dans cet article, nous considérons uniquement la protection des données personnelles des consommateurs.
- n0 « Dans cet article, nous considérons »
- n1 « uniquement la protection des données personnelles des consommateurs. »
- e0: n0 --restrictive--> n1 (conf 0.58, marge 0.91)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-247 silver-B
La jeunesse est en première ligne alors que cet outil est à peine maîtrisé.
- n0 « La jeunesse est en première ligne »
- n1 « alors que cet outil est à peine maîtrisé. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-248 silver-B
De toutes, celle de l’origine épileptique est la moins probable, car bon nombre de ces patients ont fait l’objet d’un enregistrement électroencéphalographique pendant la survenue de leurs symptômes sans qu’aucune activité épileptique n’ait été relevée.
- n0 « De toutes, celle de l’origine épileptique est la moins probable, »
- n1 « car bon nombre de ces patients ont fait l’objet d’un enregistrement électroencéphalographi »
- n2 « sans qu’aucune activité épileptique n’ait été relevée. »
- e0: n0 --restrictive--> n1 (conf 0.42, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.58, marge 0.86)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.50, marge 0.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-254 silver-B
- [3] Parce qu’il a fourni une représentation extrêmement simple et aisément communicable du problème de la crédibilité, l’article de Barro et Gordon [1983] a notamment eu une influence considérable sur l’évolution des doctrines et des institutions monétaires.
- n0 « - [3] »
- n1 « Parce qu’il a fourni une représentation extrêmement simple et aisément communicable du pro »
- e0: n0 --negative--> n1 (conf 0.58, marge 0.61)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-255 silver-B
Le problème est longtemps resté insoluble pour les neuroscientifiques, jusqu’à ce qu’une technique permette d’activer ou de désactiver par un signal lumineux l’activité des neurones de manière ciblée afin d’analyser le fonctionnement cérébral.
- n0 « Le problème est longtemps resté insoluble »
- n1 « pour les neuroscientifiques, jusqu’à ce qu’une technique permette d’activer ou de désactiv »
- e0: n0 --negative--> n1 (conf 0.55, marge 0.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-257 silver-B
Chez cette femme de 72 ans, il aura fallu dix ans pour que ce bruit, pris pour une céphalée, soit enfin identifié pour ce qu’il est.
- n0 « Chez cette femme de 72 ans, il aura fallu dix ans »
- n1 « pour que ce bruit, pris pour une céphalée, soit »
- n2 « enfin identifié pour ce qu’il est. »
- e0: n0 --negative--> n1 (conf 0.54, marge 0.64)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.61, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.57, marge 1.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-258 silver-B
- [12] La demande peut aussi être estimée à travers le « modèle spéculatif » qui consiste à repérer le potentiel de croissance économique d’un territoire pour y acquérir des terrains et y prévoir des banques de compensation.
- n0 « - [12] La demande peut aussi être estimée à travers le « modèle spéculatif » qui consiste  »
- n1 « pour y acquérir des terrains et y prévoir des banques de compensation. »
- e0: n0 --restrictive--> n1 (conf 0.43, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-259 silver-B
Le Kif4 est une protéine associée aux microtubules et spécialisée dans le transport de complexes de protéines et d'ARNms et a été purifiée uniquement en croissance cellulaire.
- n0 « Le Kif4 est une protéine associée aux microtubules et spécialisée dans le transport de com »
- n1 « uniquement en croissance cellulaire. »
- e0: n0 --restrictive--> n1 (conf 0.40, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-262 silver-B
Reste à savoir quand cette mesure sera appliquée.
- n0 « Reste à savoir »
- n1 « quand cette mesure sera appliquée. »
- e0: n0 --causale--> n1 (conf 0.51, marge 0.65)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-263 silver-B
Il a réagi à sa victoire : - Le cofondateur de Place publique considère que la primaire est un « tremplin » et que la gauche sociale-démocrate peut enfin « chasser l’esprit de défaite ».
- n0 « Il a réagi à sa victoire : - Le cofondateur de Place publique considère que la primaire es »
- n1 « enfin « chasser l’esprit de défaite ». »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-264 silver-B
On ne réglera pas le problème des retraites en augmentant les taux de cotisations, alors que le pouvoir d’achat des actifs est déjà à la peine.
- n0 « On ne réglera pas le problème des retraites en augmentant les taux de cotisations, »
- n1 « alors que le pouvoir d’achat des actifs est déjà à la peine. »
- e0: n0 --negative--> n1 (conf 0.53, marge 0.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-265 silver-B
- FOURRÉ, Martine, https://doi.org/10.3917/vst.094.0064 Notes - [1] René Collignon, « L’intérêt de l’œuvre de Bessie Head pour les cliniciens en santé mentale en Afrique », dans Psychopathologie africaine, vol. XXXIII, n° 1, 2005-2006, p. 129-141.
- n0 « - FOURRÉ, Martine, https://doi.org/10.3917/vst.094.0064 Notes - [1] René Collignon, « L’in »
- n1 « pour les cliniciens en santé mentale en Afrique », dans Psychopathologie africaine, vol. »
- n2 « XXXIII, n° 1, 2005-2006, p. 129-141. »
- e0: n0 --restrictive--> n1 (conf 0.36, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.39, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.38, marge 0.66)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v2-266 silver-B
Malgré l’élan sur le photovoltaïque, les investissements en renouvelables restent insuffisants pour tenir les objectifs de transition.
- n0 « Malgré l’élan sur le photovoltaïque, les investissements en renouvelables restent insuffis »
- n1 « pour tenir les objectifs de transition. »
- e0: n0 --restrictive--> n1 (conf 0.43, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
