# Ph5 vague 3 — validation PV1 (pipeline v2, 220 phrases fraîches + hygiène H)

Juger CHAQUE paire : [oui]/[non]+motif/[corriger→X].

## v3-2000 silver-B
Elle porte donc à 4600 le nombre de personnes et entités sanctionnées par les 27 en raison de la guerre en Ukraine.
- n0 « Elle porte »
- n1 « donc à 4600 le nombre de personnes et entités sanctionnées par les 27 »
- n2 « en raison de la guerre en Ukraine. »
- e0: n0 --restrictive--> n1 (conf 0.82, marge 2.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.65, marge 1.44)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.39, marge 0.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2001 silver-B
L’étude suivante concerne les réactions de cycloadditions dipolaires-1,3 en milieux inusuels (sans solvant, activation micro-ondes focalisées et liquides ioniques) faisant intervenir la réactivité d’imidates dérivés d’aminoesters.
- n0 « L’étude suivante concerne les réactions de cycloadditions dipolaires-1,3 en milieux inusue »
- n1 « sans solvant, activation micro-ondes focalisées et liquides ioniques) faisant intervenir l »
- e0: n0 --negative--> n1 (conf 0.42, marge 0.68)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2003 silver-B
Il considère que certains discours réactionnaires se parent de scientificité pour naturaliser des phénomènes en oblitérant leurs dimensions sociales, culturelles, environnementales ou économiques.
- n0 « Il considère que certains discours réactionnaires se parent de scientificité »
- n1 « pour naturaliser des phénomènes en oblitérant leurs dimensions sociales, culturelles, envi »
- e0: n0 --negative--> n1 (conf 0.50, marge 0.90)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2006 silver-B
Il ressort de cette analyse que le succès d’un territoire n’est pas l’apanage de la proximité géographique seule, mais il dépend de son association avec les proximités organisationnelle, sociale, cognitive et institutionnelle.
- n0 « Il ressort de cette analyse que le succès d’un territoire n’est pas l’apanage de la proxim »
- n1 « mais il »
- n2 « dépend de son association avec les proximités organisationnelle, sociale, cognitive et ins »
- e0: n0 --causale--> n1 (conf 0.94, marge 3.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.44, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.96, marge 3.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2007 silver-B
Mission : détecter des maladies rares au plus tôt chez des nourrissons, pour mieux les soigner.
- n0 « Mission : détecter des maladies rares au plus tôt chez des nourrissons, »
- n1 « pour mieux les soigner. »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2011 silver-B
Depuis septembre 2025, le programme national… Cet article est réservé aux abonnés.
- n0 « Depuis septembre 2025, le programme national… »
- n1 « Cet article est »
- n2 « réservé aux abonnés. »
- e0: n0 --causale--> n1 (conf 0.73, marge 1.75)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.53, marge 1.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.82, marge 1.81)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2012 silver-B
Nous proposons un Langage Spécifique au Domaine interne qui utilise Scala pour implanter une famille de DSLs.
- n0 « Nous proposons un Langage Spécifique au Domaine interne qui utilise Scala »
- n1 « pour implanter une famille de DSLs. »
- e0: n0 --causale--> n1 (conf 0.72, marge 1.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2013 silver-B
L'interaction des molécules de plasmide avec la membrane perméabilisée serait donc un processus rapide (de l'ordre d'une centaine de microsecondes).
- n0 « L'interaction des molécules de plasmide avec la membrane perméabilisée serait »
- n1 « donc un processus rapide (de l'ordre d'une centaine de microsecondes). »
- e0: n0 --restrictive--> n1 (conf 0.44, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2015 silver-B
Dès lors, poser la question en ces termes nécessite également de s’arrêter sur ce qui fait la spécificité de l’identité associative.
- n0 « Dès lors, poser la question en ces termes »
- n1 « nécessite également de s’arrêter sur ce qui fait la spécificité de l’identité associative. »
- e0: n0 --restrictive--> n1 (conf 0.68, marge 1.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2017 silver-B
L’économie est, en effet, d’une certaine façon, ce qui fait le sang de nos sociétés, pour prendre une analogie.
- n0 « L’économie est, en effet, d’une certaine façon, ce qui fait le sang de nos sociétés, »
- n1 « pour prendre une analogie. »
- e0: n0 --restrictive--> n1 (conf 0.28, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2018 silver-B
La stratégie nationale pour un numérique inclusif : quelles déclinaisons pour les quartiers politique de la ville ?
- n0 « La stratégie nationale »
- n1 « pour un numérique inclusif : quelles déclinaisons pour les quartiers politique de la ville »
- e0: n0 --restrictive--> n1 (conf 0.58, marge 1.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2023 silver-B
- [19] Le fait d’avoir l’exercice de l’autorité parentale permet aux parents de prendre des décisions relatives à l’administration des biens de l’enfant.
- n0 « - [19] Le fait d’avoir l’exercice de l’autorité parentale »
- n1 « permet aux parents de prendre des décisions relatives à l’administration des biens de l’en »
- e0: n0 --negative--> n1 (conf 0.39, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2024 silver-B
À Oulan-Bator (Mongolie), les représentants américains se sont opposés aux compromis proposés par les Nations unies lors du sommet pour lutter contre l’aridification et la dégradation des terres.
- n0 « À Oulan-Bator (Mongolie), les représentants américains se sont opposés aux compromis propo »
- n1 « pour lutter »
- n2 « contre l’aridification et la dégradation des terres. »
- e0: n0 --negative--> n1 (conf 0.60, marge 1.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.52, marge 0.96)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.87, marge 2.63)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2025 silver-B
À ce jour, aucune de ces théories n’est suffisamment étayée pour être considérée comme l’explication de ce syndrome spectaculaire.
- n0 « À ce jour, aucune de ces théories n’est suffisamment étayée »
- n1 « pour être considérée comme l’explication de ce syndrome spectaculaire. »
- e0: n0 --negative--> n1 (conf 0.48, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2031 silver-B
L’activité physique est indispensable… sauf quand elle devient trop exigeante pour un organisme en pleine croissance.
- n0 « L’activité physique est indispensable… »
- n1 « quand elle devient trop exigeante »
- n2 « pour un organisme en pleine croissance. »
- e0: n0 --negative--> n1 (conf 0.62, marge 1.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.52, marge 0.87)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.35, marge 0.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2033 silver-B
Mon réflexe a été de me dire que si j’allais mourir, il fallait au moins que je téléphone à mon père », se souvient Milena.
- n0 « Mon réflexe a été de me dire que »
- n1 « si j’allais mourir, il fallait au moins que je téléphone à mon père », se souvient Milena. »
- e0: n0 --negative--> n1 (conf 0.46, marge 0.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2035 silver-B
C’est probablement pour cela qu’ils disent que le jeu est comparable, voire supérieur, à la jouissance sexuelle.
- n0 « C’est probablement »
- n1 « pour cela qu’ils disent que le jeu est comparable, voire supérieur, à la jouissance sexuel »
- e0: n0 --restrictive--> n1 (conf 0.41, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2036 silver-B
Depuis 2019, des équipes australiennes, britanniques et américaines tentent d’utiliser le « chant » de coraux sains pour guérir les malades.
- n0 « Depuis 2019, des équipes australiennes, britanniques et américaines tentent d’utiliser le  »
- n1 « pour guérir les malades. »
- e0: n0 --restrictive--> n1 (conf 0.31, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2037 silver-B
« Il faut que l’Europe se protège rapidement de cette offensive, sinon il sera trop tard, prévient Christophe Aufrère, directeur général de la plateforme filière automobile (PFA), qui représente l’industrie française.
- n0 « « Il faut que l’Europe se »
- n1 « protège rapidement de cette offensive, »
- n2 « sinon il sera trop tard, »
- n3 « prévient Christophe Aufrère, directeur général de la plateforme filière automobile (PFA),  »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.46)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.35, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.41, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.39, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.28, marge 0.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2039 silver-B
Dans cette ville où il était arrivé en tête du premier tour de l’élection présidentielle, en 2022, la salle du Zénith était pleine, sans compter la foule amassée dehors pour suivre le meeting sur un écran géant.
- n0 « Dans cette ville où il était arrivé en tête du premier tour de l’élection présidentielle,  »
- n1 « sans compter la foule amassée dehors »
- n2 « pour suivre le meeting sur un écran géant. »
- e0: n0 --restrictive--> n1 (conf 0.47, marge 0.52)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.41, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.41, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2040 silver-B
Il a été maintenu malgré nos alertes », a rappelé Sea Shepherd, pour qui « le gouvernement et les responsables de son transfert doivent maintenant rendre des comptes ».
- n0 « Il a été maintenu »
- n1 « malgré nos alertes », a rappelé Sea Shepherd, »
- n2 « pour qui « le gouvernement et les responsables de son transfert doivent maintenant rendre  »
- e0: n0 --causale--> n1 (conf 0.72, marge 1.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.35, marge 0.14)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.43, marge 0.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2041 silver-B
Il est possible, par exemple, de constater que deux évènements se succèdent sans qu'il y ait un lien direct de cause à effet, ou que d'autres facteurs non considérés interviennent dans la chaîne de causalité.
- n0 « Il est possible, par exemple, de constater que deux évènements se succèdent »
- n1 « sans qu'il y ait un lien direct de »
- n2 « cause à effet, ou que d'autres facteurs non considérés interviennent dans la chaîne de cau »
- e0: n0 --negative--> n1 (conf 0.37, marge 0.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.48, marge 0.85)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.35, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2045 silver-B
La production des déchets et le risque d’accidents sont des motifs d’inquiétudes, pour respectivement 31% et 25% des personnes interrogées.
- n0 « La production des déchets et le risque d’accidents sont des motifs d’inquiétudes, »
- n1 « pour respectivement 31% et 25% des personnes interrogées. »
- e0: n0 --restrictive--> n1 (conf 0.51, marge 0.35)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2052 silver-B
Une alternative… Cet article est réservé aux abonnés.
- n0 « Une alternative… »
- n1 « Cet article est »
- n2 « réservé aux abonnés. »
- e0: n0 --systemique--> n1 (conf 0.65, marge 0.95)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.77, marge 1.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.70, marge 1.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2055 silver-B
Prenons l’exemple d’une application de santé qui collecte des informations sensibles : elle doit obtenir le consentement explicite des utilisateurs avant toute collecte de données.
- n0 « Prenons l’exemple d’une application de santé qui collecte des informations sensibles : ell »
- n1 « avant toute collecte de données. »
- e0: n0 --restrictive--> n1 (conf 0.41, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2059 silver-B
À l’échelle mondiale, les départs pour des vacances ne concernent qu’une fraction de la population.
- n0 « À l’échelle mondiale, les départs »
- n1 « pour des vacances ne concernent qu’une fraction de la population. »
- e0: n0 --negative--> n1 (conf 0.54, marge 0.99)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2060 silver-B
Une description générale est ensuite effectuée sur les micro-ondes avec quelques applications récentes en synthèse et notamment, des applications relatives aux liquides ioniques.
- n0 « Une description générale est »
- n1 « ensuite effectuée sur les micro-ondes avec quelques applications récentes en synthèse et n »
- e0: n0 --negative--> n1 (conf 0.49, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2061 silver-B
DÉCRYPTAGE - La grippe semble arriver de façon précoce aux États-Unis, mais pas encore en France.
- n0 « DÉCRYPTAGE - La grippe semble arriver de façon précoce aux États-Unis, »
- n1 « mais pas encore en France. »
- e0: n0 --causale--> n1 (conf 0.59, marge 0.79)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2064 silver-B
L'adaptation des composants déjà existants dans la communauté robotique a permis notamment d'éviter la tâche souvent ardue de réécrire le code nécessaire pour chacun des composants dans un seul et même environnement de développement.
- n0 « L'adaptation des composants déjà existants dans la communauté robotique a permis notamment »
- n1 « éviter la tâche souvent ardue de réécrire le code »
- n2 « pour chacun des composants dans un seul et même environnement de développement. »
- e0: n0 --negative--> n1 (conf 0.41, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.37, marge 0.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.62, marge 0.96)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2065 silver-B
En France, ces personnes n’ont pas droit à la régularisation du séjour, mais elles peuvent se voir assignées à résidence, et cette assignation peut être accompagnée de manière discrétionnaire d’une autorisation provisoire du séjour.
- n0 « En France, ces personnes n’ont pas droit à la régularisation du séjour, »
- n1 « mais elles peuvent se voir assignées à résidence, et cette assignation peut être accompagn »
- e0: n0 --restrictive--> n1 (conf 0.40, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2066 silver-B
Les enregistrements indiquent qu’à Sainte-Soline les gendarmes stigmatisaient les manifestants, non pour ce qu’ils faisaient, mais pour ce qu’ils étaient, c’est-à-dire des écologistes – des « pue-la-pisse », selon la terminologie en vogue parmi les forces de l’ordre mobilisées.
- n0 « Les enregistrements indiquent qu’à Sainte-Soline les gendarmes stigmatisaient les manifest »
- n1 « pour ce qu’ils faisaient, »
- n2 « mais pour ce qu’ils étaient, c’est-à-dire des écologistes – des « pue-la-pisse », selon la »
- e0: n0 --negative--> n1 (conf 0.55, marge 1.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.52, marge 1.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.82, marge 1.91)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2068 silver-B
RÉCIT - Pour sauver la vie d’un homme de 33 ans menacé par une grave infection en 2023, des médecins de l’Université de Chicago ont dû lui enlever ses poumons en attendant de le transplanter.
- n0 « RÉCIT - »
- n1 « Pour sauver la vie d’un homme de 33 ans menacé par une grave infection en 2023, des médeci »
- e0: n0 --restrictive--> n1 (conf 0.48, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2069 silver-B
Chaque année, c’est la même chose : certains passent l’hiver indemnes, quand d’autres enchaînent les infections respiratoires.
- n0 « Chaque année, c’est la même chose : certains passent l’hiver indemnes, »
- n1 « quand d’autres enchaînent les infections respiratoires. »
- e0: n0 --causale--> n1 (conf 0.59, marge 0.84)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2074 silver-B
Néanmoins, 52 % des personnes interrogées sont favorables à prolonger la durée de vie des réacteurs nucléaires, tandis que 30% ne le pensent pas, dont une petite proportion redoute un risque accru d’accidents.
- n0 « Néanmoins, 52 % des personnes interrogées sont favorables à prolonger la durée de vie des  »
- n1 « tandis que 30% ne le pensent pas, dont une petite proportion redoute un risque accru d’acc »
- e0: n0 --negative--> n1 (conf 0.49, marge 0.86)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2077 silver-B
» Et lorsque leur rhume joue les prolongations, « ils parlent de bronchite alors que c’est une authentique exacerbation de cette BPCO dont ils ignorent l’existence… » Une fois diagnostiquée, la BPCO se traite… Cet article est réservé aux abonnés.
- n0 « » Et »
- n1 « lorsque leur rhume joue les prolongations, « ils parlent de bronchite »
- n2 « alors que c’est une authentique exacerbation de cette BPCO dont ils ignorent l’existence…  »
- n3 « Une fois diagnostiquée, la BPCO se traite… »
- n4 « Cet article est »
- n5 « réservé aux abonnés. »
- e0: n0 --restrictive--> n1 (conf 0.60, marge 0.91)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.49, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.65, marge 0.92)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.45, marge 0.56)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.53, marge 0.93)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --negative--> n4 (conf 0.49, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --causale--> n4 (conf 0.82, marge 1.97)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e7: n3 --causale--> n5 (conf 0.48, marge 0.46)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e8: n4 --causale--> n5 (conf 0.72, marge 1.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2079 silver-B
Ce qui peut, à premier abord, sembler paradoxal n’est pourtant pas dénué de fondement.
- n0 « Ce qui peut, à premier abord, sembler paradoxal n’est »
- n1 « pourtant pas dénué de fondement. »
- e0: n0 --causale--> n1 (conf 0.98, marge 4.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2080 silver-B
Eléments du patrimoine culturel physique dans la commune rurale de Diabo (Province du Gourma/Région du Gulmu), Mémoire de Master, Université Joseph Ki-Zerbo, 184 p. N’dah Didier.
- n0 « Eléments du patrimoine culturel physique dans la commune rurale de Diabo (Province du Gour »
- n1 « N’dah Didier. »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.55)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2081 silver-B
DÉCRYPTAGE - L’accumulation de ce métal lourd entraîne des risques d’ostéoporose, de maladies rénales et possiblement de cancers.
- n0 « DÉCRYPTAGE - L’accumulation de ce métal lourd »
- n1 « entraîne des risques d’ostéoporose, de maladies rénales et possiblement de cancers. »
- e0: n0 --causale--> n1 (conf 0.71, marge 1.48)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2082 silver-B
RÉCIT - Après une longue carrière au CEA, le chercheur français, professeur émérite à l’université de Yale, a rejoint la firme américaine il y a près de quatre ans.
- n0 « RÉCIT - »
- n1 « Après une longue carrière au CEA, le chercheur français, professeur émérite à l’université »
- e0: n0 --systemique--> n1 (conf 0.59, marge 1.14)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2083 silver-B
Bien utilisés, toutefois, les maux peuvent devenir des remèdes.
- n0 « Bien utilisés, »
- n1 « toutefois, les maux peuvent devenir des remèdes. »
- e0: n0 --restrictive--> n1 (conf 0.62, marge 1.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2084 silver-B
Le chiffre B ne peut alors valoir que 3, pour ne pas pouvoir être inversé avec A, ni être redondant.
- n0 « Le chiffre B ne peut alors valoir que 3, »
- n1 « pour ne pas pouvoir être inversé avec A, ni être redondant. »
- e0: n0 --negative--> n1 (conf 0.48, marge 0.92)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2086 silver-B
La quasi-totalité s’est mise en conformité, mais quelques procédures pénales sont en cours.
- n0 « La quasi-totalité s’est mise en conformité, »
- n1 « mais quelques procédures pénales sont en cours. »
- e0: n0 --negative--> n1 (conf 0.42, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2087 silver-B
Au fil du temps, la compensation des îlots de cellules β pour la résistance à l’insuline échoue et entraine par conséquent une baisse progressive de la fonction des cellules β.
- n0 « Au fil du temps, la compensation des îlots de cellules β »
- n1 « pour la résistance à l’insuline échoue et entraine »
- n2 « par conséquent une baisse progressive de la fonction des cellules β. »
- e0: n0 --restrictive--> n1 (conf 0.56, marge 0.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.61, marge 0.68)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.45, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2088 silver-B
Les « insoumis » tiendront en novembre une grande « convention » consacrée à leur programme présidentiel au Bourget (Seine-Saint-Denis), avant un meeting en décembre au Havre, ville d’Edouard Philippe.
- n0 « Les « insoumis » tiendront en novembre une grande « convention » consacrée à leur programm »
- n1 « avant un meeting en décembre au Havre, ville d’Edouard Philippe. »
- e0: n0 --negative--> n1 (conf 0.39, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2089 silver-B
Les présidents de la Colombie (Abelardo de la Espriella), de l’Equateur (Daniel Noboa) et du Guatemala (Bernardo Arevalo) ont exprimé leur solidarité avec le Panama et se sont dits prêts à envoyer de l’aide si nécessaire.
- n0 « Les présidents de la Colombie (Abelardo de la Espriella), de l’Equateur (Daniel Noboa) et  »
- n1 « nécessaire. »
- e0: n0 --negative--> n1 (conf 0.36, marge 0.10)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2094 silver-B
Afin d’établir un diagnostic précis, le médecin doit donc poser des questions permettant d’exclure d’autres maladies, notamment les migraines, l’épilepsie et certaines formes de céphalées.
- n0 « Afin d’établir un diagnostic précis, le médecin doit »
- n1 « donc poser des questions permettant d’exclure d’autres maladies, notamment les migraines,  »
- e0: n0 --restrictive--> n1 (conf 0.48, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2096 silver-B
C’est sa coéquipière américaine, Jessica Meir, pourtant plus expérimentée, qui sera en soutien.
- n0 « C’est sa coéquipière américaine, Jessica Meir, »
- n1 « pourtant plus expérimentée, qui sera en soutien. »
- e0: n0 --causale--> n1 (conf 0.87, marge 2.81)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2097 silver-B
À la descente du train qui l’amène à Londres pour une conférence, il fait face à la tristesse de Gilbert Balavoine, son ancien élève devenu conseiller scientifique à l’ambassade de France.
- n0 « À la descente du train qui l’amène à Londres »
- n1 « pour une conférence, il fait face à la tristesse de Gilbert Balavoine, son ancien élève de »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2098 silver-B
Cette fois, il s’agit de protester contre l’omnibus X, un projet de texte législatif porté par la Commission européenne.
- n0 « Cette fois, il s’agit de protester »
- n1 « contre l’omnibus X, un projet de texte législatif porté par la Commission européenne. »
- e0: n0 --causale--> n1 (conf 0.35, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2100 silver-B
Si elles n’en sont pas guéries, elles témoignent qu’il est possible d’avoir une vie épanouie malgré leurs angoisses pathologiques.
- n0 « Si elles n’en sont pas guéries, elles témoignent qu’il est possible d’avoir une vie épanou »
- n1 « malgré leurs angoisses pathologiques. »
- e0: n0 --negative--> n1 (conf 0.44, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2101 silver-B
« Ils ont voulu prouver qu’ils pouvaient nous massacrer », a-t-il ajouté, avant de parler dans la session de questions-réponses « d’un démantèlement de notre activité ».
- n0 « « Ils ont voulu prouver qu’ils pouvaient nous massacrer », a-t-il ajouté, »
- n1 « avant de parler dans la session de questions-réponses « d’un démantèlement de notre activi »
- e0: n0 --causale--> n1 (conf 0.70, marge 1.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2103 silver-B
Ils aident à identifier les causes et à comprendre comment différents facteurs interagissent pour influencer les résultats en matière de santé.
- n0 « Ils aident à identifier les causes et à comprendre comment différents facteurs interagisse »
- n1 « pour influencer les résultats en matière de santé. »
- e0: n0 --negative--> n1 (conf 0.56, marge 0.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2104 silver-B
En plus de divers outils logiciels nécessaires, une annotation génomique doit être utilisée pour définir les gènes et leurs produits étudiés.
- n0 « En plus de divers outils logiciels nécessaires, une annotation génomique doit être utilisé »
- n1 « pour définir les gènes et leurs produits étudiés. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2108 silver-B
Développée en médecine réparatrice pour masquer une plaie de l’iris, la kératopigmentation est aujourd’hui majoritairement détournée à des fins esthétiques, non sans risques.
- n0 « Développée en médecine réparatrice »
- n1 « pour masquer une plaie de l’iris, la kératopigmentation est aujourd’hui majoritairement dé »
- n2 « sans risques. »
- e0: n0 --systemique--> n1 (conf 0.43, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.41, marge 0.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.48, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2114 silver-B
Les pays ayant su lutter contre les principaux facteurs de risque cardio-vasculaires et de cancer, sont aussi ceux qui ont le mieux résisté à la pandémie.
- n0 « Les pays ayant su lutter »
- n1 « contre les principaux facteurs de risque cardio-vasculaires et de cancer, sont aussi ceux  »
- e0: n0 --causale--> n1 (conf 0.98, marge 4.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2115 silver-B
On se demande alors qu’elles sont les ressources qui influencent directement l’adoption et l’utilisation du e-commerce aussi bien que les facteurs favorisants l’intention d’achat sur internet et le développement de l’économie numérique au Maroc.
- n0 « On se demande alors qu’elles sont les ressources qui influencent directement l’adoption et »
- n1 « bien que les facteurs favorisants l’intention d’achat sur internet et le développement de  »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2116 silver-B
APPEL À TÉMOIGNAGES - Passé 60 ans, certains ne cherchent pas à ralentir, mais à optimiser leur santé.
- n0 « APPEL À TÉMOIGNAGES - Passé 60 ans, certains ne cherchent pas à ralentir, »
- n1 « mais à optimiser leur santé. »
- e0: n0 --causale--> n1 (conf 0.80, marge 1.66)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2118 silver-B
Cette revue a pour but de rappeler certains effets biologiques spécifiques des neutrons et les moyens biologiques de les mesurer.
- n0 « Cette revue a »
- n1 « pour but de rappeler certains effets biologiques spécifiques des neutrons et les moyens bi »
- e0: n0 --restrictive--> n1 (conf 0.66, marge 1.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2120 silver-B
Le président de la République, Emmanuel Macron, a, lui aussi, visité quatre fois la Chine, mais en l’espace de sept ans, entre 2018 et 2025.
- n0 « Le président de la République, Emmanuel Macron, a, lui aussi, visité quatre fois la Chine, »
- n1 « mais en l’espace de sept ans, entre 2018 et 2025. »
- e0: n0 --restrictive--> n1 (conf 0.37, marge 0.06)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2121 silver-B
Les complexes plasmide/membrane ainsi formés ne diffusent pas latéralement dans la membrane.
- n0 « Les complexes plasmide/membrane »
- n1 « ainsi formés ne diffusent pas latéralement dans la membrane. »
- e0: n0 --restrictive--> n1 (conf 0.43, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2124 silver-B
Elle indique d’abord que les recommandations pour enfants adolescents sont globalement cohérentes avec celles des adultes, en adaptant les portions selon l’âge.
- n0 « Elle indique d’abord que les recommandations »
- n1 « pour enfants adolescents sont globalement cohérentes avec celles des adultes, en adaptant  »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2125 silver-B
Une étude publiée vendredi confirme que les femmes suivant un traitement hormonal contre les effets de la ménopause ont plus de risque de développer un cancer du sein.
- n0 « Une étude publiée vendredi confirme que les femmes suivant un traitement hormonal »
- n1 « contre les effets de la ménopause ont plus de risque de développer un cancer du sein. »
- e0: n0 --negative--> n1 (conf 0.52, marge 1.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2126 silver-B
Cette série, d’une ampleur inédite, contribue à faire connaître le syndrome dans la littérature neurologique.
- n0 « Cette série, d’une ampleur inédite, »
- n1 « contribue à faire connaître le syndrome dans la littérature neurologique. »
- e0: n0 --sequentielle--> n1 (conf 0.31, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2128 silver-B
La CGT appelle notamment à cesser le travail dans les raffineries du groupe et dans les stations-service TotalEnergies de la filiale Argedis, au nombre d’environ 180, principalement sur autoroute.
- n0 « La CGT »
- n1 « appelle notamment à cesser le travail dans les raffineries du groupe et dans les stations- »
- e0: n0 --causale--> n1 (conf 0.53, marge 0.97)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2133 silver-B
Ces données nous permettent de proposer un modèle de couplage transcription-traduction dans lequel les deux protéines se lient au promoteur et activent la transcription, puis sont transférées sur l'ARNm naissant où elles jouent alors le rôle d'ITAF (IRES trans-acting factor).
- n0 « Ces données nous permettent de proposer un modèle de couplage transcription-traduction dan »
- n1 « puis sont transférées sur l'ARNm naissant où elles jouent alors le rôle d'ITAF (IRES trans »
- e0: n0 --negative--> n1 (conf 0.48, marge 0.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2137 silver-B
Son amerrissage est prévu jeudi 8 octobre au large de la Californie, avec les Américains Jessica Meir et Jack Hathaway et le Russe Andreï Fediaïev, à bord de Dragon Freedom, sous réserve de la météo.
- n0 « Son amerrissage est prévu jeudi 8 octobre au large de la Californie, avec les Américains J »
- n1 « sous réserve de la météo. »
- e0: n0 --negative--> n1 (conf 0.32, marge 0.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2139 silver-B
À titre de comparaison, une montre à quartz ordinaire se décale d’une demi-seconde par jour… Ces performances sont loin d’être un absolu indépassable.
- n0 « À titre de comparaison, une montre à quartz ordinaire se décale d’une demi-seconde par jou »
- n1 « Ces performances sont loin d’être un absolu indépassable. »
- e0: n0 --negative--> n1 (conf 0.55, marge 1.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2142 silver-B
Ralentissement sur l’éolien terrestre et le solaire, construction des six EPR… Le décret sur la nouvelle feuille de route énergétique de France sera enfin signé en fin de semaine.
- n0 « Ralentissement sur l’éolien terrestre et le solaire, construction des six EPR… »
- n1 « Le décret sur la nouvelle feuille de route énergétique de France sera »
- n2 « enfin signé en fin de semaine. »
- e0: n0 --restrictive--> n1 (conf 0.48, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.41, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.39, marge 0.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2146 silver-B
Quant aux lésions auditives, elles continuent de progresser à cause de niveaux sonores trop élevés.
- n0 « Quant aux lésions auditives, elles continuent de progresser »
- n1 « à cause de niveaux sonores trop élevés. »
- e0: n0 --negative--> n1 (conf 0.32, marge 0.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2158 silver-B
En effet, une étude juridique approfondie permet de s’assurer que chaque aspect de l’application respecte les lois en vigueur.
- n0 « En effet, une étude juridique approfondie »
- n1 « permet de s’assurer que chaque aspect de l’application respecte les lois en vigueur. »
- e0: n0 --negative--> n1 (conf 0.56, marge 0.91)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2160 silver-B
Comme annoncé par la NASA, le splashdown a eu lieu à 17 h 34 (à Paris), après vingt-sept heures d’un voyage retour qui a débuté mercredi à 14 h 05.
- n0 « Comme annoncé par la NASA, le splashdown a eu lieu à 17 h 34 (à Paris), »
- n1 « après vingt-sept heures d’un voyage retour qui a débuté mercredi à 14 h 05. »
- e0: n0 --causale--> n1 (conf 0.69, marge 1.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2161 silver-B
Non retenu, ce projet a néanmoins permis de créer le premier hôpital de jour dédié à la fragilité et d’être reconnu centre collaborateur de l’OMS.
- n0 « Non retenu, ce projet a »
- n1 « néanmoins permis de créer le premier hôpital de jour dédié à la fragilité et d’être reconn »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.47)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2162 silver-B
L’intervention des pompiers a été compliquée car elle n’a pu se faire qu’à pied.
- n0 « L’intervention des pompiers a été compliquée »
- n1 « car elle n’a pu se faire qu’à pied. »
- e0: n0 --causale--> n1 (conf 0.49, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2163 silver-B
L'association des betabloquants et des derives nitres s'est montree plus efficace que les techniques de sclerose endoscopique pour reduire le risque de recidive hemorragique [6].
- n0 « L'association des betabloquants et des derives nitres s'est montree plus efficace que les  »
- n1 « pour reduire le risque de recidive hemorragique [6]. »
- e0: n0 --systemique--> n1 (conf 0.60, marge 0.95)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2164 silver-B
Pour cette méthode de contrôle sous observation partielle des événements, nous avons identifié les conditions d'existence de solution et propose une procédure pour synthétiser un superviseur sous observation partielle.
- n0 « Pour cette méthode de »
- n1 « contrôle sous observation partielle des événements, nous avons identifié les conditions d' »
- e0: n0 --negative--> n1 (conf 0.37, marge 0.54)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2171 silver-B
Bonjour, Pour le moment, nous ne disposons pas de telles informations.
- n0 « Bonjour, »
- n1 « Pour le moment, nous ne disposons pas de telles informations. »
- e0: n0 --restrictive--> n1 (conf 0.81, marge 2.04)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2173 silver-B
FIGAROVOX/TRIBUNE - Alors que tous les voyants sont au rouge, de la situation économique au niveau scolaire des élèves, le haut fonctionnaire estime que la France, dont l’histoire n’a jamais été linéaire, n’a pas dit son dernier mot.
- n0 « FIGAROVOX/TRIBUNE - »
- n1 « Alors que tous les voyants sont au rouge, de la situation économique au niveau scolaire de »
- e0: n0 --restrictive--> n1 (conf 0.56, marge 0.61)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2179 silver-B
Dans le but de caractériser ce modèle murin et d’établir un profil immunitaire, j’ai concentré mes analyses principalement sur les caractéristiques de la SSc telles que la fibrose, l’inflammation, l’hyper-γ-globulinémie polyclonale, la vasculopathie ainsi que de l’expression de cytokines.
- n0 « Dans le but de caractériser ce modèle murin et d’établir un profil immunitaire, j’ai conce »
- n1 « ainsi que de l’expression de cytokines. »
- e0: n0 --negative--> n1 (conf 0.35, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2181 silver-B
De plus, les symptômes cessent le plus souvent spontanément, ce qui est inhabituel pour des troubles convulsifs apparaissant à l’âge adulte.
- n0 « De plus, les symptômes cessent le plus souvent spontanément, ce qui est inhabituel »
- n1 « pour des troubles convulsifs apparaissant à l’âge adulte. »
- e0: n0 --restrictive--> n1 (conf 0.45, marge 0.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2184 silver-B
Quand il fait irruption, ce fléau crée une rupture avec le monde d’avant.
- n0 « Quand il fait irruption, ce fléau crée une rupture avec le monde d’ »
- n1 « avant. »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2185 silver-B
Les jeunes, sauf s'ils sont pistonnés, ne trouvent que rarement le bon stage ou la bonne alternance ...
- n0 « Les jeunes, »
- n1 « sauf s'ils sont pistonnés, ne trouvent que rarement le bon stage ou la bonne alternance .. »
- e0: n0 --restrictive--> n1 (conf 0.73, marge 1.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2187 silver-B
Au cours des dernières années, la communauté des sciences de matériaux a déployé des efforts considérables pour utiliser l'informatique afin d'accélérer le développement et la découverte de nouveaux matériaux.
- n0 « Au cours des dernières années, la communauté des sciences de matériaux a déployé des effor »
- n1 « pour utiliser l'informatique afin d'accélérer le développement et la découverte de nouveau »
- e0: n0 --restrictive--> n1 (conf 0.32, marge 0.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2190 silver-B
Ce qui peut être utile pour la géothermie et la prospection de certains minéraux ou lorsqu’il s’agit, comme ici, de différencier, dans une nappe souterraine, l’eau salée, très conductrice, de l’eau douce, qui l’est moins.
- n0 « Ce qui peut être utile »
- n1 « pour la géothermie et la prospection de certains minéraux ou »
- n2 « lorsqu’il s’agit, comme ici, de différencier, dans une nappe souterraine, l’eau salée, trè »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.38, marge 0.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.28, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2191 silver-B
Mais ces hommes-là, avec leur gilet fluo et leur écusson à gueule de loup, en ont le droit, néanmoins : il s’agit de lieutenants de louveterie, des auxiliaires de l’Etat bénévoles, dont l’origine remonte à Charlemagne.
- n0 « Mais ces hommes-là, avec leur gilet fluo et leur écusson à gueule de loup, en ont le droit »
- n1 « néanmoins : il s’agit de lieutenants de louveterie, des auxiliaires de l’Etat bénévoles, d »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2193 silver-B
La gestion des données doit être non seulement conforme aux lois en vigueur, comme le RGPD, mais aussi sécurisée afin de protéger la vie privée des utilisateurs.
- n0 « La gestion des données doit être non seulement conforme aux lois en vigueur, comme le RGPD »
- n1 « mais aussi sécurisée »
- n2 « afin de »
- n3 « protéger la vie privée des utilisateurs. »
- e0: n0 --restrictive--> n1 (conf 0.38, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.56, marge 1.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.94, marge 3.33)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.38, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --restrictive--> n3 (conf 0.78, marge 1.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2195 silver-B
Les habitants sont déjà prêts à retourner au combat pour faire échouer un projet déjà abandonné en 2022.
- n0 « Les habitants sont déjà prêts à retourner au combat »
- n1 « pour faire échouer un projet déjà abandonné en 2022. »
- e0: n0 --negative--> n1 (conf 0.55, marge 0.79)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2196 silver-B
Comment s'intéresser à la souffrance de l'autre quand on est menacé, humilié, empêché de vivre normalement ?
- n0 « Comment s'intéresser à la souffrance de l'autre »
- n1 « quand on est menacé, humilié, empêché de vivre normalement ? »
- e0: n0 --negative--> n1 (conf 0.39, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2199 silver-B
Le texte présente d'abord le contexte actuel des dispositifs de formation FLE en France et les difficultés d’écriture rencontrées par les apprenants migrants.
- n0 « Le texte présente »
- n1 « d'abord le contexte actuel des dispositifs de formation FLE en France et les difficultés d »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.99)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2201 silver-B
Isolement, anxiété, burn out… Le Dr. Romain Rey livre des clés pour se préserver.
- n0 « Isolement, anxiété, burn out… Le Dr. Romain Rey livre des clés »
- n1 « pour se préserver. »
- e0: n0 --systemique--> n1 (conf 0.37, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2203 silver-B
En modifiant la circulation atmosphérique à grande échelle, ce phénomène climatique tend à freiner les cyclones dans l’Atlantique mais à les renforcer dans le Pacifique, où trois ouragans de catégorie maximale ont été enregistrés au cours de cette année.
- n0 « En modifiant la circulation atmosphérique à grande échelle, ce phénomène climatique tend à »
- n1 « freiner les cyclones dans l’Atlantique »
- n2 « mais à les renforcer dans le Pacifique, où trois ouragans de catégorie maximale ont été en »
- e0: n0 --causale--> n1 (conf 0.90, marge 2.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.51, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.96, marge 3.48)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2205 silver-B
Mais revenir aux codes qui ont fait le succès de la « Deuche » après la seconde guerre mondiale s’est rapidement imposé.
- n0 « Mais revenir aux codes qui ont fait le succès de la « Deuche » »
- n1 « après la seconde guerre mondiale s’est rapidement imposé. »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2207 silver-B
Faire beaucoup de sport permet-il de compenser une mauvaise alimentation ?
- n0 « Faire beaucoup de sport »
- n1 « permet-il de compenser une mauvaise alimentation ? »
- e0: n0 --causale--> n1 (conf 0.84, marge 2.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2210 silver-B
Notez également que la montre-bracelet d’Aliénor affiche l’heure sur quatre chiffres et que, si vous lui proposez 7 h 30, vous aurez toutes les chances de ne pas la voir puisqu’il y a deux 0 dans 07 h 30.
- n0 « Notez également que la montre-bracelet d’Aliénor affiche l’heure sur quatre chiffres et qu »
- n1 « si vous lui proposez 7 h 30, vous aurez toutes les chances de ne pas la voir puisqu’il y a »
- e0: n0 --negative--> n1 (conf 0.46, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2217 silver-B
La Haute autorité de Santé estime que la vaccination annuelle des personnels en contact avec des patients fragiles est le seul moyen de protéger ces derniers.
- n0 « La Haute autorité de Santé estime que la vaccination annuelle des personnels en contact av »
- n1 « protéger ces derniers. »
- e0: n0 --negative--> n1 (conf 0.33, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v3-2219 silver-B
Alors que près de deux tiers des collégiens déclarent aller… Cet article est réservé aux abonnés.
- n0 « Alors que près de deux tiers des collégiens déclarent aller… »
- n1 « Cet article est »
- n2 « réservé aux abonnés. »
- e0: n0 --causale--> n1 (conf 0.61, marge 0.84)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.44, marge 0.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.82, marge 1.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
