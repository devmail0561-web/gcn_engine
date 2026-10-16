# Ph5 production — validation PV1 manuelle (69 phrases prédicatives FR, TOUTES les OK-avec-paires)

Hygiène H aveugle-modèle (49 exclues). Juger CHAQUE paire : [oui]/[non]+motif/[corriger→X].

## ph5-002 silver-B
Le prix Nobel d’économie 2026 a été décerné, lundi 12 octobre, aux Américains Michael Woodford et Robert E.
- n0 « Le prix Nobel d’économie 2026 a été décerné, lundi 12 octobre »
- n1 « , aux Américains Michael Woodford et Robert E. »
- e0: n0 --restrictive--> n1 (conf 0.42, marge 0.33)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-006 silver-B
- [6] Il s’agit du gaz présent en solution dans le pétrole.
- n0 « - [6] »
- n1 « Il s’agit du gaz présent en solution dans le pétrole. »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-007 silver-B
DÉCRYPTAGE - La plupart des recherches en génétique des addictions se sont jusqu’ici concentrées sur les facteurs de risque.
- n0 « DÉCRYPTAGE »
- n1 « - La plupart des recherches en génétique des addictions se sont jusqu’ici concentrées sur  »
- e0: n0 --restrictive--> n1 (conf 0.58, marge 1.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-012 silver-B
À lire aussi Un «Poker» à Saint-Dizier : la dissuasion nucléaire s’adapte au nouveau contexte stratégique Il est surprenant de constater que la perception de l’exposition à la radioactivité des Français est totalement erronée.
- n0 « À lire aussi Un «Poker» à Saint-Dizier : la dissuasion nucléaire s’adapte au nouveau conte »
- n1 « Il est surprenant de constater que la perception de l’exposition à la radioactivité des Fr »
- e0: n0 --causale--> n1 (conf 0.56, marge 0.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-016 silver-B
DÉCRYPTAGE - TotalEnergies et Engie sont en pointe pour fournir aux acteurs des données et de l’intelligence artificielle l’électricité verte dont ils ont besoin.
- n0 « DÉCRYPTAGE »
- n1 « - TotalEnergies et Engie sont en pointe pour fournir aux acteurs des données et de l’intel »
- e0: n0 --restrictive--> n1 (conf 0.29, marge 0.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-018 silver-B
DÉCRYPTAGE - Des scientifiques australiens montrent que l’allaitement maternel stimule les défenses immunitaires contre le cancer du sein, expliquant un phénomène connu, mais encore mal compris.
- n0 « DÉCRYPTAGE »
- n1 « - Des scientifiques australiens montrent que l’allaitement maternel stimule les défenses i »
- e0: n0 --causale--> n1 (conf 0.81, marge 2.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-020 silver-B
Pour l’association Tilikum Spirit, Rocky « est mort d’un transfert décidé pour l’industrie de la captivité ».
- n0 « Pour l’association Tilikum Spirit »
- n1 « , Rocky « est mort d’un transfert décidé pour l’industrie de la captivité ». »
- e0: n0 --negative--> n1 (conf 0.41, marge 0.31)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-023 silver-B
Pour prolonger la ligne 1 du métro parisien, Île-de-France Mobilités et les élus locaux envisagent trois scénarios.
- n0 « Pour prolonger la ligne 1 du métro parisien »
- n1 « , Île-de-France Mobilités et les élus locaux »
- n2 « envisagent trois scénarios. »
- e0: n0 --restrictive--> n1 (conf 0.57, marge 0.51)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.59, marge 0.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.47, marge 0.89)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-024 silver-B
Ecole Supérieure des Sciences Economiques et de Management - 3.
- n0 « Ecole Supérieure des Sciences Economiques et de Management »
- n1 « - 3. »
- e0: n0 --restrictive--> n1 (conf 0.63, marge 0.74)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-029 silver-B
Faculté des Sciences Juridiques Economique et Sociale d'Ait Melloul Université Ibno Zohr Description Cet article explore les différents facteurs qui influencent la survie des petites et moyennes entreprises (PME) en adoptant trois approches distinctes : émotive, volontariste et déterministe.
- n0 « Faculté des Sciences Juridiques Economique et Sociale d'Ait Melloul »
- n1 « Université Ibno »
- n2 « Zohr »
- n3 « Description »
- n4 « Cet article explore les différents facteurs qui influencent la survie des petites et moyen »
- e0: n0 --restrictive--> n1 (conf 0.51, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.54, marge 0.72)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.71, marge 1.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.57, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --systemique--> n3 (conf 0.82, marge 1.81)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --systemique--> n4 (conf 0.90, marge 2.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --systemique--> n4 (conf 0.74, marge 1.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-033 silver-B
« La couverture vaccinale a progressé l’hiver dernier, mais nous sommes encore trop loin des niveaux de protection nécessaires », a insisté jeudi devant la presse Grégory Emery, directeur général de la santé.
- n0 « « »
- n1 « La couverture vaccinale a progressé l’hiver dernier, mais nous sommes encore trop loin des »
- e0: n0 --restrictive--> n1 (conf 0.77, marge 1.99)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-037 silver-B
Suivant leurs conseils, j’ai donc décidé de subir une opération pour le faire retirer.
- n0 « Suivant leurs conseils »
- n1 « , »
- n2 « j’ai donc décidé de subir une opération pour le faire retirer. »
- e0: n0 --systemique--> n1 (conf 0.39, marge 0.03)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.54, marge 0.97)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.57, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-043 silver-B
Avec l’intérêt porté aux troubles des apprentissages ces dernières années, la meilleure connaissance qui en découle et la création de réseaux, nous pouvons espérer l’amélioration du parcours de soins de ces enfants dans les années à venir.
- n0 « Avec l’intérêt porté aux troubles des apprentissages ces dernières années »
- n1 « , la meilleure connaissance qui en découle et la création de réseaux, nous pouvons espérer »
- e0: n0 --restrictive--> n1 (conf 0.38, marge 0.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-044 silver-B
Caravanes renversées, arbres arrachés, station-service endommagée, débris : le passage de la tempête a provoqué des dégâts dans le nord-ouest de la Floride, ont constaté des journalistes de l’Agence France-Presse (AFP).
- n0 « Caravanes renversées, arbres arrachés, station-service endommagée, débris : le passage de  »
- n1 « , ont constaté des journalistes de l’Agence France-Presse (AFP). »
- e0: n0 --causale--> n1 (conf 0.32, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-054 silver-B
4Pour ce qui est « des politiques économiques », nous considérerons qu’il s’agit de « l’ensemble des interventions des collectivités territoriales, notamment des états, dans la vie économique ».
- n0 « 4Pour ce qui est « des politiques économiques » »
- n1 « , nous considérerons qu’il s’agit de « l’ensemble des interventions des collectivités terr »
- e0: n0 --causale--> n1 (conf 0.50, marge 0.49)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-055 silver-B
« Un coup de tonnerre dans un ciel serein »: beaucoup décrivent ainsi l’annonce d’un cancer.
- n0 « « »
- n1 « Un coup de tonnerre dans un ciel serein »: beaucoup décrivent ainsi l’annonce »
- n2 « d’ »
- n3 « un cancer »
- n4 « . »
- e0: n0 --causale--> n1 (conf 0.92, marge 3.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.97, marge 3.99)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.99, marge 4.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --causale--> n3 (conf 0.40, marge 0.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --restrictive--> n3 (conf 0.46, marge 0.85)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --systemique--> n4 (conf 0.70, marge 1.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --causale--> n4 (conf 0.49, marge 0.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-057 silver-B
«Jamais on n’aurait pu faire ça tout seuls!» Chacun à son rythme.
- n0 « «Jamais on n’aurait pu faire ça tout seuls! »
- n1 « » »
- n2 « Chacun à son rythme. »
- e0: n0 --causale--> n1 (conf 0.56, marge 0.80)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.43, marge 0.14)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.89, marge 2.89)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-060 silver-B
PSYCHOLOGIE - Selon notre niveau d’activités, la perception du temps change.
- n0 « PSYCHOLOGIE »
- n1 « - Selon notre niveau d’activités, la perception du temps change. »
- e0: n0 --restrictive--> n1 (conf 0.41, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-061 silver-B
Conclusion: These adverse effects can have significant consequences on patients and their anticancer treatment.
- n0 « Conclusion: »
- n1 « These adverse effects can have significant consequences on patients and their anticancer t »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-068 silver-B
Le problème avec Philémon, c’est qu’il mélange toujours tout !
- n0 « Le problème avec Philémon, c’est »
- n1 « qu’il mélange toujours tout ! »
- e0: n0 --causale--> n1 (conf 0.41, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-070 silver-B
L’administrateur de la NASA, Jared Isaacman, a aussi salué, sur X, une « pionnière dont le travail a contribué à rendre possibles le programme Apollo et les premiers voyages de l’homme vers la Lune ».
- n0 « L’administrateur de la NASA »
- n1 « , Jared Isaacman, a aussi salué, sur X, une « pionnière dont le travail a contribué à rend »
- e0: n0 --restrictive--> n1 (conf 0.55, marge 1.00)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-072 silver-B
"La législation est un guide, pas une contrainte." 📜 Protection des Données Personnelles dans les Applications Mobiles La protection des données personnelles est au cœur des préoccupations des utilisateurs d’applications.
- n0 « "La législation est un guide, pas une contrainte. »
- n1 « " 📜 »
- n2 « Protection »
- n3 « des »
- n4 « Données »
- n5 « Personnelles »
- n6 « dans les Applications Mobiles »
- n7 « La protection des données personnelles est au cœur des préoccupations des utilisateurs d’a »
- e0: n0 --causale--> n1 (conf 0.53, marge 0.72)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.79, marge 2.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.68, marge 1.14)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --restrictive--> n3 (conf 0.88, marge 2.66)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --restrictive--> n3 (conf 0.51, marge 0.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --causale--> n4 (conf 0.45, marge 0.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --negative--> n4 (conf 0.31, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e7: n3 --restrictive--> n5 (conf 0.33, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e8: n4 --causale--> n5 (conf 0.96, marge 3.89)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e9: n4 --causale--> n6 (conf 0.99, marge 4.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e10: n5 --causale--> n6 (conf 0.79, marge 2.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e11: n5 --negative--> n7 (conf 0.36, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e12: n6 --restrictive--> n7 (conf 0.46, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-073 silver-B
DÉCRYPTAGE - Ce phénomène est d’autant plus étonnant que la fonte des glaces polaires tend au contraire à ralentir sa vitesse de rotation, selon une étude publiée dans la revue Nature.
- n0 « DÉCRYPTAGE »
- n1 « - Ce phénomène est d’autant plus étonnant que la fonte des glaces polaires tend au contrai »
- e0: n0 --causale--> n1 (conf 0.80, marge 2.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-077 silver-B
Une des marques fétiches des chefs d’entreprise et des professions libérales tricolores, dès lors qu’ils ont la fibre automobile, prend un tournant spectaculaire.
- n0 « Une des marques fétiches des chefs d’ »
- n1 « entreprise »
- n2 « et des professions libérales tricolores, »
- n3 « dès lors qu’ils ont la fibre automobile, prend un tournant spectaculaire. »
- e0: n0 --restrictive--> n1 (conf 0.51, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.41, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.36, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.29, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.30, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-080 silver-B
- [2] Voir notamment la création du groupe de travail « Régulations, marchés, capitalisme » au sein de l’Association française de science politique.
- n0 « - [2] Voir notamment la création du groupe de travail « »
- n1 « Régulations, marchés, capitalisme » »
- n2 « au sein de l’Association française de science politique »
- n3 « . »
- e0: n0 --restrictive--> n1 (conf 0.34, marge 0.10)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.44, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.82, marge 2.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.29, marge 0.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --systemique--> n3 (conf 0.34, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-081 silver-B
Dans les laboratoires, les horloges dites « optiques » sont déjà capables d’une précision relative de 10−18, soit 100 fois mieux.
- n0 « Dans les laboratoires, les horloges dites »
- n1 « « optiques » sont déjà capables d’une précision relative de 10−18, soit 100 fois mieux. »
- e0: n0 --negative--> n1 (conf 0.44, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-083 silver-B
« Et enfin, on chasse l’esprit de défaite de nos âmes et de nos cœurs, et c’est ça qui est le principal.
- n0 « « »
- n1 « Et enfin, on chasse l’esprit de défaite de nos âmes et de nos cœurs, et c’est ça qui est l »
- e0: n0 --restrictive--> n1 (conf 0.80, marge 2.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-084 silver-B
Michael Dell, fondateur et patron de l’entreprise informatique du même nom, et Satya Nadella, PDG de Microsoft, ont, eux, été décorés de la médaille nationale de la technologie et de l’innovation.
- n0 « Michael Dell, fondateur et patron de l’ »
- n1 « entreprise informatique du même nom, et Satya Nadella, PDG de Microsoft »
- n2 « , ont, eux, été décorés de la médaille nationale de la technologie et de l’innovation. »
- e0: n0 --negative--> n1 (conf 0.37, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.45, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.35, marge 0.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-086 silver-B
Cette chaleur emmagasinée est ensuite redistribuée dans l’océan et l’atmosphère, modifiant… Cet article est réservé aux abonnés.
- n0 « Cette chaleur emmagasinée est ensuite redistribuée dans l’océan et l’atmosphère, modifiant »
- n1 « Cet article est réservé aux abonnés. »
- e0: n0 --causale--> n1 (conf 0.55, marge 0.64)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-087 silver-B
Un syndrome largement méconnu des médecins Le syndrome de la tête qui explose (EHS) est une parasomnie, terme désignant un trouble du comportement survenant au moment de l’endormissement ou du réveil.
- n0 « Un syndrome largement méconnu des médecins »
- n1 « Le syndrome de la tête qui explose (EHS) est une parasomnie, terme désignant un trouble du »
- e0: n0 --negative--> n1 (conf 0.46, marge 0.64)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-091 silver-B
« Ce qui a brûlé n’a pas de valeur patrimoniale en tant que tel, il s’agissait surtout d’étals en structure légère », a toutefois précisé M. Weil.
- n0 « « »
- n1 « Ce qui a brûlé n’a pas de valeur patrimoniale en tant que tel, il s’agissait surtout d’éta »
- e0: n0 --negative--> n1 (conf 0.31, marge 0.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-095 silver-B
C’est scandaleux, vite une action des écologistes et de Macron, Peut être qu’un nouvel impôt anti basculement serait efficace PHILIPPE NIETO 1 Ne serait-ce pas un des facteurs du "réchauffement climatique" dont on nous rebat les oreilles...
- n0 « C’ »
- n1 « est scandaleux, vite une action des écologistes et de Macron »
- n2 « , Peut être qu’un nouvel impôt anti basculement serait efficace PHILIPPE NIETO 1 »
- n3 « Ne serait-ce pas un des facteurs du "réchauffement climatique" dont on nous rebat les orei »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.47, marge 0.23)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.43, marge 0.63)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.36, marge 0.55)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.33, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-101 silver-B
DÉCRYPTAGE - Depuis plusieurs mois, l’exposition à ce métal via l’alimentation est accusée d’augmenter le risque de tumeur, notamment du pancréas.
- n0 « DÉCRYPTAGE »
- n1 « - Depuis plusieurs mois, l’exposition à ce métal via l’alimentation est accusée d’augmente »
- e0: n0 --systemique--> n1 (conf 0.47, marge 0.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-105 silver-B
« Une mesure de dissolution est envisagée et, à cette fin, un contradictoire est en cours, à l’issue duquel la dissolution pourra intervenir », a fait savoir le ministère à l’Agence France-Presse.
- n0 « « »
- n1 « Une mesure de dissolution est envisagée et, à cette fin, un contradictoire est en cours, à »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-108 silver-B
Résumé Objectifs : Présenter le jeu pathologique (JP), ef­fet indésirable peu connu des agonistes dopaminer­giques, au travers de quatre notifications rapportées au Centre régional de pharmacovigilance d’Amiens (France).
- n0 « Résumé Objectifs : »
- n1 « Présenter le jeu pathologique (JP), ef­fet indésirable peu connu des agonistes dopaminer­g »
- e0: n0 --negative--> n1 (conf 0.31, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-113 silver-B
Enfin l’imago, le stade adulte, avec ses ailes proportionnellement immenses, explosion de couleurs et de diversité, qui peut durer quelques jours ou quelques mois.
- n0 « Enfin l’imago »
- n1 « , le stade adulte, avec ses ailes proportionnellement immenses, explosion de couleurs et d »
- e0: n0 --restrictive--> n1 (conf 0.79, marge 2.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-114 silver-B
Adobe Stock RÉCIT - Le 13 octobre a lieu la Journée mondiale de lutte contre le cancer du sein métastatique.
- n0 « Adobe Stock »
- n1 « RÉCIT »
- n2 « - Le 13 octobre a lieu la Journée mondiale de lutte contre le cancer du sein métastatique. »
- e0: n0 --restrictive--> n1 (conf 0.48, marge 0.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.45, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.46, marge 0.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-125 silver-B
Samedi, les autorités ont commencé à rassembler l’aide qui sera envoyée au cours des prochaines heures vers les zones affectées.
- n0 « Samedi »
- n1 « , les autorités ont commencé à rassembler l’aide qui sera envoyée au cours des prochaines  »
- e0: n0 --restrictive--> n1 (conf 0.37, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-130 silver-B
En 1968, Per Asbrink, gouverneur de la Banque de Suède, bataille en vain pour arracher au gouvernement social-démocrate de Tage Erlander (1901-1985) l’indépendance de son institution.
- n0 « En 1968, Per Asbrink, gouverneur de la Banque de Suède, bataille en vain pour arracher au  »
- n1 « l’indépendance de son institution. »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.22)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-136 silver-B
Sophie Adenot et ses camarades seront ensuite transportés sur la terre ferme en hélicoptère, puis ils prendront un avion pour Houston au Texas, où se trouve le centre de la NASA chargé des astronautes.
- n0 « Sophie Adenot et ses camarades seront ensuite transportés sur la terre ferme en hélicoptèr »
- n1 « puis ils prendront un avion pour Houston au Texas, où se trouve le centre de la NASA charg »
- e0: n0 --negative--> n1 (conf 0.39, marge 0.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-144 silver-B
(CE) n° 1896/2006 du Parlement européen et du Conseil du 12 déc.
- n0 « ( »
- n1 « CE) »
- n2 « n° 1896/2006 du Parlement européen et du Conseil du 12 déc. »
- e0: n0 --causale--> n1 (conf 0.92, marge 2.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.60, marge 0.74)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.42, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-145 silver-B
Passer la publicité Passer la publicité Le premier baromètre sur la perception des risques chez les Français, commandé par la nouvelle Autorité de sûreté nucléaire et de radioprotection (ASNR) ne classe toujours pas le nucléaire en tête des risques.
- n0 « Passer la publicité Passer la publicité »
- n1 « Le premier baromètre sur la perception des risques chez les Français, commandé par la nouv »
- e0: n0 --restrictive--> n1 (conf 0.74, marge 1.55)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-148 silver-B
Impact des polluants atmosphériques sur la qualité de l’air La pollution atmosphérique est un enjeu majeur de santé publique et environnementale.
- n0 « Impact des polluants atmosphériques sur la qualité de l’air »
- n1 « La pollution atmosphérique est un enjeu majeur de santé publique et environnementale. »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.22)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-158 silver-B
Des chercheurs de Singapour ont même annoncé fin septembre avoir atteint le seuil de 10−19, soit une seconde… Cet article est réservé aux abonnés.
- n0 « Des chercheurs de Singapour ont même annoncé fin septembre avoir atteint le seuil de 10−19 »
- n1 « Cet article est réservé aux abonnés. »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-171 silver-B
Qu’il s’agisse des secours en montagne ou en mer, le cancer d’un fumeur, la cirrhose d’un buveur, il me semble nécessaire de responsabiliser les gens en leur demandant une participation financière significative.
- n0 « Qu’il s’agisse des secours en montagne ou en mer, le cancer d’un fumeur, la cirrhose d’un  »
- n1 « , il me semble nécessaire de responsabiliser les gens en leur demandant une participation  »
- e0: n0 --causale--> n1 (conf 0.60, marge 0.96)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-172 silver-B
« Il va falloir remettre les travailleurs au cœur du contrat social, parler des salaires, (…) parler d’industrie.
- n0 « « »
- n1 « Il va falloir remettre les travailleurs au cœur du contrat social, parler des salaires, (… »
- e0: n0 --restrictive--> n1 (conf 0.59, marge 0.98)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-173 silver-B
Libérus C'est surtout le dépistage pré-natal qu'il faudrait développer.
- n0 « Libérus »
- n1 « C'est surtout le dépistage pré-natal qu'il faudrait développer. »
- e0: n0 --restrictive--> n1 (conf 0.42, marge 0.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-178 silver-B
« Grâce à ces modèles, des mesures budgétaires et monétaires prises au bon moment peuvent atténuer les hauts et les bas de l’activité économique », a expliqué le jury dans un communiqué.
- n0 « « »
- n1 « Grâce à ces modèles, des mesures budgétaires et monétaires prises au bon moment peuvent at »
- e0: n0 --causale--> n1 (conf 0.78, marge 2.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-180 silver-B
La fondation est dirigée par Cheikha Mozah bint Nasser, la femme du Cheikh Hamad.
- n0 « La fondation est dirigée par Cheikha Mozah bint »
- n1 « Nasser, la femme du Cheikh Hamad. »
- e0: n0 --systemique--> n1 (conf 0.47, marge 0.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-187 silver-B
- [7] Des exemples de cette « course aux sièges » et des situations de concurrence qui en résultent sont décrits dans : Gisela Naegle, Stadt, Recht und
- n0 « - [7] »
- n1 « Des exemples de cette « course aux sièges » et des situations de concurrence qui en résult »
- e0: n0 --negative--> n1 (conf 0.54, marge 0.49)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-188 silver-B
- [8] Il existe aussi la valeur ‘-’ qui indique soit que la valeur est inconnue, soit que l’on décide de ne pas trancher entre une valeur ‘1’ ou ‘0’.
- n0 « - [8] »
- n1 « Il existe aussi la valeur ‘-’ qui indique soit que la valeur est inconnue, soit que l’on d »
- e0: n0 --restrictive--> n1 (conf 0.77, marge 1.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-192 silver-B
Le célèbre programme Apollo avait permis aux Américains de battre l’Union soviétique durant la première course à la Lune, dans les années 1960 et 1970.
- n0 « Le célèbre programme Apollo avait permis aux Américains de battre »
- n1 « l’Union soviétique durant la première course à la Lune, dans les années 1960 et 1970. »
- e0: n0 --causale--> n1 (conf 0.72, marge 1.70)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-196 silver-B
« Les vibrations et le bruit m’empêchent de dormir la nuit », raconte Sarah (prénom d’emprunt), une résidente qui vit à proximité.
- n0 « « »
- n1 « Les vibrations et le bruit m’empêchent de dormir la nuit », raconte Sarah (prénom d’emprun »
- e0: n0 --negative--> n1 (conf 0.31, marge 0.06)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-205 silver-B
Face à l’envolée des prix de l’essence et du fioul, l’argument de l’indépendance énergétique est le principal motif de soutien (pour 38% des Français selon le baromètre).
- n0 « Face à l’envolée des prix de l’essence et du fioul, l’argument de l’indépendance énergétiq »
- n1 « . »
- e0: n0 --restrictive--> n1 (conf 0.32, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-208 silver-B
Portée par une association, une anomalie à ce niveau, la course réunionnaise d’ultra-trail, qui s’élance jeudi 15 octobre, défend son indépendance malgré une gouvernance discutée et des propositions de rachat.
- n0 « Portée par une association »
- n1 « , une anomalie à ce niveau »
- n2 « , la course réunionnaise d’ultra-trail, qui s’élance jeudi 15 octobre, défend son indépend »
- e0: n0 --systemique--> n1 (conf 0.78, marge 2.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.64, marge 1.34)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.74, marge 1.71)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-213 silver-B
« En guise d’apéritif », commente Yanaël Delpech, muséologue, des amazonites, quartz, gogottes, améthystes ou autres fluorites ouvrent la visite.
- n0 « « »
- n1 « En guise d’apéritif », commente Yanaël Delpech, muséologue, des amazonites, quartz, gogott »
- e0: n0 --restrictive--> n1 (conf 0.44, marge 0.51)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-216 silver-B
ENTRETIEN - Le gouvernement vient d’annoncer un quatrième plan sur les maladies rares, porteur de progrès majeurs si des moyens financiers permettent de les réaliser.
- n0 « ENTRETIEN »
- n1 « - Le gouvernement vient d’annoncer un quatrième plan sur les maladies rares, porteur de pr »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-219 silver-B
\nL'électroperméabilisation et l'interaction plasmide/membrane perturbent la mobilité latérale des protéines membranaire du feuillet externe de la membrane plasmique.
- n0 « \nL'électroperméabilisation et l'interaction plasmide/membrane »
- n1 « perturbent la mobilité latérale des protéines membranaire du feuillet externe de la membra »
- e0: n0 --restrictive--> n1 (conf 0.50, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-224 silver-B
Panthat Sitdatvirade - stock.adobe.com Les efforts de protection montrent que l’érosion du vivant n’est pas une fatalité, explique le dernier grand rapport global du WWF, mais la fenêtre pour agir se rétrécit.
- n0 « Panthat Sitdatvirade »
- n1 « - stock.adobe.com Les efforts de protection montrent que l’érosion du vivant n’est pas une »
- e0: n0 --restrictive--> n1 (conf 0.38, marge 0.06)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-227 silver-B
« A mesure que la fin d’une mission approche, on se remet à penser à des odeurs, des goûts et des sensations familières, jusqu’à presque les ressentir.
- n0 « « A mesure »
- n1 « que la fin d’une mission approche, on se remet à penser à des odeurs, des goûts et des sen »
- e0: n0 --causale--> n1 (conf 0.33, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-231 silver-B
Les recherches menees a la grotte aux Points entre 2011 et 2015 dans le cadre du projet « Datation Grottes Ornees » (DGO) ont permis de recueillir de multiples observations d’ordre chronologique.
- n0 « Les recherches menees a la grotte aux Points entre 2011 et 2015 dans le cadre du projet « »
- n1 « Datation Grottes »
- n2 « Ornees » (DGO) »
- n3 « ont permis de recueillir de multiples observations d’ordre chronologique. »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.43, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.56, marge 0.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --causale--> n3 (conf 0.54, marge 0.89)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.41, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-236 silver-B
DÉCRYPTAGE - Une vaste étude française confirme que l’incidence du cancer du sein avant 50 ans a bel et bien augmenté de façon continue depuis trente ans.
- n0 « DÉCRYPTAGE »
- n1 « - Une vaste étude française confirme que l’incidence du cancer du sein avant 50 ans a bel  »
- e0: n0 --systemique--> n1 (conf 0.57, marge 1.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-237 silver-B
« Nous avions demandé, depuis le début, le maintien de tous les animaux à Marineland » et « avions prévenu de ce qui pouvait arriver », a-t-elle ajouté.
- n0 « « »
- n1 « Nous avions demandé, depuis le début, le maintien de tous les animaux à Marineland » et «  »
- e0: n0 --restrictive--> n1 (conf 0.45, marge 0.65)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-241 silver-B
« Ils ont annihilé mon domaine de recherche », a-t-il calmement précisé.
- n0 « « »
- n1 « Ils ont annihilé mon domaine de recherche », a-t-il calmement précisé. »
- e0: n0 --restrictive--> n1 (conf 0.40, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-252 silver-B
Les PME : levier de croissance dans une économie en développement Authors/Creators - 1.
- n0 « Les PME : levier de croissance dans une économie en développement Authors/Creators »
- n1 « - 1. »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-253 silver-B
Il a salué une participation que « personne n’attendait » et s’est moqué des « Schtroumpfs grognons » qui pensaient que la primaire à gauche ne susciterait pas d’entrain.
- n0 « Il a salué une participation que « personne n’attendait » et s’est moqué des « »
- n1 « Schtroumpfs grognons » qui pensaient que la primaire à gauche ne susciterait pas d’entrain »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-263 silver-B
Il a réagi à sa victoire : - Le cofondateur de Place publique considère que la primaire est un « tremplin » et que la gauche sociale-démocrate peut enfin « chasser l’esprit de défaite ».
- n0 « Il a réagi à sa victoire : »
- n1 « - Le cofondateur de Place publique considère que la primaire est un « tremplin » et que la »
- e0: n0 --causale--> n1 (conf 0.95, marge 3.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-266 silver-B
Malgré l’élan sur le photovoltaïque, les investissements en renouvelables restent insuffisants pour tenir les objectifs de transition.
- n0 « Malgré l’élan sur le photovoltaïque »
- n1 « , les investissements en renouvelables restent insuffisants pour tenir les objectifs de tr »
- e0: n0 --restrictive--> n1 (conf 0.76, marge 1.74)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## ph5-269 silver-B
» Cette formule apocryphe, attribuée à Henry Ford (1863-1947) aux premiers temps de l’automobile, résume un piège dans lequel l’industrie européenne semble être tombée avec la voiture électrique.
- n0 « » »
- n1 « Cette formule apocryphe, attribuée à Henry Ford (1863-1947) aux premiers temps de l’automo »
- e0: n0 --restrictive--> n1 (conf 0.44, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
