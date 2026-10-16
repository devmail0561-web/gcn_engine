# Validation minage-pw — `filter` (53 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-1 (visée `filter`)
- n1 [entite] « Sur 99 lauréats du prix , »
- n2 [processus] « trois sont des femmes , toutes primées au XXIe siècle . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-50 (visée `filter`)
- n1 [processus] « Bien loin des 75 % visés par les autorités sanitaires et l' Organisation mondiale de la sa »
- n2 [processus] « aux abonnés . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-64 (visée `filter`)
- n1 [entite] « Les jeunes , »
- n2 [processus] « s' ils sont pistonnés , ne trouvent que rarement le bon stage ou la bonne alternance . . . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-90 (visée `filter`)
- n1 [processus] « Jamais on n' aurait pu faire ça tout »
- n2 [entite] « seuls!» Chacun à son rythme . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-91 (visée `filter`)
- n1 [processus] « Jamais on n' aurait pu faire ça tout »
- n2 [entite] « seuls!» , lance Katia . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-99 (visée `filter`)
- n1 [processus] « Devenue la première Française à sortir dans l' espace , Sophie Adenot a effectué une sorti »
- n2 [processus] « quatre Français ont réalisé l' exploit . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-120 (visée `filter`)
- n1 [entite] « Le scrutin , »
- n2 [processus] « électronique , a été perturbé par une cyberattaque , lancée vendredi matin à l' ouverture  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 46]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-124 (visée `filter`)
- n1 [entite] « Sauf »
- n2 [processus] « erreur de ma part , cela n' était pas dans la charte signée les candidats à cette primaire »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 20]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-136 (visée `filter`)
- n1 [processus] « Aucun bot n' a pu voter Le vote , qui se fait »
- n2 [processus] « par voie électronique , a débuté à 8 heures vendredi , mais a été aussitôt perturbé par un »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n2 >15tok [14, 50]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-167 (visée `filter`)
- n1 [processus] « Dans cet article , nous considérons »
- n2 [processus] « la protection des données personnelles des consommateurs . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-168 (visée `filter`)
- n1 [processus] « Il existe de nombreux autres instruments de privacy (Rochelandet [2010] ; Rallet et Rochel »
- n2 [etat_local] « ici ceux susceptibles d' avoir un impact sur la concurrence . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 19]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-169 (visée `filter`)
- n1 [processus] « La portabilité est envisagée ici »
- n2 [condition] « du côté du consommateur , mais elle se pose aussi du point de vue des exploitants , par ex »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-183 (visée `filter`)
- n1 [processus] « Les actions de plus grandes dimensions nécessitées par l' état du cours d' eau , du canal  »
- n2 [etat_global] « de procédures loi sur l' eau . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-217 (visée `filter`)
- n1 [etat_local] « Les enfants devaient être en âge scolaire et être suivis »
- n2 [entite] « par un médecin généraliste . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-221 (visée `filter`)
- n1 [processus] « Les protéines hnRNP U et PARP 1 seraient directement impliquées dans J' induction de l' ap »
- n2 [processus] « lors de l' arrêt de croissance , comparativement aux CBF A et BUB3 qui sont présents dans  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]; span n2 >15tok [23, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-222 (visée `filter`)
- n1 [processus] « Le Kif4 est une protéine associée aux microtubules et spécialisée dans le transport de com »
- n2 [processus] « en croissance cellulaire . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 24]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-245 (visée `filter`)
- n1 [processus] « En France , il semblerait que l' incidence du cholangiocarcinome soit globalement en augme »
- n2 [entite] « pour le cholangiocarcinome extra-hépatique en diminution chez la femme . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 19]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-292 (visée `filter`)
- n1 [processus] « Dans cet article , nous considérons »
- n2 [processus] « la protection des données personnelles des consommateurs . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-293 (visée `filter`)
- n1 [processus] « Il existe de nombreux autres instruments de privacy (Rochelandet [2010] ; Rallet et Rochel »
- n2 [etat_local] « ici ceux susceptibles d' avoir un impact sur la concurrence . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 19]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-294 (visée `filter`)
- n1 [processus] « La portabilité est envisagée ici »
- n2 [condition] « du côté du consommateur , mais elle se pose aussi du point de vue des exploitants , par ex »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-308 (visée `filter`)
- n1 [processus] « Les actions de plus grandes dimensions nécessitées par l' état du cours d' eau , du canal  »
- n2 [etat_global] « de procédures loi sur l' eau . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-342 (visée `filter`)
- n1 [etat_local] « Les enfants devaient être en âge scolaire et être suivis »
- n2 [entite] « par un médecin généraliste . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-346 (visée `filter`)
- n1 [processus] « Les protéines hnRNP U et PARP 1 seraient directement impliquées dans J' induction de l' ap »
- n2 [processus] « lors de l' arrêt de croissance , comparativement aux CBF A et BUB3 qui sont présents dans  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]; span n2 >15tok [23, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-347 (visée `filter`)
- n1 [processus] « Le Kif4 est une protéine associée aux microtubules et spécialisée dans le transport de com »
- n2 [processus] « en croissance cellulaire . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 24]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-370 (visée `filter`)
- n1 [processus] « En France , il semblerait que l' incidence du cholangiocarcinome soit globalement en augme »
- n2 [entite] « pour le cholangiocarcinome extra-hépatique en diminution chez la femme . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 19]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-401 (visée `filter`)
- n1 [processus] « États Unis d' Amérique) de 1986 ; Un État peut souscrire à une déclaration facultative de  »
- n2 [processus] « 58 États sur 185 ont souscrit à une telle déclaration[réf . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 61]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-413 (visée `filter`)
- n1 [processus] « Originellement , les »
- n2 [processus] « sujets de ce droit sont les États , mais la multiplication des organisations international »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-414 (visée `filter`)
- n1 [processus] « Cette interprétation va néanmoins à l' encontre des fondements traditionnels volontaristes »
- n2 [processus] « les sujets du droit international peuvent créer , appliquer ou veiller à l' application de »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [19, 38]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-429 (visée `filter`)
- n1 [processus] « De même , l' article 16 de la Constitution énumère avec un apparent luxe de détails les év »
- n2 [processus] « susceptibles de justifier sa mise en œuvre . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-447 (visée `filter`)
- n1 [entite] « Tous les États de l' Union , »
- n2 [processus] « l' Irlande (qui bénéficie d' une exemption) , doivent , à terme , mettre en œuvre l' acqui »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-450 (visée `filter`)
- n1 [processus] « Étendue L' espace Schengen comprend actuellement 29 États membres dont tous »
- n2 [processus] « quatre sont membres de l' Union européenne . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-452 (visée `filter`)
- n1 [processus] « Outre les différentes causes possibles d' un risque juridique perçu ou non par une organis »
- n2 [processus] « juridiques mais peuvent conduire à d' autres impacts financiers , économiques , sociaux vo »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 32]; span n2 >15tok [34, 49]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-453 (visée `filter`)
- n1 [entite] « À noter que le terme État" ne sous entend pas »
- n2 [processus] « des États ou des pays souverains , et peut également faire référence aux conflits de lois  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [12, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-465 (visée `filter`)
- n1 [processus] « Application Dans sa version stricte (nationalité inconditionnelle pour les personnes nées  »
- n2 [processus] « la République Dominicaine et la Colombie , et seulement dans très peu de pays en dehors de »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 29]; span n2 >15tok [31, 61]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-481 (visée `filter`)
- n1 [etat_local] « Marqueur tumoral Imagerie médicale : radiographie , scanner , imagerie par résonance magné »
- n2 [processus] « rares exceptions , que par un prélèvement de la tumeur et son analyse au microscope par l' »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 35]; span n2 >15tok [37, 55]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-517 (visée `filter`)
- n1 [processus] « À »
- n2 [processus] « l' exception de l' exercice étudiant , l' inscription au tableau de l' Ordre national des  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-518 (visée `filter`)
- n1 [processus] « À »
- n2 [processus] « l' exception de l' exercice étudiant , l' inscription au tableau de l' Ordre national des  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 30]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-551 (visée `filter`)
- n1 [entite] « Selon l' OMS en 2021 , »
- n2 [etat_global] « trente huit pays ont mis en place une politique nationale de prévention du suicide . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-561 (visée `filter`)
- n1 [entite] « Cet article traite »
- n2 [processus] « du dépistage pratiqué au niveau du col de l' utérus . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-575 (visée `filter`)
- n1 [processus] « Il a défendu la possibilité d' une connaissance scientifique basée »
- n2 [processus] « sur un raisonnement inductif et une observation minutieuse des événements de la nature . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-596 (visée `filter`)
- n1 [processus] « En août 2012 , un essai randomisé en double aveugle confirme la supériorité sur le placebo »
- n2 [entite] « l' échelle de handicap de Sheehan . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-617 (visée `filter`)
- n1 [entite] « La dialectique se distingue alors de la logique et de la sophistique , car celle ci vise »
- n2 [processus] « le faux et suppose une connaissance du vrai , ce qui n' est pas le cas dans la dialectique »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [19, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-621 (visée `filter`)
- n1 [entite] « En l' état actuel du droit français »
- n2 [processus] « des aspects parcellaires de l' entreprise , comme le capital , la fidélité de la clientèle »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-634 (visée `filter`)
- n1 [processus] « Au XIXe siècle , la notion d' espace absolu est peu à peu abandonnée au profit des »
- n2 [entite] « référentiels galiléens . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-639 (visée `filter`)
- n1 [processus] « Très fréquent sur le pourtour méditerranéen , il est pratiquement absent au nord de la Loi »
- n2 [etat_global] « l' exception des côtes atlantiques[Où ?]) et ne dépasse en principe pas 700 m . d' altitud »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [18, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-647 (visée `filter`)
- n1 [processus] « En examinant la conscience morale commune , Kant dégage le fait que rien n' est bon en soi »
- n2 [processus] « une volonté bonne . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-659 (visée `filter`)
- n1 [entite] « Ici , — »
- n2 [etat_local] « peut être dans Le Crépuscule des idoles — , Nietzsche a voulu clarifier . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-664 (visée `filter`)
- n1 [processus] « La clé étant changée »
- n2 [processus] « une fois par jour , tous les messages interceptés dans une même journée étaient cassés sim »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 22]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-699 (visée `filter`)
- n1 [processus] « Cette forme diffère de la forme normale »
- n2 [processus] « par sa conformation , qui la rend particulièrement hydrophobe , ce qui explique la formati »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-708 (visée `filter`)
- n1 [etat_local] « Le virus peut être contrôlé pour infecter »
- n2 [processus] « des cellules qui répondent à la présence de glucose , de sorte que l' insuline n' est prod »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-711 (visée `filter`)
- n1 [processus] « Le film est réputé perdu dans sa version intégrale , »
- n2 [processus] « quelques fragments subsistent . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-716 (visée `filter`)
- n1 [processus] « Seuls »
- n2 [entite] « circulent encore , de nuit , quelques trains de grandes lignes . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-743 (visée `filter`)
- n1 [processus] « Enfin , la recherche et la critique des matériaux ne devraient pas se limiter aux »
- n2 [entite] « documents écrits . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___
