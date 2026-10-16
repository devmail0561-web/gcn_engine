# Validation minage-big — `data_dependency` (35 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-22550 (visée `prevent`)
- n1 [entite] « Il empêche l' angiogenèse »
- n2 [entite] « à l' oxygénation des tumeurs . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-743 (visée `data_dependency`)
- n1 [processus] « Elles ont permis de connaître le climat le plus extrême de l' Arctique et de suggérer pour »
- n2 [processus] « sur une dépression du socle rocheux (dont on sait aujourd' hui qu' elle est causée par le  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]; span n2 >15tok [25, 47]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1315 (visée `data_dependency`)
- n1 [processus] « Ceci implique que la pluviométrie maximale d' un climat sec est variable et »
- n2 [processus] « de la température et de la saisonnalité des précipitations , bien qu' en général elle incl »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [15, 46]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1897 (visée `data_dependency`)
- n1 [processus] « La vulnérabilité des populations à ces catastrophes naturelles »
- n2 [processus] « de paramètres tels que lieu de vie , âge ou encore environnement social des communautés to »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 26]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2406 (visée `data_dependency`)
- n1 [processus] « La durée optimale des relevés utilisés pour établir ces normales »
- n2 [processus] « de l' élément étudié , de la localisation géographique et de l' importance de la tendance  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [12, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4165 (visée `data_dependency`)
- n1 [etat_global] « La politique d' intégration »
- n2 [processus] « sur des politiques volontaires qui visent à créer une identité commune et à l' intériorisa »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5241 (visée `data_dependency`)
- n1 [processus] « Dans son essai Sur la France , Robert Lafont considère que chacun appartient à deux nation »
- n2 [processus] « sur une idéologie et des intérêts communs résultant du cours des événements . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 33]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5572 (visée `data_dependency`)
- n1 [entite] « Le régime »
- n2 [processus] « sur le texte la Constitution du 6 messidor de l' an I qui est élaboré par la Convention mo »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 25]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6819 (visée `data_dependency`)
- n1 [entite] « Dans ce cas , le modèle par blocs fonctionnels »
- n2 [processus] « sur l' approximation des grandeurs distribuées intervenant en l' acoustique théorique . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6822 (visée `data_dependency`)
- n1 [processus] « La validité de cette approximation »
- n2 [processus] « sur le fait que la plus courte longueur d' onde acoustique est très inférieure à la plus g »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6825 (visée `data_dependency`)
- n1 [entite] « Cette approximation »
- n2 [processus] « sur le fait que la plus courte longueur d' onde acoustique est beaucoup plus grande que la »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-7413 (visée `data_dependency`)
- n1 [processus] « La critique de la société industrielle exposée par Illich dans La convivialité en 1973 »
- n2 [entite] « sur une analyse non marxiste et non anarchiste du capitalisme . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-9034 (visée `data_dependency`)
- n1 [processus] « Impératifs de croissance/dépendance à la croissance : ensemble de mécanismes institutionne »
- n2 [processus] « ou désirable (financement des services publics , maintien du plein emploi , remboursement  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [16, 40]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-9257 (visée `data_dependency`)
- n1 [processus] « Il y affirme que la crise écologique actuelle est inséparable de la crise spirituelle et s »
- n2 [processus] « sur une compréhension renouvelée du rapport entre Dieu , l' humanité et la création . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 25]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-9563 (visée `data_dependency`)
- n1 [etat_global] « Ce mécanisme »
- n2 [etat_global] « sur la création d' une frontière symbolique structurée autour de l' opposition entre un no »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10263 (visée `data_dependency`)
- n1 [processus] « L' ambiguïté lexicale »
- n2 [processus] « sur un fait d' homonymie ; l' ambiguïté syntaxique ou grammaticale repose sur une construc »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11369 (visée `data_dependency`)
- n1 [entite] « Le diagnostic »
- n2 [processus] « sur l' observation clinique du comportement du patient . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11570 (visée `data_dependency`)
- n1 [entite] « La thérapie »
- n2 [processus] « sur la théorie selon laquelle les problèmes de mentalisation peuvent entraîner des difficu »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 38]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12025 (visée `data_dependency`)
- n1 [entite] « La Sociocratie »
- n2 [etat_global] « sur 4 principes : la prise de décision par consentement , l' organisation en cercles , le  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12138 (visée `data_dependency`)
- n1 [entite] « La rédaction de chacun des articles »
- n2 [processus] « sur une validation statistique confortable: le nombre de rêves sur lesquels s' appuient le »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n2 >15tok [8, 48]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12426 (visée `data_dependency`)
- n1 [etat_local] « L' ordre dans lesquels ces SPI doivent être traités en thérapie »
- n2 [etat_local] « de chaque personne : en effet , certaines personnes ont trop de défenses pour que le théra »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [13, 38]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12583 (visée `data_dependency`)
- n1 [processus] « On peut notamment citer les troubles psychotiques aigus , trouble dépressif sévère , un tr »
- n2 [processus] « une hospitalisation prochainement , une problématique de dépendance avec consommation actu »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 28]; span n2 >15tok [30, 50]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-13457 (visée `data_dependency`)
- n1 [etat_local] « Cette approche , due en particulier à Gödel , est tout à fait compatible avec la logique c »
- n2 [processus] « sur une interprétation en termes de valeurs de vérité . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 54]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-16634 (visée `data_dependency`)
- n1 [processus] « Il montre que la majorité de l' effet rebond est causé par un effet prix et que l' effet d »
- n2 [processus] « de la nature exogène/endogène du choix de la technologie devenue plus efficace . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 22]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-18516 (visée `data_dependency`)
- n1 [processus] « Bien que l' activité de signalisation de la voie TGF β détermine la force et la durée d' u »
- n2 [condition] « de déterminants contextuels tels que le type et l' état développemental de la cellule cibl »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 26]; span n2 >15tok [28, 53]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-23431 (visée `data_dependency`)
- n1 [entite] « Sa pédagogie »
- n2 [processus] « sur l' éducation multisensorielle et haptique de l' enfant . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-23437 (visée `data_dependency`)
- n1 [entite] « Ce projet d' éducation nouvelle s' inscrit dans la promotion de la paix et du progrès ; il »
- n2 [processus] « sur trois piliers : une posture particulière de l' éducateur , un environnement préparé et »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]; span n2 >15tok [20, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-23760 (visée `data_dependency`)
- n1 [processus] « Malgré l' abondance des ressources hydriques du pays , une grande partie des ménages rurau »
- n2 [processus] « de sources d' eau non protégées , exposées à la contamination , ce qui constitue un enjeu  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [17, 42]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-25465 (visée `data_dependency`)
- n1 [processus] « Avril 2010 11 avril : les responsables politiques des pays de la zone euro se mettent d' a »
- n2 [processus] « sur des prêts bilatéraux (durée trois ans , taux d' intérêt environ 5 %) d' un montant de  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 27]; span n2 >15tok [29, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-25610 (visée `data_dependency`)
- n1 [entite] « À notre époque , la médecine »
- n2 [processus] « sur des avancées technologiques telles que les scanners , les radiographies , et d' autres »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [8, 26]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-26248 (visée `data_dependency`)
- n1 [processus] « La formation de l' empire »
- n2 [entite] « sur la charte du Manden qui organise les rapports entre les différentes composantes social »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 36]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-26252 (visée `data_dependency`)
- n1 [entite] « Son économie »
- n2 [processus] « sur l' agriculture , l' exploitation minière , l' artisanat et le commerce transsaharien . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-26800 (visée `data_dependency`)
- n1 [entite] « L' hypothèse originelle »
- n2 [etat_global] « sur le concept d' homéostasie et soutient que les formes vivantes d' une planète hôte , as »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27132 (visée `data_dependency`)
- n1 [processus] « Par exemple , l' astrologie est aujourd' hui considérée comme obscurantiste , car elle »
- n2 [processus] « sur des interprétations peu compatibles avec les observations scientifiques modernes . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27332 (visée `data_dependency`)
- n1 [entite] « Cette critique »
- n2 [etat_global] « sur une confusion entre le caractère obligatoire d' une norme et sa validité ; qualifier u »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 44]
- [ ] oui  [ ] non  [ ] corriger : ___
