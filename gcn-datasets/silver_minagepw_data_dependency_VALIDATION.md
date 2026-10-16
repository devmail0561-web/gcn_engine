# Validation minage-pw — `data_dependency` (68 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-441 (visée `filter`)
- n1 [entite] « Il entre en vigueur le 19 juillet 2013 et »
- n2 [etat_global] « sur le même principe que les deux précédents , à savoir que , sauf critères familiaux , de »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 66]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-41 (visée `data_dependency`)
- n1 [entite] « La façon dont ce signal s' atténue »
- n2 [etat_local] « de la résistivité électrique des milieux rencontrés , explique l' expert , connu pour avoi »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n2 >15tok [9, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-88 (visée `data_dependency`)
- n1 [etat_global] « Le tyran de Syracuse Tout notre système »
- n2 [etat_global] « sur des soins coûteux , ce n' est pas une système qui vise à améliorer la santé et à maint »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 34]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-153 (visée `data_dependency`)
- n1 [processus] « Sans tenir compte de la position philosophique , il est possible de dire que le choix scol »
- n2 [etat_global] « de l' économie politique , où bénéfices et défauts sont la conséquence de l' engagement av »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [19, 40]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-235 (visée `data_dependency`)
- n1 [etat_global] « Modèle d' apprentissage [modifier | modifier le code] Un système d' apprentissage »
- n2 [processus] « sur le choix d' un modèle , c'est à dire d' un objet mathématique , décrit le plus souvent »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [14, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-260 (visée `data_dependency`)
- n1 [etat_global] « Résumés Résumé La régulation de la société internationale »
- n2 [processus] « sur la création et l' application du droit international dont l' un des principes fondamen »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-278 (visée `data_dependency`)
- n1 [processus] « Sans tenir compte de la position philosophique , il est possible de dire que le choix scol »
- n2 [etat_global] « de l' économie politique , où bénéfices et défauts sont la conséquence de l' engagement av »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [19, 40]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-360 (visée `data_dependency`)
- n1 [etat_global] « Modèle d' apprentissage [modifier | modifier le code] Un système d' apprentissage »
- n2 [processus] « sur le choix d' un modèle , c'est à dire d' un objet mathématique , décrit le plus souvent »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [14, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-385 (visée `data_dependency`)
- n1 [etat_global] « Résumés Résumé La régulation de la société internationale »
- n2 [processus] « sur la création et l' application du droit international dont l' un des principes fondamen »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-451 (visée `data_dependency`)
- n1 [entite] « Doctrines et philosophie La notion de Rechtsstaat se »
- n2 [entite] « sur les idées d' Emmanuel Kant et de Friedrich Hegel telles que comprises à travers un pri »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-461 (visée `data_dependency`)
- n1 [processus] « La légitimité »
- n2 [processus] « sur une autorité qui est fondée sur des bases juridiques ou sur des bases éthiques ou mora »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 33]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-476 (visée `data_dependency`)
- n1 [entite] « La science historique »
- n2 [processus] « sur la collecte de sources historiques ou archéologiques (textes , images , archives , tém »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 48]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-480 (visée `data_dependency`)
- n1 [processus] « La suspicion de cancer »
- n2 [entite] « sur plusieurs éléments : la clinique (les symptômes) , les examens biologiques (les prises »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-489 (visée `data_dependency`)
- n1 [processus] « Leur mode de transmission est variable et »
- n2 [processus] « de leur réservoir (humain , animal , environnemental) et parfois de vecteurs (maladies vec »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-493 (visée `data_dependency`)
- n1 [processus] « prévention primaire La prévention primaire »
- n2 [entite] « sur la promotion de la santé et la protection spécifique . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-591 (visée `data_dependency`)
- n1 [condition] « On peut remarquer : qu' elle ne porte pas obligatoirement sur l' ensemble d' une phrase ,  »
- n2 [processus] « de la langue , de la culture et de l' époque concernées . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 86]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-616 (visée `data_dependency`)
- n1 [processus] « Cette technique , qui peut apparaître comme une forme d' art , »
- n2 [processus] « sur la distinction entre la vérité objective d' une proposition et l' apparence de vérité  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [14, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-620 (visée `data_dependency`)
- n1 [processus] « Lorsqu' il s' agit d' une société , cet enregistrement lui confère la personnalité morale  »
- n2 [condition] « de l' objet social de la société , du nombre des apporteurs de capitaux , du montant des c »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 22]; span n2 >15tok [24, 54]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-657 (visée `data_dependency`)
- n1 [processus] « Politiquement , la pensée de Karl Marx »
- n2 [processus] « sur l' analyse et la participation au mouvement réel de la lutte des classes , afin de par »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 67]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-744 (visée `data_dependency`)
- n1 [entite] « La critique interne Elle »
- n2 [entite] « sur la cohérence du texte . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-782 (visée `data_dependency`)
- n1 [condition] « On peut mentionner aussi Roger Bacon , pour qui aucun discours ne peut donner la certitude »
- n2 [entite] « sur l' expérience . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-882 (visée `data_dependency`)
- n1 [entite] « L' importance de cette consommation »
- n2 [processus] « de la façon dont est produite l' électricité ; les émissions de CO2 dépendent du mix élect »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 71]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-904 (visée `data_dependency`)
- n1 [processus] « Ses causes sont multiples — psychologiques , médicales , environnementales — et son traite »
- n2 [processus] « sur des approches comportementales , médicales et hygiéniques adaptées à chaque situation) »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-907 (visée `data_dependency`)
- n1 [processus] « La gravité des lésions »
- n2 [processus] « de leur moment de survenue , de leur situation et de leurs dimensions . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-911 (visée `data_dependency`)
- n1 [processus] « Une des variables d' entrée est sélectionnée à chaque nœud intérieur (ou interne , nœud qu »
- n2 [processus] « de l' algorithme et qui sera discutée plus loin . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-913 (visée `data_dependency`)
- n1 [processus] « Ce dernier , inspiré des neurosciences et du fonctionnement du cortex visuel , »
- n2 [processus] « sur une première couche entraînée de manière non supervi »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-914 (visée `data_dependency`)
- n1 [entite] « Cette méthode »
- n2 [processus] « sur une optimisation séquentielle où chaque nouveau modèle ajuste ses paramètres en minimi »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 26]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-930 (visée `data_dependency`)
- n1 [entite] « Algorithmes La base du calcul »
- n2 [entite] « sur l' apprentissage par arbre de décision . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-947 (visée `data_dependency`)
- n1 [processus] « 1895 1899 : Charles Ernest Overton découvre que la capacité d' une substance à traverser l »
- n2 [processus] « de son caractère hydrophobe . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-950 (visée `data_dependency`)
- n1 [processus] « La mort cellulaire régulée »
- n2 [etat_local] « sur une machinerie moléculaire dédiée , ce qui implique qu' elle peut être modulée (c'est  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-951 (visée `data_dependency`)
- n1 [processus] « La respiration cellulaire »
- n2 [processus] « sur : Un ensemble de réactions métaboliques produisant par oxydoréduction des coenzymes ré »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 56]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-961 (visée `data_dependency`)
- n1 [entite] « L' orientation vers la seconde option »
- n2 [entite] « de la résultante des signaux en compétition reçus à la surface de la cellule . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-968 (visée `data_dependency`)
- n1 [processus] « Le débat sur la nature des virus (vivants ou pas) »
- n2 [processus] « sur des notions complexes et est toujours ouvert dans les années 2020 . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-990 (visée `data_dependency`)
- n1 [entite] « Résines Groupes protecteurs La performance de la synthèse peptidique »
- n2 [processus] « sur la capacité à éliminer les réactions parasites qui font chuter le rendement lors des c »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 30]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1007 (visée `data_dependency`)
- n1 [etat_global] « La communication via ces ports suit un protocole qui »
- n2 [entite] « de l' objectif de l' interaction . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1010 (visée `data_dependency`)
- n1 [entite] « Ce moyen de transmission d' événements »
- n2 [entite] « sur un bus de message store and forward . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1012 (visée `data_dependency`)
- n1 [processus] « Les ingénieurs de la CII observent que le datagramme , utilisable dans les réseaux privés  »
- n2 [processus] « sur des nœuds de commutation plus simples que ceux de la commutation de circuits , imposée »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 16]; span n2 >15tok [18, 40]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1041 (visée `data_dependency`)
- n1 [entite] « Cette initiative , à son tour , »
- n2 [condition] « de la disponibilité et de la pertinence du schéma juridique pour les personnes confrontées »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 63]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1051 (visée `data_dependency`)
- n1 [processus] « En France , l' emploi d' un ingrédient dans un soda »
- n2 [processus] « une autorisation délivrée par les pouvoirs publics après évaluation des risques par une ag »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [13, 57]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1076 (visée `data_dependency`)
- n1 [entite] « La justesse des résultats »
- n2 [processus] « de la justesse de l' analyse sur le court , moyen et long terme , et de la conscience que  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 45]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1096 (visée `data_dependency`)
- n1 [processus] « Les ODD traduisent un changement de paradigme Le changement de paradigme promu par l' Agen »
- n2 [processus] « sur une transformation en profondeur des finalités du développement . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 16]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1105 (visée `data_dependency`)
- n1 [processus] « Dotée de la personnalité juridique , l' Union européenne »
- n2 [processus] « sur un ensemble d' institutions et une administration stables , et dispose d' une incarnat »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1107 (visée `data_dependency`)
- n1 [processus] « Conséquemment , elle »
- n2 [processus] « sur une méthodologie empiriste dans laquelle les approches quantitatives sont dominantes . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1111 (visée `data_dependency`)
- n1 [processus] « La stratégie du gouvernement américain »
- n2 [processus] « sur trois caractéristiques principales : une gestion des sources ouvertes conforme à la do »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 71]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1126 (visée `data_dependency`)
- n1 [entite] « Le calcul du PRG »
- n2 [processus] « sur leur forçage radiatif cumulé sur une période donnée . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1135 (visée `data_dependency`)
- n1 [entite] « Cette définition »
- n2 [processus] « sur la classification des biens en deux dimensions proposée par Ostrom et Ostrom 1977 . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1148 (visée `data_dependency`)
- n1 [processus] « Avenarius considère que le premier objectif de la philosophie est de développer un concept »
- n2 [condition] « sur l' expérience pure et qui s' abstrait de la métaphysique et du matérialisme , abolissa »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]; span n2 >15tok [20, 48]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1149 (visée `data_dependency`)
- n1 [entite] « Cette théorie »
- n2 [etat_local] « sur l' idée que tout ce qui existe est réductible à des éléments neutres qui peuvent être  »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1156 (visée `data_dependency`)
- n1 [processus] « En ce sens , cette technique se rapproche des techniques supervisées en apprentissage auto »
- n2 [entite] « sur un cadre probabiliste . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 28]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1162 (visée `data_dependency`)
- n1 [processus] « Le capitalisme de connivence (en anglais : crony capitalism) est un terme décrivant une éc »
- n2 [processus] « de relations étroites avec les représentants du gouvernement . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1168 (visée `data_dependency`)
- n1 [processus] « Marx explique que le capitalisme , c'est à dire la croissance économique basée sur l' accu »
- n2 [processus] « de l' exploitation du surtravail des ouvriers , mais qu' une société post-capitaliste perm »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]; span n2 >15tok [25, 56]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1199 (visée `data_dependency`)
- n1 [condition] « Causes Les causes peuvent être : génétiques (dont hypersensibilité au froid) médicamenteus »
- n2 [entite] « de l' acrosyndrome et de sa cause . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 81]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1201 (visée `data_dependency`)
- n1 [processus] « Infiltration leucocytaire La formation d' un infiltrat leucocytaire dans l' inflammation »
- n2 [processus] « de l' interaction des leucocytes dans le sang avec la paroi vasculaire . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1213 (visée `data_dependency`)
- n1 [etat_global] « L' auto-régulation d' ensemble »
- n2 [etat_global] « de seuils fonctionnels locaux tels que ceux des sous-systèmes du système organique . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1225 (visée `data_dependency`)
- n1 [processus] « Le problème est que le diagramme déjà construit , à gauche de la ligne , »
- n2 [processus] « de points situés à droite de cette ligne , et donc non encore pris en compte . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [17, 33]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1226 (visée `data_dependency`)
- n1 [entite] « L' algorithme de Heap »
- n2 [processus] « sur le paradigme diviser pour régner . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1227 (visée `data_dependency`)
- n1 [processus] « La particularité de l' algorithme »
- n2 [processus] « sur le choix des deux éléments à transposer . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1228 (visée `data_dependency`)
- n1 [entite] « La méthode de Heap »
- n2 [processus] « sur un comportement remarquable par lequel , au cours de la boucle d' exécution , chacun d »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 34]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1231 (visée `data_dependency`)
- n1 [processus] « Fonctionnement technique L' ADINT »
- n2 [processus] « sur l' architecture technique de la publicité programmatique . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1282 (visée `data_dependency`)
- n1 [entite] « La mondialisation du XXIe siècle »
- n2 [entite] « sur deux facteurs principaux : la faiblesse des coûts de transport au regard des écarts de »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 36]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1285 (visée `data_dependency`)
- n1 [entite] « Le consensus de Jackson Hole »
- n2 [processus] « sur la position défendue par Ben Bernanke et Mark Gertler dans un article paru en 1999 . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1301 (visée `data_dependency`)
- n1 [entite] « Nudge marketing Ou le marketing incitatif , il »
- n2 [processus] « sur les sciences comportementales pour pousser" les individus à prendre des décisions sans »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 25]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1302 (visée `data_dependency`)
- n1 [entite] « Marketing d' influence Il »
- n2 [processus] « sur la collaboration avec des personnes influentes (influenceurs , blogueurs , célébrités) »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1326 (visée `data_dependency`)
- n1 [entite] « Victoria »
- n2 [entite] « de plus en plus du soutien et des conseils d' Albert . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1335 (visée `data_dependency`)
- n1 [processus] « Modes d' organisation La plupart des organisations »
- n2 [entite] « sur un ensemble de mécanismes formels et informels . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1359 (visée `data_dependency`)
- n1 [entite] « Cette prise en charge globale »
- n2 [processus] « sur une évaluation pré-opératoire . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1372 (visée `data_dependency`)
- n1 [processus] « Ce classement est établi par le Conseil international des aéroports (ACI World) (source en »
- n2 [processus] « sur le nombre de passagers à l' embarquement et au débarquement , les passagers en transit »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 16]; span n2 >15tok [18, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1404 (visée `data_dependency`)
- n1 [etat_local] « C' est ce cheminement sensible qui permet d' avoir un regard décalé , de créer de la nouve »
- n2 [entite] « sur l' échange . »
- `['n1']` —`data_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___
