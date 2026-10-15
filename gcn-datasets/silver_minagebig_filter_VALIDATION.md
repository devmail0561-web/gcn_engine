# Validation minage-big — `filter` (58 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-272 (visée `filter`)
- n1 [processus] « La mission de 2009 a préconisé au gouvernement français de calculer les émissions indirect »
- n2 [processus] « à celles dont on est directement responsable et sur lesquelles on peut influer (notamment  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 20]; span n2 >15tok [22, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-423 (visée `filter`)
- n1 [etat_global] « Dans les années 1920 , le climatologue Wladimir Peter Köppen met au point un système de cl »
- n2 [processus] « sur les précipitations et les températures . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 20]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-547 (visée `filter`)
- n1 [processus] « On trouve ce climat au nord et au nord est des États Unis , dans l' est du Canada , au sud »
- n2 [processus] « dans l' extrême sud , au sud du Manitoba , au sud est de la Saskatchewan , au centre de l' »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 27]; span n2 >15tok [29, 61]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-580 (visée `filter`)
- n1 [processus] « Les climats tempérés sont rares sur tout le continent »
- n2 [processus] « à très haute altitude et le long des franges et dans une partie du sud du continent . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 28]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-790 (visée `filter`)
- n1 [processus] « la majeure partie de l' Arctique est encore couverte de neige , de sorte que la surface de »
- n2 [processus] « l' exception de la mer de Norvège et du sud de la mer de Béring , où l' océan est libre de »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n1 >15tok [2, 38]; span n2 >15tok [40, 90]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-808 (visée `filter`)
- n1 [processus] « Les températures hivernales sont en moyenne inférieures au point de congélation dans l' en »
- n2 [processus] « l' exception de petites régions du sud de la mer de Norvège et de la mer de Béring , qui r »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]; span n2 >15tok [20, 50]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-809 (visée `filter`)
- n1 [processus] « En été , les températures moyennes sont supérieures au point de congélation dans toutes le »
- n2 [processus] « l' exception du centre du bassin arctique , où la glace de mer survit pendant l' été , et  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]; span n2 >15tok [19, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-848 (visée `filter`)
- n1 [processus] « Dans la majeure partie de l' Arctique , les précipitations se présentent »
- n2 [entite] « sous forme de pluie et de neige . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1018 (visée `filter`)
- n1 [processus] « Les précipitations y sont très peu abondantes ou tombent »
- n2 [processus] « sous forme de neige dans certaines régions désertiques froides . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1021 (visée `filter`)
- n1 [processus] « Il faut bien voir que les climats des déserts chauds appartiennent aux deux premières clas »
- n2 [processus] « à la première . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1022 (visée `filter`)
- n1 [processus] « De façon globale , les climats désertiques qu' ils soient chauds ou froids sont associés »
- n2 [processus] « des précipitations très faibles , à une amplitude thermique journalière et annuelle potent »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [17, 58]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1075 (visée `filter`)
- n1 [processus] « Cette zone est moyennement grande et recouvre tous les pays qui possèdent des parties dése »
- n2 [processus] « l' exception de ceux dont le désert est intégralement hyper-aride . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1082 (visée `filter`)
- n1 [processus] « Presque tous les pays couverts par le désert possèdent de vastes étendues hyper-arides »
- n2 [processus] « l' exception du Maroc et de la Tunisie mais dont certains endroits frôlent de quelques mm  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [15, 33]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1110 (visée `filter`)
- n1 [processus] « Les vents sont rares »
- n2 [entite] « quelques brises marines sur les littoraux et d' exceptionnelles tornades . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1333 (visée `filter`)
- n1 [processus] « Traditionnellement , la classification des climats secs et humides reposait »
- n2 [processus] « sur les précipitations annuelles moyennes . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1392 (visée `filter`)
- n1 [processus] « Les masses d' air étant caractérisées par des flux méridiens , les températures peuvent au »
- n2 [processus] « en été) . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 20]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1414 (visée `filter`)
- n1 [processus] « Toutes les régions de climat subtropical , »
- n2 [processus] « celles européennes et le sud est du Brésil , sont touchées par des tempêtes et des cyclone »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1432 (visée `filter`)
- n1 [processus] « En moyenne , il tombe environ 1 000 mm de précipitations »
- n2 [processus] « dans certaines régions comme près de l' Himalaya en Asie . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1436 (visée `filter`)
- n1 [processus] « Le gel est très rare dans cette région »
- n2 [entite] « parfois en altitude . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1976 (visée `filter`)
- n1 [processus] « En 2021 , dans 95 pays interrogés par l' OMS , »
- n2 [processus] « neuf incluaient un volet santé mentale et de soutien psychosocial dans leurs plans nationa »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [13, 34]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2162 (visée `filter`)
- n1 [entite] « Début 2010 , »
- n2 [processus] « les 3 premiers satellites de la série sont budgétés à la suite de restrictions budgétaires »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2335 (visée `filter`)
- n1 [entite] « Sauf »
- n2 [processus] « dans les déserts arides et parfois dans les zones tropicales , une grande partie du sol na »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2403 (visée `filter`)
- n1 [processus] « En reposant »
- n2 [etat_local] « sur une moyenne arithmétique calculée sur une fenêtre temporelle fixe , il a été montré qu »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [4, 62]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2957 (visée `filter`)
- n1 [processus] « L' esclavage est progressivement rétabli par Bonaparte en 1802 , »
- n2 [entite] « à Saint-Domingue qui accède à l' indépendance en 1804 sous le nom d' Haïti . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2970 (visée `filter`)
- n1 [processus] « Les élections sont rétablies mais elles se déroulent à plusieurs degrés et »
- n2 [processus] « les 600 citoyens les plus imposés peuvent faire partie du collège départemental . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-3478 (visée `filter`)
- n1 [etat_local] « Le Centre de recherche du château de Versailles aspire à être un lieu de rencontre interna »
- n2 [entite] « aux chercheurs . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 20]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4140 (visée `filter`)
- n1 [processus] « En France , l' utilisation de la catégorisation raciale n' a jamais eu cours en dehors de  »
- n2 [processus] « pour des questions de signalement policiers pour les délinquants recherchés , la police ut »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 49]; span n2 >15tok [51, 74]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4793 (visée `filter`)
- n1 [processus] « Ce succès sera de courte durée , et »
- n2 [etat_global] « Les verts parviendront à s' imposer durablement dans le paysage politique avec des scores  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5539 (visée `filter`)
- n1 [processus] « La LzVN prône l' indépendance des États africains , soutient surtout la défense de l' indé »
- n2 [processus] « états libres du continent) , encourage les Allemands noirs à rejoindre les syndicats allem »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 24]; span n2 >15tok [26, 60]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5659 (visée `filter`)
- n1 [entite] « En effet , »
- n2 [processus] « les Francs en tant que classe conquérante avaient le statut d' hommes libres . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5900 (visée `filter`)
- n1 [processus] « Dans cette chasse à l' anarchiste , les autorités françaises commencent à établir des list »
- n2 [processus] « , on trouve plus de cinq cent noms dans la liste d' anarchistes français , sans compter le »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 22]; span n2 >15tok [24, 50]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6263 (visée `filter`)
- n1 [processus] « Les premiers outils en pierre sont relativement peu élaborés techniquement mais impliquent »
- n2 [entite] « l' exception de l' homme . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 46]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6782 (visée `filter`)
- n1 [processus] « En effet , selon les données de l' Office fédéral de la statistique suisse , »
- n2 [processus] « 14,5% des entreprises helvétiques ont été fondées par des femmes . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-7974 (visée `filter`)
- n1 [processus] « Le terme pollution de l' air signifie généralement l' introduction directe ou indirecte da »
- n2 [etat_local] « l' exception des espaces confinés) par l' homme de toute substance susceptible d' avoir de »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]; span n2 >15tok [19, 46]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8032 (visée `filter`)
- n1 [processus] « le stockage , ou l' enfouissement dans des décharges : en général , il est préférable de s »
- n2 [entite] « les déchets ultimes , comme les résidus d' incinération ; »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8166 (visée `filter`)
- n1 [processus] « Sauf »
- n2 [processus] « indication contraire , les sources présentées ici sont exclusivement en français ((fr)) . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8248 (visée `filter`)
- n1 [processus] « Une autre étude dans une commune de Kisangani a montré que la majorité des écoles disposen »
- n2 [processus] « 25 % avaient de l' eau et du savon disponibles pour le lavage des mains . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 33]; span n2 >15tok [35, 50]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8298 (visée `filter`)
- n1 [processus] « Elle s' applique »
- n2 [entite] « toutes les eaux destinées à la consommation humaine à l' exception des eaux minérales natu »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 24]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8682 (visée `filter`)
- n1 [processus] « Faune sessile : Organismes très souvent marin , vivant »
- n2 [processus] « ou en colonies et fixés généralement sur une roche . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10654 (visée `filter`)
- n1 [entite] « Le choix des termes identitaires ne relève pas »
- n2 [condition] « d' enjeux d' autonomination précise de son orientation sexuelle , mais servent aussi de si »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [10, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10886 (visée `filter`)
- n1 [processus] « Comme exemple , ils ont utilisé l' étude à Burton et Whiting , (1960) qui a démontré que l »
- n2 [processus] « des cent sociétés qui ont une chambre séparée pour leurs nouveau nés . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 25]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10999 (visée `filter`)
- n1 [etat_local] « Le point de vue du prototype suscite cependant des inquiétudes , car les individus stockan »
- n2 [processus] « spécifiques à leur morphologie , et peuvent ne pas s' appliquer à toutes les personnes app »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 32]; span n2 >15tok [34, 80]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11135 (visée `filter`)
- n1 [entite] « L' institution forme »
- n2 [entite] « au niveau du bachelor sur une durée de neuf semestres . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11843 (visée `filter`)
- n1 [entite] « Dans les faits , il semble que »
- n2 [processus] « 20 % des gens répondent marteau rouge et 2/3 ont soit marteau soit rouge dans leur réponse »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 26]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12277 (visée `filter`)
- n1 [processus] « Le soin en santé mentale communautaire ne se concentre pas »
- n2 [processus] « sur les déficits et les limitations des personnes (perspective de la maladie) mais égaleme »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [12, 36]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12311 (visée `filter`)
- n1 [processus] « Ils tiennent la réunion quotidienne la plus importante en début de journée et ils rencontr »
- n2 [processus] « ou en groupe , dans d' autres pièces de la maison tout au long de la journée . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]; span n2 >15tok [23, 40]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12422 (visée `filter`)
- n1 [processus] « Les personnes n' internalisent pas tous les comportements de leur parent: les identificati »
- n2 [entite] « sur certains sujets . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-13148 (visée `filter`)
- n1 [processus] « Seuls »
- n2 [processus] « les deux premiers ont une interprétation naturelle , celle d' une causalité entre les deux »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-14336 (visée `filter`)
- n1 [etat_local] « Le concept d' atome est particulièrement bien admis par le grand public , pourtant , parad »
- n2 [processus] « quelques rares physiciens manipulent des atomes isolés . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15342 (visée `filter`)
- n1 [processus] « Dans un appareil électronique : les micrologiciels sont utilisés pour réaliser des automat »
- n2 [entite] « avec des circuits électroniques . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n1 >15tok [2, 17]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15421 (visée `filter`)
- n1 [processus] « Apple et Oracle (Sun) sont parmi les »
- n2 [processus] « constructeurs spécialisés à la fois dans le matériel et le logiciel . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15422 (visée `filter`)
- n1 [processus] « IBM et HP sont parmi les »
- n2 [processus] « constructeurs spécialisés à la fois dans le matériel et les services . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15427 (visée `filter`)
- n1 [processus] « Les constructeurs fournissaient »
- n2 [etat_global] « un système d' exploitation , et assistaient leurs clients par l' organisation de cours de  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [5, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15879 (visée `filter`)
- n1 [etat_global] « Les phases finales de la vie des étoiles , y compris les nébuleuses planétaires et les sup »
- n2 [processus] « formées à partir d' hydrogène et d' hélium . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 42]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15953 (visée `filter`)
- n1 [processus] « Excepté »
- n2 [processus] « pour les longueurs d' onde situées près de la lumière visible , le rayonnement infrarouge  »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [2, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-16622 (visée `filter`)
- n1 [processus] « La recherche sur l' effet rebond s' est d' abord construite en économie , en attribuant l' »
- n2 [entite] « à un effet prix . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-16711 (visée `filter`)
- n1 [etat_local] « Les résultats obtenus semblent indiquer que la recherche d' informations peut être biaisée »
- n2 [processus] « dans des situations où il existe un lien causal entre les actions de l' agent et le dommag »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 22]; span n2 >15tok [24, 42]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27554 (visée `data_dependency`)
- n1 [condition] « Il affirme ainsi que cela pourrait soigner les personnes atteintes de cancer ou SIDA , san »
- n2 [processus] « à celle ci afin de soigner de telles maladies . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 83]
- [ ] oui  [ ] non  [ ] corriger : ___
