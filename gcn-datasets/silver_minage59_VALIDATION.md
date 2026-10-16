# Validation minage59 — 58 proposées classes rares (PROPOSITION, stade propose)

Source : pipeline build_lot (lattice + balanced_auto + normalize D5) sur 59 silver purs.
54/58 relation = classe visée ; 40 litig = spans >15tok (re-découpe manuelle requise, guide).
Cocher UNE case. Promotion = validation + gate IC95.

### g0-1883 (visée `prevent`)
- n1 [entite] « À part le fait que le climat froid nous »
- n2 [processus] « de plusieurs maladies infectieuses transmises par les animaux , il est également source de »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 28]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4120 (visée `prevent`)
- n1 [processus] « Pendant les siècles qui suivent , les Français se protègent des vagues d' immigration , co »
- n2 [entite] « les Sarrazins . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4915 (visée `prevent`)
- n1 [processus] « L' année suivante , lui est adjointe la mission d' assurer le »
- n2 [processus] « contre-espionnage français afin d' éviter toute pénétration étrangère dans l' appareil d'  »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5453 (visée `prevent`)
- n1 [etat_global] « Selon la déclaration des droits de l' homme de 1795 , L' égalité consiste en ce que la loi »
- n2 [entite] « , soit qu' elle punisse . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 28]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-7302 (visée `prevent`)
- n1 [processus] « Des économistes , dont Lawrence Summers , recommandèrent des efforts de coopération pour s »
- n2 [processus] « exiger de grandes cotisations sociales en retour . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 64]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8560 (visée `prevent`)
- n1 [processus] « En absorbant la plus grande partie du rayonnement solaire ultraviolet , elle »
- n2 [entite] « les êtres vivants et les écosystèmes . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10738 (visée `prevent`)
- n1 [processus] « Ces caractéristiques comportementales comprennent , »
- n2 [etat_local] « s' y limiter , le fait de : éviter activement la confrontation , avoir de la difficulté à  »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 61]
- [ ] oui  [ ] non  [ ] corriger : ___

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

### g0-853 (visée `filter`)
- n1 [etat_local] « d' être mesurées . »
- n2 [processus] « Sauf indication contraire , toutes les quantités de précipitations indiquées dans cet arti »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 29]
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

### g0-1399 (visée `filter`)
- n1 [processus] « 8 mois sont supérieurs à 10 °C de moyenne . »
- n2 [etat_local] « Outre la classification initiale basique de Köppen , Trewartha a proposé d' affiner celle  »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 28]
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

### g0-2424 (visée `filter`)
- n1 [processus] « lieu de s' affaiblir avec les changements climatiques , selon un climatologue de l' univer »
- n2 [processus] « C' est l' un des seuls systèmes atmosphériques de la planète qui se renforce »
- `['n1']` —`opposition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [16, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2880 (visée `control_dependency`)
- n1 [entite] « À la même date , la colonie passe sous »
- n2 [entite] « royal direct . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2890 (visée `control_dependency`)
- n1 [processus] « En 1635 , la Guadeloupe et la Martinique passent sous »
- n2 [entite] « français . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-3129 (visée `control_dependency`)
- n1 [entite] « Le 18 mars 1871 , Thiers »
- n2 [processus] « de désarmer les Parisiens . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-3881 (visée `control_dependency`)
- n1 [processus] « Pendant ce temps , un des détachements a pris le »
- n2 [processus] « de la porte et a abaissé le pont suspendu aux cris de France !" . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4492 (visée `control_dependency`)
- n1 [processus] « Il bénéficie du soutien du Premier ministre Michel Rocard , avec lequel il »
- n2 [entite] « la réforme des services . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4660 (visée `control_dependency`)
- n1 [processus] « À la fin 1960 , les événements s' accélèrent : le GPRA demande un référendum sous »
- n2 [processus] « de l' ONU , le général de Gaulle évoque de plus en plus l' Algérie algérienne et organise  »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 16]; span n2 >15tok [18, 49]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5739 (visée `control_dependency`)
- n1 [processus] « En 1748 , le comte de la Galissoniere , le plus haut fonctionnaire français en Amérique du »
- n2 [processus] « à Pierre Céloron de Blainville d' emmener 250 soldats français dans l' Ohio Country pour r »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 19]; span n2 >15tok [21, 53]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5764 (visée `control_dependency`)
- n1 [processus] « L' affaire de Jumonville »
- n2 [processus] « une série d' événements qui conduisent à la Guerre de Sept Ans ou la Conquête . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [6, 21]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6106 (visée `control_dependency`)
- n1 [entite] « Le 16 mars , il »
- n2 [processus] « aux trompettes de sortir discrètement de la ville et de se positionner sur les hauteurs al »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6109 (visée `control_dependency`)
- n1 [processus] « Au milieu de la nuit du 16 au 17 mars , sombre et pluvieuse , Vaudrey »
- n2 [processus] « à toutes les trompettes de sonner en même temps . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 16]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-6112 (visée `control_dependency`)
- n1 [etat_local] « Complètement paniqué et croyant être submergé par des troupes ennemis , le sieur de Craon »
- n2 [processus] « la retraite vers l' ouest , mais celle ci tourne très rapidement en déroute . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-7757 (visée `control_dependency`)
- n1 [entite] « En août 2016 , Digital Rights Watch »
- n2 [processus] « une campagne , comprenant une couverture médiatique , exprimant son inquiétude quant aux p »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [9, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-8520 (visée `control_dependency`)
- n1 [processus] « Cerema : le Centre d' études et d' expertise sur les risques , l' environnement , la mobil »
- n2 [processus] « des moyens à mettre en œuvre . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]
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

### g0-2957 (visée `data_dependency`)
- n1 [processus] « L' esclavage est progressivement rétabli par Bonaparte en 1802 , »
- n2 [entite] « à Saint-Domingue qui accède à l' indépendance en 1804 sous le nom d' Haïti . »
- `['n1']` —`filter`→ `n2` conf=0.9 explicit=True
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
