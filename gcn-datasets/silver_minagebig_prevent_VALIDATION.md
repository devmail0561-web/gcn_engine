# Validation minage-big — `prevent` (27 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

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

### g0-11154 (visée `prevent`)
- n1 [processus] « Par exemple , l' action de la cocaïne est expliquée par le fait qu' elle »
- n2 [processus] « la recapture de certains neurotransmetteurs comme la dopamine , la noradrénaline et la sér »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [17, 45]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11489 (visée `prevent`)
- n1 [processus] « Dans la rhétorique complotiste , l' inversion accusatoire est utilisée par le créateur de  »
- n2 [processus] « leurs auteurs : ainsi , les journalistes sont accusés de cacher la vérité , les scientifiq »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 23]; span n2 >15tok [25, 76]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12296 (visée `prevent`)
- n1 [processus] « La relation entre le patient et le psychiatre est également revue , le recours à la conten »
- n2 [processus] « la future alliance thérapeutique entre le soignant et le soigné . »
- `['n1']` —`prevent`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [1, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12411 (visée `prevent`)
- n1 [entite] « L' émotion liée au traumatisme »
- n2 [processus] « le traitement des éléments émotionnels du souvenir . »
- `['n1']` —`prevent`→ `n2` conf=1.0 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12509 (visée `prevent`)
- n1 [etat_local] « Dans son comportement , la personne va avoir tendance , »
- n2 [processus] « qu' elle en soit consciente , à choisir des situations et des relations qui vont maintenir »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [12, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12517 (visée `prevent`)
- n1 [processus] « La partie adulte , n' étant pas capable de gérer les parties émotionnelles générées dans l »
- n2 [entite] « contre-attaque du SPI) . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12650 (visée `prevent`)
- n1 [etat_local] « De manière classique , ce modèle d' évaluation dure généralement de cinq à dix séances en  »
- n2 [entite] « contre-attaque qui freine la reconnaissance des SPI) . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 34]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12713 (visée `prevent`)
- n1 [processus] « Le non dit ressenti est interprété comme un contenu inavouable et donc probablement »
- n2 [entite] « ou condamnable . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15006 (visée `prevent`)
- n1 [processus] « L' extrême diversité du vivant n' empêche en »
- n2 [processus] « le groupement en entités ou taxons (taxonomie) , leurs relations les uns par rapport aux a »
- `['n1']` —`prevent`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [10, 30]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15861 (visée `prevent`)
- n1 [processus] « Le cœur peut inclure des régions solides et liquides , et dans certains cas , il peut géné »
- n2 [processus] « la planète et son atmosphère des attaques du vent solaire[17] . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 24]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-20922 (visée `prevent`)
- n1 [processus] « Les progrès technologiques , notamment le développement de souris transgéniques exprimant  »
- n2 [entite] « les agents infectieux . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 48]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-21027 (visée `prevent`)
- n1 [processus] « De nouvelles techniques d' encapsulation des cellules d' îlots pancréatiques dans un revêt »
- n2 [processus] « les cellules transplantées . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 26]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-22789 (visée `prevent`)
- n1 [processus] « Lutter »
- n2 [processus] « ces cellules immunitaires de suppression (tueuses de lymphocytes T) dans le foie lors du d »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [3, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-22850 (visée `prevent`)
- n1 [processus] « Ce lait prodigue des anticorps aux chatons , ce qui les »
- n2 [entite] « des maladies infectieuses . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27859 (visée `prevent`)
- n1 [processus] « La première a été pour rendre plus difficile l' abord des Isles ailleurs que dans les Rade »
- n2 [entite] « les descentes . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 42]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27861 (visée `prevent`)
- n1 [processus] « En troisième lieu cette réserve est faite afin que chacun ait un passage libre au long de  »
- n2 [processus] « cela les Habitants l' auroient empêché par des clôtures , et par des oppositions qui tous  »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]; span n2 >15tok [23, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-28092 (visée `prevent`)
- n1 [processus] « Liberté de circulation Les autorités peuvent prendre des mesures de couvre feu en vue d' i »
- n2 [processus] « un risque de dommage collatéral ou de causer un préjudice à une personne innocente dans la »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 54]; span n2 >15tok [56, 75]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-4713 (visée `filter`)
- n1 [processus] « Le Nigeria avait déjà agacé Paris en étant l' un des seuls pays , avec le Maroc , à protes »
- n2 [processus] « le troisième essai nucléaire français , Gerboise rouge , effectué le 27 décembre 1960 à Re »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 21]; span n2 >15tok [23, 43]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11047 (visée `filter`)
- n1 [processus] « Dans des parades avec des individus d' une autre ethnie , seuls 45% ont été correctement i »
- n2 [processus] « 60% pour les identifications de même ethnie . »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 17]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-16513 (visée `filter`)
- n1 [processus] « La médecine libérale devrait perdre 17 % de ses effectifs , et le secteur salarié 8 % , sa »
- n2 [entite] « 7 % pour les spécialistes (ophtalmologistes , oto-rhino-laryngologistes et psychiatres sur »
- `['n1']` —`prevent`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 39]
- [ ] oui  [ ] non  [ ] corriger : ___
