# Validation minage-big — `control_dependency` (17 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

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

### g0-21160 (visée `control_dependency`)
- n1 [etat_global] « Histoire d' ISO/CEI 9995 Des origines à 1994 Le coup d' envoi La mise en place du projet d »
- n2 [etat_local] « menée par le même expert , qui répondait aux préconisations du Pr Alain Souloumiac émises  »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 37]; span n2 >15tok [39, 105]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-23034 (visée `control_dependency`)
- n1 [processus] « En réponse à la liaison à son récepteur RANK , une voie de signalisation en cascade est dé »
- n2 [processus] « le recrutement de TRAF6 dans le domaine intracellulaire de RANK et aboutit à l' activation »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 20]; span n2 >15tok [22, 90]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-24919 (visée `control_dependency`)
- n1 [entite] « Chronologie Crise des subprimes La crise des subprimes se »
- n2 [processus] « au deuxième semestre 2006 avec le krach des prêts immobiliers (hypothécaires) à risque aux »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [11, 45]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27333 (visée `control_dependency`)
- n1 [etat_global] « Juge aiguilleur Selon Kelsen , le juge constitutionnel n' a pas pour vocation d' instaurer »
- n2 [processus] « de constitutionnalité aurait des aspirations politiques . »
- `['n1']` —`control_dependency`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 28]
- [ ] oui  [ ] non  [ ] corriger : ___
