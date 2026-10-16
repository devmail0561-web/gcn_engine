# Validation minage-big — `cause` (4 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-27995 (visée `prevent`)
- n1 [processus] « autorisant sous certaines conditions des dérogations pour les insecticides néonicotinoïdes »
- n2 [processus] « Elle reprend plusieurs revendications de la Fédération nationale des syndicats d' exploita »
- `['n1']` —`cause`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [17, 41]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15319 (visée `filter`)
- n1 [etat_global] « Un logiciel applicatif effectue typiquement un grand nombre d' appels système , et »
- n2 [etat_global] « conséquent , il peut fonctionner uniquement avec un système d' exploitation dont l' interf »
- `['n1']` —`cause`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [15, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-25818 (visée `control_dependency`)
- n1 [etat_local] « Dans le cas de l' être l' humain Histoire de l' avortement dans le monde Différentes situa »
- n2 [processus] « avortement provoqué ou déclenché l' interruption volontaire d' une grossesse chez une femm »
- `['n1']` —`cause`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 33]; span n2 >15tok [35, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-27955 (visée `control_dependency`)
- n1 [processus] « cause de sa préparation fortement influencée par les puissances alliées et de son approbat »
- n2 [processus] « La constitution issue de ces travaux était considérée comme un texte provisoire d' une par »
- `['n1']` —`cause`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [17, 72]
- [ ] oui  [ ] non  [ ] corriger : ___
