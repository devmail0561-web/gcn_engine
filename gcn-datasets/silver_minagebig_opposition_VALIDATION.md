# Validation minage-big — `opposition` (4 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-27918 (visée `prevent`)
- n1 [processus] « rappelle les règles régissant les prêts de la banque centrale à la puissance publique qui  »
- n2 [processus] « Elle n' interdit pas le financement de l' État par la banque centrale , »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n1 >15tok [16, 36]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-2424 (visée `filter`)
- n1 [processus] « lieu de s' affaiblir avec les changements climatiques , selon un climatologue de l' univer »
- n2 [processus] « C' est l' un des seuls systèmes atmosphériques de la planète qui se renforce »
- `['n1']` —`opposition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [16, 51]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-11412 (visée `data_dependency`)
- n1 [processus] « que la personne visée est décrite comme agresseur ou comme responsable du problème . »
- n2 [processus] « Elle repose sur un renversement logique construit : l' accusateur se présente comme victim »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-15876 (visée `data_dependency`)
- n1 [processus] « que les étoiles plus légères forment des nébuleuses planétaires et évoluent en naines blan »
- n2 [processus] « Le destin final de l' étoile dépend de sa masse: les étoiles qui sont plus de 8 fois plus  »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n2 >15tok [1, 29]
- [ ] oui  [ ] non  [ ] corriger : ___
