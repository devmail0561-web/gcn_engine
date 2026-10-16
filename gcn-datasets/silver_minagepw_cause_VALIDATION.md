# Validation minage-pw — `cause` (4 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-1024 (visée `prevent`)
- n1 [entite] « On s' interdit »
- n2 [processus] « conséquent qu' une couche invoque les services d' une couche plus basse que la couche immé »
- `['n1']` —`cause`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [5, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1050 (visée `prevent`)
- n1 [processus] « Étant »
- n2 [processus] « donné qu' aucun vaccin n' est efficace à 100 % , que Gardasil ne protège pas contre les ty »
- `['n1']` —`cause`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +1; span n2 >15tok [2, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-572 (visée `data_dependency`)
- n1 [processus] « En gestion de projet , l' obtention d' un résultat dépend de la combinaison entre trois pa »
- n2 [processus] « conséquent le volume d' effort à produire . »
- `['n1']` —`cause`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [1, 34]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1025 (visée `data_dependency`)
- n1 [entite] « étant »
- n2 [processus] « donné que Proxmox VE repose sur une distribution Debian , il est tout à fait possible de l »
- `['n1']` —`cause`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +2; span n2 >15tok [4, 38]
- [ ] oui  [ ] non  [ ] corriger : ___
