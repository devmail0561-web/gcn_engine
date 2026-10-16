# Validation minage-pw — `concession` (5 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-210 (visée `prevent`)
- n1 [processus] « Bien »
- n2 [processus] « que l' ozone stratosphérique nous protège des rayons UV nocifs , l' ozone troposphérique e »
- `['n1']` —`concession`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [2, 21]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-335 (visée `prevent`)
- n1 [processus] « Bien »
- n2 [processus] « que l' ozone stratosphérique nous protège des rayons UV nocifs , l' ozone troposphérique e »
- `['n1']` —`concession`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [2, 21]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-895 (visée `prevent`)
- n1 [processus] « si ce pesticide a été interdit en 1989 . »
- n2 [processus] « L' Inde a utilisé plus de 350 milliards de tonnes de DDT depuis 1985 , »
- `['n1']` —`concession`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-981 (visée `prevent`)
- n1 [processus] « éviter les excès par rapport aux besoins , sinon , il se produit une augmentation des reje »
- n2 [processus] « Il faut »
- `['n1']` —`concession`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [4, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-70 (visée `filter`)
- n1 [processus] « si nous l' avons pris avec des pincettes , puisque l' étude inclut les femmes à partir de  »
- n2 [processus] « Le chiffre est extrêmement faible - seulement 2,5 % de femmes traitées - »
- `['n1']` —`concession`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [15, 46]
- [ ] oui  [ ] non  [ ] corriger : ___
