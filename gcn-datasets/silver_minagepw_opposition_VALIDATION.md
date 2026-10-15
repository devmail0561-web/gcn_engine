# Validation minage-pw — `opposition` (6 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-14 (visée `prevent`)
- n1 [processus] « que la dette française inquiète les marchés . »
- n2 [processus] « Je préférerais qu' on les évite , parce que ça veut dire qu' on n' a pas vraiment capacité »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : recalage strip +1; conf=0.8 explicit=True; span n2 >15tok [1, 56]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-445 (visée `prevent`)
- n1 [entite] « son empreinte . »
- n2 [processus] « Par contre , pour plus de commodité et pour éviter les mauvais rares cas mathématiques , o »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n2 >15tok [1, 24]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1038 (visée `prevent`)
- n1 [processus] « que le protestantisme restait interdit en France , depuis la révocation de l' édit de Nant »
- n2 [processus] « Le 30 avril 1715 à Bordeaux , il épouse Jeanne de Lartigue , une protestante issue d' une  »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n1 >15tok [38, 56]; span n2 >15tok [1, 36]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1065 (visée `prevent`)
- n1 [processus] « que plus de 150 navires jettent l' ancre à l' extérieur du détroit afin d' éviter les risq »
- n2 [processus] « Les avertissements et les attaques qui suivent contre des navires entraînent une forte bai »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n1 >15tok [32, 50]; span n2 >15tok [1, 30]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-546 (visée `filter`)
- n1 [processus] « que les tableaux observés sont très disparates . »
- n2 [processus] « La Classification internationale des maladies CIM 10 et la classification américaine (DSM  »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n2 >15tok [1, 37]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-581 (visée `data_dependency`)
- n1 [processus] « que la connaissance ou la justification a posteriori dépend de l' expérience ou de preuves »
- n2 [processus] « La connaissance ou justification a priori est indépendante de l' expérience (par exemple T »
- `['n1']` —`opposition`→ `n2` conf=0.8 explicit=True
- LITIG : conf=0.8 explicit=True; span n1 >15tok [24, 48]; span n2 >15tok [1, 22]
- [ ] oui  [ ] non  [ ] corriger : ___
