# Validation minage-pw — `condition` (14 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-75 (visée `prevent`)
- n1 [processus] « compliquées . »
- n2 [condition] « Pour franchir le cap des 100 ans sans problème majeur , voire accéder au rang envié des su »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [1, 68]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-229 (visée `prevent`)
- n1 [processus] « ces dernières sont stimulées par la sphingosine-1-phosphate (S1P) , une molécule relâchée  »
- n2 [entite] « De plus , le TFPI bloque la migration des cellules endothéliales »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [13, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-354 (visée `prevent`)
- n1 [processus] « ces dernières sont stimulées par la sphingosine-1-phosphate (S1P) , une molécule relâchée  »
- n2 [entite] « De plus , le TFPI bloque la migration des cellules endothéliales »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [13, 32]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-717 (visée `prevent`)
- n1 [processus] « Si »
- n2 [processus] « son accès demeure théoriquement interdit , la Petite Ceinture est particulièrement prisée  »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [2, 19]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-827 (visée `prevent`)
- n1 [processus] « Si »
- n2 [processus] « la personne souffrant du syndrome évite de montrer sa confiance en soi , personne ne peut  »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n2 >15tok [2, 38]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1229 (visée `prevent`)
- n1 [etat_local] « chaque PC se bloque une fois tous les 10 ans , le redémarrage peut être moins dérangeant q »
- n2 [entite] « Par exemple , »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [5, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1284 (visée `prevent`)
- n1 [processus] « le pays est en effet créditeur du monde , il prête et détient des actifs dans sa propre mo »
- n2 [entite] « Le Royaume Uni évite de vivre une telle situation à la fin du siècle , car »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [18, 41]; span n2 >15tok [1, 16]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-84 (visée `filter`)
- n1 [processus] « elle devient trop exigeante pour un organisme en pleine croissance . »
- n2 [processus] « L' activité physique est indispensable… sauf »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-185 (visée `filter`)
- n1 [etat_local] « le directeur des services de greffe judiciaires , ce qui suppose leur accord , soit l' un  »
- n2 [processus] « [20] Soit les parents établissent volontairement une déclaration conjointe d' exercice en  »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +2; span n1 >15tok [20, 72]; span n2 >15tok [2, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-310 (visée `filter`)
- n1 [etat_local] « le directeur des services de greffe judiciaires , ce qui suppose leur accord , soit l' un  »
- n2 [processus] « [20] Soit les parents établissent volontairement une déclaration conjointe d' exercice en  »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : recalage strip +2; span n1 >15tok [20, 72]; span n2 >15tok [2, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-502 (visée `filter`)
- n1 [processus] « l' utérus a été retiré , afin de réduire le risque accru de cancer de l' endomètre . »
- n2 [processus] « Un progestatif est utilisé , sauf »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [8, 25]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-612 (visée `control_dependency`)
- n1 [processus] « Delphine est chez elle . »
- n2 [processus] « Son téléphone sonne pour la dernière fois à 22 h 55 (puis passera sur messagerie à 7 h 48) »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 39]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-1022 (visée `control_dependency`)
- n1 [processus] « sous contrôle et de le faire évoluer en fonction de la stratégie de l' organisation . »
- n2 [processus] « Les méthodes ITIL (IT infrastructure library) et COBIT sont par exemple , des supports per »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- LITIG : span n1 >15tok [20, 35]; span n2 >15tok [1, 18]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-919 (visée `data_dependency`)
- n1 [processus] « ce dernier dépend de variables latentes non observables . »
- n2 [processus] « L' algorithme espérance-maximisation (en anglais expectation-maximization algorithm , souv »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 29]
- [ ] oui  [ ] non  [ ] corriger : ___
