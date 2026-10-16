# Validation minage-pw — `sequence` (4 cas, PROPOSITION)

Source : presse+wiki FR mot-clé → silver B → build_lot. LITIG = spans >15tok (guide).

### pw-641 (visée `prevent`)
- n1 [processus] « le début de la tournée . »
- n2 [processus] « De ce fait , de nombreux commentateurs en Nouvelle-Zélande et au Royaume Uni considèrent c »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 46]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-819 (visée `prevent`)
- n1 [processus] « Trouble panique , avec ou sans agoraphobie Il est normal de vivre un sentiment de peur ou  »
- n2 [etat_local] « avoir évité un accident de voiture . »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [1, 30]
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-100 (visée `filter`)
- n1 [processus] « elle , seuls quatre Français ont réalisé cet exploit . »
- n2 [processus] « Une sortie extravéhiculaire qui la fera entrer dans un cercle très fermé : »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### pw-490 (visée `filter`)
- n1 [processus] « Seuls leur groupement , leur succession , une observation minutieuse ont pu , »
- n2 [processus] « de longs tâtonnements , permettre d' établir des tableaux symptomatiques particuliers et l »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [15, 31]
- [ ] oui  [ ] non  [ ] corriger : ___
