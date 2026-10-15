# Validation minage-big — `sequence` (3 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-853 (visée `filter`)
- n1 [etat_local] « d' être mesurées . »
- n2 [processus] « Sauf indication contraire , toutes les quantités de précipitations indiquées dans cet arti »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 29]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-3514 (visée `filter`)
- n1 [processus] « le 27 février 1823 (date de sa condamnation) , et non lui même , figuraient sur cette list »
- n2 [entite] « Seuls les enfants de FD Changuion nés »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [9, 27]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-5814 (visée `filter`)
- n1 [entite] « empereur) et règne en souverain sur la cité — lui et ses futurs successeurs avec l' empere »
- n2 [processus] « Il est nommé prince de l' Empire allemand (rang maximum »
- `['n1']` —`sequence`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [12, 35]
- [ ] oui  [ ] non  [ ] corriger : ___
