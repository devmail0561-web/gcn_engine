# Validation minage-big — `motivation` (2 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-19449 (visée `prevent`)
- n1 [processus] « Sinon , les ustensiles de cuisine en cuivre sont doublés d' un métal non réactif »
- n2 [processus] « empêcher le contact entre les aliments acides et l' élément structurel en cuivre de l' ust »
- `['n1']` —`motivation`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [17, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-21559 (visée `prevent`)
- n1 [processus] « La vaccination contre la rougeole »
- n2 [processus] « pour objectif d' éviter les complications graves , de prévenir un affaiblissement immunita »
- `['n1']` —`motivation`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [7, 36]
- [ ] oui  [ ] non  [ ] corriger : ___
