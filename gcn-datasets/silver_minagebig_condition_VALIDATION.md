# Validation minage-big — `condition` (7 cas, PROPOSITION)

Source : g0 mot-clé → silver B → build_lot (lattice + balanced_auto + normalize D5).
LITIG = spans >15tok → re-découpe manuelle (guide). Cocher UNE case.

### g0-11886 (visée `prevent`)
- n1 [processus] « individu n' évite pas nécessairement le sommeil , mais continue plutôt à accomplir des act »
- n2 [processus] « La procrastination du sommeil se produit généralement »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [10, 47]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-1399 (visée `filter`)
- n1 [processus] « 8 mois sont supérieurs à 10 °C de moyenne . »
- n2 [etat_local] « Outre la classification initiale basique de Köppen , Trewartha a proposé d' affiner celle  »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 28]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10087 (visée `filter`)
- n1 [processus] « lorsqu' ils étaient seuls que lorsqu' ils étaient avec d' autres personnes . »
- n2 [processus] « En revanche , les réponses des Japonais variaient considérablement en fonction de la situa »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n2 >15tok [1, 23]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-12660 (visée `filter`)
- n1 [processus] « les stratégies dysfonctionnelles sont modifiées que la thérapie sera complète . »
- n2 [processus] « C' est uniquement »
- `['n1']` —`condition`→ `n2` conf=1.0 explicit=True
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-13005 (visée `filter`)
- n1 [processus] « l' esprit humain souffre d' un biais perceptif fondamentalement ancré : celui de percevoir »
- n2 [processus] « L' équivocité de la préposition pour constitue un obstacle épistémologique puissant à la c »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [33, 73]; span n2 >15tok [1, 31]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-9428 (visée `control_dependency`)
- n1 [processus] « Guillaume le Conquérant ordonne en 1086 l' organisation d' un recensement sur son territoi »
- n2 [processus] « Les origines de l' enquête statistique remontent au XIe siècle , »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [13, 35]
- [ ] oui  [ ] non  [ ] corriger : ___

### g0-10680 (visée `control_dependency`)
- n1 [condition] « le sujet change d' environnement , il est lié à la peur et à la familiarité ; si la peur d »
- n2 [processus] « Le comportement exploratoire apparaît »
- `['n1']` —`condition`→ `n2` conf=0.9 explicit=True
- LITIG : span n1 >15tok [6, 73]
- [ ] oui  [ ] non  [ ] corriger : ___
