# Proposition silver ternaire — 20 phrases (NON gold)

Statut : proposition LLM (assistant) en attente de validation humaine.
Poids : 0.0 (R9 — jamais gold direct, jamais dans le train avant validation).
Gate de promotion : erreur ≤20% borne haute IC95 sur échantillon (`GATES:15`).

Format : texte + clauses + arête (relation 19 + rôle ternaire).
Conversion JSON gold (`tok_loader` + `third`/`joint_group`) après validation.

## ConditionalCause / ConditionalPrevent

1. {"text": "Si le pare-feu est actif, l'attaque échoue.", "clauses": ["le pare-feu est actif", "l'attaque échoue"], "edge": {"rel": "conditional_cause", "third": {"role": "Condition", "clause": 0}}}
2. {"text": "Le traitement guérit si le diagnostic est précoce.", "clauses": ["Le traitement guérit", "le diagnostic est précoce"], "edge": {"rel": "conditional_cause", "third": {"role": "Condition", "clause": 1}}}
3. {"text": "Sans sauvegarde, la panne efface les données.", "clauses": ["la panne efface les données", "sans sauvegarde"], "edge": {"rel": "conditional_prevent", "third": {"role": "Condition", "clause": 1}}}
4. {"text": "Le vaccin protège, à condition d'être conservé au froid.", "clauses": ["Le vaccin protège", "conservé au froid"], "edge": {"rel": "conditional_cause", "third": {"role": "Condition", "clause": 1}}}

## MediatedCause / MediatedPrevent

5. {"text": "La pluie provoque des inondations via la saturation des sols.", "clauses": ["La pluie", "sols saturés", "inondations"], "edge": {"rel": "mediated_cause", "third": {"role": "Mediator", "clause": 1}}}
6. {"text": "Le stress cause l'insomnie par l'hypervigilance.", "clauses": ["Le stress", "hypervigilance", "insomnie"], "edge": {"rel": "mediated_cause", "third": {"role": "Mediator", "clause": 1}}}
7. {"text": "Le couvre-feu a stoppé l'émeute grâce à l'intervention policière.", "clauses": ["Le couvre-feu", "intervention policière", "émeute stoppée"], "edge": {"rel": "mediated_prevent", "third": {"role": "Mediator", "clause": 1}}}
8. {"text": "L'antibiotique élimine l'infection en détruisant la paroi bactérienne.", "clauses": ["L'antibiotique", "paroi détruite", "infection éliminée"], "edge": {"rel": "mediated_cause", "third": {"role": "Mediator", "clause": 1}}}

## JointCause / JointPrevent

9. {"text": "La sécheresse et la canicule ont provoqué l'incendie.", "clauses": ["La sécheresse", "la canicule", "incendie"], "edge": {"rel": "joint_cause", "sources": [0, 1], "target": 2}}
10. {"text": "Le masque et la distance réduisent la contagion.", "clauses": ["Le masque", "la distance", "contagion réduite"], "edge": {"rel": "joint_prevent", "sources": [0, 1], "target": 2}}
11. {"text": "La formation et l'outillage augmentent la productivité.", "clauses": ["La formation", "l'outillage", "productivité"], "edge": {"rel": "joint_cause", "sources": [0, 1], "target": 2}}
12. {"text": "Le mot de passe et la biométrie bloquent l'intrusion.", "clauses": ["Le mot de passe", "la biométrie", "intrusion bloquée"], "edge": {"rel": "joint_prevent", "sources": [0, 1], "target": 2}}

## Analogy

13. {"text": "Comme le tabac cause le cancer, l'amiante cause le mésothéliome.", "clauses": ["tabac cause cancer", "amiante cause mésothéliome"], "edge": {"rel": "analogy"}}
14. {"text": "De même que la digue retient l'eau, le pare-feu retient les attaques.", "clauses": ["digue retient eau", "pare-feu retient attaques"], "edge": {"rel": "analogy"}}
15. {"text": "Tel le vaccin contre la grippe, ce patch protège le système.", "clauses": ["vaccin contre grippe", "patch protège système"], "edge": {"rel": "analogy"}}
16. {"text": "Comme la rouille ronge le fer, l'inflation ronge l'épargne.", "clauses": ["rouille ronge fer", "inflation ronge épargne"], "edge": {"rel": "analogy"}}

## Counterfactual

17. {"text": "Sans la ceinture, le choc aurait été mortel.", "clauses": ["sans la ceinture", "choc mortel hypothétique"], "edge": {"rel": "counterfactual"}}
18. {"text": "Si l'alerte avait sonné, l'évacuation aurait réussi.", "clauses": ["alerte non sonnée", "évacuation ratée"], "edge": {"rel": "counterfactual"}}
19. {"text": "Avec un dépistage précoce, le patient aurait survécu.", "clauses": ["pas de dépistage", "décès"], "edge": {"rel": "counterfactual"}}
20. {"text": "Sans la grève, les trains seraient arrivés à l'heure.", "clauses": ["grève", "retard"], "edge": {"rel": "counterfactual"}}
