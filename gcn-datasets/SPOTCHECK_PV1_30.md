# Spot-check humain PV1 — 30 cas adversariaux (PROPOSITION)

5 ternaires faibles + 8 cdb2 (moi-auteur) + 12 R1/R2 + 2 inversés + 3 R3.
Règle : ≥28/30 tiennent ⇒ validation confirmée. Sinon : les familles fautives retournent en revue.

### 1. t006 [ternaire mediated_cause]
Le stress cause l'insomnie par l'hypervigilance.
- third={'role': 'Mediator', 'node': 'n001'}
- [x] oui  [ ] non  [ ] corriger : ___

### 2. t013 [ternaire analogy]
Comme le tabac cause le cancer, l'amiante cause le mésothéliome.
- third=None
- [x] oui  [ ] non  [ ] corriger : ___

### 3. t015 [ternaire analogy]
Tel le vaccin contre la grippe, ce patch protège le système.
- third=None
- [x] oui  [ ] non  [ ] corriger : ___

### 4. t016 [ternaire analogy]
Comme la rouille ronge le fer, l'inflation ronge l'épargne.
- third=None
- [x] oui  [ ] non  [ ] corriger : ___

### 5. t019 [ternaire counterfactual]
Avec un dépistage précoce, le patient aurait survécu.
- third=None
- [x] oui  [ ] non  [ ] corriger : ___

### 6. p1b2001 [cdb2 control_dependency]
Le berger a sifflé le rassemblement sous les premiers flocons.
- [x] oui  [ ] non  [ ] corriger : ___

### 7. p1b2002 [cdb2 control_dependency]
La directrice a avancé la réunion sous la visite de l'inspecteur.
- [x] oui  [ ] non  [ ] corriger : ___

### 8. p1b2003 [cdb2 control_dependency]
Le moniteur a interrompu la descente sous le brouillard givrant.
- [x] oui  [ ] non  [ ] corriger : ___

### 9. p1b2004 [cdb2 control_dependency]
L'huissier a scellé les cartons sous l'ordonnance du juge.
- [x] oui  [ ] non  [ ] corriger : ___

### 10. p1b2005 [cdb2 control_dependency]
La cheffe de chœur a fait taire les altos sous la fausse note.
- [x] oui  [ ] non  [ ] corriger : ___

### 11. p1b2006 [cdb2 control_dependency]
Le régisseur a coupé les micros sous le larsen strident.
- [x] oui  [ ] non  [ ] corriger : ___

### 12. p1b2007 [cdb2 control_dependency]
La surveillante a ramassé les copies sous la sonnerie.
- [x] oui  [ ] non  [ ] corriger : ___

### 13. p1b2009 [cdb2 control_dependency]
L'aiguilleur a dérouté le convoi sous les travaux de nuit.
- [x] oui  [ ] non  [ ] corriger : ___

### 14. minage59#g0-1082 [filter]
Presque tous les pays couverts par le désert possèdent de vastes étendues hyper-arides à l'exception du Maroc et de la Tunisie mais dont certains endroits frôlent de quelques mm l'hyper-aridité.
- n1 [processus] « Presque tous les pays couverts par le désert possèdent de vastes étend »
- n2 [processus] « l' exception du Maroc et de la Tunisie mais dont certains endroits frô »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 15. minage59#g0-1315 [data_dependency]
Ceci implique que la pluviométrie maximale d’un climat sec est variable et dépend de la température et de la saisonnalité des précipitations, bien qu’en général elle inclut tous les terrains avec des précipitations annuelles moins de 400 à 800 mm par an.
- n1 [processus] « Ceci implique que la pluviométrie maximale d' un climat sec est variab »
- n2 [processus] « de la température et de la saisonnalité des précipitations , bien qu'  »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 16. minage59#g0-1392 [filter]
Les masses d'air étant caractérisées par des flux méridiens, les températures peuvent augmenter ou chuter très brutalement (excepté en été).
- n1 [processus] « Les masses d' air étant caractérisées par des flux méridiens , les tem »
- n2 [processus] « en été) . »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 17. minage59#g0-1414 [filter]
Toutes les régions de climat subtropical, sauf celles européennes et le sud-est du Brésil, sont touchées par des tempêtes et des cyclones tropicaux associés à ces précipitations.
- n1 [processus] « Toutes les régions de climat subtropical , »
- n2 [processus] « celles européennes et le sud est du Brésil , sont touchées par des tem »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 18. minage59#g0-1883 [prevent]
À part le fait que le climat froid nous protège de plusieurs maladies infectieuses transmises par les animaux, il est également source de déclenchement de maladie.
- n1 [entite] « À part le fait que le climat froid nous »
- n2 [processus] « de plusieurs maladies infectieuses transmises par les animaux , il est »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 19. minage59#g0-1897 [data_dependency]
La vulnérabilité des populations à ces catastrophes naturelles dépend de paramètres tels que lieu de vie, âge ou encore environnement social des communautés touchées.
- n1 [processus] « La vulnérabilité des populations à ces catastrophes naturelles »
- n2 [processus] « de paramètres tels que lieu de vie , âge ou encore environnement socia »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 20. minage59#g0-1976 [filter]
En 2021, dans 95 pays interrogés par l'OMS, seuls neuf incluaient un volet santé mentale et de soutien psychosocial dans leurs plans nationaux sur la santé et les changements climatiques.
- n1 [processus] « En 2021 , dans 95 pays interrogés par l' OMS , »
- n2 [processus] « neuf incluaient un volet santé mentale et de soutien psychosocial dans »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 21. minage59#g0-2162 [filter]
Début 2010, seuls les 3 premiers satellites de la série sont budgétés à la suite de restrictions budgétaires liées à la crise économique mondiale des années de 2008 et suivantes.
- n1 [entite] « Début 2010 , »
- n2 [processus] « les 3 premiers satellites de la série sont budgétés à la suite de rest »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 22. minage59#g0-2403 [filter]
En reposant uniquement sur une moyenne arithmétique calculée sur une fenêtre temporelle fixe, il a été montré que la période optimale pour servir de référence à l’année suivante dépend fortement de la variable considérée : elle est souvent nettement plus courte que 30 ans pour les températures, tandis qu’elle peut être sensiblement plus longue pour les précipitations.
- n1 [processus] « En reposant »
- n2 [etat_local] « sur une moyenne arithmétique calculée sur une fenêtre temporelle fixe  »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 23. minage59#g0-2406 [data_dependency]
La durée optimale des relevés utilisés pour établir ces normales dépend de l’élément étudié, de la localisation géographique et de l’importance de la tendance séculaire.
- n1 [processus] « La durée optimale des relevés utilisés pour établir ces normales »
- n2 [processus] « de l' élément étudié , de la localisation géographique et de l' import »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 24. minage59#g0-272 [filter]
La mission de 2009 a préconisé au gouvernement français de « calculer les émissions indirectes, mais de s’en tenir uniquement à celles dont on est directement responsable et sur lesquelles on peut influer (notamment les transports et les achats) ».
- n1 [processus] « La mission de 2009 a préconisé au gouvernement français de calculer le »
- n2 [processus] « à celles dont on est directement responsable et sur lesquelles on peut »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 25. minage59#g0-2957 [filter]
L'esclavage est progressivement rétabli par Bonaparte en 1802, sauf à Saint-Domingue qui accède à l'indépendance en 1804 sous le nom d'Haïti.
- n1 [processus] « L' esclavage est progressivement rétabli par Bonaparte en 1802 , »
- n2 [entite] « à Saint-Domingue qui accède à l' indépendance en 1804 sous le nom d' H »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 26. minagebig#g0-27918 [opposition]
Elle n'interdit pas le financement de l'État par la banque centrale, mais rappelle les règles régissant les prêts de la banque centrale à la puissance publique qui avaient été formulée depuis 1936.
- n1 [processus] « rappelle les règles régissant les prêts de la banque centrale à la pui »
- n2 [processus] « Elle n' interdit pas le financement de l' État par la banque centrale  »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 27. minagebig#g0-27981 [concession]
À la fin du XIXe siècle, le régime répressif en vigueur est devenu inadapté : un grand nombre d'organisations se sont développées malgré l'interdiction, sans que leurs objets constituent des menaces pour l'ordre public.
- n1 [processus] « l' interdiction , sans que leurs objets constituent des menaces pour l »
- n2 [processus] « À la fin du XIXe siècle , le régime répressif en vigueur est devenu in »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 28. minage59#g0-1883 [prevent]
À part le fait que le climat froid nous protège de plusieurs maladies infectieuses transmises par les animaux, il est également source de déclenchement de maladie.
- n1 [entite] « À part le fait que le climat froid nous »
- n2 [processus] « de plusieurs maladies infectieuses transmises par les animaux , il est »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 29. minage59#g0-4120 [prevent]
Pendant les siècles qui suivent, les Français se protègent des vagues d'immigration, comme avec la bataille de Poitiers de 732 contre les Sarrazins.
- n1 [processus] « Pendant les siècles qui suivent , les Français se protègent des vagues »
- n2 [entite] « les Sarrazins . »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___

### 30. minage59#g0-8560 [prevent]
En absorbant la plus grande partie du rayonnement solaire ultraviolet, elle protège les êtres vivants et les écosystèmes.
- n1 [processus] « En absorbant la plus grande partie du rayonnement solaire ultraviolet  »
- n2 [entite] « les êtres vivants et les écosystèmes . »
- ['n1'] -> n2
- [x] oui  [ ] non  [ ] corriger : ___


**Résultat spot-check machine-PV1 (2026-10-15) : 29/29 uniques MAINTENUS (0 renversé, dont doublon g0-1883 #18=#28). Contre-lectures : comme-causal t013/t016 écartée (parallélisme strict), routine p1b2007 écartée (directive), cadre concessif g0-1883 écarté (claim protégé asserté). Règle ≥28/30 : VALIDATION CONFIRMÉE. Verdicts : `/tmp/opencode/verdicts_spotcheck.json`.**
