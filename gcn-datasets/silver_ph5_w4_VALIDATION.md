# Ph5 vague 4 — validation PV1 (pipeline B-c1 deux passages, 150 phrases fraîches)

Juger CHAQUE paire : [oui]/[non]+motif/[corriger→X]. Focus : causales en tête extraites ?

## v4-3001 silver-B
Dix-huit ans après avoir été créé, le régime des microentrepreneurs affiche un bilan très mitigé.
- n0 « Dix-huit ans »
- n1 « après avoir été créé, »
- n2 « le régime des microentrepreneurs affiche un bilan très mitigé. »
- e0: n0 --restrictive--> n1 (conf 0.32, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.54, marge 0.54)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.68, marge 1.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3002 silver-B
Survenant d’abord une fois par mois, ces épisodes sont devenus quasi quotidiens au cours des deux ou trois dernières années.
- n0 « Survenant d’abord »
- n1 « une fois par mois, »
- n2 « ces épisodes sont devenus quasi quotidiens au cours des deux ou trois dernières années. »
- e0: n0 --systemique--> n1 (conf 0.37, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.44, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.45, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3006 silver-B
Le lac Annecy affecté par la sécheresse après un été particulièrement sec, fin septembre 2026.
- n0 « Le lac Annecy affecté par la sécheresse »
- n1 « après un été particulièrement sec, »
- n2 « fin septembre 2026. »
- e0: n0 --restrictive--> n1 (conf 0.62, marge 0.76)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.40, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.31, marge 0.14)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3007 silver-B
Pour 2027, le projet de loi de finances proposé aux parlementaires paraît minimaliste.
- n0 « Pour 2027, »
- n1 « le projet de loi de finances proposé aux parlementaires paraît minimaliste. »
- e0: n0 --restrictive--> n1 (conf 0.45, marge 0.30)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3008 silver-B
RENCONTRE - Depuis deux ans, ancien d’Apple devenu grand-père, Christophe ne s’entraîne plus «en aveugle».
- n0 « RENCONTRE - Depuis deux ans, ancien d’Apple devenu grand-père, Christophe ne s’ »
- n1 « entraîne plus «en aveugle». »
- e0: n0 --causale--> n1 (conf 0.73, marge 1.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3010 silver-B
Regardez comment ils vantent sans cesse la préservation de la bio diversité tout en détruisant systématiquement les populations de cerfs des forêts domaniales.
- n0 « Regardez comment ils vantent »
- n1 « sans cesse la préservation de la bio diversité tout en détruisant systématiquement les pop »
- e0: n0 --restrictive--> n1 (conf 0.49, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3011 silver-B
- [4] L’impossibilité résulte selon King du caractère collectif des décisions.
- n0 « - [4] L’impossibilité »
- n1 « résulte selon King du caractère collectif des décisions. »
- e0: n0 --negative--> n1 (conf 0.47, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3013 silver-B
L’eau d’une première retenue vient faire tourner les turbines de la centrale électrique avant d’être stockée dans un deuxième lac en contrebas, en attendant d’être remontée par pompage et ainsi de suite.
- n0 « L’eau d’une première retenue vient faire tourner les turbines de la centrale électrique »
- n1 « avant d’être stockée dans un deuxième lac en contrebas, »
- n2 « en attendant d’être remontée par pompage et »
- n3 « ainsi de suite. »
- e0: n0 --negative--> n1 (conf 0.37, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.42, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.37, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.54, marge 0.41)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.58, marge 0.83)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3018 silver-B
L’homme, grâce à la révolution industrielle et au progrès technologique, a pu développer son emprise sur l’environnement.
- n0 « L’homme, »
- n1 « grâce à la révolution industrielle et au progrès technologique, »
- n2 « a pu développer son emprise sur l’environnement. »
- e0: n0 --sequentielle--> n1 (conf 0.39, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.32, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.40, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3020 silver-B
Malgré plusieurs lacunes, les souris traitées avec la bléomycine ou portant des modifications génétiques (TSK-1) sont très utilisées dans les études précliniques de la SSc mais elles ne présentent pas toutes les caractéristiques de cette maladie.
- n0 « Malgré plusieurs lacunes, »
- n1 « les souris traitées avec la bléomycine ou portant des modifications génétiques (TSK-1) son »
- n2 « mais elles ne présentent pas toutes les caractéristiques de cette maladie. »
- e0: n0 --systemique--> n1 (conf 0.43, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.38, marge 0.15)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.50, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3021 silver-B
« On est dans une France aujourd’hui extrêmement fragmentée, et si on veut la faire avancer il va falloir faire preuve d’esprit de coalition », a estimé M. Lescure.
- n0 « « On est dans une France aujourd’hui extrêmement fragmentée, et »
- n1 « si on veut la faire avancer il va falloir faire preuve d’esprit de coalition », »
- n2 « a estimé M. Lescure. »
- e0: n0 --restrictive--> n1 (conf 0.38, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.49, marge 0.61)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.39, marge 0.02)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3022 silver-B
Surpêche, réchauffement, pollution, constructions côtières : les causes du blanchissement, puis de la mort de cet écosystème exceptionnel sont connues.
- n0 « Surpêche, réchauffement, pollution, constructions côtières : les causes du blanchissement, »
- n1 « puis de la mort de cet écosystème exceptionnel sont connues. »
- e0: n0 --restrictive--> n1 (conf 0.47, marge 0.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3026 silver-B
Une équipe turque a d’ailleurs enregistré un épisode, sans la moindre activité épileptique associée, chez un patient souffrant par ailleurs d’épilepsie.
- n0 « Une équipe turque a d’ailleurs enregistré un épisode, »
- n1 « sans la moindre activité épileptique associée, »
- n2 « chez un patient souffrant par ailleurs d’épilepsie. »
- e0: n0 --restrictive--> n1 (conf 0.41, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.45, marge 0.40)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.53, marge 0.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3028 silver-B
Chaque année, au moins un « lauréat des citations », une sélection de chercheurs annoncée quelques semaines avant les Nobel, décroche la récompense suprême.
- n0 « Chaque année, au moins un « lauréat des citations », une sélection de chercheurs annoncée  »
- n1 « avant les Nobel, »
- n2 « décroche la récompense suprême. »
- e0: n0 --negative--> n1 (conf 0.33, marge 0.17)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.30, marge 0.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.38, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3033 silver-B
La marque bientôt octogénaire devra, à l’horizon 2035, recentrer ses activités pour se concentrer sur une gamme toujours plus sportive, mais aussi plus restreinte : la valeur plutôt que le volume.
- n0 « La marque bientôt octogénaire devra, à l’horizon 2035, recentrer ses activités »
- n1 « pour se concentrer sur une gamme toujours plus sportive, »
- n2 « mais aussi plus restreinte : la valeur plutôt que le volume. »
- e0: n0 --causale--> n1 (conf 0.82, marge 2.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.39, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.37, marge 0.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3038 silver-B
DÉCRYPTAGE - De nombreuses études présentées lors du congrès de cancérologie de Chicago ont mis en avant l’intérêt de réduire les doses et d’adapter l’heure d’administration des traitements pour optimiser leur efficacité et limiter les effets secondaires.
- n0 « DÉCRYPTAGE - De nombreuses études présentées lors du congrès de cancérologie de Chicago on »
- n1 « avant l’intérêt de réduire les doses et d’adapter l’heure d’administration des traitements »
- n2 « pour optimiser leur efficacité et limiter les effets secondaires. »
- e0: n0 --restrictive--> n1 (conf 0.40, marge 0.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.48, marge 0.22)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.54, marge 1.16)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3039 silver-B
Crayon, mascara, eye-liner ou fards à paupières ne sont pas sans risque pour la santé oculaire.
- n0 « Crayon, mascara, eye-liner ou fards à paupières ne sont pas »
- n1 « sans risque »
- n2 « pour la santé oculaire. »
- e0: n0 --negative--> n1 (conf 0.60, marge 0.74)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.36, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.64, marge 1.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3044 silver-B
Septembre Quand vous êtes pris en charge par un spécialiste pour un cancer de la prostate, votre cas est discuté en réunion pluridisciplinaire (urologue, cancérologue, radiothérapeute, radiologue) qui détermine à partir de votre dossier de la meilleure démarche thérapeutique .
- n0 « Quand vous êtes pris en charge par un spécialiste »
- n1 « pour un cancer de la prostate, »
- n2 « votre cas est discuté en réunion pluridisciplinaire (urologue, cancérologue, radiothérapeu »
- n3 « détermine à partir de votre dossier de la meilleure démarche thérapeutique . »
- e0: n0 --restrictive--> n1 (conf 0.46, marge 0.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.47, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.39, marge 0.35)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.48, marge 0.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.44, marge 0.77)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3049 silver-B
Cet article vous est offert Pour lire gratuitement cet article réservé aux abonnés, connectez-vous Aliénor et Philémon, l’énigme maths du « Monde » nᵒ 114 Avec l’énigme de cette semaine, Mickaël Launay joue les entremetteurs.
- n0 « Cet article vous est offert »
- n1 « Pour lire gratuitement cet article »
- n2 « réservé aux abonnés, »
- n3 « connectez-vous Aliénor et Philémon, l’énigme maths du « Monde » nᵒ 114 Avec l’énigme de ce »
- e0: n0 --negative--> n1 (conf 0.53, marge 0.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --causale--> n2 (conf 0.47, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.37, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.53, marge 0.61)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.77, marge 1.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3050 silver-B
L’État fédéral américain exprime dès 2002 sa préférence pour le dispositif des banques de compensation et la production de titres environnementaux (US Department of Interior, 2003).
- n0 « L’État fédéral américain exprime dès 2002 sa préférence »
- n1 « pour le dispositif des banques de compensation et la production de titres environnementaux »
- n2 « 2003). »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.18)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.44, marge 0.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.37, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3051 silver-B
«Ce trajet, de la frontière italienne à la frontière espagnole, est une forme de métaphore du traitement, avec ses côtes, ses plats et ses chutes, comme dans la maladie», analyse pour sa part le Dr Ellis.
- n0 « «Ce trajet, de la frontière italienne à la frontière espagnole, est une forme de métaphore »
- n1 « pour sa part le Dr Ellis. »
- e0: n0 --negative--> n1 (conf 0.27, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3052 silver-B
Les deux hommes se sont toutefois brouillés publiquement au début de l’été 2025, après des critiques du premier sur les orientations budgétaires du second.
- n0 « Les deux hommes se sont »
- n1 « toutefois brouillés publiquement au début de l’été 2025, »
- n2 « après des critiques du premier sur les orientations budgétaires du second. »
- e0: n0 --negative--> n1 (conf 0.49, marge 0.70)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.58, marge 0.65)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.41, marge 0.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3055 silver-B
Une équipe de 22 chercheurs, essentiellement américains, a donc analysé son génome.
- n0 « Une équipe de 22 chercheurs, essentiellement américains, a »
- n1 « donc analysé son génome. »
- e0: n0 --causale--> n1 (conf 0.60, marge 0.76)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3056 silver-B
DÉCRYPTAGE - Directives européennes, superpositions de normes, arrêtés préfectoraux… Les exploitants forestiers sont très remontés contre cette « réglementation qui paralyse ».
- n0 « DÉCRYPTAGE - Directives européennes, superpositions de normes, arrêtés préfectoraux… »
- n1 « Les exploitants forestiers sont très remontés »
- n2 « contre cette « réglementation qui paralyse ». »
- e0: n0 --negative--> n1 (conf 0.44, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.47, marge 0.94)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.58, marge 1.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3061 silver-B
L’entretien régulier supplante ainsi l’extraction qui n’est plus envisagée que comme l’une des actions de ce plan de gestion.
- n0 « L’entretien régulier supplante »
- n1 « ainsi l’extraction qui n’est plus envisagée que comme l’une des actions de ce plan de gest »
- e0: n0 --negative--> n1 (conf 0.58, marge 0.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3063 silver-B
Difficile de faire abstraction du dérèglement climatique, de la « fin du pétrole » ou du coût de plus en plus élevé d’accès à de nouvelles ressources notamment du fait des externalités négatives que leur extraction engendre.
- n0 « Difficile de faire abstraction du dérèglement climatique, de la « fin du pétrole » ou du c »
- n1 « engendre. »
- e0: n0 --restrictive--> n1 (conf 0.72, marge 1.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3065 silver-B
Les choix, les orientations le protocole expérimental et la méthodologie retenus sont discutés avant les phases de tests/mise en œuvre opérationnelle du dispositif développé.
- n0 « Les choix, les orientations le protocole expérimental et la méthodologie retenus sont disc »
- n1 « avant les phases de tests/mise en œuvre opérationnelle du dispositif développé. »
- e0: n0 --negative--> n1 (conf 0.38, marge 0.44)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3067 silver-B
Enfin, il souligne la complexité de l'établissement de relations causales et l'importance de ces connaissances pour la prévention en santé publique.
- n0 « Enfin, il souligne la complexité de l'établissement de relations causales et l'importance  »
- n1 « pour la prévention en santé publique. »
- e0: n0 --restrictive--> n1 (conf 0.39, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3068 silver-B
Cela élargit le champ des transformations médiées par l'or et donne accès à de nouveaux intermédiaires réactifs ainsi qu'à de nouvelles voies de synthèse.
- n0 « Cela élargit le champ des transformations médiées par l'or et »
- n1 « donne accès à de nouveaux intermédiaires réactifs »
- n2 « ainsi qu'à de nouvelles voies de synthèse. »
- e0: n0 --restrictive--> n1 (conf 0.57, marge 0.52)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.49, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.48, marge 0.68)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3070 silver-B
Un fauteuil roulant avait été amené au cas où elle en aurait besoin, mais elle a préféré marcher.
- n0 « Un fauteuil roulant avait été amené au cas où elle en aurait besoin, »
- n1 « mais elle a préféré marcher. »
- e0: n0 --negative--> n1 (conf 0.34, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3071 silver-B
Appelé autoentrepreneur jusqu’à la fin de 2014, le microentrepreneur bénéficie d’un statut qui allège les formalités pour lancer une activité, sous réserve que le chiffre d’affaires ne dépasse pas un certain seuil (variable selon le métier occupé).
- n0 « Appelé autoentrepreneur jusqu’à la fin de 2014, le microentrepreneur bénéficie d’un statut »
- n1 « pour lancer une activité, »
- n2 « sous réserve que le chiffre d’affaires ne dépasse pas un certain seuil (variable selon le  »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.63)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.45, marge 0.45)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.59, marge 0.90)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3072 silver-B
Les deux phénomènes peuvent donc parfaitement coexister chez une même personne sans se confondre.
- n0 « Les deux phénomènes peuvent »
- n1 « donc parfaitement coexister chez une même personne »
- n2 « sans se confondre. »
- e0: n0 --systemique--> n1 (conf 0.80, marge 1.86)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.76, marge 1.80)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.54, marge 0.54)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3073 silver-B
On parle désormais de darwinisme pour exprimer le concept.
- n0 « On parle désormais de darwinisme »
- n1 « pour exprimer le concept. »
- e0: n0 --systemique--> n1 (conf 0.55, marge 1.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3077 silver-B
Celle-ci acquise, les orientations finalement adoptées par les responsables politiques et les décideurs auront d’autant plus de chances d’obtenir l’adhésion des parties prenantes, et des citoyens de manière générale.
- n0 « Celle-ci acquise, les orientations »
- n1 « finalement adoptées par les responsables politiques et les décideurs auront d’autant plus  »
- n2 « et des citoyens de manière générale. »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.62)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.62, marge 1.32)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.65, marge 1.22)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3078 silver-B
Quoique le président américain en dise, les Ukrainiens semblent décidés à continuer de frapper les raffineries, et renvoient la balle à Poutine, qui sort grand gagnant de cette séquence.
- n0 « Quoique le président américain en dise, »
- n1 « les Ukrainiens semblent décidés à continuer de frapper les raffineries, et renvoient la ba »
- e0: n0 --restrictive--> n1 (conf 0.52, marge 0.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3079 silver-B
DÉCRYPTAGE - Encore sous exploitée, cette solution, qui permet d’aller puiser dans le sous-sol de la chaleur, pourrait couvrir 70% des besoins thermiques de la France, selon le Bureau de recherche géologique et minière.
- n0 « DÉCRYPTAGE - Encore sous exploitée, cette solution, qui »
- n1 « permet d’aller puiser dans le sous-sol de la chaleur, »
- n2 « pourrait couvrir 70% des besoins thermiques de la France, selon le Bureau de recherche géo »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.42, marge 0.49)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.44, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3085 silver-B
Le dépistage néonatal permet, à partir d’une simple goutte de sang prélevé sur le nourrisson quelques heures après sa naissance, de dépister des maladies graves, mais traitables si elles sont prises à temps.
- n0 « Le dépistage néonatal »
- n1 « permet, à partir d’une simple goutte de sang prélevé sur le nourrisson quelques heures »
- n2 « après sa naissance, »
- n3 « de dépister des maladies graves, »
- n4 « mais traitables »
- n5 « si elles sont prises à temps. »
- e0: n0 --systemique--> n1 (conf 0.49, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.46, marge 0.76)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.32, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --restrictive--> n3 (conf 0.37, marge 0.03)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.54, marge 1.07)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --causale--> n4 (conf 0.93, marge 3.36)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --causale--> n4 (conf 0.84, marge 2.57)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e7: n3 --negative--> n5 (conf 0.37, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e8: n4 --causale--> n5 (conf 0.76, marge 1.58)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3086 silver-B
Une loi interdisant la captivité « Nous avons saisi la justice pour empêcher ce transfert.
- n0 « Une loi interdisant la captivité « Nous avons saisi la justice »
- n1 « empêcher ce transfert. »
- e0: n0 --restrictive--> n1 (conf 0.63, marge 0.79)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3088 silver-B
Les complexes plasmide/membrane se stabilisent en un délai de 200 ms. Un rôle distinct de l'intensité du champ électrique et du nombre d'impulsions électriques a été mis en évidence.
- n0 « Les complexes plasmide/membrane se stabilisent en un délai de 200 ms. »
- n1 « Un rôle distinct de l'intensité du champ électrique et du nombre d'impulsions électriques  »
- e0: n0 --negative--> n1 (conf 0.40, marge 0.63)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3089 silver-B
Le pape Léon dirige la prière de l’Angélus depuis la fenêtre du Palais apostolique au Vatican, le 11 octobre 2026.
- n0 « Le pape Léon »
- n1 « dirige la prière de l’Angélus depuis la fenêtre du Palais apostolique au Vatican, »
- n2 « le 11 octobre 2026. »
- e0: n0 --restrictive--> n1 (conf 0.59, marge 1.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.46, marge 0.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.49, marge 0.76)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3090 silver-B
Or, votre problème, c’est que vous êtes convaincu que Philémon et Aliénor s’entendraient à merveille si seulement il était possible de leur fixer un rendez-vous commun.
- n0 « Or, votre problème, c’est que vous êtes convaincu que Philémon et Aliénor s’entendraient à »
- n1 « si seulement il était possible de leur fixer un rendez-vous commun. »
- e0: n0 --causale--> n1 (conf 0.42, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3093 silver-B
Une étude a montré qu’elle peut entraîner une rémission complète après six mois, sans recours à aucun médicament.
- n0 « Une étude a montré qu’elle peut entraîner une rémission complète »
- n1 « après six mois, »
- n2 « sans recours à aucun médicament. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.21)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.39, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.39, marge 0.08)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3096 silver-B
par ex. European Court of Human Rights Press Unit, Factsheet : Expulsions and Extraditions, juin 2011 (consulté le 31 janv.
- n0 « par ex. »
- n1 « European Court of Human Rights Press Unit, Factsheet : Expulsions and Extraditions, juin 2 »
- e0: n0 --systemique--> n1 (conf 0.70, marge 1.53)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3098 silver-B
Cette méthode de transformation, appelée SetExp, permet de réduire le problème d'explosion des états et est appropriée pour le contrôle supervise de SEDTRs.
- n0 « Cette méthode de transformation, appelée SetExp, »
- n1 « permet de réduire le problème d'explosion des états et est appropriée »
- n2 « pour le »
- n3 « contrôle supervise de SEDTRs. »
- e0: n0 --negative--> n1 (conf 0.32, marge 0.11)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.77, marge 2.10)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.38, marge 0.25)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.54, marge 0.94)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --causale--> n3 (conf 0.44, marge 0.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3100 silver-B
Ses symptômes correspondent en revanche aux trois critères diagnostiques du « syndrome de la tête qui explose », traduction littérale de l’expression anglo-saxonne exploding head syndrome (EHS).
- n0 « Ses symptômes correspondent »
- n1 « en revanche aux trois critères diagnostiques du « syndrome de la tête qui explose », »
- n2 « traduction littérale de l’expression anglo-saxonne exploding head syndrome (EHS). »
- e0: n0 --restrictive--> n1 (conf 0.33, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.35, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.54, marge 0.82)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3104 silver-B
Evolution des fonctions de l’agriculture : quelles conséquences pour l’agronomie ?
- n0 « Evolution des fonctions de l’agriculture : quelles conséquences »
- n1 « pour l’agronomie ? »
- e0: n0 --negative--> n1 (conf 0.49, marge 0.67)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3105 silver-B
Si ce type de scénario extrême ne fait pas l’unanimité dans la communauté des géologues, l’idée que la Terre « bouge » par rapport à son axe de rotation, « bascule » en quelque sorte, est en revanche bien acceptée.
- n0 « Si ce type de scénario extrême ne fait pas l’unanimité dans la communauté des géologues, »
- n1 « l’idée que la Terre « bouge » par rapport à son axe de rotation, « bascule » en quelque so »
- n2 « en revanche bien acceptée. »
- e0: n0 --causale--> n1 (conf 0.75, marge 1.60)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.37, marge 0.09)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.37, marge 0.06)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3106 silver-B
Lorsque le Rotary lance son programme PolioPlus en 1985, la poliomyélite paralyse des centaines de milliers d’enfants chaque année dans le monde.
- n0 « Lorsque le Rotary lance son programme PolioPlus en 1985, »
- n1 « la poliomyélite paralyse des centaines de milliers d’enfants chaque année dans le monde. »
- e0: n0 --restrictive--> n1 (conf 0.67, marge 1.56)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3107 silver-B
De surcroît, entravée par un système politique aux nombreux contre-pouvoirs, la politique spatiale allemande n’a pas été guidée par une grande stratégie, si ce n’est celle de l’intégration européenne.
- n0 « De surcroît, entravée par un système politique aux nombreux »
- n1 « contre-pouvoirs, »
- n2 « la politique spatiale allemande n’a pas été guidée par une grande stratégie, »
- n3 « si ce n’est celle de l’intégration européenne. »
- e0: n0 --causale--> n1 (conf 0.44, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.42, marge 0.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.41, marge 0.22)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.49, marge 0.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --negative--> n3 (conf 0.45, marge 0.28)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3108 silver-B
DÉCRYPTAGE - Des chercheurs tentent de mettre au point un vaccin efficace contre presque toutes les souches de grippe, pour mieux protéger la population contre la maladie saisonnière mais aussi en cas de pandémie.
- n0 « DÉCRYPTAGE - Des chercheurs tentent de mettre au point un vaccin efficace »
- n1 « contre presque toutes les souches de grippe, »
- n2 « pour mieux »
- n3 « protéger la population contre la maladie saisonnière »
- n4 « mais aussi »
- n5 « en cas de pandémie. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.78)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.41, marge 0.10)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --systemique--> n2 (conf 0.52, marge 0.54)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.36, marge 0.01)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --sequentielle--> n3 (conf 0.32, marge 0.24)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e5: n2 --causale--> n4 (conf 0.71, marge 1.69)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e6: n3 --restrictive--> n4 (conf 0.60, marge 0.99)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e7: n3 --restrictive--> n5 (conf 0.49, marge 0.51)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e8: n4 --causale--> n5 (conf 0.71, marge 1.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3110 silver-B
""Aujourd'hui, la création de systèmes décisionnels en robotique mobile et autonome requiert l'intégration de nombreuses capacités motrices, sensorielles et cognitives au sein de chacun des projets réalisés.
- n0 « ""Aujourd'hui, la création de systèmes décisionnels en robotique mobile et autonome »
- n1 « requiert l'intégration de nombreuses capacités motrices, »
- n2 « sensorielles et cognitives au sein de chacun des projets réalisés. »
- e0: n0 --negative--> n1 (conf 0.41, marge 0.00)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.48, marge 0.20)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.50, marge 0.63)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3112 silver-B
Des mesures ponctuelles ont été prises pour tenter de résoudre les problèmes au fur et à mesure qu’ils apparaissaient.
- n0 « Des mesures ponctuelles ont été prises »
- n1 « pour tenter de résoudre les problèmes au fur et à mesure qu’ils apparaissaient. »
- e0: n0 --systemique--> n1 (conf 0.52, marge 0.73)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3115 silver-B
"La sécurité des données est la sécurité de la confiance." 🔐 Responsabilité des Éditeurs d’Applications La responsabilité des éditeurs d’applications est un sujet souvent négligé dans le développement mobile.
- n0 « "La sécurité des données est la sécurité de la confiance. »
- n1 « " 🔐 Responsabilité des Éditeurs d’Applications La responsabilité des éditeurs d’applicatio »
- e0: n0 --restrictive--> n1 (conf 0.51, marge 0.38)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3118 silver-B
Ce marché a d’abord une dimension physique, avec une interconnexion des différents réseaux pour pouvoir transporter facilement de l’électricité à travers les frontières.
- n0 « Ce marché a d’abord une dimension physique, avec une interconnexion des différents réseaux »
- n1 « pour pouvoir transporter facilement de l’électricité à travers les frontières. »
- e0: n0 --negative--> n1 (conf 0.43, marge 0.59)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3121 silver-B
Notre travail montre que l’espace n’a jamais été une priorité politique et stratégique de l’Allemagne, mais il l’est devenu davantage, au fur et à mesure du durcissement de la compétition stratégique et économique par et pour l’espace.
- n0 « Notre travail montre que l’espace n’a jamais été une priorité politique et stratégique de  »
- n1 « mais il l’est devenu davantage, »
- n2 « au fur et à mesure du durcissement de la compétition stratégique et économique par et »
- n3 « pour l’espace. »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.81)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.43, marge 0.29)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --causale--> n2 (conf 0.72, marge 1.64)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --negative--> n3 (conf 0.41, marge 0.51)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --restrictive--> n3 (conf 0.49, marge 0.39)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3124 silver-B
Pour la résolution du problème de contrôle centralise, nous avons identifié les conditions d'existence de solution et avons proposé une procédure algorithmique pour la synthèse du superviseur le moins restrictif.
- n0 « Pour la résolution du problème de »
- n1 « contrôle centralise, »
- n2 « nous avons identifié les conditions d'existence de solution et avons proposé une procédure »
- e0: n0 --restrictive--> n1 (conf 0.64, marge 1.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.61, marge 1.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --sequentielle--> n2 (conf 0.35, marge 0.35)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3125 silver-B
CHRONIQUE - Une étude dévoile les contradictions de l’État dans sa politique de protection des écosystèmes fragiles : fiscalité décourageante, dérogations pour les logements sociaux.
- n0 « CHRONIQUE - Une étude dévoile les contradictions de l’État dans sa politique de protection »
- n1 « pour les logements sociaux. »
- e0: n0 --negative--> n1 (conf 0.35, marge 0.27)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3126 silver-B
D’autres essais sont nécessaires pour conclure et confirmer les tendances dégagées dans cette étude.
- n0 « D’autres essais sont nécessaires »
- n1 « pour conclure et confirmer les tendances dégagées dans cette étude. »
- e0: n0 --systemique--> n1 (conf 0.40, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3129 silver-B
Alors que le pic de l’épidémie est prévu au lendemain de Noël, certains non-vaccinés se disent peut-être qu’il est trop tard pour se prémunir contre le virus.
- n0 « Alors que le pic de l’épidémie est prévu au lendemain de Noël, »
- n1 « certains non-vaccinés se disent peut-être qu’il est trop tard »
- n2 « pour se prémunir »
- n3 « contre le virus. »
- e0: n0 --causale--> n1 (conf 0.43, marge 0.13)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --negative--> n2 (conf 0.51, marge 0.51)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --negative--> n2 (conf 0.42, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e3: n1 --systemique--> n3 (conf 0.39, marge 0.26)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e4: n2 --systemique--> n3 (conf 0.61, marge 1.43)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3132 silver-B
Mardi 6 octobre, il est monté dans le bus des lycéens qui allaient défiler pour cette journée nationale de mobilisation.
- n0 « Mardi 6 octobre, il est monté dans le bus des lycéens qui allaient défiler »
- n1 « pour cette journée nationale de mobilisation. »
- e0: n0 --restrictive--> n1 (conf 0.45, marge 0.19)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3139 silver-B
Dans le cadre de ce travail, l’optimisation des paramètres et des conditions de synthèse pour la préparation d’hexaaluminates nanocristallines de grande aires spécifiques a fait l’objet d’une étude détaillée.
- n0 « Dans le cadre de ce travail, l’optimisation des paramètres et des conditions de synthèse »
- n1 « pour la préparation d’hexaaluminates nanocristallines de grande aires spécifiques a fait l »
- e0: n0 --restrictive--> n1 (conf 0.47, marge 0.64)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3141 silver-B
Près d’une semaine plus tard, elle revient sur cet événement marquant alors que l’affaire a été classée sans suite.
- n0 « Près d’une semaine plus tard, elle revient sur cet événement marquant »
- n1 « alors que l’affaire a été classée »
- n2 « sans suite. »
- e0: n0 --systemique--> n1 (conf 0.45, marge 0.42)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --systemique--> n2 (conf 0.59, marge 1.03)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.47, marge 0.12)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3144 silver-B
Résumés Résumé La régulation de la société internationale repose sur la création et l’application du droit international dont l’un des principes fondamentaux est le règlement pacifique des différends.
- n0 « Résumés Résumé La régulation de la société internationale »
- n1 « repose sur la création et l’application du droit international dont l’un des principes fon »
- e0: n0 --negative--> n1 (conf 0.42, marge 0.05)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3148 silver-B
La nouvelle ASNR a donc encore bien du travail pour expliquer les sources de rayonnements ionisants !
- n0 « La nouvelle ASNR a »
- n1 « donc encore bien du travail »
- n2 « pour expliquer les sources de rayonnements ionisants ! »
- e0: n0 --negative--> n1 (conf 0.45, marge 0.50)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e1: n0 --restrictive--> n2 (conf 0.55, marge 0.46)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
- e2: n1 --restrictive--> n2 (conf 0.54, marge 0.37)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___

## v4-3149 silver-B
JarJarBinks.Aragorn Chez les Humains ce sont les prédateurs qui prospèrent...
- n0 « JarJarBinks. »
- n1 « Aragorn Chez les Humains ce sont les prédateurs qui prospèrent... »
- e0: n0 --restrictive--> n1 (conf 0.80, marge 1.88)
  verdict : [ ] oui  [ ] non (motif : ___)  [ ] corriger → ___
