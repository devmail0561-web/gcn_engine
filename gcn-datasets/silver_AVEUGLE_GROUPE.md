# Aveugle groupé — vagues J/S/T/bulk/cond (32 cas, seeds 34/35/36/37/38)

Pour chaque cas : la relation proposée TIENT-elle ? Répondre oui/non (+ correction).
Gate : ≥95% d'accord par vague (plancher : 1 désaccord toléré sur 32).

## Cas 1 (j008)
La rivière a emporté le pont et la boue a comblé le gué : le convoi ne traversera pas.
- source « La rivière a emporté le pont »
- source « la boue a comblé le gué »
- cible « le convoi ne traversera pas. »
- relation proposée : **joint_prevent**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 2 (j036)
Le pêcheur raccommode les filets et le mousse amorce les lignes : la pêche sera bonne.
- source « Le pêcheur raccommode les filets »
- source « le mousse amorce les lignes »
- cible « la pêche sera bonne. »
- relation proposée : **joint_cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 3 (j029)
Le vent pousse les nuages et la pluie arrose les semis : les vergers fleurissent.
- source « Le vent pousse les nuages »
- source « la pluie arrose les semis »
- cible « les vergers fleurissent. »
- relation proposée : **joint_cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 4 (j016)
L'incendie dévore la grange et la foudre fend le clocher : la fête est annulée.
- source « L'incendie dévore la grange »
- source « la foudre fend le clocher »
- cible « la fête est annulée. »
- relation proposée : **joint_prevent**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 5 (sp005)
Tout comme la chouette guette la nuit, le garde guette les braconniers.
- source « la chouette guette la nuit, »
- cible « le garde guette les braconniers. »
- relation proposée : **analogy**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 6 (sp040)
Si le beurre avait manqué de glace, il aurait ranci.
- source « Si le beurre avait manqué de glace, »
- cible « il aurait ranci. »
- relation proposée : **counterfactual**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 7 (sp016)
De même que le pressoir écrase le raisin, le fisc écrase le contribuable.
- source « le pressoir écrase le raisin, »
- cible « le fisc écrase le contribuable. »
- relation proposée : **analogy**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 8 (sp007)
Tout comme le laboureur suit son sillon, le marin suit son étoile.
- source « le laboureur suit son sillon, »
- cible « le marin suit son étoile. »
- relation proposée : **analogy**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 9 (t015)
Si le ruisseau grossit au printemps, le moulin tournera jour et nuit.
- source « Si le ruisseau grossit au printemps, »
- cible « le moulin tournera jour et nuit. »
- relation proposée : **conditional_cause** | tiers {'role': 'Condition', 'node': 'n000'}
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 10 (t035)
La mère console l'enfant au moyen d'une chanson fredonnée tout bas : l'enfant s'endort.
- source « La mère console l'enfant »
- cible « l'enfant s'endort. »
- relation proposée : **mediated_cause** | tiers {'role': 'Mediator', 'node': 'n001'}
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 11 (t036)
Le vent disperse les graines par des rafales venues de l'ouest : les friches fleurissent.
- source « Le vent disperse les graines »
- cible « les friches fleurissent. »
- relation proposée : **mediated_cause** | tiers {'role': 'Mediator', 'node': 'n001'}
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 12 (t029)
La consigne retient les curieux au moyen d'une corde tendue à l'entrée : les curieux n'entreront point.
- source « La consigne retient les curieux »
- cible « les curieux n'entreront point. »
- relation proposée : **mediated_prevent** | tiers {'role': 'Mediator', 'node': 'n001'}
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 13 (bulk022)
Militant de la culture scientifique française et donc de la langue française, en pleine époque d’anglais triomphant, Dupuis s’octroyait une liberté d’expression que les jeunes chercheurs d’aujourd’hui peuvent à juste titre lui envier.
- source « Militant de la culture scientifique française et »
- cible « de la langue française , en pleine époque d ’ anglais triomphant , Dupuis s ’ oc »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 14 (bulk226)
L'informatique scientifique est surtout utilisée dans les bureaux d'étude et les entreprises d'ingénierie industrielle car elle permet de simuler, par la recherche opérationnelle ou par itération, des scénarios de façon rapide et fiable.
- source « L'informatique scientifique est surtout utilisée dans les bureaux d'étude et les »
- cible « de simuler , par la recherche opérationnelle ou par itération , des scénarios de »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 15 (bulk251)
Peu de personnes interviewées sont catégoriques sur des conséquences purement négatives ou purement positives des technologies nomades sur le bien-être au travail ; la mesure semble donc être de rigueur en la matière.
- source « Peu de personnes interviewées sont catégoriques sur des conséquences purement né »
- cible « être de rigueur en la matière . »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 16 (bulk156)
Ces transferts de carbone dépendant du folate interviennent également de façon centrale dans le métabolisme de certains acides aminés : méthionine, glycine, sérine et donc indirectement dans la synthèse des protéines.
- source « Ces transferts de carbone dépendant du folate interviennent également de façon c »
- cible « indirectement dans la synthèse des protéines . »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 17 (bulk055)
L'apparition d'un nuage qu'on dit de vapeur d'eau signifie que cette vapeur s'est transformée en très petites gouttes d'eau liquide (ce n'est donc plus un gaz).
- source « L'apparition d'un nuage qu'on dit de vapeur d'eau signifie que cette vapeur s'es »
- cible « plus un gaz ) . »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 18 (bulk283)
L'utilisation des habitudes de vie et des mises en situation écologiques permet aux patients de se projeter dans leur réadaptation.
- source « L'utilisation des habitudes de vie et des mises en situation écologiques »
- cible « aux patients de se projeter dans leur réadaptation . »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 19 (bulk057)
Ceci permet d'établir une connexion avec les théories cinématiques des ondes croissantes [1] [2], tout en précisant quels sont les modes générateurs d'énergie.
- source « Ceci »
- cible « d'établir une connexion avec les théories cinématiques des ondes croissantes [ 1 »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 20 (bulk212)
- 1830 : Michael Faraday avec sa loi d'induction permet de réaliser la première dynamo et le premier moteur électrique - 1879 : Thomas Edison invente la lampe à incandescence.
- source « - 1830 : Michael Faraday avec »
- cible « loi d'induction permet de réaliser la première dynamo et le premier moteur élect »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 21 (bulk150)
En effet l'étude théorique permet de préciser la nature des perturbations ainsi que leur origine, dans l'hypothèse où ces fluctuations obéissent à une description MHD.
- source « En effet l'étude théorique »
- cible « de préciser la nature des perturbations ainsi que leur origine , dans l'hypothès »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 22 (bulk025)
Le rapport entre E et m c2 est donc égal à 1020/109=1011 et n'est autre que le facteur d'étirement du temps .
- source « Le rapport entre E et m c2 est »
- cible « égal à 1020 / 109 = 1011 et n'est autre que le facteur d'étirement du temps . »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 23 (bulk289)
Le changement d’environnement et la rencontre avec d’autres personnes atteintes de la même pathologie permet au patient d’accepter son lymphœdème et d’apprendre à le prendre en charge.
- source « Le changement d ’ environnement et la rencontre avec d ’ autres personnes attein »
- cible « au patient d ’ accepter son lymph dème et d ’ apprendre à le prendre en charge . »
- relation proposée : **enable**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 24 (bulk077)
Les données ont permis de cartographier la fertilité P de l'Albanie en fonction des facteurs intensité, quantité et capacité ainsi que des paramètres cinétiques décrivant le flux moyen des ions phosphate entre la phase solide et liquide du sol.
- source « Les données ont permis de cartographier la fertilité P de l ' Albanie en fonctio »
- cible « que des paramètres cinétiques décrivant le flux moyen des ions phosphate entre l »
- relation proposée : **cause**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 25 (bulk058)
La validation de ces résultats permettra d'élaborer un schéma d'aménagement du bassin versant Saboun en vue de freiner le phénomène de la dégradation des sols.
- source « La validation de ces résultats permettra d'élaborer un schéma d'aménagement du b »
- cible « vue de freiner le phénomène de la dégradation des sols . »
- relation proposée : **motivation**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 26 (cond042)
Il s’agit donc d’une élection importante, même si elle est moins connue que les législatives.
- source « si elle est moins connue que les législatives . »
- cible « Il s ’ agit donc d ’ une élection importante , »
- relation proposée : **concession**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 27 (cond022)
Lorsqu'un preneur conclut un contrat de prise en pension d'animaux selon lequel il se charge de l'élevage des animaux et de l'entretien des parcelles concernées, la présomption de fraude prévue à l'article L.
- source « Lorsqu'un preneur conclut un contrat de prise en pension d'animaux selon lequel  »
- cible « présomption de fraude prévue à l'article L . »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 28 (cond100)
L’activité physique est indispensable… sauf quand elle devient trop exigeante pour un organisme en pleine croissance.
- source « elle devient trop exigeante pour un organisme en pleine croissance . »
- cible « L ’ activité physique est indispensable … sauf »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 29 (cond010)
En effet, il n'existe pas, au sens strict, d'institution financière internationale si l'on associe à cette expression l'idée d'une organisation internationale intergouvernementale au sens classique.
- source « l'on associe à cette expression l'idée d'une organisation internationale intergo »
- cible « En effet , il n'existe pas , au sens strict , d'institution financière internati »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 30 (cond098)
Dans chacune de ces formes, nous analysons comment se façonnent des articulations singulières entre morale et technique lorsque les individus publicisent des liens de causalité entre des sources de pollution et des données.
- source « les individus publicisent des liens de causalité entre des sources de pollution  »
- cible « Dans chacune de ces formes , nous analysons comment se façonnent des articulatio »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 31 (cond094)
Il constitue un anion simple lorsque son cortège électronique s'en trouve excédentaire, il est alors chargé négativement.
- source « son cortège électronique s'en trouve excédentaire , il est alors chargé négative »
- cible « Il constitue un anion simple »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___

## Cas 32 (cond085)
Comme chaque année, des économistes salueront un prix « plus que mérité », quand d’autres dénonceront une supercherie en rappelant qu’il s’agit d’un faux Nobel.
- source « d ’ autres dénonceront une supercherie en rappelant qu ’ il s ’ agit d ’ un faux »
- cible « Comme chaque année , des économistes salueront un prix « plus que mérité » , »
- relation proposée : **condition**
- verdict : [ ] oui  [ ] non  [ ] corriger → ___
