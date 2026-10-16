# Validation silver mini-campagne3 — 74 phrases (PROPOSITION)

Fichier : `gcn-datasets/silver_minicampagne3.json` (poids 0.0, hors DATA/train).
74/74 chargées par tok_loader (24 thirds, 12 joints). Cocher UNE case.

## conditional_cause

### mc3001
Si les abeilles disparaissent, un tiers des cultures dépérira.
- abeilles disparues [condition] —`conditional_cause`→ dépérissement asserté [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3002
Le pain lèvera si la pâte repose deux heures au chaud.
- levée assertée [processus] —`conditional_cause`→ repos de la pâte [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3003
Si la neige tient, les pistes ouvriront samedi.
- neige tenue [condition] —`conditional_cause`→ ouverture assertée [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3004
Les cloches sonneront si le mariage est célébré.
- sonnerie assertée [evenement] —`conditional_cause`→ mariage célébré [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3005
Si la rivière déborde, le moulin s'arrêtera.
- débordement [condition] —`conditional_cause`→ arrêt asserté [processus] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3006
Le cidre sera bon si les pommes sont pressées à froid.
- cidre bon [etat_local] —`conditional_cause`→ pressée à froid [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## conditional_prevent

### mc3007
Si les renforts tardent, la place tombera avant minuit.
- retard des renforts [condition] —`conditional_prevent`→ chute assertée [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3008
Le feu couve si les braises restent chaudes.
- feu couvant [etat_local] —`conditional_prevent`→ braises chaudes [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3009
Si la garde veille, aucun intrus ne passera le fossé.
- garde vigilante [condition] —`conditional_prevent`→ fossé tenu [etat_local] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3010
La cargaison pourrira si la cale prend l'eau.
- pourrissement asserté [processus] —`conditional_prevent`→ voie d'eau [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3011
Si le serrurier tarde, la porte restera close.
- retard du serrurier [condition] —`conditional_prevent`→ porte close [etat_local] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

## conditional_cause

### mc3012
Le blé versera si l'orage éclate avant la moisson.
- verse assertée [processus] —`conditional_cause`→ orage précoce [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## mediated_cause

### mc3013
La forge chauffe le fer via le soufflet actionné au pied.
- forge chaude [etat_local] —`mediated_cause`→ fer [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3014
Le moulin tourne et moud la farine grâce à la roue immergée dans le bief.
- moulin en marche [processus] —`mediated_cause`→ farine [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3015
La cloche sonne l'angélus via la corde tirée par le bedeau.
- cloche qui sonne [processus] —`mediated_cause`→ angélus [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3016
Le pressoir écrase le raisin grâce au vérin actionné au levier.
- pressoir qui écrase [processus] —`mediated_cause`→ raisin [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3017
Le lavoir blanchit le linge au moyen de la cendre dissoute.
- linge blanchi [processus] —`mediated_cause`→ linge [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3018
Le four cuit le pain par l'intermédiaire des braises étalées.
- four qui cuit [processus] —`mediated_cause`→ pain [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## mediated_prevent

### mc3019
La douve protège le château via un pont-levis relevé chaque nuit.
- douve qui protège [processus] —`mediated_prevent`→ château [etat_local] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3020
Le grenier reste sec contre la pluie grâce au chaume refait chaque automne.
- grenier sec [etat_local] —`mediated_prevent`→ pluie [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3021
La bergerie tient chaud contre le froid au moyen des murs en pierre sèche.
- bergerie chaude [etat_local] —`mediated_prevent`→ froid [etat_global] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3022
Le puits garde son eau contre la soif via une margelle haute.
- eau gardée [etat_local] —`mediated_prevent`→ soif [concept] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3023
La grange résiste à l'effondrement par ses contreforts en chêne.
- grange debout [etat_local] —`mediated_prevent`→ effondrement [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3024
La mare gèle et la glace porte les patineurs.
- gel de la mare [processus] —`mediated_prevent`→ patineurs portés [entite] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## joint_cause

### mc3025
Le gel et la bise ont durci la terre.
- gel [etat_global] + bise [etat_global] —`joint_cause`→ terre durcie [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3026
La pluie et le vent ont couché les blés.
- pluie [evenement] + vent [etat_global] —`joint_cause`→ blés couchés [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3027
Le froid et la faim ont décimé le régiment.
- froid [etat_global] + faim [etat_global] —`joint_cause`→ régiment décimé [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3028
La rouille et l'usure ont rongé la faux.
- rouille [processus] + usure [processus] —`joint_cause`→ faux rongée [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3029
L'âge et la maladie ont affaibli le forgeron.
- âge [concept] + maladie [etat_local] —`joint_cause`→ forgeron affaibli [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3030
La poussière et l'humidité ont grippé le mécanisme.
- poussière [entite] + humidité [etat_global] —`joint_cause`→ mécanisme grippé [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## analogy

### mc3031
Tout comme le meunier règle son moulin, le marin règle ses voiles.
- meunier→moulin [concept] —`analogy`→ marin→voiles [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3032
De même que la sage-femme veille l'accouchée, le berger veille l'agnelage.
- sage-femme→accouchée [concept] —`analogy`→ berger→agnelage [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3033
Tout comme le boulanger pétrit la pâte, le maçon gâche le mortier.
- boulanger→pâte [concept] —`analogy`→ maçon→mortier [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3034
De même que le faucon fond sur sa proie, le brochet fond sur le gardon.
- faucon→proie [concept] —`analogy`→ brochet→gardon [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3035
Tout comme le laboureur trace son sillon, le scribe trace sa ligne.
- laboureur→sillon [concept] —`analogy`→ scribe→ligne [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3036
De même que l'abeille butine les fleurs, le collecteur glane les nouvelles.
- abeille→fleurs [concept] —`analogy`→ collecteur→nouvelles [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

## counterfactual

### mc3037
Sans le coq, la basse-cour serait restée endormie.
- coq absent (irréel) [condition] —`counterfactual`→ basse-cour endormie (irréel) [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3038
Si la cloche avait sonné, le village se serait réveillé.
- cloche non sonnée (irréel) [evenement] —`counterfactual`→ village endormi (irréel) [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3039
Sans la lune, les braconniers auraient marché dans le noir.
- lune absente (irréel) [condition] —`counterfactual`→ marche aveugle (irréel) [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3040
Si le meunier avait veillé, la roue n'aurait pas cassé.
- veille non faite (irréel) [evenement] —`counterfactual`→ roue cassée (irréel) [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3041
Sans le chien, le troupeau se serait dispersé dans la lande.
- chien absent (irréel) [condition] —`counterfactual`→ dispersion hypothétique [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3042
Si l'averse avait tenu, les foins auraient pourri sur pied.
- averse durable (irréel) [evenement] —`counterfactual`→ foins pourris (irréel) [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## filter

### mc3043
Le ban seigneurial réserve la chasse au cerf.
- ban (règle) [contrainte] —`filter`→ chasse réservée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## control_dependency

### mc3044
La dîme prélève un dixième des récoltes.
- dîme (règle) [contrainte] —`control_dependency`→ prélèvement [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3045
La corvée mobilise les villageois deux jours par mois.
- corvée (règle) [contrainte] —`control_dependency`→ villageois mobilisés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3046
Le droit de péage taxe les marchands au pont.
- péage (règle) [contrainte] —`control_dependency`→ marchands taxés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### mc3047
L'interdit jeté par l'évêque frappe toute la paroisse.
- interdit (règle) [contrainte] —`cause`→ paroisse frappée [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3048
L'excommunication retranche le fidèle de la communauté.
- excommunication (règle) [contrainte] —`prevent`→ fidèle retranché [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3049
Le jubilé efface les dettes tous les cinquante ans.
- jubilé (règle) [contrainte] —`prevent`→ dettes effacées [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

## enable

### mc3050
L'amnistie libère les prisonniers politiques.
- amnistie (règle) [contrainte] —`enable`→ prisonniers libérés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3051
Le sauf-conduit protège le messager des barrages.
- sauf-conduit (règle) [contrainte] —`prevent`→ messager protégé [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3052
La trêve suspend les hostilités jusqu'à Pâques.
- trêve (règle) [contrainte] —`prevent`→ hostilités suspendues [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## control_dependency

### mc3053
Le guet impose des rondes nocturnes aux bourgeois.
- guet (règle) [contrainte] —`control_dependency`→ rondes imposées [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### mc3054
La taille frappe les roturiers chaque année.
- taille (règle) [contrainte] —`cause`→ roturiers frappés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## filter

### mc3055
Le cens réserve le vote aux plus imposés.
- cens (règle) [contrainte] —`filter`→ vote réservé [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### mc3056
La gabelle renchérit le sel du royaume.
- gabelle (règle) [contrainte] —`cause`→ sel renchéri [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3057
Le monopole interdit toute concurrence étrangère.
- monopole (règle) [contrainte] —`prevent`→ concurrence interdite [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## control_dependency

### mc3058
L'octroi taxe les denrées à l'entrée de la ville.
- octroi (règle) [contrainte] —`control_dependency`→ denrées taxées [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### mc3059
Le blocus continental affame les ports anglais.
- blocus (règle) [contrainte] —`cause`→ ports affamés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3060
Fermez les vannes avant que la cuve ne déborde !
- fermeture ordonnée [processus] —`prevent`→ débordement évité [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3061
Ne laissez aucun outil traîner sur le chantier !
- interdiction [contrainte] —`prevent`→ chantier [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## sequence

### mc3062
Sonnez l'alerte dès que la cloche tinte !
- alerte ordonnée [processus] —`sequence`→ cloche qui tinte [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### mc3063
Quel Carnot ! Ce Carnot foudroie les blés et ruine les moissons !
- foudre qui frappe [evenement] —`cause`→ moissons ruinées [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3064
Quelle tempête ! Le vent arrache les tuiles et inonde les greniers !
- vent qui arrache [processus] —`cause`→ greniers inondés [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3065
Tenez la corde, sinon la barque dérive vers les rochers !
- corde tenue [processus] —`prevent`→ dérive évitée [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3066
Poussez la charrue droit, sinon le sillon dévie vers le fossé !
- ordre de labour [processus] —`prevent`→ déviation évitée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3067
Restez groupés, sinon les loups disperseront le troupeau !
- ordre de grouper [processus] —`prevent`→ dispersion évitée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### mc3068
Gardez les chiens attachés, sinon ils effraieront les brebis !
- ordre d'attacher [processus] —`prevent`→ effroi évité [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## enable

### mc3069
Chantez plus fort, que la vallée entière vous entende !
- chant ordonné [processus] —`enable`→ vallée qui entend [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## sequence

### mc3070
Frappez trois coups avant d'entrer dans la forge !
- coups ordonnés [processus] —`sequence`→ entrée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3071
Méfiez-vous des eaux dormantes qui cachent des sables mouvants !
- méfiance ordonnée [processus] —`prevent`→ sables cachés [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## sequence

### mc3072
Courez annoncer la nouvelle au village avant la tombée du jour !
- ordre de courir [processus] —`sequence`→ jour qui tombe [etat_global]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### mc3073
Ne traversez jamais la lande sans lanterne après minuit !
- interdiction [contrainte] —`prevent`→ minuit [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

## sequence

### mc3074
Videz les seaux avant que l'orage n'éclate sur les toits !
- ordre de vider [processus] —`sequence`→ orage [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___
