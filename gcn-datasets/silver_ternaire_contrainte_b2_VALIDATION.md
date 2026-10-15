# Validation silver ternaire-batch2 + contrainte — 59 phrases (PROPOSITION)

Fichier : `gcn-datasets/silver_ternaire_contrainte_b2.json` (poids 0.0, hors DATA/train).
Déclencheurs table-11-19 v1 ; exclusions appliquées (tel que, conditionnel journalistique, code).
Vérifié : 59/59 chargées par tok_loader (22 thirds, 12 joints). Cocher UNE case.

## conditional_cause

### tb2001
Si la digue cède, le village sera inondé.
- digue qui cède [condition] —`conditional_cause`→ inondation assertée [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2002
Le blé lève si la pluie d'automne est abondante.
- levée du blé [processus] —`conditional_cause`→ pluie abondante [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2003
Si le vent tourne au nord, les températures chuteront sous zéro.
- vent au nord [condition] —`conditional_cause`→ chute assertée [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2004
La récolte pourrit si l'humidité persiste dix jours.
- pourrissement [processus] —`conditional_cause`→ humidité persistante [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2005
Si la banque relève ses taux, le crédit immobilier ralentira.
- hausse des taux [condition] —`conditional_cause`→ ralentissement asserté [processus] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2006
Les fraises mûrissent si les gelées d'avril les épargnent.
- maturation [processus] —`conditional_cause`→ absence de gel [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## conditional_prevent

### tb2007
Si le gardien dort, les cambrioleurs entreront sans bruit.
- gardien endormi [condition] —`conditional_prevent`→ intrusion assertée [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2008
Le barrage tiendra si les vannes restent ouvertes.
- barrage tenu [etat_local] —`conditional_prevent`→ vannes ouvertes [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2009
Si le témoin se tait, l'enquête s'enlisera.
- témoin muet [condition] —`conditional_prevent`→ enlisement asserté [processus] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2010
La forêt survivra si les pompiers contiennent le flanc ouest.
- forêt sauvée [etat_local] —`conditional_prevent`→ flanc contenu [condition] third={'role': 'Condition', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2011
Si la digue est surveillée, la crue sera contenue.
- surveillance [condition] —`conditional_prevent`→ crue contenue [evenement] third={'role': 'Condition', 'node': 'n000'}
- [ ] oui  [ ] non  [ ] corriger : ___

## mediated_cause

### tb2012
La ville a tenu contre la crue grâce aux barrages de sacs de sable.
- ville tenue [etat_local] —`mediated_cause`→ crue [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2013
Le malade guérit de l'infection via un protocole d'antibiotiques ciblés.
- guérison [processus] —`mediated_cause`→ infection [etat_local] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2014
L'usine tourne grâce à sa turbine et assure la production.
- usine en marche [processus] —`mediated_cause`→ production [processus] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2015
Le village survit à la famine par l'intermédiaire des convois humanitaires.
- survie du village [etat_local] —`mediated_cause`→ famine [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2016
La thèse dissipe l'ignorance grâce aux archives municipales inédites.
- ignorance dissipée [processus] —`mediated_cause`→ ignorance [concept] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2017
Le chantier résorbe le retard via la préfabrication des modules.
- retard résorbé [processus] —`mediated_cause`→ retard [concept] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## mediated_prevent

### tb2018
L'épidémie recule et stoppe la contagion grâce à la vaccination des soignants.
- épidémie en recul [processus] —`mediated_prevent`→ contagion stoppée [processus] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2019
La cave reste sèche et repousse l'inondation via un drainage périphérique continu.
- cave sèche [etat_local] —`mediated_prevent`→ inondation repoussée [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2020
La récolte est sauvée de la sécheresse grâce à l'irrigation nocturne.
- récolte sauvée [etat_local] —`mediated_prevent`→ sécheresse [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2021
Le réseau tient contre la panne par l'intermédiaire des serveurs de secours.
- réseau tenu [etat_local] —`mediated_prevent`→ panne [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2022
Le verger survit au gel grâce aux voiles d'hivernage.
- verger sauf [etat_local] —`mediated_prevent`→ gel [evenement] third={'role': 'Mediator', 'node': 'n001'}
- [ ] oui  [ ] non  [ ] corriger : ___

## joint_cause

### tb2023
La sécheresse et le vent ont attisé l'incendie.
- sécheresse [etat_global] + vent [etat_global] —`joint_cause`→ incendie attisé [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2024
Le froid et l'humidité ont rouillé les outils.
- froid [etat_global] + humidité [etat_global] —`joint_cause`→ outils rouillés [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2025
La grève et la panne ont paralysé la gare.
- grève [evenement] + panne [evenement] —`joint_cause`→ gare paralysée [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2026
Le sel et le gel ont fissuré la chaussée.
- sel [entite] + gel [etat_global] —`joint_cause`→ chaussée fissurée [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2027
La nuit et le brouillard ont égaré les randonneurs.
- nuit [etat_global] + brouillard [etat_global] —`joint_cause`→ randonneurs égarés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2028
L'orage et la grêle ont haché les vignes.
- orage [evenement] + grêle [evenement] —`joint_cause`→ vignes hachées [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## analogy

### tb2029
De même que le loup suit le troupeau, le fisc suit les fraudeurs.
- loup→troupeau [concept] —`analogy`→ fisc→fraudeurs [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2030
Tout comme la marée use la falaise, le temps use les souvenirs.
- marée→falaise [concept] —`analogy`→ temps→souvenirs [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2031
De même que la forge durcit le fer, l'effort durcit le caractère.
- forge→fer [concept] —`analogy`→ effort→caractère [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2032
Tout comme le jardinier taille les rosiers, le professeur corrige les copies.
- jardinier→rosiers [concept] —`analogy`→ professeur→copies [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2033
De même que la chouette voit la nuit, le sonar voit sous l'eau.
- chouette→nuit [concept] —`analogy`→ sonar→eau [concept]
- [ ] oui  [ ] non  [ ] corriger : ___

## counterfactual

### tb2034
Sans le phare, les navires se seraient brisés sur les récifs.
- phare absent (irréel) [condition] —`counterfactual`→ naufrage hypothétique [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2035
Si le pont avait tenu, les renforts seraient arrivés avant l'aube.
- pont tenu (irréel) [evenement] —`counterfactual`→ renforts non arrivés (irréel) [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2036
Sans les vaccins, la variole ravagerait encore des continents.
- vaccins absents (irréel) [condition] —`counterfactual`→ ravage hypothétique [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2037
Si la sentinelle avait veillé, l'attaque aurait échoué.
- veille non faite (irréel) [evenement] —`counterfactual`→ attaque réussie (irréel) [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2038
Sans les digues, la mer aurait repris les polders.
- digues absentes (irréel) [condition] —`counterfactual`→ submersion hypothétique [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2039
Si l'orage avait éclaté plus tôt, la moisson aurait été perdue.
- orage précoce (irréel) [evenement] —`counterfactual`→ moisson perdue (irréel) [evenement]
- [ ] oui  [ ] non  [ ] corriger : ___

## control_dependency

### tb2040
Le couvre-feu interdit toute sortie après vingt et une heures.
- couvre-feu (règle) [contrainte] —`control_dependency`→ sorties interdites [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2041
Le règlement intérieur impose le port du badge.
- règlement (règle) [contrainte] —`control_dependency`→ port imposé [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2042
La loi oblige les commerces à fermer le dimanche.
- loi (règle) [contrainte] —`control_dependency`→ fermeture obligée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### tb2043
L'interdiction de fumer protège les non-fumeurs.
- interdiction (règle) [contrainte] —`prevent`→ non-fumeurs protégés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2044
Le quota de pêche préserve les stocks de cabillaud.
- quota (règle) [contrainte] —`prevent`→ stocks préservés [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2045
L'obligation du masque freinait la contagion.
- obligation (règle) [contrainte] —`prevent`→ contagion freinée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2046
La consigne de tri réduit les déchets incinérés.
- consigne (règle) [contrainte] —`prevent`→ déchets réduits [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___

## control_dependency

### tb2047
Le permis à points sanctionne les chauffards.
- permis (règle) [contrainte] —`control_dependency`→ chauffards sanctionnés [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2048
La limitation à quatre-vingts punit les excès de vitesse.
- limitation (règle) [contrainte] —`control_dependency`→ excès punis [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2049
L'astreinte judiciaire force le débiteur à payer.
- astreinte (règle) [contrainte] —`control_dependency`→ paiement forcé [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## data_dependency

### tb2050
Le couvre-feu dépend du préfet et de la situation sanitaire.
- couvre-feu (règle) [contrainte] —`data_dependency`→ préfet et situation [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2051
La validité du pass dépend du décret en vigueur.
- validité du pass [concept] —`data_dependency`→ décret (règle) [contrainte]
- [ ] oui  [ ] non  [ ] corriger : ___

## filter

### tb2052
Le confinement n'autorisait que les sorties essentielles.
- confinement (règle) [contrainte] —`filter`→ sorties admises [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2053
Seuls les véhicules autorisés franchissent le barrage.
- véhicules autorisés [entite] —`filter`→ barrage (règle) [contrainte]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2054
La zone interdite exclut tout promeneur.
- zone (règle) [contrainte] —`filter`→ promeneurs exclus [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

## condition

### tb2055
Le péage conditionne l'accès à l'autoroute.
- péage (règle) [contrainte] —`condition`→ accès conditionné [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2056
Le visa conditionne l'entrée sur le territoire.
- visa (règle) [contrainte] —`condition`→ entrée conditionnée [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## cause

### tb2057
L'embargo affame la population assiégée.
- embargo (règle) [contrainte] —`cause`→ population affamée [entite]
- [ ] oui  [ ] non  [ ] corriger : ___

### tb2058
Le blocus ruine le commerce du port.
- blocus (règle) [contrainte] —`cause`→ commerce ruiné [processus]
- [ ] oui  [ ] non  [ ] corriger : ___

## prevent

### tb2059
La grève bloque les trains en gare.
- grève (contrainte collective) [contrainte] —`prevent`→ trains bloqués [etat_local]
- [ ] oui  [ ] non  [ ] corriger : ___
