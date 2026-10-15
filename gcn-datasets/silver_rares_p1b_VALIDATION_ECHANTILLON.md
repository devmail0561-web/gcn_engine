# Validation P1b échantillon — 20 risquées + 10 hasard seed 42 (PROPOSITION)

Règle : si ≥28/30 tiennent, verdict global oui (esprit IC95). Sinon revue intégrale de la famille fautive.
Risquées stratifiées = `sous` (ambigu control_dep/enable) ×5, `part` ×2, concept/contrainte ×5, marqueurs faibles ×2, +6.

### p1b029 [RISQUÉE]
Le syndic a fait évacuer l'immeuble sous la fissure du mur.
- le syndic [entite] —`control_dependency`→ immeuble évacué [etat_local] (marqueur `sous`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b040 [RISQUÉE]
Le gendarme a fait souffler le conducteur sous l'éthylotest.
- le gendarme [entite] —`control_dependency`→ conducteur contrôlé [entite] (marqueur `sous`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b033 [RISQUÉE]
Le capitaine a fait demi-tour sous la brume épaisse.
- le capitaine [entite] —`control_dependency`→ demi-tour ordonné [processus] (marqueur `sous`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b039 [RISQUÉE]
Le curateur a mis le musée sous scellés après le vol.
- le curateur [entite] —`control_dependency`→ musée sous scellés [etat_local] (marqueur `sous`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b088 [RISQUÉE]
Tous les dossiers sont traités, à part celui de l'héritier.
- tous les dossiers [concept] —`filter`→ dossier exclu [concept] (marqueur `part`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b098 [RISQUÉE]
Toute la classe part en sortie, à part les punis.
- toute la classe [entite] —`filter`→ punis exclus [etat_local] (marqueur `part`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b028 [RISQUÉE]
Le patron a réduit les horaires sous la chute des commandes.
- le patron [entite] —`control_dependency`→ horaires réduits [concept] (marqueur `réduit`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b129 [RISQUÉE]
La crédibilité du journal repose sur ses sources.
- crédibilité du journal [concept] —`data_dependency`→ sources fondatrices [entite] (marqueur `repose`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b027 [RISQUÉE]
La commission a rejeté le projet sous son irrégularité manifeste.
- la commission [entite] —`control_dependency`→ projet rejeté [concept] (marqueur `rejeté`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b117 [RISQUÉE]
Sa réputation repose sur vingt ans de chroniques.
- sa réputation [concept] —`data_dependency`→ chroniques fondatrices [processus] (marqueur `repose`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b012 [RISQUÉE]
L'éditeur a censuré le passage sous la pression du pouvoir.
- éditeur [entite] —`control_dependency`→ passage censuré [concept] (marqueur `censuré`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b073 [RISQUÉE]
Le tuteur maintient la tige malgré le vent fort.
- le tuteur [entite] —`prevent`→ tige maintenue [etat_local] (marqueur `maintient`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b077 [RISQUÉE]
Le rideau de fer contenait la foule des supporters.
- rideau de fer [entite] —`prevent`→ foule contenue [etat_local] (marqueur `contenait`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b075 [RISQUÉE]
Le rappel vaccinal écarte tout risque de récidive.
- rappel vaccinal [processus] —`prevent`→ récidive écartée [concept] (marqueur `écarte`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b078 [RISQUÉE]
Le verrouillage automatique interdit l'accès après minuit.
- verrouillage automatique [contrainte] —`prevent`→ accès interdit [processus] (marqueur `interdit`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b103 [RISQUÉE]
La bibliothèque prête tous les livres hormis les usuels.
- tous les livres [entite] —`filter`→ usuels exclus [concept] (marqueur `hormis`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b110 [RISQUÉE]
Le prix du pain dépend du cours du blé.
- le prix du pain [concept] —`data_dependency`→ cours déterminant [concept] (marqueur `dépend`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b003 [RISQUÉE]
Le ministre a limité la vitesse sous la pression des associations.
- le ministre [entite] —`control_dependency`→ vitesse limitée [concept] (marqueur `limité`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b107 [RISQUÉE]
Le verdict repose sur trois témoignages concordants.
- le verdict [concept] —`data_dependency`→ témoignages fondants [evenement] (marqueur `repose`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b127 [RISQUÉE]
Sa thèse découle de trois années d'archives.
- sa thèse [concept] —`data_dependency`→ archives fondatrices [processus] (marqueur `découle`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b108 [hasard]
Son humeur dépend de la météo du matin.
- son humeur [etat_local] —`data_dependency`→ météo déterminante [evenement] (marqueur `dépend`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b097 [hasard]
Le spectacle plaît à tous, excepté aux critiques grincheux.
- tous conquis [entite] —`filter`→ critiques exclus [entite] (marqueur `excepté`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b105 [hasard]
Seuls les riverains stationnent dans la ruelle.
- riverains admis [entite] —`filter`→ ruelle filtrée [etat_local] (marqueur `Seuls`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b082 [hasard]
Tout le village a voté, hormis les absents excusés.
- tout le village [entite] —`filter`→ absents exclus [entite] (marqueur `hormis`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b062 [hasard]
Le traitement précoce évite l'aggravation de la maladie.
- traitement précoce [processus] —`prevent`→ aggravation évitée [processus] (marqueur `évite`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b035 [hasard]
Le berger a rassemblé le troupeau sous l'orage imminent.
- le berger [entite] —`control_dependency`→ troupeau rassemblé [entite] (marqueur `rassemblé`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b066 [hasard]
L'arrosage nocturne prévient le dessèchement des semis.
- arrosage nocturne [processus] —`prevent`→ dessèchement évité [processus] (marqueur `prévient`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b089 [hasard]
Le forfait inclut toutes les remontées, hormis le téléphérique.
- toutes les remontées [entite] —`filter`→ téléphérique exclu [entite] (marqueur `hormis`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b044 [hasard]
Le chef a renvoyé le plat sous la plainte du client.
- le chef [entite] —`control_dependency`→ plat renvoyé [entite] (marqueur `renvoyé`)
- [ ] oui  [ ] non  [ ] corriger : ___

### p1b122 [hasard]
La guérison dépend de la précocité du traitement.
- la guérison [processus] —`data_dependency`→ précocité déterminante [concept] (marqueur `dépend`)
- [ ] oui  [ ] non  [ ] corriger : ___
