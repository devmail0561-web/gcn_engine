# ÉTUDE : GCN — Moteur Causal et Système Agentique
## État des lieux, limites structurelles et vision vers un agent de niveau Hermès

**Auteur** : Michel Tendeng  
**Date** : 2026-09-25  
**Statut** : Document d'étude — base de conception pour la prochaine phase de développement  
**Périmètre** : GCN-Core (Rust), GCN-Python, GCN-Tools, architecture agentique cible

---

## Avertissement préliminaire

Ce document est un état des lieux honnête. Il ne minimise pas les lacunes et ne sur-vend pas les capacités existantes. Chaque affirmation est fondée sur l'inspection directe du code source. Les points non vérifiables dans le code sont explicitement marqués comme tels. L'objectif est de fournir une base de conception rigoureuse, sans ambiguïté, pour orienter le développement vers un agent causal de haut niveau.

---

## Table des matières

1. [Contexte et objectif de cette étude](#1-contexte-et-objectif)
2. [Distinction fondamentale : le moteur est le noyau](#2-distinction-fondamentale)
3. [Ce qu'est le moteur GCN aujourd'hui](#3-ce-quest-le-moteur-gcn-aujourdhui)
4. [Capacités actuelles confirmées du moteur](#4-capacités-actuelles-confirmées)
5. [Limites structurelles du moteur actuel](#5-limites-structurelles-du-moteur-actuel)
6. [Le paradigme agentique — ce qu'il exige](#6-le-paradigme-agentique)
7. [La contrainte fondamentale : les CIRs ont des sources](#7-contrainte-fondamentale-les-cirs-ont-des-sources)
8. [Vision : l'agent Hermès](#8-vision-lagent-hermès)
9. [Niveaux de raisonnement requis](#9-niveaux-de-raisonnement-requis)
10. [Architecture de la mémoire agentique](#10-architecture-de-la-mémoire-agentique)
11. [Optimisation des recherches et du raisonnement sur les graphes](#11-optimisation-des-recherches-et-du-raisonnement)
12. [Communication et protocoles](#12-communication-et-protocoles)
13. [Directives et objectifs](#13-directives-et-objectifs)
14. [Analyse des écarts (Gap Analysis)](#14-analyse-des-écarts)
15. [Architecture cible](#15-architecture-cible)
16. [Priorités de développement](#16-priorités-de-développement)

---

## 1. Contexte et objectif de cette étude

### 1.1 Origine

GCN (Graph Causal Network) a été conçu comme un moteur d'extraction et de raisonnement causal. Son architecture repose sur deux implémentations complémentaires : un frontend symbolique en Rust (gcn-core, 9 crates) qui analyse du texte FR/EN/code via des taxonomies YAML, et un pipeline d'apprentissage en Python (gcn-python) qui apprend les représentations causales depuis des données annotées. Les deux convergent vers le même format pivot : le **CIR (Causal Intermediary Representation)**, un graphe JSON sérialisable décrivant des nœuds causaux et des relations typées entre eux.

### 1.2 Ce que cette étude adresse

Au fil du développement, le moteur a été enrichi de capacités de raisonnement (Pearl niveaux 1–3), de session (gcn-discuss), d'indexation de corpus (gcn-index), et de chaîne de données autonome (gcn-tools). Ces additions ont été construites dans le paradigme d'un **outil interactif** — un humain pose une question, le moteur répond, et la session se termine.

Cette étude constate que ce paradigme atteint ses limites dès qu'on envisage un usage réel à l'échelle : flux de données continus, surveillance autonome, intégration avec des LLMs, coopération entre agents. Le moteur est un outil puissant mais passif. Ce document décrit ce qu'il faudrait pour en faire un **agent actif et autonome**, et cartographie précisément ce qui manque.

### 1.3 Ce que ce document n'est pas

Ce document n'est pas un plan d'implémentation (il ne contient pas de code). Ce n'est pas non plus un document marketing — les performances actuelles sont citées telles qu'elles sont mesurées (val edge_f1 ≈ 0.31–0.47 selon le dataset et l'architecture), sans extrapolation. C'est un document de conception architecturale.

---

## 2. Distinction fondamentale : le moteur est le noyau

### 2.1 Principe d'architecture

Le moteur GCN (gcn-core + gcn-python) est le **noyau** du système. Il est à l'agent ce que le noyau Linux est à un système d'exploitation : un composant stable, focalisé, à responsabilité unique. Il ne connaît pas les agents, les flux, les alertes, ni les directives. Il ne gère pas de sessions, ne surveille pas de sources, ne prend pas d'initiatives.

Le **système agentique** est la couche construite *autour* du moteur. C'est elle qui orchestre, qui persiste, qui décide, qui communique. Le moteur est son outil principal — pas son équivalent.

Cette distinction n'est pas cosmétique. Elle protège le moteur d'une dérive fonctionnelle qui compromettrait sa stabilité, sa testabilité et sa réutilisabilité. Toute capacité qui relève de l'orchestration, de la persistance ou de la communication doit être dans le système agentique, jamais dans le moteur.

### 2.2 Responsabilités du moteur (noyau)

Le moteur fait exactement trois choses, et rien d'autre :

1. **Extraire des CIRs** depuis du texte (FR/EN) ou du code, par voie symbolique (Rust) ou par apprentissage (Python)
2. **Raisonner causalement** sur un graphe CIR donné (Pearl 1–3, BFS, CYCLES, GAPS)
3. **Valider** l'intégrité structurelle d'un CIR (types de nœuds, relations licites, spans 1-based)

Le moteur est **sans état persistant**. Il reçoit une entrée, produit une sortie, s'arrête. Il ne sait pas ce qui s'est passé avant lui ni ce qui se passera après.

Le moteur est **sans opinion sur les sources**. Il n'évalue pas la fiabilité d'une source, ne maintient pas de registre de confiance des producteurs, ne décide pas quoi accepter ou refuser. Ces décisions appartiennent au système agentique.

Le moteur **n'a pas de boucle**. Il ne surveille rien, n'alerte pas, n'initie rien. Il est invoqué, il produit, il s'arrête.

### 2.3 Nature de l'agent : mémoire = intelligence, graphes = connaissance

**La mémoire est l'intelligence de l'agent.** Ce n'est pas un stockage — c'est le fondement de ce qu'il est capable de faire. Un agent dont la mémoire est bien organisée, avec des sources évaluées et une connaissance catégorisée par domaine et pertinence, surpasse un agent dont la mémoire est un tas indifférencié, même si les deux partagent le même moteur de raisonnement.

**Les graphes causaux sont sa connaissance.** Pas des fichiers de données, pas des logs — des structures qui représentent comment les choses s'influencent dans son domaine, avec les preuves à l'appui.

L'agent n'est **pas un pipeline hardcodé**. Il ne suit pas une séquence fixe PERCEVOIR→RAISONNER→AGIR câblée dans le code. C'est un raisonnneur qui dispose d'un ensemble d'**outils** et d'une **mémoire de connaissance persistante**, et qui décide à chaque cycle quels outils invoquer selon la situation courante et ses directives.

**Les outils** sont les primitives d'action de l'agent. Chaque outil a une interface définie, est interchangeable, et peut être étendu sans modifier la logique agentique centrale. Exemples :

| Outil | Ce qu'il fait |
|---|---|
| `extract(text, source)` | Appelle le moteur → produit un CIR avec provenance |
| `ingest(cir)` | Intègre un CIR dans la mémoire de connaissance |
| `query_causes(concept)` | Interroge la mémoire → causes sourcées |
| `query_effects(concept)` | Interroge la mémoire → effets sourcés |
| `query_chain(a, b)` | Chemin causal entre deux concepts |
| `query_cycles()` | Cycles dans le graphe courant |
| `query_gaps(chain)` | Lacunes dans une chaîne surveillée |
| `validate(claim)` | Confronte un claim à la mémoire → verdict sourcé |
| `alert(condition, payload)` | Émet une alerte structurée |
| `report(scope, format)` | Produit un rapport depuis la mémoire |
| `fetch(adapter, config)` | Récupère des données via un adaptateur externe |
| `counterfactual(concept)` | Pearl do-calculus sur la mémoire |

L'agent choisit ses outils selon une **politique de sélection à règles** — pas un LLM orchestrateur, pas une séquence fixe. Les règles sont évaluées dans l'ordre de priorité décroissante à chaque début de cycle :

| Priorité | Condition | Outils déclenchés |
|---|---|---|
| 1 (urgence) | Alerte de monitoring active | `alert()` → `report()` |
| 2 (requête entrante) | Claim externe reçu | `validate()` |
| 3 (requête causale) | Question causale reçue | `query_causes()` / `query_chain()` / etc. |
| 4 (nouvelles données) | Adaptateur a produit des données | `fetch()` → `extract()` → `ingest()` |
| 5 (maintenance) | Relations STALE détectées | décroissance, révision |
| 6 (analyse) | Cycle de fond | `query_cycles()` → `query_gaps()` si domaine surveillé |
| — (rien) | Aucune condition | aucun outil — cycle no-op |

Les priorités 1–3 sont déclenchées par événement extérieur (push). Les priorités 4–6 sont déclenchées par l'agent lui-même selon l'horloge et l'état de la mémoire (pull). La directive définit quels niveaux sont actifs et à quelle fréquence.

**Règle anti-famine** : une priorité stricte rendrait P5/P6 (maintenance, fond) inaccessibles si P4 (flux de données) est saturé en continu. La solution est un **budget par cycle** défini dans la directive : P1–P3 consomment leur quota au-delà duquel P4–P6 sont garantis un slot minimum par cycle, indépendamment du flux entrant.

**Conflit multi-conditions** : si plusieurs conditions de niveaux différents sont vraies au même cycle (exemple : claim entrant P2 + nouvelles données P4), les conditions sont traitées **séquentiellement dans l'ordre de priorité**, pas en parallèle. Un seul outil principal par tick de priorité. Cela simplifie l'état et évite les effets de bord inter-outils dans un même cycle.

Un cycle où aucune condition n'est vraie ne déclenche aucun outil.

**La mémoire de connaissance** est le graphe causal persistant. Elle n'est pas une session — elle n'est jamais réinitialisée entre deux cycles, entre deux tâches, entre deux redémarrages de l'agent. Elle accumule la connaissance causale extraite de toutes les sources analysées, avec provenance complète, et son contenu vieillit et évolue selon le cycle de vie des relations.

### 2.4 Responsabilités du système agentique

Le SA encapsule la logique de sélection des outils, la mémoire de connaissance, et l'interprétation des directives :

- **Mémoire de connaissance** : graphe causal persistant, indexé, avec cycle de vie
- **Registre d'outils** : liste des outils disponibles, leurs interfaces, leurs dépendances
- **Boucle de raisonnement** : à chaque cycle, évaluer la situation et sélectionner les outils appropriés
- **Directives** : objectifs, domaines, seuils, actions autorisées — ce qui guide la sélection des outils
- **Registre d'adaptateurs** : composants interchangeables pour les sources externes

### 2.5 Ce que cette distinction interdit

- Ajouter une boucle d'ingestion continue **dans** le moteur → appartient au système agentique
- Implémenter la décroissance temporelle **dans** `CausalGraph` → appartient au système agentique
- Exposer une API HTTP **depuis** gcn-python directement → appartient au système agentique
- Gérer des directives **dans** le moteur → appartient au système agentique

**Exception licite** : étendre le format CIR pour inclure la provenance (URI, méthode, timestamp) est une modification du moteur, car le CIR est la frontière entre le noyau et le reste du système. Tout ce qui traverse cette frontière doit être défini dans le noyau.

---

## 3. Ce qu'est le moteur GCN aujourd'hui

### 3.1 Architecture en quatre couches

Le moteur repose sur quatre couches conceptuelles :

| Couche | Composant | Rôle |
|---|---|---|
| Représentation | CIR (JSON) | Format pivot universel entre toutes les couches |
| Extraction symbolique | gcn-core (Rust) | Texte FR/EN/code → CIR via taxonomies YAML |
| Extraction apprise | gcn-python (PyTorch/NumPy) | UD → CIR via MLP/R-GCN/GAT entraîné |
| Données | gcn-tools (scraper + annotateur) | Brut → JSONL → annotation → silver → fusion |

**Point important** : les taxonomies YAML sont exclusivement des ressources pour les outils d'extraction (gcn-core, gcn-tools). Le moteur Python ne lit jamais de YAML. Le moteur consomme uniquement du JSON. Cette séparation est une décision d'architecture délibérée et doit être préservée.

### 3.2 Le CIR — objet central et frontière du noyau

Le CIR est le format pivot de tout le système. Il décrit :

- **Nœuds** (7 types) : `etat`, `action`, `transition`, `processus`, `condition`, `entite`, `etat_systemique`
- **Arêtes** (11 relations + inverses) : `cause`, `enable`, `prevent`, `condition`, `concession`, `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency`
- **Attributs de nœud** : `type`, `label`, `token_span` (1-based), `origin`, `scope`, `temporal_index`, `modifiers`, `attributes`
- **Attributs d'arête** : `relation`, `confidence`, `explicit`, `negated`

Le CIR est conçu pour être **sérialisable en JSON**, **indépendant de la langue du texte source**, et **interopérable** entre les couches Rust et Python.

### 3.3 Invariant de traçabilité

Toute arête CIR produite par le moteur est extraite depuis un texte source réel. La confiance associée (`confidence`) mesure la certitude de l'extraction, pas une probabilité inventée. Le champ `source_text` dans `CausalGraph.edges` (4-tuple) conserve la phrase source. Ce principe de traçabilité est **non négociable** et constitue la différence fondamentale entre GCN et un LLM qui genère des relations causales sans preuve.

---

## 4. Capacités actuelles confirmées du moteur

Cette section décrit ce que le moteur **fait** aujourd'hui, vérifié dans le code source.

### 4.1 Extraction causale depuis texte FR/EN

Deux voies :

**Voie symbolique** (gcn-core Rust) : texte brut → CIR en une commande via `gcn analyze`. Couverture limitée aux taxonomies YAML versionnées. Qualité heuristique (~80–85 % revendiqué, non mesuré indépendamment). La causalité implicite est couverte avec une confiance abaissée (0.5). Limite connue : pas de coréférence inter-phrases.

**Voie ML** (gcn-python) : séquence de tokens UD → CIR via le pipeline MLP/R-GCN/GAT. Nécessite un checkpoint entraîné. Performances mesurées : val edge_f1 entre 0.31 et 0.47 selon le dataset et l'architecture. Ces chiffres sont honnêtes et ne sont pas comparables entre eux sans alignement d'échelle.

**Voie bridge** (frontend/bridge.py) : texte brut → sous-processus gcn-cli → représentations UD heuristiques. Qualité dégradée connue (is_negative à 0 %, morphologie imprécise). C'est une solution de transition, pas une solution de production.

### 4.2 Extraction depuis code

gcn-frontend-code (26 tests) parse Python/Rust/JS via tree-sitter et produit des CIR isomorphes au langage naturel. `if x < seuil: réduire(y)` produit la même structure causale que "si x dépasse le seuil, on réduit y". C'est le **seul domaine immédiatement opérationnel sans annotation métier** : la causalité dans le code est structurelle, pas lexicale.

### 4.3 Raisonnement causal formel (Pearl)

Implémenté dans les modules Rust et partiellement exposé en Python via `InstructionHandler` :

- **Pearl niveau 1** (association) : `WHY(X)` → ancêtres, `WHAT(X)` → descendants, `CHAIN(A, B)` → plus court chemin BFS
- **Pearl niveau 2** (intervention) : `DO(X)` — coupe les arêtes entrantes de X, propage les effets en aval
- **Pearl niveau 3** (contrefactuel) : `COUNTERFACTUAL(X)` — effets réels vs. monde sans X

Limite : ces trois niveaux couvrent un graphe supposé complet et correct. Ils ne raisonnent pas sur l'incertitude du graphe lui-même.

### 4.4 Analyse structurelle du graphe

- `CYCLES?` : détection et classification des boucles (positive/négative/oscillation)
- `GAPS?` : lacunes causales — arêtes concession/opposition + nœuds non résolus
- Hygiène (gcn-middleend, 17 tests) : arêtes pendantes, boucles réflexives, nœuds orphelins, violations d'ordre temporel — diagnostics Error/Warning

### 4.5 Session et persistance actuelle

`SessionStore` (cli/session.py) maintient en RAM : `CausalGraph` (nodes dict + edges list + adjacency), vecteurs enrichis par clé stable, historique JSONL. Persistance sur disque : `graph.json` (JSON complet), `session_vecs.npz` (NumPy compressed), `history.jsonl` (append-only).

### 4.6 Verbalisation

`ReferenceDecoder` : CIR → texte structuré via templates déterministes. Aucune hallucination possible — le verbalizer ne dit que ce que le graphe contient. `TrainableDecoder` existe mais n'est pas entraîné avec des données de qualité suffisante.

### 4.7 Chaîne de données autonome

gcn-scrape (Wikipedia, HAL, arXiv, RSS, GitHub, web) → gcn-annotate (LLM, regex, spaCy/UD) → gates silver (v3–v16) → fusion → gcn-train. Le moteur peut produire son propre carburant d'entraînement.

---

## 5. Limites structurelles du moteur actuel

> **Note de lecture** : cette section distingue deux types de limites.
> Les §§5.2–5.4 décrivent des lacunes **internes au moteur** — des choses que le moteur fait mal ou pas du tout dans le périmètre qui est le sien.
> Les §§5.5–5.7 décrivent des **absences qui appartiennent au système agentique**, pas au moteur. Elles sont listées ici parce qu'elles expliquent pourquoi le moteur seul ne suffit pas pour un usage agentique — non pas parce qu'il faudrait les implémenter dans le moteur.

### 5.1 Limite fondamentale : ce n'est pas un agent

Le moteur actuel est un **outil passif**. Il répond quand on l'invoque. Entre deux invocations, il n'existe pas — il n't observe rien, ne détecte rien, n'agit pas. Il n'a pas de boucle d'exécution propre, pas d'état persistant entre sessions (sauf si explicitement sauvegardé et rechargé), pas de notion d'objectif.

Cette limite n'est pas une nuance — c'est une différence de paradigme. Un outil répond à des requêtes. Un agent perçoit, raisonne, décide et agit de façon continue, orienté par des objectifs, sans attendre qu'on lui pose une question.

### 5.2 Mémoire naïve et non scalable

La structure de mémoire actuelle présente quatre défauts structurels :

**4.2.1 Persistance par snapshot complet**

`session.save()` réécrit `graph.json` et `session_vecs.npz` en intégralité à chaque appel. Sur un flux continu de 1000 phrases, cela produit 1000 réécritures complètes d'un fichier de taille croissante. Le seul composant correctement incrémental est `history.jsonl` (append-only).

**4.2.2 Aucun index sur les labels de nœuds**

`find_causes(keyword)` et `find_effects(keyword)` effectuent un scan linéaire O(n) sur la liste complète des arêtes, avec matching word-boundary sur les labels. Sur un graphe de 10 000 arêtes, chaque requête lit 10 000 entrées. Il n'existe aucun index inversé `label → [node_ids]`.

**4.2.3 Aucune résolution de conflits**

Quand deux sources extraient des relations contradictoires sur la même paire de nœuds (source A dit `cause`, source B dit `prevent`), les deux arêtes sont ajoutées au graphe sans signalement ni arbitrage. Le graphe accumule des contradictions silencieusement.

**4.2.4 Aucune décroissance temporelle**

La confiance d'une relation est figée à la valeur calculée au moment de l'extraction. Une relation extraite une seule fois il y a 6 mois, non re-confirmée, a la même confiance qu'une relation confirmée par 15 sources récentes. Il n'y a aucun mécanisme de vieillissement de la connaissance.

### 5.3 Raisonnement incomplet

Les trois niveaux de Pearl couvrent l'observation, l'intervention et le contrefactuel sur un graphe **supposé complet et correct**. Les raisonnements suivants sont absents :

**Raisonnement abductif** : Étant donné une observation (un effet constaté), remonter vers l'explication causale la plus plausible parmi plusieurs hypothèses. Pearl part de causes connues. L'abduction part de l'effet et cherche la cause — c'est le raisonnement diagnostique fondamental.

**Raisonnement temporel** : Le champ `temporal_index` existe dans le CIR mais n'est jamais utilisé dans le raisonnement. "A cause B avec un délai de 3 cycles" est traité identiquement à "A cause B instantanément". Les séquences, délais, et seuils temporels ne sont pas raisonnés.

**Raisonnement multi-échelle** : La même relation causale a des causes différentes selon le niveau d'abstraction. Un bug de sécurité est causé par une erreur de codage au niveau technique, par une pratique de développement au niveau équipe, par une politique de sécurité au niveau organisation. Ces échelles sont traitées comme un graphe plat sans hiérarchie.

**Raisonnement adversarial** : Quelle est la chaîne causale la plus fragile ? Quel nœud, s'il est supprimé ou manipulé, produit le plus grand effet sur le graphe ? Ce raisonnement sur la robustesse et les points de vulnérabilité du graphe est absent.

**Raisonnement normatif** : Le graphe décrit la causalité **observée** (descriptive). Il ne raisonne pas sur la causalité **prescrite** (normative) — ce qui devrait causer quoi selon une règle, un contrat, ou une norme. GAPS est une approximation structurelle, pas du raisonnement normatif.

**Méta-raisonnement** : L'agent ne raisonne pas sur la qualité de son propre modèle. Il ne sait pas si son graphe est assez dense pour répondre fiablement à une question donnée, ni si une réponse repose sur une seule source non confirmée ou sur vingt sources concordantes.

**Raisonnement par analogie** : La structure causale "vulnérabilité → exploitation → persistance → exfiltration" en CTI est isomorphe à "faille contractuelle → non-conformité → sanction → atteinte à la réputation" en droit. Le moteur ne reconnaît pas ces isomorphismes entre domaines et ne les exploite pas.

### 5.4 Provenance incomplète

Le moteur conserve le texte source d'une arête dans le 4-tuple de `CausalGraph.edges`. C'est insuffisant pour une provenance réelle. Il manque :

- **URI du document source** : aucun lien vers le fichier, l'URL ou la base de données d'où vient la phrase
- **Méthode d'extraction** : le CIR ne distingue pas si la relation a été extraite par le frontend Rust symbolique, par le modèle ML Python, ou par l'annotateur silver de gcn-tools — trois niveaux de confiance fondamentalement différents
- **Version du modèle** : le `checkpoint_hash` existe dans le manifest des vecteurs mais n'est pas intégré dans le CIR lui-même
- **Timestamp d'extraction** : il n'y a aucune trace de quand une relation a été extraite pour la première fois, ni quand elle a été vue pour la dernière fois

Sans provenance complète, l'agent ne peut pas répondre à la question la plus élémentaire d'un auditeur : *"d'où vient cette information, et est-elle encore valide ?"*. Il ne peut pas non plus invalider une relation dont la source a été modifiée, retirée ou discréditée.

### 5.5 Absence de communication et de protocoles *(appartient au système agentique)*

Le moteur n'expose qu'une interface CLI. Il n'y a pas d'API HTTP, pas de protocole inter-agents, pas de format de sortie adaptatif. Pour qu'un LLM consomme le résultat d'une requête, il faut un pont manuel. Pour que deux instances du moteur coopèrent, il n'existe aucun protocole défini.

Cette limite empêche toute intégration native dans un système multi-agents, un pipeline RAG, ou un service de validation en temps réel.

### 5.6 Absence de directive et d'objectifs *(appartient au système agentique)*

Le moteur n'a aucune notion de mission, de domaine de surveillance, de seuil d'alerte, ou d'action autorisée. Il traite toutes les requêtes avec la même priorité, il ne sait pas ce qu'il cherche, et il n'agit jamais spontanément. Sans directive, il ne peut pas être autonome — même avec tous les raisonnements du monde, il ne saurait pas quand ni pourquoi les appliquer.

### 5.7 Absence de boucle agentique *(appartient au système agentique)*

Le moteur est invoqué, il produit un résultat, il s'arrête. Il n'y a pas de boucle PERCEVOIR → RAISONNER → PLANIFIER → AGIR → METTRE À JOUR. Entre deux invocations, il n'existe pas. Cette propriété est la conséquence directe du paradigme CLI interactif dans lequel il a été conçu, et c'est la limite la plus fondamentale pour un usage agentique.

---

## 6. Le paradigme agentique

### 6.1 Définition opérationnelle d'un agent

Un agent est un système qui, de façon continue et autonome :
- **Perçoit** son environnement (données entrantes, événements, sorties d'autres agents)
- **Maintient un modèle interne** de l'état du monde dans son domaine
- **Raisonne** sur ce modèle pour comprendre la situation actuelle
- **Planifie** les actions appropriées compte tenu de ses objectifs
- **Agit** — émet des sorties, déclenche des processus, communique avec d'autres agents
- **Apprend** — met à jour son modèle à partir des nouvelles informations et des résultats de ses actions
- **Sait quand s'abstenir** — reconnaît les situations où ses informations sont insuffisantes pour agir de façon fiable

Un agent n'est pas un outil amélioré. C'est un paradigme différent : autonomie, continuité, orientation par objectifs.

### 6.2 Ce que le paradigme agentique change pour GCN

Dans le paradigme actuel (outil) :
- L'humain lance une commande
- Le moteur analyse et répond
- La session se termine

Dans le paradigme agentique :
- L'agent surveille des flux de données définis
- Il extrait et intègre en continu des CIRs dans son modèle du monde
- Il détecte des conditions (nouvelles chaînes, conflits, gaps critiques, cycles)
- Il agit sur ces conditions sans attendre une requête (alerte, validation, rapport)
- Il gère son propre état entre les cycles (connaissance fraîche, connaissance stale, conflits non résolus)
- Il communique avec d'autres agents ou systèmes via des protocoles définis

### 6.3 Le modèle du monde

Dans un agent GCN, le graphe causal est le **modèle du monde**. Ce n'est pas une base de données — c'est la représentation structurée de comment les choses s'influencent dans le domaine de l'agent. Tout ce que l'agent perçoit, il l'intègre dans ce modèle. Tout ce qu'il dit en sort.

Ce modèle du monde doit être :
- **Vivant** : mis à jour en continu au fil des nouvelles observations
- **Temporel** : les relations vieillissent, se confirment ou s'affaiblissent avec le temps
- **Conflictuel** : capable de représenter des contradictions entre sources sans les effacer
- **Hiérarchique** : capable de représenter des causalités à plusieurs niveaux d'abstraction
- **Borgné** : l'agent sait ce que son modèle ne contient pas, pas seulement ce qu'il contient

---

## 7. Contrainte fondamentale : les CIRs ont des sources

### 7.1 La règle

Un CIR ne peut pas être inventé. Chaque nœud et chaque arête d'un CIR doit être tracé vers une source vérifiable : un document, une URL, un fichier de code, une ligne de log. C'est cette propriété qui distingue GCN d'un LLM générant des relations causales probables — GCN prouve, le LLM opine.

Cette règle s'applique à tous les acteurs du système : le moteur symbolique, le modèle ML, l'annotateur, et tout agent utilisant GCN. Aucun agent ne peut écrire dans le graphe causal sans référencer une source.

### 7.2 Ce que ça implique pour l'architecture

**Pour l'ingestion** : l'agent ne peut pas créer de CIRs. Il ne peut qu'extraire des CIRs depuis des sources qu'il a analysées. Toute écriture dans le graphe est une extraction, jamais une déduction libre.

**Pour le raisonnement** : les relations déduites (A → C déduit de A → B et B → C) sont licites, mais elles doivent être marquées `origin: "derived"` avec les deux sources qui les justifient. Les déductions sont des inférences tracées, pas des inventions.

**Pour la validation** : quand l'agent reçoit un claim d'un LLM ou d'un humain, il doit trouver la source dans le graphe pour répondre CONFIRMÉ. S'il ne trouve pas de source, il répond NON CONFIRMÉ — pas FAUX. L'absence de preuve n'est pas une preuve d'absence.

**Pour la communication** : toute sortie de l'agent incluant une affirmation causale doit inclure la source. Émettre "A cause B" sans citation est une violation du contrat de traçabilité.

### 7.3 Ce que ça interdit

- Un agent ne peut pas "synthétiser" des relations causales depuis plusieurs CIRs sans tracer la déduction
- Un agent ne peut pas augmenter la confiance d'une relation par inférence libre — seulement par re-confirmation depuis de nouvelles sources
- Un agent ne peut pas supprimer une relation du graphe sans tracer la raison (source invalide, contradiction arbitrée)

---

## 8. Vision : l'agent Hermès

### 8.1 Pourquoi "Hermès"

Hermès, dans la mythologie grecque, est le messager des dieux, guide des âmes, dieu des transitions et de la communication. Il connaît les deux mondes, se déplace entre eux, transmet l'information avec précision et rapidité. Il sait quoi dire, à qui, quand — et quand se taire.

Pour GCN, "Hermès" désigne un agent qui :
- Connaît son domaine en profondeur (le graphe causal comme modèle du monde)
- Se déplace entre les niveaux d'abstraction (micro, méso, macro)
- Transmet l'information dans le format adapté à chaque destinataire
- Agit au bon moment, avec la bonne précision, sur la bonne cible
- Reconnaît les limites de son savoir et les communique explicitement

Ce n'est pas un objectif poétique — c'est une spécification fonctionnelle.

### 8.2 Profil de l'agent Hermès

| Propriété | Comportement attendu |
|---|---|
| Autonomie | Agit sans attendre une requête humaine |
| Continuité | Tourne en boucle, pas en mode one-shot |
| Orientation | Guidé par des directives explicites (domaine, objectifs, seuils) |
| Traçabilité | Chaque affirmation cite sa source |
| Honnêteté | Distingue ce qu'il sait, ce qu'il infère, et ce qu'il ignore |
| Adaptabilité | Format de sortie selon le destinataire |
| Sobriété | N'agit que quand c'est justifié |

### 8.3 Rôles de l'agent

L'agent Hermès remplit quatre rôles fondamentaux qui sont indépendants du domaine applicatif :

**Rôle 1 : Accumulateur intelligent**
Il ingère des sources, extrait des CIRs, les intègre dans le graphe. Il ne fait pas que stocker — il évalue chaque CIR entrant : est-ce cohérent avec ce que le graphe contient déjà ? Est-ce une confirmation, une contradiction, ou une nouvelle information ? Il filtre le bruit. Il refuse les CIRs qui ne passent pas les seuils de qualité définis dans ses directives.

Ce rôle est le seul à écrire dans le graphe. Les trois autres lisent.

**Rôle 2 : Raisonneur**
Il répond aux questions causales sur le graphe avec preuves citées. Il raisonne à plusieurs niveaux (cf. section 8). Il sait quand son graphe n'est pas assez dense pour répondre de façon fiable, et le dit.

**Rôle 3 : Validateur**
Il reçoit un claim externe (d'un LLM, d'un humain, d'un autre agent) et retourne un verdict structuré :
- `CONFIRMED` : relation présente dans le graphe, source citée, confiance X
- `UNCONFIRMED` : relation absente du graphe — pas de preuve, pas de réfutation
- `CONTRADICTED` : le graphe contient une relation opposée, source citée, confiance Y

Ce rôle est la pièce la plus différenciante du système. C'est là que GCN devient une couche de preuve pour les LLMs.

**Rôle 4 : Moniteur**
Il surveille le graphe en continu et déclenche des alertes sur des conditions définies dans ses directives : nouveau CYCLE dans un domaine critique, GAPS dans une chaîne surveillée, relation haute-confiance invalidée, source fiable qui contredit une croyance établie. Il agit sur événement, pas sur requête.

### 8.4 Ce que l'agent fait avec ce qu'il sait

Un agent qui accumule sans agir est une base de données muette. Voici les opérations que l'agent Hermès doit effectuer sur sa connaissance :

**Arbitrage des conflits** : quand deux sources produisent des relations contradictoires sur la même paire, l'agent ne les accumule pas silencieusement. Il calcule un score de confiance composite pour chaque version et élit la dominante. La formule d'arbitrage est détaillée en §10.8.

**Décroissance temporelle** : la confiance d'une relation décroît si elle n'est pas re-confirmée périodiquement. La décroissance est paramétrable dans les directives (domaine CTI : décroissance rapide, domaine droit : décroissance lente). Une relation non vue depuis N jours passe à l'état "stale" avant d'être marquée pour révision.

**Déduction tracée** : si A → B et B → C sont dans le graphe et que A → C ne l'est pas, l'agent peut inférer A → C avec `origin: "derived"`, confiance = min(conf(A→B), conf(B→C)), et les deux arêtes sources comme justification.

**Détection des lacunes orientée objectif** : le moteur actuel détecte les GAPS structurels. L'agent Hermès détecte les lacunes **par rapport à ses objectifs** — une chaîne incomplète dans un domaine surveillé est plus urgente qu'un nœud orphelin dans un domaine hors scope. Il formule la question : "pour compléter la chaîne entre X et Y, quelles sources dois-je acquérir ?"

**Déduplication sémantique** : deux nœuds avec des labels différents mais référant au même concept ("authentification", "auth", "login") doivent être fusionnés ou liés. Ce n'est pas un problème trivial — il nécessite une résolution d'entités.

---

## 9. Niveaux de raisonnement requis

### 9.1 Les trois niveaux de Pearl — socle nécessaire mais insuffisant

Pearl's ladder of causation est le fondement du raisonnement causal :

- **Niveau 1 — Association** : P(Y|X) — que se passe-t-il en présence de X ?
- **Niveau 2 — Intervention** : P(Y|do(X)) — que se passe-t-il si je force X ?
- **Niveau 3 — Contrefactuel** : P(Y_x|X', Y') — qu'aurait-il résulté si X avait été différent ?

Ces trois niveaux sont implémentés dans le moteur actuel. Ils sont nécessaires mais couvrent uniquement un graphe supposé complet et correct. Un agent intelligent doit raisonner au-delà.

### 9.2 Raisonnement abductif

**Définition** : inférence à la meilleure explication. Étant donné un effet E observé, trouver la cause C la plus plausible parmi les causes candidates dans le graphe.

**Différence avec Pearl** : Pearl niveau 1 répond à "quelles sont les causes de X dans le graphe ?" L'abduction répond à "parmi toutes les causes possibles de X, laquelle explique le mieux l'observation ?" — en pondérant par la confiance, la fréquence, et la cohérence avec d'autres observations.

**Application concrète** : en CTI, un analyste observe une exfiltration de données. L'abduction interroge le graphe pour identifier le scénario d'attaque le plus probable parmi les chaînes connues qui aboutissent à l'exfiltration.

**Ce qui manque** : un algorithme de scoring des hypothèses causales pondéré par la confiance des arêtes et la fréquence de confirmation par les sources.

### 9.3 Raisonnement temporel

**Définition** : raisonnement sur l'ordre, la durée, et les délais dans les chaînes causales.

**Ce que le CIR a** : le champ `temporal_index` sur chaque nœud (entier ordinal) et la structure `TemporalRef`/`TemporalGap` dans le CIR.

**Ce qui est absent** : l'utilisation de ces champs dans le raisonnement. L'agent devrait pouvoir répondre à : "A cause B en combien de temps ?" ou "Si A se produit, B se produira avant ou après C ?" ou "Cette chaîne peut-elle boucler dans un délai T ?"

**Application concrète** : en conformité, une obligation légale a un délai. La chaîne "infraction → détection → signalement → sanction" a des délais réglementés. Le raisonnement temporel permet de vérifier si les contraintes de délai sont respectées.

### 9.4 Raisonnement multi-échelle

**Définition** : la même relation causale s'exprime différemment à différents niveaux d'abstraction. Un agent multi-échelle navigue entre ces niveaux, agrège des causes micro en causes macro, et décompose des causes macro en mécanismes micro.

**Ce qui manque** : une hiérarchie explicite entre nœuds (nœud macro agrégant plusieurs nœuds micro), des opérateurs d'agrégation et de décomposition, et une logique de navigation entre niveaux.

**Application concrète** : en analyse de risque systémique financier, la "contagion" est une cause macro. Sous-jacents : "liquidité insuffisante", "coûts de refinancement", "ventes forcées" — causes micro. Un agent multi-échelle passe de l'une à l'autre selon la question posée.

### 9.5 Raisonnement adversarial

**Définition** : raisonnement sur la robustesse d'une chaîne causale face à des suppressions ou manipulations délibérées de nœuds ou d'arêtes.

**Questions adressées** :
- Quel nœud, s'il est supprimé, coupe le plus grand nombre de chaînes causales ?
- Quelle arête a le plus grand impact sur la propagation d'un effet ?
- Quels nœuds sont des points de défaillance unique (single points of failure) dans le graphe ?

**Ce qui manque** : des métriques de centralité causale (non pas centrality de graphe général, mais centrality pondérée par la confiance et la fréquence des chemins), des algorithmes de recherche de points de vulnérabilité.

**Application concrète** : en CTI, identifier quel contrôle de sécurité, s'il est contourné, ouvre le plus grand nombre de vecteurs d'attaque. En conformité, quel processus, s'il est défaillant, invalide le plus grand nombre d'obligations.

### 9.6 Raisonnement normatif

**Définition** : distinguer la causalité **descriptive** (ce qui cause quoi dans les faits, extrait des sources) de la causalité **prescriptive** (ce qui devrait causer quoi selon les règles, normes, contrats).

**Ce qui existe** : le graphe descriptif (toutes les relations extraites de textes réels).

**Ce qui manque** : des opérateurs de comparaison entre graphe normatif et descriptif, et une mesure d'écart de conformité.

**Clarification sur la partition** : le normatif et le descriptif ne sont pas deux domaines distincts — ce sont deux **sous-domaines** au sein du même domaine. La partition se fait ainsi :

```
domaine: conformite_rgpd
  sous-domaine: descriptif   ← extrait de logs, code, rapports d'audit
  sous-domaine: normatif     ← extrait du texte du RGPD, guides CNIL
```

Les deux sous-domaines partagent le même espace de concepts (les nœuds réfèrent aux mêmes entités) mais ont des arêtes de sources différentes. L'opérateur de comparaison prend les deux sous-graphes en entrée et produit une liste d'écarts. Cette organisation évite la double partition incompatible (domaine × mode) qui rendrait les requêtes croisées impossibles.

**Application concrète** : en audit RGPD, le graphe normatif dit "accès aux données personnelles doit activer journalisation". Le graphe descriptif (extrait des logs et du code) dit "accès aux données personnelles active journalisation 73 % du temps". L'écart est une non-conformité prouvée, sourcée, quantifiée.

### 9.7 Méta-raisonnement

**Définition** : raisonnement sur la qualité et les limites du modèle causal lui-même.

**Questions adressées** :
- Mon graphe est-il assez dense pour répondre fiablement à cette question ?
- Cette réponse repose-t-elle sur une seule source non confirmée ?
- Existe-t-il des hypothèses alternatives que mon graphe ne distingue pas encore ?
- Quelle est la couverture de mon graphe sur ce domaine ?

**Ce qui manque** : des métriques de densité et de couverture du graphe par domaine, un mécanisme d'estimation de la fiabilité d'une réponse basé sur le nombre de sources et la diversité des chemins, une interface explicite pour communiquer l'incertitude épistémique.

### 9.8 Raisonnement par analogie structurelle

**Définition** : reconnaître qu'un pattern causal dans un domaine est isomorphe à un pattern dans un autre domaine, et transférer les inférences de l'un à l'autre.

**Exemple** : "vulnérabilité → exploitation → persistance → exfiltration" (CTI) est structurellement identique à "faille → déclencheur → maintien → impact" dans d'autres domaines. Un agent qui reconnaît cet isomorphisme peut transférer les tactiques de détection et de mitigation.

**Ce qui manque** : un comparateur de structure de graphe (graph isomorphism / subgraph matching), un registre de patterns connus, un mécanisme de transfert inter-domaines.

---

## 10. Architecture de la mémoire agentique

### 10.1 La mémoire est l'intelligence

La mémoire de l'agent n'est pas un stockage passif. C'est le siège de son intelligence. Ce qu'il sait, comment il l'a organisé, la confiance qu'il accorde à chaque source, la pertinence qu'il attribue à chaque relation — tout cela constitue ce qu'il est capable de faire.

Un agent avec une mémoire mal organisée produit des réponses de mauvaise qualité même avec le meilleur moteur de raisonnement. Un agent avec une mémoire bien organisée peut répondre correctement avec un raisonnement minimal.

La métaphore juste est celle du **bibliothécaire** : il ne se contente pas de stocker des livres, il les classe par domaine, par pertinence, par auteur, par fiabilité de l'éditeur. Il sait où chercher, il sait quoi croire, il sait quand une source est douteuse. Sa valeur n'est pas dans le nombre de livres qu'il possède — c'est dans la qualité de son organisation et de son jugement sur les sources.

### 10.2 Organisation de la connaissance : le modèle bibliothécaire

La connaissance n'est pas un graphe plat. Elle est organisée selon trois dimensions :

**Dimension 1 — Domaine**
Chaque relation appartient à un domaine (cybersécurité, conformité, recherche, code, droit, finance…) et à un sous-domaine. Le domaine est assigné à l'ingestion selon la directive active. Les graphes sont partitionnés par domaine — une requête dans le domaine CTI ne traverse pas le graphe conformité.

**Dimension 2 — Pertinence**
Au sein d'un domaine, les relations ne sont pas équivalentes. La pertinence combine :
- Le statut dans le cycle de vie (CONFIRMÉE > ACTIVE > NOUVELLE)
- Le nombre de sources indépendantes qui la confirment
- La centralité dans les chaînes du domaine (une relation sur un chemin critique est plus pertinente qu'un nœud feuille)
- La fraîcheur (une relation récente est plus pertinente dans les domaines à évolution rapide)

**Dimension 3 — Source**
Chaque relation est rattachée à sa source avec provenance complète. Les sources ne sont pas toutes équivalentes — leur fiabilité est évaluée et mise à jour en continu par la mémoire elle-même.

### 10.3 Le registre de sources : l'intelligence sur les sources

La mémoire maintient un **registre de sources** — une évaluation de fiabilité construite depuis l'expérience accumulée, jamais déclarée à la main.

Pour chaque source connue, le registre tient :

```json
{
  "source_uri": "https://nvd.nist.gov/feeds/json/cve/",
  "domain": "cybersecurity_cti",
  "first_seen": "2026-01-15T10:00:00Z",
  "last_seen": "2026-09-25T08:00:00Z",
  "extraction_method": "symbolic_rust",
  "relations_contributed": 847,
  "confirmed_by_independent_source": 731,
  "contradicted_by_higher_confidence": 12,
  "confirmation_rate": 0.863,
  "contradiction_rate": 0.014,
  "reliability_score": 0.849,
  "status": "trusted"
}
```

Le `reliability_score` n'est pas déclaré — il est calculé depuis les données accumulées. Une source qui contribue des relations systématiquement confirmées par d'autres sources indépendantes gagne en fiabilité. Une source dont les relations sont fréquemment contredites perd en fiabilité. Ce processus est continu et automatique.

**Les statuts de fiabilité** :
- `trusted` : confirmation_rate ≥ 0.80, contradiction_rate ≤ 0.05
- `monitored` : confirmation_rate entre 0.60 et 0.80 — acceptée avec pondération réduite
- `suspect` : contradiction_rate > 0.10 — signalée, relations admises en NOUVELLE seulement
- `blocked` : exclue explicitement par directive ou après audit manuel

**Ce que le registre permet** :
- Pondérer la confiance initiale d'un CIR par la fiabilité de sa source
- Alerter quand une source fiable produit soudainement des relations contradictoires (signal d'anomalie)
- Répondre à la question "d'où vient cette information et est-elle fiable ?" avec un score fondé sur l'historique
- Identifier les sources de référence dans un domaine (les sources les plus confirmées)

### 10.4 Principes d'organisation

1. **Partitionnement par domaine** : graphes distincts par domaine, requêtes ciblées
2. **Écriture incrémentale** : WAL — chaque nouvelle relation s'ajoute sans réécrire l'ensemble
3. **Vieillissement** : la confiance décroît pour les relations non re-confirmées
4. **Résolution de conflits** : les contradictions sont arbitrées par fiabilité de source + fréquence de confirmation
5. **Accessibilité indexée** : index inversé sur labels, index par domaine, index de pertinence
6. **Registre de sources actif** : fiabilité mise à jour en continu depuis l'expérience accumulée

### 10.5 Structure de persistance recommandée : WAL + index

**Write-Ahead Log (WAL)** pour le graphe :
- `graph_base.json` : dernier snapshot compact (réécrit uniquement lors d'une compaction explicite)
- `graph_delta.jsonl` : nouvelles arêtes depuis le dernier snapshot (append-only, une ligne JSON par CIR)
- Chargement = `graph_base.json` + replay de `graph_delta.jsonl`
- Compaction = fusion delta dans base + truncation du delta (déclenchée sur `/save` ou sur seuil de taille)

Coût d'écriture par item : O(1) — une ligne JSON. Coût de lecture : O(n_delta) pour le replay, puis O(1) avec index.

**Index inversé sur labels** :
```
label_index: dict[str, list[node_id]]
```
Construit une seule fois après chargement, mis à jour à chaque insertion. Réduit `find_causes` et `find_effects` de O(n_edges) à O(k) où k est le nombre d'arêtes adjacentes au nœud trouvé.

**Vecteurs par fichier individuel** :
Remplacer `session_vecs.npz` (réécriture complète) par un répertoire `vecs/` avec un fichier `.npy` par vecteur, nommé par clé stable (`sha256(texte) + "_" + position`). Le manifest reste un JSONL append-only.

### 10.6 Cycle de vie de la connaissance

Chaque relation dans le graphe traverse les états suivants :

```
NOUVELLE → ACTIVE → CONFIRMÉE (multi-sources) → STALE (non re-confirmée) → EN RÉVISION → [MAINTENUE | INVALIDÉE]
```

- **NOUVELLE** : extraite d'une seule source, confiance = confiance d'extraction
- **ACTIVE** : vue dans au moins 2 sources concordantes, confiance élevée
- **CONFIRMÉE** : vue dans N ≥ seuil sources, confiance maximale dans le domaine
- **STALE** : non re-confirmée depuis T jours (T défini dans les directives), confiance décroissante
- **EN RÉVISION** : contradite par une source récente, arbitrage en cours
- **MAINTENUE** : contradiction résolue en faveur de la relation originale
- **INVALIDÉE** : contradiction résolue en faveur de la nouvelle source, relation retirée

### 10.7 Provenance complète

Chaque arête dans le graphe doit porter :

```json
{
  "source": "n001",
  "target": "n002",
  "relation": "cause",
  "attributes": {
    "confidence": 0.87,
    "explicit": true,
    "negated": false
  },
  "provenance": {
    "document_uri": "https://example.com/rapport.pdf",
    "document_hash": "sha256:abc123...",
    "sentence": "La pluie provoque l'inondation parce que le sol est saturé.",
    "token_span": [1, 5],
    "extraction_method": "symbolic_rust",
    "model_version": "gcn-core-2.5.0",
    "extracted_at": "2026-09-25T14:32:00Z"
  }
}
```

Ces 4 champs de provenance sont **immuables** — ils décrivent le fait d'observation ponctuel. Les champs `last_confirmed_at`, `confirmation_count`, et `status` sont gérés par le SA dans l'arête enrichie (§10.9) et ne figurent jamais dans le CIR-noyau.

```json
```

Le champ `extraction_method` prend l'une des valeurs : `symbolic_rust`, `ml_python`, `silver_annotator`, `derived` (pour les déductions tracées).

### 10.8 Formule d'arbitrage des conflits

Quand deux arêtes concurrentes A et B décrivent la même paire (src, dst) avec des relations différentes ou des confiances contradictoires, le SA calcule un **score composite** pour chacune :

```
score(arête) = conf_extraction
             × reliability(source)
             × norm_conf(confirmations)
             × freshness(last_confirmed_at)
```

Où :
- `conf_extraction` ∈ [0,1] : confiance produite par l'extraction (CIR-noyau, immuable)
- `reliability(source)` ∈ [0,1] : score du registre de sources (calculé depuis l'historique)
- `norm_conf(n)` = `log₂(1 + n) / log₂(1 + Nmax)` : bonus logarithmique normalisé ∈ [0,1]. Nmax est la valeur de saturation définie dans la directive (défaut : 50). Base 2 pour lisibilité. Une relation à 1 confirmation obtient log₂(2)/log₂(51) ≈ 0.16 — voulu : une seule confirmation ne suffit pas à dominer.
- `freshness(t)` = `exp(-(now - t) / T_domaine)` : décroissance exponentielle depuis la dernière confirmation. T_domaine est la demi-vie de la confiance dans ce domaine, définie dans la directive (défauts : T=7j CTI, T=90j conformité, T=365j droit). La formule précédente avait un paramètre λ redondant — supprimé.

Le score est donc toujours ∈ [0,1] et interprétable comme un score de confiance composite.

**Règle de quorum** : si `score(A) / score(B) > seuil_quorum`, A l'emporte. Le seuil par défaut est 2.0 — **valeur à calibrer empiriquement sur corpus annoté avant mise en production**, non fondée théoriquement. Si le ratio est inférieur au seuil, le conflit reste EN RÉVISION.

**Exemple corrigé** (Nmax=50, T_domaine=7j, âge arête B = 0j) :
- Arête A : `symbolic_rust`, conf 0.90, reliability 0.85, 1 confirmation → score = 0.90 × 0.85 × 0.16 × 1.0 = **0.12**
- Arête B : `ml_python`, conf 0.60, reliability 0.72, 15 confirmations → score = 0.60 × 0.72 × 0.61 × 1.0 = **0.26**

ratio = 0.26/0.12 = 2.2 > 2.0 → B l'emporte. La lecture est conservée : le consensus de 15 confirmations bat l'extraction symbolique isolée. Les valeurs sont maintenant dans [0,1] et le ratio reste interprétable.

### 10.9 Frontière CIR-noyau vs arête enrichie SA

**C'est le point d'architecture le plus délicat de la section provenance.**

Le CIR produit par le moteur est **immuable**. Il contient uniquement ce que l'extraction a observé au moment de l'analyse : la relation, les nœuds, les attributs, et la provenance statique (URI du document, méthode d'extraction, version du modèle, timestamp de l'extraction). Ces champs ne changent jamais — ils décrivent un fait d'observation ponctuel.

L'arête stockée dans la mémoire de connaissance du SA est **mutable**. Elle enveloppe le CIR-noyau et y ajoute l'état dynamique géré par le SA : confirmation_count, last_confirmed_at, status dans le cycle de vie, fiabilité calculée de la source. Ces champs évoluent à chaque nouveau cycle d'observation.

La distinction est la suivante :

```
CIR-noyau (produit par le moteur, immuable)
┌────────────────────────────────────────────────────┐
│ source, target, relation, confidence               │
│ provenance:                                        │
│   document_uri       ← où, immuable                │
│   extraction_method  ← comment, immuable           │
│   model_version      ← quelle version, immuable    │
│   extracted_at       ← quand, immuable             │
└────────────────────────────────────────────────────┘

Arête enrichie SA (gérée par la mémoire, mutable)
┌────────────────────────────────────────────────────┐
│ cir: <CIR-noyau ci-dessus>                        │
│ status: NOUVELLE | ACTIVE | CONFIRMÉE | ...        │ ← SA
│ confirmation_count: 7                              │ ← SA
│ last_confirmed_at: 2026-09-25T...                  │ ← SA
│ source_reliability: 0.849                          │ ← SA (depuis registre)
└────────────────────────────────────────────────────┘
```

**Règle** : le moteur ne reçoit jamais l'arête enrichie — il ne connaît que le CIR-noyau. La mémoire de connaissance ne modifie jamais le CIR-noyau — elle le stocke tel quel et gère l'état mutable en dehors. Ce sont deux objets distincts avec des cycles de vie distincts.

---

## 11. Optimisation des recherches et du raisonnement sur les graphes

### 11.1 Problème de fond — état actuel

Toutes les opérations de la mémoire de connaissance actuelle sont O(n_edges) :

| Opération | Implémentation actuelle | Coût |
|---|---|---|
| `find_causes(keyword)` | Scan linéaire, match word-boundary sur tous les labels | O(n_edges) |
| `find_effects(keyword)` | Idem | O(n_edges) |
| `find_path(a, b)` | BFS non pondéré — chemin le plus court en sauts | O(V + E) |
| `CYCLES?` | Parcours complet du graphe | O(V + E) |
| `GAPS?` | Parcours complet | O(V + E) |

Sur un graphe de 100 000 arêtes — objectif réaliste pour un domaine CTI actif — chaque requête lit 100 000 entrées. Ce n'est pas un problème de vitesse acceptable en production, ni pour un agent en boucle continue.

De plus, le BFS actuel retourne le chemin **le plus court en sauts**, pas le chemin **le plus fiable**. Un chemin A→B→C→D à 0.90 de confiance par arête (confiance finale 0.73) est meilleur qu'un chemin direct A→D à 0.60. Le moteur ne fait pas cette distinction aujourd'hui.

### 11.2 Recommandation — quatre optimisations dans l'ordre

Les optimisations sont présentées dans l'ordre où elles doivent être implémentées. Chacune est indépendante de la suivante, mais chacune maximise l'impact des suivantes.

---

**Optimisation 1 — Partitionnement par domaine** *(priorité absolue)*

C'est la décision architecturale qui conditionne tout le reste. La mémoire de connaissance n'est pas un graphe global unique — c'est un ensemble de graphes partitionnés par domaine.

```
mémoire/
  domaines/
    cybersecurity_cti/    ← graphe CTI isolé
    conformite_rgpd/      ← graphe conformité isolé
    code_audit/           ← graphe code isolé
  registre_sources.json   ← partagé entre domaines
```

Une requête dans le domaine CTI ne traverse jamais le graphe conformité. Un graphe de 100 000 arêtes réparti en 10 domaines = 10 graphes de 10 000 arêtes. Toutes les optimisations suivantes opèrent sur le graphe de domaine, pas sur le graphe global — leur gain est donc multiplié par le facteur de partitionnement.

**Coût d'implémentation** : faible — structure de répertoires + routage des requêtes par domaine.  
**Impact** : multiplicatif sur toutes les opérations suivantes.

---

**Optimisation 2 — Index inversé sur labels** *(vitesse)*

Toute requête commence par retrouver un nœud par son label. C'est le bottleneck de chaque opération. L'index inversé élimine le scan linéaire.

```
label_index:  dict[str, list[node_id]]   ← mot complet
tokens_index: dict[str, list[node_id]]   ← tokens individuels pour match partiel
```

Construction O(n_nodes) une seule fois à l'initialisation. Mise à jour O(1) à chaque insertion. `find_causes("ransomware")` passe de O(n_edges) à O(k) où k est le nombre d'arêtes adjacentes au nœud trouvé — typiquement quelques dizaines, jamais des milliers.

Index complémentaires à construire simultanément :

```
domain_index:    dict[domain, set[edge_id]]      ← requêtes isolées par domaine
relation_index:  dict[RelationType, list[edge_id]]  ← filtrage par type de relation
confidence_tiers: dict[str, list[edge_id]]       ← "high"≥0.85, "medium"≥0.65, "low"<0.65
```

**Coût d'implémentation** : faible.  
**Impact** : immédiat sur toutes les requêtes — gain typique ×10 à ×1000 selon la taille du graphe.

---

**Optimisation 3 — Dijkstra avec poids `-log(confidence)`** *(qualité du raisonnement)*

C'est la seule optimisation qui améliore la **qualité** des réponses, pas seulement leur vitesse. Elle corrige un défaut de raisonnement du moteur actuel.

**Problème** : le BFS trouve le chemin le plus court en nombre de sauts. Ce n'est pas le chemin le plus fiable.

**Principe** : la confiance d'un chemin se propage multiplicativement.

```
conf(A → B → C) = conf(A→B) × conf(B→C)
```

Un chemin de 3 arêtes à 0.90 chacune (conf finale 0.73) est plus fiable qu'un chemin direct de 1 arête à 0.60. Le BFS choisit le second. Dijkstra choisit le premier.

**Reformulation** : trouver le chemin de produit de confiances maximal ≡ minimiser la somme de `-log(confidence)` — c'est un problème de plus court chemin pondéré, résolu par Dijkstra en O((V + E) log V).

```
poids(arête) = -log(confidence)   # transforme max-produit en min-somme
Dijkstra sur poids → chemin de confiance maximale
```

**Extension** — BFS bidirectionnel pour les chemins longs : partir simultanément de A et de B, terminer quand les deux fronts se rencontrent. Pour un graphe de diamètre D et facteur de branchement b, le BFS standard explore O(b^D) nœuds ; le BFS bidirectionnel explore O(2b^(D/2)) — gain exponentiel sur les longues chaînes causales.

**Coût d'implémentation** : moyen.  
**Impact** : qualité des réponses CHAIN, abductif, et chemin de preuve dans `validate()`.

---

**Optimisation 4 — Cache LRU avec invalidation ciblée** *(production)*

Les mêmes requêtes reviennent sur les mêmes nœuds hubs. Un cache naïf vidé à chaque insertion est inutile sur un graphe vivant. La stratégie correcte est l'invalidation par nœud touché.

```
query_cache:        dict[(keyword, relation, domain), result]
node_to_cache_keys: dict[node_id, set[cache_key]]
```

Quand une arête (src, dst) est insérée, seules les entrées du cache référençant src ou dst sont invalidées. Toutes les autres entrées restent valides. Les requêtes répétées sur des nœuds stables sont servies en O(1).

**Coût d'implémentation** : moyen.  
**Impact** : décisif pour un agent en boucle continue où les mêmes concepts sont interrogés fréquemment.

### 11.3 Ce que l'on ne fait pas maintenant

Deux optimisations sont documentées ici mais **explicitement reportées** après les quatre fondations :

**Index sémantique (fastText + FAISS)** : projeter les labels de nœuds dans un espace vectoriel pour la recherche floue (synonymes, variantes morphologiques). Pertinent quand les labels sont très hétérogènes ou quand la couverture lexicale de l'index inversé est insuffisante. Seuil de déclenchement : graphe > 500 000 arêtes par domaine, ou taux de requêtes sans résultat > 20 %.

**Graphe hiérarchique (coarse-to-fine)** : maintenir un graphe résumé (un nœud par cluster thématique) en parallèle du graphe détaillé. Les requêtes inter-domaines utilisent le résumé d'abord, puis raffinent dans les clusters concernés. Pertinent pour les requêtes cross-domaine sur des graphes de plusieurs millions d'arêtes. En deçà, la complexité d'implémentation ne se justifie pas.

### 11.4 Ordre d'implémentation et impact attendu

```
Optimisation              Coût    Impact            Prérequis
──────────────────────────────────────────────────────────────
1. Partitionnement domaine  Faible  Multiplicatif     Aucun
2. Index inversé labels     Faible  O(n)→O(k) requêtes  §1
3. Dijkstra conf. max       Moyen   Qualité raisonnement §2
4. Cache LRU ciblé          Moyen   Production stable    §2
──────────────────────────────────────────────────────────────
Reporté : index sémantique  Élevé   >500k arêtes/domaine §1-4
Reporté : graphe hiérarchique Élevé >1M arêtes cross-dom §1-4
```

Les quatre optimisations fondamentales sont suffisantes pour tenir 500 000 arêtes par domaine avec des temps de réponse inférieurs à 50 ms. Au-delà, les optimisations reportées deviennent nécessaires.

---

## 12. Communication et protocoles

### 12.1 Le problème actuel

Le moteur n'expose qu'une interface CLI. Pour intégrer GCN dans un système multi-agents ou un pipeline LLM, il faut construire un pont manuel (subprocess + parsing de sortie texte). C'est fragile, non scalable, et incompatible avec un paradigme agentique.

### 12.2 Interface de service

L'agent Hermès doit être exposé comme un **service**, pas comme un CLI. Concrètement, une API HTTP légère (FastAPI autour de `GCNEngine` + `CausalGraph`) avec les endpoints suivants :

| Endpoint | Description |
|---|---|
| `POST /analyze` | Texte → CIR JSON |
| `POST /ingest` | CIR JSON → intégration dans le graphe (avec provenance) |
| `GET /query/causes/{concept}` | Causes d'un concept, sourcées |
| `GET /query/effects/{concept}` | Effets d'un concept, sourcées |
| `GET /query/chain/{from}/{to}` | Chaîne causale entre deux concepts |
| `POST /validate` | Claim → CONFIRMED / UNCONFIRMED / CONTRADICTED |
| `GET /status` | État du graphe (densité, staleness, conflits ouverts) |
| `POST /directive` | Soumettre ou mettre à jour les directives de l'agent |
| `GET /alerts` | Alertes actives déclenchées par le moniteur |

### 12.3 Format de sortie adaptatif

Le même raisonnement doit produire des sorties différentes selon le destinataire :

- **LLM** : JSON structuré avec faits, sources, confiances — compact, sans prose. Le LLM lit, il ne reformule pas.
- **Humain** : texte verbalisé (`ReferenceDecoder`), citations en fin de paragraphe, niveau de confiance en prose.
- **Autre agent** : CIR complet avec provenance — protocole machine à machine.
- **Système d'alerte** : payload structuré avec niveau de sévérité, condition déclenchante, et graphe partiel concerné.

### 12.4 Protocole inter-agents

Dans un système multi-agents, le CIR est le **protocole de communication**. Un agent producteur extrait un CIR depuis une source et le transmet. Un agent consommateur l'intègre dans son graphe. La règle de traçabilité s'applique à chaque transfert : la provenance suit le CIR.

Un CIR transmis entre agents doit porter :
- L'identifiant de l'agent producteur
- Son niveau d'autorité dans le système (qui peut écrire quoi dans le graphe)
- La provenance complète de chaque arête

### 12.5 Sécurité inter-agents et anti-poisoning

**Le problème** : dans un système multi-agents, un agent malveillant ou compromis peut injecter des CIRs forgés pour polluer la mémoire de connaissance d'autres agents (graph poisoning). C'est particulièrement critique dans les domaines CTI où la manipulation de la connaissance causale peut induire de fausses alertes ou masquer de vraies menaces.

**Trois mécanismes de protection** :

**Authentification des agents producteurs**
Chaque CIR transmis est signé par l'agent producteur (signature HMAC ou asymétrique selon le niveau de confiance requis). L'agent consommateur vérifie la signature avant toute ingestion. Un CIR non signé ou à signature invalide est rejeté sans être inspecté.

**Niveaux d'autorité d'écriture**
La directive définit quels agents pairs sont autorisés à écrire dans quels domaines :

```json
"trusted_agents": [
  { "agent_id": "gcn-agent-osint-001", "domains": ["cybersecurity_cti"],
    "max_confidence_accepted": 0.80 },
  { "agent_id": "gcn-agent-legal-001", "domains": ["conformite_rgpd"],
    "max_confidence_accepted": 1.0 }
]
```

Un agent CTI ne peut pas écrire dans le domaine conformité. La confiance acceptée d'un agent pair est plafonnée — même un agent `trusted` ne peut pas injecter des relations à conf 0.99 si la directive dit max 0.80.

**Quarantaine des CIRs entrants**
Tout CIR reçu d'un agent pair est d'abord mis en statut NOUVELLE avec un flag `origin: "agent_push"`. Il ne passe en ACTIVE qu'après confirmation par au moins une **source indépendante non-agent** (adaptateur http_feed, local_files, ou autre adaptateur non `agent_push`) ou après validation manuelle si le domaine est critique.

**Définition d'indépendance** : deux sources sont indépendantes si elles utilisent des adaptateurs de types différents. Deux agents du même opérateur utilisant tous deux `agent_push` ne sont pas indépendants — ils partagent un vecteur de compromission commun. La confirmation inter-agents ne compte pas comme confirmation indépendante.

**Anti-Sybil** : une attaque Sybil consiste à faire confirmer une relation poisonnée par N agents complices se confirmant mutuellement. La contre-mesure est structurelle : la sortie de quarantaine exige obligatoirement au moins une confirmation depuis un adaptateur non-agent (`http_feed`, `local_files`, etc.). Aucun nombre de confirmations `agent_push` ne peut sortir une arête de quarantaine sans cette confirmation externe. La diversité d'adaptateurs est ainsi la garantie de l'indépendance, pas le nombre de confirmants.

**Rotation des clés** : les clés de signature des agents sont liées à leur identifiant dans la directive. Une clé compromise est révoquée en supprimant ou modifiant l'entrée `trusted_agents` de la directive — toutes les arêtes provenant de cet agent depuis la date de compromission supposée sont reclassées EN RÉVISION automatiquement.

---

## 13. Directives et objectifs

### 13.1 Pourquoi les directives sont indispensables

Sans directive, un agent n'est pas un agent — c'est un système qui réagit sans but. La directive est ce qui transforme un moteur de raisonnement en acteur orienté. Elle définit ce que l'agent surveille, ce qu'il accepte, ce qu'il rejette, quand il agit.

### 13.2 Principe fondamental : la couche externe n'est pas hardcodée

La couche externe — tout ce qui arrive de l'extérieur du système — n'est jamais définie dans le code. Elle est entièrement décrite par la directive. Le SA ne "sait" pas à la compilation qu'il y a des flux RSS, des APIs, des fichiers ou des agents pairs. Il sait seulement comment recevoir des données via un adaptateur générique et comment interpréter ce qu'une directive lui demande de surveiller.

Concrètement : un adaptateur de source est un composant interchangeable qui transforme n'importe quel input externe en texte brut (ou CIR déjà formé) à destination du SA. La directive référence l'adaptateur et sa configuration — elle ne hardcode pas le type de source dans la logique agentique.

Ce principe garantit que le SA est **agnostique des sources**. Ajouter un nouveau type de source (un nouveau protocole, un nouveau format de log, une nouvelle API) ne modifie pas la logique agentique — cela ajoute un adaptateur et une entrée de directive.

### 13.3 Structure d'une directive

Une directive est une spécification structurée (JSON ou YAML) qui définit :

```json
{
  "agent_id": "gcn-agent-cti-001",
  "domain": "cybersecurity_cti",
  "taxonomy_dir": "taxonomies/cyber/",

  "sources": [
    {
      "adapter": "http_feed",
      "config": { "url": "https://nvd.nist.gov/feeds/json/cve/", "interval_seconds": 3600 }
    },
    {
      "adapter": "local_files",
      "config": { "path": "/data/reports/", "glob": "*.txt", "watch": true }
    },
    {
      "adapter": "agent_push",
      "config": { "endpoint": "/ingest", "trusted_agents": ["gcn-agent-osint-001"] }
    }
  ],

  "quality": {
    "min_confidence": 0.70,
    "min_extraction_method": "symbolic_rust",
    "min_confirmation_count": 1
  },
  "staleness": {
    "stale_after_days": 30,
    "invalidate_after_days": 90
  },
  "monitoring": {
    "watch_cycles": true,
    "watch_gaps_in_chains": ["exploitation_chain", "persistence_chain"],
    "alert_on_new_relation_involving": ["CVE-2024-.*", "ransomware"]
  },
  "actions": {
    "allowed": ["alert", "validate", "report", "ingest"],
    "forbidden": ["modify_existing_relation", "delete_node"]
  },
  "output": {
    "default_format": "json",
    "alert_endpoint": "https://siem.internal/api/gcn-alerts",
    "report_schedule": "daily"
  }
}
```

Le champ `adapter` désigne un composant interchangeable enregistré dans le SA — pas un type hardcodé. Ajouter un nouveau type de source revient à écrire un adaptateur et à le référencer dans une directive.

### 13.4 Ce que les directives contrôlent

**Périmètre** : le domaine et les sources définissent ce que l'agent surveille. Tout ce qui est hors périmètre est ignoré, même si c'est causalement intéressant.

**Qualité d'admission** : les seuils de qualité définissent ce qui entre dans le graphe. Un CIR silver non re-confirmé en dessous de 0.70 de confiance ne passe pas.

**Temporalité** : les paramètres de staleness définissent le rythme de vieillissement de la connaissance dans ce domaine. En CTI, une CVE non confirmée pendant 30 jours devient stale. En droit constitutionnel, une jurisprudence non re-confirmée pendant 5 ans reste valide.

**Actions autorisées** : l'agent ne peut faire que ce qui est dans la liste `allowed`. Un agent en mode lecture seule ne peut pas écrire dans le graphe. Un agent de validation ne peut qu'alerter et valider, pas ingérer.

---

## 14. Analyse des écarts

### 14.1 Tableau de synthèse

La colonne **Périmètre** indique si la capacité appartient au **moteur (noyau)** ou au **système agentique (SA)**.

| Capacité | Périmètre | État actuel | Ce qui manque |
|---|---|---|---|
| Extraction causale FR/EN | Moteur | ✅ Fonctionnel | Coréférence inter-phrases |
| Extraction depuis code | Moteur | ✅ Fonctionnel | Rien de bloquant |
| Pearl niveau 1–3 | Moteur | ✅ Implémenté | — |
| CYCLES / GAPS | Moteur | ✅ Partiel | GAPS orienté objectif |
| Verbalisation CIR → texte | Moteur | ✅ Fonctionnel | TrainableDecoder non entraîné |
| Chaîne de données autonome | Moteur | ✅ Fonctionnel | — |
| Provenance immuable dans le CIR | Moteur | ⚠ Partielle | 4 champs immuables seulement — statut/count dans SA (§10.9) |
| Raisonnement abductif | Moteur | ❌ Absent | Algorithme de scoring d'hypothèses |
| Raisonnement temporel | Moteur | ❌ Absent | Exploitation de temporal_index |
| Raisonnement multi-échelle | Moteur | ❌ Absent | Hiérarchie de nœuds, agrégation |
| Raisonnement adversarial | Moteur | ❌ Absent | Centralité causale, points de fragilité |
| Raisonnement normatif | Moteur | ❌ Absent | Opérateurs comparaison sous-domaines descriptif/normatif |
| Méta-raisonnement | Moteur | ❌ Absent | Métriques de densité, fiabilité réponse |
| Raisonnement par analogie | Moteur | ❌ Absent | Subgraph matching, transfert |
| Persistance incrémentale | SA | ❌ Absent | WAL, index inversé (§11) |
| Partitionnement par domaine + sous-domaines | SA | ❌ Absent | Graphes distincts, descriptif/normatif = sous-domaines (§9.6) |
| Index inversé sur labels | SA | ❌ Absent | O(n)→O(k), index tokens, relation, conf (§11.2 opt.2) |
| Dijkstra confiance max | SA | ❌ Absent | -log(conf) weights, BFS bidirectionnel (§11.2 opt.3) |
| Cache LRU ciblé | SA | ❌ Absent | Invalidation par nœud, O(1) requêtes répétées (§11.2 opt.4) |
| Registre de sources | SA | ❌ Absent | reliability_score calculé, trusted/monitored/suspect/blocked (§10.3) |
| Formule d'arbitrage des conflits | SA | ❌ Absent | score composite normalisé, règle quorum, anti-famine (§10.8) |
| Frontière CIR-noyau / arête enrichie SA | SA | ❌ Absent | État mutable hors CIR, deux objets distincts (§10.9) |
| Cycle de vie de la connaissance | SA | ❌ Absent | Décroissance T_domaine, statuts NOUVELLE→INVALIDÉE (§10.6) |
| Registre d'outils + politique de sélection | SA | ❌ Absent | 6 niveaux priorité, anti-famine, séquentiel (§2.3) |
| Registre d'adaptateurs | SA | ❌ Absent | Sources non hardcodées, extensibles par directive |
| Boucle de raisonnement | SA | ❌ Absent | Sélection d'outils selon situation, non hardcodée |
| API de service | SA | ❌ Absent | HTTP API, format adaptatif |
| Sécurité inter-agents | SA | ❌ Absent | Signature, autorité, quarantaine, anti-Sybil (§12.5) |
| Protocole inter-agents | SA | ❌ Absent | CIR avec provenance normalisée |
| Directives / objectifs | SA | ❌ Absent | Spécification structurée de mission |
| Moniteur / alertes | SA | ❌ Absent | Surveillance continue, conditions |
| Déduplication lexicale minimale | SA | ❌ Absent | Table d'alias par domaine — P1 (§16.4 note) |
| Déduplication sémantique complète | SA | ❌ Absent | Résolution d'entités embeddings — P4 |
| SLOs et benchmark agentique | SA | ❌ Absent | Définis en §16.7, à implémenter avant P3 |

### 14.2 Ce qui est solide et doit être préservé

- **Le CIR** : format pivot bien conçu, extensible, sérialisable. Ne pas le modifier, l'enrichir.
- **La séparation moteur / outils** : le moteur ne lit pas de YAML, ne dépend pas de spaCy. Cette contrainte est saine et doit être maintenue.
- **Le raisonnement Pearl** : les trois niveaux sont correctement implémentés. Ils sont le socle sur lequel construire.
- **La chaîne de données** : scraping → annotation → gates → fusion est fonctionnelle. Elle alimente le modèle ML.
- **L'architecture JSON-native** : tout est sérialisable, tout est interopérable. C'est la bonne fondation pour une API.

### 14.3 Ce qui est le plus urgent

Deux blocants qui rendent tout le reste inopérant dans un contexte agentique :

1. **L'absence de boucle agentique** : sans elle, aucune des capacités avancées ne peut s'exprimer. L'agent ne peut pas surveiller, agir, ni apprendre.

2. **L'absence de provenance complète** : sans URI, méthode, et timestamp, la règle fondamentale "les CIRs ont des sources" ne peut pas être vérifiée ni auditée.

---

## 15. Architecture cible

### 15.1 Vue d'ensemble

La frontière entre le moteur (noyau) et le système agentique est matérialisée par le CIR. Le moteur produit et consomme des CIRs. Le système agentique orchestre, persiste, surveille et communique.

```
╔═════════════════════════════════════════════════════════════════════╗
║                    COUCHE EXTERNE (non hardcodée)                  ║
║  Tout ce qui vient de l'extérieur — types définis par directive,   ║
║  jamais en dur : flux, APIs, fichiers, agents, interfaces humaines  ║
╚══════════════════════════════╤══════════════════════════════════════╝
                               │
                               ▼
╔═════════════════════════════════════════════════════════════════════╗
║                    SYSTÈME AGENTIQUE                               ║
║                                                                     ║
║   ┌─────────────────────────────────────────────────────────────┐  ║
║   │              BOUCLE DE RAISONNEMENT                         │  ║
║   │  Observer → Sélectionner outils → Exécuter → Mémoriser     │  ║
║   │  (non hardcodée — pilotée par directives + situation)       │  ║
║   └──────────────────────┬──────────────────────────────────────┘  ║
║                          │ invoque                                  ║
║                          ▼                                          ║
║   ┌─────────────────────────────────────────────────────────────┐  ║
║   │                   REGISTRE D'OUTILS                         │  ║
║   │  fetch()  extract()  ingest()  query_*()  validate()        │  ║
║   │  alert()  report()   counterfactual()  query_gaps()  …      │  ║
║   │                                                             │  ║
║   │   extract() / raisonnement  →  ┌──────────────────────┐    │  ║
║   │                                │   MOTEUR (NOYAU)     │    │  ║
║   │                                │  Extraction CIR      │    │  ║
║   │                                │  Pearl 1–3 + ext.    │    │  ║
║   │                                │  Validation CIR      │    │  ║
║   │                                └──────────────────────┘    │  ║
║   └────────────────────────────┬────────────────────────────────┘  ║
║                                │ lit / écrit                        ║
║                                ▼                                    ║
║   ┌─────────────────────────────────────────────────────────────┐  ║
║   │              MÉMOIRE DE CONNAISSANCE                        │  ║
║   │                   (= intelligence)                          │  ║
║   │                                                             │  ║
║   │  Graphes causaux par domaine (WAL + index inversé)          │  ║
║   │  Provenance complète par arête                              │  ║
║   │  Cycle de vie : NOUVELLE→ACTIVE→CONFIRMÉE→STALE→INVALIDÉE  │  ║
║   │                                                             │  ║
║   │  Registre de sources                                        │  ║
║   │    confirmation_rate | contradiction_rate | reliability      │  ║
║   │    statut : trusted / monitored / suspect / blocked         │  ║
║   │    calculé depuis l'expérience — jamais déclaré à la main   │  ║
║   │                                                             │  ║
║   │  Ne se réinitialise jamais — accumule entre les cycles      │  ║
║   └─────────────────────────────────────────────────────────────┘  ║
║                                                                     ║
║   ┌─────────────────────────────────────────────────────────────┐  ║
║   │               REGISTRE D'ADAPTATEURS                       │  ║
║   │  Couche externe (non hardcodée) — définie par directive     │  ║
║   │  http_feed | local_files | agent_push | … (extensible)     │  ║
║   └─────────────────────────────────────────────────────────────┘  ║
║                                                                     ║
║   ┌─────────────────────────────────────────────────────────────┐  ║
║   │                    DIRECTIVES                               │  ║
║   │  Domaine | Sources | Qualité | Staleness | Actions | Sortie │  ║
║   └─────────────────────────────────────────────────────────────┘  ║
╚═════════════════════════════════════════════════════════════════════╝
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    LLMs / RAG         Humains          Autres agents
    (JSON)             (texte verbalisé) (CIR + provenance)
```

**Lecture** : la boucle de raisonnement n'est pas hardcodée — elle sélectionne des outils selon la situation. Le moteur est l'un de ces outils. La mémoire de connaissance est le pivot central que tous les outils partagent. Les adaptateurs sont le seul point de contact avec la couche externe, et leur liste n'est pas fixée dans le code.

### 15.2 La boucle de raisonnement (non hardcodée)

L'agent ne suit pas une séquence fixe câblée dans le code. À chaque cycle, il évalue la situation courante et sélectionne les outils appropriés parmi ceux disponibles. La séquence d'invocation varie selon ce qu'il observe.

```
┌─────────────────────────────────────────────────────────────────┐
│                     CYCLE DE RAISONNEMENT                       │
│                                                                 │
│  1. Observer la situation                                       │
│     - Nouvelles données disponibles via adaptateurs ?           │
│     - Requête entrante (validate, query) ?                      │
│     - Condition de monitoring déclenchée ?                      │
│     - Relations STALE dans la mémoire de connaissance ?         │
│                                                                 │
│  2. Sélectionner les outils pertinents                          │
│     selon la situation + les directives                         │
│                                                                 │
│     Nouvelles données    → fetch() → extract() → ingest()       │
│     Claim entrant        → validate()                           │
│     Question causale     → query_causes() / query_chain()       │
│     Cycle détecté        → alert()                              │
│     Gap dans chaîne      → query_gaps() → fetch() si priorité   │
│     Relations STALE      → décroissance, révision               │
│     Rien de notable      → aucun outil invoqué ce cycle         │
│                                                                 │
│  3. Exécuter les outils sélectionnés                            │
│     Chaque outil opère sur ou vers la mémoire de connaissance   │
│                                                                 │
│  4. Mettre à jour la mémoire de connaissance                    │
│     WAL append — jamais de snapshot complet                     │
│                                                                 │
│  5. Retourner au cycle suivant (délai défini dans la directive) │
└─────────────────────────────────────────────────────────────────┘
```

**La mémoire de connaissance** est le pivot de chaque cycle. Tous les outils lisent depuis elle ou écrivent vers elle. Elle persiste entre les cycles, entre les redémarrages, entre les tâches. Elle n'est jamais réinitialisée.

**Les outils sont les seuls points de contact** avec l'extérieur (adaptateurs, moteur, consommateurs). L'agent n'accède à rien directement — tout passe par un outil dont l'interface est définie et testable.

---

## 16. Priorités de développement

### 16.1 Critère de priorité

Les éléments sont classés par impact sur la viabilité agentique, pas par facilité d'implémentation.

### 16.2 Priorité 1 — Fondations indispensables (sans elles, rien ne fonctionne)

**P1.1 — Provenance immuable dans le CIR** *(extension du moteur)*
Étendre le format CIR pour inclure le bloc `provenance` avec les 4 champs immuables : `document_uri`, `extraction_method`, `model_version`, `extracted_at`. Conformément à §10.9, les champs d'état dynamique (`last_confirmed_at`, `confirmation_count`, `status`) n'entrent pas dans le CIR — ils appartiennent à l'arête enrichie SA. Sans cette provenance minimale, la règle "les CIRs ont des sources" ne peut pas être vérifiée.

**P1.2 — Persistance incrémentale (WAL)** *(système agentique)*
Remplacer la réécriture complète de `graph.json` par un WAL (`graph_delta.jsonl`) et un index inversé sur les labels. Appartient au SA — le moteur n'est pas concerné. Sans ceci, le SA ne peut pas tenir un flux continu.

**P1.3 — API HTTP minimale** *(système agentique)*
Exposer `analyze`, `ingest`, `validate`, `query/causes`, `query/effects` via FastAPI. Appartient au SA — c'est la couche de communication autour du moteur, pas dans le moteur.

### 16.3 Priorité 2 — Intelligence agentique de base *(système agentique)*

Tout le contenu de cette priorité appartient au système agentique. Le moteur n'est pas modifié — il est utilisé par le SA.

**P2.1 — Cycle de vie de la connaissance**
Implémenter les statuts (NOUVELLE, ACTIVE, CONFIRMÉE, STALE, INVALIDÉE), la décroissance temporelle, et la résolution de conflits par arbitrage multi-sources. Géré par le SA autour du graphe persistant.

**P2.2 — Boucle agentique**
Implémenter la boucle PERCEVOIR → ÉVALUER → RAISONNER → PLANIFIER → AGIR → METTRE À JOUR, pilotée par un système de directives. Le SA invoque le moteur à l'étape PERCEVOIR (extraction) et RAISONNER (Pearl).

**P2.3 — Directives structurées**
Implémenter le format de directive (domaine, sources, qualité, staleness, monitoring, actions autorisées). Sans directive, la boucle agentique n'a pas d'objectif.

**P2.4 — Moniteur et alertes**
Implémenter la surveillance continue des conditions définies dans les directives : CYCLES, GAPS ciblés, relations STALE critiques, conflits non résolus.

### 16.4 Priorité 3 — Extensions du moteur (raisonnements avancés) *(moteur)*

Ces capacités étendent le moteur lui-même. Elles sont exposées via l'API du SA une fois implémentées.

**P3.1 — Raisonnement abductif**
Scoring des hypothèses causales pondéré par confiance et fréquence. Requiert P1.2 (index dans SA) et P2.1 (confiance à jour dans SA).

**P3.2 — Raisonnement temporel**
Exploitation du champ `temporal_index` dans les requêtes Pearl. Requiert P1.1 (provenance avec timestamp dans le CIR).

**P3.3 — Méta-raisonnement**
Métriques de densité et de couverture du graphe, estimation de la fiabilité des réponses. Requiert P2.1 (cycle de vie dans SA).

**P3.4 — Raisonnement normatif**
Opérateurs de comparaison descriptif/prescriptif. Conformément à §9.6, il ne s'agit pas de deux graphes parallèles mais de deux **sous-domaines** au sein du même domaine — `descriptif` (extrait des faits) et `normatif` (extrait des règles). Le moteur fournit les opérateurs de comparaison ; le SA gère les deux sous-domaines et expose les écarts. Dépend de la déduplication lexicale (P1.5) pour que les nœuds des deux sous-domaines soient comparables.

### 16.5 Priorité 4 — Extensions avancées du moteur *(moteur)*

**P4.1 — Raisonnement multi-échelle** : hiérarchie de nœuds, agrégation, décomposition.
**P4.2 — Raisonnement adversarial** : centralité causale, points de fragilité.
**P4.3 — Raisonnement par analogie** : subgraph matching, transfert inter-domaines.
**P4.4 — Déduplication sémantique** *(SA)* : résolution d'entités complète (embeddings + clustering) — appartient au SA.

**Note sur la déduplication lexicale minimale** : la déduplication sémantique complète est en P4, mais une déduplication lexicale minimale (normalisation des labels + table d'alias) doit être en P1. Sans elle, l'index inversé (P1.2) et Dijkstra (P1.3) opèrent sur un graphe fragmenté où `"auth"`, `"login"` et `"authentification"` sont trois nœuds distincts au lieu d'un seul. La déduplication lexicale ne nécessite pas d'embeddings — une table d'alias par domaine (définie dans la directive ou dans la taxonomie) suffit pour les cas les plus fréquents.

### 16.6 Ce qui ne sera pas fait

- **Raisonnement général** : GCN restera un moteur causal, pas un système de raisonnement général. Il ne fait pas d'inférences non causales.
- **Génération libre** : la verbalisation reste déterministe (ReferenceDecoder). Pas de génération LLM intégrée dans le moteur.
- **Décisions autonomes** : l'agent produit des analyses et des alertes. Les décisions (sanctions, actions légales, actions critiques) restent humaines.
- **Multimodalité** : texte et code uniquement. Pas de traitement d'images, de tableaux ou de figures.

### 16.7 Métriques de succès et SLOs

Aucun développement de P1 à P4 n'est considéré terminé sans mesure. Les SLOs suivants définissent le seuil d'acceptabilité par phase.

**Phase P1 — Fondations**

| Métrique | Seuil | Méthode de mesure |
|---|---|---|
| Latence `validate()` p99 | < 200 ms sur graphe 50k arêtes | Benchmark sur corpus CTI synthétique |
| Latence `query_causes()` p99 | < 50 ms sur graphe 100k arêtes | Idem |
| Taux CONFIRMED correct sur claims vrais | ≥ 0.85 | Évaluation sur 200 claims annotés manuellement |
| Taux faux UNCONFIRMED (claim vrai absent du graphe) | ≤ 0.10 | Idem (mesure de couverture) |
| Intégrité WAL après crash simulé | 0 perte de relation | Test d'injection de faute |

**Phase P2 — Boucle agentique**

| Métrique | Seuil | Méthode |
|---|---|---|
| Temps de cycle à vide (no-op) | < 10 ms | Profiling boucle |
| Taux d'alertes correctes sur corpus étiqueté | ≥ 0.80 précision, ≥ 0.75 rappel | Corpus d'événements CTI avec ground truth |
| Stabilité mémoire sur 72h de flux continu | Pas de fuite mémoire, croissance WAL linéaire | Test de charge |

**Phase P3 — Raisonnements avancés**

| Métrique | Seuil | Méthode |
|---|---|---|
| Qualité CHAIN Dijkstra vs BFS | Confiance moyenne +15 % sur 500 requêtes | Comparaison A/B sur corpus |
| Rappel abductif (trouver la vraie cause) | ≥ 0.70 sur corpus caché | Évaluation sur dataset CTI avec causes connues |
| Couverture méta-raisonnement | Taux de réponses avec fiabilité estimée correcte ≥ 0.80 | Annotation manuelle d'un échantillon |

**Benchmark agentique MVP (à définir avant P3)**

Un benchmark dédié doit être construit avant d'entrer en P3. Il doit couvrir :
- Un corpus de 1000 documents CTI avec ground truth causal
- 200 claims avec verdicts de référence (CONFIRMED / UNCONFIRMED / CONTRADICTED)
- 50 chaînes causales avec le chemin de référence connu
- Un scénario de flux continu sur 24h avec injection d'anomalies

---

## Conclusion

Le moteur GCN est le noyau. Il extrait des CIRs depuis du texte et du code, raisonne causalement via Pearl, et valide la structure des graphes. C'est un composant stable, focalisé, dont les fondations sont solides : CIR bien conçu, Pearl 1–3 correctement implémenté, chaîne de données fonctionnelle.

Ce noyau ne sera jamais un agent, et ce n'est pas son rôle. Le système agentique est la couche construite autour de lui : c'est elle qui persiste, surveille, orchestre, communique et décide. Elle utilise le moteur comme outil de calcul causal, sans lui déléguer ses responsabilités d'orchestration.

Les deux évolutions à mener en parallèle sont donc distinctes :

**Pour le moteur** : enrichir le format CIR avec la provenance complète, puis étendre progressivement les capacités de raisonnement (abductif, temporel, normatif, multi-échelle). Ces extensions restent dans le périmètre du noyau.

**Pour le système agentique** : construire de zéro la couche d'intelligence opérationnelle — mémoire de connaissance persistante, registre d'outils, boucle de raisonnement non hardcodée, directives, registre d'adaptateurs, protocoles inter-agents. Ces développements n'entrent pas dans le moteur.

Deux propriétés sont non négociables dans cette architecture :

1. **L'agent utilise des outils — il n'est pas hardcodé.** Sa logique de décision n'est pas une séquence fixe câblée dans le code. C'est ce qui le rend adaptable, testable, et extensible sans refactoring.

2. **L'agent a une mémoire de connaissance — pas une session.** La connaissance causale accumulée persiste et évolue. Elle n'est jamais réinitialisée. C'est ce qui lui permet de raisonner sur la durée, pas seulement sur une fenêtre de temps.

La séparation moteur / système agentique est la garantie que le noyau reste stable pendant que la couche d'intelligence évolue.

---

*Document produit le 2026-09-25. Fondé sur l'inspection directe de : discuss.py, index.py, graph_vecs.py, cli/session.py, verbalizer/instructions.py, engine.py, et l'ensemble des modules gcn-python. Toute affirmation sur le code est vérifiable dans les fichiers sources.*
