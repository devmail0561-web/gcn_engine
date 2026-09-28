# ÉTUDE : GCN — Moteur Causal
## État des lieux, limites structurelles et extensions requises

**Auteur** : Michel Tendeng
**Date** : 2026-09-25
**Statut** : Document d'étude — projet actuel (gcn-core + gcn-python)
**Périmètre** : Moteur uniquement (GCN-Core Rust + GCN-Python + GCN-Tools). Le SA (Agent Hermès) est un projet futur séparé.

---

## Avertissement préliminaire

Ce document est un état des lieux honnête. Il ne minimise pas les lacunes et ne sur-vend pas les capacités existantes. Chaque affirmation est fondée sur l'inspection directe du code source. Les points non vérifiables dans le code sont explicitement marqués comme tels. L'objectif est de fournir une base de conception rigoureuse, sans ambiguïté, pour orienter le développement du moteur.

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture fondamentale](#2-architecture-fondamentale)
3. [État des lieux — quatre couches](#3-état-des-lieux--quatre-couches)
4. [Capacités actuelles confirmées](#4-capacités-actuelles-confirmées)
5. [Limites structurelles internes au moteur](#5-limites-structurelles-internes-au-moteur)
6. [La contrainte fondamentale : les CIRs ont des sources](#6-la-contrainte-fondamentale--les-cirs-ont-des-sources)
7. [Extensions de raisonnement requises](#7-extensions-de-raisonnement-requises)
8. [Provenance complète dans le CIR](#8-provenance-complète-dans-le-cir)
9. [Analyse des écarts — moteur](#9-analyse-des-écarts--moteur)
10. [Priorités de développement — moteur](#10-priorités-de-développement--moteur)

---

## 1. Contexte et objectif

### 1.1 Origine

GCN (Graph Causal Network) a été conçu comme un moteur d'extraction et de raisonnement causal. Son architecture repose sur deux implémentations complémentaires : un frontend symbolique en Rust (gcn-core, 9 crates) qui analyse du texte FR/EN/code via des taxonomies YAML, et un pipeline d'apprentissage en Python (gcn-python) qui apprend les représentations causales depuis des données annotées. Les deux convergent vers le même format pivot : le **CIR (Causal Intermediary Representation)**, un graphe JSON sérialisable décrivant des nœuds causaux et des relations typées entre eux.

### 1.2 Ce que cette étude adresse

Au fil du développement, le moteur a été enrichi de capacités de raisonnement (Pearl niveaux 1–3), de session (gcn-discuss), d'indexation de corpus (gcn-index), et de chaîne de données autonome (gcn-tools). Ces additions ont été construites dans le paradigme d'un **outil interactif** — un humain pose une question, le moteur répond, et la session se termine.

Cette étude cartographie l'état actuel du moteur, ses limites internes, et les extensions de raisonnement requises pour atteindre un niveau de qualité supérieur. Elle ne couvre pas le système agentique (SA), qui est un projet séparé.

### 1.3 Ce que ce document n'est pas

Ce document ne couvre pas le SA (système agentique — hors périmètre moteur). Ce n'est pas non plus un document marketing — les performances actuelles sont citées telles qu'elles sont mesurées, sans extrapolation. C'est un document de conception architecturale focalisé sur le moteur.

---

## 2. Architecture fondamentale

### 2.1 Ce qu'est le moteur

Le moteur GCN (gcn-core + gcn-python) est le **noyau** du système. Il est à l'agent ce que le noyau Linux est à un système d'exploitation : un composant stable, focalisé, à responsabilité unique. Il ne connaît pas les agents, les flux, les alertes, ni les directives. Il ne gère pas de sessions, ne surveille pas de sources, ne prend pas d'initiatives.

Le moteur est une **bibliothèque** : elle est invoquée, elle produit un résultat, elle s'arrête. Elle n'a pas d'état persistant, pas de boucle, pas d'opinion sur les sources.

**Le SA (Agent Hermès) est un projet séparé** — il n'est pas dans gcn-core, ni dans gcn-python, ni dans gcn-tools. C'est un nouveau projet Python indépendant qui importe gcn-python comme dépendance. Toute capacité qui relève de l'orchestration, de la persistance ou de la communication appartient au SA, jamais au moteur.

### 2.2 Responsabilités du moteur (noyau)

Le moteur fait exactement trois choses, et rien d'autre :

1. **Extraire des CIRs** depuis du texte (FR/EN) ou du code, par voie symbolique (Rust) ou par apprentissage (Python)
2. **Raisonner causalement** sur un graphe CIR donné (Pearl 1–3, BFS, CYCLES, GAPS)
3. **Valider** l'intégrité structurelle d'un CIR (types de nœuds, relations licites, spans 1-based)

Le moteur est **sans état persistant**. Il reçoit une entrée, produit une sortie, s'arrête. Il ne sait pas ce qui s'est passé avant lui ni ce qui se passera après.

Le moteur est **sans opinion sur les sources**. Il n'évalue pas la fiabilité d'une source, ne maintient pas de registre de confiance des producteurs, ne décide pas quoi accepter ou refuser.

Le moteur **n'a pas de boucle**. Il ne surveille rien, n'alerte pas, n'initie rien. Il est invoqué, il produit, il s'arrête.

### 2.3 Ce que cette distinction interdit dans le moteur

- Ajouter une boucle d'ingestion continue **dans** le moteur → appartient au SA
- Implémenter la décroissance temporelle **dans** `CausalGraph` → appartient au SA
- Exposer une API HTTP **depuis** gcn-python directement → appartient au SA
- Gérer des directives **dans** le moteur → appartient au SA

**Exception licite** : étendre le format CIR pour inclure la provenance (`ref`, `span`, `extraction_method`, `model_version`, `extracted_at`) est une modification du moteur, car le CIR est la frontière entre le noyau et le reste du système. Tout ce qui traverse cette frontière doit être défini dans le noyau.

### 2.4 Le CIR — format pivot et frontière du noyau

Le CIR est le format pivot de tout le système. Il décrit :

- **Nœuds** (7 types) : `etat`, `action`, `transition`, `processus`, `condition`, `entite`, `etat_systemique`
- **Arêtes** (11 relations + inverses) : `cause`, `enable`, `prevent`, `condition`, `concession`, `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency`
- **Attributs de nœud** : `type`, `label`, `token_span` (1-based), `origin`, `scope`, `temporal_index`, `modifiers`, `attributes`
- **Attributs d'arête** : `relation`, `confidence`, `explicit`, `negated`

Le CIR est conçu pour être **sérialisable en JSON**, **indépendant de la langue du texte source**, et **interopérable** entre les couches Rust et Python.

### 2.5 Invariant de traçabilité

Toute arête CIR produite par le moteur est extraite depuis un texte source réel. La confiance associée (`confidence`) mesure la certitude de l'extraction, pas une probabilité inventée. Ce principe de traçabilité est **non négociable** et constitue la différence fondamentale entre GCN et un LLM qui génère des relations causales sans preuve.

---

## 3. État des lieux — quatre couches

Le moteur repose sur quatre couches conceptuelles :

| Couche | Composant | Rôle |
|---|---|---|
| Représentation | CIR (JSON) | Format pivot universel entre toutes les couches |
| Extraction symbolique | gcn-core (Rust) | Texte FR/EN/code → CIR via taxonomies YAML |
| Extraction apprise | gcn-python (NumPy) | UD → CIR via MLP/R-GCN/GAT entraîné |
| Données | gcn-tools (scraper + annotateur) | Brut → JSONL → annotation → silver → fusion |

**Point important** : les taxonomies YAML sont exclusivement des ressources pour les outils d'extraction (gcn-core, gcn-tools). Le moteur Python ne lit jamais de YAML. Le moteur consomme uniquement du JSON. Cette séparation est une décision d'architecture délibérée et doit être préservée.

---

## 4. Capacités actuelles confirmées

Cette section décrit ce que le moteur **fait** aujourd'hui, vérifié dans le code source.

### 4.1 Extraction causale depuis texte FR/EN

Deux voies :

**Voie symbolique** (gcn-core Rust) : texte brut → CIR en une commande via `gcn analyze`. Couverture limitée aux taxonomies YAML versionnées. Qualité heuristique (~80–85 % selon les constantes de bridge.py:15–24, non mesuré indépendamment). La causalité implicite est couverte avec une confiance abaissée (0.5). Limite connue : pas de coréférence inter-phrases.

**Voie ML** (gcn-python) : séquence de tokens UD → CIR via le pipeline MLP/R-GCN/GAT. Nécessite un checkpoint entraîné. Performances mesurées — **les valeurs ne sont pas comparables entre échelles** (BENCHMARK.md §347–348) :

| Échelle / run | val edge_f1 | Note |
|---|---|---|
| §1a anciens (BENCHMARK.md:26,29) | 0.32–0.47 | Échelle §1, 4 runs retenus |
| §9a re-runs seed 7 (BENCHMARK.md:309–310) | 0.16–0.25 | Échelle §9, non comparable §1 |
| Calibration croisée (BENCHMARK.md:322–323) | 0.19–0.22 | Échelle §9 |
| model_v2 embeddings 300d | 0.23 | Échelle §10/§11 |

Ces intervalles reflètent des conditions d'entraînement différentes. Aucune comparaison directe entre échelles n'est valide.

Le pipeline inclut des word embeddings apprenables (WordEmbedding, compatible GloVe/FastText, Améliorations A et B : pooling des tokens de contenu NOUN/VERB/ADJ/PROPN/ADV, embeddings sujet/objet séparés). Cette couche fournit une **généralisation sémantique à l'extraction** : deux phrases de structure causale identique mais mots différents sont reconnues comme exprimant la même relation. Ce n'est pas du pattern matching syntaxique pur — le sens des mots est approximé via l'espace vectoriel.

**Voie bridge** (frontend/bridge.py) : texte brut → sous-processus gcn-cli → représentations UD heuristiques. Qualité dégradée connue (is_negative à 0 %, morphologie imprécise). C'est une solution de transition, pas une solution de production.

### 4.2 Extraction depuis code

gcn-frontend-code (31 tests : 30 dans tests/integration_code.rs + 1 dans src/lib.rs) parse Python/Rust/JS via tree-sitter et produit des CIR isomorphes au langage naturel. `if x < seuil: réduire(y)` produit la même structure causale que "si x dépasse le seuil, on réduit y". C'est le **seul domaine immédiatement opérationnel sans annotation métier** : la causalité dans le code est structurelle, pas lexicale.

### 4.3 Raisonnement causal formel (Pearl)

Implémenté dans les modules Rust et partiellement exposé en Python via `InstructionHandler` :

- **Pearl niveau 1** (association) : `WHY(X)` → ancêtres, `WHAT(X)` → descendants, `CHAIN(A, B)` → plus court chemin BFS
- **Pearl niveau 2** (intervention) : `DO(X)` — coupe les arêtes entrantes de X, propage les effets en aval
- **Pearl niveau 3** (contrefactuel) : `COUNTERFACTUAL(X)` — effets réels vs. monde sans X

Limite : ces trois niveaux couvrent un graphe supposé complet et correct. Ils ne raisonnent pas sur l'incertitude du graphe lui-même.

### 4.4 Analyse structurelle du graphe

- `CYCLES?` : détection et classification des boucles (positive/négative/oscillation)
- `GAPS?` : lacunes causales — filtre `temporal_gap.is_some()` + `unresolved.len() > 0` (query.rs:190–204). Les arêtes de type `CONCESSION`/`OPPOSITION` sont équivalentes à des gaps uniquement après propagation middleend (propagate.rs:17–26, edge.rs:24–27) ; ce n'est pas une équivalence directe.
- Hygiène (gcn-middleend, 17 tests) : arêtes pendantes, boucles réflexives, nœuds orphelins, violations d'ordre temporel — diagnostics Error/Warning

### 4.5 Session et persistance actuelle

`SessionStore` (gcn-python/src/gcn_python/cli/session.py) maintient en RAM : `CausalGraph` (nodes dict + edges list + adjacency), vecteurs enrichis par clé stable, historique JSONL. Persistance sur disque : `graph.json` (JSON complet), `session_vecs.npz` (NumPy compressed), `session_vecs.manifest.json` (index des clés vectorielles), `history.jsonl` (append-only).

### 4.6 Verbalisation

**Rôle** : transformer un CIR en texte naturel lisible — pas une linéarisation de la structure JSON, mais une phrase ou un paragraphe qui exprime le contenu causal de façon fluide.

**Pourquoi le CIR seul ne suffit pas** : le CIR contient les labels symboliques (chaînes de caractères) et les types de relations. Il ne contient pas la sémantique fine de chaque nœud dans son contexte. Cette sémantique est capturée dans les **vecteurs enrichis produits par l'encodeur + R-GCN** après le passage dans le pipeline. Un verbalizer correct doit donc recevoir **les deux** : la structure CIR (topologie, types, confidences) et les vecteurs enrichis (sens contextuel de chaque nœud).

**État actuel — trois composants, aucun ne réalise la textualisation complète** :

- `ReferenceDecoder` : linéarisation structurelle déterministe. Produit `NodeA [type] →[relation]→ NodeB [type] (85%)`. Ce n'est pas du texte naturel — c'est la structure du graphe imprimée en ASCII. Aucune ambiguïté possible, mais aucune fluidité non plus. Entrée : CIR dict uniquement.

- `LexicalConnectorAssembler` (Amélioration G2, composant actif recommandé) : apprend à sélectionner un connecteur lexical (`"donc"`, `"parce que"`, `"entraîne"`) pour chaque type de relation CIR. Produit `"src_text connecteur dst_text"`. Ce n'est pas une textualisation — c'est une sélection dans un vocabulaire de connecteurs fixe. Entrée : `relation_idx` uniquement (0–10). Le contenu sémantique des nœuds est ignoré.

- `TrainableDecoder` : déprécié. Était le seul composant à prendre en entrée les vecteurs enrichis du R-GCN `(N, d_in)` — architecture correcte pour une vraie textualisation. Déprécié parce que le signal d'entraînement (BLEU sur labels à 1 mot) était invalide, pas parce que l'architecture était fausse.

**Gap** : la textualisation naturelle d'un CIR — qui nécessite CIR + vecteurs enrichis en entrée — n'est pas implémentée dans l'état actuel du moteur. `ReferenceDecoder` couvre le besoin fonctionnel minimal (lisibilité pour l'humain), mais ne produit pas de texte naturel fluide.

### 4.7 Chaîne de données autonome

gcn-scrape (Wikipedia, HAL, arXiv, RSS, GitHub, web) → gcn-annotate (LLM, regex, spaCy/UD) → gates silver (v3–v16) → fusion → gcn-train. Le moteur peut produire son propre carburant d'entraînement.

---

## 5. Limites structurelles internes au moteur

Cette section décrit les lacunes **internes au moteur** — des choses que le moteur fait mal ou pas du tout dans le périmètre qui est le sien.

> **Note** : les absences de communication, directives, et boucle agentique appartiennent au SA (hors périmètre moteur) et ne figurent pas dans ce document.

### 5.1 Limite fondamentale : ce n'est pas un agent

Le moteur actuel est un **outil passif**. Il répond quand on l'invoque. Entre deux invocations, il n'existe pas — il n'observe rien, ne détecte rien, n'agit pas. Il n'a pas de boucle d'exécution propre, pas d'état persistant entre sessions (sauf si explicitement sauvegardé et rechargé), pas de notion d'objectif.

Cette limite n'est pas une nuance — c'est une différence de paradigme. Un outil répond à des requêtes. Un agent perçoit, raisonne, décide et agit de façon continue, orienté par des objectifs, sans attendre qu'on lui pose une question. Le moteur est l'outil — le SA est l'agent.

### 5.2 Mémoire naïve et non scalable

La structure de mémoire actuelle présente quatre défauts structurels :

**5.2.1 Persistance par snapshot complet**

`session.save()` réécrit `graph.json` et `session_vecs.npz` en intégralité à chaque appel. Sur un flux continu de 1000 phrases, cela produit 1000 réécritures complètes d'un fichier de taille croissante. Le seul composant correctement incrémental est `history.jsonl` (append-only).

**5.2.2 Aucun index sur les labels de nœuds**

`find_causes(keyword)` et `find_effects(keyword)` effectuent un scan linéaire O(n) sur la liste complète des arêtes, avec matching word-boundary sur les labels. Sur un graphe de 10 000 arêtes, chaque requête lit 10 000 entrées. Il n'existe aucun index inversé `label → [node_ids]`.

**5.2.3 Aucune résolution de conflits**

Quand deux sources extraient des relations contradictoires sur la même paire de nœuds (source A dit `cause`, source B dit `prevent`), les deux arêtes sont ajoutées au graphe sans signalement ni arbitrage. Le graphe accumule des contradictions silencieusement.

**5.2.4 Aucune décroissance temporelle**

La confiance d'une relation est figée à la valeur calculée au moment de l'extraction. Une relation extraite une seule fois il y a 6 mois, non re-confirmée, a la même confiance qu'une relation confirmée par 15 sources récentes. Il n'y a aucun mécanisme de vieillissement de la connaissance.

### 5.3 Raisonnement incomplet

**Note préliminaire** : la généralisation sémantique à l'extraction est déjà implémentée via les word embeddings (§4.1). Les lacunes décrites ci-dessous concernent uniquement le **raisonnement sur le graphe**, pas l'extraction.

Les trois niveaux de Pearl couvrent l'observation, l'intervention et le contrefactuel sur un graphe **supposé complet et correct**. Les raisonnements suivants sont absents :

**Raisonnement abductif** : Étant donné une observation (un effet constaté), remonter vers l'explication causale la plus plausible parmi plusieurs hypothèses. Pearl part de causes connues. L'abduction part de l'effet et cherche la cause — c'est le raisonnement diagnostique fondamental.

**Raisonnement temporel** : Le champ `temporal_index` existe dans le CIR mais n'est jamais utilisé dans le raisonnement. "A cause B avec un délai de 3 cycles" est traité identiquement à "A cause B instantanément". Les séquences, délais, et seuils temporels ne sont pas raisonnés.

**Raisonnement multi-échelle** : La même relation causale a des causes différentes selon le niveau d'abstraction. Ces échelles sont traitées comme un graphe plat sans hiérarchie.

**Raisonnement adversarial** : Quelle est la chaîne causale la plus fragile ? Quel nœud, s'il est supprimé ou manipulé, produit le plus grand effet sur le graphe ? Ce raisonnement sur la robustesse et les points de vulnérabilité du graphe est absent.

**Raisonnement normatif** : Le graphe décrit la causalité **observée** (descriptive). Il ne raisonne pas sur la causalité **prescrite** (normative) — ce qui devrait causer quoi selon une règle, un contrat, ou une norme. GAPS est une approximation structurelle, pas du raisonnement normatif.

**Méta-raisonnement** : L'agent ne raisonne pas sur la qualité de son propre modèle. Il ne sait pas si son graphe est assez dense pour répondre fiablement à une question donnée, ni si une réponse repose sur une seule source non confirmée ou sur vingt sources concordantes.

**Raisonnement par analogie** : La structure causale "vulnérabilité → exploitation → persistance → exfiltration" en CTI est isomorphe à "faille contractuelle → non-conformité → sanction → atteinte à la réputation" en droit. Le moteur ne reconnaît pas ces isomorphismes entre domaines et ne les exploite pas.

### 5.4 Provenance incomplète

Le moteur conserve le texte source d'une arête dans le 4-tuple de `CausalGraph.edges`. C'est insuffisant pour une provenance réelle. Il manque :

- **Référence au document source** (`ref`) : un pointeur vers le fichier, l'URL ou le chemin réseau d'où vient la phrase. GCN ne possède pas le document — il pointe vers lui. Le document vit là où il a toujours vécu.
- **Position dans le document** (`span`) : les offsets (début, fin) permettant de localiser la phrase source dans le document si un auditeur humain veut la lire. GCN ne stocke pas la phrase elle-même — c'est une duplication inutile.
- **Méthode d'extraction** : le CIR ne distingue pas si la relation a été extraite par le frontend Rust symbolique, par le modèle ML Python, ou par l'annotateur silver de gcn-tools — trois niveaux de confiance fondamentalement différents.
- **Version du modèle** : le `checkpoint_hash` existe dans le manifest des vecteurs mais n'est pas intégré dans le CIR lui-même.
- **Timestamp d'extraction** : il n'y a aucune trace de quand une relation a été extraite pour la première fois.

**Principe fondamental** : après extraction, le document n'est plus nécessaire au runtime. Toutes les opérations (raisonnement, validation) fonctionnent sur le graphe seul. La référence `ref + span` est une citation pour l'auditeur humain — pas une dépendance fonctionnelle. Si le document disparaît, la relation reste dans le graphe et reste opérationnelle ; elle devient seulement non vérifiable par un humain.

Sans provenance complète, il est impossible de répondre à la question la plus élémentaire d'un auditeur : *"d'où vient cette information ?"*.

---

## 6. La contrainte fondamentale : les CIRs ont des sources

### 6.1 La règle

Un CIR ne peut pas être inventé. Chaque nœud et chaque arête d'un CIR doit être tracé vers une source vérifiable : un document, une URL, un fichier de code, une ligne de log. C'est cette propriété qui distingue GCN d'un LLM générant des relations causales probables — GCN prouve, le LLM opine.

Cette règle s'applique à tous les acteurs du système : le moteur symbolique, le modèle ML, l'annotateur, et tout agent utilisant GCN. Aucun acteur ne peut écrire dans le graphe causal sans référencer une source.

### 6.2 Ce que ça implique pour l'architecture

**Pour l'ingestion** : personne ne peut créer de CIRs. On ne peut qu'extraire des CIRs depuis des sources analysées. Toute écriture dans le graphe est une extraction, jamais une déduction libre.

**Pour le raisonnement** : les relations déduites (A → C déduit de A → B et B → C) sont licites, mais elles doivent être marquées `derivation: "derived"` avec les deux sources qui les justifient. Les déductions sont des inférences tracées, pas des inventions.

**Pour la validation** : quand le système reçoit un claim, il doit trouver la source dans le graphe pour répondre CONFIRMÉ. S'il ne trouve pas de source, il répond NON CONFIRMÉ — pas FAUX. L'absence de preuve n'est pas une preuve d'absence.

**Pour la communication** : toute sortie incluant une affirmation causale doit inclure la source. Émettre "A cause B" sans citation est une violation du contrat de traçabilité.

### 6.3 Ce que ça interdit

- Synthétiser des relations causales depuis plusieurs CIRs sans tracer la déduction
- Augmenter la confiance d'une relation par inférence libre — seulement par re-confirmation depuis de nouvelles sources
- Supprimer une relation du graphe sans tracer la raison (source invalide, contradiction arbitrée)

---

## 7. Extensions de raisonnement requises

> Ces raisonnements sont des extensions du moteur. Certains nécessitent des données fournies par le SA (confiances à jour, cycle de vie de la connaissance) — ces dépendances sont notées explicitement.

### 7.1 Les trois niveaux de Pearl — socle nécessaire mais insuffisant

Pearl's ladder of causation est le fondement du raisonnement causal :

- **Niveau 1 — Association** : P(Y|X) — que se passe-t-il en présence de X ?
- **Niveau 2 — Intervention** : P(Y|do(X)) — que se passe-t-il si je force X ?
- **Niveau 3 — Contrefactuel** : P(Y_x|X', Y') — qu'aurait-il résulté si X avait été différent ?

Ces trois niveaux sont implémentés dans le moteur actuel. Ils sont nécessaires mais couvrent uniquement un graphe supposé complet et correct. Le moteur doit raisonner au-delà.

### 7.2 Raisonnement abductif

**Définition** : inférence à la meilleure explication. Étant donné un effet E observé, trouver la cause C la plus plausible parmi les causes candidates dans le graphe.

**Différence avec Pearl** : Pearl niveau 1 répond à "quelles sont les causes de X dans le graphe ?" L'abduction répond à "parmi toutes les causes possibles de X, laquelle explique le mieux l'observation ?" — en pondérant par la confiance, la fréquence, et la cohérence avec d'autres observations.

**Application concrète** : en CTI, un analyste observe une exfiltration de données. L'abduction interroge le graphe pour identifier le scénario d'attaque le plus probable parmi les chaînes connues qui aboutissent à l'exfiltration.

**Ce qui manque** : un algorithme de scoring des hypothèses causales pondéré par la confiance des arêtes et la fréquence de confirmation par les sources.

**Dépendance SA** : requiert des confiances à jour (cycle de vie CONFIRMÉE/STALE géré par le SA).

### 7.3 Raisonnement temporel

**Définition** : raisonnement sur l'ordre, la durée, et les délais dans les chaînes causales.

**Ce que le CIR a** : le champ `temporal_index` sur chaque nœud (entier ordinal) et la structure `TemporalRef`/`TemporalGap` dans le CIR.

**Ce qui est absent** : l'utilisation de ces champs dans le raisonnement. L'agent devrait pouvoir répondre à : "A cause B en combien de temps ?" ou "Si A se produit, B se produira avant ou après C ?" ou "Cette chaîne peut-elle boucler dans un délai T ?"

**Application concrète** : en conformité, une obligation légale a un délai. La chaîne "infraction → détection → signalement → sanction" a des délais réglementés. Le raisonnement temporel permet de vérifier si les contraintes de délai sont respectées.

**Dépendance SA** : requiert le timestamp `extracted_at` dans la provenance CIR (P1.1).

### 7.4 Raisonnement multi-échelle

**Définition** : la même relation causale s'exprime différemment à différents niveaux d'abstraction. Un moteur multi-échelle navigue entre ces niveaux, agrège des causes micro en causes macro, et décompose des causes macro en mécanismes micro.

**Ce qui manque** : une hiérarchie explicite entre nœuds (nœud macro agrégant plusieurs nœuds micro), des opérateurs d'agrégation et de décomposition, et une logique de navigation entre niveaux.

**Application concrète** : en analyse de risque systémique financier, la "contagion" est une cause macro. Sous-jacents : "liquidité insuffisante", "coûts de refinancement", "ventes forcées" — causes micro. Un moteur multi-échelle passe de l'une à l'autre selon la question posée.

### 7.5 Raisonnement adversarial

**Définition** : raisonnement sur la robustesse d'une chaîne causale face à des suppressions ou manipulations délibérées de nœuds ou d'arêtes.

**Questions adressées** :
- Quel nœud, s'il est supprimé, coupe le plus grand nombre de chaînes causales ?
- Quelle arête a le plus grand impact sur la propagation d'un effet ?
- Quels nœuds sont des points de défaillance unique (single points of failure) dans le graphe ?

**Ce qui manque** : des métriques de centralité causale (pondérée par la confiance et la fréquence des chemins), des algorithmes de recherche de points de vulnérabilité.

**Application concrète** : en CTI, identifier quel contrôle de sécurité, s'il est contourné, ouvre le plus grand nombre de vecteurs d'attaque.

### 7.6 Raisonnement normatif

**Définition** : distinguer la causalité **descriptive** (ce qui cause quoi dans les faits, extrait des sources) de la causalité **prescriptive** (ce qui devrait causer quoi selon les règles, normes, contrats).

**Ce qui existe** : le graphe descriptif (toutes les relations extraites de textes réels).

**Ce qui manque** : des opérateurs de comparaison entre graphe normatif et descriptif, et une mesure d'écart de conformité.

**Clarification sur la partition** : le normatif et le descriptif ne sont pas deux domaines distincts — ce sont deux **sous-domaines** au sein du même domaine. La partition se fait ainsi :

```
domaine: conformite_rgpd
  sous-domaine: descriptif   ← extrait de logs, code, rapports d'audit
  sous-domaine: normatif     ← extrait du texte du RGPD, guides CNIL
```

Les deux sous-domaines partagent le même espace de concepts (les nœuds réfèrent aux mêmes entités) mais ont des arêtes de sources différentes. L'opérateur de comparaison prend les deux sous-graphes en entrée et produit une liste d'écarts.

**Application concrète** : en audit RGPD, le graphe normatif dit "accès aux données personnelles doit activer journalisation". Le graphe descriptif dit "accès aux données personnelles active journalisation 73 % du temps". L'écart est une non-conformité prouvée, sourcée, quantifiée.

**Dépendance SA** : le SA gère les deux sous-domaines et expose les écarts via son API.

### 7.7 Méta-raisonnement

**Définition** : raisonnement sur la qualité et les limites du modèle causal lui-même.

**Questions adressées** :
- Mon graphe est-il assez dense pour répondre fiablement à cette question ?
- Cette réponse repose-t-elle sur une seule source non confirmée ?
- Existe-t-il des hypothèses alternatives que mon graphe ne distingue pas encore ?
- Quelle est la couverture de mon graphe sur ce domaine ?

**Ce qui manque** : des métriques de densité et de couverture du graphe par domaine, un mécanisme d'estimation de la fiabilité d'une réponse basé sur le nombre de sources et la diversité des chemins, une interface explicite pour communiquer l'incertitude épistémique.

**Dépendance SA** : requiert le cycle de vie des relations (CONFIRMÉE/STALE) géré par le SA.

### 7.8 Raisonnement par analogie structurelle

**Définition** : reconnaître qu'un pattern causal dans un domaine est isomorphe à un pattern dans un autre domaine, et transférer les inférences de l'un à l'autre.

**Exemple** : "vulnérabilité → exploitation → persistance → exfiltration" (CTI) est structurellement identique à "faille → déclencheur → maintien → impact" dans d'autres domaines. Un moteur qui reconnaît cet isomorphisme peut transférer les tactiques de détection et de mitigation.

**Ce qui manque** : un comparateur de structure de graphe (graph isomorphism / subgraph matching), un registre de patterns connus, un mécanisme de transfert inter-domaines.

---

## 8. Provenance complète dans le CIR

### 8.1 Structure de provenance requise

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
    "ref": "/home/michel/docs/rapport.pdf",
    "span": [10240, 10287],
    "extraction_method": "symbolic_rust",
    "model_version": "gcn-core-2.5.0",
    "extracted_at": "2026-09-25T14:32:00Z"
  }
}
```

`ref` est un pointeur vers le document source (chemin, URL, identifiant) — GCN ne possède pas le document. `span` sont les offsets byte permettant de localiser la phrase dans le document si un auditeur humain le demande. La phrase source n'est **pas** stockée dans le CIR — ce serait dupliquer le document entier relation par relation.

Ces 5 champs de provenance sont **immuables** — ils décrivent le fait d'observation ponctuel. Les champs `last_confirmed_at`, `confirmation_count`, et `status` sont gérés par le SA dans l'arête enrichie et ne figurent jamais dans le CIR-noyau.

Le champ `extraction_method` prend l'une des valeurs : `symbolic_rust`, `ml_python`, `silver_annotator`, `derived` (pour les déductions tracées).

### 8.2 Frontière CIR-noyau vs arête enrichie SA

**C'est le point d'architecture le plus délicat de la provenance.**

Le CIR produit par le moteur est **immuable**. Il contient uniquement ce que l'extraction a observé au moment de l'analyse : la relation, les nœuds, les attributs, et la provenance statique (`ref`, `span`, méthode d'extraction, version du modèle, timestamp de l'extraction). Ces champs ne changent jamais — ils décrivent un fait d'observation ponctuel.

L'arête stockée dans la mémoire de connaissance du SA est **mutable**. Elle enveloppe le CIR-noyau et y ajoute l'état dynamique géré par le SA : `confirmation_count`, `last_confirmed_at`, `status` dans le cycle de vie, fiabilité calculée de la source. Ces champs évoluent à chaque nouveau cycle d'observation.

La distinction est la suivante :

```
CIR-noyau (produit par le moteur, immuable)
┌────────────────────────────────────────────────────┐
│ source, target, relation, confidence               │
│ provenance:                                        │
│   ref                ← où (pointeur), immuable     │
│   span               ← position dans doc, immuable │
│   extraction_method  ← comment, immuable           │
│   model_version      ← quelle version, immuable    │
│   extracted_at       ← quand, immuable             │
└────────────────────────────────────────────────────┘

Arête enrichie SA (gérée par la mémoire SA, mutable)
┌────────────────────────────────────────────────────┐
│ cir: <CIR-noyau ci-dessus>                        │
│ status: NOUVELLE | ACTIVE | CONFIRMÉE | ...        │ ← SA
│ confirmation_count: 7                              │ ← SA
│ last_confirmed_at: 2026-09-25T...                  │ ← SA
│ source_reliability: 0.849                          │ ← SA (depuis registre)
└────────────────────────────────────────────────────┘
```

**Règle** : le moteur ne reçoit jamais l'arête enrichie — il ne connaît que le CIR-noyau. La mémoire de connaissance du SA ne modifie jamais le CIR-noyau — elle le stocke tel quel et gère l'état mutable en dehors. Ce sont deux objets distincts avec des cycles de vie distincts.

---

## 9. Analyse des écarts — moteur

Tableau restreint aux capacités du périmètre moteur. Les écarts SA (système agentique) sont hors périmètre.

| Capacité | État actuel | Ce qui manque |
|---|---|---|
| Extraction causale FR/EN | ✅ Fonctionnel | Coréférence inter-phrases |
| Extraction depuis code | ✅ Fonctionnel | Rien de bloquant |
| Pearl niveau 1–3 | ✅ Implémenté | — |
| CYCLES / GAPS | ✅ Partiel | GAPS orienté objectif |
| Verbalisation CIR → texte naturel | ❌ Non implémenté | `ReferenceDecoder` = linéarisation structurelle ASCII, pas du texte naturel. `LexicalConnectorAssembler` = sélection de connecteur fixe, ignore la sémantique des nœuds. `TrainableDecoder` déprécié. Un verbalizer correct nécessite CIR + vecteurs enrichis R-GCN en entrée (§4.6). |
| Chaîne de données autonome | ✅ Fonctionnel | — |
| Provenance immuable dans le CIR | ✅ Implémenté (P1.1) | Struct `Provenance` avec 5 champs : `ref`, `span`, `extraction_method`, `model_version`, `extracted_at`. Backward compat CIR v1. Validation Warning si absent. |
| Déduplication lexicale | ✅ Implémenté (P1.2) | `normalize_label()` (fold accents, strip ponct), `AliasTable`, matching normalisé dans `match_score()` et `find_path()`. |
| Raisonnement abductif | ✅ Implémenté (P2.1) | `EXPLAIN <effet>` — BFS inverse, score = min_conf_path × ln(1+depth), tri décroissant. |
| Raisonnement temporel | ✅ Implémenté (P2.2) | `CHAIN_T`, `BEFORE?`, `DELAY` — exploitation de `temporal_index` et `TemporalGap`. |
| Méta-raisonnement | ✅ Implémenté (P2.3) | `DENSITY`, `COVERAGE`, `RELIABILITY` — densité, couverture, fiabilité = mean_conf × provenance_ratio. |
| Raisonnement normatif | ✅ Implémenté (P2.4) | `DIFF A -> B` — gap = 1.0 - min_conf_chemin, gap_rate_pct. |
| Raisonnement adversarial | ✅ Implémenté (P3.2) | `CENTRALITY` (centralité pondérée confiance), `SPOF?` (BFS virtuel par nœud). |
| Raisonnement multi-échelle | ✅ Implémenté (P3.1) | Champ `parent: Option<NodeId>` sur `CausalNode` (CIR v2). `ZOOM_IN`, `ZOOM_OUT`, `AGGREGATE`. |
| Raisonnement par analogie | ✅ Implémenté (P3.3 — prototype) | `ANALOGY A -> B` — matching O(E) par signature topologique (relation, types, confiance, degré). Seuil 0.3. |

### Ce qui est solide et doit être préservé

- **Le CIR** : format pivot bien conçu, extensible, sérialisable. Ne pas le modifier, l'enrichir.
- **La séparation moteur / outils** : le moteur ne lit pas de YAML, ne dépend pas de spaCy. Cette contrainte est saine et doit être maintenue.
- **Le raisonnement Pearl** : les trois niveaux sont correctement implémentés. Ils sont le socle sur lequel construire.
- **La chaîne de données** : scraping → annotation → gates → fusion est fonctionnelle. Elle alimente le modèle ML.
- **L'architecture JSON-native** : tout est sérialisable, tout est interopérable. C'est la bonne fondation.

---

## 10. Priorités de développement — moteur

> **P1–P3 implémentés le 2026-09-26.** Les éléments ci-dessous reflètent l'état final. Les seuils de performance sont des cibles de conception — aucun n'est mesuré au 2026-09-26. Les méthodes de mesure restent à définir.

### P1 — Fondations indispensables

**P1.1 — Provenance immuable dans le CIR** ✅ Implémenté

Struct `Provenance` avec 5 champs immuables (`ref`, `span`, `extraction_method`, `model_version`, `extracted_at`) sur `CausalEdge`. Frontends FR/EN/code remplissent la provenance à l'extraction. Middleend émet un Warning si absente. Backward compat CIR v1 (`#[serde(default)]`). ir_emitter.py émet `extraction_method: "ml_python"`.

**P1.2 — Déduplication lexicale minimale** ✅ Implémenté

`normalize_label()` dans gcn-ir : fold accents FR/ES, lowercase, strip ponctuation, substitution ligatures (œ→oe, æ→ae). `AliasTable` dans gcn-knowledge : insert/resolve normalisés. `match_score()` et `find_path()` utilisent `normalize_label`. Même normalisation côté Python (`_normalize()` via unicodedata NFD).

### P2 — Extensions de raisonnement de base

**P2.1 — Raisonnement abductif** ✅ Implémenté

`EXPLAIN <effet>` — BFS inverse depuis l'effet. Score = `min_conf_on_path × ln(1 + depth)`. Tri décroissant. Exposé via `InstructionHandler._abduct()`.

**P2.2 — Raisonnement temporel** ✅ Implémenté

`CHAIN_T <from> -> <to>` : chemin enrichi avec `temporal_gap` et `index_delta`, signale si `temporally_ordered`.
`BEFORE? <a>, <b>` : comparaison `temporal_index` + existence du chemin orienté.
`DELAY <from> -> <to>` : somme des `TemporalGap.min/max` ou delta `temporal_index`.

**P2.3 — Méta-raisonnement** ✅ Implémenté

`DENSITY [label]` : densité globale `n_edges / (n_nodes×(n_nodes-1))` ou locale `degree / (2×(n_nodes-1))`.
`COVERAGE <label>` : degré + ratio provenance.
`RELIABILITY <label>` : `mean_confidence × provenance_ratio`.

**P2.4 — Raisonnement normatif** ✅ Implémenté

`DIFF <from> -> <to>` : `observed_confidence = min_conf_chemin`, `gap = 1.0 - observed_confidence`, `gap_rate_pct`. Chemin absent → `gap = 1.0`.

### P3 — Extensions avancées

**P3.1 — Raisonnement multi-échelle** ✅ Implémenté

Champ `parent: Option<NodeId>` sur `CausalNode` (CIR v2, optionnel en lecture). `ZOOM_IN <label>` : enfants directs. `ZOOM_OUT <label>` : parent ou None si racine. `AGGREGATE <label>` : arêtes sortantes du groupe vers l'extérieur + confiance moyenne.

**P3.2 — Raisonnement adversarial** ✅ Implémenté

`CENTRALITY <label>` : `(Σ conf_in + Σ conf_out) / (deg_in + deg_out)`. `SPOF?` : pour chaque nœud, BFS virtuel sans ce nœud, `spof_score = paths_cut / total_pairs`. Tri décroissant.

**P3.3 — Raisonnement par analogie** ✅ Implémenté (prototype)

`ANALOGY <from> -> <to>` — matching O(E) par signature topologique dans `gcn-backend/src/analogy.rs`. Score : `0.40×rel_sim + 0.30×type_sim + 0.20×conf_prox + 0.10×degree_prox`. Seuil 0.3. Patron absent → résultats vides.

> **Note** : la déduplication sémantique complète (résolution d'entités par embeddings + clustering) appartient au SA, pas au moteur.

### Ce qui ne sera pas fait dans le moteur

- **Raisonnement général** : GCN restera un moteur causal, pas un système de raisonnement général. Il ne fait pas d'inférences non causales.
- **Génération libre** : la verbalisation reste déterministe (ReferenceDecoder). Pas de génération LLM intégrée dans le moteur.
- **Décisions autonomes** : le moteur produit des analyses. Les décisions (sanctions, actions légales, actions critiques) restent humaines ou déléguées au SA.
- **Multimodalité** : texte et code uniquement. Pas de traitement d'images, de tableaux ou de figures.
- **Boucle agentique** : le moteur n'a pas de boucle. C'est le rôle du SA.
- **API HTTP** : le moteur est une bibliothèque. L'API HTTP est exposée par le SA.

### Métriques de succès moteur

Aucun développement de P1 à P3 n'est considéré terminé sans mesure.

**Phase P1 — Fondations**

| Métrique | Seuil | Méthode de mesure |
|---|---|---|
| Intégrité provenance `ref+span` sur 100 extractions | 100 % des arêtes ont les 5 champs | Test automatisé sur corpus mixte |
| Latence `query_causes()` p99 | < 50 ms sur graphe 100k arêtes | Benchmark sur corpus CTI synthétique |
| Taux de déduplication lexicale correcte | ≥ 0.90 sur 500 labels normalisés | Évaluation manuelle d'un échantillon |

**Phase P2/P3 — Raisonnements avancés**

| Métrique | Seuil | Méthode |
|---|---|---|
| Qualité CHAIN Dijkstra vs BFS | Confiance moyenne +15 % sur 500 requêtes | Comparaison A/B sur corpus |
| Rappel abductif (trouver la vraie cause) | ≥ 0.70 sur corpus caché | Évaluation sur dataset CTI avec causes connues |
| Couverture méta-raisonnement | Taux de réponses avec fiabilité estimée correcte ≥ 0.80 | Annotation manuelle d'un échantillon |

---

## Conclusion

Le moteur GCN est un composant stable, focalisé, dont les fondations sont solides : CIR bien conçu, Pearl 1–3 correctement implémenté, chaîne de données fonctionnelle.

Ce noyau ne sera jamais un agent, et ce n'est pas son rôle. Son développement suit deux axes :

1. **Fondations** : enrichir le format CIR avec la provenance complète (`ref`, `span`, méthode, version, timestamp) — sans stocker la phrase ni le document.
2. **Extensions de raisonnement** : abductif, temporel, normatif, méta-raisonnement, multi-échelle, adversarial, analogie — dans cet ordre de priorité.

Ces extensions restent dans le périmètre du noyau. Elles seront exposées via l'API du SA une fois implémentées. Les dépendances vers le SA (données de cycle de vie, index) sont notées par capacité.

La séparation moteur / SA est la garantie que le noyau reste stable pendant que la couche d'intelligence évolue.

---

*Document produit le 2026-09-25. Fondé sur l'inspection directe de : discuss.py, index.py, graph_vecs.py, cli/session.py, verbalizer/instructions.py, engine.py, et l'ensemble des modules gcn-python. Toute affirmation sur le code est vérifiable dans les fichiers sources.*
