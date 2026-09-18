# Audit des cas d'usage — GCN Causal Engine

**Date :** 2026-09-17  
**Auteur :** Michel Tendeng  
**Scope :** Inventaire exhaustif et honnête des domaines d'application réels

---

## Principe directeur

GCN répond à une question que les LLMs ne peuvent pas répondre de façon fiable :

> **"Pourquoi X se produit-il, selon ces documents précis, et peux-tu le prouver ?"**

La valeur n'est pas dans la génération de texte — c'est dans la **vérifiabilité, la traçabilité et le raisonnement formel sur la causalité**. Là où un LLM dit *"probablement parce que..."*, GCN dit *"selon le document D, ligne 47, avec une confiance de 89%, parce que..."*.

---

## 1. Cybersécurité

### 1.1 Analyse de chaînes d'attaque (CTI)

**Problème :** Les analystes SOC reçoivent des centaines de rapports de threat intelligence (CTI), de CVEs et d'IOCs. Chaque rapport décrit une chaîne d'attaque causale. Extraire et corréler manuellement ces chaînes prend des semaines.

**Ce que GCN fait :**
- Analyse des rapports CTI → extrait les chaînes causales
- Construit un graphe agrégé : `vulnérabilité → [enable] → exploitation → [cause] → persistance → [enable] → exfiltration`
- Répond à : "quelles vulnérabilités mènent à l'exfiltration de données ?"
- Détecte les chaînes communes à plusieurs groupes d'attaquants

**Requête GCN-QL typique :**
```
SELECT path FROM vuln TO data_exfiltration WHERE confidence > 0.80
```

**Valeur ajoutée vs LLM :** Une réponse d'un LLM sur les chaînes d'attaque est plausible mais non vérifiable. GCN trace chaque lien jusqu'au rapport source. Utilisable comme preuve dans un rapport d'incident.

**Données requises :** Rapports CVE, bulletins de sécurité, rapports d'incident, frameworks MITRE ATT&CK.

---

### 1.2 Analyse forensique de logs

**Problème :** Après un incident, un analyste doit reconstruire la chaîne causale depuis des milliers de lignes de logs.

**Ce que GCN fait :**
- Analyse les logs structurés ou semi-structurés
- Extrait : `[processus A terminé anormalement] --[cause]--> [fichier de config corrompu] --[enable]--> [crash du service B]`
- Identifie la cause racine (root cause) dans une chaîne multi-étapes
- Détecte les boucles causales (cascade d'erreurs)

**Valeur ajoutée vs LLM :** La chaîne causale est extraite des logs réels, pas générée. Chaque lien est vérifiable.

---

### 1.3 Corrélation MITRE ATT&CK

**Problème :** Le framework MITRE ATT&CK est un graphe causal. Le mapper manuellement sur un incident prend des heures.

**Ce que GCN fait :**
- Ingère le framework ATT&CK → graphe causal structuré
- Analyse un rapport d'incident → extrait les techniques
- Corrèle les techniques observées avec les chaînes ATT&CK connues
- Identifie les étapes manquantes dans la chaîne ("qu'est-ce qui devrait se passer ensuite ?")

---

### 1.4 Analyse de malwares et dépendances de code

**Problème :** Comprendre ce qu'un malware ou un module de code cause réellement dans le système.

**Ce que GCN fait (`gcn-frontend-code`) :**
- Analyse le code Python/Rust/JS
- Extrait : `[fonction A appelle B] --[control_dependency]--> [module C] --[cause]--> [accès réseau D]`
- Identifie les chemins causaux vers des comportements dangereux (accès fichiers, réseau, processus)

---

## 2. Audit

### 2.1 Audit de conformité réglementaire

**Problème :** Un texte réglementaire (RGPD, NIS2, ISO 27001, Bâle III…) contient des centaines de règles causales. Vérifier la conformité manuellement est long et sujet à erreur.

**Ce que GCN fait :**
- Analyse le référentiel → extrait les chaînes d'obligation :
  `[si violation de l'article X] --[condition]--> [notification obligatoire Y] --[condition]--> [sanction Z]`
- Analyse les procédures internes → compare avec les obligations
- Détecte les lacunes : obligations du référentiel non couvertes par les procédures
- Répond à : "l'incident A déclenche-t-il l'obligation de notification B ?"

**Valeur ajoutée vs LLM :** La réponse est traçable jusqu'à l'article précis du règlement, pas une interprétation. Utilisable dans un rapport d'audit.

---

### 2.2 Audit de processus opérationnels

**Problème :** Les procédures décrivent des chaînes d'actions causales. Leur conformité avec les règles internes est difficile à vérifier à grande échelle.

**Ce que GCN fait :**
- Analyse les procédures → graphe causal des processus
- Détecte les cycles (dépendances circulaires → blocages potentiels)
- Identifie les points de défaillance unique (SPOF causaux)
- Détecte les contradictions entre versions de procédures

---

### 2.3 Audit de piste financière

**Problème :** Tracer la chaîne causale entre une décision de gestion et une anomalie financière.

**Ce que GCN fait :**
- Analyse les documents financiers, emails, comptes-rendus de réunion
- Extrait : `[décision D] --[cause]--> [transaction T] --[enable]--> [contournement de contrôle C]`
- Répond à : "la décision X a-t-elle causé l'anomalie Y ?"
- Traçabilité source par source pour le rapport d'audit

---

### 2.4 Due diligence (fusions, acquisitions)

**Problème :** Analyser des milliers de documents pour identifier les risques causaux (litiges, dépendances fournisseurs, risques réglementaires).

**Ce que GCN fait :**
- Analyse le corpus de documents → graphe causal des risques
- Identifie les chaînes de risque : `[dépendance fournisseur unique] --[enable]--> [rupture d'approvisionnement] --[cause]--> [arrêt de production]`
- Détecte les contradictions entre documents

---

## 3. Investigation

### 3.1 Journalisme d'investigation

**Problème :** Un journaliste analyse des milliers de documents (emails, contrats, rapports) pour prouver une chaîne de responsabilité.

**Ce que GCN fait :**
- Analyse le corpus → graphe causal de la chaîne de responsabilité
- Répond à : "qui a causé quoi, selon quel document ?"
- Trace les chaînes : `[décision de A] --[cause]--> [contrat B] --[enable]--> [versement C] --[cause]--> [préjudice D]`
- Détecte les contradictions entre témoignages

**Valeur ajoutée vs LLM :** Chaque lien est traçable jusqu'au document source. Utilisable comme base pour des citations dans un article ou une plainte.

---

### 3.2 Investigation judiciaire

**Problème :** La causalité est le cœur du droit pénal et civil. Prouver qu'une action a causé un préjudice requiert une chaîne causale documentée.

**Ce que GCN fait :**
- Analyse les pièces du dossier → graphe causal des faits
- Extrait les chaînes causales avec niveau de confiance et source
- Répond à : "l'action A a-t-elle causé le préjudice B, selon les pièces C, D, E ?"
- Détecte les lacunes dans la chaîne causale de l'accusation ou de la défense
- Raisonnement contrefactuel (Pearl niveau 2) : "sans l'action A, le préjudice B se serait-il produit ?"

**Valeur ajoutée vs LLM :** Admissible comme outil d'analyse — chaque affirmation est tracée jusqu'à la pièce du dossier. Un LLM produit une opinion, GCN produit une analyse documentée.

---

### 3.3 Analyse du renseignement (OSINT/HUMINT)

**Problème :** Des centaines de sources sur un sujet. Comprendre qui cause quoi, avec quelle certitude, selon quelles sources.

**Ce que GCN fait :**
- Analyse le corpus OSINT → graphe causal agrégé
- Pondère les liens par nombre de sources concordantes
- Détecte les contradictions entre sources
- Identifie les acteurs pivots (nœuds les plus causalement connectés)
- Répond à : "selon X sources, Y cause Z. Mais 3 sources le contredisent."

---

### 3.4 Analyse de fraude

**Problème :** Détecter les chaînes causales d'une fraude dans des volumes de transactions et communications.

**Ce que GCN fait :**
- Extrait les chaînes : `[accès inhabituel] --[enable]--> [modification de données] --[cause]--> [virement frauduleux]`
- Détecte les patterns causaux récurrents (même structure dans des incidents différents)
- Lie les incidents disparates via des causes communes

---

## 4. Recherche scientifique

### 4.1 Revue systématique de littérature

**Problème :** Un chercheur veut savoir ce que 10 000 articles disent sur les causes de X. La revue manuelle prend des mois.

**Ce que GCN fait :**
- Analyse le corpus → extrait toutes les affirmations causales
- Agrège par fréquence : "X est causé par Y (432 articles), Z (218 articles)"
- Détecte les contradictions : "A cause B selon 89 articles, A prévient B selon 23 articles"
- Identifie les lacunes : causes de X jamais étudiées

**Valeur ajoutée vs LLM :** Chaque affirmation est tracée jusqu'à l'article source. La synthèse est reproductible et vérifiable, pas une opinion du modèle.

---

### 4.2 Extraction de claims causaux de publications

**Problème :** Les publications scientifiques contiennent des centaines de claims causaux par article. Les extraire manuellement est impossible à grande échelle.

**Ce que GCN fait :**
- Analyse automatique → CIR par article
- Base de données de claims causaux interrogeable
- Répond à : "quels articles affirment que X cause Y ?"

---

### 4.3 Détection de contradictions dans la littérature

**Problème :** "A cause B" dans une étude, "A prévient B" dans une autre. Ces contradictions passent souvent inaperçues.

**Ce que GCN fait :**
- Analyse le corpus → détecte les arêtes contradictoires entre sources
- Alerte : "15 articles affirment A → B, 8 articles affirment A prévient B"
- Identifie les conditions qui expliquent la divergence (Pearl niveau 3)

---

## 5. Santé et médecine

### 5.1 Aide à la décision clinique basée sur l'évidence

**Problème :** Un clinicien veut les causes documentées d'une complication spécifique, avec les études à l'appui.

**Ce que GCN fait :**
- Corpus de publications médicales analysé → graphe causal médical
- Répond à : "quelles causes documentées pour ce symptôme ? Quelles études ?"
- Confidence par lien basée sur le nombre et la qualité des études

**Valeur vs LLM :** Chaque réponse cite les études sources. Pas d'hallucination de références médicales.

---

### 5.2 Pharmacovigilance

**Problème :** Extraire les effets causaux documentés d'un médicament depuis les rapports de pharmacovigilance et la littérature.

**Ce que GCN fait :**
- Analyse les rapports → extrait : `[médicament X] --[cause]--> [effet indésirable Y]` avec fréquence et conditions
- Détecte les interactions causales entre médicaments

---

### 5.3 Épidémiologie computationnelle

**Problème :** Extraire les facteurs de risque documentés depuis des milliers d'études épidémiologiques.

**Ce que GCN fait :**
- Corpus d'études → graphe causal des facteurs de risque
- Répond à : "quels facteurs causent X, selon combien d'études, avec quelle force d'association ?"

---

## 6. Ingénierie et code

### 6.1 Analyse de dépendances causales dans un codebase

**Problème :** Comprendre pourquoi un système se comporte de telle façon nécessite de tracer les dépendances causales.

**Ce que GCN fait (`gcn-frontend-code`) :**
- Analyse Python/Rust/JS → extrait `data_dependency` et `control_dependency`
- Répond à : "qu'est-ce qui cause l'appel à cette fonction ?"
- Identifie les chemins causaux vers des comportements spécifiques (accès réseau, lecture de fichiers…)

---

### 6.2 Documentation automatique de l'architecture

**Problème :** La documentation causale d'un système ("si X, alors Y se déclenche") est rarement maintenue à jour.

**Ce que GCN fait :**
- Analyse le code → génère le graphe causal des comportements
- Extrait les invariants causaux : "A cause toujours B dans ce module"
- Détecte les ruptures de causalité documentée vs code réel

---

### 6.3 Root cause analysis automatisée

**Problème :** Identifier la cause racine d'un bug ou d'une panne dans un système complexe.

**Ce que GCN fait :**
- Analyse les logs + le code + les rapports d'incident
- Construit la chaîne causale de la panne
- Remonte jusqu'à la cause racine par traversée du graphe
- Raisonnement contrefactuel : "sans ce changement de configuration, la panne se serait-elle produite ?"

---

## 7. Droit et compliance

### 7.1 Analyse de contrats

**Problème :** Les contrats contiennent des centaines de clauses causales (si X, alors Y, sauf si Z).

**Ce que GCN fait :**
- Analyse le contrat → graphe causal des obligations
- Répond à : "dans quelles conditions l'obligation Y se déclenche-t-elle ?"
- Détecte les contradictions entre clauses

---

### 7.2 Analyse de jurisprudence

**Problème :** Identifier comment des faits similaires ont causé des décisions judiciaires similaires.

**Ce que GCN fait :**
- Corpus de décisions → graphe causal fait → décision
- Répond à : "des faits similaires à X ont causé quelle décision, dans combien de cas ?"

---

## 8. Finance et économie

### 8.1 Analyse de risque systémique

**Problème :** Identifier les chaînes de contagion dans le système financier.

**Ce que GCN fait :**
- Analyse les rapports financiers, études économiques → graphe causal des interdépendances
- Identifie les nœuds pivots (une faillite qui cause une cascade)
- Répond à : "si X fait défaut, quelles entités sont causalement exposées ?"

---

### 8.2 Analyse des rapports financiers

**Problème :** Extraire les relations causales dans les rapports annuels pour comprendre les moteurs de performance.

**Ce que GCN fait :**
- Analyse les rapports → extrait : "la hausse des coûts de matières premières cause la compression des marges"
- Agrège sur plusieurs trimestres pour détecter les patterns causaux récurrents

---

## Synthèse — Matrice des cas d'usage

| Domaine | Usage principal | Volume de documents | Vérifiabilité critique | Raisonnement Pearl | Faisabilité actuelle |
|---------|----------------|---------------------|----------------------|-------------------|---------------------|
| Cyber | Analyse CTI, forensique | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Audit conformité | Compliance réglementaire | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Investigation | Journalisme, judiciaire | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Renseignement | OSINT, analyse | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Recherche | Revue de littérature | ✅ Très élevé | ✅ | ⚠️ | ⚠️ Données à annoter |
| Médecine | Aide à la décision | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Code | Root cause, dépendances | ✅ Élevé | ✅ | ⚠️ | ✅ `gcn-frontend-code` opérationnel |
| Droit | Contrats, jurisprudence | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |
| Finance | Risque systémique | ✅ Élevé | ✅ | ✅ | ⚠️ Données à annoter |

---

## Ce que GCN n'est pas

| Ce que les utilisateurs pourraient attendre | Réalité |
|--------------------------------------------|---------|
| Un assistant général (comme ChatGPT) | ❌ GCN ne répond qu'à des questions causales sur son corpus |
| Un générateur de contenu | ❌ Pas de génération de texte libre |
| Un moteur de traduction | ❌ Hors domaine |
| Un résumeur générique | ❌ Résume uniquement les structures causales |
| Un outil opérationnel sans entraînement | ❌ Le modèle actuel est entraîné sur ~800 phrases climatiques |

---

## Le vrai bloquant vers la production

L'architecture est prête. Les frontends FR, EN et code sont opérationnels. Le pipeline d'entraînement fonctionne. La publication PyPI/crates.io est faite.

**Le seul bloquant :** des données annotées dans chacun des domaines cibles.

Pour chaque cas d'usage, il faut :
- 5 000–50 000 phrases annotées (texte + CIR) dans ce domaine
- Couvrant les 7 types de nœuds et les 11 types de relations
- Avec une distribution équilibrée des classes

C'est du travail d'annotation — pas de développement supplémentaire.
