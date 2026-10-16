# Spécification linguistique et architecturale — Moteur de raisonnement causal
## Fondements théoriques, modèle de données et pipeline de traitement

**Auteur** : Michel Tendeng  
**Date** : 2026-09-27  
**Statut** : Spécification de référence — indépendante de toute implémentation  
**Objectif** : Fonder une refonte complète du moteur sur une théorie linguistique rigoureuse  
**Révision** : Architecture d'apprentissage structurel — le moteur apprend les patterns, ne charge pas de lexiques

---

## Table des matières

1. [Objectifs et principes fondateurs](#1-objectifs-et-principes-fondateurs)
2. [Le verbe comme triple pivot](#2-le-verbe-comme-triple-pivot)
3. [Types de phrases](#3-types-de-phrases)
4. [Formes de phrases](#4-formes-de-phrases)
5. [Phrases simples vs complexes](#5-phrases-simples-vs-complexes)
6. [Relations syntaxiques UD — catalogue exhaustif](#6-relations-syntaxiques-ud--catalogue-exhaustif)
7. [Traits morphologiques UD](#7-traits-morphologiques-ud)
8. [Types de nœuds — taxonomie unifiée 8 types](#8-types-de-nœuds-du-graphe-causal)
9. [Types de relations causales (ternaire + hiérarchie Pearl + algèbre négation)](#9-types-de-relations-causales)
10. [Système d'équations relationnelles (Éq. 0–12)](#10-système-déquations-relationnelles)
11. [Modèle de données (ternaire, normalisation)](#11-modèle-de-données)
12. [Architecture du pipeline (bootstrap, cold start)](#12-architecture-du-pipeline)
13. [Application à l'exemple pivot](#13-application-à-lexemple-pivot)
14. [Problème de l'annotation — solution par habitudes positionnelles](#14-problème-de-lannotation--solution-par-habitudes-positionnelles)
15. [Taxonomie hiérarchique — relations et nœuds](#15-taxonomie-hiérarchique--relations-et-noeuds)
16. [Solution pour les connecteurs](#16-solution-pour-les-connecteurs)
17. [Composants externes — comment ils apprennent (ternaire)](#17-composants-externes--comment-ils-apprennent)
18. [Phrases complexes et conditionnelles — problèmes et solutions](#18-phrases-complexes-et-conditionnelles--problèmes-et-solutions)
19. [Réduction du moteur à sa fonction core](#19-réduction-du-moteur-à-sa-fonction-core)
20. [Hypothèses de travail — tests falsificateurs](#20-hypothèses-de-travail--à-prouver-avant-de-considérer-comme-acquises)
21. [Références](#21-références)

---

## 1. Objectifs et principes fondateurs

### 1.0 Registre des décisions architecturales gelées

Les décisions suivantes sont **gelées**. Elles ne sont plus des options à discuter. Tout développement doit s'y conformer. Toute contradiction détectée dans le document ou le code est un bug à corriger, pas un point ouvert.

```
D1 — Périmètre des lemmes
  Couche 1 (déterministe) : zéro lemme, uniquement features UD structurelles
  Couche 2 (apprise)      : lemma_emb[3×128] autorisé (vecteurs continus, pas règles statiques)
  NLU routing (externe)   : lemmes via lang_markers.json uniquement
  Moteur core             : aucun lemme sous aucune forme

D2 — Modèle ternaire
  CIREdge = (sources:[], target, relation, third:{role,node}?)
  Relations directes (third=None) : sources=[id], third=None — cas sans tiers explicite
  JOINT_CAUSE = sources=[id₁,id₂], CONDITIONAL_CAUSE = third.role=condition,
  MEDIATED_CAUSE = third.role=mediator
  19 types de relations canoniques :
    11 directes (third=None) + 3 ternaires positives + 3 *_prevent + 2 systémiques

D3 — Exemple pivot FILTER direct (third=None)
  "Le médicament n'est prescrit que si les traitements ont échoué."
  → FILTER direct (third=None) : sources=[échec], target=prescription, third=None
  Un ternaire CONDITIONAL_CAUSE exige agent A ≠ condition C (A≠∅)

D4 — Progression coarse → fin
  Niveau 1 coarse : 5 relations + 3 nœuds, depuis position+POS, N_min=20-50
  Niveau 2 fin    : 11+3 relations + 8 nœuds, requiert Mood+N_fine≥400
  Bascule par type, pas globale — un type bascule dès que ses seuils sont atteints

D5 — Taxonomie nœuds 8 types
  processus / état_local / état_global / entité / condition /
  concept / événement / contrainte
  Migration : action+transition→processus, etat→état_local, etat_systemique→état_global

D6 — Formule de confiance unique
  confiance = rung × f(Mood) × g(connecteur) + Δ_saillance
  R1=0.50, R2=0.75, R3=0.65
  f : Ind=1.0, Sub=0.85, Cnd=0.80
  g : explicite=1.0, VERB_causal=0.90, parataxis=0.60, aucun=0.50
  Δ : +0.10 si advcl_before_root
  Toutes les autres échelles dans ce document sont caduques.
  Ces valeurs sont des PRIORS — remplacés par calibration isotonique si N_val≥100.

D7 — Direction causale invariante
  Source = clause introduite par SCONJ (subordonnée)
  Cible  = proposition principale
  TOUJOURS — indépendamment de l'ordre d'apparition
  Exception SEQUENCE : direction = ordre temporel (Tense/position)

D8 — Objectif
  Production FR robuste sur moteur UD+GNN actuel.
  Pas de passage transformer/LLM — le raisonnement causal traçable
  (Pearl, graphe typé, requêtes WHY/CHAIN/SPOF) est l'objectif central.
  Multilingue = généralisé par les features UD universelles (T1 test wenn).

D9 — Stratégie d'implémentation
  Strangler incrémental — pas de freeze total.
  Ordre : Phase A (NLU/Layer1) → Phase B (features) → Phase C (ternaire) → Phase D (preuves)
  Le socle Pearl + benchmarks + datasets reste intact pendant toute la migration.
  Gate : T1-T5 tous verts avant merge de chaque phase.

D10 — lemma_emb (décision finale, ne plus rouvrir)
  lemma_emb[3×128] = lookups ponctuels sur les 3 tokens pivot (nsubj, root, obj)
  Source : embedding.py.vocab (infrastructure existante dim=128)
  Distinct du pooling clause entière (features.py:187) — les deux coexistent
  Vecteur zéro[128] si lemme absent du vocabulaire
```

### 1.1 But du moteur

Le moteur est un **moteur de raisonnement causal** sur du texte naturel. Son rôle est triple :

```
1. EXTRAIRE   — lire un texte et en tirer les relations causales
               Texte brut → représentation UD → graphe causal (CIR)

2. RAISONNER  — interroger le graphe construit
               CIR → réponses aux questions causales (pourquoi, comment, sans X...)

3. RÉPONDRE   — produire une réponse intelligible
               Résultat du raisonnement → texte structuré
```

Le moteur n'est pas un générateur de texte, ni un classifieur de documents, ni un résumeur générique. Il construit une **représentation logique des causes et effets** dans un domaine, puis répond à des questions sur cette structure avec un raisonnement traçable — chaque réponse est justifiée par les relations du graphe.

### 1.2 Les trois axiomes fondateurs

**Axiome 1 — La phrase est la vérité.**

Une phrase bien formée est une unité **auto-suffisante et complète**. Elle contient toutes les informations nécessaires à son interprétation : qui fait quoi, dans quelle modalité, sous quelle condition, avec quelle polarité. Le moteur n'a besoin d'aucune ressource externe pour lire une phrase — la phrase se décode elle-même par sa structure interne.

```
"Le médicament n'est prescrit que si les traitements ont échoué."

Contient en elle-même :
  ├── L'entité concernée     (médicament)
  ├── L'action               (prescrire)
  ├── La voix                (passif : Voice=Pass)
  ├── La polarité            (négatif : Polarity=Neg)
  ├── La restriction         (ne...que)
  ├── La condition           (si les traitements ont échoué)
  ├── La modalité            (indicatif : fait établi)
  └── Le type de relation    (filter : prescription uniquement sous condition)
```

Aucune de ces informations n'est externe à la phrase. Tout est là.

**Axiome 2 — Le verbe est le pilier.**

Le verbe est le seul élément obligatoire qui rend une phrase grammaticalement complète. Supprimer n'importe quel autre élément (sujet, complément, adverbe) laisse une structure partiellement interprétable. Supprimer le verbe détruit la phrase. Le verbe est le **point d'ancrage unique** depuis lequel tous les autres éléments prennent leur sens — leur rôle, leur relation, leur importance.

```
VERBE
  ├── détermine qui est sujet, qui est complément    (via les dep_rel)
  ├── détermine la modalité de la relation           (via Mood, Tense, Aspect)
  ├── détermine la polarité                          (via Polarity, Voice)
  ├── détermine la nature de la relation causale     (via sa valence + connecteur)
  └── détermine l'intention de la phrase             (via Mood + présence/absence nsubj)
```

**Axiome 3 — La position est du sens.**

L'ordre des éléments dans la phrase n'est pas arbitraire. En français (et dans toutes les langues à ordre canonique), la position de chaque constituant **par rapport au verbe** porte une information sémantique et pragmatique distincte :

```
Position pré-verbale  (id < root.id)  → information donnée, topic, présupposé
Position post-verbale (id > root.id)  → information nouvelle, rhème, focus

Position pré-verbale du sujet         → ordre canonique, déclaratif
Position post-verbale du sujet        → inversion → interrogatif
                                       (nsubj.id > root.id)

Subordonnée en tête de phrase         → condition/cause mise en relief
  (advcl avant la principale)          signale que la condition est proéminente

Subordonnée en fin de phrase          → condition/cause en arrière-plan
  (advcl après la principale)          la principale est l'information centrale

Négateur pré-verbal (ne)              → porte sur tout le prédicat
Négateur post-verbal (pas)            → renforce et spécifie la portée négative

Connecteur en tête de subordonnée    → toujours : mark.id < advcl_head.id
  (si échoué → "si" avant "échoué")   position canonique du connecteur
```

Ces positions sont encodées dans le champ `id` de chaque token et dans la relation `dep_head`. Elles constituent des **features positionnelles** que le moteur doit apprendre à lire au même titre que `dep_rel` et `morph`.

### 1.3 Principes architecturaux

**P1 — Le moteur core n'utilise jamais de lemme comme règle statique.**  
Le moteur core (extraction + raisonnement = couches 1 à 5) n'encode aucune liste de lemmes et ne charge aucun fichier de configuration linguistique. Il apprend depuis des exemples annotés à reconnaître les marqueurs à leur **position structurelle** — `dep_rel`, `pos`, `morph`, `id` relatif au root. Les lemmes sont acceptés sous deux formes uniquement : (a) **embeddings appris** par le classifieur de couche 2 (représentation vectorielle dense, pas règle statique) ; (b) **configuration externe du NLU routing** (composant externe au moteur, voir §19) sous forme de données JSON optionnelles — jamais dans le code du moteur.

```
Moteur core (P1 s'applique) :
  Couche 1 déterministe : dep_rel, pos, morph, id → aucun lemme
  Couche 2 apprise      : FeatureVector + embeddings lemmes → appris, pas codés

NLU routing (externe au moteur, P1 ne s'applique pas) :
  Peut utiliser lemmes comme signaux de dispatch (Éq. 6, §12.2)
  Peut charger lang_markers.json (données de configuration, pas règles moteur)
```

**P2 — La structure UD + la position distinguent les rôles sans ambiguïté.**  
`dep_rel=nsubj` → sujet. `dep_rel=obj` → complément direct. `dep_rel=mark` → connecteur. `nsubj.id > root.id` → inversion → interrogatif. `advcl.id < root.id` → subordonnée en tête → condition proéminente. Ces patterns sont universels dans UD.

**P3 — Le verbe est le pivot unique.**  
Toute extraction part du verbe root. Un seul parcours depuis le root, enrichi des positions relatives de chaque dépendant, produit simultanément le type de phrase, les rôles sémantiques, le type de nœud, le type de relation et la saillance informationnelle.

**P4 — Invariance causale active/passive.**  
La même relation causale doit être extraite indépendamment de la voix syntaxique. Le moteur normalise toujours vers `(Agent → cause, Patient → effet)`.

**P5 — Deux couches d'apprentissage.**  
Couche 1 : règles déterministes sur les features UD + position (robuste, sans données).  
Couche 2 : classifieur appris sur des données annotées. Il apprend des patterns de features UD, de positions relatives, et d'embeddings de lemmes — jamais de règles statiques sur des lemmes spécifiques.

**P6 — L'intention est dans la structure de la phrase.**  
Interrogative → chercher et répondre. Impérative → agir. Déclarative → extraire. Le DSL interne est généré automatiquement depuis la structure, jamais saisi par l'utilisateur.

**P7 — Confiance graduée selon la hiérarchie causale de Pearl.**  
La confiance est calculée par la formule unique de §9.0 : `confiance = rung × f(Mood) × g(connecteur) + Δ_saillance`. Toute autre échelle de confiance dans ce document est caduque et référence §9.0.

---

## 2. Le verbe comme triple pivot

### 2.1 Principe structural — verbe et position

En grammaire de dépendances (UD), le verbe est la **tête** (root) de la clause. Tous les constituants lui sont subordonnés. La position de chaque constituant par rapport au verbe root est une **dimension de sens** encodée dans le champ `id` de chaque token.

```
    id:  1        2        3      4        5          6
         Le    médicament est  prescrit   si      échouent
                  │              │        │           │
               nsubj:pass      root     mark        advcl
                  │              │        │           │
             [Patient]       [VERBE]  [Connecteur] [Cause]

    Position relative au root (id=4) :
      médicament : id=2 < root.id=4  → pré-verbal  → topic (information donnée)
      si         : id=5 > root.id=4  → post-verbal → rhème (condition en arrière-plan)
      échouent   : id=6 > root.id=4  → post-verbal → condition backgroundée

    Si la subordonnée était en tête :
      "Si les traitements ont échoué, le médicament est prescrit."
      → advcl.id < root.id → condition foregroundée → mise en relief de la condition
```

**Les positions clés et leur sens :**

| Position | Condition | Signal | Sens |
|----------|-----------|--------|------|
| Sujet pré-verbal | `nsubj.id < root.id` | Ordre canonique | Déclaratif — topic connu |
| Sujet post-verbal | `nsubj.id > root.id` | Inversion | Interrogatif ou emphatique |
| Subordonnée en tête | `advcl.min_id < root.id` | Condition frontée | Cause/condition en relief |
| Subordonnée en queue | `advcl.min_id > root.id` | Condition backgroundée | Cause/condition secondaire |
| Négateur pré-verbal | `neg.id < root.id` | `ne` en français | Portée négative sur le prédicat |
| Négateur post-verbal | `neg.id > root.id` | `pas/jamais` | Renforcement et spécification |
| Connecteur en tête de sub. | `mark.id < advcl_head.id` | Position canonique | Connecteur structurel |

### 2.2 Les trois classifications depuis le verbe

Un seul parcours depuis le verbe root produit trois classifications simultanées :

```
                    ┌─────────────────────────────┐
                    │         VERBE (root)         │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
     MORPHOLOGIE DU VERBE    VALENCE + ARGS        CONNECTEUR (mark/cc)
     Mood / Voice / Person   nsubj / obj / advcl   SCONJ sur advcl/cc
              │                    │                     │
              ▼                    ▼                     ▼
     TYPE DE PHRASE          TYPE DE NŒUD          TYPE DE RELATION
     (→ intention)           (→ représentation)    (→ lien causal)
```

**Classification 1 — Type de phrase (depuis la morphologie) :**

| Signal UD | Valeur | Type détecté |
|-----------|--------|-------------|
| `morph.Mood` | `Imp` + ¬nsubj | IMPÉRATIVE |
| `morph.PronType` | `Int` sur un token | INTERROGATIVE |
| inversion | `nsubj.id > root.id` | INTERROGATIVE |
| `punct` | `?` | INTERROGATIVE |
| `morph.Mood` | `Ind` + `punct=!` | EXCLAMATIVE |
| `morph.Mood` | `Ind` + nsubj présent | DÉCLARATIVE |

**Classification 2 — Type de nœud (depuis la valence et les arguments) :**

| Structure | Type de nœud |
|-----------|-------------|
| VERB transitif + nsubj + obj | `processus` |
| VERB d'état + cop + attribut | `état_local` (entité) ou `état_global` (système) |
| VERB existentiel + expl | `entité` |
| VERB dans advcl + mark=condition | `condition` |
| NOUN déverbal (nominalisation) | `processus` |
| NOUN abstrait (mesure, grandeur) | `concept` |
| NOUN + obl:tmod (fait daté) | `événement` |
| advcl + Polarity=Neg + restriction | `contrainte` |

**Classification 3 — Type de relation (depuis la structure apprise) :**

Le type de relation n'est pas lu dans un dictionnaire : il est **inféré depuis la configuration structurelle** du connecteur et son contexte. Le modèle apprend ces patterns depuis les données annotées.

```
Signaux structurels appris par le modèle :

dep_rel=mark + pos=SCONJ + Mood(V_sub)=Sub     → apprendre : condition ou concession ou but
dep_rel=mark + pos=SCONJ + Mood(V_sub)=Ind     → apprendre : cause ou temporel ou conséquence
dep_rel=cc   + pos=CCONJ + position_dans_conj  → apprendre : cause ou concession ou séquence
dep_rel=advmod + pos=ADV + Polarity=Neg        → négation → prevent ou filter
pos=VERB + dep_rel=root + valence(transitive)  → apprendre : cause ou enable ou prevent
dep_rel=parataxis                              → relation implicite, confiance basse
```

Ce que le modèle distingue structurellement sans lemme :
- Un `mark` dont le verbe subordonné est au subjonctif (`Mood=Sub`) ≠ un `mark` à l'indicatif
- Un `advmod` portant `Polarity=Neg` ≠ un `advmod` ordinaire
- Un `nsubj` ≠ un `mark` ≠ un `obj` — roles distincts, jamais confondus

---

## 3. Types de phrases

Les 4 types sont **exhaustifs et mutuellement exclusifs** au niveau de la proposition principale.

### 3.1 Déclarative

Exprime un fait, une information, une assertion.

```
Structure    : [nsubj] + V(Mood=Ind) + [arguments] + [adjuncts]
Ponctuation  : .  ou  ...  ou absence
Mood UD      : Ind
Ordre        : canonique S avant V
```

Exemples :
```
"Le médicament réduit la douleur."
  nsubj=médicament, obj=douleur, VERB=réduire
  → cause(médicament, réduction_douleur)

"La demande baisse depuis deux ans."
  nsubj=demande, VERB=baisser, obl:tmod=depuis_deux_ans
  → processus(baisse_demande) + temporel(2_ans)
```

### 3.2 Interrogative

Demande une information. Trois sous-types structurellement distincts.

**Interrogative totale** — réponse oui/non uniquement

```
Structure FR  : inversion V-S  OU  est-ce que + S-V  OU  intonation
Ponctuation   : ?
Mood UD       : Ind
Signal clé    : nsubj.id > root.id  (inversion syntaxique)
                OU punct="?"
```

```
"Le médicament est-il efficace ?"
  root.id=2, nsubj.id=1  (médicament)... mais avec inversion:
  syntaxiquement : V-S → nsubj postposé au verbe
  → INTERROGATIVE TOTALE → requête sur le graphe
```

**Interrogative partielle** — demande un constituant précis

```
Ponctuation  : ?
PronType UD  : Int  sur le mot-Wh
```

| Mot-Wh | dep_rel du mot-Wh | PronType | Requête sur le graphe |
|--------|-------------------|----------|-----------------------|
| pourquoi | advmod | Int | explain(C) → chaîne causale vers C |
| comment | advmod | Int | chain(S, C) → chemin causal |
| quand | advmod | Int | chain_t(S, C) → chemin temporel |
| combien | advmod | Int | delay(S, C) → délai estimé |
| qui | nsubj | Int | find_effects(C) → agents de C |
| quel | det | Int | explain(C) |
| quoi / que | obj | Int | find_effects(S) → effets de S |

**Interrogative alternative** — choix entre deux options

```
Signal : conj + cc("ou") + punct="?"
"Prescrit-on le médicament A ou le médicament B ?"
```

### 3.3 Impérative

Exprime un ordre, une consigne, une instruction directe.

```
Ponctuation   : .  ou  !
Mood UD       : Imp
              OU VerbForm=Inf sans nsubj  (consigne écrite)
Signal clé    : ¬nsubj  +  (Mood=Imp  OU  VerbForm=Inf)
Person        : 2 (tu/vous) ou 1 pluriel (nous)
```

```
"Prescris le médicament."
  Mood=Imp, ¬nsubj, Person=2
  → action : prescrire(médicament)

"Analyser le fichier traitement.json"
  VerbForm=Inf, ¬nsubj
  → action : analyze("traitement.json")

"Ne pas dépasser la dose."
  VerbForm=Inf, ¬nsubj, Polarity=Neg
  → action : prevent(dépasser_dose)
```

Cas particulier — impératif indirect (politesse) :
```
"Veuillez entrer votre identifiant."
  V principal Ind (Veuillez) + ccomp(entrer) sans nsubj propre
  → Signal : verbe modal de politesse + ccomp sans sujet
  → Traitée comme impérative
```

### 3.4 Exclamative

Exprime une émotion, une intensité, un jugement affectif.

```
Ponctuation  : !
Mood UD      : Ind  ← critère de distinction avec l'impérative
```

**Règle de distinction exclamative / impérative — critique :**
```
!  +  Mood=Imp   →  IMPÉRATIVE   "Prescris le médicament !"
!  +  Mood=Ind   →  EXCLAMATIVE  "Les ventes s'effondrent !"
!  sans Mood     →  EXCLAMATIVE  (défaut si Mood absent)
```

```
"Quel médicament efficace !"
  PronType=Int + dep_rel=det + ! + Mood=Ind → EXCLAMATIVE

"Comme la progression est rapide !"
  advmod(rapide, Comme) + ! + Mood=Ind → EXCLAMATIVE
```

---

## 4. Formes de phrases

Les formes sont **orthogonales** aux types : toute combinaison est possible.  
Exemple valide : interrogative + passive + négative.

### 4.1 Affirmative / Négative

**Affirmative :** aucun marqueur de négation. Valeur par défaut.

**Négative :** deux niveaux sémantiquement distincts :

```
Négation totale    : ¬P
  "Le médicament n'est PAS prescrit."
  → fait nié → relation PREVENT ou absence de relation

Négation partielle : P | condition  (restriction)
  "Le médicament n'est prescrit QUE si..."
  → restriction exclusive → relation FILTER, pas PREVENT
```

Signaux UD par ordre de priorité :
```
1. root_morph.Polarity = Neg           (annotation morphologique directe)
2. token_morph.Polarity = Neg          (sur la particule de négation)
3. dep_rel=advmod sur négateur lexical  (fallback couche 2 via lang_markers)
```

Impact sur la relation causale :
```
cause(A, B)  +  Polarity=Neg            →  prevent(A, B)
cause(A, B)  +  Polarity=Neg + ne...que →  filter(A, B)
```

### 4.2 Active / Passive

**Active :** Agent = sujet syntaxique, Patient = objet.
```
nsubj(V, Agent) + obj(V, Patient)
"Le médecin [Agent] prescrit le médicament [Patient]."
```

**Passive :** Patient = sujet syntaxique, Agent optionnel (peut être effacé).
```
nsubj:pass(V, Patient) + aux:pass(V, être) + obl:agent(V, Agent)?
"Le médicament [Patient] est prescrit (par le médecin [Agent])."
```

**Passive agentive effacée :**
```
"Le médicament est prescrit."
  → Agent absent → entité responsable indéterminée ou générique
```

Signaux UD :
```
Voice=Pass dans root_morph     (priorité 1 — annotation directe)
dep_rel=aux:pass               (présence de l'auxiliaire être)
dep_rel=nsubj:pass             (sujet en position passif)
dep_rel=obl:agent              (complément d'agent introduit par "par")
```

**Invariance causale — propriété fondamentale :**
```
∀ Φ_actif, Φ_passif décrivant le même fait :
  Extraire(Φ_actif) ≡ Extraire(Φ_passif)

Actif  : nsubj(prescrit, médecin)      + obj(prescrit, médicament)
Passif : nsubj:pass(prescrit, médecin) + obl:agent(prescrit, médicament)

→ Le moteur normalise toujours : Agent=cause, Patient=effet
  avant construction du graphe, indépendamment de la voix.
```

### 4.3 Neutre / Emphatique

**Neutre :** ordre canonique S-V-O.

**Emphatique :** un constituant est focalisé.

```
Clivée         : "C'est le médicament qui réduit la douleur."
                  cop + nsubj + acl:relcl sur le constituant focalisé

Pseudo-clivée  : "Ce qui réduit la douleur, c'est le médicament."
                  csubj:relcl + cop

Topicalisation : "Le médicament, le médecin le prescrit."
                  dislocated(médicament) + obj coréférent
```

Le constituant focalisé est le candidat principal pour le nœud de la relation causale.

### 4.4 Personnelle / Impersonnelle

**Personnelle :** sujet = référent réel.

**Impersonnelle :** sujet = "il" explétif sans référent réel.

| Type | Structure UD | Exemple | Interprétation causale |
|------|-------------|---------|------------------------|
| Météorologique | expl(V, il) | "Il pleut." | propriété du système |
| Existentielle | expl + ccomp/obl | "Il existe un risque." | état du système |
| Falloir/suffire | expl + xcomp/ccomp | "Il faut prescrire." | contrainte |
| Passive impersonnelle | expl:pass | "Il se prescrit facilement." | processus générique |
| Présentative | expl + obl | "Il y a une hausse." | état |

Les impersonnelles expriment des **propriétés globales** du système — conditions structurelles, états généraux sans agent précis. Nœud extrait = `état_local` ou `état_global` ou `condition`.

---

## 5. Phrases simples vs complexes

### 5.1 Phrase simple

Une seule proposition, un seul verbe fléchi.

```
Φ_simple = [nsubj] + V(root) + [arguments] + [adjuncts*]
```

Critère : absence de dep_rels subordonnants sur des verbes fléchis (`advcl`, `ccomp`, `xcomp` sur `VerbForm=Fin`).

### 5.2 Phrase complexe — taxonomie complète

Au moins deux propositions. La relation inter-propositionnelle définit le type.

#### 5.2.1 Subordination adverbiale (advcl)

La proposition subordonnée modifie le verbe principal comme un adverbe.  
**C'est la structure la plus importante pour l'extraction causale.**

```
dep_rel : advcl  sur le verbe de la subordonnée
dep_rel : mark   sur le connecteur (SCONJ)
```

Taxinomie complète des sous-types par connecteur :

| Sous-type | Connecteurs FR | Connecteurs EN | Mood | Relation causale |
|-----------|---------------|----------------|------|-----------------|
| Condition | si, à condition que, pourvu que, seulement si, dans le cas où | if, provided that, as long as | Sub/Ind | CONDITION |
| Cause | parce que, car, puisque, comme, vu que, étant donné que | because, since, as, given that | Ind | CAUSE |
| Concession | bien que, malgré que, quoique, même si, encore que | although, even though, despite | Sub | CONCESSION |
| Conséquence | si bien que, de sorte que, au point que | so that, such that | Ind | ENABLE |
| But | pour que, afin que, de peur que | so that, in order that | Sub | MOTIVATION |
| Temporelle | quand, lorsque, après que, avant que, pendant que, depuis que, dès que | when, after, before, while, since | Ind | SEQUENCE |
| Comparaison | comme, ainsi que, de même que, plus que, moins que | as, just as, more than | Ind | ANALOGY |
| Restriction | à moins que, sauf si, excepté si | unless, except if | Sub | FILTER |

#### 5.2.2 Complétive (ccomp)

Proposition complément d'un verbe de cognition, de parole ou de perception.

```
dep_rel : ccomp
Déclencheurs : savoir, penser, croire, dire, constater, montrer, prouver...

"Le médecin sait que le traitement échoue."
  → ccomp(sait, échoue)
  → Relation épistémique : le verbe principal qualifie la certitude
  → Impact confiance : sait > pense > croit > suppose
```

#### 5.2.3 Complétive infinitive (xcomp)

Infinitif sans sujet propre — sujet partagé avec la principale.

```
dep_rel : xcomp

"Le médecin veut prescrire le médicament."
  → xcomp(veut, prescrire), sujet partagé = médecin
  → Relation intentionnelle : motivation(médecin, prescription)
```

#### 5.2.4 Relative (acl:relcl)

Modifie un nom en lui adjoignant une proposition.

```
dep_rel : acl:relcl
Introduite par : pronom relatif (qui, que, dont, où, lequel...)

"Le médicament qui réduit la douleur est prescrit."
  → acl:relcl(médicament, réduit)
  → La relative spécifie le référent → nœud enrichi
```

#### 5.2.5 Coordination (conj + cc)

Deux propositions de même statut syntaxique.

| Conjonction | Relation logique | Relation causale extraite |
|-------------|-----------------|--------------------------|
| `et` | addition | SEQUENCE ou CAUSE implicite |
| `ou` | alternative | alternative causale |
| `mais` | opposition | CONCESSION |
| `donc` | conséquence | CAUSE directe |
| `car` | cause | CAUSE (inversée : effet car cause) |
| `or` | opposition logique | CONCESSION |
| `ni` | négation double | double PREVENT |

#### 5.2.6 Parataxe — juxtaposition

Deux propositions sans connecteur. Relation implicite.

```
dep_rel : parataxis

"Le traitement échoue : le médecin prescrit un autre médicament."
  → parataxis(échoue, prescrit)
  → Relation implicite, souvent CAUSE ou SEQUENCE
  → Confiance réduite (relation inférée, non explicitée)
```

---

## 6. Relations syntaxiques UD — catalogue exhaustif

### 6.1 Arguments du verbe (core — obligatoires selon la valence)

| dep_rel | Nom grammatical | Description | Exemple |
|---------|----------------|-------------|---------|
| `nsubj` | Sujet nominal | Agent ou Thème en position sujet | *Le médecin* prescrit |
| `nsubj:pass` | Sujet passif | Patient promu en position sujet passif | *Le médicament* est prescrit |
| `obj` | Objet direct (COD) | Patient de l'action | prescrit *le médicament* |
| `iobj` | Objet indirect (COI) | Récipiendaire | prescrit au *patient* |
| `csubj` | Sujet clausal | Proposition entière en position sujet | *Guérir* est l'objectif |
| `csubj:pass` | Sujet clausal passif | Proposition en position sujet passif | *Qu'il guérisse* est espéré |
| `ccomp` | Complétive | Proposition complément de verbe | sait *que le traitement échoue* |
| `xcomp` | Complétive ouverte | Infinitif / attribut sans sujet propre | veut *prescrire* |

### 6.2 Modificateurs nominaux

| dep_rel | Nom grammatical | Description | Exemple |
|---------|----------------|-------------|---------|
| `det` | Déterminant | Article, démonstratif, possessif | *le* médicament |
| `amod` | Épithète adjectivale | Adjectif qualifiant le nom | médicament *efficace* |
| `nmod` | Complément du nom | GN modifiant un nom | médicament *de synthèse* |
| `nmod:poss` | Complément possessif | Génitif | médicament *du patient* |
| `nummod` | Complément numéral | Quantificateur | *trois* médicaments |
| `appos` | Apposition | Reformulation nominale | le médecin, *Dr Martin* |
| `acl` | Relative infinitive | Proposition adjectivale infinitive | médicament *à prescrire* |
| `acl:relcl` | Proposition relative | Relative avec pronom relatif | médicament *qui réduit la douleur* |
| `compound` | Composé | Noms composés | *base de données* |
| `flat` | Séquence plate | Noms propres multi-mots | *Jean-Paul Dupont* |

### 6.3 Modificateurs du verbe

| dep_rel | Nom grammatical | Description | Exemple |
|---------|----------------|-------------|---------|
| `advmod` | Complément adverbial | Modificateur adverbial | prescrit *rapidement* |
| `obl` | Complément oblique | Circonstanciel avec préposition | prescrit *en urgence* |
| `obl:agent` | Complément d'agent | Agent passif introduit par "par" | prescrit *par le médecin* |
| `obl:tmod` | Circonstanciel temporel | Complément de temps | prescrit *hier* |
| `obl:lmod` | Circonstanciel locatif | Complément de lieu | administré *à l'hôpital* |
| `advcl` | Subordonnée adverbiale | Proposition modifiant le verbe | prescrit *si les autres échouent* |

### 6.4 Relations fonctionnelles

| dep_rel | Nom grammatical | Description | Exemple |
|---------|----------------|-------------|---------|
| `aux` | Auxiliaire | Auxiliaire de temps / aspect / modal | *a* prescrit / *peut* prescrire |
| `aux:pass` | Auxiliaire passif | Être dans la construction passive | *est* prescrit |
| `cop` | Copule | Être / sembler / devenir + attribut | *est* efficace |
| `mark` | Subordonnant | Conjonction de subordination (SCONJ) | *si* les traitements échouent |
| `case` | Préposition | Marqueur de cas syntaxique | *par* le médecin |
| `cc` | Conjonction coordination | Et, ou, mais, donc, car, or, ni | *et* le patient guérit |
| `conj` | Conjoint | Deuxième membre d'une coordination | prescrit *et* administre |
| `punct` | Ponctuation | Tout signe de ponctuation | prescrit *.* |
| `fixed` | Expression figée | Locution grammaticale multi-mots | *bien que*, *afin de* |

### 6.5 Sujets explétifs

| dep_rel | Construction | Exemple |
|---------|-------------|---------|
| `expl` | Impersonnel standard | *Il* pleut / *Il* faut |
| `expl:pass` | Pronominal passif | *Il* se prescrit facilement |
| `expl:subj` | Sujet impersonnel | *Il* existe un problème |

### 6.6 Relations de discours et de structure textuelle

| dep_rel | Nom grammatical | Description | Exemple |
|---------|----------------|-------------|---------|
| `discourse` | Marqueur discursif | Interjection, marqueur d'attitude | *Eh bien*, le résultat... |
| `vocative` | Apostrophe | Interpellation directe | *Docteur*, prescrivez... |
| `parataxis` | Parataxe | Proposition juxtaposée | Il échoue *: on change* |
| `orphan` | Nœud orphelin | Nœud sans tête (ellipse) | Pierre aime Marie, Paul *aussi* |
| `list` | Item de liste | Élément d'énumération | 1. *efficace* 2. *sûr* |
| `dislocated` | Dislocation | Constituant disloqué | *Le médicament*, je le prescris |
| `reparandum` | Disfluence | Auto-correction (oral) | — |
| `dep` | Dépendance inconnue | Relation non classifiable | — |

---

## 7. Traits morphologiques UD

### 7.1 Traits du verbe — signaux primaires

| Feature | Valeurs | Rôle dans la classification |
|---------|---------|----------------------------|
| `Mood` | `Ind` / `Imp` / `Sub` / `Cnd` | **Type de phrase** : Imp→impérative, Sub→subordonnée, Cnd→hypothèse |
| `Tense` | `Pres` / `Past` / `Fut` / `Imp` | Ancrage temporel → qualification de la relation causale |
| `Aspect` | `Imp` / `Perf` | Imperfectif (en cours) / Perfectif (accompli) |
| `Person` | `1` / `2` / `3` | Person=2 + ¬nsubj → signal impératif secondaire |
| `Number` | `Sing` / `Plur` | Accord |
| `VerbForm` | `Fin` / `Inf` / `Part` / `Ger` | Inf + ¬nsubj → consigne écrite (impérative) |
| `Voice` | `Act` / `Pass` | **Forme** : Pass → inverser Agent/Patient |
| `Polarity` | `Pos` / `Neg` | **Forme** : Neg → relation niée → prevent ou filter |

### 7.2 Traits du nom et du pronom

| Feature | Valeurs | Rôle dans la classification |
|---------|---------|----------------------------|
| `Gender` | `Masc` / `Fem` | Accord grammatical |
| `Number` | `Sing` / `Plur` | Plur → généralisation possible |
| `Definite` | `Def` / `Ind` | Def → référent spécifique ; Ind → générique |
| `PronType` | `Prs` / `Int` / `Rel` / `Dem` / `Ind` / `Art` | **Int = signal interrogatif prioritaire** |
| `Reflex` | `Yes` | Construction pronominale (se) |
| `Case` | `Nom` / `Acc` / `Dat` / `Gen` / `Loc` | Cas grammatical |

### 7.3 Traits de l'adjectif

| Feature | Valeurs | Rôle |
|---------|---------|------|
| `Degree` | `Pos` / `Cmp` / `Sup` | Comparatif/superlatif → intensité → EXCLAMATIVE |

### 7.4 Table des signaux structurels — couche 1 (déterministe) et couche 2 (apprise)

Le moteur opère en deux couches. La couche 1 est déterministe sur les features UD universelles. La couche 2 est un classifieur appris — il ne charge pas de lexique, il généralise depuis les patterns de features observés dans les données annotées.

```
Type/Forme      Couche 1 — déterministe (features UD)  Couche 2 — apprise (patterns UD)
──────────────────────────────────────────────────────────────────────────────────────────
INTERROGATIVE   PronType=Int sur un token              pattern inversion + punct=?
                nsubj.id > root.id                     (le modèle apprend la combinaison)
                punct="?"
IMPÉRATIVE      Mood=Imp sur root                      pattern ¬nsubj + Person=2
                VerbForm=Inf + ¬nsubj                  (le modèle apprend l'absence)
EXCLAMATIVE     punct="!" + Mood=Ind                   —
DÉCLARATIVE     défaut absolu                          —
NÉGATIVE        Polarity=Neg (root ou token)           pattern dep_rel=advmod + pos=ADV
                                                       (modèle apprend : advmod négatif)
PASSIVE         Voice=Pass / dep_rel=nsubj:pass        dep_rel=aux:pass
COMPLEXE        dep_rel ∈ {advcl,ccomp,xcomp,acl}      —
TYPE RELATION   —                                      pattern complet :
                                                       (dep_rel(K), pos(K), Mood(V_sub),
                                                        Polarity(V), présence restriction,
                                                        valence(V), dep_rel(S), dep_rel(C))
                                                       → appris depuis données annotées
```

**Ce que la couche 2 apprend — et ce qu'elle n'apprend PAS :**

```
Apprend :
  - La combinaison (dep_rel=mark, pos=SCONJ, Mood(V)=Sub) → souvent condition ou concession
  - La combinaison (dep_rel=mark, pos=SCONJ, Mood(V)=Ind) → souvent cause ou temporel
  - La combinaison (dep_rel=advmod, pos=ADV, Polarity=Neg) → négation fonctionnelle
  - La combinaison (¬nsubj, Mood=Imp, Person=2) → impérative
  - La présence de (dep_rel=nsubj:pass + dep_rel=aux:pass) → voix passive

N'apprend PAS :
  - Que "si" signifie condition  (c'est le pattern structurel qui l'enseigne)
  - Que "ne...pas" signifie négation  (c'est Polarity=Neg dans morph qui l'enseigne)
  - Des règles spécifiques à une langue  (les features UD sont universelles)
```

---

## 8. Types de nœuds du graphe causal

### 8.1 Taxonomie unifiée — 8 types

La taxonomie intègre les types existants du moteur (`action`, `transition`, `etat_systemique`) et les nouveaux types requis par la spécification. La table de migration garantit la compatibilité descendante.

| Type | Définition | Signal UD principal | Exemples | Migration depuis moteur |
|------|-----------|--------------------|-----------|-----------------------|
| `processus` | Action ou transformation dynamique avec durée | VERB transitif ou NOUN déverbal | "prescription", "chute", "hausse" | fusion : `action` + `transition` + `processus` |
| `état_local` | Propriété stable d'une entité spécifique | VERB d'état + cop, ADJ prédicatif | "efficacité", "toxicité", "résistance d'un patient" | renomme : `etat` |
| `état_global` | Condition générale du système ou du domaine | NOUN abstrait systémique, contexte macro | "crise", "récession", "état du marché" | renomme : `etat_systemique` |
| `entité` | Agent, objet ou institution identifiable | NOUN concret non déverbal | "médecin", "médicament", "entreprise" | inchangé : `entite` |
| `condition` | Précondition nécessaire ou suffisante | advcl + mark conditionnel | "si les traitements échouent" | inchangé : `condition` |
| `concept` | Abstraction, mesure, grandeur systémique | NOUN abstrait + contexte quantitatif (numéral, mesure adjacent) | "taux", "demande", "coût marginal" | **NOUVEAU** — absent du moteur actuel |
| `événement` | Fait ponctuel daté ou datable | NOUN + obl:tmod ou apposition temporelle | "l'effondrement de 2008", "la crise du Covid" | **NOUVEAU** — absent du moteur actuel |
| `contrainte` | Restriction exclusive ou condition bloquante | advcl + Polarity=Neg + has_restriction | "ne...que si", restriction + condition conjointe | **NOUVEAU** — absent du moteur actuel |

### 8.2 Table de migration (moteur actuel → taxonomie unifiée)

```
Moteur actuel      → Taxonomie unifiée    Justification
────────────────────────────────────────────────────────────────────────
action             → processus            action = processus dynamique
transition         → processus            transition = processus de changement d'état
processus          → processus            inchangé
etat               → état_local           renommage — propriété d'une entité
etat_systemique    → état_global          renommage — état du système entier
entite             → entité               inchangé
condition          → condition            inchangé
—                  → concept              NOUVEAU : NOUN + contexte quantitatif
—                  → événement            NOUVEAU : NOUN + obl:tmod présent
—                  → contrainte           NOUVEAU : advcl + Neg + restriction
```

### 8.3 Signaux de détection pour les 3 nouveaux types

```
concept :
  NOUN abstrait
  ∧ ¬déverbal (pas de VERB correspondant)
  ∧ ¬nom propre
  ∧ (précédé ou suivi d'un numéral OU d'une unité de mesure)
  Exemples : "taux de 3%", "demande mondiale", "coût marginal"

événement :
  NOUN (ou PROPN)
  ∧ (∃ obl:tmod dans la clause  OU  date/période en apposition)
  Exemples : "l'effondrement (de 2008)", "la crise (du Covid)", "la réforme (de 2020)"

contrainte :
  Proposition advcl
  ∧ Polarity=Neg sur le verbe (root de la clause)
  ∧ has_restriction = True (pattern ne...que ou équivalent)
  Exemple : "ne...que si les traitements ont échoué"
```

### 8.4 Rôles sémantiques dans la relation causale

| Rôle | Définition | dep_rel en actif | dep_rel en passif |
|------|-----------|-----------------|------------------|
| **Agent** | Entité qui initie ou cause l'action | `nsubj` | `obl:agent` |
| **Patient** | Entité qui subit l'action ou l'effet | `obj` | `nsubj:pass` |
| **Thème** | Entité dont on prédicte un état | `nsubj` + cop | — |
| **Résultat** | État final produit par le processus | `xcomp` attributif | `xcomp` |
| **Instrument** | Moyen utilisé pour produire l'action | `obl` + "avec"/"par" | — |
| **Source** | Origine de l'action | `obl` + "de"/"depuis" | — |
| **But** | Objectif visé (non encore atteint) | `advcl` + "pour que" | — |
| **Bénéficiaire** | Entité qui bénéficie du résultat | `iobj` | — |

### 8.5 Hiérarchie multi-échelle

```
Niveau macro    : domaine / secteur              ("santé publique", "finance")
  ↳ Niveau méso : processus / entité principale  ("prescription médicale", "marché")
      ↳ Niveau micro : actions / états atomiques ("prescrire", "chuter", "être efficace")
```

Les requêtes `zoom_in` / `zoom_out` sur le graphe naviguent dans cette hiérarchie.

---

## 9. Types de relations causales

Le modèle est **ternaire** : toute relation est exprimée sous la forme `(sources, target, relation, third?)` où le tiers optionnel encode une condition, un médiateur ou une co-cause. Les relations sans tiers (`third=None`) sont le cas fondamental — relation directe A→B.

### 9.0 Hiérarchie causale de Pearl — mapping des relations

Pearl distingue trois niveaux de causalité. Chaque type de relation appartient à un rung qui détermine sa **confiance de base** et les **requêtes applicables**.

```
RUNG 1 — Association   P(B | A)              OBSERVER
  Sémantique : A et B co-occurrent dans les données. Corrélation, pas causalité prouvée.
  Relations  : cause (faible), sequence, analogy
  Confiance base : 0.50
  Requêtes   : find_causes, find_effects, chain (statistiques observées)
  Exemple    : "Les ventes baissent quand la demande baisse."
               → corrélation observée, pas lien mécaniste établi

RUNG 2 — Intervention   P(B | do(A))          AGIR
  Sémantique : modifier A structurellement change B. Lien mécaniste établi.
  Relations  : condition, enable, prevent, filter, motivation, conditional_cause,
              conditional_prevent, mediated_prevent, joint_prevent,
              data_dependency, control_dependency
  Confiance base : 0.75
  Requêtes   : explain, counterfactual, before?, spof, do
  Exemple    : "Si la demande baisse, les ventes chutent."
               → structure causale explicite (do-calculus applicable)

RUNG 3 — Contrefactuel   P(Bₐ | A=faux)       IMAGINER
  Sémantique : raisonnement sur des scénarios non observés. Hypothèse sur des mondes possibles.
  Relations  : counterfactual, mediated_cause, joint_cause, concession, opposition
  Confiance base : 0.65
  Requêtes   : counterfactual, analogy (structurelle)
  Exemple    : "La pluie cause les inondations via la saturation du sol."
               → mécanisme hypothétique, non directement observable

Équation de confiance unifiée depuis le rung :
  confiance = confiance_rung × f(Mood) × g(connecteur)
  où :
    f(Mood=Ind) = 1.00    g(connecteur_explicite) = 1.00
    f(Mood=Sub) = 0.85    g(VERB_causal_lexical)  = 0.90
    f(Mood=Cnd) = 0.80    g(parataxis)            = 0.60
                          g(aucun_signal)          = 0.50  # aucun signal < parataxe(0.60)

  + 0.10 si advcl_before_root = True  (condition proéminente — Éq. 9)

  Exemples (avant calibration — priors uniquement) :
    CONDITION (Rung 2=0.75) + Mood=Sub + mark explicite :
      0.75 × 0.85 × 1.00 = 0.64
    CAUSE (Rung 1=0.50) + Mood=Ind + mark explicite :
      0.50 × 1.00 × 1.00 = 0.50
    CAUSE (Rung 1=0.50) + Mood=Ind + parataxis :
      0.50 × 1.00 × 0.60 = 0.30
    CAUSE (Rung 1=0.50) + Mood=Ind + aucun signal :
      0.50 × 1.00 × 0.50 = 0.25
    FILTER (Rung 2=0.75) + Mood=Ind + mark + restriction :
      0.75 × 1.00 × 1.00 + 0 = 0.75  [restriction = has_restriction=True, pas ×]
    MEDIATED_CAUSE (Rung 3=0.65) + Mood=Ind + tiers détecté :
      0.65 × 1.00 × 1.00 = 0.65

Cohérence ordinale vérifiée :
  mark_explicite > VERB_causal > parataxis > aucun_signal
  Rung2 > Rung3 > Rung1  (intervention > contrefactuel > association)
  1.00 > 0.90 > 0.60 > 0.50  ✓

Note : PRIORS uniquement. Remplacés par calibration isotonique dès N_val ≥ 100 (§9.5).
```

### 9.1 Relations directes (third = None)

| Relation | Notation | Définition précise | Mood UD | Polarity |
|----------|----------|--------------------|---------|----------|
| `cause` | A → B | A est condition suffisante de B | Ind | Pos |
| `condition` | A ⇒ B | B se produit si et seulement si A | Sub/Ind | Pos |
| `enable` | A ⊃ B | A rend B possible (nécessaire non suffisant) | Ind | Pos |
| `prevent` | A ⊄ B | A empêche B de se produire | Ind | Neg |
| `concession` | A ∧¬(A→¬B) | Malgré A, B se produit quand même | Sub | Pos |
| `motivation` | goal(A)←B | A est entrepris en vue d'atteindre B | Sub/Inf | Pos |
| `sequence` | A ≺ B | A précède B dans le temps | Ind | Pos |
| `filter` | B↑A | B se produit exclusivement si A | Sub/Ind | Neg+restrict |
| `opposition` | A ⊕ B | A s'oppose à B ou le remplace | Ind | — |
| `analogy` | A≈C, B≈D | Structure A→B similaire à C→D | Ind | — |
| `counterfactual` | P(B\|do(¬A)) | Ce qui se passerait sans A | Cnd/Sub | Neg |

### 9.2 Relations ternaires (third = condition | mediator) — joint déprécié

Les relations ternaires expriment ce qu'une relation directe (third=None) ne peut pas capturer sans perte d'information.

| Relation | Notation | Définition | Tiers (rôle) | Signal linguistique |
|----------|----------|------------|--------------|---------------------|
| `conditional_cause` | A→B \| C | A cause B uniquement sous condition C | `condition` | "A cause B si C" / "A → B pourvu que C" |
| `mediated_cause` | A→B via C | A cause B, C est le mécanisme explicite | `mediator` | "A cause B en/par/via C" / "A entraîne B grâce à C" |
| `joint_cause` | (A₁∧A₂)→B | A₁ et A₂ ensemble causent B (ni l'un ni l'autre seul) | `joint` (second source) | "A et B causent C" / "La conjonction de A et B produit C" |

**Illustration des trois patterns ternaires :**

```
CONDITIONAL_CAUSE :
  "Le médicament réduit la douleur uniquement si la dose est correcte."

  Sans tiers — third=None (perd l'information) :
    CIR_edge(médicament → réduction, CAUSE)
    CIR_edge(dose_correcte → réduction, CONDITION)
    → Deux arêtes séparées. Le lien entre A et C est perdu.

  Ternaire (fidèle à la sémantique) :
    CIR_ternary(
      source = médicament
      target = réduction_douleur
      relation = CONDITIONAL_CAUSE
      third = {role: "condition", node: dose_correcte}
    )
    → Encode : médicament cause réduction SOUS CONDITION dose_correcte.

────────────────────────────────────────────────────────────────────────

MEDIATED_CAUSE :
  "La pluie cause les inondations en saturant le sol."

  Sans tiers — chaîne (perd le lien direct) :
    CIR_edge(pluie → saturation_sol)
    CIR_edge(saturation_sol → inondations)
    → A→B direct disparu. On ne sait plus que pluie CAUSE inondations.

  Ternaire (fidèle à la sémantique) :
    CIR_ternary(
      source   = pluie
      target   = inondations
      relation = MEDIATED_CAUSE
      third    = {role: "mediator", node: saturation_sol}
    )
    → Encode : la pluie cause les inondations,
               la saturation du sol est le mécanisme.

────────────────────────────────────────────────────────────────────────

JOINT_CAUSE :
  "La hausse des coûts et la baisse de la demande ont provoqué la faillite."

  Sans tiers (perd la nécessité conjointe) :
    CIR_edge(hausse_coûts → faillite)
    CIR_edge(baisse_demande → faillite)
    → Suggère que chaque cause seule suffit. Faux.

  Ternaire (fidèle à la sémantique) :
    CIR_ternary(
      sources  = [hausse_coûts, baisse_demande]
      target   = faillite
      relation = JOINT_CAUSE
      third    = None   [le tiers = le second source, dans sources[1]]
    )
    → Encode : les DEUX sont nécessaires conjointement.
               Ni hausse_coûts seul, ni baisse_demande seul n'y suffit.
```

### 9.3 Relations systémiques

| Relation | Définition | Rung |
|----------|-----------|------|
| `data_dependency` | A utilise les données produites par B | 1 |
| `control_dependency` | A conditionne l'exécution de B | 2 |

### 9.4 Algèbre de négation ternaire

La négation d'une relation ternaire dépend du **nœud sur lequel porte la négation** — cible, condition ou source. Trois règles distinctes.

```
RÈGLE 1 — Négation sur la CIBLE (B)
  Sémantique : A produit l'absence de B (ou empêche B de se produire)

  Direct (third=None)                       → PREVENT(A, B)
  CONDITIONAL_CAUSE(A, B, cond=C) + Neg(B)  → CONDITIONAL_PREVENT(A, B, cond=C)
    "A empêche B uniquement si C"
  MEDIATED_CAUSE(A, B, via=M) + Neg(B)      → MEDIATED_PREVENT(A, B, via=M)
    "A empêche B par l'intermédiaire de M"
  JOINT_CAUSE([A₁,A₂], B) + Neg(B)         → JOINT_PREVENT([A₁,A₂], B)
    "A₁ et A₂ ensemble empêchent B"

RÈGLE 2 — Négation sur la CONDITION (C dans third)
  Sémantique : la relation est vraie quand la condition est fausse

  CONDITIONAL_CAUSE(A, B, cond=C) + Neg(C)
    → CONDITIONAL_CAUSE(A, B, cond=¬C)
    → third = {role:"condition", node:C, polarity:"negative"}
    "A cause B uniquement si C ne se produit PAS"
    Exemple : "Le médicament est efficace uniquement si le patient
               ne présente PAS de contre-indication."

RÈGLE 3 — Négation sur la SOURCE (A)
  Sémantique : l'absence de A est elle-même une cause/condition

  Direct CAUSE(A, B) + Neg(A) → COUNTERFACTUAL(¬A, B) (Rung 3)
    "En l'absence de A, B ne se produit pas"
  JOINT_CAUSE([A₁,A₂], B) + Neg(A₁)
    → sources = [{node:A₁, polarity:"negative"}, {node:A₂, polarity:"positive"}]
    "L'absence de A₁ ET la présence de A₂ causent B"
    Exemple : "Sans la pluie, mais avec les fontes des neiges,
               des inondations se produisent."

Types ternaires issus de la négation (à ajouter au RelationType) :
  conditional_prevent  → CONDITIONAL_CAUSE + Neg(cible)
  mediated_prevent     → MEDIATED_CAUSE + Neg(cible)
  joint_prevent        → JOINT_CAUSE + Neg(cible)

Illustration :
  "La dose correcte empêche la toxicité uniquement si le patient est adulte."
  → CONDITIONAL_PREVENT(dose_correcte, toxicité, cond=patient_adulte)
    sources = [dose_correcte]
    target  = toxicité
    relation = conditional_prevent
    third   = {role:"condition", node:patient_adulte}
    polarity = "negative"  (sur la cible toxicité)
```

### 9.5 Calibration des confiances

Les valeurs de confiance sont des **hyperparamètres** initialisés par les priors de rung. Ils sont calibrés empiriquement par régression isotonique sur l'ensemble de validation.

```
Procédure :
  1. Entraîner le modèle sur le corpus d'entraînement
  2. Sur l'ensemble de validation (N_val ≥ 100) :
     Pour chaque phrase → prédire la relation → calculer confiance_brute
     y_true = 1 si relation_prédite == relation_réelle, 0 sinon
  3. Régression isotonique :
     confiance_calibrée = IsotonicRegression(confiance_brute, y_true)
     → garantit que P_calibrée est monotone croissante avec P_correcte
  4. Remplacer les priors de l'Éq. 5 par les valeurs calibrées

État avant calibration (N_val < 100) : utiliser les priors de §9.0
État après calibration (N_val ≥ 100) : utiliser les valeurs calibrées

Propriété garantie :
  confiance_calibrée = P(relation_correcte | features, rung)
  → La confiance est une probabilité réelle, pas un proxy intuitif.
```

### 9.6 Patterns structurels appris → relation causale

Le modèle apprend ces associations depuis les données annotées. La colonne "Lemme (exemple)" est donnée à titre indicatif pour la compréhension. **Couche 1** (déterministe) : zéro lemme — uniquement features UD structurelles. **Couche 2** (apprise) : peut utiliser `lemma_emb[3×128]` (§11.3) mais jamais comme règle statique `if lemma=X then Y`.

```
Pattern structurel (features UD)                              Relation apprise
─────────────────────────────────────────────────────────────────────────────────
── Relations directes (third=None) ──────────────────────────────────────────────
mark, SCONJ, advcl_Mood=Sub, ¬Neg, ¬goal                  → CONDITION ou CONCESSION
mark, SCONJ, advcl_Mood=Sub, ¬Neg, has_goal               → MOTIVATION
mark, SCONJ, advcl_Mood=Ind, ¬Neg, Tense=—                → CAUSE
mark, SCONJ, advcl_Mood=Ind, ¬Neg, Tense=Past             → SEQUENCE
mark, SCONJ, advcl_Mood=Ind, ¬Neg, Tense=Pres             → CAUSE ou ENABLE
mark, SCONJ, advcl_Mood=Sub, Neg+restrict                  → FILTER
case, ADP,   Mood=Ind, ¬Neg                                → CONCESSION
cc,   CCONJ, conj, Mood=Ind                                → CAUSE ou SEQUENCE ou CONCESSION
advmod, ADV, Polarity=Neg, ¬restrict                       → PREVENT
advmod, ADV, Polarity=Neg, has_restrict                    → FILTER
parataxis                                                  → CAUSE  [conf=0.30 = 0.50×1.0×0.60, avant calib.]
VERB, has_obj, ¬mark, ¬cc, ¬Neg                           → CAUSE ou ENABLE
VERB, has_obj, ¬mark, ¬cc, Polarity=Neg                   → PREVENT

── Relations ternaires (third présent) ───────────────────────────────────────────
VERB causal + mark=cond + obl/advcl mécanisme             → CONDITIONAL_CAUSE
  (deux subordonnées : une condition + un médiateur)         third={role:condition}

VERB + obl avec pos=ADP, "en/par/via" + VERB mécanisme    → MEDIATED_CAUSE
  (obl de mécanisme explicite)                               third={role:mediator}

deux NOUN/sources + cc="et" + VERB transitif commun       → JOINT_CAUSE
  (co-sujets coordonnés partageant le même verbe)            sources=[N₁, N₂]

Exemples d'apprentissage (lemmes pédagogiques uniquement — non utilisés) :
  "si"              → mark, SCONJ, Mood=Sub/Ind → CONDITION (third=None)
  "bien que"        → mark, SCONJ, Mood=Sub     → CONCESSION (third=None)
  "parce que"       → mark, SCONJ, Mood=Ind     → CAUSE (third=None)
  "quand"           → mark, SCONJ, temporel     → SEQUENCE (third=None)
  "ne...que si"     → Neg+restrict + mark       → FILTER (third=None)
  "A cause B si C"  → VERB causal + mark=cond   → CONDITIONAL_CAUSE (ternaire)
  "A via/en C → B"  → VERB + obl mécanisme      → MEDIATED_CAUSE (ternaire)
  "A et B causent C"→ deux sources + cc="et"    → JOINT_CAUSE (ternaire)
```

---

## 10. Système d'équations relationnelles

### Équation 0 — Décomposition universelle (modèle ternaire)

Toute phrase se décompose depuis le verbe root en un quintuplet. Le modèle est **ternaire** : le tiers T capture le troisième élément structural (condition, médiateur, ou co-cause).

```
Φ = (S, V, C*, K?, T?)

  S  =  sujet(s)    → {nsubj, nsubj:pass, expl, csubj}
                       ou [S₁, S₂] si deux sujets coordonnés (joint_cause)
  V  =  verbe root  → pivot central
  C  =  compléments → {obj, iobj, obl, xcomp, ccomp, advcl}*
  K  =  connecteur  → {mark, cc, parataxis}?
  T  =  tiers       → {role, proposition}?                  ← NOUVEAU
           role      : "condition" | "mediator"   # "joint" déprécié → sources[1]
           proposition: la clause ou le nœud tiers

  Cas selon T :
    T = None          → relation directe (third=None) : A → B
    T.role=condition  → CONDITIONAL_CAUSE (A → B | C)
    T.role=mediator   → MEDIATED_CAUSE   (A → B via C)
    T.role=joint      → JOINT_CAUSE      ([A₁,A₂] → B)

  Illustration — les quatre cas :

    Φ_direct         = (pluie,        cause,  inondation,  None,  None)
    Φ_conditional    = (médicament,   réduit, douleur,     si,    T{cond=dose})
    Φ_mediated       = (pluie,        cause,  inondation,  en,    T{med=saturation})
    Φ_joint          = ([coûts,demande], cause, faillite,  et,    None)
```

### Équation 1 — Type de phrase

```
TypeΦ(V) =

  INTERROGATIVE  si  punct(Φ) = "?"
                 ∨   ∃t : morph(t).PronType = Int
                         ∧ dep_rel(t) ∈ {advmod, det, nsubj, obj}
                 ∨   ∃t : dep_rel(t) = nsubj ∧ t.id > V.id     [inversion]

  IMPÉRATIVE     si  morph(V).Mood = Imp
                 ∨   (morph(V).VerbForm = Inf ∧ ¬∃nsubj(Φ))
                 ∨   (¬∃nsubj(Φ) ∧ morph(V).Person ∈ {1,2}
                         ∧ morph(V).Mood ≠ Ind)

  EXCLAMATIVE    si  punct(Φ) = "!" ∧ morph(V).Mood ≠ Imp

  DÉCLARATIVE    sinon                                           [défaut absolu]
```

**Priorité d'évaluation :** dans l'ordre ci-dessus. La première condition vraie détermine le type.

### Équation 2 — Rôles sémantiques

La voix du verbe pivote les rôles :

```
Rôles(S, V, C) =

  Si morph(V).Voice = Act  ∨  dep_rel(S) = nsubj :
    Agent    ← S[nsubj]
    Patient  ← C[obj]
    Récip.   ← C[iobj]            (si présent)

  Si morph(V).Voice = Pass  ∨  dep_rel(S) = nsubj:pass :
    Patient  ← S[nsubj:pass]
    Agent    ← C[obl:agent]?      (optionnel — peut être effacé)

  Si dep_rel(V) = cop :
    Thème    ← S[nsubj]
    Attribut ← C[xcomp ou ADJ prédicatif]

  Si dep_rel(S) ∈ {expl, expl:subj, expl:pass} :
    Agent    ← ∅                  (impersonnel — entité générique)
    Thème    ← C[ccomp, xcomp, obl]
```

**Invariant formel :**
```
∀ Φ_actif, Φ_passif décrivant le même événement :
  Rôles(Φ_actif).Agent   ≡  Rôles(Φ_passif).Agent
  Rôles(Φ_actif).Patient ≡  Rôles(Φ_passif).Patient
```

### Équation 3 — Type de nœud

```
TypeNœud(x) =

  PROCESSUS    si  x est VERB ou NOUN déverbal
               ∧   valence(x.V) ∈ {transitif, ditransitif, intransitif_dynamique}

  ÉTAT_LOCAL   si  x est VERB d'état (être, sembler, paraître, devenir)
               ∨   x est ADJ prédicatif via cop
               ∧   contexte = propriété d'une entité spécifique

  ÉTAT_GLOBAL  si  x est NOUN abstrait systémique
               ∧   contexte = condition générale du système ou du domaine
               (ex : crise, récession, état du marché — §8.1)

  ENTITÉ       si  x est NOUN concret non déverbal

  CONDITION    si  x est proposition advcl
               ∧   dep_rel(mark(x)) ∈ markers.condition

  CONCEPT      si  x est NOUN abstrait, mesure ou grandeur systémique

  ÉVÉNEMENT    si  x est NOUN ∧ ∃obl:tmod(x)

  CONTRAINTE   si  x est proposition advcl
               ∧   morph(V_x).Polarity = Neg
               ∧   restriction(x)
```

### Équation 4 — Type de relation

L'équation centrale de l'extraction. Deux couches : règles déterministes sur les features UD (couche 1, zéro lemme), puis classifieur appris sur les patterns structurels et les embeddings de lemmes (couche 2). **Couche 1 : aucun lemme sous aucune forme. Couche 2 : embeddings continus autorisés, listes statiques interdites.**

```
TypeR(V₁, V₂, K) =

  ── Couche 1 : features UD déterministes (priorité absolue) ────────────

  Polarity(V₂) = Neg  ∧  has_restriction(tokens)       →  FILTER
  Polarity(V₂) = Neg  ∧  ¬has_restriction(tokens)      →  PREVENT
  dep_rel(K) = parataxis                                →  CAUSE  [conf=0.30 = Rung1×g(parataxis), avant calib.]

  ── Couche 2 : classifieur appris sur features UD ──────────────────────

  Le modèle reçoit FeatureVector(Φ) — features UD structurelles (couche 1) + embeddings de lemmes (couche 2) — et prédit le type de relation.
  Il apprend les patterns suivants depuis les données annotées :

  Pattern appris → Relation prédite
  ───────────────────────────────────────────────────────────────────────
  (mark, SCONJ, advcl_mood=Sub, ¬neg, ¬restriction)   → CONDITION ou CONCESSION ou MOTIVATION
  (mark, SCONJ, advcl_mood=Ind, ¬neg)                 → CAUSE ou SEQUENCE ou ENABLE
  (cc,   CCONJ, conj, ¬neg)                            → CAUSE ou CONCESSION ou SEQUENCE
  (mark, SCONJ, advcl_mood=Sub, neg, restriction)      → FILTER
  (root=VERB, obj, ¬mark, ¬neg, transitive)            → CAUSE ou ENABLE (verbe causal)
  (root=VERB, obj, ¬mark, neg, transitive)             → PREVENT (verbe causal négatif)
  (cop, attribut, ¬mark)                               → ÉTAT_LOCAL ou ÉTAT_GLOBAL (nœud d'état, pas de relation causale directe)

  La distinction fine au sein des patterns ci-dessus (ex: CONDITION vs CONCESSION
  quand advcl_mood=Sub) est apprise depuis les exemples annotés.
  Ce n'est pas le lemme du connecteur qui la détermine —
  c'est la combinaison de (Mood du verbe subordonné, Aspect, Tense, Polarity,
  présence d'une restriction, valence du verbe principal).

  ── Défaut ────────────────────────────────────────────────────────────

  ∅  →  CAUSE  [conf=0.25 = Rung1×g(aucun_signal), avant calib.]
```

### Équation 5 — Extraction causale complète (modèle ternaire)

```
Extraire(Φ_déclarative) =

  ── Étape 1 : Identifier le pattern ─────────────────────────────────

  Si S = [S₁, S₂] coordonnés (cc="et" entre deux sujets/sources) :
    → pattern JOINT_CAUSE
    nœud_a₁  ← TypeNœud(S₁)
    nœud_a₂  ← TypeNœud(S₂)
    nœud_b   ← TypeNœud(Patient)
    → CIR_ternary([nœud_a₁, nœud_a₂], nœud_b, JOINT_CAUSE, conf)

  Sinon si T.role = "condition" détecté (obl/advcl mécanisme conditionnel) :
    → pattern CONDITIONAL_CAUSE
    nœud_a   ← TypeNœud(Agent)
    nœud_b   ← TypeNœud(Patient)
    nœud_c   ← TypeNœud(T.proposition)
    → CIR_ternary(nœud_a, nœud_b, CONDITIONAL_CAUSE, conf,
                  third={role:"condition", node:nœud_c})

  Sinon si T.role = "mediator" détecté (obl avec "en/par/via") :
    → pattern MEDIATED_CAUSE
    nœud_a   ← TypeNœud(Agent)
    nœud_b   ← TypeNœud(Patient)
    nœud_c   ← TypeNœud(T.proposition)
    → CIR_ternary(nœud_a, nœud_b, MEDIATED_CAUSE, conf,
                  third={role:"mediator", node:nœud_c})

  Sinon :
    → pattern direct (third=None)
    (Agent, Patient) ← Rôles(S, V, C)           [Éq. 2]
    nœud_a           ← TypeNœud(Agent)           [Éq. 3]
    nœud_b           ← TypeNœud(Patient)         [Éq. 3]
    relation         ← TypeR(V, advcl(V), K)     [Éq. 4]
    → CIR_ternary(nœud_a, nœud_b, relation, conf, third=None)

  ── Étape 2 : Calculer la confiance (formule unique §9.0) ────────────

  confiance = confiance_rung(relation) × f(Mood) × g(connecteur) + Δ_saillance

  confiance_rung :
    Rung 1 (CAUSE, SEQUENCE, ANALOGY)                      = 0.50
    Rung 2 (CONDITION, ENABLE, PREVENT, FILTER,
            CONDITIONAL_CAUSE, MOTIVATION)                 = 0.75
    Rung 3 (COUNTERFACTUAL, MEDIATED_CAUSE, JOINT_CAUSE,
            CONCESSION, OPPOSITION)                        = 0.65

  f(Mood) : Ind=1.00 / Sub=0.85 / Cnd=0.80

  g(connecteur) : explicite=1.00 / VERB_causal_lexical=0.90
                  parataxis=0.60 / aucun_signal=0.50

  Δ_saillance : +0.10 si advcl_before_root=True (condition proéminente)

  Note : ces valeurs sont des PRIORS. Remplacées par calibration isotonique
         dès N_val ≥ 100 (§9.5). Ne pas modifier les priors manuellement.

  Exemples :
    CAUSE(Rung1=0.50) + Mood=Ind + mark explicit :
      0.50 × 1.00 × 1.00 = 0.50  → Rung 1 pur, confiance modérée
    CONDITION(Rung2=0.75) + Mood=Sub + mark explicit :
      0.75 × 0.85 × 1.00 = 0.64  → Sub réduit la certitude
    FILTER(Rung2=0.75) + Mood=Ind + mark + restriction + frontée :
      0.75 × 1.00 × 1.00 + 0.10 = 0.85
```

**Illustration — les quatre cas d'extraction :**

```
CAS 1 — Direct (third=None)
  "La pluie cause les inondations."
  Φ = (pluie, cause, inondation, None, None)
  → CIR_ternary(pluie, inondation, CAUSE, 1.0, third=None)

CAS 2 — Conditional_cause
  "La pluie cause des inondations uniquement si le sol est saturé."
  Φ = (pluie, cause, inondation, si, T{cond=saturation})
  → CIR_ternary(pluie, inondation, CONDITIONAL_CAUSE, 0.9,
                third={role:"condition", node:saturation_sol})

CAS 3 — Mediated_cause
  "La pluie cause les inondations en saturant le sol."
  Φ = (pluie, cause, inondation, en, T{med=saturation})
  → CIR_ternary(pluie, inondation, MEDIATED_CAUSE, 0.9,
                third={role:"mediator", node:saturation_sol})

CAS 4 — Joint_cause
  "La pluie et la fonte des neiges causent les inondations."
  Φ = ([pluie, fonte_neiges], cause, inondation, et, None)
  → CIR_ternary([pluie, fonte_neiges], inondation, JOINT_CAUSE, 0.9,
                third=None)
```

### Équation 6 — Classification d'intention (tête MLP apprise — moteur core)

L'intent est **appris** par le moteur à partir des annotations `SentenceRecord.intent`.
Zéro lemme, zéro config externe — langue-agnostique par construction.

```
Intent(Φ_interrogative) =

  Input  : clause_vec[d_clause=106]  ← même vecteur que la tête nœuds
  Output : logit[N_intents=21]       ← argmax → intent_type ∈ INTENT_TYPES

  Architecture (non-breaking, n_intent_types=0 par défaut) :
    d_clause → mlp_hidden → 64 → 21   [He-init, même pattern tête nœuds]

  Supervision  : SentenceRecord.intent ∈ INTENT_TYPES  ("" = non annoté)
  N_min D4     : un type n'est entraîné qu'à N ≥ N_min exemples annotés
  Persistance  : intent_layer_*_W / intent_layer_*_b dans le checkpoint

  Dispatch     : intent_idx = argmax(softmax(logit[0])) → commande DSL Pearl

INTENT_TYPES (21 types ordonnés par catégorie) :

  Causal       : explain        — Pourquoi X ? (WHY backward)
                 effects        — Quelles conséquences de X ? (WHAT forward)
                 abduct         — Qu'aurait pu causer X ? (abductif)
                 counterfactual — Sans X, que se passe-t-il ?

  Path         : chain          — Comment A mène à B ? (chemin causal)
                 chain_t        — Dans quel ordre temporel ?
                 before         — A précède-t-il B ?
                 delay          — Délai estimé A→B ?

  Structure    : spof           — Nœud le plus critique ?
                 centrality     — Nœud le plus central ?
                 analogy        — Pattern similaire à A→B ?

  Méta-qualité : summarize      — Vue d'ensemble du graphe
                 density        — Le graphe est-il dense ?
                 coverage       — X est-il bien couvert ?
                 reliability    — Les réponses sur X sont-elles fiables ?
                 diff           — Y a-t-il un écart normatif ?

  Navigation   : zoom_in        — Détail de X
                 zoom_out       — Vue d'ensemble de X
                 aggregate      — X et ses sous-composants

  Génération   : verbalize      — Décris le graphe en texte naturel

  Extraction   : none           — Phrase déclarative → extraction CIR
```

### Équation 7 — Normalisation voix active/passive

```
Normaliser(Φ) =

  Si dep_rel(S) = nsubj:pass :
    Agent   ← token avec dep_rel = obl:agent   (∅ si absent)
    Patient ← S[nsubj:pass]
  Sinon :
    Agent   ← S[nsubj]
    Patient ← C[obj]

  Résultat : CIR_edge(Agent, Patient, R)  identique en actif et en passif
```

### Équation 8 — Phrase complexe ternaire (deux ou trois propositions)

```
Extraire(Φ_complexe) =

  Φ₁ = (S₁, V₁, C₁)   [proposition principale]
  Φ₂ = (S₂, V₂, C₂)   [subordonnée : advcl / ccomp / xcomp]
  K   = mark(V₂)        [connecteur sur la subordonnée]

  nœud_1   ← Extraire(Φ₁)      [récursif — peut être lui-même ternaire]
  nœud_2   ← Extraire(Φ₂)
  relation ← TypeR(V₁, V₂, K)              [Éq. 4]

  ── Cas 1 : relation simple → ternaire avec tiers=condition ──────────

  K ∈ {condition, cause, filter} :
    → CIR_ternary(
        source   = nœud_2         [Φ₂ est la source causale]
        target   = nœud_1         [Φ₁ est la cible]
        relation = relation
        third    = None           [sauf si troisième proposition présente]
      )

  K ∈ {enable, sequence, motivation} :
    → CIR_ternary(nœud_1, nœud_2, relation, conf, third=None)

  K ∈ {concession} :
    → CIR_ternary(nœud_2, nœud_1, CONCESSION, conf, third=None)

  K = parataxis :
    → CIR_ternary(nœud_1, nœud_2, CAUSE, 0.4, third=None)

  ── Cas 2 : trois propositions → tiers explicite ─────────────────────

  Si Φ₃ présent (deuxième subordonnée) :
    K₁ = mark(Φ₂)    [connecteur principal : condition/cause/...]
    K₂ = mark(Φ₃)    [connecteur du tiers : médiateur/exception/...]

    Si K₂ = mediator (en/par/via) :
      → CIR_ternary(nœud_2, nœud_1, MEDIATED_CAUSE, conf,
                    third={role:"mediator", node:nœud_3})

    Si K₂ = exception (sauf si/à moins que) :
      → CIR_ternary(nœud_2, nœud_1, CONDITIONAL_CAUSE, conf,
                    third={role:"condition", node:nœud_3})

  ── Illustration ─────────────────────────────────────────────────────

  Deux propositions (relation directe, third=None) :
    "Si la demande baisse, les ventes chutent."
    Φ₂=demande_baisse, Φ₁=ventes_chutent, K=si
    → CIR_ternary(demande_baisse, ventes_chutent, CONDITION, 0.9, None)

  Trois propositions (ternaire explicite) :
    "Les ventes chutent si la demande baisse, à cause de la crise."
    Φ₁=ventes_chutent, Φ₂=demande_baisse (K₁=si), Φ₃=crise (K₂=à cause de)
    → CIR_ternary(demande_baisse, ventes_chutent, CONDITIONAL_CAUSE, 0.9,
                  third={role:"mediator", node:crise})

  Pattern "ne...que si" (filter ternaire) :
    "Les ventes ne chutent que si la demande baisse et les stocks augmentent."
    [S₁,S₂]=[demande_baisse, stocks_augmentent], target=ventes_chutent
    → CIR_ternary([demande_baisse, stocks_augmentent],
                  ventes_chutent, JOINT_CAUSE+FILTER, 0.9,
                  third=None)   [les deux sources sont dans sources[]]
```

### Équation 9 — Saillance positionnelle

La position des éléments dans la phrase porte une information sémantique et pragmatique. Cette équation qualifie chaque nœud et chaque relation extraits avec leur degré de **proéminence informationnelle** — ce qui est mis en avant vs ce qui est présupposé.

```
Saillance(Φ) =

  ── Saillance du sujet ────────────────────────────────────────────────
  subj_before_root = True    → sujet = TOPIC (information donnée)
                                 → nœud extrait = entrée du graphe (cause probable)
  subj_after_root  = True    → sujet postposé → signal interrogatif ou emphatique

  ── Saillance de la subordonnée ───────────────────────────────────────
  advcl_before_root = True   → subordonnée en TÊTE de phrase
                                 → cause/condition PROÉMINENTE
                                 → confiance += 0.1 (locuteur l'a mise en avant)
  advcl_after_root  = True   → subordonnée en QUEUE de phrase
                                 → cause/condition secondaire ou clarification

  ── Saillance de l'objet ──────────────────────────────────────────────
  obj_before_root   = True   → topicalisation de l'objet
                                 → Patient = TOPIC → nœud d'entrée du graphe
  obj_after_root    = True   → ordre canonique V-O → Patient = rhème (information nouvelle)

  ── Saillance de la négation ──────────────────────────────────────────
  neg_before_root   = True   → négateur pré-verbal (ne)
                                 → portée négative sur tout le prédicat
  neg_after_root    = True   → négateur post-verbal (pas, jamais)
                                 → spécification de la portée

  ── Saillance des éléments périphériques ─────────────────────────────
  premier token non-DET     → candidat TOPIC de la phrase
  dernier token non-PUNCT   → candidat FOCUS de la phrase (information la plus nouvelle)

  ── Saillance de la relation causale ─────────────────────────────────
  Structure canonique   : principale → subordonnée
    "Le médicament est prescrit si les traitements échouent."
    → effet annoncé avant la condition → relation FILTER backgroundée

  Structure inversée    : subordonnée → principale
    "Si les traitements échouent, le médicament est prescrit."
    → condition annoncée en premier → condition PROÉMINENTE
    → confiance += 0.1 sur la relation CONDITION/FILTER

  → CIR_edge annotée avec salience : {topic_node, focus_node, condition_prominence}
```

### Équation 10 — Relation discursive inter-phrasale

Étend l'extraction aux **paires de phrases consécutives**. Opère sur `(Φᵢ₋₁, Φᵢ, D?)` où D est le connecteur discursif optionnel en position initiale de Φᵢ.

```
Extraire_discours(Φᵢ₋₁, Φᵢ, D?) =

  nœud_pivot(Φ) = focus_node(Φ)     [dernier constituant informatif non-PUNCT — Éq. 9]

  ── Cas 1 : connecteur discursif explicite D ──────────────────────────
  D présent ∧ D.pos ∈ {CCONJ, ADV} ∧ D.id ≤ 2  [position initiale de Φᵢ]

    D ∈ {donc, par conséquent, ainsi, c'est pourquoi, de ce fait}
      → CIR_ternary(nœud_pivot(Φᵢ₋₁), nœud_pivot(Φᵢ), CAUSE, 0.80, None)

    D ∈ {cependant, néanmoins, pourtant, toutefois, malgré cela}
      → CIR_ternary(nœud_pivot(Φᵢ₋₁), nœud_pivot(Φᵢ), CONCESSION, 0.80, None)

    D ∈ {ensuite, puis, après, d'abord, premièrement, enfin}
      → CIR_ternary(nœud_pivot(Φᵢ₋₁), nœud_pivot(Φᵢ), SEQUENCE, 0.80, None)

    D ∈ {en effet, car, parce que}  [justification rétrospective]
      → CIR_ternary(nœud_pivot(Φᵢ), nœud_pivot(Φᵢ₋₁), CAUSE, 0.75, None)
        [direction inversée : Φᵢ justifie Φᵢ₋₁]

  ── Cas 2 : anaphore (pronom en début de Φᵢ réfère à un nœud de Φᵢ₋₁) ─
  Φᵢ commence par {il, elle, ils, elles, ce, cela, ceci, ça}
  ∧ nœud_pivot(Φᵢ₋₁) est une entité ou un processus

    → Lier les deux CIR : le pronom = co-référence de nœud_pivot(Φᵢ₋₁)
    → Relation = CAUSE si Φᵢ₋₁ précède logiquement Φᵢ (confiance=0.50)

  ── Cas 3 : juxtaposition sans connecteur ni anaphore ─────────────────
    → CIR_ternary(nœud_pivot(Φᵢ₋₁), nœud_pivot(Φᵢ), CAUSE, 0.35, None)
       [Rung 1, relation implicite — confiance basse]

  Fenêtre contextuelle : k=2 phrases par défaut (i-1, i)
  Extension : k=3 si connecteur discursif présent à chaque frontière

  Illustration :
    Φᵢ₋₁ : "Les traitements conventionnels ont échoué."
    Φᵢ   : "C'est pourquoi le médecin a prescrit ce médicament."
            [D = "c'est pourquoi", pos=ADV, id=1]
    → CIR_ternary(échec_traitements, prescription_médicament, CAUSE, 0.80)
```

### Équation 11 — Résolution d'ambiguïté

Quand le classifieur ne peut pas trancher entre deux types de relation, l'ambiguïté est **encodée explicitement dans le CIR** plutôt que forcée vers un seul type.

```
θ_ambiguity = 0.65  (hyperparamètre — calibré sur validation, initialisé à 0.65)

Résolution(features, candidats) =

  P_max = max(P(typeᵢ | features)) sur tous les types candidats

  Si P_max ≥ θ_ambiguity :
    → CIR_edge(
          relation   = argmax(candidats),
          ambiguous  = False,
          candidates = None
      )

  Si P_max < θ_ambiguity :
    → CIR_edge(
          relation   = type_le_plus_probable,
          ambiguous  = True,
          candidates = [(type₁, P₁), (type₂, P₂), ...]
                        triés par probabilité décroissante
      )

  Résolution par cohérence de graphe (si ambiguous=True) :
    Si ∃ arête voisine de même type que candidat₁ liée aux mêmes nœuds
      → favoriser candidat₁ (cohérence locale)
    Si ∃ arête voisine de même type que candidat₂
      → favoriser candidat₂
    Sinon : conserver les deux candidats, confiance partagée :
      P(candidat₁) = P₁ / (P₁ + P₂)
      P(candidat₂) = P₂ / (P₁ + P₂)

  Cas typique résolu par cette équation :
    CONDITION vs CONCESSION sans Mood : P(CONDITION)=0.52, P(CONCESSION)=0.48
    → 0.52 < θ_ambiguity=0.65 → ambiguous=True
    → Le raisonneur Pearl test les deux hypothèses et retourne celle
      cohérente avec le contexte du graphe

  Illustration :
    "Bien que le traitement soit lourd, il est efficace."
    P(CONCESSION)=0.58, P(CONDITION)=0.42  → ambiguous=True
    Contexte graphe : arêtes voisines = CONCESSION × 3 → résoudre vers CONCESSION
```

### Équation 12 — Langage de requêtes ternaire

Extension des requêtes Pearl pour les arêtes ternaires. Chaque requête gère les trois patterns `CONDITIONAL_CAUSE`, `MEDIATED_CAUSE`, `JOINT_CAUSE` explicitement.

```
WHY(B) — causes de B
────────────────────────────────────────────────────────────
  CAUSE(A→B)                       → retourner A
  CONDITIONAL_CAUSE(A→B | cond=C)  → retourner "A sous condition C"
                                      (A ET C sont tous les deux requis)
  MEDIATED_CAUSE(A→B via M)        → retourner "A (mécanisme : M)"
                                      (M explique comment A cause B)
  JOINT_CAUSE([A₁,A₂]→B)          → retourner "A₁ ET A₂ conjointement"
                                      (ni l'un ni l'autre seul ne suffit)

CHAIN(X, Y) — chemin causal de X à Y
────────────────────────────────────────────────────────────
  Relations directes (third=None) : traversée standard du graphe
  JOINT_CAUSE([A₁,A₂]→B) dans le chemin :
    Valide uniquement si X peut atteindre A₁ ET A₂ séparément
    → Deux sous-chemins convergents requis
    → Si un seul sous-chemin existe : chemin INVALIDE (nécessité conjointe)
  CONDITIONAL_CAUSE(A→B|C) dans le chemin :
    Annoter le segment avec la condition C requise
    Retourner : [X →…→ A] ∧ condition(C) →[CONDITIONAL_CAUSE]→ B →…→ Y

COUNTERFACTUAL(¬A) — effets de la suppression de A
────────────────────────────────────────────────────────────
  Pour toute arête ayant A dans sources[] :
    CAUSE(A→B)                    : supprimer l'arête → B disparaît
    JOINT_CAUSE([A, A₂]→C)       : sources → [A₂] seul
                                    → nécessité conjointe cassée → C ne se produit plus
                                    (même si A₂ reste présent)
    CONDITIONAL_CAUSE(A→B|C)     : si C reste présent → B disparaît (A supprimé)
                                    la condition C seule ne produit pas B sans A
    MEDIATED_CAUSE(A→B via M)    : propager la suppression sur M
                                    si M est causé uniquement par A → M disparaît aussi
                                    si M a d'autres sources → M reste, B peut rester

SPOF — Single Point of Failure (nœuds critiques)
────────────────────────────────────────────────────────────
  Extension ternaire :
    Un nœud N est SPOF si :
      N est source d'au moins 1 JOINT_CAUSE
      → sa suppression casse la contrainte de nécessité conjointe
      → tous les effets de ce JOINT_CAUSE disparaissent
    Un nœud N est SUPER-SPOF si :
      N est source à la fois d'relations directes (third=None) ET de JOINT_CAUSE
      → suppression à effet cumulatif (direct + ternaire avec tiers)

EXPLAIN(A, B) — expliquer la relation entre A et B
────────────────────────────────────────────────────────────
  Si ∃ CIR_ternary(A, B, MEDIATED_CAUSE, via=M) :
    → Retourner : "A cause B par l'intermédiaire de M"
  Si ∃ CIR_ternary(A, B, CONDITIONAL_CAUSE, cond=C) :
    → Retourner : "A cause B sous condition C"
  Si ∃ CIR_ternary([A, A₂], B, JOINT_CAUSE) :
    → Retourner : "A et A₂ causent conjointement B"
```

### Vue synthétique — Pipeline complet

```
Texte brut
    │
    ▼  Tokenisation + annotation UD
Tokens  {id, form, lemma, pos, dep_rel, morph{Mood,Voice,Polarity,PronType,...}, dep_head}
    │
    ▼  [Éq. 0]  Localiser V(root) : dep_rel="root" ou dep_head=0
    │
    ├──[Éq. 1]──→ TypeΦ(V.morph, punct, nsubj.id)
    │                  │
    │                  ├─ INTERROGATIVE  →  [Éq. 6]  Router vers requête graphe
    │                  ├─ IMPÉRATIVE     →  Dispatcher vers action
    │                  └─ DÉCLARATIVE    →  [Éq. 5]  Extraction causale
    │
    ├──[Éq. 2]──→ Rôles(S, V, C) = (Agent, Patient, ...)
    │
    ├──[Éq. 3]──→ TypeNœud(Agent), TypeNœud(Patient)  →  8 types
    │
    ├──[Éq. 4]──→ TypeR(V, advcl, K, markers)          →  11 types
    │
    ├──[Éq. 7]──→ Normaliser voix (actif / passif)
    │
    ├──[Éq. 9]──→ Saillance positionnelle (topic / focus / condition_prominence)
    │              subj_before_root, advcl_before_root, neg_before_root...
    │
    └──[Éq. 5/8]─→ CIR_edge(nœud_a, nœud_b, relation, confiance, salience)
                         │
                         ▼
                   Graphe causal (DAG orienté, nœuds typés, saillance annotée)
                         │
                         ▼
                   Raisonnement Pearl
                   WHY / CHAIN / SPOF / COUNTERFACTUAL / ANALOGY
                         │
                         ▼
                   Réponse verbalisée en langage naturel
```

---

## 11. Modèle de données

### 11.1 Token UD

Représentation minimale d'un token annoté. Toute implémentation doit supporter ces champs :

```
Token = {
    id       : int          # position 1-based dans la phrase
    form     : str          # forme de surface (orthographe originale)
    lemma    : str          # lemme (forme canonique)
    pos      : str          # UPOS : VERB, NOUN, ADJ, ADV, SCONJ, CCONJ, ADP, DET, PRON, AUX, PUNCT...
    dep_rel  : str          # relation de dépendance UD (cf. §6)
    dep_head : int          # id du token tête (0 = root)
    morph    : {            # traits morphologiques (cf. §7)
        Mood?    : "Ind" | "Imp" | "Sub" | "Cnd"
        Tense?   : "Pres" | "Past" | "Fut" | "Imp"
        Aspect?  : "Imp" | "Perf"
        Person?  : "1" | "2" | "3"
        Number?  : "Sing" | "Plur"
        Gender?  : "Masc" | "Fem"
        Voice?   : "Act" | "Pass"
        Polarity?: "Pos" | "Neg"
        VerbForm?: "Fin" | "Inf" | "Part" | "Ger"
        PronType?: "Prs" | "Int" | "Rel" | "Dem" | "Ind" | "Art"
        Definite?: "Def" | "Ind"
        Reflex?  : "Yes"
        Degree?  : "Pos" | "Cmp" | "Sup"
    }
}
```

### 11.2 SentenceProfile

Résultat du classifieur de phrase. Produit par `classify(tokens, markers?) → SentenceProfile`.

```
SentenceProfile = {
    sentence_type  : "declarative" | "interrogative" | "imperative" | "exclamative"
    polarity       : "affirmative" | "negative"
    voice          : "active" | "passive"
    modality       : "indicative" | "subjunctive" | "conditional" | "imperative"
    complexity     : "simple" | "complex"
    subordination  : "none"       # pas de subordonnée (phrase simple)
                   | "condition"  # si, à condition que...
                   | "cause"      # parce que, car, puisque...
                   | "concession" # bien que, malgré...
                   | "temporal"   # quand, après que, avant que...
                   | "motivation" # pour que, afin que...
                   | "filter"     # ne...que si, seulement si...
                   | "sequence"   # puis, ensuite, d'abord... (coordination temporelle)
                   | "opposition" # mais, au lieu de... (opposition/concession légère)
                   | "enable"     # de sorte que, si bien que... (conséquence)
    has_restriction: bool     # pattern "ne...que" ou équivalent

    # Propriétés dérivées (calculées depuis les champs)
    expects_response : bool   # sentence_type == "interrogative"
    expects_action   : bool   # sentence_type == "imperative"
    asserts_fact     : bool   # sentence_type ∈ {"declarative", "exclamative"}
    is_negated       : bool   # polarity == "negative"
    is_passive       : bool   # voice == "passive"
    is_complex       : bool   # complexity == "complex"
    has_condition    : bool   # subordination == "condition"
}
```

### 11.3 FeatureVector — vecteur d'entrée du classifieur appris

Le modèle reçoit un **vecteur de features** composé de deux parties : (1) features structurelles UD (couche 1, zéro lemme) ; (2) embeddings de lemmes appris (couche 2, vecteurs continus). La couche 1 est toujours active. La couche 2 enrichit la discrimination sémantique — c'est le Mécanisme 2 de §12.7 (désambiguïsation sémantique). C'est depuis ce vecteur complet qu'il apprend à distinguer les rôles et les types.

```
FeatureVector = {

    # ── Verbe root (pivot central) ─────────────────────────────────────
    root_pos        : one-hot(UPOS)           # VERB / AUX / NOUN / ADJ...
    root_dep_rel    : one-hot(UD_DEP_RELS)    # root / cop / advcl...
    root_mood       : one-hot(Mood)           # Ind / Imp / Sub / Cnd
    root_tense      : one-hot(Tense)          # Pres / Past / Fut / Imp
    root_aspect     : one-hot(Aspect)         # Imp / Perf
    root_voice      : one-hot(Voice)          # Act / Pass
    root_polarity   : one-hot(Polarity)       # Pos / Neg
    root_verbform   : one-hot(VerbForm)       # Fin / Inf / Part / Ger
    root_person     : one-hot(Person)         # 1 / 2 / 3

    # ── Sujet (S) ─────────────────────────────────────────────────────
    subj_dep_rel    : one-hot(UD_DEP_RELS)    # nsubj / nsubj:pass / expl / csubj
    subj_pos        : one-hot(UPOS)           # NOUN / PRON / PROPN / —
    subj_present    : bool                    # ∃ nsubj dans la clause
    subj_id_gt_root : bool                    # nsubj.id > root.id → INVERSION S-V → interrogatif

    # ── Compléments (C) ───────────────────────────────────────────────
    has_obj         : bool                    # dep_rel=obj présent
    has_iobj        : bool                    # dep_rel=iobj présent
    has_ccomp       : bool                    # dep_rel=ccomp présent
    has_xcomp       : bool                    # dep_rel=xcomp présent

    # ── Connecteur (K) ────────────────────────────────────────────────
    connector_dep   : one-hot(UD_DEP_RELS)    # mark / cc / case / parataxis
    connector_pos   : one-hot(UPOS)           # SCONJ / CCONJ / ADP / —
    connector_present: bool                   # connecteur explicite présent

    # ── Subordonnée (advcl si présente) ──────────────────────────────
    has_advcl       : bool
    advcl_mood      : one-hot(Mood)           # Mood du verbe de la subordonnée
    advcl_polarity  : one-hot(Polarity)       # Polarity du verbe subordonné
    advcl_verbform  : one-hot(VerbForm)

    # ── Ponctuation finale ────────────────────────────────────────────
    punct_type      : one-hot({".", "?", "!", "...", "?!", "—", ""})

    # ── Flags structurels ─────────────────────────────────────────────
    has_aux_pass    : bool                    # dep_rel=aux:pass présent
    has_nsubj_pass  : bool                    # dep_rel=nsubj:pass présent
    has_obl_agent   : bool                    # dep_rel=obl:agent présent
    has_neg_advmod  : bool                    # advmod avec Polarity=Neg
    has_pron_int    : bool                    # PronType=Int sur un token
    has_restriction : bool                    # structure de restriction (advmod Neg + advmod non-Neg)
    has_expl        : bool                    # dep_rel=expl présent
    has_temporal_obl: bool                    # obl:tmod ou advmod temporel présent

    # ── Features positionnelles ────────────────────────────────────────
    # La position de chaque élément par rapport au verbe root est du sens.
    subj_before_root    : bool                # sujet avant le verbe  → ordre canonique
    subj_after_root     : bool                # sujet après le verbe  → inversion → interrogatif
    advcl_before_root   : bool                # subordonnée en tête   → condition/cause proéminente
    advcl_after_root    : bool                # subordonnée en queue  → condition/cause backgroundée
    obj_after_root      : bool                # objet après le verbe  → ordre canonique (V-O)
    obj_before_root     : bool                # objet avant le verbe  → topicalisation
    neg_before_root     : bool                # négateur pré-verbal   → portée sur tout le prédicat
    neg_after_root      : bool                # négateur post-verbal  → renforcement/spécification
    connector_before_advcl_head : bool        # connecteur en tête de subordonnée → canonique
    sentence_initial_id : float               # position normalisée du premier token (≠ det)
                                              # = id / len(tokens) ∈ [0.0, 1.0]
    sentence_final_id   : float               # position normalisée du dernier token (≠ punct)
                                              # = id / len(tokens) ∈ [0.0, 1.0]
    relative_depth      : float               # profondeur normalisée de la subordonnée
                                              # = depth / max_depth_in_sentence ∈ [0.0, 1.0]

    # ── Features ternaires ─────────────────────────────────────────────
    has_two_sources     : bool                # deux NOUN/processus coordonnés avant VERB
    has_third_condition : bool                # troisième proposition conditionnelle détectée
    has_obl_mediator    : bool                # obl avec "en/par/via" après VERB causal
    third_clause_mood   : one-hot(Mood)?      # Mood du verbe du tiers (si présent)
    third_connector_dep : one-hot(UD_DEP_RELS)? # dep_rel du connecteur du tiers
}
```

**Ce que le modèle apprend depuis ce vecteur :**

```
Tâche 1 — Classification du type de phrase
  Input  : FeatureVector(Φ)
  Output : {declarative, interrogative, imperative, exclamative}
  Signal principal appris :
    has_pron_int=True + subj_id_gt_root=True + punct_type="?"  → interrogative
    root_mood=Imp + subj_present=False                          → imperative
    has_neg_advmod=True + has_restriction=True                  → négation restrictive

Tâche 2 — Classification du type de relation (ternaire)
  Input  : FeatureVector(Φ) + FeatureVector(subordonnée) + FeatureVector(tiers?)
  Output : {cause, condition, enable, prevent, concession, motivation, sequence,
            filter, opposition, analogy, counterfactual,
            conditional_cause, mediated_cause, joint_cause}
  Signal principal appris :
    connector_dep=mark + connector_pos=SCONJ + advcl_mood=Sub  → {condition, concession, motivation}
    connector_dep=mark + connector_pos=SCONJ + advcl_mood=Ind  → {cause, temporal, consequence}
    root_polarity=Neg + has_restriction=True                   → filter
    root_polarity=Neg                                          → prevent
    ── Ternaires ──
    has_two_sources=True + cc="et" entre sources               → joint_cause
    has_third_condition=True + VERB causal principal            → conditional_cause
    has_obl_mediator=True (obl avec "en/par/via")              → mediated_cause

Tâche 3 — Classification du type de nœud (taxonomie §8, 8 types)
  Input  : FeatureVector(nœud_candidat)
  Output : {processus, état_local, état_global, entité, condition,
            concept, événement, contrainte}
  Signal principal appris :
    root_pos=VERB + has_obj=True + ¬cop                        → processus
    root_pos=VERB + connector_dep=cop + ADJ_pred               → état_local
    root_pos=NOUN abstrait + contexte_macro                    → état_global
    root_pos=NOUN concret + ¬déverbal + ¬abstrait              → entité
    has_advcl=True + connector_dep=mark                        → condition
    root_pos=NOUN abstrait + numéral/mesure adjacent           → concept
    root_pos=NOUN + has_temporal_obl=True                      → événement
    has_advcl=True + Polarity=Neg + has_restriction            → contrainte

  Note de migration depuis le moteur actuel (AUDIT_COMPARATIF) :
    action, transition → processus   (fusion — signaux identiques)
    etat              → état_local   (renommage)
    etat_systemique   → état_global  (renommage)

Tâche 4 — Détection du rôle du token dans la clause
  Pour chaque token t, classifier :
  Input  : (t.pos, t.dep_rel, t.morph, t.dep_head_morph)
  Output : {sujet, complement_direct, complement_indirect,
            connecteur, negateur, modificateur, ponctuation, autre}
  Signal principal appris :
    dep_rel=nsubj                → sujet
    dep_rel=obj                  → complément direct
    dep_rel=mark + pos=SCONJ    → connecteur subordonnant
    dep_rel=advmod + Polarity=Neg → négateur/restricteur
    dep_rel=cc + pos=CCONJ      → connecteur coordinant
```

### 11.3-N Normalisation du FeatureVector

Chaque type de feature suit une règle de normalisation précise. Sans normalisation, les features entières (positions, profondeur) dominent les features binaires et faussent l'apprentissage.

```
Type de feature              Normalisation              Formule                Intervalle
──────────────────────────────────────────────────────────────────────────────────────────
one-hot (pos, dep_rel, morph) Aucune                   ∈ {0, 1}              [0, 1]
bool                          Aucune                   ∈ {0.0, 1.0}          [0, 1]

position absolue              Position relative         id / len(tokens)       [0.0, 1.0]
(sentence_initial_id,          par phrase               0.0 = premier token
 sentence_final_id)                                     1.0 = dernier token

profondeur syntaxique         Profondeur relative       depth / max_depth_in_phrase  [0.0, 1.0]
(relative_depth)               par phrase               0.0 = root
                                                        1.0 = feuille la plus profonde

distance root-subj             Distance relative         |subj.id - root.id| / len(tokens)  [0.0, 1.0]
(subj_id_gt_root est bool,
 la distance brute est normalisée)

Règle générale :
  Toute feature entière est normalisée par la longueur de la phrase (len(tokens)).
  Normalisation locale (par phrase), PAS globale (par corpus).
  → Invariance à la longueur de phrase garantie.
  → Une phrase de 5 tokens et une phrase de 50 tokens ont des features comparables.

Cas particulier — features ternaires :
    # ── Embeddings de lemmes (couche 2 uniquement — Mécanisme 2 §12.7) ──────
    # Ces champs sont absents de la couche 1 déterministe.
    # Valeur par défaut = vecteur zéro si embedding non disponible.
    lemma_emb_src   : float[128]              # embedding ponctuel du lemme source (nsubj)
    lemma_emb_verb  : float[128]              # embedding ponctuel du lemme root (VERB)
    lemma_emb_dst   : float[128]              # embedding ponctuel du lemme cible (obj)
                                              # dim = 128 (aligne sur embedding.py existant)
                                              # Source : lookup dans WordEmbedding.vocab
                                              #   lemma_emb_src  = embed(S[nsubj].lemma)
                                              #   lemma_emb_verb = embed(V.root.lemma)
                                              #   lemma_emb_dst  = embed(C[obj].lemma)
                                              # Valeur par défaut si lemme absent : vecteur zéro
                                              # DISTINCT du pooling root/mean/max (features.py:187)
                                              # — pooling agrège la clause entière,
                                              #   lemma_emb cible les 3 tokens pivot seulement

  # ── Features ternaires ──────────────────────────────────────────────────
  has_two_sources      : bool → {0.0, 1.0} — aucune normalisation
  has_third_condition  : bool → {0.0, 1.0}
  has_obl_mediator     : bool → {0.0, 1.0}
  third_clause_mood    : one-hot → [0, 1]
  third_connector_dep  : one-hot → [0, 1]
```

### 11.4 Nœud du graphe causal (CIR Node)

```
CIRNode = {
    id        : int | str      # identifiant unique dans le graphe
    label     : str            # lemme ou expression du concept
    node_type : "processus" | "état_local" | "état_global" | "entité" | "condition"
               | "concept" | "événement" | "contrainte"
    pos       : str?           # UPOS du token source
    morph     : dict?          # features morphologiques du token source
    source    : str?           # texte source de la proposition
}
```

### 11.5 Arête causale ternaire (CIR Edge)

Le modèle est **ternaire**. Une relation sans tiers (`third=None`, `sources=[id]`) est le cas fondamental — relation directe A→B. La compatibilité descendante est garantie.

```
CIREdge = {
    ── Sources (une ou plusieurs) ──────────────────────────────────────
    sources      : [int | str]     # liste d'ids sources
                                   # [A]        → relation directe (third=None)
                                   # [A₁, A₂]  → JOINT_CAUSE (deux sources)

    ── Cible ───────────────────────────────────────────────────────────
    target       : int | str       # id du nœud cible (effet)

    ── Type de relation ────────────────────────────────────────────────
    relation     : "cause"              # third=None : A → B
                 | "condition"          # third=None : A ⇒ B
                 | "enable"             # third=None : A ⊃ B
                 | "prevent"            # third=None : A ⊄ B
                 | "concession"         # third=None : malgré A, B
                 | "motivation"         # third=None : A en vue de B
                 | "sequence"           # third=None : A ≺ B
                 | "filter"             # third=None : B↑A (restriction)
                 | "opposition"         # third=None : A ⊕ B
                 | "analogy"            # third=None : A≈C, B≈D
                 | "counterfactual"     # third=None : P(B|do(¬A))
                 | "conditional_cause"    # ternaire : A→B | C
                 | "mediated_cause"       # ternaire : A→B via C
                 | "joint_cause"          # ternaire : (A₁∧A₂)→B
                 # algèbre de négation §9.4 — Règle 1 (négation sur la cible)
                 | "conditional_prevent"  # CONDITIONAL_CAUSE + Neg(cible)
                 | "mediated_prevent"     # MEDIATED_CAUSE + Neg(cible)
                 | "joint_prevent"        # JOINT_CAUSE + Neg(cible)
                 | "data_dependency"
                 | "control_dependency"

    ── Tiers (optionnel — présent uniquement pour les ternaires) ───────
    third        : {
        role     : "condition"   # C conditionne la relation A→B (CONDITIONAL_CAUSE)
                 | "mediator"    # C est le mécanisme de A→B (MEDIATED_CAUSE)
                 # "joint" est DÉPRÉCIÉ — utiliser sources=[id₁,id₂] à la place
        node     : int | str     # id du nœud tiers
        polarity : "positive" | "negative"?  # négation sur la condition (Règle 2 §9.4)
    }?                           # None (Python) / null (JSON) si relation directe ou JOINT_CAUSE

    ── Qualifications ──────────────────────────────────────────────────
    confidence   : float         # ∈ [0.0, 1.0]
    polarity     : "positive" | "negative"
    voice        : "active" | "passive"
    modality     : "indicative" | "subjunctive" | "conditional" | "imperative"
    has_restriction : bool       # restriction exclusive (ne...que)
    condition_prominence : "foreground" | "background"  # D7 + Éq. 9
    temporal_gap : {min, max, unit}?
    source_text  : str?
}
```

**Exemples de schéma JSON pour les 4 patterns :**

```json
// Relation directe (third=null)
{"sources": [0], "target": 1, "relation": "cause",
 "third": null, "confidence": 0.50}  // CAUSE Rung1×f(Ind)×g(explicit) = 0.50

// Conditional_cause  (Rung2=0.75 × f(Ind=1.0) × g(mark=1.0) = 0.75, avant calibration)
{"sources": [0], "target": 1, "relation": "conditional_cause",
 "third": {"role": "condition", "node": 2}, "confidence": 0.75}

// Mediated_cause  (Rung3=0.65 × f(Ind=1.0) × g(obl=1.0) = 0.65, avant calibration)
{"sources": [0], "target": 1, "relation": "mediated_cause",
 "third": {"role": "mediator", "node": 2}, "confidence": 0.65}

// Joint_cause  (Rung3=0.65 × f(Ind=1.0) × g(cc=1.0) = 0.65, avant calibration)
{"sources": [0, 1], "target": 2, "relation": "joint_cause",
 "third": null, "confidence": 0.65}
```

### 11.6 Graphe causal (CIR)

```
CIR = {
    nodes            : list[CIRNode]
    edges            : list[CIREdge]      # toutes ternaires (third=null si relation directe)
    source_text      : str?
    lang             : str?
    sentence_profile : SentenceProfile?
    salience         : {                  # annotations de saillance (Éq. 9)
        topic_node          : int | str?
        focus_node          : int | str?
        condition_prominence: "foreground" | "background"?
    }?
}
```

---

## 12. Architecture du pipeline

### 12.1 Couches du moteur

```
┌─────────────────────────────────────────────────────────────────────┐
│  COUCHE 0 — Annotation UD (hors moteur)                             │
│  Entrée : texte brut (FR, EN, multilingue)                          │
│  Sortie : list[Token{id, form, lemma, pos, dep_rel, morph, dep_head}]│
│  Outil  : annotateur UD externe (spaCy, Stanza, UDPipe...)          │
│  Contrainte : le moteur reçoit des tokens, jamais du texte brut      │
└────────────────────────────┬────────────────────────────────────────┘
                             │ list[Token]
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 1 — Vectorisation structurelle                               │
│  Entrée : list[Token]                                                │
│  Sortie : FeatureVector(Φ)                                           │
│  Algorithme : extraction des features UD (dep_rel, pos, morph,       │
│               flags booléens, one-hot encodings) — Éq. 0            │
│  Aucune dépendance aux lemmes                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ FeatureVector
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 2 — Classification phrastique                                │
│  Entrée : FeatureVector(Φ)                                           │
│  Sortie : SentenceProfile                                            │
│  Algorithme :                                                        │
│    Partie déterministe : Éq. 1 (type) + Éq. 2 (rôles) + formes     │
│    Partie apprise      : classifieur entraîné sur FeatureVector      │
│                          → type de phrase, type de nœud, type R     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SentenceProfile
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 3 — Extraction CIR                                           │
│  Entrée : list[Token] + SentenceProfile + FeatureVector             │
│  Sortie : CIR (graphe partiel)                                       │
│  Algorithme : Éq. 3 (nœuds) + Éq. 4 (relations apprise) +          │
│              Éq. 7 (normalisation voix) + Éq. 5/8 (simple/complexe) │
└────────────────────────────┬────────────────────────────────────────┘
                             │ CIR
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 4 — Graphe causal cumulatif                                  │
│  Entrée : CIR + graphe existant                                      │
│  Sortie : graphe enrichi                                             │
│  Structure : DAG orienté, nœuds typés, arêtes avec confiance        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Graphe causal
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 5 — Raisonnement causal (Pearl)                              │
│  Entrée : Graphe + requête (type + arguments)                       │
│  Requêtes : WHY / CHAIN / CHAIN_T / DELAY / SPOF / BEFORE /         │
│             COUNTERFACTUAL / ANALOGY / ABDUCT / CENTRALITY...        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Résultat structuré
┌────────────────────────────▼────────────────────────────────────────┐
│  COUCHE 6 — Verbalisation                                            │
│  Entrée : résultat structuré                                         │
│  Sortie : texte naturel structuré                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.2 Routage NLU — point d'entrée utilisateur

```
Entrée utilisateur (texte naturel)
    │
    ▼
classify(tokens, markers) → SentenceProfile
    │
    ├─ expects_response == True   (INTERROGATIVE)
    │      │
    │      └─ Router(Φ) [Éq. 6] → requête sur le graphe (EXPLAIN, CHAIN, DELAY...)
    │
    ├─ expects_action == True     (IMPÉRATIVE)
    │      │
    │      └─ Dispatcher(V.lemma, C) → action du moteur (ANALYZE, LOAD, SAVE...)
    │
    └─ asserts_fact == True       (DÉCLARATIVE / EXCLAMATIVE)
           │
           └─ Extraire(Φ) [Éq. 5/8] → enrichir le graphe causal
```

### 12.3 Données d'entraînement — ce que le modèle doit apprendre

Le modèle apprend depuis des phrases annotées avec :
- le **type de phrase** (déclarative, interrogative, impérative, exclamative)
- le **type de relation** sur chaque arête (cause, condition, prevent, filter...)
- le **type de nœud** pour chaque argument (processus, état, entité, concept...)
- le **rôle** de chaque token (sujet, complément, connecteur, négateur, autre)

**Ce que les données d'entraînement doivent couvrir :**

```
Diversité structurelle requise :
  ├─ Phrases actives ET passives (même relation, deux structures)
  ├─ Phrases simples ET complexes (une proposition vs plusieurs)
  ├─ Toutes les combinaisons (Mood × Voice × Polarity × dep_rel connecteur)
  ├─ Négation totale (prevent) vs restriction (filter) dans des contextes distincts
  ├─ Subjonctif vs indicatif en subordonnée (même connecteur, relations différentes)
  └─ Impersonnels, ellipses, coordinations

Ce que les données NE doivent PAS faire :
  ├─ Biaiser vers un lemme précis comme signal de relation
  │    (plusieurs connecteurs différents doivent produire le même type de relation)
  └─ Sur-représenter une langue
       (les features UD sont universelles — le modèle doit généraliser)
```

**Propriété de généralisation attendue :**

Un modèle entraîné sur des phrases françaises avec `mark=SCONJ, Mood=Sub` → CONDITION  
doit correctement classer une phrase allemande ou espagnole avec le même pattern structurel,  
**sans avoir jamais vu les connecteurs de ces langues dans les données d'entraînement.**

### 12.4 Calcul de confiance — formule unique (D6)

⚠ L'ancienne échelle 1.0/0.8/0.7/.../0.3 avec modificateurs multiplicatifs est **caduque (D6)**.
Formule unique : `confiance = rung × f(Mood) × g(connecteur) + Δ_saillance` — voir §9.0.

```
Récapitulatif §9.0 pour référence rapide :
  Rungs     : R1(CAUSE,SEQUENCE,ANALOGY)=0.50
              R2(CONDITION,ENABLE,PREVENT,FILTER,CONDITIONAL_CAUSE)=0.75
              R3(COUNTERFACTUAL,MEDIATED_CAUSE,JOINT_CAUSE,CONCESSION)=0.65
  f(Mood)   : Ind=1.00 / Sub=0.85 / Cnd=0.80
  g(conn.)  : explicite=1.00 / VERB_causal=0.90 / parataxis=0.60 / aucun=0.50
  Δ         : +0.10 si advcl_before_root

  Cas courants (avant calibration) :
    CAUSE    + mark + Ind   → 0.50 × 1.00 × 1.00       = 0.50
    CONDITION+ mark + Sub   → 0.75 × 0.85 × 1.00       = 0.64
    FILTER   + mark + Ind   → 0.75 × 1.00 × 1.00       = 0.75
    CAUSE    + parataxis    → 0.50 × 1.00 × 0.60       = 0.30
    CAUSE    + aucun signal → 0.50 × 1.00 × 0.50       = 0.25
    CAUSE    + frontée      → 0.50 × 1.00 × 1.00 + 0.10 = 0.60

  Cohérence ordinale (vérifiée) :
    g : explicite(1.0) > VERB_causal(0.90) > parataxis(0.60) > aucun(0.50)
    Rung : R2(0.75) > R3(0.65) > R1(0.50)  [intervention > contrefactuel > association]

PRIORS uniquement — remplacés par calibration isotonique dès N_val ≥ 100 (§9.5).
```

### 12.5 Ce que le modèle ne peut pas apprendre seul

```
Limite                          Cause                          Traitement
──────────────────────────────────────────────────────────────────────────
Ambiguïté structurelle réelle   Même pattern UD, deux relations Données annotées avec contexte
  (ex: sub. condition vs          contextes distincts          plus large (fenêtre élargie)
       sub. conséquence)
Connecteur absent (parataxe)    Aucun token K dans la clause   conf=0.30 [Rung1×g(parataxis), formule §9.0]
Tokens sans morph               Annotateur non configuré       Couche 1 seule (ponctuation)
Coréférence pronominale         "il" → référent hors tokens    Résolution = couche 0
Ambiguïté des rôles PP          "avec" = instrument ou cause   Couche 0 ou annotation manuelle
Ellipse                         "Paul aussi" → référent implicite  Orphan UD → hors scope
Discours indirect libre         Frontières de proposition floues   Segmentation = couche 0
```

### 12.6 Bootstrap — démarrage à zéro exemple (cold start)

Le moteur est fonctionnel dès le premier lancement sans aucune donnée annotée. Il monte en précision progressivement selon trois phases.

```
PHASE 0 — Zéro exemple  (couche 1 déterministe seule)
────────────────────────────────────────────────────────────────────────
  Opérationnel immédiatement.

  Type de phrase   : Éq. 1 déterministe (punct + Mood + inversion S-V)
  Type de relation : CAUSE par défaut pour tout advcl         [Rung 1, conf=0.50]
                     PREVENT si Polarity=Neg                  [Rung 2, conf=0.75]
                     FILTER si Polarity=Neg + has_restriction  [Rung 2, conf=0.75]
  Type de nœud     : processus (VERB), nominal (NOUN), condition (SCONJ+VERB)
  Requêtes Pearl   : toutes fonctionnelles sur le CIR extrait
  Confiances       : priors de §9.0 uniquement (non calibrés)

  Illustration :
    "Si la demande baisse, les ventes chutent."
    → advcl détecté → CAUSE (défaut) → confiance = 0.50
    [Insuffisant pour distinguer CONDITION de CAUSE — acceptable en Phase 0]

PHASE 1 — N ∈ [1, N_min] exemples par type   (classifieur coarse actif)
────────────────────────────────────────────────────────────────────────
  N_min = 50 (seuil d'activation du classifieur coarse)

  Niveau coarse (§15) remplace les règles proxy pour les types couverts :
    5 relations : CAUSALE / RESTRICTIVE / NÉGATIVE / SÉQUENTIELLE / SIMPLE
    3 nœuds    : ACTION / NOMINAL / CONDITION

  Confiances calibrées sur les N_min premiers exemples (calibration légère).

  Bootstrap depuis UD existant (optionnel, accélère la Phase 1) :
    Corpus UD_French-GSD ou UD_English-EWT (librement disponibles)
    → Extraire les patterns (dep_rel=mark, SCONJ, Mood) sans label causal
    → Initialiser les distributions de Phase 1
    → Réduit N_min de 50 à 20 par type

PHASE 2 — N ≥ N_fine exemples par type   (classifieur fin actif)
────────────────────────────────────────────────────────────────────────
  N_fine = 400 (seuil d'activation du classifieur fin)

  Transition par relation : dès que N_fine exemples disponibles pour un type,
                            ce type bascule de coarse vers fin.
  Progression typique (ordre attendu selon fréquence dans les corpus) :
    CAUSE fine → CONDITION fine → SEQUENCE fine → CONCESSION fine → ...

  Calibration complète (§9.5) activée dès N_val ≥ 100.

Résumé :
  N=0        → Phase 0 : opérationnel, coarse, priors
  N ≥ 20-50  → Phase 1 : coarse calibré
  N ≥ 400    → Phase 2 : fin calibré, confiances réelles
```

### 12.7 Désambiguïsation sémantique — phrases de même structure

Deux phrases peuvent avoir un FeatureVector UD structurellement identique mais des sens causaux différents. Le moteur dispose de quatre mécanismes de désambiguïsation ordonnés par priorité.

#### Problème posé

```
"La maladie cause la fatigue."   → cause(maladie → fatigue)
"La fatigue cause la maladie."   → cause(fatigue → maladie)

FeatureVector IDENTIQUE sauf les lemmes de nsubj et obj.
```

#### Mécanisme 1 — Positions syntaxiques (couvre ~80% des cas)

`nsubj` = source causale. `obj` = cible causale. Règle invariante de l'Éq. 2.

```
La position IS le rôle. Le moteur extrait fidèlement ce que le texte affirme.
"Les inondations causent la pluie." → cause(inondation → pluie, conf=0.80)
Le moteur ne valide pas la plausibilité causale — la phrase EST la vérité (Axiome 1).
```

#### Mécanisme 2 — Embeddings de lemmes appris (couvre ~15% des cas restants)

Quand les positions sont identiques et le sens diffère, le signal discriminant est la **sémantique des lemmes**. Le moteur apprend des représentations vectorielles denses — pas de règles statiques.

```
Distinction fondamentale :
  Hardcoding (INTERDIT) : if lemma=="empêcher" → PREVENT
                           règle statique, fragile, langue-dépendante

  Embedding (CORRECT)  : vector("empêcher") ≈ vector("bloquer") ≈ vector("inhiber")
                          Le modèle apprend que ce cluster vectoriel → PREVENT
                          Généralise à des verbes sémantiquement proches mais jamais vus.

Cas typique :
  "Le médicament réduit la douleur."   → nsubj=médicament, VERB=réduire, obj=douleur
  "La douleur réduit le confort."      → nsubj=douleur, VERB=réduire, obj=confort

  Structure UD identique.
  Différence : vector("médicament") ≠ vector("douleur") dans l'espace sémantique
  Le modèle apprend depuis les données que (médicament + réduire + douleur)
  co-occurrent dans des contextes ENABLE thérapeutique,
  et que (douleur + réduire + confort) co-occurrent dans des contextes CAUSE.
  Ce n'est pas du hardcoding — c'est de l'apprentissage distribué.
```

#### Mécanisme 3 — Types de nœuds comme contrainte sémantique (couvre ~3%)

Les types de nœuds (`processus`, `état_local`, `état_global`, `entité`, `concept`) portent une sémantique partielle qui contraint les relations plausibles.

```
Patterns appris depuis les données :
  entité       → cause  → état_local   (le virus cause la maladie)
  processus    → cause  → processus    (la chute cause la blessure)
  état_global  → enable → processus    (la crise facilite la réforme)
  processus    → prevent→ état_local   (le vaccin empêche l'infection)

Si nœud_source = concept (taux, mesure)
ET nœud_cible  = état_global
→ Relation probable : CAUSE ou CONDITION (pas PREVENT ni JOINT_CAUSE)
→ Confiance ajustée selon la cohérence de types
```

#### Mécanisme 4 — Accumulation et consensus inter-textes (couvre ~1%)

Sur de multiples textes, les arêtes contradictoires ou convergentes s'accumulent. La répétition augmente la confiance ; une arête isolée contredisant le consensus reçoit une confiance réduite.

```
Texte 1 : cause(pluie → inondation, conf=0.90)
Texte 2 : cause(pluie → inondation, conf=0.85)  ← confirmation
Texte 3 : cause(inondation → pluie, conf=0.50)  ← contradictoire, isolé

→ cause(pluie → inondation) : confiance agrégée = haute (consensus)
→ cause(inondation → pluie) : seule occurrence → ANOMALIE détectable via CYCLES
```

#### Résidu irréductible — ce que le moteur ne peut pas résoudre

```
Cas 1 — Direction causalement implausible mais textuellement affirmée
  "Les inondations causent la pluie."
  → Le moteur extrait fidèlement. Il ne juge pas la plausibilité.
  → La validation causale est hors scope du moteur.

Cas 2 — Même verbe, deux sens selon le domaine
  "La chaleur dilate le métal."      → cause (physique)
  "La chaleur dilate les artères."   → enable (médecine)
  Structure UD identique. Seul le lemme de l'obj diffère.
  → Mécanisme 2 approxime via embeddings
  → Si P_max < θ_ambiguity=0.65 → Éq. 11 : ambiguous=True dans le CIR

Cas 3 — Métaphore causale
  "L'amour cause la souffrance."
  → Sens littéral ou métaphorique ? Impossible sans contexte discursif.
  → confiance réduite, Rung 1
```

#### Pipeline de désambiguïsation complet

```
Deux phrases, même structure UD
        │
        ▼  Méc. 1 — Positions syntaxiques
   nsubj ≠ obj → direction résolue (80%)
        │
        ▼  Méc. 2 — Embeddings de lemmes
   vector(nsubj) + vector(VERB) + vector(obj) → type de relation (15%)
        │
        ▼  Méc. 3 — Types de nœuds
   TypeNœud(source) × TypeNœud(cible) → contrainte (3%)
        │
        ▼  Méc. 4 — Consensus inter-textes
   Arêtes contradictoires → anomalie → confiance réduite (1%)
        │
        ▼  Résidu (~1%)
   P_max < θ_ambiguity → Éq. 11 → ambiguous=True
   Le moteur signale l'ambiguïté sans forcer une interprétation.
```

### 12.8 Limites architecturales inhérentes

| Limite | Cause structurelle | Couche concernée |
|--------|-------------------|-----------------|
| Parataxe sans connecteur | advcl sans mark → aucun signal K | Couche 3 — conf=0.30 [Rung1×g(parataxis), avant calib.] |
| Morph features absentes | Annotateur UD mal configuré | Couche 2 dégradée — couche 1 seule |
| Coréférence | Référent hors tokens de la clause | Hors scope moteur — couche 0 |
| Ambiguïté PP | "avec" instrument vs cause | Hors scope — couche 0 |
| Ellipse | Orphan UD | Hors scope couche 2 |
| Discours indirect libre | Frontières de proposition floues | Segmentation = couche 0 |
| Plausibilité causale | Connaissance du monde requise | Hors scope — validation externe |

---

## 13. Application à l'exemple pivot

### Phrase analysée
> « Le médicament n'est prescrit que si les traitements conventionnels ont échoué. »

### Analyse UD token par token

| id | form | lemma | pos | dep_rel | morph |
|----|------|-------|-----|---------|-------|
| 1 | Le | le | DET | det | Definite=Def |
| 2 | médicament | médicament | NOUN | nsubj:pass | Gender=Masc\|Number=Sing |
| 3 | n' | ne | ADV | advmod | Polarity=Neg |
| 4 | est | être | AUX | aux:pass | Mood=Ind\|Number=Sing\|Person=3\|Tense=Pres |
| 5 | prescrit | prescrire | VERB | **root** | Mood=Ind\|Voice=Pass\|VerbForm=Part |
| 6 | que | que | ADV | advmod | — |
| 7 | si | si | SCONJ | **mark** | — |
| 8 | les | le | DET | det | Definite=Def |
| 9 | traitements | traitement | NOUN | nsubj | Gender=Masc\|Number=Plur |
| 10 | conventionnels | conventionnel | ADJ | amod | Gender=Masc\|Number=Plur |
| 11 | ont | avoir | AUX | aux | Mood=Ind\|Number=Plur\|Person=3\|Tense=Pres |
| 12 | échoué | échouer | VERB | **advcl** | Mood=Ind\|VerbForm=Part\|Tense=Past |

### Application des équations

```
Éq. 0 — Décomposition
  V = prescrit (id=5, dep_rel=root)
  S = médicament (dep_rel=nsubj:pass)
  C = échoué (dep_rel=advcl)
  K = si (dep_rel=mark, pos=SCONJ)
  Φ = (médicament, prescrit, [échoué], si)

Éq. 1 — Type de phrase
  punct = "."
  PronType=Int : absent
  nsubj.id=2 < root.id=5 → pas d'inversion
  Mood=Ind + nsubj présent
  → DÉCLARATIVE ✓

Éq. 2 — Rôles sémantiques
  dep_rel(médicament) = nsubj:pass → Voice=Pass
  Patient ← médicament (nsubj:pass)
  Agent   ← ∅ (obl:agent absent → agent effacé)
  [subordonnée] nsubj(échoué) = traitements → Agent subordonnée

Éq. 3 — Types de nœuds
  médicament   → ENTITÉ (NOUN concret)
  prescription → PROCESSUS (VERB passif)
  échouer      → PROCESSUS (action accomplie, VERB)
  traitements  → ENTITÉ (NOUN concret)

Éq. 4 — Type de relation
  Couche 1 : advmod(n', id=3) → Polarity=Neg, pré-verbal (id=3 < root=5)
             advmod(que, id=6) → post-verbal, co-présent avec négateur → RESTRICTION
             → signal FILTER
  Couche 2 : pattern (dep_rel=mark, pos=SCONJ, advcl_mood=Ind, Polarity=Neg, restriction)
             → FILTER appris depuis données annotées

Éq. 7 — Normalisation voix
  nsubj:pass(prescrit, médicament) → Patient = médicament
  obl:agent absent → Agent = ∅ (générique)

Éq. 8 — Phrase complexe
  Φ₁ = (médicament, prescrit, ∅)        [principale]
  Φ₂ = (traitements, échoué, ∅)         [subordonnée]
  K  = si → CONDITION → direction : Φ₂ cause Φ₁
  Restriction → FILTER
  → CIR_edge(id=0, id=1, FILTER, confiance=0.75)  [Rung2×f(Ind)×g(mark), avant calib.]

Éq. 9 — Saillance positionnelle
  subj_before_root  = True  (médicament id=2 < root id=5) → topic
  advcl_after_root  = True  (échoué id=12 > root id=5)    → condition backgroundée
  neg_before_root   = True  (n' id=3 < root id=5)         → portée négative totale
  neg_after_root    = False  (pas de "pas" post-verbal)

  Structure : principale AVANT subordonnée
    → condition en arrière-plan (la prescription est l'information principale)
    → si la phrase avait été "Si les traitements échouent, le médicament n'est prescrit que..."
      advcl_before_root = True → condition proéminente → confiance += 0.1

  Annotations de saillance sur le CIR :
    topic_node   = "prescription_médicament"  (sujet de la principale, pré-verbal)
    focus_node   = "échec_traitements"        (dernier constituant informatif)
    condition_prominence = "background"       (subordonnée post-posée)
```

### SentenceProfile résultant

```
SentenceProfile(
    sentence_type   = "declarative"
    polarity        = "negative"      # Polarity=Neg sur "n'"
    voice           = "passive"       # nsubj:pass + aux:pass
    modality        = "indicative"    # Mood=Ind sur prescrit et échoué
    complexity      = "complex"       # advcl présent
    subordination   = "condition"     # mark="si"
    has_restriction = True            # pattern "ne...que"

    expects_response  = False
    expects_action    = False
    asserts_fact      = True
    is_negated        = True
    is_passive        = True
    is_complex        = True
    has_condition     = True
)
```

### CIR extrait — pourquoi FILTER direct (third=None) et non CONDITIONAL_CAUSE

**Correction de doctrine :** la phrase pivot est un **FILTER direct (third=None)**, pas un ternaire. Elle a été incorrectement identifiée comme ternaire dans les versions précédentes.

**Raisonnement :**

```
Un ternaire CONDITIONAL_CAUSE(A, B, cond=C) requiert TROIS éléments distincts :
  A = cause explicite (agent actif)
  B = effet
  C = condition distincte de A

Dans "Le médicament n'est prescrit que si les traitements ont échoué." :
  Agent   = ∅  (effacé — passif sans obl:agent)
  Source  = échec_traitements  (l'unique acteur causal)
  Cible   = prescription

Il n'y a PAS de cause distincte de la condition.
→ L'échec est à la fois la source ET la condition.
→ C'est un FILTER direct (third=None) (restriction exclusive).

Un ternaire aurait requis un agent explicite distinct :
  "Le médecin prescrit le médicament uniquement si les traitements échouent."
  → CONDITIONAL_CAUSE(médecin, prescription, cond=échec)
     A=médecin (cause active), B=prescription, C=échec (condition distincte)
```

```json
{
  "nodes": [
    {
      "id": 0,
      "label": "prescription_médicament",
      "node_type": "processus",
      "pos": "VERB",
      "morph": {"Mood": "Ind", "Voice": "Pass", "VerbForm": "Part"}
    },
    {
      "id": 1,
      "label": "échec_traitements_conventionnels",
      "node_type": "processus",
      "pos": "VERB",
      "morph": {"Mood": "Ind", "VerbForm": "Part", "Tense": "Past"}
    }
  ],
  "edges": [
    {
      "sources": [1],
      "target": 0,
      "relation": "filter",
      "third": null,
      "confidence": 0.75,
      "polarity": "negative",
      "voice": "passive",
      "modality": "indicative",
      "has_restriction": true,
      "prominence": "background",
      "source_text": "Le médicament n'est prescrit que si les traitements conventionnels ont échoué."
    }
  ],
  "salience": {
    "topic_node": 0,
    "focus_node": 1,
    "condition_prominence": "background"
  },
  "sentence_profile": {
    "sentence_type": "declarative",
    "polarity": "negative",
    "voice": "passive",
    "modality": "indicative",
    "complexity": "complex",
    "subordination": "condition",
    "has_restriction": true
  }
}
```

**Lecture du CIR :**

```
sources = [1]  (échec_traitements)  → source et condition confondus (passif, agent effacé)
target  =  0   (prescription)       → cible
relation = "filter"                  → direct (third=None) — restriction exclusive
third   =  null                      → PAS de tiers distinct
has_restriction = true               → ne...que = exclusivité
polarity = "negative"                → Polarity=Neg sur "n'"
voice   = "passive"                  → nsubj:pass + aux:pass
prominence = "background"            → subordonnée en queue de phrase

Interprétation :
  La prescription ne se produit exclusivement que sous la condition
  de l'échec des traitements. L'agent prescripteur n'est pas exprimé.
  Calcul de confiance (formule §9.0) :
    FILTER → Rung 2, base = 0.75
    f(Mood=Ind) = 1.00
    g(mark=si, explicite) = 1.00
    Δ_saillance = 0  (advcl_after_root → pas de frontée)
    has_restriction = True → caractérisation de FILTER (pas de multiplicateur)
    → confiance = 0.75 × 1.00 × 1.00 + 0 = 0.75  [valeur avant calibration]

Contre-exemple ternaire pour comparaison :
  "Le médecin prescrit le médicament uniquement si les traitements échouent."
  → CONDITIONAL_CAUSE(médecin, prescription, cond=échec, conf=0.75  [Rung2×f(Ind)×g(mark), avant calib.])
     sources=[médecin], target=prescription, third={role:condition, node:échec}
  (pas une cause simple, mais une cause conditionnelle exclusive)
```

---

## 14. Problème de l'annotation — solution par habitudes positionnelles

### 14.1 Le goulot d'étranglement de l'annotation

L'annotation UD complète (dep_rel + morph sur chaque token) suppose trois prérequis qui peuvent tous faire défaut :

```
1. Un parseur UD fiable pour la langue cible
2. Un corpus suffisamment grand et représentatif dans cette langue
3. Des annotateurs formés capables de corriger les erreurs du parseur
```

Un moteur qui dépend de ces trois prérequis s'effondre dès qu'on change de langue, de domaine, ou que le parseur commet des erreurs systématiques. **L'annotation token par token ne peut pas être un prérequis.**

### 14.2 Ce que la phrase contient sans annotation

La phrase contient toutes ses informations (Axiome 1). Certaines sont accessibles **directement depuis le texte brut**, sans aucune annotation :

```
Information                    Accessible sans annotation    Signal direct
───────────────────────────────────────────────────────────────────────────
Type de phrase (partiel)       Oui                          Ponctuation finale
                                                             (. / ? / !)

Complexité (partiel)           Oui                          Présence d'un SCONJ
                                                             dans la phrase

Frontière de propositions      Oui                          Virgule + SCONJ
                                                             + deux groupes verbaux

Position des éléments          Oui                          Index des tokens
                                                             (arithmétique pure)

POS de surface                 Avec tagger minimal          VERB / NOUN / SCONJ /
                               (bien plus simple qu'UD)      ADV / PUNCT
```

### 14.3 Le principe des habitudes d'apparition

Chaque type de phrase produit une **configuration positionnelle caractéristique** — une habitude d'apparition des éléments qui la composent. Ces habitudes sont statistiquement régulières et apprises depuis le corpus, pas hardcodées.

**Définition formelle :**

```
HabitureΦ(type) = distribution de probabilité sur les positions relatives
                  des éléments {VERB, NOUN, SCONJ, ADV, PUNCT}
                  conditionnellement au type de phrase

P(NOUN.pos < VERB.pos | type=DÉCLARATIVE)  ≈ 0.85   [sujet avant verbe]
P(NOUN.pos > VERB.pos | type=DÉCLARATIVE)  ≈ 0.15   [inversion rare]

P(NOUN.pos > VERB.pos | type=INTERROGATIVE) ≈ 0.60  [inversion fréquente]
P(punct=?            | type=INTERROGATIVE)  ≈ 0.95  [ponctuation quasi-certaine]

P(NOUN.pos absent    | type=IMPÉRATIVE)     ≈ 0.80  [pas de sujet pré-verbal]
P(VERB.pos = 0       | type=IMPÉRATIVE)     ≈ 0.70  [verbe en tête]

P(SCONJ.pos < VERB1.pos | type=COMPLEXE)   ≈ 0.45  [subordonnée frontée]
P(SCONJ.pos > VERB1.pos | type=COMPLEXE)   ≈ 0.55  [subordonnée en queue]
```

**Ce que le moteur apprend depuis ces habitudes :**

Le moteur n'apprend pas des règles — il apprend des **profils de distribution**. À l'inférence, il compare le profil positionnel de la phrase inconnue aux profils connus et retient le type le plus probable.

```
Phrase inconnue → extraire profil :
  (VERB.pos, NOUN.pos, punct, SCONJ.present, ADV.before_VERB)

→ P(déclarative | profil) = 0.87
→ P(interrogative | profil) = 0.09
→ P(impérative | profil) = 0.04

→ Type = DÉCLARATIVE ✓
```

### 14.4 Minimum requis pour apprendre ces habitudes

```
Observation minimale                   Justification
──────────────────────────────────────────────────────────────────────────
1. Tokenisation                        Gratuite, universelle. Donne les
                                       positions absolues (id de chaque token).

2. POS de surface : 6 catégories       Beaucoup plus simple que l'annotation UD.
   VERB / NOUN / SCONJ / ADV /         Disponible pour la quasi-totalité des langues.
   CCONJ / PUNCT                       Erreurs rares sur ces 6 catégories.

3. Labels au niveau phrase             Un seul label par phrase, pas par token.
   (type de phrase)                    500 exemples par type suffisent pour
                                       apprendre les distributions positionnelles
                                       car l'espace de features est petit (~5 vars).
```

**Ce qui n'est PAS nécessaire :**

```
dep_rel par token  → remplacé par la position relative
morph.Mood         → approximé par ponctuation + position VERB/NOUN
morph.Polarity     → approximé par ADV avant VERB
lemme (règle statique) → jamais nécessaire en couche 1 (déterministe)
lemme (embedding)     → optionnel en couche 2, améliore Méc.2 sans changer couche 1
annotation complète   → jamais nécessaire pour la couche de classification de type
```

### 14.5 Propriété de généralisation

Un modèle entraîné sur des habitudes positionnelles généralise **sans voir les tokens** — il n'a jamais besoin d'avoir rencontré le verbe ou le connecteur spécifique dans son corpus d'entraînement.

```
Entraîné sur :
  "Est-il efficace ?"        → INTERROGATIVE (VERB avant NOUN + ?)
  "Guérit-il rapidement ?"   → INTERROGATIVE (VERB avant NOUN + ?)

Généralise à :
  "Prescrit-il le médicament ?" → INTERROGATIVE
  même si "prescrire" n'a jamais été vu en position inversée

Justification : le modèle a appris le PROFIL (VERB avant NOUN + ?),
pas les mots qui l'instancient. Tout triplet (VERB, NOUN, ?) en inversion
active ce profil.
```

---

## 15. Taxonomie hiérarchique — relations et nœuds

### 15.1 Le problème de précision sans annotation Mood

Sans `morph.Mood` sur le verbe subordonné, certaines distinctions entre relations causales sont **structurellement indiscernables** depuis la position et le POS seuls.

**Paires ambiguës sans Mood :**

```
CONDITION vs CONCESSION
  Les deux : SCONJ + Clause₂ + Clause₁ (structure identique)
  Signal discriminant manquant : Mood=Sub (concession) vs Mood=Ind (condition)
  → Sans Mood, le modèle confond les deux dans ~30% des cas

CAUSE vs SEQUENCE
  Les deux : Clause₁ + SCONJ + Clause₂ (structure identique)
  Signal discriminant manquant : Tense (séquence → Tense=Past dominant)
                                  Aspect (cause → Aspect=Perf fréquent)
  → Sans Tense/Aspect, le modèle confond dans ~25% des cas

ENABLE vs CAUSE
  Les deux : NOUN + VERB + NOUN (structure transitif identique)
  Signal discriminant manquant : valence sémantique du verbe (dep_rel=obj)
  → Sans dep_rel, le modèle ne peut pas séparer structurellement
```

### 15.2 La solution : taxonomie à deux niveaux

Le moteur apprend d'abord les **distinctions fiables** (niveau coarse), puis affine quand les données le permettent (niveau fine).

```
NIVEAU 1 — Coarse (appris depuis position + POS surface)

  Relations (5 types structurellement distincts) :
  ┌────────────────┬──────────────────────────────────────────────────────┐
  │ CAUSALE        │ Clause₁ + SCONJ/CCONJ + Clause₂, ¬NEG, ¬restriction │
  │                │ Signal : SCONJ ou CCONJ entre deux VERB               │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ RESTRICTIVE    │ ADV_NEG pré-verbal + ADV post-verbal + SCONJ          │
  │                │ Signal : deux ADV encadrant le VERB + SCONJ           │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ NÉGATIVE       │ ADV_NEG pré-verbal + VERB, ¬restriction               │
  │                │ Signal : ADV avant VERB sans second ADV restrictif     │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ SÉQUENTIELLE   │ CCONJ temporel ou SCONJ temporel entre deux VERB      │
  │                │ Signal : position temporelle (ADV temporel adjacent)  │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ SIMPLE         │ Un seul VERB, ¬SCONJ → pas de relation inter-clause   │
  └────────────────┴──────────────────────────────────────────────────────┘

  Nœuds (4 types structurellement distincts au Niveau 1) :
  ┌────────────────┬──────────────────────────────────────────────────────┐
  │ ACTION         │ VERB en position root (pas précédé d'un SCONJ)        │
  │                │ → regroupe PROCESSUS + ÉTAT_LOCAL + ÉTAT_GLOBAL        │
  │                │   (distinction fine = Niveau 2)                       │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ NOMINAL        │ NOUN en position pré- ou post-verbale                 │
  │                │ → regroupe ENTITÉ + CONCEPT (distinction Niveau 2)   │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ ÉVÉNEMENT      │ NOUN + ADV/obl temporel adjacent                      │
  │                │ Signal : présence d'indicateur temporel               │
  │                │ → regroupe ENTITÉ + CONCEPT                           │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ CONDITION      │ SCONJ + VERB dans une clause secondaire               │
  │                │ → regroupe CONDITION + CONTRAINTE                     │
  └────────────────┴──────────────────────────────────────────────────────┘

  Fiabilité du niveau 1 : ~85-90%
  Supervision requise   : 200-300 exemples par type, labels au niveau phrase
```

```
NIVEAU 2 — Fine (appris si Mood/Tense disponibles OU labels supplémentaires)

  Relations (11 types complets) :
    CAUSE / CONDITION / CONCESSION / MOTIVATION / ENABLE
    PREVENT / FILTER / SEQUENCE / OPPOSITION / ANALOGY / COUNTERFACTUAL

  Nœuds (8 types complets) :
    PROCESSUS / ÉTAT_LOCAL / ÉTAT_GLOBAL / ENTITÉ / CONCEPT / CONDITION / ÉVÉNEMENT / CONTRAINTE

  Signal supplémentaire requis :
    morph.Mood   → distingue CONDITION (Ind/Sub) de CONCESSION (Sub)
    morph.Tense  → distingue CAUSE (Pres) de SEQUENCE (Past)
    dep_rel=cop  → distingue ÉTAT de PROCESSUS (copule vs verbe plein)
    dep_rel=obj  → distingue ENABLE de CAUSE (valence transitive)

  Obtenir Mood sans annotation complète :
    → Tagger morphologique minimal (Mood seul)
    → Beaucoup moins coûteux que l'annotation UD complète
    → Disponible pour FR/EN depuis des taggers légers (TreeTagger, spaCy)

  Fiabilité du niveau 2 : ~92-95% (avec Mood disponible)
  Supervision requise   : 400-500 exemples par type fin
```

### 15.3 Principe de progression

```
Le moteur commence au niveau 1 dès le premier lancement.
Il monte au niveau 2 pour un type de relation spécifique
dès que les conditions sont réunies :

  Condition A : Mood disponible pour les phrases de ce type
  Condition B : ≥ 400 exemples étiquetés couvrant les deux
                sous-types ambigus (ex: CONDITION + CONCESSION)

  Sinon, le type reste à la granularité coarse.
  Le graphe causal reste cohérent — il porte juste moins d'information
  sur les relations de niveau fin.

Avantage :
  Le moteur est utile immédiatement, sans corpus annoté.
  Il améliore sa précision de façon incrémentale et autonome.
  Aucune refonte de l'architecture n'est nécessaire entre les deux niveaux.
```

---

## 16. Solution pour les connecteurs

### 16.1 Le problème

Un connecteur (`si`, `parce que`, `bien que`...) est un token ordinaire dans la phrase. Le moteur ne peut pas le reconnaître par son lemme (pas de liste hardcodée). Il doit reconnaître sa **fonction structurelle** — c'est-à-dire qu'il est un connecteur — avant même de déterminer quel type de relation il introduit.

**Trois questions successives :**

```
Q1 : Ce token est-il un connecteur ?        (identification de rôle)
Q2 : Quel type de relation introduit-il ?   (classification sémantique)
Q3 : Dans quelle direction ?                (direction causale)
```

### 16.2 Réponse à Q1 — Identifier un connecteur sans annotation

Un connecteur a une **position structurelle caractéristique** qui ne dépend pas de son lemme :

```
Signature positionnelle d'un connecteur de subordination :
  1. pos = SCONJ                              (POS de surface)
  2. apparaît à la FRONTIÈRE entre deux clauses
     → précède immédiatement un groupe VERB   (SCONJ + ... + VERB)
  3. n'est ni sujet (¬précédé d'un DET+NOUN comme tête)
     ni objet (¬suit immédiatement un VERB transitif sans autre NOUN)

Signature positionnelle d'un connecteur de coordination :
  1. pos = CCONJ
  2. apparaît ENTRE deux groupes de même type  (NOUN+VERB ... CCONJ ... NOUN+VERB)
  3. position symétrique par rapport aux deux membres qu'il relie

Signature du négateur-restricteur :
  1. pos = ADV
  2. position PRÉ-verbale (ADV.id < VERB.id)
  3. co-présence possible d'un second ADV POST-verbal
     → si oui : RESTRICTIF (ne...que pattern)
     → si non  : NÉGATIF pur
```

**Ce que le moteur apprend pour Q1 :**

```
Depuis un corpus non étiqueté (tokenisation + POS surface) :
  Apprendre la distribution de position des SCONJ :
    P(SCONJ précède un VERB dans les N tokens suivants)    → connecteur sub.
    P(SCONJ en position 0 de la phrase)                   → frontière clause frontée
    P(SCONJ en position > premier VERB)                   → frontière clause backgroundée

  Apprendre la distribution de position des CCONJ :
    P(CCONJ entre deux NOUN)                              → coord. nominale
    P(CCONJ entre deux VERB)                              → coord. verbale/propositionnelle

  Ces distributions sont appris sans labels — elles émergent
  de la régularité syntaxique de la langue.
```

### 16.3 Réponse à Q2 — Classifier le type de relation sans lemme

Un connecteur identifié (Q1 résolu) doit être classé en type de relation. Sans lemme, le signal discriminant est le **contexte structural de la subordonnée qu'il introduit**.

**L'hypothèse fondamentale :**

> La sémantique d'un connecteur est partiellement encodée dans la morphologie du verbe qu'il introduit.

```
CONDITION   : SCONJ + Clause(Mood=Sub OU Mood=Ind)
              Le verbe de la subordonnée est au subjonctif ou indicatif.
              Contenu : un événement hypothétique ou une circonstance.

CONCESSION  : SCONJ + Clause(Mood=Sub, souvent)
              Le verbe de la subordonnée est quasi-systématiquement au subjonctif.
              Contenu : un fait admis qui ne suffit pas à bloquer la principale.

CAUSE       : SCONJ + Clause(Mood=Ind, Tense=Past/Pres)
              Le verbe de la subordonnée est à l'indicatif.
              Contenu : un événement établi comme réel.

SÉQUENCE    : SCONJ + Clause(Mood=Ind, Tense=Past/Pres, aspect achevé)
              Le verbe de la subordonnée est à l'indicatif passé.
              Contenu : un événement antérieur achevé.

MOTIVATION  : SCONJ + Clause(Mood=Sub, VerbForm=Inf souvent)
              Le verbe de la subordonnée est au subjonctif ou à l'infinitif.
              Contenu : un but non encore réalisé.
```

**Ce que cela signifie pour l'apprentissage :**

```
La couche 1 (déterministe) N'A PAS BESOIN du lemme du connecteur.
La couche 2 (apprise) PEUT utiliser embedding(connecteur) pour affiner Q2.
La couche 1 a besoin de :
  - (pos=SCONJ) + (Mood du verbe qu'il introduit)
  - Si Mood disponible → Q2 résolu à ~85% de précision (couche 1 seule)

Sans Mood (niveau coarse) :
  Le modèle apprend depuis le corpus :
    P(Clause suivante = Sub | SCONJ en position X) ≈ f(position)
    P(Clause suivante = Ind | SCONJ en position X) ≈ g(position)
  Ces probabilités conditionnelles approximent Mood sans le lire directement.
  Précision dégradée mais non nulle : ~65-70%.
```

### 16.4 Réponse à Q3 — Direction causale

La direction de la relation causale (qui cause quoi) est déterminée par la **position relative de la subordonnée par rapport à la principale** — et cela se lit depuis les positions seules, sans annotation.

```
RÈGLE FONDAMENTALE :
  La subordonnée introduite par le connecteur est la SOURCE (cause/condition).
  La proposition principale est la CIBLE (effet/conséquence).

  Structure A (subordonnée en queue) :
    Clause_principale + SCONJ + Clause_subordonnée
    "Le médicament est prescrit [si les traitements échouent]."
    → CIR_edge(échec_traitements → prescription)
    → SCONJ.id > root_principal.id → subordonnée en queue → condition backgroundée

  Structure B (subordonnée en tête) :
    SCONJ + Clause_subordonnée + , + Clause_principale
    "[Si les traitements échouent], le médicament est prescrit."
    → CIR_edge(échec_traitements → prescription)   [même relation]
    → SCONJ.id < root_principal.id → subordonnée frontée → condition proéminente
    → confiance += 0.1 (le locuteur a mis la cause en premier)

Signal pour détecter la structure A vs B :
  SCONJ.position < position(premier VERB de la proposition sans SCONJ)
    → Structure B (frontée) → condition proéminente
  SCONJ.position > position(premier VERB)
    → Structure A (en queue) → condition backgroundée

Ce signal est calculable depuis tokenisation + POS seuls.
Aucune annotation dep_rel nécessaire.
```

### 16.5 Apprentissage du connecteur — vue d'ensemble

```
PHASE 1 — Auto-supervisé (aucun label causal)
────────────────────────────────────────────────────────────────────────────
Input  : corpus brut + tokenisation + POS surface
Appris : distribution positionnelle des SCONJ/CCONJ/ADV
         → identifier qu'un token est un connecteur (Q1)
         → identifier la structure A vs B (Q3 partiel)
Signal : régularité statistique de la langue (aucun label)


PHASE 2 — Faiblement supervisé (labels au niveau phrase)
────────────────────────────────────────────────────────────────────────────
Input  : corpus de phrases avec type de relation labélé au niveau phrase
         (ex: "cette phrase contient une relation CONDITION")
Appris : association entre (pos(K), structure A/B, Mood approx) et type de relation
         → Q2 au niveau coarse (5 types)
Signal : 200-300 exemples par type de relation (pas par connecteur)


PHASE 3 — Supervisé avec Mood (labels phrase + tagger morpho minimal)
────────────────────────────────────────────────────────────────────────────
Input  : corpus + labels phrase + Mood du verbe subordonné
Appris : association précise (pos(K), Mood(V_sub), position) → 11 types
         → Q2 au niveau fin
Signal : 400-500 exemples par type + Mood disponible


PROPRIÉTÉ CLÉ :
  À chaque phase, le moteur est fonctionnel.
  Il ajoute de la précision sans jamais changer d'architecture.
  Un connecteur jamais vu dans le corpus d'entraînement est classé
  depuis son PROFIL STRUCTUREL — pas depuis son lemme.
```

### 16.6 Vérification — ce que le modèle doit pouvoir faire

```
Test de généralisation :
  Entraîné sur phrases FR avec SCONJ de type CONDITION.
  Présenté avec une phrase DE (allemand) contenant un SCONJ de type CONDITION.

  Si le modèle a appris le profil :
    (SCONJ, VERB_sub à l'indicatif, structure A ou B)
  → il classifie correctement CONDITION
  → sans avoir jamais vu le connecteur allemand "wenn"

  Cela est possible UNIQUEMENT si le modèle n'a pas mémorisé les lemmes
  mais a appris les patterns structurels.
  C'est la vérification que l'apprentissage est bien générique.
```

---

## 17. Composants externes — comment ils apprennent

### 17.1 Architecture du système complet

Le moteur est le composant central — constructeur et interrogateur du CIR. Deux composants externes s'y connectent, l'un en amont (NLU routing), l'autre en aval (verbalisation). Ils sont **symétriques** : l'un convertit du texte vers une requête formelle, l'autre convertit un résultat formel vers du texte.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Utilisateur : "Pourquoi les ventes ont-elles chuté ?"                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ texte naturel
                             ▼
              ┌──────────────────────────────┐
              │     NLU ROUTING (externe)     │
              │  texte → requête formelle     │
              │  explain("ventes")            │
              └──────────────┬───────────────┘
                             │ requête formelle + contexte
                             ▼
              ┌──────────────────────────────┐
              │     MOTEUR CORE              │
              │  CIR builder + reasoner      │
              │  → résultat structuré        │
              └──────────────┬───────────────┘
                             │ résultat CIR + contexte
                             ▼
              ┌──────────────────────────────┐
              │     VERBALISATION (externe)   │
              │  résultat → texte naturel     │
              │  "Les ventes ont chuté        │
              │   parce que la demande a      │
              │   baissé (confiance 90%)."    │
              └──────────────────────────────┘
                             │
                             ▼
              Utilisateur confirme ou corrige
                             │
              ┌──────────────┴───────────────┐
              │  Signal partagé par les 3    │
              │  composants (boucle fermée)  │
              └──────────────────────────────┘
```

**Principe de symétrie :**

```
Corpus annoté (texte, CIR, type_phrase)
       │
       ├── [Forward]  texte → CIR        = Moteur apprend
       ├── [Backward] CIR → texte        = Verbaliseur apprend
       └── [Lateral]  texte → profil → requête = NLU routing dérivé
```

Les trois composants partagent **le même corpus** et **le même signal de retour utilisateur**. Aucun ne requiert un corpus propre.

---

### 17.2 NLU routing — comment il apprend

#### Cas 1 — Phrases simples (dérivation déterministe)

Pour les phrases simples, le NLU routing est **entièrement dérivé du SentenceProfile** calculé par le moteur. Aucun apprentissage supplémentaire n'est nécessaire.

```
Illustration — Table de dispatch depuis SentenceProfile

┌──────────────────────────────────┬────────────────────┬─────────────────────┐
│ Input utilisateur                │ SentenceProfile    │ Requête dérivée     │
├──────────────────────────────────┼────────────────────┼─────────────────────┤
│ "Les ventes ont chuté."          │ DECLARATIVE        │ extract → CIR       │
│ "Pourquoi les ventes ont chuté?" │ INTERROG + pourquoi│ explain("ventes")   │
│ "Comment A mène-t-il à B ?"      │ INTERROG + comment │ chain(A, B)         │
│ "Qui a causé la baisse ?"        │ INTERROG + qui     │ find_effects(baisse) │
│ "Quand X précède-t-il Y ?"       │ INTERROG + quand   │ chain_t(X, Y)       │
│ "Analyse ce fichier."            │ IMPERATIVE         │ action(analyze, arg)│
│ "Quelle chute !"                 │ EXCLAMATIVE        │ extract + intensité  │
└──────────────────────────────────┴────────────────────┴─────────────────────┘
```

#### Cas 2 — Signal de retour moteur (apprentissage endogène)

Pour les cas ambigus, le routeur apprend depuis la **qualité de la réponse du moteur** — sans intervention humaine.

```
Illustration — Boucle d'apprentissage endogène

  Query A : explain("ventes") pour "Que se passe-t-il si les ventes baissent ?"
  Moteur  : retourne "aucune cause directe de 'ventes'"  → résultat vide
  Signal  : négatif → mapping incorrect
            ↓
  Ajustement : INTERROGATIVE + has_condition → counterfactual("baisse_ventes")

  Query B : counterfactual("baisse_ventes")
  Moteur  : retourne [(chute_revenus, 0.9), (licenciements, 0.7)]  → riche
  Signal  : positif → mapping correct
            ↓
  Renforcement : INTERROGATIVE + has_condition + wh=que → counterfactual(condition)
```

Aucun label humain — le moteur est l'oracle de qualité du routage.

---

### 17.3 Verbalisation — comment elle apprend

#### Principe — les paires inversées

Le verbaliseur apprend depuis les mêmes paires `(texte, CIR)` que le moteur, utilisées dans le sens inverse :

```
Illustration — Même paire, deux directions d'apprentissage

  Paire d'entraînement :
    Texte : "La pluie cause les inondations."
    CIR   : CIR_edge(pluie → inondation, CAUSE, 1.0)

  ──────────────────────────────────────────────────────
  Moteur apprend (→) :
    "La pluie cause les inondations."
         ↓
    CIR_edge(pluie, inondation, CAUSE, 1.0)

  Verbaliseur apprend (←) :
    CIR_edge(pluie, inondation, CAUSE, 1.0)
         ↓
    "La pluie cause les inondations."
  ──────────────────────────────────────────────────────
```

#### Ce que le verbaliseur apprend par type de relation

```
Illustration — Patterns appris par type

  CAUSE(A, B, conf=1.0)   → "A cause B."
                             "A entraîne B."
                             "B est causé par A."

  CAUSE(A, B, conf=0.7)   → "A semble causer B."
                             "A contribue à B."

  CAUSE(A, B, conf=0.4)   → "A pourrait être lié à B."

  CONDITION(A, B)         → "Si A, alors B."
                             "B uniquement si A."

  FILTER(A, B)            → "B n'est possible que si A."    [A=condition, B=effet restreint]
                             "B exclusivement sous condition de A."
                             Note : A est la condition nécessaire, B est l'effet — NE PAS inverser.

  PREVENT(A, B)           → "A empêche B."
                             "A bloque B."

  CONCESSION(A, B)        → "Bien que A, B se produit."
                             "Malgré A, B."

  COUNTERFACTUAL(¬A, B)  → "Sans A, B ne se produirait pas."
                             "En l'absence de A, B disparaît."

  SEQUENCE(A, B)          → "D'abord A, puis B."
                             "A précède B."
```

#### Templates ternaires

Les trois relations ternaires ont leurs propres templates avec graduation de confiance et mode verbal selon le contexte de requête.

```
CONDITIONAL_CAUSE(A, B, cond=C) :
  Indicatif (fait extrait)       → "{A} cause {B} uniquement si {C}."
  Conditionnel (hypothèse)       → "{A} causerait {B} uniquement si {C}."
  conf < 0.70                    → "{A} pourrait causer {B}, à condition que {C}."
  Résumé compact                 → "{A} → {B} | condition : {C}"
  Négatif (cible) :              → "{A} empêche {B} uniquement si {C}."

MEDIATED_CAUSE(A, B, via=M) :
  Indicatif                      → "{A} cause {B} par l'intermédiaire de {M}."
                                   "{A} entraîne {B} en passant par {M}."
  Conditionnel                   → "{A} causerait {B} via {M}."
  conf < 0.70                    → "{A} semble causer {B} par le biais de {M}."
  Résumé compact                 → "{A} → {M} → {B}  (mécanisme explicite)"

JOINT_CAUSE([A₁, A₂], B) :
  Indicatif                      → "{A₁} et {A₂} causent conjointement {B}."
                                   "La combinaison de {A₁} et {A₂} produit {B}."
  Conditionnel                   → "{A₁} et {A₂} causeraient conjointement {B}."
  conf < 0.70                    → "{A₁} et {A₂} semblent produire {B} ensemble."
  Note obligatoire               → "Ni {A₁} seul ni {A₂} seul ne suffit."
  Résumé compact                 → "({A₁} ∧ {A₂}) → {B}"

CONDITIONAL_PREVENT(A, B, cond=C) :
  Indicatif                      → "{A} empêche {B} uniquement si {C}."
  Résumé compact                 → "{A} ⊄ {B} | condition : {C}"

MEDIATED_PREVENT(A, B, via=M) :
  Indicatif                      → "{A} empêche {B} par l'intermédiaire de {M}."
```

#### Règle du mode verbal selon le contexte de requête

Le même CIR_edge peut être verbalisé en indicatif ou en conditionnel selon la requête qui l'a produit.

```
┌─────────────────────┬──────────────────────┬──────────────────────────┐
│ Contexte requête    │ Mode verbal           │ Exemple (CAUSE(pluie,B)) │
├─────────────────────┼──────────────────────┼──────────────────────────┤
│ explain / chain     │ Indicatif présent     │ "La pluie cause B."      │
│ find_causes         │ Indicatif présent     │ "La pluie cause B."      │
│ counterfactual      │ Conditionnel          │ "La pluie causerait B."  │
│ chain (hypothèse)   │ Modal possible        │ "La pluie pourrait..."   │
│ conf < 0.50         │ Atténuation lexicale  │ "La pluie semble liée à" │
└─────────────────────┴──────────────────────┴──────────────────────────┘
```

#### Noyau de templates déterministes (cold start)

Le verbaliseur démarre avec ces templates — un par type — fonctionnels dès la Phase 0. L'apprentissage les enrichit progressivement.

```
Relations directes (third=None) :
  CAUSE(A, B)      → "{A} cause {B}."
  CONDITION(A, B)  → "Si {A}, alors {B}."
  PREVENT(A, B)    → "{A} empêche {B}."
  FILTER(A, B)     → "{B} uniquement si {A}."
  SEQUENCE(A, B)   → "{A} précède {B}."
  ENABLE(A, B)     → "{A} permet {B}."
  CONCESSION(A, B) → "Bien que {A}, {B}."

Ternaires :
  CONDITIONAL_CAUSE(A, B, C) → "{A} cause {B} uniquement si {C}."
  MEDIATED_CAUSE(A, B, M)    → "{A} cause {B} par l'intermédiaire de {M}."
  JOINT_CAUSE([A₁,A₂], B)   → "{A₁} et {A₂} causent conjointement {B}."
```

---

### 17.4 Boucle d'apprentissage fermée — illustration complète

```
═══════════════════════════════════════════════════════════════════════
  EXEMPLE COMPLET : du corpus à la réponse en production
═══════════════════════════════════════════════════════════════════════

  ── PHASE 1 : Entraînement (une fois, offline) ─────────────────────

  Corpus :
    (1) "La pluie cause les inondations."
        → CIR_edge(pluie, inondation, CAUSE, 1.0)     [type=DECLARATIVE]

    (2) "Pourquoi les inondations ?"
        → explain("inondation")                         [type=INTERROGATIVE]

    (3) "La pluie cause les inondations."
        ← CIR_edge(pluie, inondation, CAUSE, 1.0)     [verbaliseur inverse]

  Moteur apprend depuis (1) : extraction causale
  NLU apprend depuis (2)    : INTERROGATIVE + pourquoi → explain(C)
  Verbaliseur apprend depuis (3) : CAUSE(A,B) → "A cause B."

  ── PHASE 2 : Production (runtime) ────────────────────────────────

  Input utilisateur : "Les inondations sont-elles causées par la pluie ?"

  [NLU routing]
    SentenceProfile :
      type = INTERROGATIVE  (inversion S-V + ?)
      voice = PASSIVE       (nsubj:pass présent)
    → requête : find_causes("inondation")
    → contexte : {voice: passive, mode: indicatif}

  [Moteur core]
    find_causes("inondation") sur le CIR
    → résultat : [(pluie, CAUSE, 1.0)]

  [Verbaliseur]
    CAUSE(pluie, inondation, 1.0) + contexte(voice=passive)
    → "Les inondations sont causées par la pluie." (voix passive préservée)

  [Utilisateur]
    ✓ Confirme
    → Signal + sur les 3 composants pour ce pattern

═══════════════════════════════════════════════════════════════════════
```

---

## 18. Phrases complexes et conditionnelles — problèmes et solutions

### 18.1 Le problème structurel fondamental

Dans une phrase simple, le moteur traite un triplet `(Agent, VERB, Patient)`. Dans une phrase complexe, il traite **deux propositions reliées** — et la difficulté est que la proposition syntaxiquement subordonnée est souvent logiquement la **source causale**.

```
Illustration — Inversion logique vs syntaxique

  Phrase :
    "Si les traitements échouent, le médicament est prescrit."

  Hiérarchie SYNTAXIQUE (UD) :
    prescrit (root)          ← tête syntaxique = proposition principale
      ├── médicament (nsubj:pass)
      ├── est (aux:pass)
      └── échouent (advcl)   ← subordonné syntaxiquement
            ├── traitements (nsubj)
            └── si (mark)

  Direction LOGIQUE (causal) :
    échec_traitements  →  prescription_médicament
    ↑ SOURCE causale        ↑ CIBLE causale
    (subordonnée)           (principale)

  ══► La hiérarchie syntaxique est INVERSÉE par rapport à
      la direction causale. Le moteur doit apprendre cet inversion.
```

### 18.2 Règle déterministe — direction causale invariante

Pour toutes les relations introduites par un SCONJ (`advcl + mark`), la direction est une **propriété logique**, pas statistique. Elle s'encode comme contrainte dure :

```
╔════════════════════════════════════════════════════════════════════╗
║  RÈGLE INVARIANTE : relations advcl-based                          ║
║                                                                    ║
║  Source ← clause introduite par le SCONJ (la subordonnée)         ║
║  Cible  ← proposition principale                                  ║
║  TOUJOURS — indépendamment de l'ordre d'apparition                ║
╚════════════════════════════════════════════════════════════════════╝

  Illustration — Même relation, deux ordres, même direction CIR

  Structure A (subordonnée en queue) :
    "Le médicament est prescrit  [SI  les traitements échouent]."
                       ↑ principale    ↑ subordonnée
    CIR_edge(échec → prescription, CONDITION)

  Structure B (subordonnée en tête) :
    "[SI  les traitements échouent],  le médicament est prescrit."
      ↑ subordonnée                        ↑ principale
    CIR_edge(échec → prescription, CONDITION)  ← identique

  Seule la saillance change (Éq. 9) :
    Structure A : condition backgroundée   (confiance = 0.9)
    Structure B : condition proéminente   (confiance += 0.10 → max Rung×f×g + 0.10)
```

**Exception — SEQUENCE temporelle :**

```
  "Après que A s'est produit, B se produit."
    → CIR_edge(A → B, SEQUENCE)   [A est antérieur dans le temps]

  "Avant que B se produise, A se produit."
    → CIR_edge(A → B, SEQUENCE)   [même relation — A précède toujours B]

  Direction déterminée par l'ordre TEMPOREL (Tense/position),
  pas par la hiérarchie syntaxique principale/subordonnée.
```

### 18.3 Ce que le moteur apprend pour les phrases complexes

**Tâche 1 — Segmentation des propositions**

```
  Illustration — Apprendre les frontières de propositions

  Depuis corpus non étiqueté (POS + position) :

  Phrase : "Si les traitements échouent , le médicament est prescrit ."
  POS    :  SCONJ  DET     NOUN    VERB  PUNCT DET   NOUN   AUX  VERB  PUNCT

  Pattern observé statistiquement :
    SCONJ en position 0 → commence une subordonnée frontée
    PUNCT(,) après VERB₁ → frontière entre P₁ et P₂
    VERB₂ après la frontière → démarre la principale

  Distribution apprise :
    P(SCONJ démarre une subordonnée) ≈ 0.92
    P(PUNCT + VERB après SCONJ+clause = frontière) ≈ 0.85
    P(clause après frontière = principale) ≈ 0.90
```

**Tâche 2 — Classification du type de relation**

```
  Illustration — Signal structurel par type (niveau coarse vs fine)

  ─────────────────────────────────────────────────────────────────────
  Niveau COARSE (position + POS seuls)

    CAUSALE    : SCONJ + Clause₂ + Clause₁  OU  Clause₁ + SCONJ + Clause₂
                 ¬NEG, ¬restriction
                 → regroupe CAUSE + CONDITION + CONCESSION + MOTIVATION

    RESTRICTIVE : ADV_NEG pré-verbal + SCONJ + Clause₂
                 → regroupe FILTER

    NÉGATIVE    : ADV_NEG pré-verbal, ¬SCONJ
                 → regroupe PREVENT

    SÉQUENTIELLE : SCONJ/CCONJ + ADV temporel adjacent
                 → SEQUENCE

  ─────────────────────────────────────────────────────────────────────
  Niveau FIN (+ Mood du verbe subordonné)

    (SCONJ, Mood_sub=Sub, ¬goal_marker)  → CONDITION ou CONCESSION
    (SCONJ, Mood_sub=Sub, has_goal)      → MOTIVATION
    (SCONJ, Mood_sub=Ind, ¬Neg)          → CAUSE
    (SCONJ, Mood_sub=Ind, Tense=Past)    → SEQUENCE

    Discrimination CONDITION vs CONCESSION :
      CONDITION  : Mood=Sub/Ind,  le fait de P₂ dépend de P₁
      CONCESSION : Mood=Sub,      le fait de P₂ est indépendant de P₁
      → Appris depuis 400+ exemples étiquetés couvrant les deux
```

**Tâche 3 — Conditions imbriquées** *(niveau 2 uniquement)*

```
  Illustration — Phrase à 3 propositions

  "Si A, alors B, sauf si C."

  Propositions :
    P₁ = A (condition principale)
    P₂ = B (effet)
    P₃ = C (condition d'exception)

  Relations attendues :
    CIR_edge(A → B, CONDITION)
    CIR_edge(C → B, FILTER)    ← C modifie B, pas A

  ════ Difficulté ════
  Sans dep_rel, la portée de "sauf si C" est ambiguë :
    Porte-t-il sur A ?  → "A, sauf si C, entraîne B"
    Porte-t-il sur B ?  → "B, sauf si C, se produit"

  Solution : requires dep_rel=advcl ou labels spécifiques
  → Classé en Niveau 2, pas en Niveau 1.
```

### 18.4 NLU routing pour les interrogatives conditionnelles

Une interrogative conditionnelle cumule deux types. Le routeur doit les traiter ensemble.

```
  Illustration — Décomposition d'une interrogative conditionnelle

  Input : "Que se passe-t-il si les ventes baissent ?"

  ─── Analyse du SentenceProfile ─────────────────────────────────────
  sentence_type   = INTERROGATIVE  (inversion "se passe-t-il" + ?)
  has_condition   = True           (advcl + SCONJ "si" présent)
  subordination   = CONDITION
  wh_type         = "que/quoi"     (objet interrogatif)

  ─── Table de dispatch étendue ───────────────────────────────────────
  ┌──────────────────────────────────────────┬──────────────────────────────┐
  │ Profil                                   │ Requête                      │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ INTERROG + condition + wh=que/quoi       │ counterfactual(condition)    │
  │ "Que se passe-t-il si X ?"               │                              │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ INTERROG + condition + wh=pourquoi       │ explain(effet) filtré sur A  │
  │ "Pourquoi B si A ?"                      │                              │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ INTERROG + condition + wh=comment        │ chain(condition, effet)       │
  │ "Comment A conduit-il à B ?"             │                              │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ INTERROG + condition + wh=quand          │ chain_t(A, B)                │
  │ "Quand B si A ?"                         │                              │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ INTERROG simple, ¬condition + pourquoi   │ explain(C)                   │
  │ "Pourquoi B ?"                           │                              │
  └──────────────────────────────────────────┴──────────────────────────────┘

  ─── Apprentissage du routeur pour les cas conditionnels ─────────────
  Signal endogène :

    Route mal choisie : explain("ventes") pour "Que se passe-t-il si X ?"
    → Moteur retourne : "aucune cause directe de ventes"  → résultat vide
    → Signal − : ajuster vers counterfactual(X)

    Route correcte : counterfactual("baisse_ventes")
    → Moteur retourne : [(chute_revenus, 0.9), (licenciements, 0.7)]
    → Signal + : renforcer le mapping INTERROGATIVE+condition+que → counterfactual

  Données requises : 100-200 exemples d'interrogatives conditionnelles
  (bien moins que pour les relations fines — profil positionnel distinct)
```

### 18.5 Verbalisation des résultats conditionnels

Le résultat d'une requête conditionnelle doit être verbalisé au **mode conditionnel** — pas à l'indicatif. Le verbaliseur reçoit le résultat CIR ET le contexte de requête.

```
  Illustration — Contexte de requête détermine le mode verbal

  ════ Même CIR, deux contextes, deux verbalisations ════

  CIR_edge(pluie, inondation, CAUSE, 0.9)

  Contexte A : query_type = "explain"  (fait extrait du corpus)
    → "La pluie cause les inondations (90%)."    [INDICATIF]

  Contexte B : query_type = "counterfactual"  (scénario hypothétique)
    → "La pluie causerait des inondations (90%)."  [CONDITIONNEL]

  ══════════════════════════════════════════════════════════

  Règle apprise :
    query_type = extract/explain  → INDICATIF (fait établi)
    query_type = counterfactual   → CONDITIONNEL (hypothèse)
    query_type = chain_t          → PRÉSENT factuel

  Graduation par confiance + contexte hypothèse :
    conf=0.9 + hypothèse  → "causerait" (conditionnel présent)
    conf=0.7 + hypothèse  → "pourrait causer" (modal)
    conf=0.4 + hypothèse  → "semblerait lié à" (atténuation)
```

### 18.6 Boucle complète — phrase conditionnelle de bout en bout

```
═══════════════════════════════════════════════════════════════════════════
  EXEMPLE COMPLET : phrase conditionnelle de l'input à la réponse
═══════════════════════════════════════════════════════════════════════════

  ── Corpus d'entraînement ──────────────────────────────────────────────

    Phrase : "Si la demande baisse, les ventes chutent."
    CIR    : CIR_edge(baisse_demande → chute_ventes, CONDITION, 0.95)
    Type   : DECLARATIVE + COMPLEX + CONDITION + subordonnée frontée

    Moteur apprend    (→) : texte → CIR_edge(CONDITION, direction correcte)
    Verbaliseur apprend (←) : CIR_edge → "Si A, B."
    NLU apprend     (↔) : DECLARATIVE + condition → extract (pas une requête)

  ── Production (runtime) ───────────────────────────────────────────────

    Input : "Que se passe-t-il si la demande baisse ?"

    ① NLU routing
      SentenceProfile :
        type = INTERROGATIVE (inversion + ?)
        has_condition = True (SCONJ "si" + advcl)
        wh = "que" (objet interrogatif)
      → Requête : counterfactual("baisse_demande")
      → Contexte : {hypothesis: True, mode: conditionnel}

    ② Moteur core
      counterfactual("baisse_demande") sur le CIR
      → Effets trouvés :
          [(chute_ventes,   CONDITION, 0.95),
           (perte_revenus,  CAUSE,     0.80),
           (licenciements,  ENABLE,    0.60)]
      → Direction : tous les nœuds EN AVAL de baisse_demande

    ③ Verbaliseur
      Reçoit : (effets, {hypothesis:True, mode:conditionnel})
      Génère :
        "Si la demande baissait :
         → les ventes chuteraient (très probable, 95%)
         → les revenus baisseraient en conséquence (80%)
         → des licenciements deviendraient possibles (60%)"

    ④ Utilisateur confirme ✓
      → Signal + sur les 3 composants

  ── Ce que chaque composant a appris de cet échange ───────────────────

    NLU routing :
      INTERROGATIVE + has_condition + wh=que → counterfactual  [renforcé]

    Moteur core :
      CIR étendu avec le contexte de la requête comme validation

    Verbaliseur :
      CONDITION(A,B, hypothèse) → "Si A, B-rait"  [renforcé]
      Graduation : conf=0.95 → "très probable"
                   conf=0.60 → "deviendraient possibles"

═══════════════════════════════════════════════════════════════════════════
```

### 18.7 Limites réelles des phrases conditionnelles

```
  ┌─────────────────────────────┬────────────────────────┬─────────────────┐
  │ Cas                         │ Problème               │ Niveau requis   │
  ├─────────────────────────────┼────────────────────────┼─────────────────┤
  │ Condition simple            │ Aucun                  │ Niveau 1        │
  │ "Si A, B."                  │                        │                 │
  ├─────────────────────────────┼────────────────────────┼─────────────────┤
  │ CONDITION vs CONCESSION     │ Même structure surface │ Niveau 2 (Mood) │
  │ "si A, B" vs "bien que A,B" │ sans Mood              │                 │
  ├─────────────────────────────┼────────────────────────┼─────────────────┤
  │ Condition niée              │ Négation d'une         │ Niveau 2        │
  │ "si A n'échoue pas"         │ condition déjà         │ (Mood + labels) │
  │                             │ conditionnelle         │                 │
  ├─────────────────────────────┼────────────────────────┼─────────────────┤
  │ Conditions imbriquées       │ Portée du 2e SCONJ     │ Niveau 2        │
  │ "Si A, B, sauf si C"        │ ambiguë sans dep_rel   │ (dep_rel requis)│
  ├─────────────────────────────┼────────────────────────┼─────────────────┤
  │ Concession conditionnelle   │ "même si" ≈ "si" en    │ Niveau 2 (Mood) │
  │ "Même si A, B"              │ surface POS            │                 │
  └─────────────────────────────┴────────────────────────┴─────────────────┘

  Conclusion :
    Cas simples (une condition, une conséquence) → Niveau 1 → sans Mood
    Cas ambigus (concession vs condition, imbrication) → Niveau 2 → Mood requis
    Mood reste bien moins coûteux que l'annotation UD complète.
```

---

## 19. Réduction du moteur à sa fonction core

### 19.1 Définition formelle

```
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  Le moteur est un constructeur et interrogateur de représentation     ║
║  logique des relations causales.                                      ║
║                                                                       ║
║  Input  : tokens structurés (POS + position + Mood si disponible)    ║
║  Output : CIR valide — graphe causal typé, orienté, avec confiances  ║
║                                                                       ║
║  Le moteur ne sait pas :                                              ║
║    - d'où viennent les tokens (REPL, API, fichier, batch...)          ║
║    - ce que l'utilisateur voulait dire                                ║
║    - comment présenter la réponse                                     ║
║                                                                       ║
║  Le moteur sait uniquement :                                          ║
║    - quelles relations causales existent dans le texte                ║
║    - avec quelle certitude                                            ║
║    - dans quelle direction                                            ║
║    - et peut répondre à des requêtes causales formelles sur ce CIR   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### 19.2 Ce qui entre dans le moteur, ce qui en sort

```
  DANS LE MOTEUR (responsabilité unique)
  ──────────────────────────────────────
  ① Extraction  : tokens → CIR_edges (nœuds + relations + confiances)
  ② Accumulation: CIR partiel + CIR existant → CIR enrichi
  ③ Inférence   : CIR + requête → résultat structuré
                  (WHY, CHAIN, SPOF, BEFORE, DELAY, COUNTERFACTUAL...)
                  Ces opérations font partie de la sémantique causale
                  — elles ne sont pas une couche applicative séparée.

  HORS DU MOTEUR (interfaces externes)
  ─────────────────────────────────────
  NLU routing  : texte → requête formelle  (interface d'entrée)
  Verbalisation: résultat → texte naturel  (interface de sortie)
  Persistance  : sauvegarder/charger le CIR (infrastructure)

  Illustration :
    ┌──────────┐    requête       ┌─────────────────┐    résultat    ┌───────────┐
    │  NLU     │────────────────→ │  MOTEUR CORE    │ ────────────→  │  VERBAL.  │
    │ routing  │                  │  CIR + reasoner │                │           │
    └──────────┘                  └─────────────────┘                └───────────┘
         ↑                               │                                 │
         │                               ↓                                 ↓
    texte brut                      CIR (état)                        texte naturel
    (utilisateur)                   (persistant)                    (utilisateur)
```

### 19.3 Ce qui est gagné par cette réduction

```
  ① Responsabilité unique et testable
     Input/output déterministes → test automatique fiable
     CIR(texte_1) == CIR(texte_2) si les deux décrivent le même fait

  ② Découplage total
     NLU peut être remplacé sans toucher au moteur
     Verbaliseur peut être remplacé sans toucher au moteur
     Moteur peut être amélioré sans casser les interfaces

  ③ Composabilité
     Plusieurs NLU (différentes langues) → un seul moteur
     Plusieurs verbaliseurs (FR, EN, JSON, graphe) → un seul moteur
     CIRs de plusieurs sources mergés → un seul graphe cohérent

  ④ Extension naturelle
     Ajouter un nouveau type de requête (ex: DENSITY, RELIABILITY) :
     → modifier uniquement le reasoner dans le moteur
     → NLU et verbaliseur s'adaptent automatiquement
       (le NLU apprend le nouveau pattern depuis les retours moteur)

  ⑤ Rien n'est perdu
     La "réponse en langage naturel" n'était pas dans le moteur —
     elle était dans la couche verbalisation, qui reste intacte.
     Séparer les deux rend la limite explicite, pas invisible.
```

---

## 20. Hypothèses de travail — à prouver avant de considérer comme acquises

Les valeurs quantitatives ci-dessous sont des **hypothèses de travail** (H), pas des résultats mesurés. Elles sont raisonnables comme priors mais ne constituent pas de preuves empiriques. Chaque H doit être validée ou réfutée par les 5 tests falsificateurs (§20.1-20.2) avant de figurer dans un résultat.

| Hypothèse | Claim | Source dans le doc | Statut | Test falsificateur |
|-----------|-------|-------------------|--------|-------------------|
| H1 | Désambiguïsation : positions 80%, embeddings 15%, types 3%, consensus 1% | §12.7 | **Non prouvé** — ventilation assertée, 0 mesure | 25 phrases hors-lexique, mesurer chaque mécanisme séparément |
| H2 | Coarse 85-90%, fin 92-95% | §15 | **Non prouvé** — edge réel 0.21-0.30 | Évaluer sur corpus équilibré 8 nœuds + 14 relations |
| H3 | CONDITION/CONCESSION ~30% confusion sans Mood | §15 | **Non prouvé** — pas de matrice de confusion | Ablation Mood sur corpus annoté : mesurer F1 avant/après |
| H4 | Q2 connecteur : 85% avec Mood, 65-70% sans | §16 | **Non prouvé** | Classifier CONDITION vs CONCESSION avec/sans feature Mood |
| H5 | Habitudes positionnelles : distributions P(NOUN<VERB\|décl)≈0.85 etc. | §14 | **Non prouvé** — jamais mesuré sur corpus | Calculer distributions depuis UD_French-GSD |
| H6 | Seuils N_min=50→20, N_fine=400, N_val=100, θ=0.65 | §12.6, §11.3 | **Partiel** — gold 219 ≈ Phase1 basse seulement, isotonique jamais appliquée | Mesurer calibration isotonique dès N_val≥100 réel |
| H7 | Priors rung : R1=0.50, R2=0.75, R3=0.65 et g-table | §9.0 | **Priors uniquement** — cohérence ordinale vérifiée, calibration empirique non faite | Isotonique sur N_val≥100 (cf. §9.5) |

### 20.1 Tests officiels — falsificateurs (H1-H7) + non-régression (S1-S5)

Les tests T1-T5 sont des **tests falsificateurs** (validés comme officiels, D1-D7 v7).
Les scénarios S1-S5 sont des **tests de non-régression bout en bout** (validés comme officiels, D1-D7 v7).

**Scénarios de non-régression S1-S5 :**

```
S1 — Pivot FILTER direct (third=None) (§13)
  Input  : "Le médicament n'est prescrit que si les traitements ont échoué."
  Attendu : FILTER direct (third=None), sources=[échec], third=null, conf=0.75, passive, background
  Écueils : ne pas forcer CONDITIONAL_CAUSE (Agent=∅), ne pas ajouter +0.10 (advcl_after_root)

S2 — Boucle inondations → find_causes → signal+ (§17.4)
  Input  : "Les inondations sont-elles causées par la pluie ?"
  Attendu : INTERROGATIVE détectée, find_causes("inondation"),
            réponse en voix passive préservée, signal+ sur les 3 composants
  Écueils : obl:agent ignoré, ? perdu en extract, query_type non propagé au verbaliseur

S3 — Interrogative conditionnelle → counterfactual (§18.4)
  Input  : "Que se passe-t-il si les ventes baissent ?"
  Attendu : INTERROGATIVE + has_condition + wh=que → counterfactual("baisse_ventes")
  Écueils : SCONJ manqué → explain vide, wh confondu avec pourquoi, hypothesis non propagé

S4 — Verbaliseur mode conditionnel (§18.5)
  Input  : même CIR_edge(pluie, inondation, CAUSE, 0.50)
  Attendu A : query_type=explain   → "La pluie cause les inondations."   [indicatif]
  Attendu B : query_type=counterfactual → "La pluie causerait des inondations."  [conditionnel]
  Écueils : query_type non transmis au verbaliseur, voix non préservée

S5 — Conditionnelle E2E (§18.6)
  Input  : "Que se passe-t-il si la demande baisse ?"
  Attendu : counterfactual("baisse_demande") → [(chute_ventes, 0.50), ...]
            verbalisation conditionnelle, conf≠0.95 (Phase 0 → priors ≤ 0.75)
  Écueils : arête inversée (surface order), 0.95 asserté sans calibration, portée sauf-si non résolue
```

### 20.2 Les 5 tests falsificateurs (H1-H7)

```
T1 — Test de généralisation structurelle (réfute H1, H2, H4)
  Entraîner sur FR, évaluer sur phrases DE/ES contenant SCONJ+Mood=Sub
  → Si accuracy < 60% : l'apprentissage est lexical, pas structurel (H1 tombe)
  → Phrase cible : « Wenn die Behandlungen scheitern, wird das Medikament verschrieben. »

T2 — Test hors-lexique (réfute H1, H2)
  25-50 phrases FR réelles avec verbes et noms ABSENTS du corpus d'entraînement
  → Si node < 70% (vs 99% in-vocabulary) : mémorisation lexicale confirmée (H1 tombe)
  → Source : textes médicaux réels non issus des templates EXPERIMENT_990

T3 — Ablation Mood (réfute H3, H4)
  Évaluer classification CONDITION vs CONCESSION avec et sans feature Mood
  → Mesurer : F1(Mood=présent) vs F1(Mood=absent) sur corpus annoté
  → Si ΔF1 < 10% : Mood n'est pas le signal discriminant (H3 et H4 tombent)

T4 — Correction has_advcl (réfute H2, isole le plafond réel)
  Corriger bridge.py:187,203 → has_advcl=True dérivé de node_type=condition
  → Mesurer edge avant/après sur les phrases complexes en inférence
  → Si edge ne change pas : le plafond est l'architecture paire, pas le bridge

T5 — Calibration isotonique (réfute H6, H7)
  Dès N_val ≥ 100, appliquer isotonique sur confiances brutes
  → Mesurer ECE (Expected Calibration Error) avant/après
  → Si ECE < 0.05 : priors validés comme initialisations
  → Si ECE > 0.15 : priors significativement biaisés — revoir R1/R2/R3
```

---

## 21. Références

- **Universal Dependencies Guidelines** : https://universaldependencies.org/guidelines.html  
- **Universal Dependencies Tagset FR** : https://universaldependencies.org/fr/  
- **Tesnière, L.** — *Éléments de syntaxe structurale* (1959) — fondateur de la valence verbale et de la grammaire de dépendances  
- **Pearl, J.** — *Causality: Models, Reasoning and Inference* (2000) — do-calculus, contrefactuels, DAG causaux  
- **Mel'čuk, I.** — *Dependency Syntax: Theory and Practice* (1988) — formalisation des dépendances syntaxiques  
- **Manning & Schütze** — *Foundations of Statistical NLP* (1999) — extraction d'information et parsing  
- **Nivre, J. et al.** — *Universal Dependencies* (2016) — guidelines UD multilingues  
- **Marneffe, M.-C. & Manning, C.** — *Stanford Typed Dependencies Manual* (2008) — sémantique des dep_rels  
- **Pitler, E. et al.** — *Easily Identifiable Discourse Relations* (2008) — les SCONJ/CCONJ sont les marqueurs discursifs les plus fiables structurellement, sans annotation  
- **Prasad, R. et al.** — *Penn Discourse Treebank* (2008) — taxonomie empirique des relations discursives, base des 11 types de relations causales  
