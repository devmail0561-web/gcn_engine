# Étude comparative — Moteur actuel vs Spécification théorique
## Analyse des lacunes pour la refonte de GCN-Core

**Auteur** : Michel Tendeng  
**Date** : 2026-09-27 (révisé au fil des analyses v1-v5)  
**Base de comparaison** : `docs/ETUDE_LINGUISTIQUE_NLU.md` vs code source inspecté  
**Méthode** : lecture directe du code source — aucune affirmation sans vérification  
**Référence décisions gelées** : D1-D7 dans `ETUDE_LINGUISTIQUE_NLU.md §1.0`

### Désalignements satellites à réconcilier avant implémentation

| Document | Désalignement vs ETUDE | Action |
|----------|----------------------|--------|
| `REGLE_EQUILIBRE_DATASET.md` | 7 nœuds anciens (sans état_local/global, sans concept/événement/contrainte), 11 relations (sans ternaires), quotas inexistants pour les 3 nouveaux types | Mettre à jour quotas ±10% pour 8 nœuds + 14 relations |
| `EXPERIMENT_990_DATASET.md` | 5 types nœuds pré-migration (action/etat/etat_systemique), 11 relations ternaires (third=None implicite) seules | Résultats valides historiquement, non reproductibles sur spec actuelle |
| `AUDIT_COMPARATIF_MOTEUR.md` (ce fichier) | Tableau nœuds §3 mis à jour (8 types, D5), tableau relations §2 mis à jour (11+3+3_prevent) | Mis à jour — voir §2 et §3 ci-dessous |
| Code Python (`constants.py`) | 7 nœuds, 11 relations, zéro ternaire | Migration D5 = Phase C, D2 = Phase A |
| Code Rust (`gcn-ir/src/edge.rs`) | 11 relations, zéro ternaire, analogy/counterfactual absents | Migration D2 = Phase A |

---

## Résumé exécutif

| Gravité | Nombre | Domaines principaux |
|---------|--------|---------------------|
| **CRITIQUE** | 9 | Modèle ternaire, sentence_type non branché, features ternaires, execute() incomplet |
| **MAJEURE** | 16 | RelationType Rust incomplet, nœuds divergents, features positionnelles, verbalisation |
| **MINEURE** | 4 | Types nœuds granularité, salience absente du CIR |

**Ce qui fonctionne bien :** le raisonnement Pearl (WHY, CHAIN, SPOF, COUNTERFACTUAL, ANALOGY, BEFORE, DELAY, 20+ requêtes) est complet et testé. L'accumulation CIR est fonctionnelle. La suite de tests est solide (28 fichiers, 425+ tests).

---

## 1. Modèle ternaire

La spécification adopte un modèle **ternaire** `(sources[], target, relation, third?)`. Le moteur actuel utilise un tuple sans tiers `(NodeId, NodeId, CausalEdge)` — sources[] et third non supportés.

### 1.1 Structure d'arête

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Format d'arête | `{sources:[NodeId,…], target, third:{role,node}?}` | Tuple sans tiers `(NodeId, NodeId, CausalEdge)` — sources[] et third absents | **CRITIQUE — ABSENT** |
| Multi-sources (joint_cause) | `sources=[A₁,A₂]` | `edge_norm.py` L121 : liste normalisée en mono-source — **le deuxième source est silencieusement perdu** | **CRITIQUE — DIVERGENT** |
| Champ `third` | `{role:"condition"\|"mediator", node:NodeId}` | Absent du CIR Rust et Python | **CRITIQUE — ABSENT** |

**Impact :** trois patterns causaux fondamentaux sont **impossibles à représenter** :

```
JOINT_CAUSE :
  "La hausse des coûts ET la baisse de la demande ont provoqué la faillite."
  Moteur actuel : deux arêtes indépendantes → perd la nécessité conjointe
  Attendu       : CIR_ternary([coûts, demande], faillite, JOINT_CAUSE)

CONDITIONAL_CAUSE :
  "Le médicament réduit la douleur uniquement si la dose est correcte."
  Moteur actuel : deux arêtes séparées → lien entre cause et condition perdu
  Attendu       : CIR_ternary(médicament, réduction, CONDITIONAL_CAUSE,
                               third={role:"condition", node:dose})

MEDIATED_CAUSE :
  "La pluie cause les inondations en saturant le sol."
  Moteur actuel : chaîne A→C→B → le lien direct A→B disparaît
  Attendu       : CIR_ternary(pluie, inondations, MEDIATED_CAUSE,
                               third={role:"mediator", node:saturation_sol})
```

---

## 2. Types de relations

### 2.1 Relations dans le Rust (`gcn-ir/src/edge.rs`)

| Relation | Spécification | Moteur Rust | Lacune |
|----------|--------------|-------------|--------|
| cause | ✓ | ✓ Cause | OK |
| condition | ✓ | ✓ Condition | OK |
| enable | ✓ | ✓ Enable | OK |
| prevent | ✓ | ✓ Prevent | OK |
| concession | ✓ | ✓ Concession | OK |
| motivation | ✓ | ✓ Motivation | OK |
| sequence | ✓ | ✓ Sequence | OK |
| filter | ✓ | ✓ Filter | OK |
| opposition | ✓ | ✓ Opposition | OK |
| **analogy** | ✓ | **ABSENT** de `RelationType` | **MAJEURE — ABSENT** |
| **counterfactual** | ✓ | **ABSENT** — existe comme `Query::Counterfactual` mais pas comme type d'arête stockable | **MAJEURE — DIVERGENT** |
| data_dependency | ✓ | ✓ DataDependency | OK |
| control_dependency | ✓ | ✓ ControlDependency | OK |
| **conditional_cause** | ✓ ternaire (D2) | **ABSENT** | **CRITIQUE — ABSENT** |
| **mediated_cause** | ✓ ternaire (D2) | **ABSENT** | **CRITIQUE — ABSENT** |
| **joint_cause** | ✓ ternaire (D2) | **ABSENT** | **CRITIQUE — ABSENT** |
| **conditional_prevent** | ✓ algèbre négation | **ABSENT** | **MAJEURE — ABSENT** |
| **mediated_prevent** | ✓ algèbre négation | **ABSENT** | **MAJEURE — ABSENT** |
| **joint_prevent** | ✓ algèbre négation | **ABSENT** | **MAJEURE — ABSENT** |

**Décisions gelées :** D2 (ternaire) et D6 (algèbre négation). Les 3 `*_prevent` découlent directement de D2 — ils sont issus de l'algèbre de négation ternaire (§9.4) et doivent être ajoutés en même temps que les 3 ternaires positifs.

**Conséquence :** `analogy` est requêtable mais non stockable en CIR. `counterfactual` est calculé à la requête mais non persisté. Ces deux lacunes bloquent la construction de graphes causaux qui "se souviennent" des analogies et contrefactuels détectés.

---

## 3. Types de nœuds

### 3.1 Comparaison des taxonomies (décision D5 — ETUDE_LINGUISTIQUE_NLU §8)

La taxonomie unifiée à 8 types est **gelée (D5)**. Le tableau ci-dessous montre le delta code → spec.

| Spec (8 types, D5) | Moteur actuel (`constants.py`) | Migration | Lacune |
|--------------------|-------------------------------|-----------|--------|
| `processus` | `action` + `transition` + `processus` | fusion → `processus` | **MAJEURE — MIGRATION** |
| `état_local` | `etat` | renommage | **MAJEURE — MIGRATION** |
| `état_global` | `etat_systemique` | renommage | **MAJEURE — MIGRATION** |
| `entité` | `entite` | inchangé | OK |
| `condition` | `condition` | inchangé | OK |
| **`concept`** | **ABSENT** | nouveau signal : NOUN+quantitatif | **MAJEURE — ABSENT** |
| **`événement`** | **ABSENT** | nouveau signal : NOUN+obl:tmod | **MINEURE — ABSENT** |
| **`contrainte`** | **ABSENT** | nouveau signal : advcl+Neg+restriction | **MINEURE — ABSENT** |

**Impact :** `action` et `transition` ne sont pas supprimés mais fusionnés — migration conservatrice. Les 3 nouveaux types (concept/événement/contrainte) requièrent des signaux UD spécifiques documentés dans §8.3 de l'ETUDE.

---

## 4. Pipeline d'extraction (Équations 0 à 9)

### 4.1 Équation 0 — Décomposition quintuplet `(S, V, C*, K?, T?)`

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Quintuplet avec tiers T | `Φ = (S, V, C*, K?, T?)` | `UDRepresentation` = 10 champs — pas de tiers T | **CRITIQUE — ABSENT** |
| Détection JOINT_CAUSE | `S=[S₁,S₂]` si deux sujets coordonnés | Non implémenté dans `ir_emitter.py` | **CRITIQUE — ABSENT** |
| Détection CONDITIONAL_CAUSE | `T.role=condition` si obl conditionnel | Non implémenté | **CRITIQUE — ABSENT** |
| Détection MEDIATED_CAUSE | `T.role=mediator` si obl avec "en/par/via" | Non implémenté | **CRITIQUE — ABSENT** |

### 4.2 Équation 5 — Extraction ternaire

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Extraction ternaire | Détecte les 3 patterns, produit `CIR_ternary(…)` | `ir_emitter.py` : `edge_triples: list[tuple[int,int,str,float,bool,int\|None]]` — **third et sources[] absents** | **CRITIQUE — ABSENT** |

### 4.3 Équation 7 — Normalisation voix active/passive

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Normalisation Agent/Patient | Passif → inverser source/cible avant CIR | Non implémenté — voix passive non détectée dans le pipeline d'extraction | **MAJEURE — ABSENT** |
| `Voice=Pass` dans morph | Signal primaire passif | Absent du `FeatureVocabulary` et des features encodées | **MAJEURE — ABSENT** |

**Conséquence :** "Le médicament est prescrit par le médecin" et "Le médecin prescrit le médicament" produisent des CIR différents pour la même relation causale. L'invariance active/passive (P4) n'est pas garantie.

### 4.4 Équation 9 — Saillance positionnelle

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Features `subj_before_root` | Signal inversion S-V → interrogatif | Absent du `FeatureVocabulary` | **MAJEURE — ABSENT** |
| Features `advcl_before_root` | Condition proéminente vs backgroundée | Absent | **MAJEURE — ABSENT** |
| Features `neg_before_root` | Portée négative totale vs partielle | Absent | **MAJEURE — ABSENT** |
| `prominence` dans CIR Edge | `"foreground"\|"background"` | Absent du schéma | **MAJEURE — ABSENT** |
| `salience` dans CIR | `{topic_node, focus_node, condition_prominence}` | Absent | **MAJEURE — ABSENT** |

---

## 5. Features d'apprentissage

### 5.1 FeatureVector structurel

| Feature | Spécification | Moteur actuel | Lacune |
|---------|--------------|---------------|--------|
| `morph.Mood` | ✓ signal type de phrase | ✓ Encodé dans `FeatureVocabulary` | OK |
| `morph.Tense` | ✓ SEQUENCE vs CAUSE | ✓ Encodé | OK |
| `morph.Aspect` | ✓ Imp/Perf | ✓ Encodé | OK |
| `morph.Polarity` | ✓ signal PREVENT/FILTER | ✓ Encodé | OK |
| `morph.PronType=Int` | ✓ signal interrogatif primaire | **ABSENT** des features UD encodées | **MAJEURE — ABSENT** |
| `morph.Voice=Pass` | ✓ signal passif | **ABSENT** | **MAJEURE — ABSENT** |
| Features positionnelles | `subj_before_root`, `advcl_before_root`, etc. | **Totalement absentes** | **MAJEURE — ABSENT** |
| Features ternaires | `has_two_sources`, `has_third_condition`, `has_obl_mediator` | **Totalement absentes** | **CRITIQUE — ABSENT** |
| Taxonomie coarse/fine | Niveau 1 (position+POS) + Niveau 2 (Mood+labels) | Un seul niveau d'apprentissage | **MAJEURE — ABSENT** |

### 5.2 `has_advcl` dans le bridge

```
Spécification : has_advcl = True si un token a dep_rel=advcl dans la clause
Moteur actuel : bridge.py L187 et L203 → has_advcl=False TOUJOURS en inférence
```

**C'est la lacune la plus impactante sur la qualité d'extraction.** Toutes les phrases complexes traitées par le bridge (inférence depuis texte brut) perdent leur signal de subordination. `has_advcl` est toujours à 0.0 dans le vecteur de features — le modèle ne "voit" jamais de subordonnée en mode production.

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| `has_advcl` en inférence | Calculé depuis les tokens | `has_advcl=False` hardcodé dans bridge | **CRITIQUE — DIVERGENT** |

---

## 6. Annotation et habitudes positionnelles

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Apprentissage positionnel | POS surface + position (id) → type de phrase sans annotation lourde | Requiert annotation UD complète — pas de chemin dégradé POS-only | **MAJEURE — ABSENT** |
| Labels au niveau phrase | 500 exemples par type de phrase suffisent | Labels par arête (relation_type) uniquement — tâche différente | **MAJEURE — DIVERGENT** |
| Habitudes d'apparition | Distributions positionnelles apprises depuis corpus non étiqueté | Non implémenté — supervisé uniquement | **MAJEURE — ABSENT** |

---

## 7. NLU Routing

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Dérivation depuis SentenceProfile | INTERROGATIVE → routing automatique | `parse_command()` : DSL manuel — **l'utilisateur doit taper `explain:` etc.** | **CRITIQUE — DIVERGENT** |
| `sentence_type.py` branché | SentenceProfile guide le routing | `sentence_type.py` **zéro import** depuis `discuss.py`, `instructions.py` ou le pipeline — module isolé | **CRITIQUE — ABSENT** |
| `execute(text, tokens, markers)` | Paramètres optionnels pour NLU | `execute(self, text: str)` — tokens et markers absents | **CRITIQUE — ABSENT** |
| Interrogative conditionnelle | `has_condition` → `counterfactual(X)` | Non implémenté | **CRITIQUE — ABSENT** |
| Retour moteur comme oracle | Résultat vide → signal − → ajustement routing | Non implémenté | **MAJEURE — ABSENT** |

---

## 8. Verbalisation

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Templates langage naturel | Un template par type de relation (`CAUSE → "A cause B."`) | `ReferenceDecoder` : sortie structurée lisible (`src →[rel]→ dst (conf%)`) — sérialisation, pas verbalisation naturelle | **MAJEURE — DIVERGENT** |
| Paires inversées (CIR → texte) | Même corpus que le moteur, direction inverse | `TrainableDecoder` mentionné comme TODO dans `decoder.py` — non implémenté | **MAJEURE — ABSENT** |
| Mode conditionnel selon requête | `query_type=counterfactual → conditionnel` | Non implémenté | **MAJEURE — ABSENT** |
| Graduation confiance | `conf=0.7 → "semble causer"` vs `conf=1.0 → "cause"` | Confiance affichée en % mais n'influence pas le choix lexical | **MINEURE — ABSENT** |
| Boucle retour utilisateur | Confirmation → signal + sur 3 composants | Non implémenté | **MAJEURE — ABSENT** |

---

## 9. Frontières du moteur core

| Aspect | Spécification | Moteur actuel | Lacune |
|--------|--------------|---------------|--------|
| Séparation NLU / Moteur / Verbalisation | Trois composants distincts | `discuss.py` intègre NLU, REPL, session, verbalisation, accumulation CIR **dans un seul module** | **MAJEURE — DIVERGENT** |
| Raisonnement Pearl (20+ requêtes) | WHY, CHAIN, SPOF, BEFORE, DELAY, COUNTERFACTUAL, ANALOGY... | **Implémentés et fonctionnels** — couverture complète ✓ | **OK** |
| Accumulation CIR | CIR partiel + existant → enrichi | `CausalGraph.add_cir()` ✓ | **OK** |
| `salience` dans CIR | `{topic_node, focus_node, condition_prominence}` | Absent du schéma CIR | **MAJEURE — ABSENT** |

---

## Synthèse des lacunes par ordre de priorité de refonte

### Priorité 1 — Lacunes bloquantes pour la refonte ternaire (CRITIQUE)

```
L1. Modèle sans tiers → ternaire complet (sources[], third?)
    Fichiers : gcn-ir/src/edge.rs, ir_emitter.py, edge_norm.py L121
    Impact   : impossibilité de représenter JOINT/CONDITIONAL/MEDIATED_CAUSE

L2. has_advcl=False hardcodé dans bridge
    Fichier  : frontend/bridge.py L187, L203
    Impact   : toutes les phrases complexes perdent leur signal en production

L3. sentence_type.py non branché
    Fichier  : layer1/sentence_type.py (zéro import)
    Impact   : NLU routing impossible — DSL manuel obligatoire

L4. execute() sans tokens/markers
    Fichier  : verbalizer/instructions.py L395
    Impact   : routing NLU depuis SentenceProfile impossible

L5. Features ternaires absentes du FeatureVector
    Fichier  : layer1/features.py
    Impact   : le modèle ne peut pas détecter ni apprendre les patterns ternaires
```

### Priorité 2 — Lacunes majeures pour la complétude (MAJEURE)

```
L6.  analogy et counterfactual absents de RelationType Rust
L7.  PronType=Int absent des features → signal interrogatif manquant
L8.  Voice=Pass absent des features → invariance active/passive non garantie
L9.  Features positionnelles totalement absentes (Éq. 9 — saillance)
L10. Normalisation voix active/passive non implémentée (Éq. 7)
L11. Types nœuds manquants : concept, événement, contrainte
L12. Taxonomie coarse/fine absente (un seul niveau d'apprentissage)
L13. Verbalisation en langage naturel absente
L14. Frontières moteur/NLU/verbalisation non séparées
L15. Salience absente du schéma CIR
```

### Priorité 3 — Lacunes mineures (MINEURE)

```
L16. etat_systemique non documenté dans la spec (à intégrer ou supprimer)
L17. action/transition vs processus : granularité divergente
L18. Graduation confiance dans la verbalisation lexicale
L19. Boucle retour utilisateur non implémentée
```

---

## Ce qui fonctionne et doit être préservé

```
✓ Raisonnement Pearl (20+ requêtes) — gcn-backend/src/query.rs + pearl.rs
  WHY, WHAT, CHAIN, CHAIN_T, BEFORE?, DELAY, SPOF, COUNTERFACTUAL,
  ANALOGY, ZOOM_IN/OUT, AGGREGATE, CENTRALITY, DIFF, DENSITY,
  COVERAGE, RELIABILITY, CYCLES, GAPS, DO

✓ Accumulation CIR (CausalGraph.add_cir)

✓ Suite de tests (28 fichiers, 425+ tests Python, 185+ tests Rust)

✓ Relations directes sans tiers (9 sur 11 dans Rust, 11 dans Python)

✓ Encodage morph : Mood, Tense, Aspect, Polarity dans FeatureVocabulary

✓ Architecture 4 couches (UD → MLP → R-GCN → Pearl) — structure saine

✓ Export DOT / sérialisation CIR JSON

✓ CLI (gcn-train, gcn-eval, gcn-discuss, gcn-export)
```

---

## Carte de migration vers la spécification

```
Phase A — Modèle ternaire (L1, L5)
  ├── gcn-ir : étendre CausalEdge avec sources[] et third?
  ├── ir_emitter.py : détecter JOINT/CONDITIONAL/MEDIATED
  ├── edge_norm.py : ne plus écraser le deuxième source
  └── layer1/features.py : ajouter has_two_sources, has_third_*, etc.

Phase B — Features manquantes (L2, L7, L8, L9, L10)
  ├── bridge.py : corriger has_advcl=False (déduire depuis node_type=condition)
  ├── layer1/features.py : ajouter PronType, Voice, features positionnelles
  └── layer1/representation.py : étendre UDRepresentation avec tiers T

Phase C — Types nœuds et relations (L6, L11)
  ├── gcn-ir/src/edge.rs : ajouter Analogy comme RelationType, Counterfactual
  ├── constants.py : ajouter concept, événement, contrainte aux NODE_TYPES
  └── Documenter etat_systemique, action, transition dans la spec

Phase D — NLU routing (L3, L4)
  ├── layer1/sentence_type.py : brancher depuis discuss.py et instructions.py
  ├── instructions.py execute() : ajouter paramètres tokens, markers
  └── discuss.py : charger SentenceProfile et router depuis le profil

Phase E — Verbalisation naturelle (L13)
  ├── verbalizer/ : implémenter templates déterministes par relation
  └── Séparer verbalisation de la sérialisation (ReferenceDecoder)

Phase F — Architecture (L14, L15)
  ├── discuss.py : séparer NLU routing du moteur core
  └── CIR schema : ajouter champ salience
```
