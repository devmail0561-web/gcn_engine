# Progression de l'implémentation — GCN-Core

Suivi détaillé de l'avancement par phase, composant et critère de vérification.

---

## Vue d'ensemble

```
Phase 1  ████████████████████  100%  gcn-ir + gcn-knowledge
Phase 2a ████████████████████  100%  gcn-frontend-fr
Phase 2b ████████████████░░░░   80%  gcn-python couches ML
Phase 2c ████████████████████  100%  gcn-python évaluation
Phase 3  ████████████████████  100%  gcn-middleend
Phase 4  ████████████████████  100%  gcn-backend + gcn-cli
Phase 5  ░░░░░░░░░░░░░░░░░░░░    0%  gcn-verbalizer
Phase 6  ░░░░░░░░░░░░░░░░░░░░    0%  gcn-frontend-code
Phase 7  ░░░░░░░░░░░░░░░░░░░░    0%  Pearl 2-3, R-GCN, wolof/arabe
```

**Tests : 51 / 51 passent** (`cargo test --workspace`)

---

## Phase 1 — Infrastructure IR ✅

**Objectif :** définir le contrat central `CausalIR` et le chargement des connaissances.

| Composant | Fichier | Statut |
|---|---|---|
| Types CIR | `gcn-ir/src/ir.rs` | ✅ |
| NodeType (7 variantes) | `gcn-ir/src/node.rs` | ✅ |
| RelationType (11 variantes) | `gcn-ir/src/edge.rs` | ✅ |
| Scope (6 variantes) | `gcn-ir/src/scope.rs` | ✅ |
| Modifier (12 variantes) | `gcn-ir/src/modifier.rs` | ✅ |
| TemporalRef + TemporalGap | `gcn-ir/src/temporal.rs` | ✅ |
| NodeOrigin, CausalCycle, Ambiguity | `gcn-ir/src/ir.rs` | ✅ |
| CodeCausalType | `gcn-ir/src/code.rs` | ✅ |
| Chargeur YAML taxonomies | `gcn-knowledge/src/loader.rs` | ✅ |
| Lexicon (POS → classe GCN) | `gcn-knowledge/src/lexicon.rs` | ✅ |
| InferenceEngine | `gcn-knowledge/src/inference.rs` | ⚠️ Stub — à enrichir |
| 11 taxonomies YAML (fr) | `gcn-references/taxonomies/` | ✅ |

**Vérification :** `cargo test -p gcn-ir -p gcn-knowledge` ✅

---

## Phase 2a — Frontend français ✅

**Objectif :** parser symbolique français → `CausalIR`.

| Composant | Fichier | Statut |
|---|---|---|
| Tokenizer | `gcn-frontend-fr/src/tokenizer.rs` | ✅ |
| Tagger (POS basé sur lexicon) | `gcn-frontend-fr/src/tagger.rs` | ✅ |
| Règles causales | `gcn-frontend-fr/src/rules.rs` | ✅ |
| Annotateur | `gcn-frontend-fr/src/annotator.rs` | ✅ |
| Chargement LexicalResources | `gcn-frontend-fr/src/resources.rs` | ✅ |
| Émetteur CausalIR | `gcn-frontend-fr/src/emitter.rs` | ✅ |
| API publique `FrenchParser` | `gcn-frontend-fr/src/lib.rs` | ✅ |

**Tests (16/16) :** condition, cause backward/forward, séquence, depuis-compositionnel, concession, motivation, verbe causal, scope universel, causalité implicite, négation sur nœud, négation sur arête, hypothétique, enable, input vide.

**Limites connues :**
- Parser symbolique : couverture limitée aux marqueurs du lexique YAML
- Résolution coréférentielle inter-phrases : non implémentée
- Compositionnalité verbale (aspect) : partiellement implémentée

---

## Phase 2b — Couches ML Python 🔄

**Objectif :** implémentations de référence NumPy des 3 couches CGNP.

| Composant | Fichier | Statut |
|---|---|---|
| UDRepresentation | `gcn-python/layer1/representation.py` | ✅ |
| Extracteur spaCy | `gcn-python/layer1/extractor.py` | ✅ |
| FeatureVocabulary + vectorize | `gcn-python/layer1/features.py` | ✅ |
| CausalEncoder Protocol | `gcn-python/layer2/interface.py` | ✅ |
| MLPEncoder référence NumPy | `gcn-python/layer2/reference.py` | ✅ |
| CausalGraph Protocol | `gcn-python/layer3/interface.py` | ✅ |
| RGCNLayer référence NumPy | `gcn-python/layer3/reference.py` | ✅ |
| CGNPipeline.forward() | `gcn-python/pipeline/cgnp.py` | ✅ |
| ir_emitter (→ JSON Rust) | `gcn-python/pipeline/ir_emitter.py` | ✅ |
| gcn-forward CLI | `gcn-python/pipeline/cli.py` | ✅ |
| backward() rétropropagation | `gcn-python/pipeline/cgnp.py` | ⚠️ No-op — à implémenter par le DS |
| loss() cross-entropy | `gcn-python/pipeline/cgnp.py` | ⚠️ 0/1 référence — à remplacer |

**Manquant pour compléter 2b :**
- [ ] La `loss()` référence utilise une erreur 0/1 — doit être une cross-entropy pour un entraînement réel
- [ ] `backward()` est un no-op intentionnel : le data scientist implémente sa propre rétropropagation

---

## Phase 2c — Évaluation ✅

| Composant | Fichier | Statut |
|---|---|---|
| node_accuracy, node_f1_per_class | `gcn-python/evaluation/metrics.py` | ✅ |
| edge_accuracy, edge_f1_per_class | `gcn-python/evaluation/metrics.py` | ✅ |
| causal_graph_similarity | `gcn-python/evaluation/metrics.py` | ✅ |
| TrainingRecorder | `gcn-python/evaluation/recorder.py` | ✅ |

---

## Phase 3 — Middle-end ✅

**Objectif :** enrichir le `CausalIR` — cycles, contraintes, validation.

| Composant | Fichier | Statut |
|---|---|---|
| Construction DiGraph | `gcn-middleend/src/graph.rs` | ✅ |
| Détection SCC (Tarjan) | `gcn-middleend/src/cycle.rs` | ✅ |
| Classification cycles | `gcn-middleend/src/cycle.rs` | ✅ |
| Propagation TemporalGap | `gcn-middleend/src/propagate.rs` | ✅ |
| Vérif ordre temporel | `gcn-middleend/src/propagate.rs` | ✅ |
| Self-loop, orphelins | `gcn-middleend/src/validate.rs` | ✅ |
| Confidence basse, Condition | `gcn-middleend/src/validate.rs` | ✅ |
| API `process()` | `gcn-middleend/src/lib.rs` | ✅ |

**Tests (17/17) :** chaîne sans cycle, feedback positif/négatif, oscillation (concession), chemin du cycle, TemporalGap sur concession, violation ordre temporel, self-loop, orphelin, confidence basse, condition pendante, pipeline appended.

---

## Phase 4 — Backend + CLI ✅

**Objectif :** raisonnement Pearl niveau 1, GCN-QL, export, CLI complète.

| Composant | Fichier | Statut |
|---|---|---|
| WHY (BFS inverse) | `gcn-backend/src/pearl.rs` | ✅ |
| WHAT (BFS avant) | `gcn-backend/src/pearl.rs` | ✅ |
| CHAIN (chemin causal) | `gcn-backend/src/pearl.rs` | ✅ |
| Parseur GCN-QL | `gcn-backend/src/query.rs` | ✅ |
| Exécuteur GCN-QL | `gcn-backend/src/query.rs` | ✅ |
| Export JSON | `gcn-backend/src/export.rs` | ✅ |
| Export DOT (Graphviz) | `gcn-backend/src/export.rs` | ✅ |
| `gcn analyze` | `gcn-cli/src/main.rs` | ✅ |
| `gcn query` | `gcn-cli/src/main.rs` | ✅ |
| `gcn export` | `gcn-cli/src/main.rs` | ✅ |
| `gcn forward` (Rust↔Python) | `gcn-cli/src/main.rs` | ✅ |

**Tests (18/18) :** parse WHY/WHAT/CHAIN/CYCLES/GAPS, parse invalide, WHY cause directe, WHY non trouvé, WHAT effet, CHAIN chemin trouvé, CHAIN absent, CYCLES détecté, CYCLES vide, GAPS concession, GAPS propre, export JSON valide, export DOT, roundtrip JSON.

**Limites connues :**
- Pearl niveau 1 uniquement (association / observationnel)
- `gcn forward` requiert que `gcn-python` soit installé et `gcn-forward` dans le PATH

---

## Phase 5 — Verbalisateur ⬜

**Objectif :** graphe causal → texte naturel **et** code simultanément.

| Composant | Fichier | Statut |
|---|---|---|
| Crate `gcn-verbalizer` | — | ⬜ Non démarré |
| Verbalisation texte (fr) | — | ⬜ |
| Verbalisation code (Python) | — | ⬜ |
| Sortie simultanée texte + code | — | ⬜ |

**Critère de vérification :**
- `gcn-cli verbalize <ir.json> --lang fr` → phrase française cohérente avec le graphe
- `gcn-cli verbalize <ir.json> --lang python` → code Python implémentant la logique causale
- Le texte et le code produits depuis le même graphe encodent la même structure causale

---

## Phase 6 — Frontend code ⬜

**Objectif :** AST Python, Rust, JavaScript → `CausalIR`.

| Composant | Statut |
|---|---|
| Bridge tree-sitter | ⬜ |
| Mapping AST → NodeType/RelationType | ⬜ |
| Frontend Python | ⬜ |
| Frontend Rust | ⬜ |
| Frontend JavaScript | ⬜ |

**Critère de vérification :** `if x < y: reduce(z)` et *"Si x est inférieur à y, on réduit z"* produisent des CIR isomorphes.

---

## Phase 7 — Extensions ⬜

**Objectif :** Pearl niveaux 2-3, R-GCN optimisé, langues additionnelles.

| Composant | Statut |
|---|---|
| Pearl niveau 2 — do-calculus (intervention) | ⬜ |
| Pearl niveau 3 — contrefactuels | ⬜ |
| R-GCN PyTorch/JAX (remplace NumPy référence) | ⬜ |
| Frontend Wolof | ⬜ |
| Taxonomies Wolof | ⬜ |
| Frontend Arabe | ⬜ |

---

## Décisions architecturales actives

| Code | Décision | Impact |
|---|---|---|
| DA-1 | `CausalIR` est le contrat central — tout frontend produit un `CausalIR` | Interopérabilité garantie |
| DA-2 | Couches ML dans `gcn-python` (NumPy référence, DS substitue PyTorch/JAX) | Agnosticité framework |
| DA-3 | Graphe orienté **avec cycles** (petgraph DiGraph, pas DAG) | Boucles de rétroaction natives |
| DA-4 | Lacunes causales = données de première classe | Concession/Opposition = signal explicite |
| DA-5 | Compositionnalité verbale implémentée (aspect modifie le type) | `depuis` → Processus |
| DA-6 | DiGraph vit dans middle-end, pas dans `CausalIR` | CausalIR 100% sérialisable |
| DA-7 | `NodeOrigin` (Explicit/Inferred/Hypothetical) sur chaque nœud | Traçabilité complète |
| DA-8 | Causalité implicite détectée avec confidence 0.5, `explicit: false` | Signalement des inférences |
