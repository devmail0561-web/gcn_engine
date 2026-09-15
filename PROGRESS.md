# Progression de l'implémentation — GCN-Core

Suivi détaillé de l'avancement par phase, composant et critère de vérification.

---

## Vue d'ensemble

```
Phase 1  ████████████████████  100%  gcn-ir + gcn-knowledge
Phase 2a ████████████████████  100%  gcn-frontend-fr
Phase 2b ████████████████████  100%  gcn-python couches ML
Phase 2c ████████████████████  100%  gcn-python évaluation
Phase 3  ████████████████████  100%  gcn-middleend
Phase 4  ████████████████████  100%  gcn-backend + gcn-cli
Phase 5  ████████████████████  100%  gcn-verbalizer
Phase 6  ████████████████████  100%  gcn-frontend-code
Phase 7  ████████████████████  100%  Pearl 2-3, R-GCN PyTorch, frontend anglais
Phase 8  ████████████████████  100%  Mise en production + publication
```

**Tests Rust : 137 / 137 passent** (`cargo test --workspace`)
**Tests Python : 112 / 112 passent** (`pytest gcn-python/tests/`)

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
| InferenceEngine | `gcn-knowledge/src/inference.rs` | ✅ |
| 11 taxonomies YAML (fr) | `gcn-references/taxonomies/` | ✅ |

**Vérification :** `cargo test -p gcn-ir -p gcn-knowledge` ✅

**Corrections appliquées (post-audit) :**
- `InferenceEngine` enrichi : `infer_node_type`, `infer_scope`, `score_confidence`, `enrich` — 23 tests sur données synthétiques (indépendants des YAML)

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

**Tests (21/21) :** condition, cause backward/forward, séquence, depuis-compositionnel, concession, motivation, verbe causal, scope universel, causalité implicite, négation sur nœud, négation sur arête, hypothétique, enable, input vide, + 5 tests post-audit (BUG-1, BUG-3/4, paper-013, C-1, C-4).

**Limites connues :**
- Parser symbolique : couverture limitée aux marqueurs du lexique YAML
- Résolution coréférentielle inter-phrases : non implémentée dans FrenchParser (outil de
  bootstrap uniquement). Dans le pipeline ML de production, les labels des nœuds sont les
  référents résolus manuellement dans `gcn-datasets/` — la coréférence n'intervient pas à
  l'inférence. Le slot `AmbiguousField::CorefTarget` dans `gcn-ir` est réservé pour une
  future extension sur texte brut entrant. **Non-bloquant en production.**
- Compositionnalité verbale (aspect) : partiellement implémentée

---

## Phase 2b — Couches ML Python ✅

**Objectif :** implémentations de référence NumPy des 3 couches CGNP + boucle d'entraînement SGD.

| Composant | Fichier | Statut |
|---|---|---|
| UDRepresentation | `gcn-python/layer1/representation.py` | ✅ |
| FeatureVocabulary + vectorize | `gcn-python/layer1/features.py` | ✅ |
| CausalEncoder Protocol | `gcn-python/layer2/interface.py` | ✅ |
| MLPEncoder référence NumPy | `gcn-python/layer2/reference.py` | ✅ |
| CausalGraph Protocol | `gcn-python/layer3/interface.py` | ✅ |
| RGCNLayer référence NumPy | `gcn-python/layer3/reference.py` | ✅ |
| CGNPipeline.forward() | `gcn-python/pipeline/cgnp.py` | ✅ |
| ir_emitter (→ JSON Rust) | `gcn-python/pipeline/ir_emitter.py` | ✅ |
| gcn-forward CLI | `gcn-python/pipeline/cli.py` | ✅ |
| backward() rétropropagation + SGD | `gcn-python/pipeline/cgnp.py` | ✅ |
| loss() cross-entropie NumPy | `gcn-python/pipeline/cgnp.py` | ✅ |
| GCNDataLoader + TrainingSample | `gcn-python/data/loader.py` | ✅ |
| gcn-train CLI (boucle SGD) | `gcn-python/training/train.py` | ✅ |
| gcn-bootstrap CLI (génération YAML) | `gcn-python/training/bootstrap.py` | ✅ |
| checkpoint save/load (.npz) | `gcn-python/training/checkpoint.py` | ✅ |
| gcn-forward --model-path | `gcn-python/pipeline/cli.py` | ✅ |
| backward_message_pass RGCNLayer | `gcn-python/layer3/reference.py` | ✅ |
| Protocol CausalGraph backward | `gcn-python/layer3/interface.py` | ✅ |

**Tests (9/9 + 11 post-audit) :** cross-entropie, gradient shape, gradient sum, backward R-GCN shape, backward updates weights, checkpoint roundtrip, dataloader batches, edge_map alignment, reps alignment. **89 / 89 passent au total.**

**Corrections audit appliquées :**
- `recorder.py` : docstring mise à jour avec la nouvelle signature `loss()`
- `layer2/reference.py` : `_backward_mlp` retourne `(grads, d_input)` — élimine la duplication dans `backward_node_dx`
- `layer3/reference.py` : `.copy()` sur les tableaux d'entrée au `message_pass` — évite les mutations externes
- `rules.rs` : normalisation NFC avant `strip_suffix` (unicode-normalization)
- `annotator.rs` : paramètre `_res` mort supprimé de `extract_object`
- `lib.rs` : test debug AST avec assertions réelles

**Corrections post-audit (bugs pipeline) :**
- `yaml_reader.py` : `_parse_edge` lisait `confidence`/`explicit`/`negated`/`marker_token` au niveau racine alors qu'ils sont imbriqués sous `attributes` dans le YAML — corrigé avec fallback
- `checkpoint.py` : `load_checkpoint` ne restaurait pas `FeatureVocabulary` — corrigé, vocab rechargé depuis `_vocab_json`
- `cgnp.py` : `backward_message_pass` recevait `d_enriched` de shape `(min(N,M), D)` au lieu de `(N, D)` quand gold labels < clauses spaCy → crash shape mismatch — corrigé par padding à la taille N
- `cgnp.py` : assemblage `flat_grads` sans garde-fou sur les bornes → potentiel IndexError avec encodeur custom — bornes ajoutées
- **Alignement entraînement** : `reps_from_sentence()` construit les `UDRepresentation` depuis les tokens YAML (alignement garanti features ↔ gold labels) ; `CGNPipeline.forward(reps, text)` est la seule entrée publique du forward — spaCy retiré du moteur
- **Misalignement reps ↔ gold labels** (bug 0.9.1) : `reps_from_sentence` filtrait les clauses à span vide sans retourner les indices valides — `logit[i]` était comparé au label de la clause `i-1`. Corrigé : `reps_from_sentence` retourne `(reps, valid_indices)` ; `train.py` indexe `gold_node_labels[valid_indices]` avant la loss
- **Indice 0 silencieux** (bug 0.9.3) : `node_type`/`relation` inconnus tombaient silencieusement à l'indice 0 — `ValueError` levée avec sentence ID et valeur fautive
- **Troncature silencieuse** (bug 0.9.3) : `loss()` absorbait les désalignements taille logits/gold avec `min()` — `ValueError` levée pour forcer un alignement explicite en amont
- **Supervision arêtes inverses perdue** (bug 0.9.5) : arêtes gold `(k+1, k)` jamais retrouvées par le lookup `(k, k+1)` du forward — zéro gradient d'arête pour ces samples. Corrigé par lookup bidirectionnel dans `_to_sample` (gap == 1 uniquement) et `train.py`
- **Champ mort `gold_edge_labels`** (0.9.5) : tableau insertion-order jamais aligné avec les logits, supprimé de `TrainingSample` ; fallback `train.py` → `gold_edge = None`
- **Warning arêtes longue distance** (0.9.5) : `_to_sample` émet un `UserWarning` pour toute arête gold avec `gap > 1` (aucune supervision possible avec le forward consécutif)
- **Mismatch R-GCN dimensionnel** (0.9.10) : `cgnp.py` lève `ValueError` au premier `forward()` si `d_out ≠ d_clause` — remplace le `UserWarning` qui laissait les poids R-GCN ne jamais apprendre
- **Double normalisation des gradients** (0.9.8) : `_cross_entropy` normalisait déjà par N ; `backward()` renormalisait à nouveau par `n`/`e`, produisant un gradient `1/N²` au lieu de `1/N`. Les deux renormalisations redondantes supprimées dans `cgnp.py`.
- **TaxonomyIndex retiré du pipeline ML** (C1) : `FeatureVocabulary` devient purement syntaxique (UPOS + DEP_REL + tense/aspect/mood + polarity + flags structurels). Les taxonomies restent des guides pour les annotateurs, jamais des dépendances runtime du moteur.
- **`backward_message_pass` retiré du Protocol `CausalGraph`** (C2) : levait `NotImplementedError` dans `RGCNLayerPT` ; la garde `hasattr` dans `cgnp.py:backward()` retourne `False` pour les impls PyTorch.
- **`d_out: int` ajouté au Protocol `CausalGraph`** (C5) : dimension de sortie vérifiable sans instanciation.

---

## Phase 2c — Évaluation ✅

| Composant | Fichier | Statut |
|---|---|---|
| node_accuracy, node_f1_per_class | `gcn-python/evaluation/metrics.py` | ✅ |
| edge_accuracy, edge_f1_per_class | `gcn-python/evaluation/metrics.py` | ✅ |
| causal_graph_similarity | `gcn-python/evaluation/metrics.py` | ✅ |
| TrainingRecorder | `gcn-python/evaluation/recorder.py` | ✅ |
| decoder_causal_fidelity | `gcn-python/evaluation/metrics.py` | ✅ |
| cross_modal_consistency | `gcn-python/evaluation/metrics.py` | ✅ |
| roundtrip_similarity | `gcn-python/evaluation/metrics.py` | ✅ |
| generation_bleu | `gcn-python/evaluation/metrics.py` | ✅ |

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

## Phase 5 — Décodeur ✅

**Objectif :** décodeur de l'architecture encodeur-décodeur GCN. Entraîné conjointement avec l'encodeur sur les mêmes paires `(texte, CausalIR)`. Évalué symétriquement.

| Composant | Fichier | Statut |
|---|---|---|
| Crate `gcn-verbalizer` (pont Rust) | `gcn-verbalizer/src/lib.rs` | ✅ |
| `ReferenceDecoder` (linéarisation structurelle, CLI) | `gcn-python/verbalizer/decoder.py` | ✅ |
| `TrainableDecoder` (NumPy, entraînable, remplaçable) | `gcn-python/verbalizer/trainable.py` | ✅ |
| `SurfaceVocabulary` | `gcn-python/verbalizer/trainable.py` | ✅ |
| Interface Protocol `VerbalizerDecoder` | `gcn-python/verbalizer/interface.py` | ✅ |
| CLI `gcn-verbalize` (Python, comme gcn-forward) | `gcn-python/verbalizer/cli.py` | ✅ |
| `VerbalizerDataLoader` + `VerbalizeSample` | `gcn-python/data/verbalize_loader.py` | ✅ |
| `CGNPipeline` étendu (decoder optionnel) | `gcn-python/pipeline/cgnp.py` | ✅ |
| Checkpoint decoder (save/load) | `gcn-python/training/checkpoint.py` | ✅ |
| `gcn-train --verbalize-dir` (entraînement conjoint) | `gcn-python/training/train.py` | ✅ |
| Schéma dataset verbalization | `gcn-datasets/schemas/gcn-verbalize.schema.yaml` | ✅ |
| Exemples cross-modal (fr+python même CausalIR) | `gcn-datasets/examples/verbalize_cross_modal.json` | ✅ |
| Métriques décodeur (fidelity, consistency, roundtrip, bleu) | `gcn-python/evaluation/metrics.py` | ✅ |

**Architecture d'entraînement conjoint :**
- `forward()` appelle `decoder.forward_decode(enriched_vecs)` après le R-GCN → `_cached_decode_logits`
- `loss(gold_surface=...)` ajoute la loss décodeur au total → `_cached_decode_gradient`
- `backward()` propage `d_mean` du décodeur vers `d_enriched` avant le R-GCN backward (gradient couplé)
- `gcn-train --verbalize-dir` : joint training via `source_text_map` + boucle standalone sur paires verbalize

**Critère de vérification :**
- `Verbalizer::decode(ir)` retourne une surface non vide pour tout CausalIR valide
- La sortie dépend de l'entraînement, aucun type de sortie présupposé par l'architecture
- `gcn-train --verbalize-dir gcn-datasets/examples/ --data-dir gcn-datasets/examples/ --epochs 2` — sans erreur, checkpoint contient `decoder_*`
- `roundtrip_similarity` ≥ 0.7 sur exemples gold du dataset
- `cross_modal_consistency` ≥ 0.7 entre surfaces fr et python du même CausalIR
- `cargo test --workspace` passe, `cargo clippy -- -D warnings` propre

**Tests (21/21) :** 13 tests `test_trainable_decoder.py` + 8 tests `test_verbalize_loader.py`

---

## Phase 6 — Frontend code ✅

**Objectif :** AST Python, Rust, JavaScript → `CausalIR`.

| Composant | Fichier | Statut |
|---|---|---|
| Bridge tree-sitter | `gcn-frontend-code/src/python.rs`, `rust.rs`, `js.rs` | ✅ |
| Mapping AST → NodeType/RelationType | `gcn-references/taxonomies/python/python_ast.yaml`, `rust/rust_ast.yaml`, `js/js_ast.yaml` | ✅ |
| LabelStrategy (YAML, pas de hardcoding) | `gcn-frontend-code/src/mapper.rs`, `resources.rs` | ✅ |
| Frontend Python | `gcn-frontend-code/src/python.rs` | ✅ |
| Frontend Rust | `gcn-frontend-code/src/rust.rs` | ✅ |
| Frontend JavaScript | `gcn-frontend-code/src/js.rs` | ✅ |

**Tests (26/26) :** Python base (7), Rust base (7), JS base (7), isomorphisme fr↔py (1), isomorphisme py↔rs↔js (1), arête séquentielle (1), except body (1), label typé (1).

**Critère de vérification :** `if x < y: reduce(z)` et *"Si x est inférieur à y, on réduit z"* produisent des CIR isomorphes (même nombre de nœuds, même arête `Condition`). ✅

**Corrections appliquées lors de l'audit :**
- Arête séquentielle `prev → cur` : était `data_edge(edge_rel)` (type du nœud courant), corrigée en `control_edge(Sequence)`
- `node_label` : tronquait au premier `:` (bug sur `def f(x: int)`) → remplacé par `LabelStrategy` YAML + `child_by_field_name`
- `walk_try` : le corps des `except_clause` était silencieusement ignoré → ajout de la récursion
- `has_negation_in_clause` : paramètre `_res` inutile supprimé (les flags sont déjà set dans le tagger)
- `generation_bleu` : brevity penalty utilisait `min(ref_lengths)` → standard BLEU (longueur la plus proche de l'hypothèse)

---

## Phase 7 — Extensions ✅

**Objectif :** Pearl niveaux 2-3, R-GCN optimisé, langues additionnelles (fr + en).

| Composant | Fichier | Statut |
|---|---|---|
| Pearl niveau 2 — `DO <nœud>` (intervention do-calculus) | `gcn-backend/src/pearl.rs` | ✅ |
| Pearl niveau 3 — `COUNTERFACTUAL <nœud>` (contrefactuels) | `gcn-backend/src/pearl.rs` | ✅ |
| GCN-QL étendu (`DO`, `COUNTERFACTUAL`) | `gcn-backend/src/query.rs` | ✅ |
| R-GCN PyTorch (remplace NumPy référence, GPU/MPS) | `gcn-python/layer3/pytorch_rgcn.py` | ✅ |
| Frontend anglais `EnglishParser` | `gcn-frontend-en/` | ✅ |
| Taxonomies anglaises (11 fichiers) | `gcn-references/taxonomies/en/` | ✅ |
| Isomorphisme fr↔en (même CIR pour condition fr/en) | `gcn-frontend-en/tests/integration_en.rs` | ✅ |

**Tests (26+8=34 backend, 16 en, 16 fr, 11 PyTorch, ...) :** tous passent.

**Sémantique Pearl :**
- **DO X** = coupe toutes les arêtes entrantes de X (causes naturelles de X sont court-circuitées), propage en avant depuis X. Retourne `severed_count` + effets.
- **COUNTERFACTUAL X** = "Que se serait-il passé si X n'avait pas eu lieu ?" Algorithme : compare les effets réels de X avec les nœuds atteignables depuis les vraies racines sans X. `unique_effects` = effets qui n'auraient PAS eu lieu sans X.

**Corrections appliquées lors de l'implémentation :**
- Algorithme contrefactuel initial : nœuds orphelins de X devenaient faussement des racines. Corrigé en partant des VRAIES racines originelles (nœuds sans aucune arête entrante dans le graphe complet).
- `lemmatize_verb` anglais : strip `'s'` avant strip `"es"` pour "causes" → "cause", "enables" → "enable".
- Annotateur anglais : marqueurs multi-mots avant verbes causaux (priorité plus haute) pour éviter que "prevent" vole le cas `in order to prevent`.

**Corrections post-audit (pearl.rs) :**
- `CausalLink` gagne `to_id: NodeId` ; `counterfactual()` fait le lookup par id au lieu de label — corrige la perte de cardinalité quand deux nœuds distincts partagent le même label (C-4 / BUG-2).
- Emitters FR et EN : propagation `temporal_index` max cumulé depuis la source — A→B→C produit `0, 1, 2` au lieu de `0, 1, 1` (C3).

**Décision : Wolof et Arabe exclus de la phase 7** (focalisé fr + en).

---

## Phase 8 — Mise en production ✅

**Objectif :** rendre le projet déployable et évaluable sur données réelles.

| Composant | Fichier | Statut |
|---|---|---|
| `Makefile` (install / install-dev / build-release / test / release) | `Makefile` | ✅ |
| `GCN_PYTHON_BIN` env var override | `gcn-cli/src/main.rs` | ✅ |
| `gcn-eval` CLI | `gcn-python/evaluation/eval_runner.py` | ✅ |
| Métriques dans `gcn-train` (node_accuracy, edge_macro_f1) | `gcn-python/training/train.py` | ✅ |
| `TrainingRecorder` câblé + export JSON | `gcn-python/training/train.py` | ✅ |
| `gcn-bootstrap` — 2 bugs corrigés | `gcn-python/training/bootstrap.py` | ✅ |
| Corpus réel — structure + workflow | `gcn-datasets/corpus/phrases_fr.txt` | 🔲 (annotation manuelle) |
| **README gcn-python** (vision, architecture, API complète) | `gcn-python/README.md` | ✅ |
| **README 9 crates Rust** (gcn-ir … gcn-cli) | `gcn-core/crates/*/README.md` | ✅ |
| **Publication PyPI** — `gcn-python 1.0.2` | pypi.org/project/gcn-python | ✅ |
| **Publication crates.io** — 9 crates v1.0.1 | crates.io/crates/gcn-ir (+ 8) | ✅ |
| Tag git `v1.0.0` | git | ✅ |

**Vérification :**
- `make install-dev` depuis la racine installe Rust + Python d'un coup
- `make test` lance `cargo test --workspace` + `pytest`
- `gcn-eval --data-dir gcn-datasets/examples/ --model-path model.npz` sort un rapport JSON à 5 métriques
- `pip install gcn-python` installe le package depuis PyPI
- `cargo add gcn-ir` ajoute la crate depuis crates.io

**Limites restantes (non-bloquantes) :**
- Corpus `gcn-datasets/corpus/` à remplir manuellement (40 phrases, 8 patterns causaux)
- `GCNDataLoader` charge les `SentenceRecord` bruts en RAM — pré-vectorisation via `FeatureVocabulary` prévue pour une version future

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
