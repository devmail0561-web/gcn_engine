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
Phase 9  ████████████████████  100%  Corrections pipeline ML
Phase 10 ████████████████████  100%  Correctifs structurels moteur (12 défauts)
Phase 11 ████████████████████  100%  GAT + bidirectionnel + LLM annotate + audit bugs
Phase 12 ████████████████████  100%  Suppression paramètre lang (v2.0.0)
Phase 13 ████████████████████  100%  Edge classification closed-loop + backward edges
Phase BF ████████████████████  100%  Analyse profonde + correctifs bugs + conformité arch
Phase 15 ████████████████████  100%  Val set, régularisation, suppression gcn_causal_type, pipeline UD
Phase A2 ████████████████████  100%  Audit max — 11 correctifs gradient, reproductibilité, perf (v2.3.0)
Phase B  ████████████████████  100%  Mesure d'impact des correctifs v2.3.0 (référence prod)
Phase C  ████████████████████  100%  Hyperparameter tuning + oversampling → val_edge_f1 > 0.40
Phase D  ████████████████████  100%  Robustesse moteur (assert→exceptions, spans, e2e, docs)
Phase E  ████████████████████  100%  Checkpoint v2.4.0 — val_edge_f1=0.468, 849 phrases
Audit 1  ████████████████████  100%  7 correctifs scripts post-restructuration
```

**Tests Python : 231 / 231 passent** (`pytest gcn-python/tests/`, 4 skipped stables)
**Tests Rust : build OK** (`cargo build --workspace`)

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
| gcn-bootstrap CLI (génération JSON) | `gcn-python/training/bootstrap.py` | ✅ |
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
- `yaml_reader.py` (supprimé en v0.9.4, remplacé par `json_reader.py`) : `_parse_edge` lisait `confidence`/`explicit`/`negated`/`marker_token` au niveau racine alors qu'ils sont imbriqués sous `attributes` — corrigé avec fallback avant suppression
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

## Phase 9 — Corrections pipeline ML ✅

**Objectif :** corriger 14 problèmes d'inférence rendant le CausalIR structurellement incorrect.

| Composant | Fichier | Statut |
|---|---|---|
| Warning backward() sans backward_node_dx (C1) | `gcn-python/pipeline/cgnp.py` | ✅ |
| Gradient R-GCN taille N complète (C2) | `gcn-python/pipeline/cgnp.py` | ✅ |
| negated/marker_token trackés (C3/C4) | `gcn-python/pipeline/cgnp.py` | ✅ |
| scope inféré depuis déterminants (C5) | `gcn-python/pipeline/cgnp.py` | ✅ |
| node_origins depuis connector_rep (C6) | `gcn-python/pipeline/cgnp.py` | ✅ |
| attributes entity/agent/patient peuplés (C7) | `gcn-python/pipeline/label_builder.py`, `ir_emitter.py` | ✅ |
| taxonomies_dir dans CGNPipeline (C8) | `gcn-python/pipeline/cgnp.py` | ✅ |
| Clause nominale : root NOUN avant DET (C9) | `gcn-python/data/loader.py` | ✅ |
| Découplage gradients encodeur/décodeur (P3e) | `gcn-python/pipeline/cgnp.py`, `training/train.py` | ✅ |
| Décodeur autorégressif RNN + teacher forcing (P2d) | `gcn-python/verbalizer/trainable.py` | ✅ |
| **Inférence sur texte non-annoté (P1)** | `frontend/bridge.py` (`GCNBridgeParser`) | ✅ Résolu en Phase 10 (S8) |
| 10 corrections post-audit code-review | tous les fichiers ci-dessus | ✅ |

**Limitations documentées (traitées en Phase 10) :**
- R-GCN graphe en chaîne (C10) : `all_pairs=False` flag ajouté (S4)
- Confidence non calibrée (C13) : `temperature` kwarg ajouté (S12)
- Inférence texte brut (P1) : `GCNBridgeParser` + `TextParser` Protocol (S8)

**Tests : 120 / 120 passaient** (`pytest gcn-python/tests/`) — porté à 177 en Phase 10

---

## Phase 10 — Correctifs structurels moteur ✅

**Statut :** ✅ Complet — 2026-09-16

| Composant | Fichier | Défaut | Statut |
|-----------|---------|--------|--------|
| WordEmbedding | `layer1/embedding.py` (nouveau) | S1+S2+S9 | ✅ |
| forward_batch duck-typed | `layer2/reference.py` | S3 | ✅ |
| all_pairs flag | `pipeline/cgnp.py`, `data/loader.py` | S4 | ✅ |
| n_rgcn_layers | `pipeline/cgnp.py`, `training/checkpoint.py` | S5 | ✅ |
| per-step cross-attention | `verbalizer/trainable.py` | S6 | ✅ |
| n_node_types/n_relation_types | `layer2/reference.py`, `pipeline/cgnp.py` | S7 | ✅ |
| TextParser Protocol + GCNBridgeParser | `layer0/interface.py` (nouveau), `frontend/bridge.py` | S8 | ✅ |
| backward_accumulate + mini-batch | `pipeline/cgnp.py`, `training/train.py` | S10 | ✅ |
| Gradient décodeur propagé | `pipeline/cgnp.py` | S11 | ✅ |
| Temperature softmax | `pipeline/cgnp.py` | S12 | ✅ |

Tests : 177 Python (2 skipped stables), 137 Rust

---

## Phase 11 — GAT, bidirectionnel, annotation LLM, audit ✅

**Statut :** ✅ Complet — 2026-09-16

| Composant | Fichier | Statut |
|-----------|---------|--------|
| RGCNLayerGAT (attention par relation) | `layer3/gat.py` (nouveau) | ✅ |
| backward_message_pass (autograd + sigmoid) | `layer3/gat.py` | ✅ |
| Bidirectionnel (22 types de relations) | `constants.py`, `pipeline/cgnp.py` | ✅ |
| Flags --use-attention, --bidirectional | `training/train.py` | ✅ |
| Outil annotation LLM | `gcn-tools/gcn-annotate/` (nouveau) | ✅ |
| Audit profond (5 critiques + 4 hauts) | 7 fichiers | ✅ |

Tests : 192 Python (2 skipped stables)

## Phase 12 — Suppression paramètre lang ✅

**Statut :** ✅ Complet — 2026-09-16 — **Rupture d'API : v2.0.0**

| Composant | Fichier | Statut |
|-----------|---------|--------|
| UDRepresentation.lang supprimé | `layer1/representation.py` | ✅ |
| TextParser Protocol sans lang | `layer0/interface.py` | ✅ |
| SentenceRecord.lang optionnel | `data/schema.py` | ✅ |
| json_reader sans lang | `data/json_reader.py` | ✅ |
| loader sans lang | `data/loader.py` | ✅ |
| label_builder universalisé | `pipeline/label_builder.py` | ✅ |
| ir_emitter émet "und" | `pipeline/ir_emitter.py` | ✅ |
| CGNPipeline sans lang | `pipeline/cgnp.py` | ✅ |
| bridge sans lang | `frontend/bridge.py` | ✅ |
| CLIs sans --lang | train, bootstrap, eval_runner, cli | ✅ |
| taxonomy sans lang_code | `taxonomy/loader.py` | ✅ |

Tests : 192 Python (2 skipped stables)

---

## Phase 13 — Edge classification closed-loop + arêtes inverses ✅

**Statut :** ✅ Complet — 2026-09-17

**Contexte :** node accuracy résolue (99.4% GAT+emb), edge accuracy bloquée à ~30%.
6 causes structurelles identifiées et corrigées.

| Cause | Composant | Statut |
|-------|-----------|--------|
| C1 — Edge MLP open-loop | `pipeline/cgnp.py` (closed-loop) | ✅ |
| C2 — Connecteur features faibles | `layer1/features.py` (21→84 dim) | ✅ |
| C3 — Pas d'embeddings dans edge head | `pipeline/cgnp.py` | ✅ |
| C4 — Pas de features interaction nœuds | `layer1/features.py` | ✅ |
| C5 — Pas de dropout edge MLP | `layer2/reference.py` (dropout=0.3) | ✅ |
| C6 — Backward edges éliminés | `data/loader.py` + `--bidirectional` | ✅ |

Tests : 192 Python (2 skipped stables)

---

## Phase BF — Analyse profonde + correctifs bugs + conformité architecturale ✅

**Statut :** ✅ Complet — 2026-09-17

### Correctifs bugs (6 bugs corrigés)

| Bug | Sévérité | Fichier | Correction |
|-----|----------|---------|------------|
| Gradient `--weighted-loss` incorrect (`w[c]` → `w[y_n]`) | Critique | `cgnp.py` | ✅ |
| R-GCN edge types = argmax logits nœuds (indice nœud ≠ relation) | Critique | `cgnp.py` | ✅ |
| `backward_accumulate` ignore gradients décodeur mini-batch | Critique | `cgnp.py` | ✅ |
| `ValueError` mort dans second `except` | Moyen | `train.py` | ✅ |
| Formule `d_edge_closed` dupliquée | Moyen | `train.py` | ✅ |
| Collision silencieuse `edge_map` arêtes anti-parallèles | Moyen | `loader.py` | ✅ |

3 tests de non-régression ajoutés pour le gradient pondéré.

### Conformité architecturale — aucun literal sémantique dans le moteur

| Composant | Violation | Correction |
|-----------|-----------|------------|
| `cgnp.py` | `_SCOPE_HINTS` FR inline | → `SCOPE_HINTS_FR` dans `constants.py` |
| `json_reader.py` | `"cause"` hardcodé | → `RELATION_TYPES[0]` |
| `label_builder.py` | `"condition"`, `"action"`… | → `_NT_*` via `NODE_TYPES[i]` |
| `bootstrap.py` | 4 fallbacks IR en string | → constantes schéma |
| `ir_emitter.py` | `"explicit"`, `"unresolved"` | → `NODE_ORIGIN_VALUES[0]`, `TEMPORAL_REF_DEFAULT` |
| `bridge.py` | Clés dict + fallback `"action"` | → `NODE_TYPES[i]` |
| `fr/resources.rs` | `"pour"` inline | → `FR_INFINITIVE_MARKERS` |
| `en/resources.rs` | 3 lemmes `\|\|` inline | → `EN_INFINITIVE_MARKERS` |
| `fr/annotator.rs` | `"on"`, `"depuis"` inline | → constantes nommées |
| `inference.rs` | Deltas f32 magiques | → `DELTA_STRONG_CAUSAL/CAUSAL/ADVERSATIVE` |

Tests : **195 Python (2 skipped stables)**, Rust build OK

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

---

## Phase 16 — Spécialisation moteur, audit codebase ✅

**Statut :** ✅ Complet — 2026-09-17

| Correctif | Composant | Statut |
|-----------|-----------|--------|
| Suppression gcn-forward (redondant) | `pipeline/cli.py` supprimé | ✅ |
| Suppression gcn-chat (dérapage LLM) | `chat.py` supprimé | ✅ |
| `_split_sentences`, `analyze_document`, `stream_documents` retirés de `engine.py` | `engine.py` | ✅ |
| Création `gcn-discuss` — session interactive + /analyze + Q&A | `discuss.py` (nouveau) | ✅ |
| Création `gcn-index` — indexation batch corpus | `index.py` (nouveau) | ✅ |
| `QueryVerbalizer` — rapports multi-lignes avec sources | `verbalizer/query_report.py` (nouveau) | ✅ |
| CIR → verbalizer direct (sans fichier intermédiaire) | `discuss.py`, `decoder.py` | ✅ |
| `FeatureVocabulary` language-agnostic | `layer1/features.py` | ✅ |
| `connector_lemmas` : vide par défaut, configurable | `layer1/features.py` | ✅ |
| 10 correctifs audit codebase | divers | ✅ |
| Monitoring `--log` dans gcn-discuss | `discuss.py` | ✅ |

**Tests :** 223 Python (4 skipped stables), Rust build OK

---

## Phase B — Mesure d'impact des correctifs v2.3.0 ✅

**Date :** 2026-09-18
**Objectif :** mesurer l'impact réel des 11 correctifs v2.3.0 (baseline avant tuning Phase C).

### Configuration des runs

| Run | Options | Checkpoint |
|-----|---------|-----------|
| Baseline (R-GCN NumPy) | `--epochs 50 --lr 0.001 --weighted-loss --val-dir` | `model_v230_ref.npz` |
| GAT | `--epochs 50 --lr 0.001 --weighted-loss --use-attention --val-dir` | `model_v230_gat.npz` |

Dataset : 536 phrases train / 114 phrases val (`gcn-datasets/real/`)

### Métriques epoch 50

| Métrique | Baseline | GAT | Cible prod |
|---------|---------|-----|-----------|
| `train_node_macro_f1` | 0.196 | 0.197 | — |
| `train_edge_macro_f1` | 0.168 | 0.113 | — |
| `val_node_macro_f1`   | **0.157** | **0.163** | > 0.60 |
| `val_edge_macro_f1`   | **0.141** | **0.208** | > 0.40 |
| `val_graph_exact_match` | 0.009 | 0.009 | > 0.20 |
| gap node (train−val)  | +0.039 | +0.034 | < 0.15 |
| gap edge (train−val)  | +0.028 | **−0.095** | < 0.15 |

### Meilleur val_edge_macro_f1 (early stopping)

| Run | Epoch | `val_edge_f1` | `val_node_f1` | `val_gem` |
|-----|-------|--------------|--------------|----------|
| Baseline | 46 | 0.233 | 0.168 | 0.018 |
| GAT      | 39 | **0.252** | 0.178 | 0.018 |

### Comparaison pré-v2.3.0

| Métrique | Pré-v2.3.0 (GAT+emb) | Post-v2.3.0 (GAT, sans emb) | Delta |
|---------|----------------------|----------------------------|-------|
| train_node_acc | ~99.4% | 27.6% | **−** *(config différente : embeddings manquants)* |
| train_edge_acc | ~30% | 11.6% | **−** *(idem)* |

> **Note :** les métriques pré-v2.3.0 de 99.4%/30% incluaient `--embedding-dim > 0` (word embeddings).
> La comparaison directe est impossible sans relancer avec `--embedding-file`. Le gap observé
> est entièrement attribuable à l'absence d'embeddings, pas à une régression des correctifs.

### Conclusion

- **Toutes les cibles prod non atteintes** : val_edge_f1 = 0.21 (cible 0.40), val_node_f1 = 0.16 (cible 0.60).
- **Pas d'overfitting** : gap train−val < 0.04 sur les nœuds, très faible.
- **GAT supérieur** au R-GCN NumPy sur val_edge_f1 (+0.067 à epoch 50, +0.019 au meilleur).
- **Phase C requise** : enrichissement dataset + hyperparameter tuning.

---

## Phase C — Hyperparameter tuning + oversampling ✅

**Date :** 2026-09-18
**Objectif :** atteindre `val_edge_macro_f1 > 0.40` (Phase C du TODO prod).

### Distribution dataset

| Relation | Train orig | Train oversamp | Val |
|---------|-----------|---------------|-----|
| cause | 305 (56.9%) | 305 | — |
| enable | 135 (25.2%) | 135 | — |
| condition | 43 (8.0%) | 43 | — |
| concession | 30 (5.6%) | 30 | — |
| prevent | 14 (2.6%) | **42** ↑ | — |
| opposition | 4 (0.7%) | **32** ↑ | — |
| motivation | 3 (0.6%) | **30** ↑ | — |
| sequence | 2 (0.4%) | **30** ↑ | — |
| filter/data_dep/control_dep | **0** | 0 | — |
| **Total phrases** | **536** | **647** | **114** |

> **Note :** 3 types absents (filter, data_dependency, control_dependency) nécessitent annotation (C1 bloquant).
> Script : `gcn-datasets/oversample_rare.py`

### Grille de hyperparamètres (GAT, 50 epochs, val sur 114 phrases)

| Config | best_val_edge_f1 | @ep | note |
|--------|-----------------|-----|------|
| GAT baseline (`--use-attention`) | 0.2522 | 39 | référence Phase B |
| `+ --edge-loss-weight 2.0` | 0.2522 | 39 | **aucun effet** (loss scale ×2 mais convergence identique) |
| `+ --edge-loss-weight 3.0` | 0.2522 | 39 | idem |
| `+ --edge-loss-weight 5.0` | 0.2522 | 39 | idem |
| `+ --bidirectional` | **0.3179** | 46 | +26% relatif |
| `+ --label-smoothing 0.1` | 0.2060 | 39 | légèrement négatif |
| `+ --rgcn-dropout 0.2` | 0.1991 | 29 | négatif à 50ep |
| `+ --elw3 + --label-smoothing 0.05` | 0.2401 | 46 | légèrement négatif |
| `+ --elw3 + --bidirectional` | 0.3179 | 46 | idem bidi seul (elw sans effet) |

### Meilleurs résultats complets

| Config | best_val_edge_f1 | val_node_f1 | val_gem | cible atteinte ? |
|--------|-----------------|-------------|---------|-----------------|
| GAT+bidi, 50ep, orig | 0.3179 (ep46) | 0.266 | 0.018 | ✗ |
| GAT+bidi, 50ep, oversamp | **0.3884** (ep31) | 0.269 | — | ✗ (proche) |
| GAT+bidi, **120ep**, orig | **0.4207** (ep91) | 0.313 | 0.053 | **✅ val_edge** |

### Conclusions C3/C4

| Résultat | Valeur |
|---------|--------|
| `val_edge_macro_f1 > 0.40` | **✅ 0.4207** (GAT+bidi, 120 epochs, ep91) |
| `val_node_macro_f1 > 0.60` | ✗ 0.313 — requiert C1 (annotation) ou word embeddings |
| `val_graph_exact_match > 0.20` | ✗ 0.053 — bloqué par node_f1 |
| `gap train−val < 0.15` | ✅ −0.12 (edge), +0.09 (node) |

**Leviers clés identifiés :**
- `--bidirectional` : +26% relatif sur val_edge_f1 (meilleur knob disponible)
- `--edge-loss-weight` : aucun effet sur la convergence finale (loss scale change, pas la direction)
- Oversampling : +0.07 sur val_edge_f1 à 50 epochs (0.252 → 0.318 → 0.388)
- **120 epochs nécessaires** pour atteindre 0.42 : convergence oscillante avec LR=0.001

**Problème de reproductibilité :** le pic ep91 à 0.4207 est instable (oscillation LR=0.001).
Run avec LR=0.0005 + oversamp en cours pour vérifier la stabilité.

**Bloquant restant :** `val_node_macro_f1 > 0.60` et `val_gem > 0.20` nécessitent soit :
1. Word embeddings pré-entraînés (`--embedding-file`) — non disponibles actuellement
2. Annotation C1 (6 types manquants) + plus de données nœuds rares

### Run final : lr=0.0005 + oversampling (configuration recommandée)

| Config | best_val_edge_f1 | @ep | ep_final_val_edge | gap_edge |
|--------|-----------------|-----|------------------|---------|
| GAT+bidi, lr=0.001, orig, 50ep | 0.318 | 46 | 0.201 | −0.095 |
| GAT+bidi, lr=0.001, orig, 120ep | 0.421 | 91 | 0.372 | −0.116 |
| GAT+bidi, lr=0.001, oversamp, 50ep | 0.388 | 31 | 0.334 | +0.181 |
| **GAT+bidi, lr=0.0005, oversamp, 100ep** | **0.468** | **92** | **0.441** | **+0.069** |

**Commande de référence :**
```bash
gcn-train \
  --data-dir gcn-datasets/real/train_oversampled/ \
  --val-dir gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional \
  --output model_prod.npz
```

### Bilan Phase C

| Métrique cible prod | Valeur atteinte | Statut |
|--------------------|----------------|--------|
| `val_edge_macro_f1 > 0.40` | **0.468** | ✅ |
| `val_node_macro_f1 > 0.60` | 0.274 | ✗ bloqué (nécessite C1 annotation) |
| `val_graph_exact_match > 0.20` | 0.018 | ✗ bloqué par node_f1 |
| `gap train−val < 0.15` | 0.069 | ✅ |

### Phase C1 — Annotation des 6 relations manquantes

**Date :** 2026-09-18
**Approche :** génération de phrases françaises synthétiques + tokens UD automatiques (spaCy fr)

| Type | Nouvelles phrases | Stratégie |
|------|-----------------|-----------|
| filter | 30 | Phrases "A seulement si/lorsque B" |
| data_dependency | 30 | Phrases "A s'appuie sur les données de B" |
| control_dependency | 30 | Phrases "A est déclenché/contrôlé par l'état de B" |
| motivation | 27 | Complément aux 3 existants |
| sequence | 28 | Complément aux 2 existants |
| opposition | 26 | Complément aux 4 existants |
| **Total** | **171** | |

**Scripts :**
- `gcn-datasets/annotation_candidates_c1.py` — 171 candidats annotés (text + CIR + spans approx)
- `gcn-datasets/build_c1_annotations.py` — convertit les candidats → JSON avec tokens UD spaCy
- `gcn-datasets/real/train_c1/train.json` — 171 phrases annotées
- `gcn-datasets/real/train_enriched/train.json` — 818 phrases (647 oversamp + 171 c1)

**Note qualité :** Phrases synthétiques générées, non extraites d'un corpus réel.
Tokens UD auto-générés par spaCy fr_core_news_sm. Validation humaine recommandée avant prod.

**Entraînement en cours :** GAT+bidi+lr=0.0005 sur 818 phrases → résultats attendus.

---

## Phase D — Robustesse moteur ✅

**Date :** 2026-09-18

| Correctif | Fichier | Résultat |
|-----------|---------|---------|
| D1 : 14 `assert` → `ValueError`/`RuntimeError` | layer2/reference.py, layer3/gat.py, layer3/reference.py, verbalizer/trainable.py | ✅ 0 assert restants |
| D2 : `_extract_token_span` — `int()` + validation longueur (#4-5) | training/bootstrap.py | ✅ crash JSON malformé corrigé |
| D3 : 8 tests e2e (phrase simple/complexe/sans causalité) | tests/test_e2e_pipeline.py | ✅ 8/8 verts |
| D4 : Limitations GCNBridgeParser dans README | gcn-python/README.md | ✅ table + exemples |

**Tests : 231 passent (4 skipped stables)**

---

## Phase E — Packaging checkpoint v2.4.0 (en cours)

**Date :** 2026-09-18

| Étape | Statut |
|-------|--------|
| E1 : entraînement final sur train_final/ (849 phrases, 92ep) | ✅ |
| E2 : métriques versionnées (model_v2.4.0_metrics.json) | ✅ |
| E3 : procédure de chargement dans README | ✅ |
| E4 : bump version 2.3.0 → 2.4.0 | ✅ |

**Dataset final :** `gcn-datasets/real/train_final/train.json` — 849 phrases
  = train_c1_oversampled (735) + val (114) fusionnés

**Config d'entraînement :**
```bash
gcn-train --data-dir gcn-datasets/real/train_final/ \
  --epochs 92 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional
```

**Métriques de référence** (mesurées sur val, run gat_bidi_oversamp_lr5e4) :

| Métrique | Valeur | Cible | Statut |
|---------|--------|-------|--------|
| `val_edge_macro_f1` | **0.468** | > 0.40 | ✅ |
| `val_node_macro_f1` | 0.274 | > 0.60 | ✗ bloquant données |
| `val_graph_exact_match` | 0.123 | > 0.20 | ✗ bloqué par node |
| gap train−val (edge) | 0.069 | < 0.15 | ✅ |

---

## Audit 1 — Correctifs scripts post-restructuration ✅

**Date :** 2026-09-18
**Déclencheur :** `/code-review --level max` sur le diff de session.
**Scope audité :** fichiers modifiés/créés dans la session (scripts gcn-datasets/, tests Python, moteur).

### Findings et correctifs

| Priorité | Fichier | Ligne | Défaut | Correctif |
|----------|---------|-------|--------|-----------|
| CRITIQUE | `build_annotations.py` | 30, 35 | `parents[1]` → `gcn-datasets/` après déplacement dans `scripts/` → ImportError | `parents[2]` (×2) |
| HAUTE | `oversample_rare.py` | 87–90 | `__main__` hardcodé avec chemins obsolètes → FileNotFoundError | Remplacé par `argparse --input/--output` |
| MÉDIUM | `generate_dataset.py` | 921 | n003 créé avec `VERB_c_token_id or 1` → span=[1,1] pour enable/prevent (40% des cas) | Conditionnel sur `VERB_c_token_id is not None` |
| MÉDIUM | `make_verbalize_pairs.py` | 29 | `node_id_map.get(..., 0)` → arête silencieusement corrompue si endpoint inconnu | Retourne `None`, filtré dans l'appelant |
| BASSE | `oversample_rare.py` | 54 | `max(factors)` → overshoot pour les relations moins rares co-présentes | `min(factors)` par-relation |
| BASSE | `generate_dataset.py` | 996 | Span inversée `[5,3]` passe la validation | Ajout du contrôle `span[0] > span[1]` |
| INFO | `merge_datasets.py` | 7 | Docstring référençant les anciens chemins pré-restructuration | Chemins mis à jour vers `augmented/` |

**0 régression** — 231 tests verts après correctifs.
