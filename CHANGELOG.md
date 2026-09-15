# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [Unreleased]

---

## [0.9.2] — 2026-09-15

### Corrigé
- **Misalignement gold edges ↔ arêtes prédites** (`data/loader.py`, `pipeline/cgnp.py`, `training/train.py`) — les arêtes YAML sont des triplets sémantiques arbitraires (source/target quelconques) alors que les arêtes prédites sont strictement des paires consécutives `i → i+1`. Le `min()` dans `loss()` prenait le Kième label YAML pour la Kième paire, sans vérifier la correspondance source/target ; pour les phrases à ≥ 3 nœuds avec des arêtes non-séquentielles, les gradients d'arête étaient faux.
  - `TrainingSample` gagne `edge_map: dict[(src_idx, tgt_idx), rel_idx]` construit depuis les `node_id` YAML résolus en indices de clauses
  - `CGNPipeline.filter_edge_cache(valid_idxs)` filtre les caches MLP d'arêtes (`_cached_edge_vecs`, `_cached_edge_logits`, `_cached_edge_snapshots`) — les caches R-GCN ne sont pas filtrés (le message-passing forward a utilisé toutes les arêtes)
  - `train.py` construit `gold_edge` aligné sur les paires consécutives via `edge_map`, masque les paires sans label YAML, filtre les logits et les caches en conséquence

### Tests
- `test_edge_map_alignment` — vérifie que `edge_map` résout correctement une arête non-consécutive n001→n003 vers `(0, 2)` et que les paires `(0,1)` et `(1,2)` restent absentes

---

## [0.9.1] — 2026-09-15

### Corrigé
- **Misalignement reps ↔ gold labels** (`data/loader.py`, `training/train.py`) — `reps_from_sentence` filtre silencieusement les clauses à span vide, produisant moins de reps que de gold labels ; le décalage d'index corrompait la loss (logit[i] comparé au label de la clause i-1). `reps_from_sentence` retourne maintenant `(reps, valid_indices)` et `train.py` indexe `gold_node_labels[valid_indices]` avant l'appel à `loss()`.

### Tests
- `test_reps_from_sentence_alignment` — vérifie qu'une clause à span vide en position 0 ne décale pas les labels des clauses suivantes

---

## [0.9.0] — 2026-09-15

### Ajouté
- **`InferenceEngine`** — implémentation complète dans `gcn-knowledge/src/inference.rs` (remplace le stub)
  - `InferenceNote` enum : `NodeTypeResolved`, `ConfidenceAdjusted`, `CausalGapSignaled`
  - `InferenceEngine::new(lexicon)` + `from_dir(path)`
  - `infer_node_type(lemma, pos)` : lookup taxonomie → `NodeType` (verbes + noms ; `auxiliaire` → `None`)
  - `infer_scope(lemma, pos)` : lookup DET/PRON → `Scope` (priorité `entry.scope`, fallback classe)
  - `score_confidence(from, to, rel, explicit)` : score structurel `[0.1, 1.0]` basé sur compatibilité des types causaux
  - `enrich(&mut CausalIR)` : 3 passes — résolution types nœuds inférés, recalibrage confiance arêtes implicites, signalement lacunes causales
  - `Lexicon::from_taxonomies_for_test` sous `#[cfg(test)]` pour tests sans YAML
  - `InferenceNote` ajouté au re-export public de `gcn-knowledge`
  - 23 tests sur données synthétiques — indépendants des fichiers YAML
- **Bypass spaCy à l'entraînement** — `data/loader.py` : `reps_from_sentence(SentenceRecord)` construit une `UDRepresentation` par nœud CIR directement depuis les tokens annotés du YAML, garantissant l'alignement exact features ↔ gold labels
  - `CGNPipeline.forward_from_reps(reps, text)` expose la passe avant sans extraction spaCy
  - `CGNPipeline._forward_from_reps` factorise le corps commun
  - `train.py` utilise `reps_from_sentence` quand les tokens sont disponibles, fallback spaCy sinon

### Corrigé
- `data/yaml_reader.py` : `_parse_edge` lisait `confidence`, `explicit`, `negated`, `marker_token` au niveau racine — ces champs sont imbriqués sous `attributes` dans le schéma GCN-NL, corrigé avec fallback racine pour compatibilité
- `training/checkpoint.py` : `load_checkpoint` ne restaurait pas `FeatureVocabulary` depuis le `_vocab_json` sauvegardé — le vocab est maintenant rechargé, évitant une inférence incorrecte si les taxonomies changent
- `pipeline/cgnp.py` : `backward_message_pass` recevait `d_enriched` de shape `(min(N,M), D_out)` au lieu de `(N, D_out)` quand le nombre de gold labels diffère du nombre de clauses spaCy — padding à `N` ajouté avant le backward R-GCN
- `pipeline/cgnp.py` : assemblage `flat_grads` sans garde-fou sur les indices — ajout de bornes `if i < len(flat_grads)` pour robustesse avec encodeurs custom non-3+3 couches

---

## [0.8.0] — 2026-09-14

### Ajouté
- **Boucle d'entraînement Phase 2b** — `gcn-python/training/`
  - `CGNPipeline.loss(node_logits, edge_logits, gold_node, gold_edge)` : cross-entropie NumPy réelle sur nœuds (7 classes) et arêtes (11 classes) — retourne `(float, d_node, d_edge)`
  - `CGNPipeline.backward(d_node, d_edge, lr)` : SGD complet avec sauvegarde de snapshots de cache MLP par nœud/arête au `forward()` — élimine tout re-run forward au `backward()` et la fragilité du cache partagé
  - `RGCNLayer.backward_message_pass(d_output)` : rétropropagation R-GCN NumPy, retourne `(d_input, [dW_r, dW_0])`
  - `CausalGraph` Protocol étendu : `backward_message_pass(d_output) -> tuple[ndarray, list[ndarray]]`
  - `MLPEncoder.backward_node_dx(d_logits)` : gradient vers l'entrée pour le R-GCN ; `snapshot_node_cache()` / `restore_node_cache()` pour backward déterministe
  - `GCNDataLoader` (`data/loader.py`) : itération sur datasets YAML annotés → `TrainingSample(sentence, gold_node_labels, gold_edge_labels)`
  - `gcn-train` CLI : boucle SGD multi-epoch, export CSV optionnel, vérification décroissance loss en fin d'entraînement
  - `gcn-bootstrap` CLI : génération automatique de données YAML depuis `gcn analyze` (Rust subprocess)
  - `training/checkpoint.py` : `save_checkpoint(pipeline, path)` / `load_checkpoint(pipeline, path)` en format `.npz` NumPy
  - `gcn-forward --model-path` : chargement d'un checkpoint entraîné (avertissement stderr si absent)
  - `RGCNLayerPT.backward_message_pass()` : stub `NotImplementedError` pour satisfaire le Protocol (`autograd` PyTorch à la place)
  - 9 tests dans `tests/test_training.py` (7 passent, 2 skippés sans modèle spaCy fr)

### Corrigé (audit)
- `recorder.py` : docstring mettait à jour l'ancienne API `loss(pred, gold)` → corrigée avec la nouvelle signature
- `layer2/reference.py` : `backward_node_dx` dupliquait la boucle de `_backward_mlp` — `_backward_mlp` retourne maintenant `(grads, d_input)`, `backward_node_dx` délègue
- `layer3/reference.py` : `message_pass` stockait des références directes aux tableaux d'entrée → `.copy()` sur les 3 tableaux
- `rules.rs` : ajout `unicode-normalization` (dépendance), normalisation NFC avant `strip_suffix`/`ends_with` — corrige le cas des tokens NFD français
- `annotator.rs` : paramètre `_res: &LexicalResources` jamais utilisé supprimé de `extract_object` et de son callsite
- `lib.rs` : test `debug_ast_if` (sans assertion, toujours vert) remplacé par `python_parser_produces_expected_ast_nodes` avec assertions réelles
- `python.rs` : commentaire expliquant que le dispatch structurel AST est topologique (pas des données lexicales)
- `metrics.py` : commentaire documentant le BLEU tronqué comme comportement intentionnel pour les courtes hypothèses NLP

---

## [0.7.0] — 2026-09-14

### Ajouté
- **Pearl niveau 2 — `DO <nœud>`** (do-calculus, intervention) dans `gcn-backend`
  - `pearl::intervene()` : coupe toutes les arêtes entrantes du nœud cible (ses causes naturelles sont court-circuitées), propage les effets en aval
  - `QueryResult::Intervention` : `target`, `severed_count`, `severed` (arêtes coupées), `effects`
  - Requête GCN-QL : `gcn query "DO <nœud>"`
- **Pearl niveau 3 — `COUNTERFACTUAL <nœud>`** dans `gcn-backend`
  - `pearl::counterfactual()` : "que se serait-il passé si X n'avait pas eu lieu ?"
  - Algorithme : BFS depuis les vraies racines sans X → `unique_effects` = effets qui n'auraient pas eu lieu sans X
  - `QueryResult::CounterfactualDiff` : `target`, `actual_effects`, `unique_effects`
  - Requête GCN-QL : `gcn query "COUNTERFACTUAL <nœud>"`
- **`gcn-frontend-en`** — frontend anglais symbolique (bootstrap d'annotation)
  - `EnglishParser::new(taxonomies_root)` + `EnglishParser::parse(text) -> CausalIR`
  - Pipeline identique à `gcn-frontend-fr` : tokenizer → tagger → rules → annotator → emitter
  - Marqueurs multi-mots prioritaires sur les verbes causaux (évite misattribution "in order to prevent" → Prevent)
  - Lemmatiseur anglais : strip `'s'` avant strip `'es'` ("causes" → "cause")
  - 16 tests d'intégration dont isomorphisme fr↔en sur la Condition
- **Taxonomies anglaises** : 10 fichiers dans `gcn-references/taxonomies/en/`
  - `conjonctions.yaml`, `verbes_causaux.yaml`, `verbes.yaml`, `adverbes.yaml`, `pronoms.yaml`, `determinants.yaml`, `noms.yaml`, `prepositions.yaml`, `prepositions_causales.yaml`, `nominalizations.yaml`
- **Réorganisation taxonomies** : `gcn-references/taxonomies/fr/` (11 fichiers déplacés dans sous-répertoire de langue)
- **`TaxonomyClass.examples`** : nouveau champ générique (en plus de `examples_fr`) pour les langues non-françaises
- **`RGCNLayerPT`** — couche 3 R-GCN PyTorch dans `gcn-python/layer3/pytorch_rgcn.py`
  - Remplace `RGCNLayer` (NumPy référence) pour les DS souhaitant GPU/MPS
  - `scatter_add_` vectorisé, `autograd` intégré, `torch_parameters()` pour optimizer PyTorch standard
  - Compatible Protocol `CausalGraph` : `parameters()` et `update()` restent NumPy pour compatibilité `CGNPipeline`
  - PyTorch déclaré optionnel : `ImportError` explicite si non installé
  - 11 tests incluant gradient flow, Protocol compliance, empty edges

### Modifié
- `gcn-backend/src/query.rs` : ajout variantes `Intervene(String)` et `Counterfactual(String)` au `Query` enum, message d'erreur GCN-QL étendu
- `gcn-knowledge/src/taxonomy.rs` : champ `examples: Option<Vec<LexicalEntry>>` ajouté à `TaxonomyClass`

---

## [0.6.0] — 2026-09-14

### Ajouté
- **`gcn-frontend-code`** — frontend code symbolique via tree-sitter (bootstrap d'annotation)
  - `CodeParser` pour Python, Rust et JavaScript (AST → CausalIR)
  - Bridge tree-sitter : `python.rs`, `rust.rs`, `js.rs`
  - `LabelStrategy` YAML — labels de nœuds dérivés des taxonomies AST, pas hardcodés
  - `gcn-references/taxonomies/python/python_ast.yaml`, `rust/rust_ast.yaml`, `js/js_ast.yaml`
  - 26 tests : Python (7), Rust (7), JS (7), isomorphisme fr↔py (1), py↔rs↔js (1), arête séquentielle (1), except body (1), label typé (1)
  - Critère de vérification : `if x < y: reduce(z)` et `"Si x est inférieur à y, on réduit z"` → CIR isomorphes ✅

### Corrigé
- `gcn-frontend-code` — arête séquentielle : était `data_edge(edge_rel)`, corrigée en `control_edge(Sequence)`
- `gcn-frontend-code` — `node_label` : tronquait au premier `:` (bug sur `def f(x: int)`), remplacé par `LabelStrategy` YAML
- `gcn-frontend-code` — `walk_try` : corps des `except_clause` silencieusement ignorés → récursion ajoutée
- `gcn-python/evaluation/metrics` — `generation_bleu` : brevity penalty utilisait `min(ref_lengths)` → standard BLEU (longueur la plus proche de l'hypothèse)

---

## [0.5.0] — 2026-09-14

### Ajouté
- **`gcn-verbalizer`** — décodeur de l'architecture encodeur-décodeur GCN (Phase 5)
  - `gcn-verbalizer` crate Rust : pont `Verbalizer::decode(ir)` → subprocess `gcn-verbalize`
  - `gcn-python/verbalizer/interface.py` : Protocol `VerbalizerDecoder`
  - `gcn-python/verbalizer/decoder.py` : implémentation NumPy de référence (remplaçable par PyTorch/JAX)
  - `gcn-python/verbalizer/cli.py` : CLI `gcn-verbalize` — entrée JSON stdin, sortie surface stdout
  - `gcn-datasets/schemas/gcn-verbalize.schema.yaml` : schéma des paires `(CausalIR, surface_text, lang)`
  - `gcn-datasets/examples/verbalize_cross_modal.yaml` : exemples fr + python du même CausalIR
- Métriques décodeur dans `gcn-python/evaluation/metrics.py` :
  - `decoder_causal_fidelity` : re-parser la sortie, comparer au CIR source
  - `cross_modal_consistency` : deux surfaces (fr + python) du même CIR
  - `roundtrip_similarity` : fidélité du cycle texte → CIR → texte
  - `generation_bleu` : qualité de surface (standard BLEU avec brevity penalty)

---

## [0.4.0] — 2026-09-14

### Ajouté
- `gcn-backend` — moteur de raisonnement causal (Pearl niveau 1)
  - `pearl.rs` : `WHY` (BFS inverse — ancêtres causaux), `WHAT` (BFS avant — descendants), `CHAIN` (chemin causal via `has_path_connecting` petgraph)
  - `query.rs` : parseur GCN-QL (`WHY`, `WHAT`, `CHAIN`, `CYCLES`, `GAPS`) + exécuteur ; `QueryResult` sérialisable JSON
  - `export.rs` : `to_json` (CausalIR → JSON) + `to_dot` (CausalIR → Graphviz DOT coloré par type de relation)
  - 18 tests d'intégration
- `gcn-cli` — interface ligne de commande complète (4 sous-commandes)
  - `gcn analyze <text> --data-dir <path>` — texte → frontend-fr → middleend → JSON ou DOT
  - `gcn query <query> --ir <file>` — charge un IR JSON et exécute une requête GCN-QL
  - `gcn export --ir <file> --format <json|dot>` — conversion de format
  - `gcn forward <text> --lang <fr> --enrich` — interface Rust↔Python via subprocess `gcn-forward`
- Variable d'environnement `GCN_TAXONOMY_DIR` pour configurer le chemin des taxonomies
- Feature `env` activée sur `clap` (workspace)
- `README.md` — documentation complète du projet

### Modifié
- `SAD.md` — 4 corrections d'alignement avec l'objectif moteur :
  - Phase 4 : Pearl niveau 1 uniquement (pas 1-2-3) + interface Rust↔Python explicite
  - Phase 5 : verbalisateur produit texte **et** code (pas "ou")
  - Phase 7 : Pearl 2-3 clarifié comme extension séparée
  - Vérification Phase 5 corrigée (testait frontend-code, pas verbalisateur)

---

## [0.3.0] — 2026-09-14

### Ajouté
- `gcn-middleend` — enrichissement du graphe causal (Phase 3)
  - `graph.rs` : construction `petgraph::DiGraph<NodeId, usize>` depuis `CausalIR`
  - `cycle.rs` : détection SCC (algorithme de Tarjan) + classification `FeedbackPositive` / `FeedbackNegative` / `Oscillation` ; marque les arêtes participantes (`in_cycle`)
  - `propagate.rs` : `TemporalGap` automatique sur arêtes `Concession`/`Opposition` ; avertissement sur inversions d'ordre temporel dans les arêtes `Sequence`
  - `validate.rs` : self-loop (erreur), nœuds orphelins, confidence < 0.3, nœud `Condition` sans arête sortante (avertissements)
  - `error.rs` : `Diagnostic`, `DiagnosticSeverity`, `DiagnosticKind`, `MiddleendError`
  - API publique : `process(CausalIR) -> Result<MiddleendResult, MiddleendError>`
  - 17 tests d'intégration

### Modifié
- `SAD.md` — mise à jour structure projet, statuts de phases, Phase 5 verbalisateur ajoutée

---

## [0.2.0] — 2026-09-14

### Modifié
- **Réorganisation de la structure projet** : séparation moteur / références / données
  - `gcn-core/data/taxonomies/` → `gcn-references/taxonomies/`
  - `gcn-core/datasets/` → `gcn-datasets/`
  - `gcn-core/` devient un workspace Rust pur (aucun fichier de données)
  - Chemin mis à jour dans `gcn-frontend-fr/tests/integration_fr.rs`

---

## [0.1.0] — 2026-09-14

### Ajouté
- **Phase 1 — Infrastructure IR**
  - `gcn-ir` : types `CausalIR`, `CausalNode`, `CausalEdge`, `NodeType` (7 variantes), `RelationType` (11 variantes), `Scope`, `Modifier` (12 variantes), `TemporalRef`, `TemporalGap`, `NodeOrigin`, `CausalCycle`, `Ambiguity`, `CodeCausalType`
  - `gcn-knowledge` : chargeur YAML (`Lexicon`, `Taxonomy`), `InferenceEngine`, `load_taxonomy`, `load_all_taxonomies`
  - 11 taxonomies GCN dans `gcn-references/taxonomies/` : verbes, verbes causaux, adverbes, adjectifs, conjonctions, prépositions, prépositions causales, noms, pronoms, déterminants, nominalisations

- **Phase 2a — Frontend français symbolique**
  - `gcn-frontend-fr` : pipeline tokenizer → tagger → rules → annotator → emitter
  - `FrenchParser::new(data_dir)` + `FrenchParser::parse(text) -> CausalIR`
  - Couverture : 12 patterns causaux du papier de recherche (condition, cause backward/forward, séquence, concession, motivation, verbes causaux, scope universel, causalité implicite, négation, hypothétique)
  - 16 tests d'intégration

- **Phase 2b — Couches ML Python**
  - `gcn-python/layer1` : extraction UD (spaCy) → `UDRepresentation` → vecteurs de clauses (`D_clause ≈ 136`, `D_edge ≈ 346`)
  - `gcn-python/layer2` : `CausalEncoder` Protocol + `MLPEncoder` référence NumPy (3 couches, He init)
  - `gcn-python/layer3` : `CausalGraph` Protocol + `RGCNLayer` référence NumPy (R-GCN, supporte les cycles)
  - `gcn-python/pipeline` : `CGNPipeline.forward()`, `loss()`, `backward()` (no-op référence)
  - `gcn-forward` CLI Python

- **Phase 2c — Évaluation**
  - `gcn-python/evaluation/metrics` : `node_accuracy`, `node_f1_per_class`, `node_macro_f1`, `edge_accuracy`, `edge_f1_per_class`, `edge_macro_f1`, `causal_graph_similarity`
  - `gcn-python/evaluation/recorder` : `TrainingRecorder` — enregistrement loss + métriques par epoch, export CSV/JSON

- **Datasets et schémas**
  - `gcn-datasets/schemas/gcn-nl.schema.yaml` — schéma langues naturelles
  - `gcn-datasets/schemas/gcn-pl.schema.yaml` — schéma langages de programmation
  - `gcn-datasets/examples/` — 4 exemples annotés (fr basic, fr cycles, python, rust)

- `SAD.md` — Software Architecture Document initial

---

[Unreleased]: https://github.com/Maik-start/projet_CNM/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Maik-start/projet_CNM/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Maik-start/projet_CNM/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Maik-start/projet_CNM/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Maik-start/projet_CNM/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Maik-start/projet_CNM/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Maik-start/projet_CNM/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Maik-start/projet_CNM/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Maik-start/projet_CNM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Maik-start/projet_CNM/releases/tag/v0.1.0
