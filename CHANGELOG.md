# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [Unreleased]

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

[Unreleased]: https://github.com/Maik-start/projet_CNM/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/Maik-start/projet_CNM/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Maik-start/projet_CNM/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Maik-start/projet_CNM/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Maik-start/projet_CNM/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Maik-start/projet_CNM/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Maik-start/projet_CNM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Maik-start/projet_CNM/releases/tag/v0.1.0
