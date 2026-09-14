# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [Unreleased]

### Planifié
- Phase 5 : `gcn-verbalizer` — graphe causal → texte naturel **et** code simultanément
- Phase 6 : `gcn-frontend-code` — AST Python, Rust, JavaScript via tree-sitter
- Phase 7 : Pearl niveaux 2-3 (intervention, contrefactuel), R-GCN optimisé, wolof/arabe

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

[Unreleased]: https://github.com/Maik-start/projet_CNM/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Maik-start/projet_CNM/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Maik-start/projet_CNM/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Maik-start/projet_CNM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Maik-start/projet_CNM/releases/tag/v0.1.0
