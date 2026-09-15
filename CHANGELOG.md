# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [Unreleased]

### Corrigé

**Correctifs architecturaux (43a20f1)**
- **C1 — TaxonomyIndex retiré du pipeline ML** (`layer1/features.py`, `pipeline/cgnp.py`, `pipeline/label_builder.py`, `pipeline/cli.py`, `training/train.py`) — `FeatureVocabulary` devient purement syntaxique (UPOS + DEP_REL + tense/aspect/mood + polarity + flags structurels). Les taxonomies restent des guides pour les annotateurs, jamais des dépendances runtime du moteur.
- **C2 — `backward_message_pass` retiré de `RGCNLayerPT` et du Protocol `CausalGraph`** (`layer3/interface.py`, `layer3/pytorch_rgcn.py`) — levait `NotImplementedError` ; la garde `hasattr` dans `cgnp.py:backward()` retourne `False` pour les impls PyTorch, `autograd` gère le backward nativement.
- **C3 — Propagation du `temporal_index` corrigée dans les emitters FR et EN** (`gcn-frontend-fr/src/emitter.rs`, `gcn-frontend-en/src/emitter.rs`) — la chaîne `A→B→C` produisait `0, 1, 1` (index de la source répété) au lieu de `0, 1, 2`. Corrigé : propagation du max cumulé depuis la source.
- **C4 — Label condition langue-dépendant** (`gcn-frontend-fr`, `gcn-frontend-en`) — `hidden_cause` (EN) vs `cause_cachée` (FR) ; clé `examples_fr` → `examples` pour EN dans `_load_nominalizations`.
- **C5 — `d_out: int` ajouté au Protocol `CausalGraph`** (`layer3/interface.py`) — rend la dimension de sortie vérifiable sans instanciation.

**Correctifs post-audit frontend/backend (2f0d7f4)**
- **`pearl.rs` — Déduplication contrefactuelle par `NodeId`** — `CausalLink` gagne `to_id: NodeId` ; `counterfactual()` fait le lookup par id au lieu de label — corrige la perte de cardinalité quand deux nœuds distincts partagent le même label (C-4 / BUG-2).
- **`annotator.rs` — Arêtes implicites inter-phrases étendues à N phrases** — la src cherche la dernière clause non-`Hypothetical` via `.rev().find()` — corrige la limite aux 2 premières phrases et le double appel `annotate_sentence` (C-1, BUG-1).
- **`rules.rs` — `concession` → `MarkerDir::Backward`** (était `Forward`) ; `GoalToAction` initial : `(src_idx, dst_idx) = (0, 1)` fixe dans tous les cas (BUG-3/4, paper-013).
- **Lemmatiseur FR** — exclure `merci/aussi/demi/semi` du strip `-i` passé composé (faux positifs).

**`gcn-bootstrap` — Deux bugs corrigés (841ce9a)**
- Flag `--compact` inexistant dans `gcn analyze` supprimé (crash silencieux).
- `--taxonomy-dir` ajouté (`envvar GCN_TAXONOMY_DIR`) et transmis à `gcn analyze --data-dir` (was absent — le binaire Rust échouait toujours).

### Ajouté

- **`Makefile`** (racine) — cibles `install` / `install-dev` / `build-release` coordonnent `cargo build` + `pip install` depuis la racine.
- **`GCN_PYTHON_BIN`** (`gcn-cli/src/main.rs`) — variable d'environnement pour override du chemin `gcn-forward` ; fallback sur le nom nu si absente.
- **`gcn-eval` CLI** (`evaluation/eval_runner.py`) — `run_eval(data_dir, model_path)` agrège `node_accuracy`, `node_macro_f1`, `edge_accuracy`, `edge_macro_f1` sur un répertoire de données ; entry point `gcn-eval` dans `pyproject.toml`.
- **Métriques dans `gcn-train`** (`training/train.py`) — `TrainingRecorder` câblé : `node_accuracy` et `edge_macro_f1` loguées par époque dans le CSV et exportées en JSON (`.json` aux côtés du `.csv`) si `--log-csv` fourni.
- **`gcn-datasets/corpus/`** — sous-dossier pour le corpus réel ; `phrases_fr.txt` documente le workflow bootstrap (40 phrases cibles, 8 patterns causaux).

### Documentation

- **`PROGRESS.md`** — limite coréférentielle de `FrenchParser` qualifiée explicitement "bootstrap only, non-bloquante en production" pour neutraliser les faux positifs d'audit.

### Tests
- `fix_c4_unique_effects_two_same_label_nodes_both_unique` — deux nœuds distincts à label identique → 2 `unique_effects`
- `fix_bug2_counterfactual_duplicate_labels` — seul le nœud non-atteignable sans X apparaît dans `unique_effects`
- `fix_bug1_three_sentence_juxtaposition` — 3 phrases juxtaposées → 2 arêtes implicites
- `fix_bug3_hypothetical_targets_main_clause` — nœud hypothétique pointe vers l'événement surprenant
- `paper_013_goal_to_action_initial` — `Pour réussir, il travaille.` → Motivation(réussir→travaille)
- `fix_c1_concession_sentence_implicit_edge_src_not_hypothetical` — src arête implicite ≠ Hypothetical
- `lemmatize_non_verb_i_words_unchanged` / `lemmatize_real_participes_ir` (2 tests `rules.rs`)
- **89 / 89 tests Python passent** (`pytest gcn-python/tests/`)
- **137 / 137 tests Rust passent** (`cargo test --workspace`)

---

## [0.9.11] — 2026-09-15

### Corrigé
- **CLI `gcn-forward` cassée — unpack 2-tuple** (`pipeline/cli.py:69`) — `reps_from_sentence` retourne un triplet depuis 0.9.6 mais `cli.py` l'unpackait encore comme une paire, levant `ValueError: too many values to unpack` sur toute invocation. Corrigé : unpack complet `(reps, valid_clause_idxs, connector_reps)` ; `clause_positions`, `n_total_clauses` et `connector_reps` sont maintenant transmis à `pipeline.forward()`.
- **`except Exception` étouffait les `ValueError` de misconfiguration** (`training/train.py:92`) — le handler de samples malformés interceptait aussi les `ValueError` légitimes (d_out ≠ d_clause, clause_positions mal dimensionné), laissant un pipeline mal configuré entraîner 50 epochs sans gradient utile. Corrigé : `except ValueError: raise` avant le handler large.
- **`_relation_idx` appelé avant les guards skip-edge** (`data/loader.py:77`) — une arête non-supervisable (backward ou longue distance) avec une relation inconnue levait `ValueError` avant le `continue`, rejetant toute la phrase y compris ses labels de nœuds valides. Corrigé : `_relation_idx` déplacé après les guards.
- **`load_checkpoint` mutait `vocabulary` avant validation des shapes** (`training/checkpoint.py:33`) — un `ValueError` de mismatch laissait le pipeline dans un état incohérent (nouveau vocab, anciens poids). Corrigé : toutes les shapes sont validées avant toute mutation (opération atomique).
- **Check `d_out ≠ d_clause` trop tardif** (`pipeline/cgnp.py`) — la vérification avait lieu après le calcul complet `message_pass` (O(E·D²)) au lieu d'échouer à la construction. Déplacée dans `__init__` ; supprimée de `_forward_from_reps`.
- **Guard falsy sur `n_total_clauses=0`** (`pipeline/cgnp.py:129`) — `n_total_clauses if n_total_clauses else len(reps)` traitait `0` comme absent. Corrigé : `is not None`.
- **Aucun guard sur la longueur de `clause_positions`** (`pipeline/cgnp.py`) — une liste trop courte causait `IndexError`. Corrigé : `ValueError` explicite si `len(clause_positions) != len(reps)`.
- **Labels négatifs non détectés dans `_cross_entropy`** (`pipeline/cgnp.py:345`) — la sentinelle `-1` (arête non supervisée) passait silencieusement, interprétée comme index `-1` (dernière classe). Corrigé : `ValueError` si `labels.min() < 0`.
- **`AttributeError` sur `update_node`/`update_edge` pour encodeurs tiers** (`pipeline/cgnp.py:315`) — `backward()` vérifiait uniquement `backward_node_dx` mais appelait `update_node`/`update_edge` sans guard. Corrigé : appels conditionnels `hasattr`.

### Simplifié
- **Duplication `update_node`/`update_edge`** (`layer2/reference.py`) — corps identiques extraits dans `_apply_grads(layers, grads, lr)`.

### Tests
- `test_forward_rgcn_dout_mismatch_raises` mis à jour : le `ValueError` est maintenant attendu à la construction (`CGNPipeline.__init__`), plus au `forward()`.
- **86 / 86 tests Python passent** (`pytest gcn-python/tests/`)

---

## [0.9.10] — 2026-09-15

### Corrigé
- **Décalage forward/backward silencieux quand `d_out ≠ d_clause`** (`pipeline/cgnp.py`) — quand `RGCNLayer` était instancié avec `d_out ≠ vocabulary.d_clause`, le forward utilisait les logits pré-R-GCN, le backward opérait sur les snapshots pré-R-GCN (cohérent), mais les poids du R-GCN n'étaient jamais mis à jour : la couche 3 tournait en avant sans jamais apprendre, sans aucune erreur visible. Le `warnings.warn` était insuffisant car il n'empêchait pas l'entraînement de continuer dans cet état incohérent. Corrigé : le warning est remplacé par un `ValueError` levé au premier `forward()`, avant tout update de poids ; le double `if` conditionnel est supprimé. L'import `warnings` devenu orphelin est retiré.

### Tests
- `test_forward_rgcn_dout_mismatch_raises` ajouté (`tests/test_pipeline.py`) : vérifie que `ValueError` est levé avec le pattern `d_out=.*≠.*d_clause`.
- **86 / 86 tests Python passent** (`pytest gcn-python/tests/`)

---

## [0.9.9] — 2026-09-15

### Corrigé
- **Convention d'ordre implicite dans `backward()`** (`pipeline/cgnp.py`, `layer2/interface.py`, `layer2/reference.py`) — `backward()` assemblait une liste plate de gradients et les répartissait sur les paramètres de l'encodeur par position, supposant que `encoder.parameters()` retourne d'abord les paramètres nœud puis les paramètres arête. Une implémentation tierce retournant les paramètres dans un ordre différent aurait appliqué silencieusement les mauvais gradients aux mauvais poids, sans erreur visible.

  Corrigé : deux méthodes nommées `update_node(grads, lr)` et `update_edge(grads, lr)` sont ajoutées au protocole `CausalEncoder` et implémentées dans `MLPEncoder`. Chacune itère sur ses propres couches (`_node_layers` / `_edge_layers`). `backward()` appelle ces deux méthodes directement, sans assemblage plat ni hypothèse sur l'ordre de `parameters()`. `parameters()` et `update()` sont conservés pour le checkpoint et la compatibilité externe.

### Tests
- **85 / 85 tests Python passent** (inchangé)

---

## [0.9.8] — 2026-09-15

### Corrigé
- **Double normalisation des gradients** (`pipeline/cgnp.py`) — `_cross_entropy` normalise déjà ses gradients par N (`d_logits /= N`). `backward()` renormalisait à nouveau les gradients accumulés par `n` (nœuds) et `e` (arêtes), produisant un gradient effectif `1/N²` au lieu de `1/N`. Pour une phrase de 3 clauses, le learning rate réel était 9× trop faible ; pour 10 clauses, 100× trop faible — et variable selon la longueur des phrases. Les deux renormalisations redondantes sont supprimées ; `_cross_entropy` reste la seule source de normalisation.

### Tests
- **85 / 85 tests Python passent** (inchangé)

---

## [0.9.7] — 2026-09-15

### Corrigé
- **Arêtes backward mortes dans `edge_map`** (`data/loader.py`) — les arêtes gold en direction inverse (`src > tgt`) étaient stockées dans `edge_map` malgré le warning indiquant qu'elles ne sont jamais supervisées (le forward produit uniquement des paires croissantes). Ces entrées mortes consommaient de la mémoire inutilement. Corrigé : `continue` ajouté avant l'insertion, les arêtes backward sont désormais comptées pour le warning mais exclues de `edge_map`.

### Tests
- `test_backward_edge_not_supervised` mis à jour : vérifie désormais que `(1, 0)` **n'est pas** dans `edge_map` (comportement attendu après correction).

---

## [0.9.6] — 2026-09-15

### Corrigé
- **C1 — Crash sur node_type/relation inconnu** (`data/loader.py`) — `_to_sample()` appelée depuis `__iter__()` levait `ValueError` hors du `try/except` de `train.py`, tuant l'intégralité de la boucle d'entraînement. Corrigé : `__iter__` enveloppe chaque `yield` dans un `try/except ValueError` et émet un `UserWarning`.
- **M1 — Arêtes backward inversent la sémantique** (`data/loader.py`, `training/train.py`) — l'injection de la direction inverse `edge_map[(tgt, src)] = rel_idx` supervisait le modèle à prédire `cause(A→B)` quand le gold disait `cause(B→A)`. Corrigé : injection supprimée ; `train.py` utilise désormais un lookup strict `edge_map.get(p, -1)` ; les arêtes backward émettent un `UserWarning`.
- **M3 — Poids chargés avant vocabulaire** (`training/checkpoint.py`) — `load_checkpoint` restaurait les poids encoder/graph avant le vocabulaire ; si la taxonomie avait changé, `p[:] = data[key]` crashait ou corrompait silencieusement. Corrigé : vocabulaire rechargé en premier, puis validation des formes avant chaque `p[:] = data[key]`.
- **M4 — Features de position erronées pour clauses non-contiguës** (`pipeline/cgnp.py`, `training/train.py`) — quand des clauses étaient exclues (span vide), `vectorize_edge` recevait des indices dans la liste filtrée au lieu des positions originales. Corrigé : `forward()` accepte `clause_positions` et `n_total_clauses` ; `train.py` passe `valid_clause_idxs` et `len(sentence.clauses)`.
- **m2 — node_id manquant dans une arête silencieusement ignoré** (`data/loader.py`) — arête ignorée sans aucune trace. Corrigé : `UserWarning` émis.
- **m3 — Arêtes longue distance insérées dans `edge_map`** (`data/loader.py`) — entrées jamais consommées. Corrigé : `continue` après le warning, pas d'insertion.
- **m4 — Span inversée produit une clause vide silencieuse** (`data/loader.py`) — span `(start > end)` retournait silencieusement `None`. Corrigé : `UserWarning` émis.
- **m6 — `except Exception: continue` muet** (`training/train.py`) — les erreurs de forward étaient absorbées sans log. Corrigé : `warnings.warn` avec type et message de l'exception.

### Ajouté
- **M2 — Connecteur causal extrait et vectorisé** (`data/loader.py`, `pipeline/cgnp.py`) — `reps_from_sentence()` retourne maintenant un triplet `(reps, valid_indices, connector_reps)` ; le token connecteur (SCONJ/CCONJ/ADP ou `gcn_causal_type: "conjonction"`) entre deux spans consécutives est extrait via `_connector_between()` et passé à `vectorize_edge`. Le slot `d_conn` du vecteur d'arête est désormais actif.
- **m1 — Warnings sur champs JSON absents** (`data/json_reader.py`) — `UserWarning` si `type` ou `relation` absents d'un nœud/arête (défauts `"action"`/`"cause"` conservés).
- **m5 — Garde label hors-bornes dans `_cross_entropy`** (`pipeline/cgnp.py`) — lève `ValueError` explicite si `labels.max() >= n_classes`.
- **m7 — Pondération configurable `edge_loss_weight`** (`pipeline/cgnp.py`) — paramètre `edge_loss_weight: float = 1.0` dans `loss()` pour équilibrer la contribution des arêtes.

### Tests
- `test_invalid_node_type_warns_not_crashes` : C1 — node_type invalide → warning, pas crash
- `test_backward_edge_not_supervised` : M1 — arête backward → `edge_map` direction naturelle, lookup strict retourne -1
- `test_checkpoint_dimension_mismatch_raises` : M3 — `ValueError` sur dimensions incompatibles
- `test_forward_connector_slot_nonzero` : M2 — slot UPOS du connecteur non-nul quand connecteur fourni
- `test_forward_position_features_noncontiguous` : M4 — features de distance corrigées pour clauses exclues
- `test_rgcn_gradient_finite_differences` : vérification numérique `dW_r` / `dW_0` par différences finies (`eps=1e-4`)
- `test_edge_map_alignment` mis à jour : arêtes longue distance absentes de `edge_map`
- `test_reps_from_sentence_alignment` mis à jour : dépaquetage triplet `(reps, indices, connectors)`
- **85 / 85 tests Python passent**

---

## [0.9.5] — 2026-09-15

### Corrigé
- **Supervision des arêtes inverses perdue** (`data/loader.py`, `training/train.py`) — le forward prédit toujours les paires `(k, k+1)` mais le gold CIR encode parfois la causalité dans le sens inverse `(k+1, k)` ; `edge_map.get((0,1), -1)` retournait `-1` pour ces arêtes, éliminant leur gradient. Corrigé : `_to_sample` ajoute maintenant les deux sens pour toute arête consécutive (`gap == 1`) ; `train.py` effectue un lookup bidirectionnel `edge_map.get(p, edge_map.get((p[1], p[0]), -1))`.

### Supprimé
- **Champ mort `TrainingSample.gold_edge_labels`** (`data/loader.py`) — ce tableau (ordre d'insertion JSON) n'était jamais aligné avec les `edge_logits` et ne participait à aucun gradient. Le chemin actif passe exclusivement par `edge_map`. Champ et calcul associé retirés ; le fallback mort dans `train.py` remplacé par `gold_edge = None`.

### Ajouté
- **Warning arêtes longue distance** (`data/loader.py:_to_sample`) — une arête gold avec `|src_idx - tgt_idx| > 1` émet maintenant un `UserWarning` explicite : le forward ne prédit que les paires consécutives, aucune supervision d'arête n'est possible pour ces cas.
- **Warning R-GCN dimensionnel** (`pipeline/cgnp.py:_forward_from_reps`) — si `RGCNLayer` est instancié avec `d_out ≠ vocabulary.d_clause`, les logits ne sont pas recalculés après enrichissement R-GCN ; un `UserWarning` l'indique maintenant explicitement avec le remède (`d_out=d_clause`).

### Tests
- `test_edge_map_alignment` : vérifie que `_to_sample` émet un `UserWarning` pour une arête longue distance et que le reverse `(2, 0)` n'est pas ajouté (aucune supervision possible de toute façon)
- `test_dataloader_yields_batches` : assertion `gold_edge_labels.dtype` retirée (champ supprimé)
- **79 / 79 tests Python passent**

---

## [0.9.4] — 2026-09-15

### Modifié
- **Format des datasets migré de YAML vers JSON** (`gcn-datasets/examples/`, `gcn-core/tests/fixtures/`) — les fichiers annotés (sentences + tokens + CIR) sont désormais en `.json` ; les taxonomies linguistiques (`gcn-references/taxonomies/`, `gcn-knowledge/`) restent en YAML
- `data/yaml_reader.py` remplacé par `data/json_reader.py` — même API (`load_sentences`, `load_all_sentences`), `import json` stdlib au lieu de PyYAML, glob `*.json` au lieu de `*.yaml`
- `data/loader.py` — import mis à jour vers `json_reader`
- `pipeline/cli.py` — argument `yaml_path` renommé `dataset_path`, docstring mise à jour
- `training/bootstrap.py` — génère des `.json` via `json.dumps` (plus de dépendance `yaml` dans ce module)

### Supprimé
- `data/yaml_reader.py` — remplacé par `json_reader.py`
- `gcn-datasets/examples/*.yaml` (5 fichiers) — remplacés par leurs équivalents `.json`
- `gcn-core/tests/fixtures/paper_examples.yaml` — remplacé par `paper_examples.json`

### Tests
- `tests/test_yaml_reader.py` remplacé par `tests/test_json_reader.py`
- `tests/conftest.py` — fixture `paper_examples_yaml` pointe vers `paper_examples.json`
- **79 / 79 tests Python passent**

---

## [0.9.3] — 2026-09-15

### Supprimé
- **spaCy retiré du moteur** (`layer1/extractor.py` supprimé, `constants.SPACY_MODELS` retiré) — le moteur ne dépend plus d'un parser externe ; `UDRepresentation` est construite exclusivement depuis les tokens YAML annotés
- `CGNPipeline.forward(text)` supprimé — remplacé par `forward(reps, text)` qui prend une liste de `UDRepresentation` déjà construites
- `CGNPipeline.forward_from_reps()` fusionné dans `forward()` (alias supprimé)
- `train.py` : le fallback `pipeline.forward(sample.sentence.text)` supprimé — les samples sans tokens YAML sont ignorés (`continue`)
- `ir_emitter.py` : `"spacy-layer1"` retiré de la liste pipeline → `"cgnp-layer1"`

### Modifié
- `pipeline/cli.py` — `gcn-forward` accepte maintenant un fichier YAML annoté (format dataset) au lieu de texte brut ; option `--sentence-id` pour cibler une sentence spécifique
- `layer1/representation.py` — docstring épurée (mention spaCy supprimée)

### Corrigé
- **Indice 0 silencieux** (`data/loader.py`) — un `node_type` ou `relation` inconnu dans le YAML levait silencieusement l'indice 0 (`"etat"` / `"cause"`) ; lève maintenant `ValueError` avec le nom de la sentence et la valeur fautive
- **Troncature silencieuse** (`pipeline/cgnp.py`) — `loss()` et `_cross_entropy()` utilisaient `min(len(logits), len(gold))` pour absorber les désalignements ; lèvent maintenant `ValueError` si les tailles diffèrent, forçant un alignement explicite en amont

### Tests
- `test_pipeline.py` réécrit sans dépendance spaCy — tous les tests utilisent `forward(reps, text)` avec des `UDRepresentation` construites directement ; test `test_forward_two_reps` ajouté pour le cas multi-clauses
- `test_training.py` : `test_backward_updates_encoder_weights` et `test_loss_decreases_over_epochs` utilisent des `UDRepresentation` directes
- **79 / 79 tests Python passent** (0 skippé)

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
