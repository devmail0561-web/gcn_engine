# Architecture du moteur — branche refacto

Date : 2026-10-14
Branche : `refacto/arch-cible-refacto-2026-10`
Mode : lecture codebase, sans supposition. Faits en `fichier:ligne`.

## 1. Vue d'ensemble

```text
Texte
  ↓
L0 Tokenizer + Socle B
  ↓
L1 Embeddings + Pooling
  ↓
L2 Classification noeuds / liens / actes
  ↓
L3 Enrichissement graphe R-GCN
  ↓
Émission CIR JSON
  ↓
Rust : Middleend → Backend → Réponse
```

But : `BYTES + temps → CIR sourcé → réponse prouvée`. Pas de frontend par langue dans le chemin cible.

## 2. Rust — noyau

```text
CausalIR
  ├── nodes : 8 types
  ├── edges : 19 relations
  ├── cycles : 3 types
  └── unresolved : manques

Middleend
  ├── build graphe
  ├── Tarjan cycles
  ├── propagation gaps
  └── validation 8 contrôles

Backend
  ├── requêtes 25 (WHY, CHAIN, DO, ...)
  ├── raisonnement BFS / Dijkstra
  └── réponse UserResponse
```

| Bloc | Entrée | Sortie | Format |
|------|--------|--------|--------|
| IR | mémoire | mémoire | JSON snake_case, schema 2.0 (`gcn-ir/src/lib.rs:37-40`) |
| Middleend | CIR brut | CIR + diagnostics | Error = pendante / boucle, Warning = reste (`middleend/src/validate.rs`, `lib.rs:21-50`) |
| Backend | CIR + texte requête | QueryResult | JSON, confiance = produit (`backend/src/query.rs:755-796`, `pearl.rs`) |
| Respond | QueryResult | UserResponse | status ok / partiel / degrade + phrase FR + caveats (`backend/src/respond.rs:24-45,312-330`) |

Détail IR (`gcn-ir/src`) :
- `ir.rs` : `CausalIR {source_lang, source_text, nodes, edges: Vec<(NodeId,NodeId,CausalEdge)>, cycles, unresolved, metadata, intent?}`
- `node.rs` : `CausalNode {id u32, type 8, label, source_span, scope 6, modifiers, temporal_ref, temporal_index?, origin, parent?, kind?}` — 8 types : Processus, EtatLocal, EtatGlobal, Entite, Condition, Concept, Evenement, Contrainte + alias.
- `edge.rs` : `CausalEdge {relation 19, confidence f32, temporal_gap?, explicit, negated, marker_token?, in_cycle?, provenance?, derivation?, joint_group?, third?}` — 19 relations : 11 directes + 8 v3 ternaires. `Provenance {ref?, span Token|Code|Synthetic, method SymbolicRust|MlPython|Silver|Derived, version, at}`.
- `temporal.rs` : `TemporalRef {Absolute, Relative, Epsilon, Range, Unresolved}`, `TemporalGap {min?, max?, nature Immediate|Deferred|Continuous}`.
- `lattice.rs` : `Lattice {tokens:[{index 1-based, form, pos Verb|Noun|Adv|Punct|Other, dep_rel Root|Nsubj|Obj|Mark|Advcl|Det|Advmod|Punct|Other, dep_head, lemma, clause}], clauses}`.
- `normalize.rs` : trim + fold accents + lower + filtre a-z0-9 _-.

Détail Middleend (`gcn-middleend/src`) :
- `graph.rs:build` : DiGraph<NodeId,usize>, arêtes indexées, pendantes hors-graphe mais signalées.
- `cycle.rs` : tarjan_scc taille>1 ou self-loop, Concession → Oscillation sinon niée/Prevent → Negative sinon Positive.
- `propagate.rs` : gap Deferred auto sur Concession/Opposition, check ordre Sequence.
- `validate.rs` : 8 contrôles, seuil conf basse 0.3 + NaN.

Détail Backend (`gcn-backend/src`) :
- `pearl.rs` : WHY/WHAT BFS ancêtres/descendants, CHAIN Dijkstra `-ln(conf×poids)` + Yen top-k, DO entrantes coupées + BFS aval (`pearl.rs:1339`), COUNTERFACTUAL BFS moins atteignable sans X (`pearl.rs:1376,1414`), ANALOGY `[0.40/0.30/0.20/0.10]`, SPOF garde 500 noeuds.
- `query.rs` : 25 requêtes WHY WHAT CHAIN CYCLES GAPS DO COUNTERFACTUAL EXPLAIN ANALOGY ASK WHYNOT ZOOM_IN/OUT AGGREGATE CENTRALITY SPOF DOUBTS PROVENANCE DIFF DENSITY COVERAGE RELIABILITY CHAIN_T BEFORE DELAY.
- `export.rs` : JSON pretty + DOT.
- Honnête : 0 backdoor/frontdoor, cycles annotés jamais rejetés. Index sourcé utile, preuve causale identifiée non.

Hors-noyau :
- `frontend-fr/en` : texte → Lattice JSON infaillible, langue = sous-commande (`lib.rs:6-19`, `lattice.rs` 640/527 lignes, suffixes + positions).
- `frontend-code` : source tree-sitter → CIR Programming + CodeSpan + kind, table kinds en dur (`kinds.rs:55-321`).
- `frontend-graph` : bundle STIX → CIR Graph{Stix21} conf 1.0 Synthetic, inconnu → Concept (`mapper.rs:14-64`).
- `frontend-table` : CSV + TableSchema imposé → CIR Graph{Table} (`table.rs:27-55`).
- `cli gcn` : analyze → Lattice, analyze-code/graph/table → CIR middleend → json/dot/txt, query --ir --strict → UserResponse JSON cap 50Mo (`main.rs:374-412`), export --force.
- `verbalizer` : decode via subprocess `gcn-verbalize` stdin JSON (`lib.rs:24-29`), fallback pur Rust jamais mensonger (`fallback.rs:4-9`).

## 3. Python L0 — entrée

```text
Texte str
  → split
    → ids stables
    → offsets [début, fin)
    → tokens normalisés

Texte + langue
  → socle B
    → clauses c001
    → connecteurs
    → source silver-B / fallback
```

| Objet | Contenu |
|-------|---------|
| ids | u64 sha256 8 octets, sans vocab, PYTHONHASHSEED-indépendant (`tokenizer.py:94-97,124-183`) |
| offsets | spans caractères originaux, alignés 1:1 ids |
| tokens | normalisés par famille latin / CJK / arabe / autre (`tokenizer.py:70-91`, RULES_VERSION l0-v1 `:34`) |
| clauses | `{id:c001, texte, lang, span:[s,e]}` (`engine_b.py:171-181`) |
| connecteurs | `{span, clause_id}` (`engine_b.py:184-192`) |
| source | silver-B \| fallback obligatoire (`engine_b.py:40-42`), lazy spacy 3.8 timeout 30s (`:126-144`), compteur seuil 5% (`:57-62,94-105`), `to_record` format s0 20261007 (`:247-262`) |
| hash | `tokenizer_hash = sha256(règles+vocab trié)` (`tokenizer.py:100-113`), s0 vocab vide admis |

`nlu/tok.py:23-57` seul autorisé à appeler B : `parse_text → {ids, offsets, clause_spans[[s,e]], clause_textes, connecteurs, source, intent}`. `intent = detect_act` règles v0 `{assert|question|request, saillant, lang}` (`act.py:115-143`). Vide → segments vides, jamais exception. Cache FIFO 1024 (`:82-106`).

## 4. Python L1 — vecteurs

```text
ids + offsets + spans clauses
  → lookup E_tok
    → vecteurs (N, d=128)
  → alignement
    → tokens par clause
  → pooling moyenne
    → reps (C, d)
```

| Étape | Entrée | Sortie |
|-------|--------|--------|
| Vocab | ids triés | taille V + matrice E(V,128) N(0,sqrt1/d) (`tok_embedding.py:42-68`) |
| Lookup | id | vecteur copie, OOV → zéros comptés (`:98-118`), `oov_rate` |
| Align | offsets + [s,e) | indices tokens, règle début ∈ [s,e) (`l1_forward.py:59-72`) |
| Pool | (N,d) + groupes | (C,d) moyenne fixe, vide → zéro compté + PoolMap (`l1_forward.py:75-98`) |
| Persist | — | s1 `{format 20261007, tok_vocab, hash, d_model, E}` (`:154-236`), manifest `{hash, d_model}` exact (`chunk.py:24-73`) |

Gates : hash fail-fast, `vec_dim == manifest` sinon ValueError, OOV train >5% ValueError / val warn (`forward_ids.py:344-367`, `OOV_WARN_RATE=0.05 :71-72`).

## 5. Python L2 / L3 — décisions

L2 :

```text
reps (C, d)
  → MLP noeuds → logits (C, 8)
  → MLP liens  → logits (19,)
  → tête actes → (3,) assert / question / request
```

- `MLPEncoder` NumPy `D→128→64→out` He-init (`reference.py:46-56`), `forward_node (d,)→(8,)`, `forward_batch (N,d)→(N,8)`, `forward_edge (d_edge,)→(19,)` + têtes intent/sentence/quals.
- `ActHead (d→64→3)+softmax` s3 `{format, act_types[3], params}` (`act_head.py`).
- `stage2` hors-tok : `(N,36)→(N,12)` 11+REJECT.
- Dims canoniques `constants.py` : 8 noeuds, 19 relations.

L3 :

```text
H (C,d) + paires + types
  → messages voisins R-GCN
  → H enrichi (C,d)
```

- `RGCNLayer` NumPy : `σ(Σ_r 1/c W_r h + W0 h)` +1 slot no-edge (`reference.py:55-63`), backward complet.
- `GAT` torch optionnel même contrat + attention LeakyReLU + softmax/dst.
- `LinkPredHead` bilinéaire `sigmoid(uᵀWv+b)` séparée.
- Paires fenêtre `WINDOW_K=2` (`forward_ids.py:68`), `1 ≤ j-i ≤ 2`, `all_pairs` interdit en tok.
- Base arête `[src|dst|gap]` dim `2*d+1` (`:77-79`), full + enriched + probs (`:82-84`).
- Triplets `(i,j,relation,conf,negated=False,marqueur,ambiguous)` (`:444-456`), `scope=unknown, origin=explicit`, marqueurs B observés span→token (`:315-324,428-443`), `theta_ambiguity` + temperature (`:419-426`).

## 6. Émission CIR

```text
types + labels + spans + triples
  → noeuds [{id, type, label, span, scope, origin}]
  → arêtes [[src, dst, {relation, confiance, provenance}]]
  → CIR {nodes, edges, cycles=[], metadata, intent}
```

- `emit(...) → dict` (`ir_emitter.py:196-285,410-442,510-538`) : nodes `source_span{token_span}`, scopes, `origin`, attributes ; edges `confidence, confidence_ml, confidence_d6, temporal_gap, explicit, negated, marker_token, third, joint_group, ambiguous, in_cycle, provenance{ref, span, ml_python, version, at}`.
- Confiance D6 `rung(0.50 R1 / 0.75 R2 / 0.65 R3) × fMood(Ind1/Sub0.85/Cnd0.80) × gConn(explicite1/verb0.9/parataxis0.6/aucun0.5) + salience` (`:12-79,387-394`). Non appris — à remplacer par calibration.
- `cycles=[]` émetteur, Rust recalcule. `metadata{schema 2.0, pipeline}`, `source_lang und` volontaire (`:510-513`), `intent{act,lang,saillant}` fail-fast (`:528-537`).
- `forward_ids.py:530-594` : `forward_from_spans → CIR doc_ref=tok:source`, `forward_supervise` mêmes nombres + caches, `forward_cached` FIFO 1024. Gates vocab vide ValueError.

## 7. Train / eval tok

```text
Gold UD + texte
  → offsets chars (SpanError si échec, jamais deviné)
  → sample {ids, offsets, spans clauses, node_gold[8], edge_gold{(i,j):19}}
  → train (CE masquée, GelError si <30/classe, rejet alignement >5% STOP)
  → checkpoint .npz {format 20261007, hash, d_model, E, têtes}
```

- `gold_spans.py:40-78` : `reconstruire_offsets` find ordonné + plier apos 1:1, `span_clause` UD inclusif → [s,e) jamais clamp.
- `tok_loader.py:30-106` : `convertir_phrase` types/rels inconnus → SpanError, `i≥j` arrières / `j-i>k` hors-fenêtre exclus comptés, `charger_split` stats, `couverture_vocab` OOV.
- `train_tok.py:311-465` : rejet alignement >5% STOP (`SEUIL_REJET=0.05 :40`), vocab train, OOV sanity, espace fin 8+19 GEL vs coarse 4+5, `require_min_counts → GelError` (`refacto_loss.py:25-50`), boucle phrase `forward_supervise → backward_tok → update` (sans-gold sans grad `:106-205`), best `val_edge_macro_f1` + restore + CSV `{data_hash, git_sha, seed}`, silver weight 0.0 exclu.
- Checkpoint s1+ (`forward_ids.py:162-288`) : `{format, d_model, seed, temp, threshold, theta, bidir, window_k, vocab u64, hash, rules, E, node/edge Wb, rgcn Wr/W0/norm}`, `allow_pickle=False`, legacy refusé convertir-ou-réentraîner.
- Eval : `train_tok.evaluer :267-308` argmax → F1 macro NaN si vide ; `eval_runner` legacy interdit final/c1_* + legacy-UD (`:28-57`) ; calibration ECE 10 bins + T 0.1-5.0 + isotonie (`calibration.py:30-114`), ordre ML→D6→isotonie ; grid 250 essais + tripwire `≠(0.0,0.65,1.0)` (`grid_f1.py:33-103`), gate `N>=30` sinon GEL exit 2.

Chiffres actuels : val `0.239 ±0.04 (0.184/0.265/0.269)` vs train `0.99` (`README:633-645,681`), OOV `23.7%` vs seuil `5%` (`CHANGELOG:29`), ECE `0.33-0.55` vs `0.15` NO-GO Ph5 (`GATES:39`), 8 relations à 0 gel R8/R10 (`README:684-686`). Mémorisation, pas généralisation.

## 8. Flux legacy — divergence

```text
Legacy : texte → subprocess lattice → UD lemma/pos/dep_rel → vectorize 106d → two-pass + teacher gold → CIR
Tok    : texte → split in-process → ids + E_tok → pool → CIR
```

| Point | Legacy (`cgnp.py`, `loader.py`) | Tok (`forward_ids.py`) |
|-------|-------------------------------|------------------------|
| Segmentation | subprocess `gcn lattice` + pseudo-dep (`bridge.py:88-136`) | `tokenizer.split` + B in-process |
| Représentation | UD + `FeatureVocabulary` 106d (`features.py:75-128`) | ids + E_tok + moyenne fixe |
| Arêtes | base morpho + `all_pairs=True` + teacher gold + two-pass | base `[src\|dst\|gap]` + fenêtre k=2, no-edge sans teacher |
| Négation/scope | détectés / inférés | `False` / `unknown` / `explicit` figés |
| Masques | classes vides legacy | pas de masque, têtes non entraînées gelées R8 |

Gates refacto : `subprocess==0` (test monkeypatch `test_forward_ids.py:227-232`, Makefile grep, CI guards), `WINDOW_K=2` (test `dst-src≤K`), `format_refacto` partout, `LegacyUDError` + `allow_legacy=False` en eval.

## 9. Verdict honnête

Bon : lattice sans décision, ids stables hash, E_tok appris OOV compté, pooling/alignement figés testés, no-edge explicite jamais 0=cause, SpanError jamais deviné, gates GEL/format/subprocess/fenêtre, UserResponse jamais Err, offline CPU.
Mauvais : 2 chemins divergents, D6/D7 non appris, negated/scope/origin figés en tok, B non appris + SUPPORTED 6 + pt/it vides, backend dépend verbalizer PATH, middleend jette pendantes du graphe avant diagnostic, normalize collisionne, IR vide passe, harnais incomparables, val 0.239 + OOV + ECE NO-GO.
