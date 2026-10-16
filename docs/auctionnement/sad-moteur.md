# SAD — Moteur GCN (branche refacto, sans SA)

Date : 2026-10-14
Branche : `refacto/arch-cible-refacto-2026-10`
Statut : fidèle au code vérifié `fichier:ligne`. Remplace `gcn-core/SAD.md` archivé pré-refacto (7 noeuds / 11 relations / 21 requêtes).

## 1. Objet et finalité

Moteur = bibliothèque locale CPU. 3 actes seuls : extraire CIR, raisonner dessus, valider sa structure.

- Entrée : `BYTES + temps`. Sortie : réponse prouvée. Pivot : CIR sourcé.
- Assertions sourcées, pas faits (`README:27-32`, `ETUDE_SA:129-131`). DO = coupe + propage, counterfactual = monde-sans-X cité, sans confondeurs.
- Principes : CIR immuable, span obligatoire, <10 Mo CPU offline, confiance non calibrée = provisoire, toujours-répondre ok/partiel/degrade jamais vide.

## 2. Vue ensemble

```text
Texte brut
  ↓
L0 — split + socle B
  ids stables + clauses + source

  ↓
L1 — embeddings + pooling
  reps (C, d)

  ↓
L2 — classification
  noeuds (C,8) / liens (19,) / actes (3,)

  ↓
L3 — graphe R-GCN
  reps enrichies

  ↓
CIR JSON schema 2.0
  nodes + edges + provenance

  ↓
Middleend
  graphe + cycles + gaps + validation

  ↓
Backend
  25 requêtes → UserResponse
```

| Flèche | Format | Dims |
|--------|--------|------|
| split | ids u64 + offsets [s,e) | N variable |
| clauses | c001 + texte + span | C clauses |
| reps | moyenne fixe | (C,128) |
| noeuds | logits | (C,8) |
| liens | logits | (19,) |
| CIR | JSON snake_case 2.0 | nodes + edges |
| réponse | UserResponse | status + phrase + caveats |

## 3. gcn-ir — vocabulaire

`EXCHANGE_SCHEMA_VERSION = 2.0` (`gcn-ir/src/lib.rs:40`).

CIR (`ir.rs:9-22`) : `source_lang / source_text / nodes / edges Vec<(NodeId,NodeId,CausalEdge)> / cycles / unresolved / metadata / intent?`

Noeuds 8 (`node.rs:13-30`) : Processus, EtatLocal, EtatGlobal, Entite, Condition, Concept, Evenement, Contrainte + alias action/transition→Processus, etat→EtatLocal.

Relations 19 (`edge.rs:11-33`) : 11 directes Cause Enable Prevent Condition Concession Sequence Motivation Filter Opposition DataDependency ControlDependency + 8 v3 Analogy Counterfactual ConditionalCause MediatedCause JointCause ConditionalPrevent MediatedPrevent JointPrevent. `signals_causal_gap` = Concession|Opposition seuls. `is_joint` = Joint*.

Noeud (`node.rs:133-156`) : `id u32 / type / label / span TokenSpan|CodeSpan|Synthetic / scope 6 / modifiers / temporal_ref / temporal_index? / origin Explicit|Inferred|Hypothetical / attributes / parent? / kind?`

Arête (`edge.rs:181-206`) : `relation / confidence f32 / gap? / explicit / negated / marker? / in_cycle? / provenance? / derivation? / joint_group? sha256[:16] / third? Condition|Mediator`

Provenance 5 immuables (`edge.rs:104-112`) : ref span méthode SymbolicRust|MlPython|Silver|Derived version date (horloge sans chrono `edge.rs:120-163`). Derivation (`edge.rs:114-118`) : sources A→C de A→B+B→C.

Cycle (`ir.rs:115-128`) : `id / path / FeedbackPositive|Negative|Oscillation`. Ambiguity (`ir.rs:130-154`) : noeud + champ NodeType|Scope|AgentType|CorefTarget + candidats + indice. Intent (`ir.rs:24-48`) : `act Assert|Question|Request / lang défaut fr / saillant`, None = legacy. Temporal (`temporal.rs`) : Absolu Relatif Epsilon Plage NonRésolu + gap min?/max?/Immédiat|Différé|Continu. Scope (`scope.rs`) : Universal Existential Partial Null Specific Unknown. Modifiers 12 + Frequency 5 + Location 3 + Maturity 4. Code : 10 types causaux + attributs flux.

Normalisation (`normalize.rs:20-48`) : trim + fold accents + ligatures + filtre a-z0-9 _- + lower.

Lattice (`lattice.rs:13-107`) : `tokens [{index 1-based, forme, pos Verb|Noun|Adv|Punct|Other, dep Root|Nsubj|Obj|Mark|Advcl|Det|Advmod|Punct|Other, tête, lemme, clause}] + clauses`, `as_ud / as_upos`. Pseudo-dep positionnels, aucune décision.

## 4. middleend — graphe et validation

`process(ir) -> {ir, diagnostics}` (`lib.rs:21-50`) : doublons id → GraphBuildError, `graph::build`, `cycle::detect_and_classify` + in_cycle, `propagate::run`, `validate::run`, pipeline += middleend.

| Étape | Fonction |
|-------|----------|
| build (`graph.rs:19-35`) | DiGraph<NodeId,usize>, pendantes hors-graphe mais signalées |
| Tarjan (`cycle.rs:12-73`) | SCC taille>1 ou self-loop, Concession→Oscillation sinon niée/Prevent→Negative sinon Positive |
| gaps (`propagate.rs:17-27`) | Deferred auto sur Concession/Opposition sans gap |
| ordre (`propagate.rs:31-52`) | Warning si Sequence src.index > dst.index |

8 contrôles (`validate.rs:10-142`, seuil 0.3) :

| Contrôle | Niveau |
|----------|--------|
| Arête pendante | Error |
| Boucle réflexive | Error |
| Orphelin (si >1 noeud) | Warning |
| Confiance <0.3 ou NaN | Warning |
| Condition sans sortie | Warning |
| Label vide | Warning |
| Doublon arête | Warning |
| Provenance manquante | Warning |

Limite : pendantes ignorées du graphe avant diagnostic, conf>1 non contrôlée, pas de fusion multi-phrases.

## 5. backend — requêtes et réponse

25 requêtes (`query.rs:433-526`, parse `755-795`, execute `2174`) :

| Famille | Requêtes |
|---------|----------|
| Association | WHY WHAT CHAIN CYCLES GAPS |
| Intervention | DO |
| Contrefactuel | COUNTERFACTUAL |
| Abductif | EXPLAIN + contraste |
| Analogie | ANALOGY poids [0.40/0.30/0.20/0.10] seuil 0.3 (`analogy.rs:189-191`) |
| Fiche | ASK WHY-NOT |
| Hiérarchie | ZOOM_IN ZOOM_OUT AGGREGATE |
| Graphe | CENTRALITY SPOF garde 500 DOUBTS PROVENANCE DIFF DENSITY COVERAGE RELIABILITY |
| Temporel | CHAIN_T BEFORE DELAY |

Options : AMBIGUITY_LIMIT 5, TOPK défaut 3, SCOPE AGENT METHOD SINCE ONLY_EXPLICIT WEIGHTS MIN_SCORE HOPS CONTRAST. Filtres après recherche sans couper (`query.rs:197-272`). Caveats (`query.rs:274-344`) : Hypothetical, scope, unresolved.

Pearl (`pearl.rs`) : WHY/WHAT BFS (`351-404`), CHAIN Dijkstra `-ln(conf×poids_origine)` (`423-529`) + Yen top-k (`554-615`), DO entrantes coupées + BFS aval (`1308-1349`), COUNTERFACTUAL BFS moins atteignable depuis racines sans X (`1355-1491`, appelé `1414`), abduction `min_conf/(1+depth)` (`1067-1211`), SPOF paires BFS (`1020-1065`), `ORIGIN 1.0/0.7/0.3` (`983-1002`), `match_score` exact/préfixe/contient (`100-127`). 0 backdoor/frontdoor. Borné assertions sourcées.

Réponse (`respond.rs:24-45,312-350`) : `UserResponse{status ok|partiel|degrade, phrase FR 1-3, lang, result?, caveats[], suggestions[]}` jamais Err. Degrade = parse/introuvable/vide/incident/IR Error → exit 1 CLI (`main.rs:409-411`). Partiel = ambigu/sans-chemin/flou/coupures/trop-gros. Batch doit tester statut.

Export (`export.rs`) : JSON pretty + DOT formes/couleurs. Erreurs (`error.rs`) : NodeNotFound ParseError Serialization NoPath-strict Ambiguous.

## 6. frontends / CLI / verbalizer

| Bloc | Entrée → Sortie | Règles / limites |
|------|-----------------|------------------|
| FR (`lattice.rs:640`) / EN (`527`) | texte → Lattice JSON infaillible (`lib.rs:18-19`) | suffixes (`FR:90-204` / `EN:83-122`) + positions (`FR:281-468`), 1 marqueur, virgule=clause, sans dico/YAML, langue=sous-commande |
| Code (`kinds:332` + py/rs/js) | source tree-sitter → CIR Programming + CodeSpan (`lib.rs:19-52`) | table kinds en dur `55-321`, transparents traversés, non-mappés comptés, labels 256 (`common.rs`) |
| Table (`table:233` + `csv:119`) | CSV + schéma imposé → CIR Graph Table (`table.rs:80-233`) | `TableSchema{cause/effect/relation?/confidence?/défauts/délimiteur}` jamais deviné, 1 ligne=1 arête, dédup stricte, vide/inconnu/hors[0,1] ignoré compté |
| Graph STIX (`stix:195` + `mapper:91`) | bundle JSON → CIR Graph Stix21 conf 1.0 Synthetic (`lib.rs:40-42`) | inconnu→Concept (`mapper:23-61`), 12 relations mappées (`:76-84`), autres comptées |
| CLI `gcn` (`main.rs:43-448`) | analyze→Lattice, analyze-code/graph/table→CIR+middleend→json/dot/txt, extract-text→phrases/ligne, query --ir --strict→UserResponse cap 50Mo warn version, export --force, forward retirée | recalcul cycles à chaque query, strict CHAIN→Err si aucun |
| Verbalizer | decode CIR→subprocess `gcn-verbalize` stdin JSON (`lib.rs:29-64`) | drop stdin anti-deadlock, fallback pur Rust jamais mensonger `say(Status,SpeechKind)` (`fallback.rs`) |

## 7. L0-L3 Python

L0 split (`tokenizer.py`) : `text → ids u64 sha256[:8] + offsets [s,e) original + tokens`, `RULES l0-v1`, familles latin/CJK/arabe/autre, latin minuscule sans accent autres NFC, CJK 1char/token, vide→vide, re+unicodedata seuls. Hash `sha256(règles+vocab)` (`:100-113`).

L0 B (`engine_b.py`) : `(text,lang 6)→clauses c001 + connecteurs + source silver-B|fallback` (`:40-42,203-231`), spacy 3.8 lazy timeout 30s (`:46-62,126-144`), compteur seuil 5%, `to_record` format s0 20261007. Fallback 100% stdlib (`fallback_regex.py`) : segmentation ponct forte + tables EXISTANTES + snapshot minimal, pt/it sans table → vide (`:20-22,37,210-218`).

L1 E_tok (`tok_embedding.py`, d=128) : vocab trié, matrice E(V,d), lookup copie OOV zéros comptés, s1 `{format, vocab, hash, d_model, E}` fail-fast, manifest exact (`chunk.py`), `vec_dim==manifest`. `forward_ids` lookup+masque, `align_spans` début∈[s,e) (`l1_forward.py:59-72`), `pool` moyenne fixe vide→zéro. Legacy 106d = 18+38+5+5+4+5+3+7+1+3+12+5 (`features.py:75-90`) + embeddings lemmes, word_embedding obligatoire.

L2 MLP (`reference.py`) `D→128→64→out` He : noeuds (d,)→(8,), liens →(19,), têtes intent/sentence/quals, dropout train-only. Actes règles v0 (`act.py`) `?→question !→assert+saillant consigne/politesse→request défaut assert` + tête apprise `(d→64→3)` s3 (`act_head.py`). Stage2 hors-tok `(36,)→(12,)` 11+REJECT (`stage2.py`).

L3 RGCN (`reference.py`) `σ(Σ 1/c W_r h + W0 h)` +1 slot no-edge, GAT torch même + attention, link_pred bilinéaire séparée. Paires fenêtre k=2 (`forward_ids.py:68`), base `[src|dst|gap]` 2d+1, triplets `negated=False scope=unknown origin=explicit`, marqueurs B observés.

## 8. Pipeline tok vs legacy

Tok (`forward_ids.py`) inférence seule pré-segmenté, jamais segmentation/B/texte brut, RGCN sauté si 1 clause, no-edge sans teacher, pas de masque, vocab vide ValueError, OOV>5% warn, cache 1024. Persist s1+ `{format, d_model, vocab u64, hash, rules, E, têtes, arch}` allow_pickle=False.

Legacy UD (`cgnp.py`) : `vectorize 106d` + two-pass + RGCN all_pairs + teacher gold + négation détectée + scope inféré. `emit` D6 `rung(0.50/0.75/0.65)×humeur×connecteur+salience` + D7 orientation/voix/ternaire/négation (`ir_emitter.py:12-193`) non appris, `cycles=[]`, `und` volontaire, intent fail-fast.

Gates : subprocess==0 (`test_forward_ids.py:227-232`), WINDOW 2, masks N>=30 GelError exit 2 (`grid_f1.py:101-103`), format 20261007, SEUIL rejet 5% STOP, LegacyUDError.

## 9. Data / train / eval

Spans token→char find ordonné + plier apos, SpanError jamais deviné/clamp (`gold_spans.py:40-78`). Sample `{ids, offsets, spans, node_gold[8], edge_gold{(i,j):19}, hors, arrières}` types inconnus SpanError, `i≥j` arrière `j-i>k` hors exclus comptés (`tok_loader.py:30-106`).

Train tok supervision gold pure chaîne `d_edge→MLP→split→d_node→RGCN→pool⁻¹→E_tok` (`train_tok.py:106-205`), sans-gold sans grad, GelError, silver 0.0 exclu, best `val_edge_macro_f1` + restore + CSV `{data_hash, git_sha, seed}` (`:311-465`).

Checkpoint atomique tmp+replace, refuse symlink, `guarded_np_load` même trusted (`checkpoint.py`, `security/npz_guard.py`).

Eval argmax F1 macro NaN si vide, legacy interdit final/c1_* + legacy-UD (`eval_runner.py:28-57`), calibration ECE 10 bins + T 0.1-5 + isotonie (`calibration.py`), grid 250 + tripwire `≠(0.0,0.65,1.0)` (`grid_f1.py`).

Chiffres actuels : val `0.239 ±0.04 (0.184/0.265/0.269)` vs train `0.99` (`README:633-645,681`), split 572/132 quota64 (`README:628-630`), OOV `23.7%` vs `5%` (`CHANGELOG:29`), ECE `0.33-0.55` vs `0.15` NO-GO N=81 (`CHANGELOG:270`, `GATES:39`), 8 relations à 0 gel R8/R10 (`README:684-686`). Mémorisation, pas généralisation. Harnais 0.468 vs 0.1867 archivés incomparables (`README:650-652`).

## 10. NLU / dialogue / engine / reasoning / sécurité

NLU tok sans UD zéro lexique (`nlu/tok.py`), intent actes, cache hits sautent B. NLU legacy B+actes+concepts + silver traçé (`nlu/infer.py`). Dialogue session pile 3 + digits Unicode + paraphrase 0.75 + clarification unique + `lang:code` opaque + journal 200 + `revoke:` + caps 500/2000 (`dialogue.py`). Engine `from_pretrained` détecte s1+ sans désérialisation → tok-only sinon legacy + parser lattice (`engine.py`). Reasoning R1 pur borné profondeur 2 topk5 budget16 (`reasoning/engine.py`). Sécurité `assert_npz_pickle_safe` allowlist + `guarded_np_load` obligatoire. Verbalizer linéarisation + teacher surfaces gold + instructions commandes fermées.

## 11. Limites et blocages

Double chemin à unifier vers tok, D6/D7 à calibrer appris, B non appris + SUPPORTED 6 + pt/it vides, backend→verbalizer PATH fragile, pendantes hors-graphe, normalize collisions, IR vide passe, Ph5 NO-GO → provisoire, checkpoint non re-mesuré, dataset 3M 5% 50-95j, MVP gate P3.
