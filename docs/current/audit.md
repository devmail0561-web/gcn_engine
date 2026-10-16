Audit complet — Pipeline et flux de données du moteur
Architecture : 5 chemins d'entrée → CausalIR unifié → Middleend → Backend
Texte FR/EN ──→ gcn-frontend-fr/en ──┐
Code Py/Rust/JS → gcn-frontend-code ──┤
STIX 2.x ──────→ gcn-frontend-graph ──┼──→ CausalIR ──→ gcn-middleend ──→ gcn-backend
CSV ───────────→ gcn-frontend-table ──┤
Texte+ML ──────→ gcn-python (bridge) ─┘
PATH 1 : Symbolique NL (gcn-frontend-fr/en)
Flux
1. Tokenisation : &str → Vec<Token> (découpage whitespace, clitiques FR séparés)
2. POS tagging : Vec<Token> → Vec<TaggedToken> (heuristiques + lexiques YAML)
3. Annotation : Vec<TaggedToken> → SentenceAnnotation (stratégie verbe OU marqueur, pas les deux)
4. Émission : SentenceAnnotation → CausalIR
Pertes de données
- Ponctuation : supprimée en tokenisation (sauf . et ,)
- Stratégie unique : si une phrase a à la fois un verbe causal et un marqueur, seule la première stratégie est utilisée. Incohérence FR/EN : FR essaie verbe d'abord, EN essaie marqueur d'abord → mêmes phrases, structures différentes
- temporal_gap : toujours None (calculé par middleend)
- cycles/unresolved : toujours vides (calculés par middleend)
- derivation/joint_group_id/third : toujours None
Incohérence temporelle
- FR : TemporalRef::Range { start: -60, end: 0 } pour "depuis"
- EN : TemporalRef::Unresolved pour aspect progressif
- Sémantique asymétrique
PATH 2 : Code (gcn-frontend-code)
Flux
1. Tree-sitter : &str → AST (échec total si erreur de syntaxe)
2. Mapping AST : nœuds tree-sitter → NodeType/RelationType via YAML
3. Marche AST : émet nœuds + arêtes ControlDependency/Sequence
4. Émission : CausalIR avec CodeSpan (ligne/colonne)
Pertes de données critiques
- CodeAttributes jamais peuplé : CodeCausalType, DataFlow, ControlFlow définis dans code.rs mais jamais utilisés — la sémantique code (data flow, side effects, pureté) est perdue
- scope : toujours Unknown
- modifiers : toujours vides
- temporal_ref : toujours Unresolved
- temporal_index : séquentiel next_id - 1 (pas sémantique)
- attributes : toujours Default::default() (pas d'entity, agent, patient)
- Nœuds non mappés : silencieusement ignorés (continue)
Comparaison NL vs Code
Feature	NL Frontend	Code Frontend
Scope	Depuis déterminant	Toujours Unknown
Modifiers	Negation, Temporality	Toujours vides
TemporalRef	Unresolved/Range	Toujours Unresolved
TemporalIndex	Propagé par arêtes	Séquentiel
Attributes	entity, agent, patient	Toujours default
Label	Riche (verbe(agent))	Texte tronqué 64 chars
PATH 3 : ML Pipeline (gcn-python)
Flux GCNDataLoader
JSON → SentenceRecord → TrainingSample
  ├── gold_node_labels: np.ndarray (N,) int64
  ├── edge_map: dict[(src,tgt)] → rel_idx
  ├── hyperedge_map: dict[(frozenset,tgt)] → rel_idx
  └── edge_conf_map: dict[(src,tgt)] → float
Filtrage et pertes
- Arêtes asymétriques inverses (src > tgt) : cause/enable/prevent → supprimées silencieusement avec warning
- Relations inconnues : ValueError, sample entier sauté
- Hyperedges (N-aires) : routées vers hyperedge_map mais jamais supervisées (warning seulement)
- IDs nœuds inconnus : arêtes supprimées avec warning
Flux CGNPipeline.forward()
UDRepresentation[] → vectorize_clause() → clause_vecs (N, 106+d_emb)
  → encoder.forward_node() → node_logits (N, 8)
  → R-GCN message passing (all_pairs ou adjacent)
  → encoder.forward_node() → node_logits (N, 8) [2e passe]
  → vectorize_edge() + enriched → encoder.forward_edge() → edge_logits
  → softmax(temperature) → confidence
  → filtrage edge_threshold
  → emit() → CausalIR dict
Pertes dans l'émission ML
- confidence_ml, confidence_d6, source_polarity, ambiguous, candidates : champs Python-only, perdus si round-trip via Rust serde
- unresolved : toujours vide
- parent : jamais émis
- is_inferred : émis mais noté "Python-only, ignoré par serde Rust"
Bridge (frontend/bridge.py) — compression lossy
Texte → subprocess gcn analyze → CausalIR → UDRepresentation[]
- root_pos : heuristique depuis node_type (~85% précision)
- root_dep_rel : heuristique (~80%)
- root_morph : toujours {} (Tense/Aspect/Mood toujours _absent)
- is_negative : toujours False
- has_advcl : toujours False (sauf Condition)
- subject_pos : depuis attributes.agent (~70%)
14 dimensions de features sont nulles après le bridge.
PATH 4 : Nouveaux frontends
gcn-frontend-graph (STIX)
- Relations non causales : ignorées mais comptées (GraphParseReport)
- Attributs nœuds : tous par défaut
- scope : toujours Unknown
- temporal_ref : toujours Unresolved
gcn-frontend-table (CSV)
- Lignes vides/relation inconnue/confiance invalide : ignorées mais comptées
- Tous les nœuds : même node_type (défaut Entite)
- scope : toujours Unknown
extract-text
- PDF : nécessite pdftotext externe, erreur explicite sinon
- HTML : détaggage interne, entités limitées à un set fixe
- IPYNB : cellules code ignorées (markdown seul)
- Découpage phrases : frontières potentielles sur abréviations hors liste
PATH 5 : Backend
Moteur de requêtes
- 21 types : WHY, WHAT, CHAIN, CYCLES, GAPS, DO, COUNTERFACTUAL, EXPLAIN, ANALOGY, ZOOM_IN/OUT, AGGREGATE, CENTRALITY, SPOF, DIFF, DENSITY, COVERAGE, RELIABILITY, CHAIN_T, BEFORE, DELAY
- Label matching : normalize_label() (accents, casse, ponctuation) + scoring exact=0, prefix=1, contains=2
Pertes dans les requêtes
- third_node_label : toujours None dans link_to_dto() (query.rs:954)
- WHY/WHAT : retournent seulement la première correspondance, les autres silencieusement ignorées
- CHAIN : retourne Path { found: false } vs chain_strict() retourne Err(NoPath) — deux comportements différents
Export DOT
- Perte : IDs, scopes, modifiers, temporel, provenance, cycles — seule la structure visuelle est préservée
Incohérences transversales
I-1 : Temporal Index — 4 sémantiques différentes
Frontend	Sémantique
NL	Propagé par arêtes causales
Code	Séquentiel next_id - 1
Graph/Table	node_id.0 (séquentiel)
ML	Rang par position token
Un index de 2 signifie "3e dans l'ordre causal" pour NL, "3e instruction" pour code, "3e objet" pour graph, "3e dans le texte" pour ML.
I-2 : Confiance — 4 échelles différentes
Frontend	Valeurs
NL	1.0 explicite, 0.5 implicite, 0.3 hypothétique
Code	Toujours 1.0
Graph/Table	1.0 ou depuis CSV
ML	Softmax (0.0-1.0) + D6 prior séparé
I-3 : Couverture des relations
Frontend	Relations
NL	11 (directes)
Code	3 (ControlDependency, Sequence, Concession)
Graph	4 (Enable, Cause, Prevent, Analogy)
Table	19 (toutes, depuis CSV)
ML	19 (toutes)
I-4 : Cycles — 2 algorithmes différents
- Rust middleend : Tarjan SCC sur petgraph
- ML pipeline : DFS sur liste d'arêtes
- Résultats peuvent différer pour graphes complexes
I-5 : Relations ternaires — jamais émises
- Rust IR : supporte TernaryThird (role, node, polarity)
- ML pipeline : detect_ternary() détecte rôles mais third_node_label toujours None
- Frontends : n'émettent jamais d'arêtes ternaires
- Backend : link_to_dto() toujours third_node_label: None
Code mort et chemins inutilisés
Élément	Statut
InferenceEngine::enrich()	Défini, jamais appelé
CodeAttributes, CodeCausalType, DataFlow, ControlFlow	Définis, jamais peuplés
gcn-veralizer crate	Non référencé par aucun autre crate
Query::chain_strict()	Défini, jamais appelé (CLI utilise execute())
AGENT_TYPE_VALUES	Marqué "RÉSERVÉ — non utilisé"
RELATION_TYPES_INV	Utilisé seulement si bidirectional=True (défaut False)
Gaps de gestion d'erreurs
Point	Problème
NL Frontend	Pas de récupération partielle si clauses.is_empty()
Code Frontend	All-or-nothing : une erreur de syntaxe = aucun IR
Graph Frontend	parse_bundle() discard le report (seul parse_bundle_with_report l'expose)
Table Frontend	FieldCount sur première incohérence, pas de traitement partiel
ML Loader	Si tous les samples échouent, l'epoch produit aucun signal
Bridge	Pas de fallback si subprocess échoue
ML Loss	Loss non fini → warning seulement, training continue avec gradients potentiellement corrompus
Préoccupations performance
Point	Impact
SPOF O(n²·(n+e))	500 nœuds max, pire cas 500×500×(500+e)
All-pairs edge classification	N=20 clauses → 190 classifications
Rebuild graphe par requête	Chaque why()/what()/chain() reconstruit le DiGraph
Bridge subprocess	1 spawn par texte, pas de batch
CSV parser byte-by-byte	Plus lent qu'un bulk read pour gros fichiers
Tree-sitter par fichier	Pas d'incrémental ni cache
Résumé des pertes de données
Étape	Perte	Impact
Tokenisation NL	Ponctuation	Faible
Annotation NL	Stratégie unique (verbe XOR marqueur)	Moyen — incohérence FR/EN
Émission NL	temporal_gap, cycles, derivation, third	Moyen
Code Frontend	CodeAttributes, scope, modifiers, temporal	Élevé — sémantique code perdue
Graph Frontend	Relations non causales, attributs	Moyen
Table Frontend	Lignes invalides	Faible — comptées
Bridge ML	Structure token complète, morphologie, négation, scope	Élevé — 14 dims nulles
Émission ML	confidence_ml, confidence_d6, source_polarity, ambiguous	Moyen — perdu si round-trip Rust
Training ML	Hyperedges, arêtes asymétriques	Moyen — jamais supervisées
Export DOT	IDs, scopes, modifiers, temporel, provenance	Élevé — visuel seulement
Résultats requêtes	third_node_label toujours None	Faible
Verdict honnête
Le pipeline fonctionne pour les cas simples (texte FR/EN avec connecteurs explicites, code Python bien formé, STIX avec relations causales). Mais :
1. Le bridge ML est le maillon faible : 14 dimensions nulles, morphologie perdue, négation perdue — le ML travaille sur des features dégradées
2. Le frontend code perd sa sémantique : DataFlow, ControlFlow, side effects définis mais jamais capturés
3. Incohérences FR/EN : ordre des stratégies d'annotation différent, modélisation temporelle asymétrique
4. Relations ternaires : supportées dans l'IR Rust mais jamais émises par aucun frontend, jamais supervisées en ML
5. Code mort significatif : InferenceEngine, CodeAttributes, gcn-veralizer, chain_strict — la moitié de l'IR n'est pas utilisée
6. 4 sémantiques de temporal_index : un même index veut dire 4 choses différentes selon le chemin
Le moteur est un excellent raisonneur symbolique sur des graphes propres, mais la chaîne d'approvisionnement en graphes (frontends + bridge) est le point faible — particulièrement le bridge ML qui dégrade les features avant même le premier calcul.
