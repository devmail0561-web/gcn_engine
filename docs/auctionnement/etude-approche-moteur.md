# Étude — approche actuelle du moteur : problèmes, limites, blocages

Date : 2026-10-14
Branche vérifiée : `refacto/arch-cible-refacto-2026-10` (`2778fb4`), projet entier 3,6 Go
Mode : lecture codebase + docs, sans supposition. Chaque fait porte son `fichier:ligne`.

## 1. Finalité réelle : l'usage, pas le graphe

- Moteur = bibliothèque 3 actes seuls (`ETUDE_SA:50-54`) : extraire CIR, raisonner Pearl 1-3/BFS/CYCLES/GAPS, valider pendantes/réflexives/orphelins. Sans état, sans boucle, sans opinion sources (`ETUDE_SA:40-44`). Interdits (`ETUDE_SA:115-121`) : boucle ingestion, décroissance dans graphe, API HTTP, directives dans moteur.
- Borné (`ETUDE_SA:129-131` + `README:27-32`) : assertions sourcées, pas faits. DO = coupe + propage, counterfactual = monde-sans-X cité. Sans confondeurs, pas do-calculus identifié.
- 4 rôles Evidra (`ETUDE_SA:241-260`) : Accumulateur seul écrit, Raisonneur avec preuves, Validateur `CONFIRMED/UNCONFIRMED/CONTRADICTED` table fermée, Moniteur alertes. 12 outils (`ETUDE_SA:67-81`), 6 priorités séquentielles (`ETUDE_SA:82-100`).
- 4 sorties même raisonnement (`ETUDE_SA:645-650`) : LLM JSON compact, humain verbalisé + citations, agent CIR protocole, alerte payload. Service 9 endpoints (`ETUDE_SA:627-639`). Offline noyau 100% (`ETUDE_SA:695-697`), test dogfood code réseau coupé.
- Enveloppe (`respond.rs:26-45`) : `UserResponse{status ok/partiel/degrade, phrase 1-3, result, caveats, suggestions}` jamais Err (`respond.rs:312-330`), `respond_validated:336-350` dégrade sur Error middleend. Verbalizer pont subprocess (`verbalizer/lib:24-29`) + fallback déterministe jamais mensonger (`fallback:4-9`).
- Échec d'usage acté (`ROADMAP:24-30` + `ETUDE_SA:1166-1172`) : Ph5 NO-GO → conf provisoires, checkpoint non re-mesuré → rien consommé, dataset 3M à 5% 50-95j, MVP 1000 docs / 200 claims / 50 chaînes 24h gate P3. Métriques SA non mesurées SA inexistant (`ETUDE_SA:970-989`).

Le graphe est le pivot, pas le produit. Succès = claims validés + chaînes surveillées + alertes justes, pas graphe dumpé.

## 2. Périmètre moteur vs outils

Moteur strict :
- `gcn-ir ~1178` : CIR, 8 NodeType, 19 RelationType, Provenance, Derivation
- `gcn-middleend ~416` : `graph.rs:35` DiGraph, `cycle.rs:74` Tarjan, propagate/validate
- `gcn-backend ~5500` : `pearl.rs:1616` + `query.rs:2473` + `respond.rs:999`, 21 requêtes GCN-QL (`README:270-295`)
- `gcn-python apprentissage` : `layer2/reference:417` MLP + `layer3` R-GCN/GAT + `pipeline/cgnp:2295` + `forward_ids:594` chemin tok + `nlu/tok:106` sans UD
- `gcn-transformers` option protocole `CausalEncoder`, jamais dépendance inverse (`Makefile:25`, CI guards). 0 run (`USAGE:7-12`, `BENCHMARK-README:239-240`, `FINAL-REPORT:42,125`, `STATUS:6`).

Hors-moteur câblé au runtime (à sortir) :
- 5 frontends 3123 lignes : FR `lattice.rs:640` + EN `527` sortie Lattice jamais CIR (`lib:6-19`), règles suffixes (`FR:90-204` / `EN:83-122`) + positions (`FR:281-297,304-468`), code `kinds:55-321`, table schéma imposé (`table:27-55`), STIX repli Concept (`mapper:14-64`) confiance 1.0 Synthetic (`stix:4-10`)
- `gcn-verbalizer:372` pont PATH + `gcn-cli:1119` 2 contrats `analyze`, `gcn-knowledge` fantôme absent (SAD caduque)
- `layer0:800` : `fallback_regex:34` SUPPORTED 6 langues, pt/it sans table (`fallback:20-22,37,215`), lecture fichiers scraper/annotate à l'import (`fallback:128-169`), snapshot minimal (`fallback:181-192`), `engine_b:46-62` pin spacy 3.8 timeout 30s, source silver-B/fallback obligatoire (`engine_b:40-42,247-262`)
- `pipeline/ir_emitter:20-53` RUNG R1 0.50/R2 0.75/R3 0.65 + MOOD + CONNECTOR verb 0.90, D6 rung×mood×connecteur (`ir_emitter:56-79`), THETA 0.65 (`constants:155`, vivant dans `cgnp + forward_ids + train`), grid inclut 0.9 (`grid_f1:72-73`)
- `layer1/features:75-90` d_clause 106 = 18+38+5+5+4+5+3+7+1+3+12+5, `data/loader:55-56` gold_node/edge_map seule vérité, `loader:157-266` raise inconnu, `train:1020-1074` gold filtré, `cgnp:1294-1343` raise désalignement
- `gcn-tools` annotate LLM `claude-sonnet-5 + gpt-4o` lazy + regex + tables CONN, scraper offline JSONL — offline sauf lecture tables à l'import fallback
- `gcn-datasets` schemas nl/pl/verbalize, `DATA/supervision` 332 fichiers 321k phrases 35k avec CIR 285k sans, train 536/6,9M val 114 test 117, scripts build/generate/merge/oversample anti-leak, gold 219 gold_mini 85 cible 3000
- `gcn-references` que fr/12 + en/11 YAML, zéro python/js. `loader:30-40` scan YAML, `test_pipeline:305-07` zéro YAML runtime. SAD obsolète.

## 3. Problèmes de conception

P1. Frontend par langue/type = impasse. Ajouter langue = nouveau crate. Cible T6 (1 tokenizer + profils script, langue=DONNÉES) non tenue en legacy.
P2. Annotation obligatoire. `train + loader + eval + gold_spans SpanError` exigent UD/lemme/dep_rel/spans gold. Sans gold, pas de loss. Inverse auto-supervisé Transformers.
P3. Moteur de règles déguisé. RUNG/MOOD/CONNECTOR/THETA/0.9/SUPPORTED/KEYWORDS/d_clause/WINDOW_K/discourse 0.50/act listes closes.
P4. Causal faux. `pearl:1339` intervene = filter + `bfs_descendants:748-753`, `pearl:1376,1414` counterfactual = BFS + `reachable_from_roots_without:1445-90`, set-diff `1423-28`. 0 backdoor/frontdoor. Cycles annotés jamais rejetés. ANALOGY 0.4/0.3/0.2/0.1 + ORIGIN 1.0/0.7/0.3 arbitraires.
P5. Double vérité. Legacy 106/297/1115 vs tok ids+d_model. `plan-inference §2` 8 blocs à supprimer non supprimés.
P6. Backend dépend sortie. `backend Cargo:11-13` dépend verbalizer PATH, `respond` always-answer masque erreur en dégradé.
P7. Middleend ampute. `graph:19-34` ignore pendantes avant diagnostic, conf>1 non contrôlée, tout mixte+concession = oscillation, seul gap auto différé, pas de fusion multi-phrases SAD:592-602.
P8. Normalisation casse preuve. `normalize:20-28` fold + lower + strip → collisions, perte `/()`.
P9. IR vide passe (`ir:18-22`), version absente tolérée.
P10. Docs mentent. SAD archivé 7/11 vs 8/19 réel, knowledge inexistant, taxos python/js inexistantes, BENCHMARK-RESULTS supprimé 2026-09-25, transformers 0 run, badges vs `lastfailed`.

## 4. Limites mesurées

- Val `0.239±0.04 (0.184@119, 0.265@36, 0.269@128)` (`README:633-638`, `CHANGELOG:13`), cible >0.40 facteur 1,7x (`README:642`), Δ gold+silver 0,01 = bruit (`README:647`). Train 0,98-0,99 (`README:635-637`), gap (`README:645,681`). Split 572/132 quota64 (`README:628-630`).
- OOV 23,7% (`CHANGELOG:29,66-67`) vs seuil 0,05 (`forward_ids:71-72,360-363`).
- ECE 0,33-0,55 (`CHANGELOG:270-271`, `GATES:39`, `ROADMAP:26`), N=81<100, quantification enterrée (`GATES:13`).
- 8 rels à 0 → gel R8/R10 (`README:684-686`, `CHANGELOG:106`, `GATES:34`, `ROADMAP:28`), <30/classe GEL exit 2 (`grid:10,109`, `GATES:14`).
- Tests : backend 67/67 4,2s (`baselines:12`, `CHANGELOG:103`, `GATES:33`), P0 102 (`GATES:36`), 3 snapshots (`CHANGELOG:136`). Badges 260/887 (`README:6-7,570,586`) vs 932/939/952 revendiqués (`CHANGELOG:30,49,72`). Transformers 31/10 skipped (`README:597`).
- Benchmarks : validation 30 fichiers, archive 17 runs 0,4676→0,1202 (`BENCHMARK:24-42,88`), re-runs 0,1615/0,2508, chute même val, données réécrites irrécupérables, ne jamais comparer (`BENCHMARK:309-347,361-365`), archivé (`README:650-652,707`), protocole SHA + seeds x3 + <0,05 bruit (`PROTOCOLE:1-15`).
- Ressources : CPU ms 5 Mo vs HF 110-280M 3-4 Go GPU. Offline noyau OK.

## 5. Blocages actés

- Gates : slice GO jouet 4 phrases, socle 0c GO 85, s0 GO, P0 GO, Ph8 livré, Ph5 NO-GO → 6b sans chiffres, silver 0→0,3 si IC95≤20%, J7 exige table 11→19, STOP si slice/Ph5+6b/J7 2 cycles (`GATES`).
- Roadmap : Ph5 NO-GO → provisoire, checkpoint non re-mesuré → rien consommé, dataset 3M 5% 50-95j, MVP gate P3 (`ROADMAP:24-30`).
- Collecte : T2 gold-219 Silver-B 219/219 97,7%, 8 rels à 0, kappa/campagne en attente.

## 6. Moteur vs Transformers — honnête

Voir `comparaison-moteur-vs-transformers.md` (vérifiée 91 lignes, fidèle).
Moteur ~5 Mo CPU graphe typé + spans + doute, interrogeable WHY/CHAIN/DO. Transformer 100M ~400 Mo GPU texte fluide sans source. Moteur gagne preuve/traçabilité/efficacité, transformer gagne fluence/multi-tâche/écosystème. Sans bench tête-à-tête mêmes 572 phrases même métrique usage, gagnant global = supposition.
Sur 50 tokens : moteur ~5 Mo + 50 Ko, transformer ~400 Mo + 1 Mo KV. ~80× plus léger, sortie différente : graphe prouvé vs texte plausible.

## 7. Pourquoi l'annotation obligatoire est fausse ici

Transformers : 1 entrée, 0 frontend, masqué sans gold, milliards tokens. Nous : UD/lemme/dep_rel/spans gold exigés, 1 frontend par langue, CE sur 572. Léger mais aveugle hors gold.

Voie qui tient : `BYTES+T → E unique → biaffine → calibration apprise`, <1M <10 Mo CPU, auto-supervisé ordre/span/gap/cycle, gold-300 éval seule. Jugée usage : 200 claims, 50 chaînes 24h, p99 <200ms validate / <50ms causes, offline réseau coupé, ECE<0,15 sinon UNCONFIRMED.
