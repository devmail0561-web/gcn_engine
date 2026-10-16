# Plan — Durcissement `gcn-backend`

> Décisions [D1][D2][D3] figées le 2026-10-07. Périmètre : `gcn-core/crates/gcn-backend` + tests + README crate.
> Hors périmètre : moteur Python, dataset, frontends, CLI (hors thin wiring), write-back vers le train (refusé).
> Invariants : lecture seule du CIR, déterminisme (tri par label à égalité), garde SPOF 500 nœuds, codes d'erreur existants inchangés.
>
> Décision [M1] figée le 2026-10-07 : Python passe par Rust. Chemin cible unique :
> Python (état du dialogue + moteur ML) → CIR JSON → `gcn query` → Rust
> (middleend + backend + enveloppe always-answer) → UserResponse JSON → Python affiche.
> Le raisonnement Python existant (raisonneur d'instructions, rapports de requêtes,
> décodeur à règles côté discussion) est supprimé après bascule ; le routage des
> 3 actes et l'état de dialogue (sujets, clarification, langue — Phase 9) restent en
> Python ; le Rust reste sans état entre requêtes. Pont temps 1 = sous-processus
> `gcn query` avec CIR en fichier (format d'échange verrouillé : CIR JSON actuel +
> UserResponse Phase 8, test de roundtrip) ; temps 2 = liaison native seulement si
> les mesures de la Phase 10 l'exigent. Le contrat d'échange est versionné des deux
> côtés, avec test de compatibilité en CI.
>
> Note nommage (2026-10-08) : le « P0 » du présent plan (Phases 1-4) est distinct du P0 historique « audit adversarial » dont les marqueurs `P0-1…P0-5` parsèment déjà le code (agrégation des matchs, `chain_strict`, parse reports, partiels, alias aveugles).

## Phase 1 — Négation bloqueuse + tiers [D1] (~0.5j)

Sémantique figée : arête niée = propagation COUPÉE (pas d'inversion de signe) ; chemin traversant une niée = exclu + listé dans `blocked_by`. `prevent` = relation positive, se propage normalement. Inversion éventuelle = future requête INVERT, jamais le défaut.

Travail :
- 1a. `pearl.rs` — `bfs_ancestors`, `bfs_descendants`, `bfs_path_links`, `intervene` (effects), `counterfactual` (actual + résiduel) : sauter les arêtes `negated=true`, collecter les sautées pour `blocked_by`.
- 1b. Tiers Éq.12 : `chain_strict` (`main.rs:383` + pearl) exige la présence du `third_node` sur le chemin ; WHY/WHAT annotent le lien du tiers.
- 1c. `DO(X)` : severed inclut les entrantes niées (coupées 2x = coupées) ; effects exclut tout chemin à arête niée.
- 1d. `query.rs` — champ `blocked_by: Vec<LinkDto>` (`#[serde(default)]`) sur Path/Causes/Effects/Intervention/CounterfactualDiff ; CHAIN sans chemin à cause de négation → `found:false` + `blocked_by` rempli (vs NoPath vide quand vraiment aucun chemin).

Acceptation : jouet A→B nié, B→C : WHY(C) sans A ; CHAIN(A→C) `found:false` + `blocked_by=[A→B]` ; `DO(B).effects` sans A ; prevent A→B : WHY(non-B) contient A.

## Phase 2 — Résolution exacte-puis-floue + garde ambiguïté [D2] (~0.5j)

Sémantique figée : exact (score 0) toujours répondu ; flou (préfixe/contient, puis similarité injectable) seulement si 0 exact ; `AMBIGUITY_LIMIT` défaut 5 compté sur FLOUS seuls ; dépassement → erreur `Ambiguous` typée (pas de concaténation silencieuse) ; limite paramétrable en requête.

Travail :
- 2a. `pearl.rs` — `match_score` inchangé ; `find_best`/`find_all_ranked` : séparer exacts vs flous ; nouveau trait `LabelSim` (cosinus sur embeddings fournis par l'appelant, défaut = normalisé actuel) ; backend sans dépendance ML (reste pur Rust).
- 2b. `query.rs` — parse `AMBIGUITY_LIMIT=N` (défaut 5) ; Why/What : si exacts → répondre (tous, + scores) ; sinon si flous ≤ N → répondre + warning de flou ; sinon → `Err(Ambiguous{candidates})`.
- 2c. DTO : `match_scores: Vec<(label, score)>` (serde default) sur Causes/Effects ; `target`/`source` = meilleur exact.

Acceptation : {vent, ventes, ventilation} : WHY("vent") → "vent" seul ; requête floue 7 matchs → Err Ambiguous + 7 candidats ; exacts multiples (6× "ventes") → réponse normale, quota non déclenché.

## Phase 3 — Ranking Dijkstra + TOPK [D3] (~0.5j)

Sémantique figée : CHAIN = Dijkstra confiance-max (coût −ln(conf), conf<=0 → epsilon) ; TOPK défaut k=3 paramétrable ; `path_confidence` (produit) + `rank` exposés ; WHY/WHAT triés par force (produit depuis la cible) ; poids ANALOGY (0.40/0.30/0.20/0.10) + seuil 0.3 → paramètres `WEIGHTS=`/`MIN_SCORE=` (défauts actuels, documentés README).

Travail :
- 3a. `pearl.rs` — `chain()` → Dijkstra + k-chemins ; tris WHY/WHAT/EXPLAIN.
- 3b. `analogy.rs` — poids/seuil en arguments (défauts = actuels).
- 3c. `query.rs` — parse `TOPK=`/`WEIGHTS=`/`MIN_SCORE=` ; Path DTO + liens des k chemins ordonnés (+ `ranked_paths` si TOPK>1).

Acceptation : 2 chemins A→B (0.9 direct vs 0.5·0.5) → rank1 = direct, `path_confidence=0.9` ; TOPK=2 → les deux ordonnés.

## Phase 4 — Verrouillage tests (~0.5j, en filigrane des phases 1-3)

- 4a. `tests/test_query_gold.rs` (NOUVEAU) : diamant, chaîne niée, joint A₁∧A₂→B, cycle, temporel désordonné, prevent — résultats attendus en dur incluant `blocked_by`, `match_scores`, `path_confidence`, ranks.
- 4b. Property : déterminisme (2 runs = mêmes octets), monotonie (couper ≯ effets), joint-nécessité (1 co-source retirée = joint invalide), négation-bloque, temporel-strict-rejette, ambiguïté (7 flous → Err).
- 4c. Fuzz `Query::parse` (vide, unicode, "->", casse) : erreur typée, jamais panic.
- 4d. Existant conservé : `integration_backend.rs`, `test_ternary_pearl.rs`.

GATE P0 : `cargo test -p gcn-backend` + `cargo clippy -D warnings` verts + snapshot JSON stable → P0 LIVRÉ (~2j cumulés).

## Phase 5 — GATE calibration (~0.5j d'analyse, zéro code backend)

Mesurer côté Python (APRÈS slice GO/NO-GO moteur) : ECE + monotonie des confidences sur gold. SI acceptable → calibrer (isotonic) PUIS Phase 6a. SINON → enterrer la quantification, basculer Phase 6b. Livrable : "ECE=…, monotone oui/non → GO/NO-GO P1".

## Phase 6a — Quantification Pearl (3-5j, SI gate GO)

C1 Noisy-OR (indépendantes) / AND (jointes) → P calculées. C2 DO quantifié : P(Y|do(X)) + Δ par effet. C3 CF gradué : nécessité = 1−P(effet|sans X). C4 abduction bayésienne (ancre `confirmations`, `pearl.rs:423`). Ordre strict C1→C2→C3→C4 + dérivations exposées.

## Phase 6b — Produit sans chiffres faux (2-3j, SI gate NO-GO ou en //)

ASK, TOPK causes/effets, WHY-NOT (réutilise `blocked_by`), EXPLAIN contrastif, ANALOGY multi-sauts, dérivations D3 systématiques.

## Phase 7 — Gisement CIR : exploiter ce que le backend ignore (~2.5-3.5j, après P0)

> Décisions [7-i][7-ii][7-iii] figées le 2026-10-07. 7c seul conditionné au gate calibration ; le reste = logique/signalement, pas quantification.

### 7a. Gaps causaux + cycles typés : étendre [D1] (~0.5j)

Concession/opposition (`signals_causal_gap`, `edge.rs:36`) = BLOQUEUSES comme les niées [D1] (même `blocked_by`, motif distinct `causal_gap`). FeedbackNegative traversé = signalé `damped` ; FeedbackPositive = `amplifying` ; Oscillation = CHAIN_T exige la phase. Nouvelle requête CYCLES_TYPED ou extension CycleDto (type + nœuds + effet).

Acceptation : chaîne via opposition → `found:false` + `blocked_by` motif `causal_gap` ; boucle feedback-negative : WHY(effet) annote le lien `damped`.

### 7b. Origine + unresolved : l'honnêteté dans les réponses [7-ii] (~0.5j)

Poids d'origine FIGÉS : Explicit=1.0 > Inferred=0.7 > Hypothetical=0.3 (paramétrables, documentés), multipliés dans le tri WHY/WHAT/CHAIN ; `origin` exposé dans LinkDto. Caveat SYSTÉMATIQUE dès qu'un chemin gagnant contient ≥1 nœud Hypothetical. `caveats: Vec<Caveat>` (serde default) sur tous les résultats : pour chaque nœud du chemin porteur d'Ambiguity → {node, field, candidates, hint}. Nouvelle requête DOUBTS (liste brute des unresolved).

Acceptation : 2 chemins (attesté 0.6 vs hypothétique 0.9·0.3=0.27) → rank1 = attesté ; WHY sur nœud ambigu → caveats rempli avec hint du moteur.

### 7c. Modifiers quantitatifs (~1j, SEULEMENT si gate GO)

Probability(p) × confidence par produit (indépendance assumée ET documentée) ; Frequency : Always=1.0/Often=0.75/Sometimes=0.5/Rarely=0.25/Never=BLOQUEUR (comme niée) ; Negation{total=true}=BLOQUEUR (cohérent [D1]), total=false=atténuateur 0.5 ; Intensity/Quantitative exposés (tri secondaire), jamais multipliés sans gate. Manner/Location/IntrinsicProperty/TemporaryState/Relational/TemporalQuality/Temporality : filtres de requête (WITH/WHERE) + affichés, pas de sémantique inventée.

Acceptation : Frequency=Never sur arête → même effet que niée ; Probability combinée par produit documenté ; filtres Manner/Location testés.

### 7d. Agents + réversibilité : DO responsable [7-iii] (~0.5j)

`reversible=false` sur cible de DO → warning FIGÉ `irreversible`, effects calculés quand même (pas de blocage : le DO sur irréversible est informatif, surtout en analyse de risque). `agent_type` exposé + filtre (`DO ... AGENT=human`) ; EXPLAIN scinde hypothèses par agent_type ; patient/agent affichés dans LinkDto.

Acceptation : DO sur nœud irréversible → warning présent, effects calculés ; EXPLAIN groupé par agent_type.

### 7e. Scope : quantificateurs [7-i] (~0.5j)

Règle FIGÉE : Universal = propage normalement ; Existential/Specific = WHY annote `witness_only` (généralisation interdite) ; Null = BLOQUEUR [7-i] (motif `null_scope`, principe unique : pas de référent, pas de propagation) ; Unknown/Partial = caveat (comme 7b). Scope exposé dans les DTO + filtre WHERE scope=universal.

Acceptation : arête scope=null traversée → `blocked_by` motif `null_scope` ; existential → lien annoté `witness_only`, jamais généralisé en agrégat.

### 7f. Provenance + dérivation + ancrage : requêtes de confiance (~0.5j)

LinkDto + : `explicit`, `extraction_method`, `model_version`, `span` (retour texte/code), `kind` (code). Filtres : ONLY_EXPLICIT, METHOD=silver/ml/symbolic, SINCE=date. `derivation.source_edges` → `derived_from` dans la réponse (= D3 alimentée par le CIR). Nouvelle requête PROVENANCE(X) : fiche complète (qui, quand, comment, span texte).

Acceptation : ONLY_EXPLICIT élimine tout lien inféré ; PROVENANCE retourne méthode+modèle+span ; `derived_from` == source_edges du CIR.

Ordre : 7a → 7b → 7f → 7d → 7e → 7c (SI gate GO). Chaque item : gold tests + property (suites Phase 4) ; aucun changement de contrat sans serde(default).

## Risques → réponses

- DTO change casse CLI/Python → `#[serde(default)]` + snapshot + version mineure crate.
- Dijkstra change des réponses → voulu, gold archivé ("legacy_documented").
- P1 sur mauvais socle → gate APRÈS slice moteur, jamais avant.
- Boucle auto-label → aucun write-back : candidats + dérivation, validation externe seule (humain/teacher S4-opt), jamais gold direct.

## Engagement

À valider : 2j (Phases 1-4). Le reste est optionnel (gate 0.5j, 6a 3-5j SI GO, 6b 2-3j, Phase 7 ~2.5j sans 7c / ~3.5j avec). Pour exécuter : Phase 1.

## Phase 8 — Toujours répondre, chaîne complète + phrase verbalisée (~2j, parallélisable avec P0)

> [R-ALWAYS] Tout point d'entrée UTILISATEUR (CLI, API) retourne TOUJOURS : {phrase NL 1-3 phrases + contenu structuré même partiel + statut ok|partiel|degrade + caveats + suggestions}. Les Err typées restent en INTERNE, converties à la frontière. Batch/CI : contrat inchangé (raise R3, exit non-zéro).
> [R-LANG] Aucun paramètre de langue : la langue de réponse est COMPRISE (déduite de l'intention, `cir.intent.lang`), pas configurée. Défaut français ; message utilisateur en wolof → réponse wolof ; demande explicite (« réponds en wolof ») → bascule. Données (labels, spans) gardent leur langue d'origine.

- 8a. Nouveau `respond(query, ir) -> UserResponse` (query.rs) enveloppant `execute()` : NodeNotFound → degrade + candidats flous (match_scores [D2]) + reformulation ; Ambiguous [D2] → partiel + candidats classés + question (l'Err interne subsiste, la frontière la convertit) ; found:false/NoPath → partiel + proches + WHY-NOT ; SPOF>500 → partiel + CENTRALITY top-k + annonce ; QueryParseError → degrade + aide + « vouliez-vous… ».
- 8b. `execute()` inchangé (Err internes, tests intacts) ; gold tests assertent sur UserResponse (phrase non-vide + statut).
- 8c. Verbalizer de secours déterministe (gcn-verbalizer : `say(status, lang, slots)`, pur Rust, 0 dépendance, testé sans Python) : ~12 templates FR d'abord (introuvable, ambigu, sans-chemin, trop-gros, requête-inconnue, cir-vide, irréversible-info, hypothétique-avertit, incident, aide…), autres langues à la demande ; voie noble = décodeur appris quand il fonctionne. Règle : la phrase de secours ne MENT jamais (constat + miettes réelles + suggestions).
- 8d. Property « always-answer » : chaque requête × entrée adverse (vide, unicode, inconnu, ambigu×7, graphe vide, SPOF>500 simulé, décodeur absent simulé) → réponse non-vide + statut ∈ {ok,partiel,degrade} + phrase non-vide. Jamais d'Err nue ni de None à la frontière.

## Phase 9 — Mémoire conversationnelle (~3-4j)

> But : le système enchaîne les échanges sans perdre le fil. Chaque tour se souvient du sujet en cours, des questions restées ouvertes et de la langue utilisée. Décisions figées : (9-1) les questions posées par l'utilisateur enrichissent le graphe mais avec un poids réduit et une étiquette d'origine, et une question ne pose jamais un fait ; (9-2) une seule question de clarification en attente à la fois, abandonnée après 3 échanges avec annonce.

### 9a. Marquer l'origine de chaque information (0.5-1j, socle du reste)

Situation actuelle : tout ce qui entre dans le graphe est mélangé. Une question comme « le stress cause l'insomnie ? » ajoute l'arête stress → insomnie au même niveau que les faits issus des documents. Les questions polluent le savoir et les réponses suivantes s'appuient dessus comme sur des faits.

Changement prévu : chaque arête reçoit une étiquette d'origine (document, affirmation de l'utilisateur, question de l'utilisateur) et un poids de confiance par origine : document = 1.0, affirmation = 0.8, question = 0.5. Les tris des réponses multiplient force du lien par poids d'origine, et chaque réponse indique d'où vient l'information (« d'après le document X », « d'après la question posée plus tôt »). Règle : une question ne pose jamais un fait — ses arêtes servent à comprendre la question, pas à répondre aux suivantes.

Exemple concret : l'utilisateur demande « le stress cause l'insomnie ? » puis « pourquoi l'insomnie ? ». Si la seule cause connue vient de la question précédente, la réponse dit : « la seule cause connue vient de votre question précédente, aucun document ne le confirme », au lieu d'affirmer le lien comme un fait.

Vérification : cas monté où un chemin documenté de force 0.6 doit gagner contre un chemin issu d'une question de force 0.9 × 0.5 = 0.45 ; l'origine est visible dans la réponse.

### 9b. Suivre le sujet de la conversation (1j)

Situation actuelle : le système prend le dernier mot de la question comme sujet cherché. « Et ses effets ? » fait chercher le mot « effets » dans le vide et répond qu'il n'a rien trouvé. Les pronoms (il, ses, ça, celui-ci) et les désignations (« le premier ») ne sont pas résolus.

Changement prévu : le système retient les 3 derniers sujets abordés. Avant chaque recherche, un résolveur relie les pronoms et démonstratifs au sujet en cours, et les ordinaux (« le premier », « le 2 ») à la dernière liste de candidats proposée. La méthode du dernier mot n'est gardée qu'en dernier recours. Le sujet compris est toujours annoncé dans la réponse (« compris : les effets de l'insomnie ») pour permettre la correction.

Exemple concret : après « le stress cause l'insomnie », la question « et ses effets ? » donne la liste des effets de l'insomnie, avec mention du sujet compris. Après une liste A, B, C, « le premier » sélectionne A.

Vérification : scénarios de dialogue écrits d'avance (3 à 4 échanges avec pronoms et numéros), réponses attendues fixées ; le test échoue si le sujet est mal résolu.

### 9c. Finir les questions commencées (0.5j)

Situation actuelle : quand le système demande « laquelle : A ou B ? », la réponse suivante repart de zéro — le choix proposé est oublié.

Changement prévu : une seule question en attente à la fois. Le message suivant est d'abord interprété comme une réponse à ce choix (numéro, répétition du nom, oui/non) avant tout autre traitement. La question en attente se consomme en un tour ; après 3 échanges sans réponse, elle est déclarée abandonnée. Si l'utilisateur change de sujet, l'abandon est annoncé (« je laisse de côté le choix A/B »). Jamais de file d'attente de clarifications.

Exemple concret : le système demande « laquelle : A ou B ? », l'utilisateur répond « la première » → la réponse porte sur A. Après 3 échanges sans sélection, le système annonce que le choix est abandonné.

Vérification : tests de propriété — jamais deux questions en attente, jamais de question fantôme après 3 tours ; scénarios en dur pour la sélection par numéro et par nom.

### 9d. Retenir la langue de l'échange (0.25j)

Situation actuelle : rien ne retient la langue utilisée — chaque message est traité isolément.

Changement prévu : la session retient la langue du dernier message (détection par message, français par défaut initial). Une consigne explicite (« réponds en wolof ») est prioritaire et persiste jusqu'à contre-ordre. Ordre de priorité figé : consigne explicite, puis langue du message en cours, puis langue du fil, puis français. La phrase de secours de la phase 8 utilise cette langue.

Exemple concret : fil en français, consigne « réponds en wolof » → les réponses suivantes sont en wolof jusqu'à contre-ordre. Un message isolé en wolof reçoit une réponse en wolof immédiatement.

Vérification : scénarios bilingues — bascule sur consigne, bascule sur message isolé, persistance de la consigne, retour sur contre-ordre.

### 9e. Journal utile et droit à la correction (1j, avec M2)

Situation actuelle : le journal des échanges est écrit mais jamais relu par le système ; les réponses données n'y figurent même pas. Il n'existe aucun moyen de corriger une information apportée par l'utilisateur.

Changement prévu : chaque tour stocke la question, le sujet résolu, un résumé d'une ligne de la réponse, le statut et le numéro du tour. Le résolveur du point 9b consulte les 3 derniers tours avant d'interroger le graphe. Au-delà d'une limite de taille, les vieux échanges sont résumés automatiquement (avec la phase 10). Une correction (« non, je parlais d'autre chose ») retire ou requalifie les ajouts de l'utilisateur du tour visé — jamais ceux du corpus — et l'opération est annoncée.

Exemple concret : après une correction, les arêtes apportées par l'utilisateur sont retirées, le corpus reste intact, et la réponse suivante utilise le bon sujet.

Vérification : test de révocation — seules les arêtes d'origine utilisateur sont effacées ; test de reprise — une référence à un sujet vieux de 2 tours est résolue via le journal.

## Phase 10 — Mémoire, RAM et limites (~4-6j)

> But : arrêter le gaspillage de calcul et de mémoire, borner la session avec rangement annoncé au lieu de refus brutal. Principe : construire une seule fois ce qui est partagé, ne pas recalculer l'identique, borner tôt, optimiser la mémoire vive en dernier (le temps est le vrai goulot, pas la RAM).

### 10-M4. Un seul calcul de graphe par requête (0.5j, en premier, risque faible)

Situation actuelle : pour une seule requête, le backend reconstruit le même graphe interne 7 fois (une fois par fonction d'analyse : causes, effets, chaîne, intervention, contrefactuel, etc.).

Changement prévu : le graphe interne est construit une seule fois à l'arrivée de la requête, dans un contexte partagé passé à toutes les fonctions d'analyse. Aucun changement de résultat — c'est une réorganisation mécanique. Sans risque car le graphe ne change pas pendant une requête (le backend est en lecture seule).

Vérification : les tests existants passent sans modification des résultats attendus ; mesure du temps — une requête multi-opérations doit prendre 3 à 4 fois moins de temps.

### 10-M1. Ne pas recalculer les mots déjà vus (1-2j, le vrai goulot du temps)

Situation actuelle : les vecteurs de mots sont recalculés à chaque requête, même pour des phrases déjà vues ; les requêtes sont traitées une par une.

Changement prévu : (a) un cache des vecteurs de mots entre requêtes, avec clé incluant la version du modèle et du tokenizer — si le modèle change, le cache est invalidé ; jamais de cache sur les graphes eux-mêmes ; (b) traitement des phrases par lots au lieu d'une par une ; (c) sélection des meilleures paires avant tri complet.

Vérification : une phrase répétée ne déclenche aucun nouveau calcul la seconde fois ; changement de modèle prouvé par un test d'invalidation ; mesures de temps avant/après chiffrées.

### 10-M2. Borner la session : compacter au lieu de refuser (2-3j)

Situation actuelle : la session grossit sans limite (graphe, vecteurs, journal) ; la sauvegarde réécrit tout à chaque fois ; une paraphrase (« fatigue » vs « être fatigué ») n'est jamais reconnue comme identique.

Changement prévu : (a) des plafonds configurables sur la taille du graphe et de l'historique — au dépassement, rangement automatique et ANNONCÉ (fusion des doublons, résumé des vieux échanges, avertissement dans la réponse), jamais de refus ; (b) sauvegarde incrémentale — journal des ajouts plus photo périodique, au lieu de tout réécrire ; (c) reconnaissance des paraphrases par similarité, avec seuil et repli sur la correspondance exacte.

Exemple concret : au 501e nœud, au lieu d'une erreur « graphe trop gros », le système répond : « j'ai rangé les vieux échanges pour faire de la place, voici la réponse… ».

Vérification : dépassement de plafond → rangement annoncé, zéro refus ; sauvegarde plus rapide (mesuré) ; paraphrase proche retrouvée.

### 10-M3. Alléger la mémoire vive (1j, en dernier, avec garde-fou chiffré)

Situation actuelle : tout est stocké en pleine précision, le fichier de vecteurs est chargé en entier en mémoire.

Changement prévu : (a) chargement paginé du fichier de vecteurs (pages à l'usage) ; (b) précision réduite pour la phase de calcul des vecteurs uniquement, avec mesure de l'écart de qualité sur les tests de référence — si l'écart dépasse le seuil, retour en arrière acté ; (c) zones de calcul pré-allouées à la taille maximale des lots. En dernier car la RAM se compte en dizaines de Mo, pas en Go : ce n'est pas là que ça fait mal.

Vérification : écart de qualité sous le seuil et empreinte mémoire en baisse, mesurés ; sinon retour en arrière acté et documenté.

## Ordre d'exécution des phases 9 et 10, effort, paliers

Ordre : M4, puis 9a, 9b, 9c, 9d, puis M1, puis M2a, puis 9e avec M2b/c, puis M3.

Premier palier utilisable (~2.5j) : M4 + 9a + 9b + 9c — le système dialogue sans gaspiller le calcul de graphe. Deuxième palier (~2j de plus) : M1 + M2a — requêtes et sessions assainies. Le reste (9e complet, M2b/c, M3) est optionnel selon les mesures. Total estimé : 7 à 10 jours.

Tests : dialogues multi-tours écrits d'avance pour la phase 9 ; mesures de temps et de mémoire avant/après chiffrées pour la phase 10 ; tests de propriété (une seule clarification en attente, une question ne pose jamais un fait, la révocation ne touche que les apports utilisateur, invalidation du cache au changement de modèle).
