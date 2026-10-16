# Plan raisonnement R1 — `reason()` pure et bornée (moteur)

> Statut : plan v3 FIGÉ (5 amendements + 5 résolutions de gaps intégrés,
> audit de vérification contre le code : 9/10 points relus en session,
> 1 confirmé au build — fixtures).
> Ligne conforme ETUDE_SA.md : moteur = bibliothèque (extraire, raisonner
> causalement, valider), boucle/outils/directives = projet SA séparé
> (hors workspace, dépendance à sens unique versionnée).
> R1 en parallèle du dataset (décision 2026-10-10).

## 1. Périmètre et interdits

- Neuf seul : `gcn_python/reasoning/` (`steps.py`, `engine.py`, `trace.py`).
  Python seul, Rust inchangé, zéro toucher à l'existant.
- `reason()` = fonction pure : entrée (but, graphe), sortie (réponse + trace).
  Lecture graphe seule ; aucun I/O, aucun outil/adaptateur, aucune persistance,
  aucune directive. Les `derived` sont retournés dans la trace, jamais ingérés
  (l'ingestion relève du SA et de ses directives).
- R1 : un but = une paire (from, to). Multi-buts → `Degrade` motif `multigoal:`.
- Échelle : graphes ≤ caps session (500 nœuds / 2000 arêtes) ; au-delà →
  `Degrade` motif `oversize:`. Perf gros graphes = sujet R3/index.

## 2. Signatures figées

- `reason(kw_from, kw_to, graph, budget=16) -> ReasonResponse`
  `{ version: "reason/1.0", status: ok|degrade, answer, trace }`
  (versionnement + test « clé future ignorée », modèle `EXCHANGE_SCHEMA_VERSION`).
- `reason_question(text, graph, ...)`: extrait `(kw_from, kw_to)` via
  `parse_command` / `DialogueSession.resolve` existants — aucun nouveau parseur.
- But vide/non résolu → `Degrade` immédiat (pas de boucle).

## 3. Boucle et étapes (codes D-AGNOSTIC, aucun nouveau code réponse)

`goal:` → `subgoal:` (≤3, prémisses manquantes) → `query:` (top-k=5 sur tri
level/conf existant, `min_level=2` forcé, warnings substring filtrés) →
`observe:` (relation de chaque maillon notée `rel:`, **traversée toutes
relations en R1, filtre causal en R2 sur données de traces**) → `infer:`
(transitivité A→B ∧ B→C ⇒ A→C `derived`, conf = min sans seuil, prémisses
citées + `origin:`, cf. ETUDE_SA §5.4 ; seuil `min_premise_conf` en R2) →
`verify:` (prémisses présentes ? contradiction = arête même (src,dst) avec
`negated=True` [Ph1]) → `answer:` | `abandoned:` (code existant réutilisé).
Échec `verify:` → backtrack au dernier point de choix. Cycles sur états gelés
(`_freeze` canonique). Budget : 1 `query:`/`infer:` = 1 étape ; épuisement →
`Degrade`. Graphe vide → `answer:` négative sourcée (convention `chain:`).

Référence d'algorithme :
```
reason(kw_from, kw_to, graph, budget=16):
  trace = [goal: (kw_from → kw_to)]
  path = graph.find_path(kw_from, kw_to)          # observe direct
  if path et verify_path(path): return answer(path)
  subgoals = maillons manquants (≤3)              # points de choix
  visited = {freeze(état)}
  for sg in subgoals:
    cands = graph.find_causes(sg.to)[:5] + graph.find_effects(sg.from)[:5]
    for c in cands:                               # 1 query + 1 infer = 2 étapes
      étape! si budget épuisé → degrade
      d = infer_transitive(c, sg)                 # conf = min, prémisses citées
      if verify(d): return answer(chaîne complète)
      trace += abandoned: <motif>
  return degrade (trace complète conservée)
```

Schéma d'étape (trace JSONL) : `{ code, tour, prémisses: [ids], conf?, motif? }`.

## 4. Primitives réutilisées (pures, vérifiées en session)

`find_causes` / `find_effects` (copient les attrs), `find_path` (BFS trié,
seed-agnostique), `_chain` (format-only). Aucun nouveau chemin de requête.

## 5. Tests (`test_r1_reason.py`, ~10 cas)

Chaînes 2-3 sauts + distracteurs ; fausse piste imposée → backtrack réel ;
budget épuisé → `Degrade` ; `multigoal:`/`oversize:` couverts ; property 0
`infer:` sans prémisse ; déterminisme 2 runs ; non-mutation du graphe
(snapshot avant/après).

## 6. Gate R1

10/10 + property + 855 Python / 258 Rust verts → commit ; sinon STOP avec
sortie écrite (règle STOP GATES.md). Constantes top-k/budget recalibrées sur
bench R3 (métrique étapes médianes).

## 7. Dettes R2 actées (non bloquantes) — R2 livré partiel 2026-10-11

Livré : filtre causal strict (groupe §15 `causale` via `COARSE_RELATION_GROUPS`,
`None` accepté, opt-out `causal_only=False`), `min_premise_conf` (défaut 0.0 =
R1, compagnons pré-filtrés, nié jugé par `verify:`), contradictions opposées
(`NEGATION_PREVENT_MAP` §9.4 les deux sens + `opposition`, scan tous doublons).
Reportés R3 avec motif : substitution (pas de données d'alias), marqueurs `?`
(schéma versionné — changement de contrat), bench mini-gold + recalibration.
Côté SA (projet séparé) : outil `reason_complex`, adaptateur `calculator`,
quarantaine des résultats.
