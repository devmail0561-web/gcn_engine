# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..constants import NEGATION_PREVENT_MAP, NODE_ORIGIN_VALUES, TEMPORAL_REF_DEFAULT

# ---------------------------------------------------------------------------
# D6 — Formule de confiance unique (ETUDE_LINGUISTIQUE_NLU.md §9.0 / §12.4).
# Transcription à l'identique des priors gelés — ne pas modifier à la main ;
# remplacés par calibration isotonique dès N_val >= 100 (§9.5).
# ---------------------------------------------------------------------------

#: Priors de rung par relation (ETUDE §9.0).
#: R1=0.50 (CAUSE, SEQUENCE, ANALOGY), R2=0.75 (CONDITION, ENABLE, PREVENT,
#: FILTER, CONDITIONAL_CAUSE, MOTIVATION), R3=0.65 (COUNTERFACTUAL,
#: MEDIATED_CAUSE, JOINT_CAUSE, CONCESSION, OPPOSITION).
#: Extensions documentées (absentes de l'ETUDE, §DECISIONS_NON_COUVERTES) :
#: *_prevent → R2 (analogie PREVENT), data/control_dependency → R1
#: (dépendances systémiques observées, confiance modérée).
CONFIDENCE_RUNG: dict[str, float] = {
    "cause": 0.50, "sequence": 0.50, "analogy": 0.50,
    "data_dependency": 0.50, "control_dependency": 0.50,
    "condition": 0.75, "enable": 0.75, "prevent": 0.75,
    "filter": 0.75, "conditional_cause": 0.75, "motivation": 0.75,
    "counterfactual": 0.65, "mediated_cause": 0.65, "joint_cause": 0.65,
    "concession": 0.65, "opposition": 0.65,
    "conditional_prevent": 0.75, "mediated_prevent": 0.65, "joint_prevent": 0.65,
}

#: f(Mood) — ETUDE §9.0 : Ind=1.00 / Sub=0.85 / Cnd=0.80.
#: Mood Imp/absent/_absent → 1.00 (neutre, non spécifié par l'ETUDE).
MOOD_FACTOR: dict[str, float] = {"Ind": 1.00, "Sub": 0.85, "Cnd": 0.80}

#: g(connecteur) — ETUDE §9.0/§12.4 : explicite=1.00 / VERB_causal=0.90 /
#: parataxis=0.60 / aucun=0.50.
CONNECTOR_FACTOR: dict[str, float] = {
    "explicite": 1.00, "verb": 0.90, "parataxis": 0.60, "aucun": 0.50,
}


def confidence_d6(
    relation: str,
    mood: str = "Ind",
    connector: str = "explicite",
    salience_bonus: float = 0.0,
) -> float:
    """Prior de confiance D6 : rung × f(Mood) × g(connecteur) + Δ_saillance.

    Args:
        relation: type de relation (clé RELATION_TYPES).
        mood: valeur UD Mood ("Ind"/"Sub"/"Cnd", défaut "Ind").
        connector: "explicite" (marker_token présent), "verb", "parataxis"
            ou "aucun" (défaut explicite côté appelant selon marker_token).
        salience_bonus: +0.10 si advcl_before_root (condition proéminente),
            sinon 0.0. Le câblage positionnel complet est un follow-up ;
            l'appelant passe la valeur explicite.

    Sortie clampée dans [0, 1]. Relation inconnue → rung R2=0.75
    (choix documenté, jamais silencieux : pas de KeyError).
    """
    rung = CONFIDENCE_RUNG.get(relation, 0.75)
    f_mood = MOOD_FACTOR.get(mood, 1.00)
    g_conn = CONNECTOR_FACTOR.get(connector, 0.50)
    return min(1.0, max(0.0, rung * f_mood * g_conn + float(salience_bonus)))


def _clause_has_sconj(tokens: list[dict]) -> bool:
    """True si la clause porte une subordonnée SCONJ (mark/SCONJ)."""
    for t in tokens or []:
        if t.get("dep_rel") == "mark" or t.get("pos") == "SCONJ":
            return True
    return False


def orient_edge_d7(
    src: int,
    dst: int,
    relation: str,
    token_spans: list[tuple[int, int]] | None = None,
    tokens_by_clause: dict[int, list[dict]] | None = None,
) -> tuple[int, int]:
    """D7 — orientation causale invariante (ETUDE §D7, plan §DECISIONS_NON_COUVERTES).

    Règle : source = clause subordonnée SCONJ, sauf SEQUENCE où
    source = clause chronologiquement antérieure (position token).
    Sans métadonnées (token_spans/tokens_by_clause absents) → identité.
    Pure et testable indépendamment du pipeline.
    """
    if token_spans is None:
        return (src, dst)
    if relation == "sequence":
        if token_spans[src][0] <= token_spans[dst][0]:
            return (src, dst)
        return (dst, src)
    if tokens_by_clause:
        src_sub = _clause_has_sconj(tokens_by_clause.get(src, []))
        dst_sub = _clause_has_sconj(tokens_by_clause.get(dst, []))
        if dst_sub and not src_sub:
            return (dst, src)
    return (src, dst)


def apply_voice_eq7(
    src: int,
    dst: int,
    voice_by_node: dict[int, str] | None,
) -> tuple[int, int]:
    """Éq.7 — normalisation voix : si Voice=Pass sur la cible, inverser
    Agent/Patient dans (src, dst) pour que src soit l'Agent logique.
    Sans métadonnées voix → identité. Appliquer APRÈS D7 (plan §C.5)."""
    if voice_by_node and voice_by_node.get(dst) == "Pass":
        return (dst, src)
    return (src, dst)


def detect_ternary(
    relation: str,
    src_tokens: list[dict] | None = None,
    joint_signature: str | None = None,
) -> tuple[str | None, int | None, str | None]:
    """Détecteur ternaire minimal C.6 (heuristique UD, stockage seul).

    Returns:
        (third_role, third_node, joint_group_id) — chaque champ None
        si non détecté. `third_node` est résolu par l'appelant
        (index de clause) ; ici seul le rôle est détecté depuis les tokens.
        Phase E (supervision + groupement Pearl) reste à chiffrer.
    """
    if relation in ("conditional_cause", "conditional_prevent"):
        if src_tokens is not None and _clause_has_sconj(src_tokens):
            return ("condition", None, None)
        return (None, None, None)
    if relation in ("mediated_cause", "mediated_prevent"):
        if src_tokens is not None and any(
            t.get("dep_rel") == "obl" for t in src_tokens
        ):
            return ("mediator", None, None)
        return (None, None, None)
    if relation in ("joint_cause", "joint_prevent"):
        return (None, None, joint_signature)
    return (None, None, None)


def _infer_temporal_ref(rep) -> str:
    """Dérive temporal_ref depuis les features UD déjà calculées (S-1)."""
    tense = getattr(rep, 'tense', '_absent')
    if tense == 'Past':
        return 'past'
    if tense in ('Fut', 'Futur'):
        return 'future'
    if tense == 'Pres':
        return 'present'
    if getattr(rep, 'has_temporal_obl', False):
        return 'anchored'
    return TEMPORAL_REF_DEFAULT


def apply_negation_algebra(
    relation: str,
    negation_site: str | None,
    third: dict | None,
    negated: bool,
) -> tuple[str, dict | None, str | None]:
    """Applique l'algèbre de négation §9.4 ETUDE.

    Retourne (relation, third, source_polarity).
    R1 Neg(dst) : relation directe → variante prevent.
    R2 Neg(condition) : third["polarity"] = "negative".
    R3 Neg(src) : cause → counterfactual, joint_cause → source_polarity.
    """
    source_polarity = None
    if not negated:
        return relation, third, source_polarity
    if negation_site == "dst":
        relation = NEGATION_PREVENT_MAP.get(relation, relation)
    elif negation_site == "condition" and third is not None:
        third = {**third, "polarity": "negative"}
    elif negation_site == "src":
        if relation == "cause":
            relation = "counterfactual"
        elif relation == "joint_cause":
            source_polarity = "negative"
    return relation, third, source_polarity


def emit(
    text: str,
    node_types: list[str],
    node_labels: list[str],
    token_spans: list[tuple[int, int]],
    scopes: list[str],
    edge_triples: list[tuple],
    node_origins: list[str] | None = None,
    node_attributes: list[dict] | None = None,
    node_inferred: list[bool] | None = None,
    temporal_refs: list[str] | None = None,  # S-1 : temporal_ref par nœud
    doc_ref: str | None = None,
    mood_by_node: dict[int, str] | None = None,
    connector_by_edge: dict[int, str] | None = None,
    salience_by_edge: dict[int, float] | None = None,
    voice_by_node: dict[int, str] | None = None,
    apply_orientation: bool = False,
    negation_sites: list[str | None] | None = None,
    ambiguous_edges: list[bool] | None = None,
    edge_candidates: list[list | None] | None = None,
) -> dict:
    """
    Produit un dict CausalIR conforme au schéma serde Rust de gcn-ir.
    JSON-serializable. Tous les noms en snake_case.

    edge_triples : 6-tuple legacy
        (src_idx, dst_idx, relation, confidence, negated, marker_token)
        ou 9-tuple v3 (plan §C.3) :
        (src, dst, relation, confidence, negated, marker_token,
         third_role, third_node, joint_group_id).
    node_attributes : list de dicts {entity, agent, patient, quality, agent_type, reversible}
    node_inferred : flag par nœud (origine inférée) — métadonnée Python
      "is_inferred" sur le dict nœud, PAS un token spécial. Ignoré par serde Rust
      (champ supplémentaire côté Python uniquement).
    mood_by_node / connector_by_edge / salience_by_edge : métadonnées D6-shadow.
      Chaque arête émise porte `confidence` (ML, inchangé), `confidence_ml`
      (alias) et `confidence_d6` (formule ETUDE). L'ordre pipeline est
      brut ML → D6 → isotonie T5 (plan §C.5).
    voice_by_node + apply_orientation : Éq.7 puis D7 (plan §C.5).
      apply_orientation=False par défaut (opt-in jusqu'à validation T4 e2e, K3).
    """
    if node_origins is None:
        node_origins = [NODE_ORIGIN_VALUES[0]] * len(node_types)

    # S-2 : temporal_index = rang dans l'ordre du texte (par position de token start)
    # Ordre de création ≠ ordre temporel quand une cause est mentionnée après son effet.
    text_order = sorted(range(len(token_spans)), key=lambda k: token_spans[k][0])
    temporal_rank = [0] * len(token_spans)
    for rank, orig_idx in enumerate(text_order):
        temporal_rank[orig_idx] = rank

    nodes = []
    for i, (nt, label, span, scope, origin) in enumerate(
        zip(node_types, node_labels, token_spans, scopes, node_origins, strict=False)
    ):
        attrs = node_attributes[i] if node_attributes and i < len(node_attributes) else {
            "entity": None,
            "quality": None,
            "agent": None,
            "patient": None,
            "agent_type": None,
            "reversible": None,
        }
        inferred = bool(node_inferred[i]) if node_inferred and i < len(node_inferred) else False
        nodes.append({
            "id": i,
            "node_type": nt,
            "label": label,
            "source_span": {"token_span": {"start": span[0], "end": span[1]}},
            "scope": scope,
            "modifiers": [],
            "temporal_ref": temporal_refs[i] if temporal_refs and i < len(temporal_refs) else TEMPORAL_REF_DEFAULT,
            "temporal_index": temporal_rank[i],  # S-2 : rang texte, pas rang création
            "origin": origin,
            "is_inferred": inferred,
            "attributes": attrs,
        })

    import datetime as _dt
    _now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from .. import __version__ as _gcn_version
    except Exception:
        _gcn_version = "unknown"

    import math
    from collections import defaultdict as _defaultdict
    edges = []
    for edge_idx, tup in enumerate(edge_triples):
        if len(tup) == 6:
            src, dst, relation, confidence, negated, marker_token = tup
            third_role = third_node = joint_group_id = None
            negation_site = (negation_sites[edge_idx]
                             if negation_sites and edge_idx < len(negation_sites) else None)
            ambiguous = (ambiguous_edges[edge_idx]
                         if ambiguous_edges and edge_idx < len(ambiguous_edges) else False)
            candidates = (edge_candidates[edge_idx]
                          if edge_candidates and edge_idx < len(edge_candidates) else None)
        elif len(tup) == 9 and not isinstance(tup[7], bool):
            # 9-tuple §C.3 : (src,dst,rel,conf,neg,marker,third_role,third_node,joint_group_id)
            (src, dst, relation, confidence, negated, marker_token,
             third_role, third_node, joint_group_id) = tup
            negation_site = (negation_sites[edge_idx]
                             if negation_sites and edge_idx < len(negation_sites) else None)
            ambiguous = (ambiguous_edges[edge_idx]
                         if ambiguous_edges and edge_idx < len(ambiguous_edges) else False)
            candidates = (edge_candidates[edge_idx]
                          if edge_candidates and edge_idx < len(edge_candidates) else None)
        else:
            # Nouveau format cgnp.py : (src,dst,rel,conf,neg,marker,negation_site,ambiguous,candidates)
            (src, dst, relation, confidence, negated, marker_token,
             negation_site, ambiguous, candidates) = tup
            third_role = third_node = joint_group_id = None
        if apply_orientation:
            src, dst = orient_edge_d7(src, dst, relation, token_spans)
            src, dst = apply_voice_eq7(src, dst, voice_by_node)
        # C.6 detect_ternary — câblé ici pour les tuples sans third pré-calculé
        if third_role is None and joint_group_id is None:
            src_tokens = None
            if token_spans and src < len(token_spans):
                # Les tokens ne sont pas dans emit() — on passe None ici,
                # detect_ternary opère sur src_tokens si fournis par cgnp
                pass
            _t_role, _t_node, _jgid = detect_ternary(relation, src_tokens, joint_group_id)
            if _t_role:
                third_role, third_node = _t_role, _t_node
            if _jgid:
                joint_group_id = _jgid

        # §9.4 algèbre de négation — promotion "condition" si third présent sans site
        third_dict = ({"role": third_role, "node": third_node} if third_role else None)
        if negated and negation_site is None and third_dict is not None:
            negation_site = "condition"
        relation, third_dict, source_polarity = apply_negation_algebra(
            relation, negation_site, third_dict, negated
        )
        # D6-shadow : ML inchangé, prior ETUDE en champ séparé.
        mood = (mood_by_node or {}).get(dst, (mood_by_node or {}).get(src, "Ind"))
        if connector_by_edge is not None:
            connector = connector_by_edge.get(edge_idx, "explicite")
        else:
            connector = "explicite" if marker_token is not None else "aucun"
        salience = (salience_by_edge or {}).get(edge_idx, 0.0)
        conf_d6 = confidence_d6(relation, mood=mood, connector=connector,
                               salience_bonus=salience)
        conf = float(confidence)
        if not math.isfinite(conf):
            raise ValueError(
                f"emit : confidence non finie ({confidence!r}) pour l'arête "
                f"{src}->{dst} — JSON refusé par serde Rust."
            )
        # S-7 temporal_gap : différence d'indices temporels entre les deux clauses
        t_src = temporal_rank[src] if src < len(temporal_rank) else None
        t_dst = temporal_rank[dst] if dst < len(temporal_rank) else None
        t_gap = (t_dst - t_src) if (t_src is not None and t_dst is not None) else None
        edges.append([src, dst, {
            "relation": relation,
            "confidence": conf,
            "confidence_ml": conf,
            "confidence_d6": conf_d6,
            "temporal_gap": t_gap,
            "explicit": marker_token is not None,
            "negated": negated,
            "source_polarity": source_polarity,
            "marker_token": marker_token,
            "third": third_dict,
            "joint_group_id": joint_group_id,
            "ambiguous": ambiguous,
            "candidates": candidates,
            "in_cycle": None,
            "provenance": {
                "ref": doc_ref,
                "span": ({"token_span": {"start": marker_token, "end": marker_token}}
                         if marker_token is not None else "synthetic"),
                "extraction_method": "ml_python",
                "model_version": _gcn_version,
                "extracted_at": _now_iso,
            },
        }])

    # Éq.11 — cohérence locale : résoudre les arêtes ambiguës par voisinage
    _confirmed_by_node: dict[int, set[str]] = _defaultdict(set)
    for _s, _d, _r in edges:
        if not _r.get("ambiguous"):
            _confirmed_by_node[_s].add(_r["relation"])
            _confirmed_by_node[_d].add(_r["relation"])
    for _s, _d, _r in edges:
        if _r.get("ambiguous") and _r.get("candidates"):
            _neighbor = _confirmed_by_node[_s] | _confirmed_by_node[_d]
            _c0_type, _c0_p = _r["candidates"][0]
            _c1_type, _c1_p = _r["candidates"][1]
            if _c1_type in _neighbor and _c0_type not in _neighbor:
                _r["relation"] = _c1_type
                _c0_p, _c1_p = _c1_p, _c0_p
            else:
                _r["relation"] = _c0_type
            _total = _c0_p + _c1_p
            if _total > 0:
                _r["confidence"] = _c0_p / _total

    # S-7 : détection de cycles par DFS sur le graphe d'arêtes (orienté)
    # Utilise un graphe orienté pour éviter de confondre arête retour avec parent.
    adj_directed: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for src, dst, _ in edges:
        adj_directed[src].append(dst)

    visited: set[int] = set()
    in_path: set[int] = set()       # nœuds dans le chemin de récursion courant
    cycle_node_sets: list[frozenset[int]] = []

    def _dfs_cycle(node: int, path: list[int]) -> None:
        visited.add(node)
        in_path.add(node)
        path.append(node)
        for nb in adj_directed[node]:
            if nb in in_path:
                # Arête retour → cycle : extraire la boucle
                idx = path.index(nb)
                cycle_node_sets.append(frozenset(path[idx:]))
            elif nb not in visited:
                _dfs_cycle(nb, path)
        path.pop()
        in_path.discard(node)

    for start in range(len(nodes)):
        if start not in visited:
            _dfs_cycle(start, [])

    # Dédupliquer les cycles
    unique_cycles = list({frozenset(c) for c in cycle_node_sets})
    cycles_output = [sorted(c) for c in unique_cycles]

    node_to_cycle: dict[int, int] = {}
    for cycle_id, c in enumerate(unique_cycles):
        for nid in c:
            node_to_cycle[nid] = cycle_id
    for edge in edges:
        src_e, dst_e = edge[0], edge[1]
        src_cid = node_to_cycle.get(src_e)
        dst_cid = node_to_cycle.get(dst_e)
        edge[2]["in_cycle"] = (src_cid is not None and src_cid == dst_cid)

    import datetime
    return {
        # Moteur language-agnostic : "und" volontaire (la langue n'est jamais
        # passée au modèle, même à l'entraînement).
        "source_lang": {"natural": {"lang": "und"}},
        "source_text": text,
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles_output,
        "unresolved": [],
        "metadata": {
            "schema_version": "2.0",
            "pipeline": ["cgnp-layer1", "cgnp-layer2", "cgnp-layer3"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        },
    }
