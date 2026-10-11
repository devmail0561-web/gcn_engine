# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""R1 — `reason()` pure et bornée (ETUDE_SA : bibliothèque, jamais d'I/O).

Lit le graphe seul (primitives pures `find_causes` / `find_effects` /
`find_path`), n'écrit jamais dedans : les `derived` sont retournés dans la
trace, pas ingérés. R1 : un but = une paire (from, to), profondeur ≤ 2
(direct + 1 maillon transitif), top-k=5 par requête, budget=16 par défaut
(1 `query:` / 1 `infer:` = 1 étape).
"""
from __future__ import annotations

import warnings as _warnings

from .steps import (
    STATUS_DEGRADE,
    STATUS_OK,
    make_step,
    reason_response,
)

#: Caps session (même ordre que M2a : 500 nœuds / 2000 arêtes).
MAX_NODES = 500
MAX_EDGES = 2000
#: Candidats par requête (tri level/conf existant, on tronque).
TOPK = 5
#: Sous-buts max (profondeur bornée → pas d'explosion).
MAX_SUBGOALS = 3


def _label(graph, nid: str) -> str:
    return graph.nodes.get(nid, {}).get("label", nid)


def _edge_conf(attrs) -> float:
    if not isinstance(attrs, dict):
        return 0.5
    try:
        return float(attrs.get("confidence", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _quiet_find(graph, method: str, keyword: str, topk: int = TOPK) -> list:
    """find_* sans warnings substring (boucle : pas de pollution de logs)."""
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        res = getattr(graph, method)(keyword)
    return list(res[:topk])


def _resolve(graph, keyword: str) -> list:
    """Mot-clé → ids de nœuds (id direct, sinon match label ≤ 2, cap TOPK)."""
    if keyword in graph.nodes:
        return [keyword]
    out = []
    for nid, node in graph.nodes.items():
        if graph._match_level(node.get("label", ""), keyword) in (1, 2):
            out.append(nid)
            if len(out) >= TOPK:
                break
    return out


def verify_path(path: list, graph) -> tuple[bool, str]:
    """Chaque maillon = arête présente et non `negated` (Ph1)."""
    for a, b in zip(path, path[1:]):
        edge = graph._edge_between(a, b)
        if edge is None:
            return False, f"missing-link: {a} -> {b}"
        if isinstance(edge, dict) and edge.get("negated"):
            return False, f"contradiction: {a} -> {b} [negated]"
    return True, "verified"


def _chain_answer(path: list, graph, derived: list) -> dict:
    confs = [_edge_conf(graph._edge_between(a, b)) for a, b in zip(path, path[1:])]
    conf = min(confs) if confs else 1.0
    return {
        "summary": "chain:\n  " + " --> ".join(_label(graph, n) for n in path),
        "path": list(path),
        "confidence": conf,
        "derived": derived,
    }


def _bad_edge(graph, a: str, b: str) -> bool:
    """Maillon inutilisable : absent ou `negated`."""
    edge = graph._edge_between(a, b)
    return edge is None or (isinstance(edge, dict) and bool(edge.get("negated")))


def reason(kw_from: str, kw_to: str, graph, budget: int = 16) -> dict:
    """But (from → to) sur le graphe donné. Pur, borné, déterministe."""
    trace: list = []
    tour = 0
    spent = 0

    def _spend(n: int = 1) -> bool:
        nonlocal spent, tour
        spent += n
        tour += 1
        return spent <= budget

    def _degrade(motif: str, summary: str) -> dict:
        trace.append(make_step("abandoned:", tour, motif=motif))
        return reason_response(STATUS_DEGRADE, {"summary": summary}, trace)

    kw_from = (kw_from or "").strip()
    kw_to = (kw_to or "").strip()
    trace.append(make_step("goal:", 0, extra={"from": kw_from, "to": kw_to}))
    if not kw_from or not kw_to:
        return _degrade("empty-goal", "goal: unresolved")
    if len(graph.nodes) > MAX_NODES or len(graph.edges) > MAX_EDGES:
        return _degrade("oversize:", "graph: oversize")

    # Direct : un observe + un verify.
    if not _spend():
        return _degrade("budget:", "budget: exhausted")
    trace.append(make_step("query:", tour, extra={"op": "find_path"}))
    path = graph.find_path(kw_from, kw_to)
    trace.append(make_step("observe:", tour, extra={"path": path}))
    if path:
        ok, detail = verify_path(path, graph)
        trace.append(make_step("verify:", tour, motif=None if ok else detail))
        if ok:
            trace.append(make_step("answer:", tour, premises=tuple(path),
                                    conf=_chain_answer(path, graph, [])["confidence"]))
            return reason_response(STATUS_OK, _chain_answer(path, graph, []), trace)
        missing = [(path[i], path[i + 1]) for i in range(len(path) - 1)
                   if _bad_edge(graph, path[i], path[i + 1])][:MAX_SUBGOALS]
    else:
        missing = []

    # Repli : paire brute (résolue en ids ci-dessous).
    if not missing:
        missing = [(kw_from, kw_to)]
    for u, v in missing:
        u_ids = _resolve(graph, u)
        v_ids = _resolve(graph, v)
        if not u_ids or not v_ids:
            continue
        trace.append(make_step("subgoal:", tour, extra={"from": u, "to": v}))
        if not _spend():
            return _degrade("budget:", "budget: exhausted")
        trace.append(make_step("query:", tour,
                               extra={"op": "find_effects+find_causes"}))
        cands = _quiet_find(graph, "find_effects", _label(graph, u_ids[0]))
        cands += _quiet_find(graph, "find_causes", _label(graph, v_ids[0]))
        seen: set = set()
        for s, d, attrs, _text in cands:
            if (s, d) in seen:
                continue
            seen.add((s, d))
            bridges = []
            # Pont A : s côté u, puis d → v (direct ou chemin court vérifié).
            if s in u_ids:
                for v2 in v_ids:
                    if graph._edge_between(d, v2) is not None:
                        bridges.append([s, d, v2])
                    elif _spend():
                        trace.append(make_step(
                            "query:", tour, extra={"op": "find_path"}))
                        p = graph.find_path(_label(graph, d), _label(graph, v2))
                        if len(p) >= 2 and p[0] == d and len(p) <= 3:
                            bridges.append([s] + p)
            # Pont B : d côté v, précédé de u → s.
            if d in v_ids:
                for u2 in u_ids:
                    if graph._edge_between(u2, s) is not None:
                        bridges.append([u2, s, d])
                    elif _spend():
                        trace.append(make_step(
                            "query:", tour, extra={"op": "find_path"}))
                        p = graph.find_path(_label(graph, u2), _label(graph, s))
                        if len(p) >= 2 and p[-1] == s and len(p) <= 3:
                            bridges.append(p + [d])
            for bridge in bridges:
                if not _spend():
                    return _degrade("budget:", "budget: exhausted")
                confs = [_edge_conf(graph._edge_between(a, b))
                         for a, b in zip(bridge, bridge[1:])]
                conf = min(confs) if confs else 0.5
                rels = [graph._edge_between(a, b).get("relation")
                        if isinstance(graph._edge_between(a, b), dict) else None
                        for a, b in zip(bridge, bridge[1:])]
                trace.append(make_step("infer:", tour,
                                       premises=tuple(f"{a}->{b}" for a, b
                                                      in zip(bridge, bridge[1:])),
                                       conf=conf, extra={"rel": rels}))
                ok, detail = verify_path(bridge, graph)
                trace.append(make_step("verify:", tour, motif=None if ok else detail))
                if ok:
                    derived = [{"path": bridge, "derivation": "derived",
                                "confidence": conf,
                                "premises": [f"{a}->{b}" for a, b
                                             in zip(bridge, bridge[1:])]}]
                    trace.append(make_step("answer:", tour,
                                           premises=tuple(bridge), conf=conf))
                    return reason_response(STATUS_OK,
                                           _chain_answer(bridge, graph, derived),
                                           trace)
                trace.append(make_step("abandoned:", tour, motif=detail))
    return _degrade("budget:" if spent >= budget else "no-bridge",
                    "budget: exhausted" if spent >= budget
                    else "chain: no causal path")


def reason_question(text: str, graph, budget: int = 16) -> dict:
    """Texte → but (from, to) via `parse_command` existant, puis `reason()`.

    Comprend `chain A B` / `chain A -> B` (séparateurs `->`, `,`, espaces).
    Texte non analysable → `Degrade` motif `unparsed-goal` (jamais d'invention).
    Multi-buts explicites (`;`) → `Degrade` motif `multigoal:`.
    """
    from ..verbalizer.instructions import parse_command

    text = (text or "").strip()
    if ";" in text:
        trace = [make_step("goal:", 0, extra={"from": text, "to": ""}),
                 make_step("abandoned:", 1, motif="multigoal:")]
        return reason_response(STATUS_DEGRADE, {"summary": "goal: multigoal"}, trace)
    cmd, arg = parse_command(text)
    kw_from = kw_to = ""
    raw = arg
    if cmd == "text":
        # Forme nue `chain A B` (canonique : `chain: A -> B`).
        import re as _re0
        _m = _re0.match(r"(?i)^chain\s+(.+)$", text)
        if _m:
            raw = _m.group(1)
    if raw:
        import re as _re
        parts = [p for p in _re.split(r"->|,|\s+", raw) if p]
        if len(parts) >= 2:
            kw_from, kw_to = parts[0], parts[-1]
    if not kw_from or not kw_to:
        trace = [make_step("goal:", 0, extra={"from": text, "to": ""}),
                 make_step("abandoned:", 1, motif="unparsed-goal")]
        return reason_response(STATUS_DEGRADE, {"summary": "goal: unparsed"}, trace)
    return reason(kw_from, kw_to, graph, budget=budget)
