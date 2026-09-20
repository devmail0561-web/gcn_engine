# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
QueryVerbalizer — formate les résultats de requêtes CausalGraph
en rapports structurés multi-lignes avec sources et contradictions.

Aucun texte en dur dans une langue spécifique : les labels viennent
du CIR extrait (quelle que soit la langue du corpus analysé).
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ..constants import RELATION_TYPES

if TYPE_CHECKING:
    from .instructions import CausalGraph

_SEP = "  " + "─" * 54

# Groupes sémantiques dérivés de RELATION_TYPES — aucun string littéral
# Relations qui "bloquent/empêchent" un effet
_BLOCKING_RELS = {r for r in RELATION_TYPES if r in {"prevent", "filter"}}
# Relations qui "causent/permettent" un effet
_ENABLING_RELS = {r for r in RELATION_TYPES if r in {"cause", "enable", "motivation", "sequence"}}


def _sanitize(s: str) -> str:
    import re as _re
    if not isinstance(s, str):
        return s
    s = _re.sub(r"\x1b\[[0-9;]*m", "", s)
    return s.replace("\n", " ").replace("\r", " ").strip()


def _truncate(text: str, max_chars: int = 200) -> str:
    text = _sanitize(text or "")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut if cut else text[:max_chars]


class QueryVerbalizer:
    """
    Produit des rapports analytiques depuis les résultats de CausalGraph.

    Les rapports incluent :
    - Toutes les relations trouvées (pas une seule)
    - Sources exactes par relation
    - Niveau de confiance et fréquence
    - Contradictions flaggées explicitement
    """

    def __init__(self, graph: "CausalGraph", max_source_chars: int = 200):
        self.graph = graph
        self.max_source_chars = int(max_source_chars)

    # ------------------------------------------------------------------
    # Rapport : causes d'un concept
    # ------------------------------------------------------------------

    def causes(self, keyword: str) -> str:
        results = self.graph.find_causes(keyword)
        if not results:
            return f"  causes_of: {keyword!r} — not found in corpus."

        lines = [f"  causes_of: {keyword!r}", _SEP]

        # Grouper par source → dst pour détecter les contradictions
        effects_by_src: dict[str, list] = {}
        for src, dst, attrs, text in results:
            src_lbl = self.graph._label(src)
            key = src_lbl
            effects_by_src.setdefault(key, []).append((src, dst, attrs, text))

        # Chercher les contradictions (même src, relations opposées)
        prevent_srcs = {
            self.graph._label(src)
            for src, dst, attrs, _ in self.graph.find_effects(keyword)
            if attrs.get("relation") in _BLOCKING_RELS
        }

        for i, (src, dst, attrs, text) in enumerate(results, 1):
            src_lbl  = self.graph._label(src)
            src_type = self.graph._type(src)
            rel      = attrs.get("relation", "?")
            conf     = attrs.get("confidence")
            neg      = " [negated]" if attrs.get("negated") else ""
            conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "?"
            lines.append(
                f"  {i}. [{src_type}] {src_lbl}"
                f"  --[{rel}{neg}]-->  {keyword!r}"
                f"  (conf={conf_s})"
            )
            if text:
                lines.append(f"     source: {_truncate(text, self.max_source_chars)}")

        # Groupement sources convergentes : (frozenset(src_ids), dst, rel) -> (n, min, max)
        _groups: dict[tuple, list] = {}
        for src, dst, attrs, text in results:
            key = (frozenset([src]), dst, attrs.get("relation"))
            conf = attrs.get("confidence")
            c = float(conf) if isinstance(conf, (int, float)) else 0.0
            _groups.setdefault(key, []).append(c)
        if len(_groups) > 1:
            ranked = sorted(
                ((len(v), min(v), max(v), k) for k, v in _groups.items()),
                key=lambda t: (t[0] * t[2], t[2]),
                reverse=True,
            )
            lines.append("  convergent groups (n, min_conf, max_conf):")
            for n, mn, mx, k in ranked[:5]:
                lines.append(f"    {k[2]} n={n} min={mn:.0%} max={mx:.0%}")

        # Contradictions : prevent OU filter + negated=True
        contradictions = [
            (src, dst, attrs, text)
            for src, dst, attrs, text in self.graph.edges
            if (self.graph._label(dst) == keyword or keyword.lower() in self.graph._label(dst).lower())
            and (attrs.get("relation") in _BLOCKING_RELS or attrs.get("negated") is True)
        ]
        if contradictions:
            lines.append(f"\n  ⚠ CONTRADICTION — {len(contradictions)} source(s) bloquantes:")
            for src, dst, attrs, text in contradictions:
                lines.append(f"     {self.graph._label(src)} --[{attrs.get('relation')}]--> {keyword!r}")
                if text:
                    lines.append(f"     source: {_truncate(text, self.max_source_chars)}")

        lines.append(_SEP)
        lines.append(f"  {len(results)} cause(s) found.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rapport : effets d'un concept
    # ------------------------------------------------------------------

    def effects(self, keyword: str) -> str:
        results = self.graph.find_effects(keyword)
        if not results:
            return f"  effects_of: {keyword!r} — not found in corpus."

        lines = [f"  effects_of: {keyword!r}", _SEP]
        for i, (src, dst, attrs, text) in enumerate(results, 1):
            dst_lbl  = self.graph._label(dst)
            dst_type = self.graph._type(dst)
            rel      = attrs.get("relation", "?")
            conf     = attrs.get("confidence")
            neg      = " [negated]" if attrs.get("negated") else ""
            conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "?"
            lines.append(
                f"  {i}. --[{rel}{neg}]-->  [{dst_type}] {dst_lbl}"
                f"  (conf={conf_s})"
            )
            if text:
                lines.append(f"     source: {_truncate(text, self.max_source_chars)}")

        lines.append(_SEP)
        lines.append(f"  {len(results)} effect(s) found.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rapport : chemin causal A → B
    # ------------------------------------------------------------------

    def path(self, kw_from: str, kw_to: str) -> str:
        path_ids = self.graph.find_path(kw_from, kw_to)
        if not path_ids:
            return f"  chain: no causal path from {kw_from!r} to {kw_to!r}."

        lines = [f"  chain: {kw_from!r} → {kw_to!r}", _SEP]
        steps = []
        for nid in path_ids:
            steps.append(f"[{self.graph._type(nid)}] {self.graph._label(nid)}")
        lines.append("  " + " --> ".join(steps))

        # Collecter les sources des arêtes du chemin
        sources = set()
        for i in range(len(path_ids) - 1):
            src, dst = path_ids[i], path_ids[i + 1]
            for s, d, attrs, text in self.graph.edges:
                if s == src and d == dst and text:
                    sources.add(_truncate(text, self.max_source_chars))
        if sources:
            lines.append("  sources: " + " | ".join(sorted(sources)[:3]))

        lines.append(_SEP)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rapport : contradictions dans le corpus
    # ------------------------------------------------------------------

    def contradictions(self) -> str:
        conflicts = []
        edge_map: dict[tuple, list] = {}
        for src, dst, attrs, text in self.graph.edges:
            key = (src, dst)
            edge_map.setdefault(key, []).append((attrs.get("relation"), text))

        for (src, dst), rels_texts in edge_map.items():
            rels = [r for r, _ in rels_texts]
            if any(r in _BLOCKING_RELS for r in rels) and any(r in _ENABLING_RELS for r in rels):
                conflicts.append((src, dst, rels_texts))

        if not conflicts:
            return "  contradictions: none found in corpus."

        lines = [f"  contradictions: {len(conflicts)} found", _SEP]
        for src, dst, rels_texts in conflicts:
            src_lbl = self.graph._label(src)
            dst_lbl = self.graph._label(dst)
            lines.append(f"  {src_lbl}  ↔  {dst_lbl}")
            for rel, text in rels_texts:
                lines.append(f"    [{rel}]  source: {_truncate(text or '', self.max_source_chars)}")

        lines.append(_SEP)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rapport : raisonnement contrefactuel
    # ------------------------------------------------------------------

    def counterfactual(self, keyword: str) -> str:
        effects  = self.graph.find_effects(keyword)
        causes   = self.graph.find_causes(keyword)
        if not effects and not causes:
            return f"  counterfactual: {keyword!r} — not connected in causal graph."

        lines = [f"  counterfactual: do(not {keyword!r})", _SEP]

        if effects:
            lines.append("  blocked downstream:")
            for _, dst, attrs, _ in effects:
                lines.append(
                    f"    [{attrs.get('relation','?')}] --> "
                    f"[{self.graph._type(dst)}] {self.graph._label(dst)}"
                )

        if causes:
            lines.append("  interrupted upstream:")
            for src, _, attrs, _ in causes:
                lines.append(
                    f"    [{self.graph._type(src)}] {self.graph._label(src)}"
                    f" --[{attrs.get('relation','?')}]--> (removed)"
                )

        lines.append("  [Pearl level 2 — intervention]")
        lines.append(_SEP)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rapport : résumé du corpus
    # ------------------------------------------------------------------

    def summarize(self) -> str:
        if not self.graph.edges:
            return "  summarize: corpus is empty."

        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)
        sources = {text for _, _, _, text in self.graph.edges if text}
        rel_counts = Counter(
            attrs.get("relation", "?")
            for _, _, attrs, _ in self.graph.edges
        )

        lines = [
            "  summarize:",
            _SEP,
            f"  nodes     : {n_nodes}",
            f"  relations : {n_edges}",
            f"  sources   : {len(sources)} document(s)",
            "",
            "  relation distribution:",
        ]
        for rel, cnt in rel_counts.most_common():
            bar = "█" * min(cnt, 20)
            lines.append(f"    {rel:22s} {cnt:4d}  {bar}")

        # Hub causal
        out_deg = Counter(src for src, _, _, _ in self.graph.edges)
        if out_deg:
            top_id = out_deg.most_common(1)[0][0]
            lines.append(
                f"\n  top causal node: [{self.graph._type(top_id)}]"
                f" {self.graph._label(top_id)}"
            )

        lines.append(_SEP)
        return "\n".join(lines)
