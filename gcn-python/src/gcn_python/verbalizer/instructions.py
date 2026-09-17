"""
InstructionHandler — exécute des commandes structurées sur le graphe causal.

Commandes language-agnostic (mots-clés fixes, contenu en n'importe quelle langue) :

    explain: <concept>          — causes de ce concept
    effects: <concept>          — effets de ce concept
    chain: <concept_a> <concept_b>  — chemin causal entre deux concepts
    counterfactual: <concept>   — que se passe-t-il sans ce concept ?
    summarize                   — aperçu de la session
    verbalize                   — CIR → texte structuré

Les labels des nœuds viennent du texte analysé — n'importe quelle langue.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from typing import Optional


# ---------------------------------------------------------------------------
# Commandes reconnues (fixes, indépendantes de la langue)
# ---------------------------------------------------------------------------

COMMANDS = {
    "explain",        # causes de X
    "effects",        # effets de X
    "chain",          # chemin de A à B
    "counterfactual", # sans X
    "summarize",      # résumé
    "verbalize",      # CIR → texte
    "help",           # aide
    "clear",          # vider
    "quit",           # quitter
}


def parse_command(text: str) -> tuple[str, Optional[str]]:
    """
    Parse une entrée utilisateur en (commande, argument).

    Format attendu :
        explain: ventes
        effects: hausse des coûts
        chain: demande ventes
        counterfactual: demande
        summarize
        verbalize

    Si l'entrée ne correspond à aucune commande connue, retourne
    ("text", None) — le texte sera traité comme texte à analyser.
    """
    t = text.strip()

    # Format "commande: argument" ou "commande argument"
    for cmd in COMMANDS:
        # Avec deux-points
        if re.match(rf"^{cmd}\s*:", t, re.I):
            arg = re.sub(rf"^{cmd}\s*:\s*", "", t, flags=re.I).strip()
            return cmd.lower(), arg or None
        # Sans deux-points, si la commande est en premier mot seul
        if re.match(rf"^{cmd}\s*$", t, re.I):
            return cmd.lower(), None

    return "text", None


# ---------------------------------------------------------------------------
# Graphe causal en mémoire
# ---------------------------------------------------------------------------

class CausalGraph:
    """Graphe causal construit depuis les CIR accumulés en session."""

    def __init__(self):
        self.nodes: dict = {}                            # id → node dict
        self.edges: list[tuple] = []                     # (src, dst, attrs, source_text)
        self.adjacency: dict  = defaultdict(list)        # src → [dst]
        self.reverse_adj: dict = defaultdict(list)       # dst → [src]

    def add_cir(self, cir: dict) -> None:
        for n in cir.get("nodes", []):
            if isinstance(n, dict):
                self.nodes[n.get("id")] = n
        src_text = cir.get("source_text", "")
        for e in cir.get("edges", []):
            if isinstance(e, (list, tuple)) and len(e) == 3:
                src, dst, attrs = e
                self.edges.append((src, dst, attrs, src_text))
                self.adjacency[src].append(dst)
                self.reverse_adj[dst].append(src)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.reverse_adj.clear()

    def find_causes(self, keyword: str) -> list[tuple]:
        """Arêtes dont la cible contient keyword."""
        kw = keyword.lower()
        return [
            (src, dst, attrs, text)
            for src, dst, attrs, text in self.edges
            if kw in (self.nodes.get(dst, {}).get("label", "")).lower()
        ]

    def find_effects(self, keyword: str) -> list[tuple]:
        """Arêtes dont la source contient keyword."""
        kw = keyword.lower()
        return [
            (src, dst, attrs, text)
            for src, dst, attrs, text in self.edges
            if kw in (self.nodes.get(src, {}).get("label", "")).lower()
        ]

    def find_path(self, kw_from: str, kw_to: str) -> list:
        """BFS : chemin de nœuds contenant kw_from vers nœuds contenant kw_to."""
        src_ids = [nid for nid, n in self.nodes.items()
                   if kw_from.lower() in n.get("label", "").lower()]
        dst_ids = {nid for nid, n in self.nodes.items()
                   if kw_to.lower() in n.get("label", "").lower()}
        for start in src_ids:
            visited = {start}
            queue = deque([[start]])
            while queue:
                path = queue.popleft()
                if path[-1] in dst_ids:
                    return path
                for nb in self.adjacency.get(path[-1], []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(path + [nb])
        return []

    def _label(self, nid) -> str:
        return self.nodes.get(nid, {}).get("label", str(nid))

    def _type(self, nid) -> str:
        return self.nodes.get(nid, {}).get("node_type", "?")


# ---------------------------------------------------------------------------
# Génération des réponses (structure seule, aucun hardcoding linguistique)
# ---------------------------------------------------------------------------

class InstructionHandler:
    """
    Exécute des commandes sur le graphe causal de session.

    Toutes les réponses sont construites depuis la structure du CIR.
    Aucun texte en dur dans une langue spécifique — seuls les labels
    du CIR (qui viennent du texte analysé) apparaissent dans les réponses.
    """

    def __init__(self):
        self.graph = CausalGraph()

    def add_cir(self, cir: dict) -> None:
        self.graph.add_cir(cir)

    def clear(self) -> None:
        self.graph.clear()

    def execute(self, text: str) -> Optional[str]:
        """
        Parse et exécute une commande. Retourne None si c'est du texte à analyser.
        """
        cmd, arg = parse_command(text)

        if cmd == "text":
            return None  # à analyser, pas une commande

        if cmd == "explain":
            return self._explain(arg)
        if cmd == "effects":
            return self._effects(arg)
        if cmd == "chain":
            return self._chain(arg)
        if cmd == "counterfactual":
            return self._counterfactual(arg)
        if cmd == "summarize":
            return self._summarize()
        if cmd == "verbalize":
            return self._verbalize()
        if cmd == "help":
            return self._help()

        return None

    # ------------------------------------------------------------------

    def _explain(self, keyword: Optional[str]) -> str:
        if not keyword:
            return "explain: <concept>  — missing argument"
        causes = self.graph.find_causes(keyword)
        if not causes:
            return f"explain: no causes found for '{keyword}'"
        lines = [f"explain: {keyword}"]
        for src, dst, attrs, src_text in causes:
            conf = attrs.get("confidence", 0.0)
            rel  = attrs.get("relation", "?")
            neg  = " [negated]" if attrs.get("negated") else ""
            lines.append(
                f"  {self.graph._label(src)}"
                f"  --[{rel}]{neg}-->"
                f"  {self.graph._label(dst)}"
                f"  conf={conf:.2f}"
            )
        return "\n".join(lines)

    def _effects(self, keyword: Optional[str]) -> str:
        if not keyword:
            return "effects: <concept>  — missing argument"
        effects = self.graph.find_effects(keyword)
        if not effects:
            return f"effects: no effects found for '{keyword}'"
        lines = [f"effects: {keyword}"]
        for src, dst, attrs, _ in effects:
            rel = attrs.get("relation", "?")
            neg = " [negated]" if attrs.get("negated") else ""
            lines.append(
                f"  --> [{rel}]{neg}  {self.graph._label(dst)}"
                f"  [{self.graph._type(dst)}]"
            )
        return "\n".join(lines)

    def _chain(self, arg: Optional[str]) -> str:
        if not arg:
            return "chain: <concept_a> <concept_b>  — missing arguments"
        parts = arg.split()
        if len(parts) < 2:
            return "chain: provide two concepts separated by a space"
        kw_from, kw_to = parts[0], parts[-1]
        path = self.graph.find_path(kw_from, kw_to)
        if not path:
            return f"chain: no causal path found from '{kw_from}' to '{kw_to}'"
        steps = [
            f"[{self.graph._type(nid)}] {self.graph._label(nid)}"
            for nid in path
        ]
        return "chain:\n  " + " --> ".join(steps)

    def _counterfactual(self, keyword: Optional[str]) -> str:
        if not keyword:
            return "counterfactual: <concept>  — missing argument"
        effects  = self.graph.find_effects(keyword)
        causes   = self.graph.find_causes(keyword)
        if not effects and not causes:
            return f"counterfactual: '{keyword}' not connected in causal graph"
        lines = [f"counterfactual: do(not {keyword})"]
        if effects:
            lines.append("  blocked downstream:")
            for _, dst, attrs, _ in effects:
                lines.append(
                    f"    [{attrs.get('relation','?')}] --> "
                    f"{self.graph._label(dst)} [{self.graph._type(dst)}]"
                )
        if causes:
            lines.append("  interrupted upstream:")
            for src, _, attrs, _ in causes:
                lines.append(
                    f"    {self.graph._label(src)} [{self.graph._type(src)}]"
                    f" --[{attrs.get('relation','?')}]--> (removed)"
                )
        lines.append("  [Pearl level 2 — intervention]")
        return "\n".join(lines)

    def _summarize(self) -> str:
        if not self.graph.edges:
            return "summarize: no causal relations in session"
        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)
        rel_counts = Counter(
            attrs.get("relation", "?")
            for _, _, attrs, _ in self.graph.edges
        )
        lines = [
            f"summarize: {n_nodes} node(s), {n_edges} relation(s)",
            "  relations:",
        ]
        for rel, cnt in rel_counts.most_common():
            lines.append(f"    {rel}: {cnt}")
        # Hub causal (nœud source avec le plus d'arêtes sortantes)
        out_deg = Counter(src for src, _, _, _ in self.graph.edges)
        if out_deg:
            top = out_deg.most_common(1)[0][0]
            lines.append(
                f"  top causal node: [{self.graph._type(top)}]"
                f" {self.graph._label(top)}"
            )
        return "\n".join(lines)

    def _verbalize(self) -> str:
        if not self.graph.edges:
            return "verbalize: no relations to verbalize"
        lines = ["verbalize:"]
        for i, (src, dst, attrs, _) in enumerate(self.graph.edges, 1):
            rel  = attrs.get("relation", "?")
            neg  = " [negated]" if attrs.get("negated") else ""
            conf = attrs.get("confidence", 0.0)
            lines.append(
                f"  {i}. [{self.graph._type(src)}] {self.graph._label(src)}"
                f"  --[{rel}{neg}]-->"
                f"  [{self.graph._type(dst)}] {self.graph._label(dst)}"
                f"  ({conf:.0%})"
            )
        return "\n".join(lines)

    def _help(self) -> str:
        return (
            "commands:\n"
            "  explain: <concept>              — what causes <concept>\n"
            "  effects: <concept>              — what does <concept> cause\n"
            "  chain: <concept_a> <concept_b>  — causal path between two concepts\n"
            "  counterfactual: <concept>       — Pearl do-calculus: remove <concept>\n"
            "  summarize                       — session overview\n"
            "  verbalize                       — structured CIR as text\n"
            "  clear                           — reset session\n"
            "  quit                            — exit"
        )
