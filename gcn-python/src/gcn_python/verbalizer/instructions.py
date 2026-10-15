# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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

# ---------------------------------------------------------------------------
# Commandes reconnues (fixes, indépendantes de la langue)
# ---------------------------------------------------------------------------

COMMANDS = {
    "explain",        # causes de X (Pearl WHY)
    "abduct",         # hypothèses causales pour un effet (Pearl abductif)
    "effects",        # effets de X
    "chain",          # chemin de A à B
    "analogy",        # patterns causaux similaires à un patron A→B
    "zoom_in",        # enfants dans la hiérarchie multi-échelle
    "zoom_out",       # parent dans la hiérarchie
    "aggregate",      # vue agrégée d'un nœud et ses enfants
    "centrality",     # centralité causale d'un nœud
    "spof",           # nœuds dont la suppression coupe le plus de chaînes
    "diff",           # écart normatif A→B vs attente (conf=1.0)
    "density",        # densité du graphe (global ou local)
    "coverage",       # couverture causale d'un concept
    "reliability",    # fiabilité des réponses pour un concept
    "chain_t",        # chemin temporel ordonné
    "before",         # A précède-t-il B ?
    "delay",          # délai estimé A→B
    "counterfactual", # sans X
    "summarize",      # résumé
    "verbalize",      # CIR → texte
    "lang",           # dialogue 9d : lang: <code> (état session, intercepté avant execute)
    "revoke",         # dialogue 9e : revoke: [tour] (état session, intercepté avant execute)
    "help",           # aide
    "clear",          # vider
    "quit",           # quitter
}


def parse_command(text: str) -> tuple[str, str | None]:
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
        if re.match(rf"^{cmd}\s*:", t, re.IGNORECASE):
            arg = re.sub(rf"^{cmd}\s*:\s*", "", t, flags=re.IGNORECASE).strip()
            return cmd.lower(), arg or None
        # Sans deux-points, si la commande est en premier mot seul
        if re.match(rf"^{cmd}\s*$", t, re.IGNORECASE):
            return cmd.lower(), None

    return "text", None


# ---------------------------------------------------------------------------
# Origine dialogue 9a (FIGÉ D-ORIG + spec-origine-addcir.md)
#
# Chaque écriture porte session_origin ∈ {document, affirmation_utilisateur,
# question_utilisateur}. Le poids 9a (document 1.0 / affirmation 0.8 /
# question 0.5, FIGÉ) est appliqué À L'ÉCRITURE : la confiance stockée est
# déjà effective (raw × poids) — tout consommateur aval (tri, affichage,
# futur export vers le backend Rust qui applique ses propres poids 7b)
# la respecte sans recalcul, en un seul point de multiplication par axe.
# Une question ne pose jamais un fait : ses arêtes restent visibles avec
# mention d'origine mais ne prouvent rien (cf. origin_only_note).
# ---------------------------------------------------------------------------

SESSION_ORIGINS = ("document", "affirmation_utilisateur", "question_utilisateur")

SESSION_WEIGHTS = {
    "document": 1.0,
    "affirmation_utilisateur": 0.8,
    "question_utilisateur": 0.5,
}

# Apports utilisateur (poids < 1.0) — révocables par 9e, jamais le corpus.
_USER_ORIGINS = frozenset({"affirmation_utilisateur", "question_utilisateur"})

# Précédence d'attestation des nœuds : un nœud vu en question puis attesté
# en document est promu (upgrade-only, jamais de downgrade).
_SESSION_PRECEDENCE = {"question_utilisateur": 0, "affirmation_utilisateur": 1, "document": 2}

# Tags d'origine affichés (codes stables, language-agnostic — pas de traduction).
_ORIGIN_TAG = {"question_utilisateur": "question", "affirmation_utilisateur": "affirmation"}


def edge_origin_tag(attrs: dict) -> str:
    """Tag d'origine pour une ligne de réponse ("" si document/inconnu)."""
    if not isinstance(attrs, dict):
        return ""
    tag = _ORIGIN_TAG.get(attrs.get("session_origin"))
    return f" [{tag}]" if tag else ""


def origin_only_note(edges: list) -> str:
    """Mention 9a quand TOUTES les arêtes viennent de l'utilisateur.

    Ligne de code structurée (moteur language-agnostic — pas de prose) :
    `origin: question-only` / `origin: user-only`. Sens : les arêtes montrées
    ne sont attestées par aucun document — une question ne pose jamais un fait.
    edges : séquences (src, dst, attrs, text). "" si mixte/corpus/vide.
    """
    if not edges:
        return ""
    origins = set()
    for e in edges:
        try:
            _s, _d, attrs, _t = e
        except (TypeError, ValueError):
            continue
        origins.add(attrs.get("session_origin") if isinstance(attrs, dict) else None)
    if not origins or not origins <= _USER_ORIGINS:
        return ""
    if "question_utilisateur" in origins:
        return "  origin: question-only"
    return "  origin: user-only"

class CausalGraph:
    """Graphe causal construit depuis les CIR accumulés en session."""

    def __init__(self):
        self.nodes: dict = {}                            # id → node dict
        self.edges: list[tuple] = []                     # (src, dst, attrs, source_text)
        self.adjacency: dict  = defaultdict(list)        # src → [dst]
        self.reverse_adj: dict = defaultdict(list)       # dst → [src]
        # M2b — compteurs de mutation (save incrémentale : log ssi purs ajouts).
        self._adds = 0
        self._dels = 0

    def add_cir(self, cir: dict, session_origin: str | None = None) -> None:
        """Fusionne un CIR en session avec origine 9a REQUISE (pas de défaut silencieux).

        - nœuds : origine de l'écriture, upgrade-only par précédence
          (question → affirmation → document) ;
        - arêtes : copie des attrs (jamais d'alias avec le CIR appelant),
          stamp session_origin + session_weight, confiance stockée = raw ×
          poids 9a (FIGÉ D-ORIG) ; arête déjà stampée = premier stamp conservé
          (jamais de double multiplication).
        Lève ValueError si session_origin absente ou inconnue.
        P2 (rigidité) : relation inconnue (hors 19 types, None accepté) →
        ValueError ; extrémité ni dans le CIR ni dans le graphe (références
        inter-CIR via from_cirs acceptées) → ValueError nommant les manquants.
        Formes d'arêtes hétérogènes (non 3-tuple/dict, sans src/dst) : sautées
        comme avant (tolérance de format, pas de validation).
        """
        from ..constants import RELATION_TYPES as _REL19
        if session_origin not in SESSION_WEIGHTS:
            raise ValueError(
                f"add_cir : session_origin requise parmi {list(SESSION_ORIGINS)} "
                f"(reçu {session_origin!r})."
            )
        weight = SESSION_WEIGHTS[session_origin]
        import math as _m
        # Ids normalisés en str : les CIR moteur portent des ids entiers (0, 1, …)
        # que JSON.stringify en clés ("0", "1", …) au save — sans normalisation,
        # tout graphe rechargé perd ses labels (nodes.get(dst) → None).
        for n in cir.get("nodes", []):
            if isinstance(n, dict) and n.get("id") is not None:
                nid = str(n.get("id"))
                n = dict(n)
                n["id"] = nid
                cur = self.nodes.get(nid)
                if cur is None:
                    n["session_origin"] = session_origin
                    self.nodes[nid] = n
                else:
                    for k, v in n.items():
                        if k not in ("id", "session_origin"):
                            cur[k] = v
                    if _SESSION_PRECEDENCE[session_origin] > _SESSION_PRECEDENCE.get(
                        cur.get("session_origin"), -1
                    ):
                        cur["session_origin"] = session_origin
        src_text = cir.get("source_text", "")
        _cir_ids = {str(n.get("id")) for n in cir.get("nodes", [])
                    if isinstance(n, dict) and n.get("id") is not None}
        for e in cir.get("edges", []):
            if isinstance(e, (list, tuple)) and len(e) == 3:
                src, dst, attrs = e
            elif isinstance(e, dict):
                # format dict {source(s), target, relation...}
                src = (e.get("sources") or [e.get("source")])[0] if isinstance(e.get("sources"), list) else e.get("source")
                dst = e.get("target", e.get("dst"))
                attrs = e
                if src is None or dst is None:
                    continue
            else:
                continue
            if not isinstance(attrs, dict):
                continue
            attrs = dict(attrs)  # copie : le stamp ne doit jamais fuiter dans le CIR appelant
            _rel = attrs.get("relation")
            if _rel is not None and _rel not in _REL19:
                raise ValueError(
                    f"add_cir : relation inconnue {_rel!r} "
                    f"(19 types : {', '.join(_REL19)}).")
            src, dst = str(src), str(dst)
            _missing = [x for x in (src, dst)
                        if x not in _cir_ids and x not in self.nodes]
            if _missing:
                raise ValueError(
                    f"add_cir : extrémité(s) pendante(s) {_missing} "
                    "(ni dans le CIR ni dans le graphe).")
            if "session_origin" not in attrs:
                attrs["session_origin"] = session_origin
                attrs["session_weight"] = weight
                raw = attrs.get("confidence")
                if isinstance(raw, (int, float)) and _m.isfinite(raw):
                    attrs["confidence"] = max(0.0, min(1.0, float(raw) * weight))
            self.edges.append((src, dst, attrs, src_text))
            self.adjacency[src].append(dst)
            self.reverse_adj[dst].append(src)
        self._adds += 1

    def add_discourse_block(self, merged_cir: dict, session_origin: str | None = None) -> None:
        """Ajoute un bloc de discours multi-phrases (ids pré-préfixés bNNNNN_nMMM).

        Format réel produit par index.py : b{block_idx:05d}_ (lettre b, 5 chiffres).
        Délègue à add_cir après validation minimale : les ids doivent être
        uniques dans le graphe courant (sinon warn, écrasement documenté).
        L'origine se propage du bloc (requise, cf. add_cir).
        """
        import warnings as _w
        for n in merged_cir.get("nodes", []):
            if isinstance(n, dict) and str(n.get("id")) in self.nodes:
                _w.warn(f"CausalGraph : node id {n.get('id')!r} écrasé par bloc de discours.",
                        UserWarning, stacklevel=2)
        self.add_cir(merged_cir, session_origin=session_origin)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.reverse_adj.clear()
        self._dels += 1  # M2b : destruction → prochaine save en photo

    def _rebuild_adjacency(self) -> None:
        """Reconstruit les index depuis self.edges (après fusion/suppression)."""
        self.adjacency = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        for s, d, _a, _t in self.edges:
            self.adjacency[s].append(d)
            self.reverse_adj[d].append(s)

    def merge_duplicate_nodes(self) -> int:
        """Fusionne les nœuds de même label normalisé (garde le plus petit id).

        Tri lexicographique d'ids (ordre string : '10' < '9') — déterministe
        mais pas numérique ; les ids préfixés (bNNNNN_, qNNNNN_) sont à largeur
        fixe donc l'ordre y est chronologique, les ids moteur bruts non.
        Recâble les arêtes vers le gardé ; origine fusionnée par précédence
        (max, jamais de downgrade). Retourne le nombre de nœuds supprimés.
        Déterministe (tri d'ids). M2a : première source de rangement.
        """
        groups: dict[str, list] = defaultdict(list)
        for nid, nd in self.nodes.items():
            norm = self._normalize(nd.get("label", nid))
            if not norm:
                continue  # sans label : jamais fusionnés (pas d'identité)
            groups[norm].append(nid)
        remap: dict[str, str] = {}
        for key in sorted(groups):
            ids = sorted(groups[key])
            if len(ids) < 2:
                continue
            keeper = ids[0]
            for other in ids[1:]:
                remap[other] = keeper
                oth = self.nodes[other]
                cur = self.nodes[keeper]
                if _SESSION_PRECEDENCE.get(oth.get("session_origin"), -1) > \
                        _SESSION_PRECEDENCE.get(cur.get("session_origin"), -1):
                    cur["session_origin"] = oth["session_origin"]
        if not remap:
            return 0
        for rid in remap:
            del self.nodes[rid]
        self.edges = [
            (remap.get(s, s), remap.get(d, d), a, t) for s, d, a, t in self.edges
        ]
        self._rebuild_adjacency()
        self._dels += 1
        return len(remap)

    #: Clés d'attrs exclues de la dédup (métadonnées volatiles, pas le fait).
    _DEDUP_VOLATILE = frozenset({"session_turn", "_match_level"})

    @staticmethod
    def _freeze(v):
        """Forme canonique hachable (récursive, ordre stable).

        dict → tuple trié par clé (le repr() brut dépend de l'ordre
        d'insertion → dedup manquée) ; list/tuple → tuple ; scalaire
        hachable tel quel ; sinon repr().
        """
        if isinstance(v, dict):
            return tuple((k, CausalGraph._freeze(v[k])) for k in sorted(v))
        if isinstance(v, (list, tuple)):
            return tuple(CausalGraph._freeze(x) for x in v)
        try:
            hash(v)
        except TypeError:
            return repr(v)
        else:
            return v

    @staticmethod
    def _edge_key(attrs: dict) -> tuple:
        """Clé de dédup d'arête : tout l'attrs sauf volatiles (canonique)."""
        return tuple(
            (k, CausalGraph._freeze(attrs[k]))
            for k in sorted(attrs)
            if k not in CausalGraph._DEDUP_VOLATILE
        )

    def merge_duplicate_edges(self) -> int:
        """Déduplique les arêtes identiques (src, dst + attrs non volatiles).

        Garde la première occurrence (ordre d'écriture). Retourne le nombre
        d'arêtes supprimées. M2a : deuxième source de rangement.
        Limite actée : `session_turn` est volatile donc deux faits identiques
        de tours distincts fusionnent — `revoke(tour)` ultérieur peut ne rien
        trouver (tour absorbé) ou retirer le survivant (tour gardé). Ne se
        produit que passé les plafonds (seul déclencheur du merge).
        """
        seen: set = set()
        keep = []
        for s, d, a, t in self.edges:
            key = (s, d, self._edge_key(a if isinstance(a, dict) else {}))
            if key in seen:
                continue
            seen.add(key)
            keep.append((s, d, a, t))
        removed = len(self.edges) - len(keep)
        if removed:
            self.edges = keep
            self._rebuild_adjacency()
            self._dels += 1
        return removed

    def _merge_node_dict(self, nid: str, n: dict) -> None:
        """Fusion brute last-wins (replay du journal M2b : l'ordre = écritures)."""
        cur = self.nodes.get(nid)
        if cur is None:
            self.nodes[nid] = dict(n)
        else:
            for k, v in n.items():
                if k != "id":
                    cur[k] = v

    def drop_edges(self, pred) -> int:
        """Supprime les arêtes vérifiant pred(src, dst, attrs) ; reconstruit l'adjacence.

        Retourne le nombre d'arêtes supprimées. Utilisé par la révocation 9e
        (jamais le corpus : le prédicat filtre sur session_origin).
        """
        keep = [(s, d, a, t) for s, d, a, t in self.edges if not pred(s, d, a)]
        removed = len(self.edges) - len(keep)
        if removed:
            self.edges = keep
            self._rebuild_adjacency()
            self._dels += 1
        return removed

    def save(self, path) -> None:
        """Persiste le graphe en JSON (écriture atomique tmp+replace).

        Sans atomicité, un crash mid-write corrompt graph.json et la session
        suivante repart vide (seul le journal post-photo serait rejoué).
        """
        import json
        data = {
            "nodes": self.nodes,
            "edges": [
                {"src": src, "dst": dst, "attrs": attrs, "text": text}
                for src, dst, attrs, text in self.edges
            ],
        }
        from pathlib import Path as _P
        _p = _P(path)
        _tmp = _p.with_suffix(".tmp.json")
        _tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _tmp.replace(_p)

    @classmethod
    def load(cls, path) -> CausalGraph:
        """Charge un graphe depuis un fichier JSON produit par save().

        Migration 9a : nœuds/arêtes sans session_origin (graphes pré-9a) →
        document + migrated:true (distinguer l'attesté du supposé).
        Confiance inchangée (poids document = 1.0).
        """
        import json
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(  # noqa: TRY004 — ValueError = convention API validation (données invalides)
                f"CausalGraph.load : {path} n'est pas un graphe "
                f"(racine {type(data).__name__}, objet JSON attendu).")
        g = cls()
        _nodes = data.get("nodes", {})
        if _nodes is None:
            _nodes = {}
        if not isinstance(_nodes, dict):
            raise ValueError(  # noqa: TRY004 — ValueError = convention API validation (données invalides)
                f"CausalGraph.load : {path} a des nœuds invalides "
                f"({type(_nodes).__name__}, objet attendu).")
        # Clés re-stringifiées (vieux fichiers avec clés entières -> str, cf add_cir).
        g.nodes = {str(k): v for k, v in _nodes.items()}
        for nd in g.nodes.values():
            if isinstance(nd, dict) and "session_origin" not in nd:
                nd["session_origin"] = "document"
                nd["migrated"] = True
        _edges = data.get("edges", [])
        if _edges is None:
            _edges = []
        if not isinstance(_edges, list):
            raise ValueError(  # noqa: TRY004 — ValueError = convention API validation (données invalides)
                f"CausalGraph.load : {path} a des arêtes invalides "
                f"({type(_edges).__name__}, liste attendue).")
        for i, e in enumerate(_edges):
            if not isinstance(e, dict):
                raise ValueError(  # noqa: TRY004 — ValueError = convention API validation (données invalides)
                    f"CausalGraph.load : {path} arête #{i} invalide "
                    f"({type(e).__name__}, objet attendu).")
            try:
                src = str(e["src"])
                dst = str(e["dst"])
                attrs = e["attrs"]
            except KeyError as exc:
                raise ValueError(
                    f"CausalGraph.load : {path} arête #{i} incomplète "
                    f"(clé {exc} manquante).") from exc
            if isinstance(attrs, dict) and "session_origin" not in attrs:
                attrs["session_origin"] = "document"
                attrs["session_weight"] = 1.0
                attrs["migrated"] = True
            text  = e.get("text", "")
            g.edges.append((src, dst, attrs, text))
            g.adjacency[src].append(dst)
            g.reverse_adj[dst].append(src)
        return g

    @classmethod
    def from_cirs(cls, cirs: list, session_origin: str | None = None) -> CausalGraph:
        """Construit depuis une liste de CIR (origine 9a requise, cf. add_cir)."""
        g = cls()
        for cir in cirs:
            g.add_cir(cir, session_origin=session_origin)
        return g

    @staticmethod
    def _safe_conf(attrs: dict, key: str = "confidence", default: float = 1.0) -> float:
        """Extract confidence as float, clamped [0,1]. Returns default on None/NaN/Inf."""
        import math as _m
        c = attrs.get(key)
        if c is None:
            return default
        try:
            v = float(c)
        except (TypeError, ValueError):
            return default
        if _m.isnan(v) or _m.isinf(v):
            return default
        return max(0.0, min(1.0, v))

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalise un label : ligatures, fold accents (NFD→ASCII), lowercase, strip ponctuation."""
        import re as _re
        import unicodedata as _ud
        pre = (s or "").replace('œ', 'oe').replace('Œ', 'oe').replace('æ', 'ae').replace('Æ', 'ae')
        nfd = _ud.normalize('NFD', pre)
        ascii_str = nfd.encode('ascii', 'ignore').decode('ascii')
        return _re.sub(r'[^a-z0-9 _\-]', '', ascii_str.lower()).strip()

    @staticmethod
    def _match_level(label: str, keyword: str) -> int | None:
        """1=exact, 2=mot complet (word-boundary), 3=préfixe, 4=substring, None=pas de match.
        Normalise les deux termes avant comparaison (accents, casse, ponctuation)."""
        import re as _re
        lab = CausalGraph._normalize(label)
        kw = CausalGraph._normalize(keyword)
        if not kw or not lab:
            return None
        if lab == kw:
            return 1
        if _re.search(r'\b' + _re.escape(kw) + r'\b', lab):
            return 2
        if lab.startswith(kw):
            return 3
        if kw in lab:
            return 4
        return None

    def find_causes(self, keyword: str, min_level: int = 2) -> list[tuple]:
        """Arêtes dont la cible matche keyword. attrs['_match_level'] stocké sans breaking (4-tuple conservé)."""
        import warnings as _w
        scored = []
        n_substring = 0
        for idx, (src, dst, attrs, text) in enumerate(self.edges):
            level = self._match_level(self.nodes.get(dst, {}).get("label", ""), keyword)
            if level is None or level > min_level:
                continue
            if not isinstance(attrs, dict):
                attrs = {}
            attrs = dict(attrs)
            attrs["_match_level"] = level
            conf = attrs.get("confidence")
            try:
                conf_val = float(conf) if conf is not None else 0.5
            except (TypeError, ValueError):
                conf_val = 0.5
            if level == 4:
                n_substring += 1
            scored.append((level, -conf_val, idx, (src, dst, attrs, text)))
        if n_substring:
            _w.warn(f"find_causes : {n_substring} match(s) substring (niveau 4) — qualité dégradée.",
                    UserWarning, stacklevel=2)
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in scored]

    def find_effects(self, keyword: str, min_level: int = 2) -> list[tuple]:
        """Arêtes dont la source matche keyword. Même convention que find_causes."""
        import warnings as _w
        scored = []
        n_substring = 0
        for idx, (src, dst, attrs, text) in enumerate(self.edges):
            level = self._match_level(self.nodes.get(src, {}).get("label", ""), keyword)
            if level is None or level > min_level:
                continue
            if not isinstance(attrs, dict):
                attrs = {}
            attrs = dict(attrs)
            attrs["_match_level"] = level
            conf = attrs.get("confidence")
            try:
                conf_val = float(conf) if conf is not None else 0.5
            except (TypeError, ValueError):
                conf_val = 0.5
            if level == 4:
                n_substring += 1
            scored.append((level, -conf_val, idx, (src, dst, attrs, text)))
        if n_substring:
            _w.warn(f"find_effects : {n_substring} match(s) substring (niveau 4).",
                    UserWarning, stacklevel=2)
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in scored]

    def find_path(self, kw_from: str, kw_to: str) -> list:
        """BFS déterministe : src_ids et voisins triés (seed-agnostique). Matching word-boundary normalisé."""
        import re as _re
        _nf = CausalGraph._normalize(kw_from)
        _nt = CausalGraph._normalize(kw_to)
        if not _nf or not _nt:
            return []
        _from = _re.compile(r'\b' + _re.escape(_nf) + r'\b')
        _to   = _re.compile(r'\b' + _re.escape(_nt) + r'\b')
        src_ids = sorted(nid for nid, n in self.nodes.items()
                         if _from.search(CausalGraph._normalize(n.get("label", ""))))
        dst_ids = {nid for nid, n in self.nodes.items()
                   if _to.search(CausalGraph._normalize(n.get("label", "")))}
        for start in src_ids:
            visited = {start}
            queue = deque([[start]])
            while queue:
                path = queue.popleft()
                if path[-1] in dst_ids:
                    return path
                for nb in sorted(self.adjacency.get(path[-1], [])):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append([*path, nb])
        return []

    def _edge_between(self, src_id, dst_id) -> dict | None:
        for s, d, attrs, _ in self.edges:
            if s == src_id and d == dst_id:
                return attrs
        return None

    def find_temporal_path(self, kw_from: str, kw_to: str) -> tuple[list, bool]:
        path = self.find_path(kw_from, kw_to)
        if not path:
            return [], False
        ordered = True
        for i in range(len(path) - 1):
            ti_s = self.nodes.get(path[i], {}).get("temporal_index")
            ti_d = self.nodes.get(path[i + 1], {}).get("temporal_index")
            if ti_s is not None and ti_d is not None and ti_s > ti_d:
                ordered = False
                break
        return path, ordered

    @staticmethod
    def _gap_value(g) -> int | None:
        """Extract a scalar from a TemporalGap (dict {min,max,nature}) or a plain int."""
        if isinstance(g, dict):
            lo = g.get("min")
            hi = g.get("max")
            if lo is not None and hi is not None:
                return (int(lo) + int(hi)) // 2
            return int(lo) if lo is not None else (int(hi) if hi is not None else None)
        return int(g)

    def estimate_delay(self, kw_from: str, kw_to: str) -> dict:
        path = self.find_path(kw_from, kw_to)
        if not path:
            return {"found": False, "gap_sum": None, "index_delta": None}
        gap_total: int | None = 0
        idx_delta: int | None = 0
        for i in range(len(path) - 1):
            attrs = self._edge_between(path[i], path[i + 1]) or {}
            g = attrs.get("temporal_gap")
            if g is not None:
                if gap_total is None:
                    pass
                else:
                    try:
                        gap_total = gap_total + self._gap_value(g)
                    except (TypeError, ValueError):
                        gap_total = None
            ti_s = self.nodes.get(path[i], {}).get("temporal_index")
            ti_d = self.nodes.get(path[i + 1], {}).get("temporal_index")
            if ti_s is not None and ti_d is not None:
                if idx_delta is not None:
                    idx_delta = idx_delta + (ti_d - ti_s)
            else:
                idx_delta = None
        return {"found": True, "gap_sum": gap_total, "index_delta": idx_delta}

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

    def add_cir(self, cir: dict, session_origin: str | None = None) -> None:
        self.graph.add_cir(cir, session_origin=session_origin)

    def clear(self) -> None:
        self.graph.clear()

    def execute(self, text: str) -> str | None:
        """
        Parse et exécute une commande. Retourne None si c'est du texte à analyser.
        """
        cmd, arg = parse_command(text)

        if cmd == "text":
            return None  # à analyser, pas une commande

        if cmd == "explain":
            return self._explain(arg)
        if cmd == "abduct":
            return self._abduct(arg)
        if cmd == "effects":
            return self._effects(arg)
        if cmd == "chain":
            return self._chain(arg)
        if cmd == "analogy":
            return self._analogy(arg)
        if cmd == "zoom_in":
            return self._zoom_in(arg)
        if cmd == "zoom_out":
            return self._zoom_out(arg)
        if cmd == "aggregate":
            return self._aggregate(arg)
        if cmd == "centrality":
            return self._centrality(arg)
        if cmd == "spof":
            return self._spof()
        if cmd == "diff":
            return self._diff(arg)
        if cmd == "density":
            return self._density(arg)
        if cmd == "coverage":
            return self._coverage(arg)
        if cmd == "reliability":
            return self._reliability(arg)
        if cmd == "chain_t":
            return self._chain_t(arg)
        if cmd == "before":
            return self._before(arg)
        if cmd == "delay":
            return self._delay(arg)
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

    def _explain(self, keyword: str | None) -> str:
        if not keyword:
            return "explain: <concept>  — missing argument"
        causes = self.graph.find_causes(keyword)
        if not causes:
            return f"explain: no causes found for '{keyword}'"
        lines = [f"explain: {keyword}"]
        for src, dst, attrs, _src_text in causes:
            conf = attrs.get("confidence")
            rel  = attrs.get("relation", "?")
            neg  = " [negated]" if attrs.get("negated") else ""
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            lines.append(
                f"  {self.graph._label(src)}"
                f"  --[{rel}]{neg}-->"
                f"  {self.graph._label(dst)}"
                f"  conf={conf_s}{edge_origin_tag(attrs)}"
            )
        note = origin_only_note(causes)
        if note:
            lines.append(note)
        return "\n".join(lines)

    def _abduct(self, keyword: str | None) -> str:
        if not keyword:
            return "abduct: <effet>  — missing argument"
        from collections import deque as _dq
        effect_ids = [nid for nid, nd in self.graph.nodes.items()
                      if CausalGraph._match_level(nd.get("label", ""), keyword) is not None]
        if not effect_ids:
            return f"abduct: node '{keyword}' not found"

        edge_map = {}
        for s, d, attrs, _ in self.graph.edges:
            edge_map[(s, d)] = attrs

        candidates = []
        visited = set(effect_ids)
        queue = _dq()
        for eid in effect_ids:
            for pred in self.graph.reverse_adj.get(eid, []):
                if pred not in visited:
                    visited.add(pred)
                    edge_attrs = edge_map.get((pred, eid), {})
                    edge_conf = CausalGraph._safe_conf(edge_attrs, default=0.5)
                    queue.append((pred, 1, edge_conf, eid))
                    score = edge_conf / 2.0
                    candidates.append((
                        score,
                        self.graph._label(pred),
                        edge_attrs.get("relation", "?"),
                        edge_conf,
                        1,
                    ))

        while queue:
            ni, depth, path_conf, _ = queue.popleft()
            for pred in self.graph.reverse_adj.get(ni, []):
                if pred not in visited:
                    visited.add(pred)
                    edge_attrs = edge_map.get((pred, ni), {})
                    edge_conf = CausalGraph._safe_conf(edge_attrs, default=0.5)
                    new_path_conf = min(path_conf, edge_conf)
                    new_depth = depth + 1
                    score = new_path_conf / (1.0 + new_depth)
                    queue.append((pred, new_depth, new_path_conf, ni))
                    candidates.append((
                        score,
                        self.graph._label(pred),
                        edge_attrs.get("relation", "?"),
                        edge_conf,
                        new_depth,
                    ))

        if not candidates:
            return f"abduct: no causal hypotheses found for '{keyword}'"
        candidates.sort(key=lambda t: (-t[0], t[1]))
        lines = [f"abduct: hypotheses for '{keyword}' (score = min_conf_path / (1+depth))"]
        for score, label, rel, conf, depth in candidates:
            lines.append(f"  [{rel}] {label}  conf={conf:.2f}  depth={depth}  score={score:.3f}")
        return "\n".join(lines)

    def _effects(self, keyword: str | None) -> str:
        if not keyword:
            return "effects: <concept>  — missing argument"
        effects = self.graph.find_effects(keyword)
        if not effects:
            return f"effects: no effects found for '{keyword}'"
        lines = [f"effects: {keyword}"]
        for _src, dst, attrs, _ in effects:
            rel = attrs.get("relation", "?")
            neg = " [negated]" if attrs.get("negated") else ""
            lines.append(
                f"  --> [{rel}]{neg}  {self.graph._label(dst)}"
                f"  [{self.graph._type(dst)}]{edge_origin_tag(attrs)}"
            )
        note = origin_only_note(effects)
        if note:
            lines.append(note)
        return "\n".join(lines)

    def _analogy(self, arg: str | None) -> str:
        if not arg or "->" not in arg:
            return "analogy: <from> -> <to>  — missing arguments or '->' separator"
        kw_from, kw_to = [x.strip() for x in arg.split("->", 1)]
        nf = CausalGraph._normalize(kw_from)
        nt = CausalGraph._normalize(kw_to)
        # Trouver l'arête patron
        pattern_edge = None
        for s, d, attrs, _ in self.graph.edges:
            sl = CausalGraph._normalize(self.graph._label(s))
            dl = CausalGraph._normalize(self.graph._label(d))
            if sl == nf and dl == nt:
                pattern_edge = (s, d, attrs)
                break
        if pattern_edge is None:
            return f"analogy: edge '{kw_from}' → '{kw_to}' not found in graph"
        p_src, p_dst, p_attrs = pattern_edge
        p_rel = p_attrs.get("relation", "")
        p_conf = CausalGraph._safe_conf(p_attrs)
        p_type_src = self.graph.nodes.get(p_src, {}).get("node_type", "")
        p_type_dst = self.graph.nodes.get(p_dst, {}).get("node_type", "")
        p_out_deg = sum(1 for s,_,_,_ in self.graph.edges if s == p_src)
        p_in_deg  = sum(1 for _,d,_,_ in self.graph.edges if d == p_dst)
        causal_family = {"cause", "enable", "condition"}
        flow_family = {"data_dependency", "control_dependency", "sequence"}

        def rel_sim(r):
            if r == p_rel:
                return 1.0
            if r in causal_family and p_rel in causal_family:
                return 0.6
            if r in flow_family and p_rel in flow_family:
                return 0.6
            return 0.0

        def type_sim(t, pt):
            if t == pt:
                return 1.0
            dyn = {"action", "transition"}
            sta = {"etat", "etat_systemique"}
            if t in dyn and pt in dyn:
                return 0.5
            if t in sta and pt in sta:
                return 0.5
            return 0.0

        from collections import Counter as _Ctr
        _out_deg = _Ctr(s for s,_,_,_ in self.graph.edges)
        _in_deg  = _Ctr(d for _,d,_,_ in self.graph.edges)
        matches = []
        for s, d, attrs, _ in self.graph.edges:
            if s == p_src and d == p_dst:
                continue
            rel = attrs.get("relation", "")
            conf = CausalGraph._safe_conf(attrs)
            t_src = self.graph.nodes.get(s, {}).get("node_type", "")
            t_dst = self.graph.nodes.get(d, {}).get("node_type", "")
            out_deg = _out_deg[s]
            in_deg  = _in_deg[d]
            rs = rel_sim(rel)
            ts = (type_sim(t_src, p_type_src) + type_sim(t_dst, p_type_dst)) / 2.0
            cp = 1.0 - abs(conf - p_conf)
            d_out_diff = abs(p_out_deg - out_deg)
            d_in_diff  = abs(p_in_deg  - in_deg)
            dp = 1.0 / (1.0 + d_out_diff + d_in_diff)
            score = 0.40 * rs + 0.30 * ts + 0.20 * cp + 0.10 * dp
            if score >= 0.3:
                matches.append((score, self.graph._label(s), self.graph._label(d), rel, conf))
        matches.sort(key=lambda t: (-t[0], t[1]))
        if not matches:
            return f"analogy: no analogues found for '{kw_from}' → '{kw_to}' (score < 0.30)"
        lines = [f"analogy: '{kw_from}' → '{kw_to}'  ({len(matches)} match(es))"]
        for score, fl, tl, rel, conf in matches[:10]:
            lines.append(f"  {fl} --[{rel}]--> {tl}  conf={conf:.2f}  sim={score:.2f}")
        return "\n".join(lines)

    def _zoom_in(self, arg: str | None) -> str:
        if not arg:
            return "zoom_in: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"zoom_in: node '{arg}' not found"
        pid = matching[0]
        children = [(nid, nd) for nid, nd in self.graph.nodes.items()
                    if nd.get("parent") == pid]
        if not children:
            return f"zoom_in: '{arg}' has no children (leaf node or hierarchy not set)"
        lines = [f"zoom_in: '{arg}' ({len(children)} children)"]
        for _, nd in children:
            lines.append(f"  [{nd.get('node_type','?')}] {nd.get('label', '?')}")
        return "\n".join(lines)

    def _zoom_out(self, arg: str | None) -> str:
        if not arg:
            return "zoom_out: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        node = next((nd for nd in self.graph.nodes.values()
                     if CausalGraph._normalize(nd.get("label","")) == kw), None)
        if node is None:
            return f"zoom_out: node '{arg}' not found"
        pid = node.get("parent")
        if pid is None:
            return f"zoom_out: '{arg}' is a root node (no parent)"
        parent = self.graph.nodes.get(pid, {})
        return f"zoom_out: '{arg}' → parent: [{parent.get('node_type','?')}] {parent.get('label','?')}"

    def _aggregate(self, arg: str | None) -> str:
        if not arg:
            return "aggregate: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"aggregate: node '{arg}' not found"
        pid = matching[0]
        child_ids = {nid for nid, nd in self.graph.nodes.items() if nd.get("parent") == pid}
        out_edges = [(s,d,a) for s,d,a,_ in self.graph.edges
                     if s in child_ids and d not in child_ids]
        confs = [CausalGraph._safe_conf(a) for _,_,a in out_edges]
        mean_c = sum(confs) / len(confs) if confs else 0.0
        lines = [f"aggregate: '{arg}'  children={len(child_ids)}  outgoing_edges={len(out_edges)}"]
        lines.append(f"  mean_confidence={mean_c:.2f}")
        for s,d,a in out_edges:
            lines.append(f"  {self.graph._label(s)} --[{a.get('relation','?')}]--> {self.graph._label(d)}")
        return "\n".join(lines)

    def _centrality(self, arg: str | None) -> str:
        if not arg:
            return "centrality: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"centrality: node '{arg}' not found"
        nid = matching[0]
        in_e  = [CausalGraph._safe_conf(a) for s,d,a,_ in self.graph.edges if d==nid]
        out_e = [CausalGraph._safe_conf(a) for s,d,a,_ in self.graph.edges if s==nid]
        total = len(in_e) + len(out_e)
        w_in  = sum(in_e)
        w_out = sum(out_e)
        score = (w_in + w_out) / total if total > 0 else 0.0
        return (f"centrality: '{arg}'\n"
                f"  degree_in={len(in_e)}  degree_out={len(out_e)}\n"
                f"  weighted_in={w_in:.2f}  weighted_out={w_out:.2f}\n"
                f"  centrality_score={score:.3f}")

    _SPOF_MAX_NODES = 500

    def _spof(self) -> str:
        from collections import deque as _dq
        nodes = self.graph.nodes
        adj = self.graph.adjacency

        if len(nodes) > self._SPOF_MAX_NODES:
            return (f"spof: graph too large ({len(nodes)} nodes > {self._SPOF_MAX_NODES}). "
                    f"Use CENTRALITY on specific nodes instead.")

        def reachable_pairs(exclude_id=None):
            count = 0
            for src in nodes:
                if src == exclude_id:
                    continue
                visited = {src}
                if exclude_id is not None:
                    visited.add(exclude_id)
                q = _dq([src])
                while q:
                    cur = q.popleft()
                    for nb in adj.get(cur, []):
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                            count += 1
            return count

        total = reachable_pairs()
        if total == 0:
            return "spof: no reachable pairs in graph"
        scores = []
        for nid, nd in nodes.items():
            cut = total - reachable_pairs(nid)
            scores.append((cut, nd.get("label", str(nid))))
        scores.sort(key=lambda t: (-t[0], t[1]))
        lines = [f"spof: total_reachable_pairs={total}"]
        for cut, lbl in scores[:10]:
            pct = cut / total * 100
            lines.append(f"  {lbl}  paths_cut={cut}  ({pct:.1f}%)")
        return "\n".join(lines)

    def _diff(self, arg: str | None) -> str:
        if not arg or "->" not in arg:
            return "diff: <from> -> <to>  — missing arguments or '->' separator"
        kw_from, kw_to = [x.strip() for x in arg.split("->", 1)]
        path = self.graph.find_path(kw_from, kw_to)
        if not path:
            return (f"diff: '{kw_from}' → '{kw_to}'\n"
                    f"  path: NOT FOUND  gap=1.00  gap_rate=100%")
        confs = []
        for i in range(len(path) - 1):
            attrs = self.graph._edge_between(path[i], path[i+1]) or {}
            confs.append(CausalGraph._safe_conf(attrs))
        min_conf = min(confs) if confs else 0.0
        gap = 1.0 - min_conf
        return (f"diff: '{kw_from}' → '{kw_to}'\n"
                f"  observed_confidence={min_conf:.2f}  gap={gap:.2f}  gap_rate={gap*100:.1f}%\n"
                f"  path_length={len(path)-1} edges")

    def _density(self, arg: str | None) -> str:
        n = len(self.graph.nodes)
        e = len(self.graph.edges)
        global_d = e / (n * (n - 1)) if n > 1 else 0.0
        if not arg:
            return f"density: global={global_d:.3f}  ({n} nodes, {e} edges)"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"density: node '{arg}' not found"
        nid = matching[0]
        degree = sum(1 for s,d,_,_ in self.graph.edges if nid in (s, d))
        local_d = degree / (2 * (n - 1)) if n > 1 else 0.0
        return (f"density: '{arg}'  local={local_d:.3f}  degree={degree}\n"
                f"  global={global_d:.3f}  ({n} nodes, {e} edges)")

    def _coverage(self, arg: str | None) -> str:
        if not arg:
            return "coverage: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"coverage: node '{arg}' not found"
        nid = matching[0]
        incident = [(s,d,a) for s,d,a,_ in self.graph.edges if nid in (s, d)]
        degree = len(incident)
        n_prov = sum(1 for _,_,a in incident if a.get("provenance") is not None)
        n_e = len(self.graph.edges)
        cov = degree / n_e if n_e > 0 else 0.0
        prov_r = n_prov / degree if degree > 0 else 1.0
        return (f"coverage: '{arg}'\n"
                f"  degree={degree}  provenance={n_prov}/{degree}\n"
                f"  coverage_score={cov:.3f}  provenance_ratio={prov_r:.2f}")

    def _reliability(self, arg: str | None) -> str:
        if not arg:
            return "reliability: <concept>  — missing argument"
        kw = CausalGraph._normalize(arg)
        matching = [nid for nid, nd in self.graph.nodes.items()
                    if CausalGraph._normalize(nd.get("label","")) == kw]
        if not matching:
            return f"reliability: node '{arg}' not found"
        nid = matching[0]
        confs = []
        n_prov = 0
        for s,d,attrs,_ in self.graph.edges:
            if nid in (s, d):
                c = attrs.get("confidence")
                try:
                    confs.append(float(c) if c is not None else 1.0)
                except (TypeError, ValueError):
                    confs.append(1.0)
                if attrs.get("provenance") is not None:
                    n_prov += 1
        if not confs:
            return f"reliability: '{arg}' has no incident edges"
        mean_c = sum(confs) / len(confs)
        min_c = min(confs)
        prov_r = n_prov / len(confs)
        score = mean_c * prov_r
        return (f"reliability: '{arg}'\n"
                f"  mean_conf={mean_c:.2f}  min_conf={min_c:.2f}\n"
                f"  provenance_ratio={prov_r:.2f}  reliability_score={score:.3f}")

    @staticmethod
    def _split_two(arg: str, cmd: str) -> tuple[str, str] | str:
        """Split arg into two concepts using '->' or ',' separator, fallback to whitespace."""
        if not arg:
            return f"{cmd}: <concept_a> -> <concept_b>  — missing arguments"
        if "->" in arg:
            a, b = [x.strip() for x in arg.split("->", 1)]
        elif "," in arg:
            a, b = [x.strip() for x in arg.split(",", 1)]
        else:
            parts = arg.split()
            if len(parts) < 2:
                return f"{cmd}: provide two concepts separated by '->'"
            a, b = parts[0], parts[-1]
        if not a or not b:
            return f"{cmd}: provide two non-empty concepts"
        return (a, b)

    def _chain(self, arg: str | None) -> str:
        r = self._split_two(arg, "chain")
        if isinstance(r, str):
            return r
        kw_from, kw_to = r
        path = self.graph.find_path(kw_from, kw_to)
        if not path:
            return f"chain: no causal path found from '{kw_from}' to '{kw_to}'"
        steps = [
            f"[{self.graph._type(nid)}] {self.graph._label(nid)}"
            for nid in path
        ]
        return "chain:\n  " + " --> ".join(steps)

    def _chain_t(self, arg: str | None) -> str:
        r = self._split_two(arg, "chain_t")
        if isinstance(r, str):
            return r
        kw_from, kw_to = r
        path, ordered = self.graph.find_temporal_path(kw_from, kw_to)
        if not path:
            return f"chain_t: no causal path found from '{kw_from}' to '{kw_to}'"
        steps = [
            f"[ti={self.graph.nodes.get(nid, {}).get('temporal_index', '?')}] "
            f"{self.graph._label(nid)}"
            for nid in path
        ]
        order_tag = "temporally ordered" if ordered else "NOT temporally ordered"
        return f"chain_t [{order_tag}]:\n  " + " --> ".join(steps)

    def _before(self, arg: str | None) -> str:
        r = self._split_two(arg, "before")
        if isinstance(r, str):
            return r
        a, b = r
        # Find best matching nodes
        a_node = next(
            (n for n in self.graph.nodes.values() if CausalGraph._normalize(n.get("label", "")) == CausalGraph._normalize(a)),
            None
        )
        b_node = next(
            (n for n in self.graph.nodes.values() if CausalGraph._normalize(n.get("label", "")) == CausalGraph._normalize(b)),
            None
        )
        ti_a = a_node.get("temporal_index") if a_node else None
        ti_b = b_node.get("temporal_index") if b_node else None
        path = self.graph.find_path(a, b)
        path_exists = bool(path)
        if ti_a is not None and ti_b is not None:
            a_before_b = ti_a < ti_b and path_exists
        else:
            a_before_b = path_exists
        verdict = "YES" if a_before_b else "NO"
        return (
            f"before: '{a}' before '{b}'? {verdict}\n"
            f"  temporal_index: {a}={ti_a} / {b}={ti_b}\n"
            f"  directed path: {'exists' if path_exists else 'not found'}"
        )

    def _delay(self, arg: str | None) -> str:
        r = self._split_two(arg, "delay")
        if isinstance(r, str):
            return r
        kw_from, kw_to = r
        res = self.graph.estimate_delay(kw_from, kw_to)
        if not res["found"]:
            return f"delay: no causal path found from '{kw_from}' to '{kw_to}'"
        gap = res["gap_sum"]
        delta = res["index_delta"]
        lines = [f"delay: '{kw_from}' → '{kw_to}'"]
        if gap is not None:
            lines.append(f"  temporal_gap sum: {gap} units")
        if delta is not None:
            lines.append(f"  temporal_index delta: {delta} steps")
        if gap is None and delta is None:
            lines.append("  no temporal metadata available on this path")
        return "\n".join(lines)

    def _counterfactual(self, keyword: str | None) -> str:
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
            conf = attrs.get("confidence")
            conf_s = f"({conf:.0%})" if isinstance(conf, (int, float)) else "(?)"
            lines.append(
                f"  {i}. [{self.graph._type(src)}] {self.graph._label(src)}"
                f"  --[{rel}{neg}]-->"
                f"  [{self.graph._type(dst)}] {self.graph._label(dst)}"
                f"  {conf_s}"
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
