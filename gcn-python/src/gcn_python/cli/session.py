# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Session persistante gcn-discuss / gcn-index (anti-perte de discussion).

Une session = un répertoire contenant :
- graph.json          : graphe causal (CausalGraph.save/load)
- session_vecs.npz    : vecteurs enrichis par clé stable (graph_vecs)
- session_vecs.manifest.json : manifest auto (checkpoint_hash, d_eff, created_at)
- history.jsonl       : échanges (analyze + questions/réponses), un JSON par ligne

Rien n'est obligatoire : chaque pièce manquante -> warn + fallback
(graphe vide, vecs None, historique vide). Le moteur tourne sans session.
"""
from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path

import numpy as np

from ..data.graph_vecs import checkpoint_hash, save_graph_vecs, stable_key
from ..verbalizer.instructions import CausalGraph

GRAPH_NAME = "graph.json"
VECS_NAME = "session_vecs.npz"
HISTORY_NAME = "history.jsonl"
GRAPH_LOG_NAME = "graph_log.jsonl"

#: Plafond de vecteurs en mémoire (FIFO silencieux : cache recomputable
#: par ré-analyse — pas d'annonce, cf. M2a).
SESSION_MAX_VECS = 5000

#: Plafond d'entrées d'historique conservées en mémoire (déjà persistées).
#: Le disque (history.jsonl, audit) garde tout ; la liste mémoire est
#: élaguée des entrées `_saved` au-delà du cap à chaque save().
HISTORY_KEEP_SAVED = 1000


def _node_hash(node: dict) -> str:
    """Empreinte canonique d'un nœud (détection de mise à jour M2b)."""
    import hashlib
    return hashlib.sha1(
        json.dumps(node, sort_keys=True, ensure_ascii=False, default=str
                   ).encode("utf-8"), usedforsecurity=False).hexdigest()


class SessionStore:
    """Graphe + vecs + historique d'une session de discussion."""

    def __init__(self, session_dir: Path | None = None,
                 max_vecs: int = SESSION_MAX_VECS,
                 snapshot_every: int = 10) -> None:
        self.session_dir = Path(session_dir) if session_dir else None
        self.graph = CausalGraph()
        self.vecs: dict[str, np.ndarray] = {}
        self.vecs_meta: dict[str, dict] = {}
        self.history: list[dict] = []
        self.max_vecs = max_vecs
        self.snapshot_every = max(1, int(snapshot_every))
        self._saves = 0
        self._photo_nodes: set[str] = set()
        self._photo_nodes_data: dict[str, str] = {}
        self._photo_edges = 0
        self._photo_adds = 0
        self._photo_dels = 0
        self._checkpoint_hash: str | None = None
        self._d_eff: int | None = None

    # ------------------------------------------------------------------
    # Collecte : après chaque engine.analyze(), appeler collect()
    # ------------------------------------------------------------------
    def collect(self, engine, text: str, cir: dict) -> int:
        """Stocke les vecteurs enrichis du dernier forward + trace l'analyse.

        Retourne le nombre de vecteurs collectés (0 si aucun).
        Ne lève jamais : warn + 0 en cas d'échec.
        """
        try:
            vecs = engine._pipeline.get_enriched_vectors()
        except Exception as exc:  # noqa: BLE001  # résilience : warn + 0 si pipeline indisponible
            warnings.warn(f"session : vecteurs inaccessibles ({exc}) — non collectés.",
                          UserWarning, stacklevel=2)
            return 0
        if vecs is None or len(vecs) == 0:
            return 0
        nodes = cir.get("nodes", []) or []
        n = 0
        for pos, node in enumerate(nodes):
            if pos >= len(vecs):
                break
            key = stable_key(text, pos)
            self.vecs[key] = np.asarray(vecs[pos], dtype=np.float32)
            self.vecs_meta[key] = {
                "node_id": str(node.get("id", pos)),
                "node_label": str(node.get("label", "")),
                "shape": [int(v) for v in np.asarray(vecs[pos]).shape],
                "source_text": text[:200],
            }
            n += 1
        # Mémoriser le hash du checkpoint pour le manifest (une fois)
        if self._checkpoint_hash is None:
            try:
                params = {}
                for i, p in enumerate(engine._pipeline.encoder.parameters()):
                    params[f"encoder_{i}"] = np.asarray(p)
                self._checkpoint_hash = checkpoint_hash(params)
                self._d_eff = int(vecs.shape[1]) if getattr(vecs, "ndim", 1) == 2 else len(vecs[0])
            except Exception as exc:  # noqa: BLE001  # hash best-effort : warn unique, manifest sans hash
                if not getattr(self, "_hash_warned", False):
                    warnings.warn(
                        f"session : hash du checkpoint indisponible ({exc}) — manifest sans hash.",
                        UserWarning, stacklevel=2,
                    )
                    self._hash_warned = True
        # M2a : plafond vecs (FIFO — cache recomputable, éviction silencieuse).
        while len(self.vecs) > self.max_vecs:
            old = next(iter(self.vecs))
            del self.vecs[old]
            self.vecs_meta.pop(old, None)
        self.history.append({
            "ts": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
            "event": "analyze",
            "text": text[:500],
            "n_nodes": len(nodes),
            "n_vecs": n,
        })
        self._prune_memory()
        return n

    def lookup(self, text: str, pos: int) -> np.ndarray | None:
        """Retrouve un vecteur collecté par (texte, position). None si absent."""
        return self.vecs.get(stable_key(text, pos))

    def _prune_memory(self) -> None:
        """Borne mémoire de l'historique entre deux save().

        Sans session_dir, save() part tôt et rien ne purge : usage pure-mémoire
        (bibliothèque, REPL) croissait sans borne (3000+ entrées mesurées).
        Règle : au-delà de 2×HISTORY_KEEP_SAVED, éjecter les plus vieilles
        `_saved` d'abord, puis les plus vieilles tout court. Le disque
        (audit) garde tout ; la RAM n'est pas la piste d'audit.
        """
        cap = 2 * HISTORY_KEEP_SAVED
        while len(self.history) > cap:
            for i, e in enumerate(self.history):
                if e.get("_saved"):
                    del self.history[i]
                    break
            else:
                del self.history[0]

    def record_exchange(self, question: str, answered: bool, *,
                          turn: int | None = None, subject: str | None = None,
                          summary: str | None = None, status: str | None = None,
                          lang: str | None = None, added: int = 0) -> None:
        """Trace un échange. Avec turn (DialogueSession 9e) : entrée journal
        event="turn" rejouable par DialogueSession.from_history ; sans turn :
        forme historique event="query" (compat)."""
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
            "event": "query",
            "question": question[:500],
            "answered": bool(answered),
        }
        if turn is not None:
            entry.update({
                "event": "turn",
                "n": turn,
                "subject": subject,
                "summary": summary,
                "status": status,
                "lang": lang,
                "added": added,
            })
        self.history.append(entry)
        self._prune_memory()

    def record_summary(self, summary: dict) -> None:
        """Persiste une entrée `summary` M2a (vieux tours repliés, audit)."""
        entry = dict(summary)
        entry["ts"] = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
        self.history.append(entry)
        self._prune_memory()

    # ------------------------------------------------------------------
    # Persistance (M2b : photo périodique + journal des ajouts)
    # ------------------------------------------------------------------
    def _write_photo(self) -> None:
        """Photo complète (graphe) + troncature du journal. Converge tout."""
        self.graph.save(self.session_dir / GRAPH_NAME)
        self._photo_nodes = set(self.graph.nodes)
        self._photo_nodes_data = {
            nid: _node_hash(nd) for nid, nd in self.graph.nodes.items()
            if isinstance(nd, dict)}
        self._photo_edges = len(self.graph.edges)
        self._photo_adds = self.graph._adds
        self._photo_dels = self.graph._dels
        log = self.session_dir / GRAPH_LOG_NAME
        if log.exists():
            log.unlink()

    def _append_log(self) -> None:
        """Journal des ajouts depuis la photo (nœuds nouveaux + modifiés + arêtes).

        Les mises à jour de nœuds existants (add_cir upgrade-only, fusions)
        étaient silencieuses → perdues sur crash avant la prochaine photo.
        """
        new_nodes = {nid: nd for nid, nd in self.graph.nodes.items()
                     if nid not in self._photo_nodes}
        changed = {nid: nd for nid, nd in self.graph.nodes.items()
                   if isinstance(nd, dict)
                   and nid in self._photo_nodes
                   and _node_hash(nd) != self._photo_nodes_data.get(nid)}
        logged_nodes = {**new_nodes, **changed}
        new_edges = self.graph.edges[self._photo_edges:]
        if not logged_nodes and not new_edges:
            return
        with open(self.session_dir / GRAPH_LOG_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps({"nodes": logged_nodes, "edges": [
                {"src": s, "dst": d, "attrs": a, "text": t}
                for s, d, a, t in new_edges]}, ensure_ascii=False) + "\n")
        self._photo_nodes.update(new_nodes)
        for nid, nd in logged_nodes.items():
            if isinstance(nd, dict):
                self._photo_nodes_data[nid] = _node_hash(nd)
        self._photo_edges = len(self.graph.edges)
        self._photo_adds = self.graph._adds

    def save(self) -> Path | None:
        """Persiste graphe + vecs + historique. Retourne le répertoire ou None.

        M2b : photo complète à la 1re save puis toutes les `snapshot_every`
        (ou après destruction : revoke/fusion/clear), journal des ajouts
        entre-temps. Historique et vecs : inchangés (append / réécriture).
        """
        if self.session_dir is None:
            return None
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._saves += 1
        g = self.graph
        pure_appends = (
            g._dels == self._photo_dels
            and len(g.nodes) >= len(self._photo_nodes)
            and len(g.edges) >= self._photo_edges
        )
        if ((self._saves - 1) % self.snapshot_every == 0
                or not (self.session_dir / GRAPH_NAME).exists()
                or not pure_appends):
            self._write_photo()
        else:
            self._append_log()
        if self.vecs:
            save_graph_vecs(
                self.session_dir / VECS_NAME,
                self.vecs,
                self._checkpoint_hash or "sha256:unknown",
                self._d_eff or 0,
                self.vecs_meta,
            )
        with open(self.session_dir / HISTORY_NAME, "a", encoding="utf-8") as f:
            for entry in self.history:
                if not entry.get("_saved"):
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    entry["_saved"] = True
        # Cap mémoire : le disque garde tout (audit), la liste ne retient
        # que les non-sauvées + les HISTORY_KEEP_SAVED dernières sauvées.
        if len(self.history) > HISTORY_KEEP_SAVED:
            _unsaved = [e for e in self.history if not e.get("_saved")]
            _saved = [e for e in self.history if e.get("_saved")]
            self.history = _unsaved + _saved[-HISTORY_KEEP_SAVED:]
        return self.session_dir

    def load(self) -> dict:
        """Restaure graphe + vecs + historique. Rapport {graph_edges, vecs, history}.

        M2b : rejoue le journal des ajouts (`graph_log.jsonl`) sur la photo,
        puis re-snapshotte (convergence : toute lecture finit en photo propre).
        Clés ajoutées : `replayed` (arêtes rejouées), `resnapshotted`.
        """
        report = {"graph_edges": 0, "vecs": 0, "history": 0,
                  "replayed": 0, "resnapshotted": False,
                  "skipped": 0, "skipped_history": 0}
        if self.session_dir is None or not self.session_dir.exists():
            return report
        gpath = self.session_dir / GRAPH_NAME
        if gpath.exists():
            try:
                self.graph = CausalGraph.load(gpath)
                report["graph_edges"] = len(self.graph.edges)
            except Exception as exc:  # noqa: BLE001  # graphe corrompu : warn + session vide
                warnings.warn(f"session : graphe illisible ({exc}) — session vide.",
                              UserWarning, stacklevel=2)
        # M2b : rejouer le journal des ajouts, puis re-snapshottter.
        lpath = self.session_dir / GRAPH_LOG_NAME
        if lpath.exists():
            try:
                # Fenêtre de crash photo→unlink : la photo peut déjà contenir
                # des arêtes du journal → doublons au rejouement. Déduplique
                # exacte (clé complète, volatiles inclus : même objet = sauté,
                # ré-ajout légitime à un autre tour = rejoué).
                def _full_key(s, d, a, t):
                    return (s, d, json.dumps(a, sort_keys=True,
                                            ensure_ascii=False, default=str)
                            if isinstance(a, dict) else repr(a), t)
                _seen = {_full_key(s, d, a, t)
                         for s, d, a, t in self.graph.edges}
                _skipped = 0
                with open(lpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            delta = json.loads(line)
                        except json.JSONDecodeError:
                            _skipped += 1  # une ligne pourrie ne jette pas le journal
                            continue
                        try:
                            for nid, nd in (delta.get("nodes", {}) or {}).items():
                                if isinstance(nd, dict):
                                    self.graph._merge_node_dict(str(nid), nd)
                            for e in delta.get("edges", []) or []:
                                s, d = str(e["src"]), str(e["dst"])
                                _t = e.get("text", "")
                                _k = _full_key(s, d, e["attrs"], _t)
                                if _k in _seen:
                                    continue
                                _seen.add(_k)
                                self.graph.edges.append((s, d, e["attrs"], _t))
                                self.graph.adjacency[s].append(d)
                                self.graph.reverse_adj[d].append(s)
                                report["replayed"] += 1
                        except (KeyError, TypeError, AttributeError):
                            _skipped += 1  # entrée malformée : sautée, journal préservé
                            continue
                if _skipped:
                    warnings.warn(
                        f"session : {_skipped} entrée(s) de journal sautée(s) "
                        "(malformées) — reste rejoué.", UserWarning, stacklevel=2)
                    report["skipped"] = _skipped
                report["graph_edges"] = len(self.graph.edges)
                self._write_photo()
                report["resnapshotted"] = True
            except Exception as exc:  # noqa: BLE001  # journal illisible : warn + photo seule
                warnings.warn(f"session : journal illisible ({exc}) — photo seule.",
                              UserWarning, stacklevel=2)
        self._photo_nodes = set(self.graph.nodes)
        self._photo_nodes_data = {
            nid: _node_hash(nd) for nid, nd in self.graph.nodes.items()
            if isinstance(nd, dict)}
        self._photo_edges = len(self.graph.edges)
        self._photo_adds = self.graph._adds
        self._photo_dels = self.graph._dels
        self._saves = 1  # la prochaine save part en journal (photo déjà propre)
        vpath = self.session_dir / VECS_NAME
        if vpath.exists():
            try:
                import numpy as _np
                # M3a : mmap (pages à l'usage ; `np.array` copie par clé —
                # rien ne s'échappe du handle, fermeture sûre).
                handle = _np.load(vpath, allow_pickle=False, mmap_mode="r")
                for k in handle.files:
                    self.vecs[k] = _np.array(handle[k])
                report["vecs"] = len(self.vecs)
                mpath = vpath.with_suffix(".manifest.json")
                if mpath.exists():
                    manifest = json.loads(mpath.read_text(encoding="utf-8"))
                    for k, meta in (manifest.get("entries") or {}).items():
                        if k in self.vecs:
                            self.vecs_meta[k] = meta
            except Exception as exc:  # noqa: BLE001  # npz illisible : warn + vecs ignorés
                warnings.warn(f"session : vecs illisibles ({exc}) — ignorés.",
                              UserWarning, stacklevel=2)
        hpath = self.session_dir / HISTORY_NAME
        if hpath.exists():
            try:
                _hskipped = 0
                with open(hpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self.history.append(json.loads(line))
                        except json.JSONDecodeError:
                            _hskipped += 1  # une ligne pourrie ne jette pas la queue
                            continue
                if _hskipped:
                    warnings.warn(
                        f"session : {_hskipped} entrée(s) d'historique sautée(s) "
                        "(malformées) — reste chargé.", UserWarning, stacklevel=2)
                    report["skipped_history"] = _hskipped
                for e in self.history:
                    e["_saved"] = True
                report["history"] = len(self.history)
            except Exception as exc:  # noqa: BLE001  # fichier illisible : warn + historique ignoré
                warnings.warn(f"session : historique illisible ({exc}) — ignoré.",
                              UserWarning, stacklevel=2)
        return report
