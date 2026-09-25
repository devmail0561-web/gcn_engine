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


class SessionStore:
    """Graphe + vecs + historique d'une session de discussion."""

    def __init__(self, session_dir: Path | None = None) -> None:
        self.session_dir = Path(session_dir) if session_dir else None
        self.graph = CausalGraph()
        self.vecs: dict[str, np.ndarray] = {}
        self.vecs_meta: dict[str, dict] = {}
        self.history: list[dict] = []
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
            except Exception:  # noqa: S110, BLE001  # hash du checkpoint best-effort, silencieux
                pass
        self.history.append({
            "ts": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
            "event": "analyze",
            "text": text[:500],
            "n_nodes": len(nodes),
            "n_vecs": n,
        })
        return n

    def lookup(self, text: str, pos: int) -> np.ndarray | None:
        """Retrouve un vecteur collecté par (texte, position). None si absent."""
        return self.vecs.get(stable_key(text, pos))

    def record_exchange(self, question: str, answered: bool) -> None:
        self.history.append({
            "ts": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
            "event": "query",
            "question": question[:500],
            "answered": bool(answered),
        })

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def save(self) -> Path | None:
        """Persiste graphe + vecs + historique. Retourne le répertoire ou None."""
        if self.session_dir is None:
            return None
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.graph.save(self.session_dir / GRAPH_NAME)
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
        return self.session_dir

    def load(self) -> dict:
        """Restaure graphe + vecs + historique. Rapport {graph_edges, vecs, history}."""
        report = {"graph_edges": 0, "vecs": 0, "history": 0}
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
        vpath = self.session_dir / VECS_NAME
        if vpath.exists():
            try:
                import numpy as _np
                handle = _np.load(vpath, allow_pickle=False)
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
                with open(hpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.history.append(json.loads(line))
                for e in self.history:
                    e["_saved"] = True
                report["history"] = len(self.history)
            except Exception as exc:  # noqa: BLE001  # jsonl corrompu : warn + historique ignoré
                warnings.warn(f"session : historique illisible ({exc}) — ignoré.",
                              UserWarning, stacklevel=2)
        return report
