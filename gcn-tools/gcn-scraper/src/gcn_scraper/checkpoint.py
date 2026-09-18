"""Checkpoint de reprise : sauvegarde l'état du scraping après chaque source."""
from __future__ import annotations
import json
from pathlib import Path


class ScrapingCheckpoint:
    """Persist le statut de chaque source pour permettre une reprise sans perte."""

    def __init__(self, checkpoint_file: Path):
        self.path = checkpoint_file
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def is_done(self, source_key: str) -> bool:
        return self.state.get(source_key, {}).get("done", False)

    def mark_done(self, source_key: str, count: int) -> None:
        self.state[source_key] = {"done": True, "count": count}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_count(self, source_key: str) -> int:
        return self.state.get(source_key, {}).get("count", 0)

    def _save(self) -> None:
        """Sauvegarde l'état courant (utilisé pour persister session_timestamp)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self) -> None:
        self.state = {}
        if self.path.exists():
            self.path.unlink()
