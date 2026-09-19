"""Checkpoint de reprise : sauvegarde l'état du scraping après chaque source."""
from __future__ import annotations
import json
import os
import tempfile
import warnings
from pathlib import Path

_BUDGET_COUNTS_KEY = "budget_counts"


class ScrapingCheckpoint:
    """Persist le statut de chaque source pour permettre une reprise sans perte."""

    def __init__(self, checkpoint_file: Path):
        self.path = checkpoint_file
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception as exc:
                warnings.warn(
                    f"Checkpoint {self.path} illisible ({exc}) — reprise impossible, "
                    "démarrage fresh.",
                    UserWarning,
                    stacklevel=3,
                )
                return {}
        return {}

    def is_done(self, source_key: str) -> bool:
        return self.state.get(source_key, {}).get("done", False)

    def mark_done(
        self, source_key: str, count: int, budget_counts: dict | None = None
    ) -> None:
        self.state[source_key] = {"done": True, "count": count}
        if budget_counts is not None:
            self.state[_BUDGET_COUNTS_KEY] = dict(budget_counts)
        self.save()

    def get_count(self, source_key: str) -> int:
        return self.state.get(source_key, {}).get("count", 0)

    def get_budget_counts(self) -> dict:
        """Compteurs du BalanceTracker persistés (pour réhydratation en --resume)."""
        saved = self.state.get(_BUDGET_COUNTS_KEY, {})
        return dict(saved) if isinstance(saved, dict) else {}

    def save(self) -> None:
        """Sauvegarde atomique de l'état courant (tmp + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".checkpoint.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def reset(self) -> None:
        self.state = {}
        if self.path.exists():
            self.path.unlink()
