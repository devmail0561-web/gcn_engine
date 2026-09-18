"""Suivi du volume de phrases par type de relation — contrôle l'équilibre du dataset."""
from __future__ import annotations
from collections import defaultdict

DEFAULT_BUDGET: dict[str, int] = {
    "cause": 6000,
    "enable": 4500,
    "prevent": 4500,
    "condition": 6000,
    "concession": 4000,
    "sequence": 5000,
    "motivation": 4500,
    "filter": 3500,
    "opposition": 4500,
    "data_dependency": 3500,
    "control_dependency": 4000,
    "unknown": 1000,
}


class BalanceTracker:
    """Suit le budget par type de relation et décide si une source est saturée."""

    def __init__(self, budget: dict[str, int] | None = None):
        self.budget: dict[str, int] = budget or dict(DEFAULT_BUDGET)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, hints: list[str]) -> None:
        """Enregistre une phrase avec ses hints détectés."""
        if hints:
            for h in hints:
                self.counts[h] += 1
        else:
            self.counts["unknown"] += 1

    def is_full(self, hints: list[str]) -> bool:
        """True si tous les hints de cette phrase ont atteint leur budget."""
        if not hints:
            return self.counts["unknown"] >= self.budget.get("unknown", 1000)
        return all(self.counts[h] >= self.budget.get(h, 4000) for h in hints)

    def is_globally_full(self) -> bool:
        """True si tous les types ont atteint leur budget."""
        return all(self.counts.get(k, 0) >= v for k, v in self.budget.items())

    def summary(self) -> dict[str, str]:
        """Résumé sous forme 'actuel/budget' par type."""
        return {k: f"{self.counts.get(k, 0)}/{self.budget[k]}" for k in self.budget}

    def total(self) -> int:
        return sum(self.counts.values())

    def progress_bar(self, width: int = 40) -> str:
        """Barre de progression textuelle."""
        total_budget = sum(self.budget.values())
        filled = min(self.total(), total_budget)
        pct = filled / total_budget if total_budget else 0
        bar = "█" * int(pct * width) + "░" * (width - int(pct * width))
        return f"[{bar}] {filled}/{total_budget} ({pct*100:.1f}%)"
