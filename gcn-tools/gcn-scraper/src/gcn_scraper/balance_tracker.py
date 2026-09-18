"""Suivi du volume de phrases par langue — contrôle l'équilibre FR/EN du dataset.

Le scraper collecte du texte brut ; l'équilibre se fait par langue et source,
PAS par type de relation causale (c'est le moteur GCN qui annotera les relations).
"""
from __future__ import annotations
from collections import defaultdict

DEFAULT_BUDGET: dict[str, int] = {
    "fr": 25000,  # 50 % français
    "en": 25000,  # 50 % anglais
}


class BalanceTracker:
    """Suit le budget par langue et décide si une source est saturée."""

    def __init__(self, budget: dict[str, int] | None = None):
        self.budget: dict[str, int] = budget or dict(DEFAULT_BUDGET)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, lang: str) -> None:
        """Enregistre une phrase pour la langue détectée."""
        key = lang if lang in self.budget else "fr"
        self.counts[key] += 1

    def is_full(self, lang: str) -> bool:
        """True si le budget pour cette langue est atteint."""
        key = lang if lang in self.budget else "fr"
        return self.counts[key] >= self.budget.get(key, 25000)

    def is_globally_full(self) -> bool:
        """True si toutes les langues ont atteint leur budget."""
        return all(self.counts.get(k, 0) >= v for k, v in self.budget.items())

    def summary(self) -> dict[str, str]:
        """Résumé 'actuel/budget' par langue."""
        return {k: f"{self.counts.get(k, 0)}/{self.budget[k]}" for k in self.budget}

    def total(self) -> int:
        return sum(self.counts.values())

    def progress_bar(self, width: int = 40) -> str:
        total_budget = sum(self.budget.values())
        filled = min(self.total(), total_budget)
        pct = filled / total_budget if total_budget else 0
        bar = "█" * int(pct * width) + "░" * (width - int(pct * width))
        return f"[{bar}] {filled}/{total_budget} ({pct*100:.1f}%)"
