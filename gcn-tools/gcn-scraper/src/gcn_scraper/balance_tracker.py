"""Suivi du volume par langue — budget calculé automatiquement depuis sources.yaml.

Agnostique à la langue : ajouter wikipedia_ar dans sources.yaml suffit,
zéro changement de code. Le budget se recalcule à chaque instanciation.
"""
from __future__ import annotations
from collections import defaultdict


def _compute_budget(
    selected_langs: list[str],
    target_total: int,
    include_code: bool = True,
) -> dict[str, int]:
    """Budget proportionnel aux poids des langues sélectionnées.

    include_code=False (--prog-langs none) : pas de bucket "code" — sinon
    is_globally_full() ne devient jamais vrai et 10% du budget est perdu
    (audit-2 Fix 4).
    """
    try:
        from .config.loader import get_config
        cfg = get_config()
    except Exception:
        cfg = {}
    weights: dict[str, float] = {}
    for lang in selected_langs:
        source_cfg = cfg.get("sources", {}).get(f"wikipedia_{lang}", {})
        weights[lang] = float(source_cfg.get("budget_weight", 1.0))
    code_budget = max(int(target_total * 0.10), 500) if include_code else 0
    text_budget = target_total - code_budget
    total_weight = sum(weights.values()) or 1.0
    budget = {lang: max(100, int(text_budget * w / total_weight)) for lang, w in weights.items()}
    if include_code:
        budget["code"] = code_budget
    return budget


def _build_default_budget() -> dict[str, int]:
    """
    Calcule le budget depuis sources.yaml.

    - Découvre toutes les sources wikipedia_XX activées.
    - Pondère par `budget_weight` (défaut 1.0).
    - Réserve 10% du total pour le code (GitHub + docs).
    - FR et EN ont budget_weight=2.0 par défaut dans le YAML.
    """
    try:
        from .config.loader import get_config
        cfg = get_config()
    except Exception:
        return {"fr": 25000, "en": 25000, "code": 5000}

    target: int = cfg.get("scraping", {}).get("target_total", 50000)
    sources: dict = cfg.get("sources", {})

    lang_weights: dict[str, float] = {}
    for key, val in sources.items():
        if key.startswith("wikipedia_") and val.get("enabled", True):
            lang = val.get("lang", key.replace("wikipedia_", ""))
            lang_weights[lang] = float(val.get("budget_weight", 1.0))

    if not lang_weights:
        lang_weights = {"fr": 2.0, "en": 2.0}

    # 10% code, 90% texte
    code_budget = max(int(target * 0.10), 500)
    text_budget = target - code_budget
    total_weight = sum(lang_weights.values())

    budget = {
        lang: max(100, int(text_budget * w / total_weight))
        for lang, w in lang_weights.items()
    }
    budget["code"] = code_budget
    return budget


class BalanceTracker:
    """Suit le budget par langue et décide si une source est saturée.

    Budget auto-calculé depuis sources.yaml — aucun hardcoding.
    Pour changer les proportions : éditer budget_weight dans sources.yaml.
    """

    def __init__(self, budget: dict[str, int] | None = None):
        self.budget: dict[str, int] = budget or _build_default_budget()
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, lang: str) -> None:
        """Enregistre une phrase pour la langue détectée.
        Les langues hors budget sont ignorées — elles ne corrompent pas les compteurs."""
        if lang in self.budget:
            self.counts[lang] += 1

    def is_full(self, lang: str) -> bool:
        """True si le budget pour cette langue est atteint.
        Une langue hors budget retourne True pour bloquer les phrases non budgétées."""
        if lang not in self.budget:
            return True
        return self.counts.get(lang, 0) >= self.budget[lang]

    def is_globally_full(self) -> bool:
        """True si toutes les langues ont atteint leur budget."""
        return all(self.counts.get(k, 0) >= v for k, v in self.budget.items())

    def summary(self) -> dict[str, str]:
        return {k: f"{self.counts.get(k, 0)}/{v}" for k, v in self.budget.items()}

    def total(self) -> int:
        return sum(self.counts.values())

    def progress_bar(self, width: int = 40) -> str:
        total_budget = sum(self.budget.values())
        filled = min(self.total(), total_budget)
        pct = filled / total_budget if total_budget else 0
        bar = "█" * int(pct * width) + "░" * (width - int(pct * width))
        return f"[{bar}] {filled}/{total_budget} ({pct*100:.1f}%)"
