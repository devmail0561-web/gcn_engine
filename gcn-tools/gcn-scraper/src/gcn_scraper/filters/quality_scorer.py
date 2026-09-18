"""Score de qualité neutre — longueur, densité lexicale, propreté.

N'utilise PAS de mots-clés causaux : le scraper collecte du texte brut diversifié,
l'annotation causale est faite séparément par le moteur GCN.
"""
from __future__ import annotations
import re

_REPEAT_RE = re.compile(r'(.{10,})\1{2,}')  # détecte les répétitions de blocs


class QualityScorer:
    """Score de qualité d'une phrase sur [0, 1]."""

    def score(self, text: str) -> float:
        """
        Calcule un score de qualité neutre.

        Critères :
        - Longueur 30–600 chars
        - Au moins 5 mots
        - Ratio chars/mots dans [3, 15] (texte propre ≈ 5-8)
        - Pas de répétitions de blocs
        """
        if not text:
            return 0.0
        n = len(text)
        if n < 30 or n > 600:
            return 0.0
        words = text.split()
        if len(words) < 5:
            return 0.2
        ratio = n / len(words)
        if ratio < 3 or ratio > 15:
            return 0.4
        if _REPEAT_RE.search(text):
            return 0.3
        # Score longueur : optimum à 150-300 chars
        if n <= 300:
            length_score = min(1.0, n / 150)
        else:
            length_score = max(0.5, 1.0 - (n - 300) / 300)
        return round(length_score, 2)
