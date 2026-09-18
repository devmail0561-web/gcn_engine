"""Scraper Wikipedia EN — catégories ciblées sur les 11 types de relations GCN."""
from __future__ import annotations
import requests
import time


class WikipediaENScraper:
    """Récupère des articles de Wikipedia EN via l'API officielle."""

    CATEGORIES: list[str] = [
        # cause / enable / prevent
        "Epidemiology", "Causality", "Pharmacology", "Climate_change",
        "Toxicology", "Pathophysiology", "Ecology",
        # condition / filter
        "Contract_law", "Constitutional_law", "Legal_reasoning",
        "Formal_methods", "Logic", "Probability_theory",
        # concession / opposition
        "Philosophy_of_science", "Critical_thinking", "Debate",
        "Political_philosophy",
        # sequence
        "Algorithms", "Computer_science", "Chemical_engineering",
        "History", "Protocols_(science)",
        # motivation
        "Decision_theory", "Behavioral_economics", "Psychology",
        # data_dependency / control_dependency
        "Software_engineering", "Machine_learning", "Operating_systems",
        "Distributed_computing", "Programming_paradigms",
    ]

    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def get_category_members(self, category: str, limit: int = 50) -> list[str]:
        """Récupère les titres d'articles d'une catégorie."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "cmtype": "page",
            "format": "json",
        }
        try:
            resp = self.session.get(self.API_URL, params=params, timeout=30)
            if resp.status_code != 200:
                return []
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            return [m["title"] for m in members if m.get("ns") == 0]
        except Exception:
            return []

    def get_article_text(self, title: str, max_chars: int = 5000) -> str | None:
        """Récupère le texte brut d'un article via l'API extracts."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
        }
        try:
            resp = self.session.get(self.API_URL, params=params, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return extract[:max_chars]
        except Exception:
            pass
        return None

    def scrape(self, categories: list[str] | None = None, max_per_category: int = 50) -> list[dict]:
        """Scrape des articles de Wikipedia EN."""
        if categories is None:
            categories = self.CATEGORIES

        results = []
        for cat in categories:
            print(f"  EN/{cat}...", end=" ", flush=True)
            titles = self.get_category_members(cat, limit=max_per_category)
            count = 0
            for title in titles[:max_per_category]:
                text = self.get_article_text(title)
                if text and len(text) > 200:
                    results.append({
                        "source": f"wikipedia_en:{cat}",
                        "category": cat,
                        "title": title,
                        "text": text,
                        "lang": "en",
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    })
                    count += 1
                time.sleep(0.2)
            print(f"{count} articles")
        return results
