"""Scraper Wikipedia EN pour le dataset causale."""

import requests
import time
import re


class WikipediaENScraper:
    """Recupere des articles de Wikipedia en anglais."""

    CATEGORIES = [
        "Climate_change", "Causality", "Machine_learning",
        "Computer_science", "Physics", "Biology",
        "Economics", "Psychology", "History",
        "Mathematics", "Engineering", "Medicine",
        "Ecology", "Genetics", "Neuroscience",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def get_category_members(self, category: str, limit: int = 50) -> list[str]:
        """Recupere les titres d'articles d'une categorie."""
        titles = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "format": "json",
        }
        url = "https://en.wikipedia.org/w/api.php"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return titles
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
        except Exception:
            return titles
        for m in members:
            if m.get("ns") == 0:  # Articles only
                titles.append(m["title"])
        return titles

    def get_article_text(self, title: str, max_chars: int = 3000) -> str | None:
        """Recupere le texte brut d'un article."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        url = "https://en.wikipedia.org/w/api.php"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                extract = page.get("extract", "")
                if extract:
                    return extract[:max_chars]
        except Exception:
            pass
        return None

    def scrape(self, categories: list[str] | None = None, max_per_category: int = 30) -> list[dict]:
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
                        "source": "wikipedia_en",
                        "category": cat,
                        "title": title,
                        "text": text,
                    })
                    count += 1
                time.sleep(0.1)
            print(f"{count} articles")
        return results
