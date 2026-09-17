"""Scraper Wikipedia FR via l'API officielle."""

import time
import wikipediaapi


class WikipediaFRScraper:
    """Extrait des articles Wikipedia en francais."""

    CATEGORIES = [
        "Climat",
        "Economie",
        "Sante",
        "Histoire_de_France",
        "Technologie",
        "Environnement",
        "Epidemiologie",
        "Geologie",
        "Sociologie",
        "Psychologie",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language="fr",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    def scrape_category(self, category: str, max_articles: int = 50) -> list[dict]:
        """Extrait les articles d'une categorie."""
        cat = self.wiki.page(f"Category:{category}")
        if not cat.exists():
            return []

        articles = []
        members = cat.categorymembers

        for title, page in members.items():
            if len(articles) >= max_articles:
                break
            if page.namespace != 0:
                continue
            if not page.exists():
                continue

            text = page.text
            if text and len(text) > 200:
                articles.append({
                    "title": page.title,
                    "text": text,
                    "url": page.fullurl,
                    "source": f"wikipedia:{category}",
                })
                time.sleep(0.5)

        return articles

    def scrape_all(self, max_per_category: int = 30) -> list[dict]:
        """Scrape toutes les categories cibles."""
        all_articles = []
        for cat in self.CATEGORIES:
            articles = self.scrape_category(cat, max_per_category)
            all_articles.extend(articles)
            print(f"  {cat}: {len(articles)} articles")
        return all_articles
