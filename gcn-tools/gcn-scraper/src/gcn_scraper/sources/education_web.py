"""Scraper de sites educatifs francais."""

import time
import trafilatura


class EducationScraper:
    """Extrait du texte depuis des pages web educatives."""

    URLS = [
        "https://fr.wikipedia.org/wiki/Connecteur_logique",
        "https://fr.wikipedia.org/wiki/Proposition_causale",
        "https://fr.wikipedia.org/wiki/Proposition_conditionnelle",
        "https://fr.wikipedia.org/wiki/Proposition_concessive",
        "https://fr.wikipedia.org/wiki/Relation_causale",
        "https://fr.wikipedia.org/wiki/Logique",
        "https://fr.wikipedia.org/wiki/Physique",
        "https://fr.wikipedia.org/wiki/Chimie",
        "https://fr.wikipedia.org/wiki/Biologie",
        "https://fr.wikipedia.org/wiki/Mathematiques",
        "https://fr.wikipedia.org/wiki/Informatique",
        "https://fr.wikipedia.org/wiki/Astronomie",
        "https://fr.wikipedia.org/wiki/Ecologie",
        "https://fr.wikipedia.org/wiki/Medecine",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.user_agent = user_agent

    def scrape_url(self, url: str) -> str | None:
        """Telecharge et nettoie une page web."""
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
        return None

    def scrape_all(self) -> list[dict]:
        """Scrape toutes les URLs."""
        results = []
        for url in self.URLS:
            text = self.scrape_url(url)
            if text and len(text) > 200:
                title = url.split("/")[-1].replace("_", " ")
                results.append({
                    "title": title,
                    "text": text,
                    "url": url,
                    "source": "education",
                })
            time.sleep(0.5)
        return results
