"""Scraper Wikipedia FR — catégories ciblées sur les patterns causaux du moteur GCN."""
from __future__ import annotations
import time
import wikipediaapi


class WikipediaFRScraper:
    """Extrait des articles Wikipedia FR catégorisés selon les 11 types de relations."""

    # Catégories ciblées par famille de relations
    CATEGORIES: list[str] = [
        # cause / enable / prevent
        "Épidémiologie", "Climatologie", "Toxicologie", "Pharmacologie",
        "Physiopathologie", "Médecine", "Géologie", "Volcanologie",
        # condition / filter
        "Droit_civil_français", "Droit_du_travail_en_France",
        "Droit_administratif_français", "Procédure_pénale_en_France",
        # concession / opposition
        "Débat_politique_en_France", "Philosophie_française",
        "Sciences_sociales", "Sociologie",
        # sequence
        "Histoire_de_France", "Chimie",
        "Informatique", "Génie_chimique",
        # motivation
        "Économie", "Politique_économique_de_la_France", "Psychologie",
        # data_dependency / control_dependency
        "Algorithme", "Architecture_des_ordinateurs", "Génie_logiciel",
        "Apprentissage_automatique", "Réseau_de_neurones_artificiels",
    ]

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language="fr",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    def scrape_category(
        self,
        category: str,
        max_articles: int = 50,
        max_chars: int = 5000,
    ) -> list[dict]:
        """Extrait les articles d'une catégorie Wikipedia FR."""
        cat = self.wiki.page(f"Catégorie:{category}")
        if not cat.exists():
            # Essai avec préfixe Category: (fallback)
            cat = self.wiki.page(f"Category:{category}")
        if not cat.exists():
            return []

        articles = []
        for title, page in cat.categorymembers.items():
            if len(articles) >= max_articles:
                break
            if page.namespace != 0:
                continue
            if not page.exists():
                continue
            text = page.text[:max_chars] if page.text else ""
            if text and len(text) > 200:
                articles.append({
                    "title": page.title,
                    "text": text,
                    "url": page.fullurl,
                    "source": f"wikipedia_fr:{category}",
                    "lang": "fr",
                })
                time.sleep(0.3)

        return articles

    def scrape_all(self, max_per_category: int = 50) -> list[dict]:
        all_articles = []
        for cat in self.CATEGORIES:
            articles = self.scrape_category(cat, max_per_category)
            all_articles.extend(articles)
            print(f"  FR/{cat}: {len(articles)} articles")
        return all_articles
