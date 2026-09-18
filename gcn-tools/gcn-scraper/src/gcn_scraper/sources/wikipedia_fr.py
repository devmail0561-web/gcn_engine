"""Scraper Wikipedia FR — wrapper rétrocompat autour de WikipediaLangScraper."""
from .wikipedia import WikipediaLangScraper


class WikipediaFRScraper(WikipediaLangScraper):
    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        super().__init__("fr", user_agent)

    def scrape_all(self, max_per_category: int = 50) -> list[dict]:
        return self.scrape()
