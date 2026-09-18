"""Scraper Wikipedia EN — wrapper rétrocompat autour de WikipediaLangScraper."""
from .wikipedia import WikipediaLangScraper


class WikipediaENScraper(WikipediaLangScraper):
    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        super().__init__("en", user_agent)

    def scrape(self, categories=None, max_per_category: int = 30) -> list[dict]:
        return super().scrape()
