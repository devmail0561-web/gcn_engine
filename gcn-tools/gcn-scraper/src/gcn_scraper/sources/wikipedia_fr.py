# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Scraper Wikipedia FR — wrapper rétrocompat autour de WikipediaLangScraper."""
from .wikipedia import WikipediaLangScraper
from ._net import DEFAULT_USER_AGENT


class WikipediaFRScraper(WikipediaLangScraper):
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        super().__init__("fr", user_agent)

    def scrape_all(self, max_per_category: int = 50) -> list[dict]:
        return self.scrape()
