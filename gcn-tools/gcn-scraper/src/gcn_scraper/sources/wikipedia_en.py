# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Scraper Wikipedia EN — wrapper rétrocompat autour de WikipediaLangScraper."""
from .wikipedia import WikipediaLangScraper
from ._net import DEFAULT_USER_AGENT


class WikipediaENScraper(WikipediaLangScraper):
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        super().__init__("en", user_agent)

    def scrape(self, categories=None, max_per_category: int = 30) -> list[dict]:
        return super().scrape()
