"""Scraper Wikipedia FR — dynamique : requêtes depuis config, pagination, expansion catégories."""
from __future__ import annotations
import time
import requests
from ._net import retry_get as _retry_get
from ..config.loader import get_source_config, get_config


def _build_queries(lang: str) -> list[str]:
    """Requêtes de recherche depuis config/sources.yaml — indépendant de la taxonomie causale."""
    return get_config().get("search_queries", {}).get(lang, [])


class WikipediaFRScraper:
    """Scraper Wikipedia FR — dynamique par API Search + expansion catégories + pagination."""

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self._cfg = get_source_config("wikipedia_fr")
        self._api_url: str = self._cfg.get("api_url", "https://fr.wikipedia.org/w/api.php")
        self._article_base: str = self._cfg.get("article_base_url", "https://fr.wikipedia.org/wiki/")
        self._cat_prefix: str = self._cfg.get("category_prefix", "Catégorie")
        self._delay: float = float(self._cfg.get("rate_limit_delay", 0.3))
        self._max_chars: int = int(self._cfg.get("max_chars_per_article", 5000))
        self._page_limit: int = int(self._cfg.get("pagination_limit", 50))
        self._fallback_cats: list[str] = self._cfg.get("fallback_categories", [])
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def _article_url(self, title: str) -> str:
        return self._article_base + title.replace(" ", "_")

    def search_articles(self, query: str, max_results: int = 30) -> list[str]:
        """Recherche simple (une page)."""
        params = {
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": min(max_results, 50),
            "srnamespace": 0, "format": "json",
        }
        resp, self.session = _retry_get(self.session, self._api_url, params)
        if resp is None:
            return []
        try:
            return [h["title"] for h in resp.json().get("query", {}).get("search", [])]
        except Exception:
            return []

    def search_articles_paginated(self, query: str, max_total: int = 60) -> list[str]:
        """Recherche paginée — parcourt plusieurs pages de résultats."""
        titles: list[str] = []
        offset = 0
        limit = min(self._page_limit, max_total)
        while len(titles) < max_total:
            params = {
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": limit,
                "sroffset": offset, "srnamespace": 0, "format": "json",
            }
            resp, self.session = _retry_get(self.session, self._api_url, params)
            if resp is None:
                break
            try:
                data = resp.json()
                hits = data.get("query", {}).get("search", [])
                if not hits:
                    break
                titles.extend(h["title"] for h in hits)
                if "continue" not in data:
                    break
                offset += len(hits)
            except Exception:
                break
            time.sleep(self._delay)
        return titles[:max_total]

    def get_article_text(self, title: str, max_chars: int | None = None) -> str | None:
        """Récupère le texte extrait d'un article."""
        chars = max_chars or self._max_chars
        params = {
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": True,
            "exsectionformat": "plain", "format": "json",
        }
        resp, self.session = _retry_get(self.session, self._api_url, params)
        if resp is None:
            return None
        try:
            for page in resp.json().get("query", {}).get("pages", {}).values():
                text = page.get("extract", "")
                if text and len(text) > 100:
                    return text[:chars]
        except Exception:
            pass
        return None

    def get_article_categories(self, title: str) -> list[str]:
        """Récupère les catégories d'un article — pour l'expansion dynamique."""
        params = {
            "action": "query", "titles": title,
            "prop": "categories", "cllimit": 10, "format": "json",
        }
        resp, self.session = _retry_get(self.session, self._api_url, params)
        if resp is None:
            return []
        try:
            for page in resp.json().get("query", {}).get("pages", {}).values():
                return [
                    c["title"].split(":")[-1]
                    for c in page.get("categories", [])
                    if "stub" not in c["title"].lower()
                    and "maintenance" not in c["title"].lower()
                    and "portail" not in c["title"].lower()
                ]
        except Exception:
            pass
        return []

    def get_category_articles(self, category: str, max_articles: int = 30) -> list[str]:
        """Articles d'une catégorie ; descend dans les sous-catégories (niveau 1) si vide."""
        titles = self._category_members(category, max_articles, ns=0)
        if not titles:
            subcats = self._category_members(category, 20, ns=14)
            for subcat in subcats[:5]:
                name = subcat.replace(f"{self._cat_prefix}:", "").replace("Category:", "")
                titles.extend(self._category_members(name, 10, ns=0))
                if len(titles) >= max_articles:
                    break
        return titles[:max_articles]

    def _category_members(self, category: str, limit: int, ns: int) -> list[str]:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"{self._cat_prefix}:{category}",
            "cmlimit": min(limit, 500), "cmnamespace": ns, "format": "json",
        }
        resp, self.session = _retry_get(self.session, self._api_url, params)
        if resp is None:
            return []
        try:
            return [
                m["title"]
                for m in resp.json().get("query", {}).get("categorymembers", [])
                if m.get("ns") == ns
            ]
        except Exception:
            return []

    def scrape(
        self,
        max_per_query: int | None = None,
        max_per_category: int | None = None,
        tracker=None,
    ) -> list[dict]:
        """
        Scraping dynamique : requêtes dérivées du CausalScorer + expansion catégories.

        tracker : BalanceTracker optionnel — s'arrête automatiquement quand quota atteint.
        """
        cfg_max = int(self._cfg.get("max_results_per_query", 20))
        mq = max_per_query or cfg_max
        mc = max_per_category or cfg_max

        query_list = _build_queries("fr")
        seen: set[str] = set()
        discovered_cats: set[str] = set()
        results: list[dict] = []

        # Phase 1 — recherche paginée depuis les requêtes config
        if tracker and tracker.is_full("fr"):
            print("  FR: budget langue atteint, skip")
            return results
        for query in query_list:
            if tracker and tracker.is_full("fr"):
                break
            print(f"  FR/search: '{query[:50]}'...", end=" ", flush=True)
            titles = self.search_articles_paginated(query, max_total=mq * 2)
            count = 0
            for title in titles:
                if title in seen:
                    continue
                if tracker and tracker.is_full("fr"):
                    break
                seen.add(title)
                text = self.get_article_text(title)
                if text:
                    for cat in self.get_article_categories(title)[:3]:
                        discovered_cats.add(cat)
                    results.append({
                        "title": title, "text": text, "lang": "fr",
                        "source": "wikipedia_fr",
                        "url": self._article_url(title),
                    })
                    count += 1
                time.sleep(self._delay)
            print(f"{count} articles")
            time.sleep(self._delay * 2)

        # Phase 2 — expansion dynamique depuis les catégories découvertes
        if discovered_cats and (tracker is None or not tracker.is_globally_full()):
            print(f"  FR/expansion: {len(discovered_cats)} catégories découvertes...")
            for cat in list(discovered_cats)[:20]:
                if tracker and tracker.is_globally_full():
                    break
                for title in self.get_category_articles(cat, max_articles=8):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "title": title, "text": text, "lang": "fr",
                            "source": f"wikipedia_fr:expanded:{cat}",
                            "url": self._article_url(title),
                        })
                    time.sleep(self._delay)
                time.sleep(self._delay * 2)

        # Phase 3 — fallback catégories de config
        if not tracker or not tracker.is_globally_full():
            print("  FR/fallback catégories...")
            for cat in self._fallback_cats:
                if tracker and tracker.is_globally_full():
                    break
                for title in self.get_category_articles(cat, mc):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "title": title, "text": text, "lang": "fr",
                            "source": f"wikipedia_fr:cat:{cat}",
                            "url": self._article_url(title),
                        })
                    time.sleep(self._delay)
                time.sleep(self._delay * 2)

        print(f"  FR total: {len(results)} articles ({len(seen)} uniques)")
        return results

    def scrape_all(self, max_per_category: int = 50) -> list[dict]:
        """Rétrocompat — délègue à scrape()."""
        return self.scrape(max_per_query=max_per_category, max_per_category=max_per_category)
