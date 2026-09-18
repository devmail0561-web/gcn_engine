"""Scraper Wikipedia générique — supporte toutes les langues via l'API MediaWiki."""
from __future__ import annotations
import time
import requests
from ._net import retry_get
from ..config.loader import get_source_config, get_config


class WikipediaLangScraper:
    """
    Scraper Wikipedia pour n'importe quelle langue.

    Configuration lue depuis sources.yaml (clé wikipedia_<lang>).
    Si la clé n'existe pas, utilise les valeurs par défaut MediaWiki.
    Aucune URL hardcodée.
    """

    def __init__(self, lang: str, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.lang = lang
        cfg = get_source_config(f"wikipedia_{lang}")
        self._api_url: str = cfg.get("api_url", f"https://{lang}.wikipedia.org/w/api.php")
        self._article_base: str = cfg.get("article_base_url", f"https://{lang}.wikipedia.org/wiki/")
        self._cat_prefix: str = cfg.get("category_prefix", "Category")
        self._delay: float = float(cfg.get("rate_limit_delay", 0.3))
        self._max_chars: int = int(cfg.get("max_chars_per_article", 5000))
        self._page_limit: int = int(cfg.get("pagination_limit", 50))
        self._fallback_cats: list[str] = cfg.get("fallback_categories", [])
        self._max_per_query: int = int(cfg.get("max_results_per_query", 20))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def _article_url(self, title: str) -> str:
        return self._article_base + title.replace(" ", "_")

    def search_articles_paginated(self, query: str, max_total: int = 60) -> list[str]:
        """Recherche paginée sur l'API MediaWiki search."""
        titles: list[str] = []
        offset = 0
        limit = min(self._page_limit, max_total)
        while len(titles) < max_total:
            params = {
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": limit,
                "sroffset": offset, "srnamespace": 0, "format": "json",
            }
            resp, self.session = retry_get(self.session, self._api_url, params)
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

    def get_article_text(self, title: str) -> str | None:
        """Récupère le texte extrait d'un article Wikipedia."""
        params = {
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": True,
            "exsectionformat": "plain", "format": "json",
        }
        resp, self.session = retry_get(self.session, self._api_url, params)
        if resp is None:
            return None
        try:
            for page in resp.json().get("query", {}).get("pages", {}).values():
                text = page.get("extract", "")
                if text and len(text) > 100:
                    return text[:self._max_chars]
        except Exception:
            pass
        return None

    def get_article_categories(self, title: str) -> list[str]:
        """Retourne les catégories d'un article (pour expansion dynamique)."""
        params = {
            "action": "query", "titles": title,
            "prop": "categories", "cllimit": 10, "format": "json",
        }
        resp, self.session = retry_get(self.session, self._api_url, params)
        if resp is None:
            return []
        try:
            for page in resp.json().get("query", {}).get("pages", {}).values():
                return [
                    c["title"].split(":")[-1]
                    for c in page.get("categories", [])
                    if not any(skip in c["title"].lower()
                               for skip in ("stub", "maintenance", "portail", "portal", "vorlage"))
                ]
        except Exception:
            pass
        return []

    def get_category_articles(self, category: str, max_articles: int = 30) -> list[str]:
        """Articles d'une catégorie ; descend dans les sous-catégories si vide."""
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
        """Membres d'une catégorie, avec pagination cmcontinue (audit-2 Fix 3).

        Sans boucle, seul la première page (cmlimit) était retournée et les
        catégories de plus de `limit` membres étaient tronquées.
        """
        titles: list[str] = []
        cmcontinue: str | None = None
        while len(titles) < limit:
            params = {
                "action": "query", "list": "categorymembers",
                "cmtitle": f"{self._cat_prefix}:{category}",
                "cmlimit": min(limit - len(titles), 500),
                "cmnamespace": ns, "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp, self.session = retry_get(self.session, self._api_url, params)
            if resp is None:
                break
            try:
                data = resp.json()
                titles.extend(
                    m["title"]
                    for m in data.get("query", {}).get("categorymembers", [])
                    if m.get("ns") == ns
                )
                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue:
                    break
            except Exception:
                break
            time.sleep(self._delay)
        return titles[:limit]

    def scrape(self, tracker=None) -> list[dict]:
        """
        Scraping dynamique : recherche paginée + expansion catégories + fallback config.

        tracker : BalanceTracker optionnel — arrêt quand budget langue atteint.
        """
        lang = self.lang
        if tracker and tracker.is_full(lang):
            print(f"  {lang.upper()}: budget langue atteint, skip")
            return []

        queries = get_config().get("search_queries", {}).get(lang, [])
        seen: set[str] = set()
        discovered_cats: set[str] = set()
        results: list[dict] = []

        # Phase 1 — recherche paginée
        for query in queries:
            if tracker and tracker.is_full(lang):
                break
            print(f"  {lang.upper()}/search: '{query[:50]}'...", end=" ", flush=True)
            titles = self.search_articles_paginated(query, max_total=self._max_per_query * 2)
            count = 0
            for title in titles:
                if title in seen or (tracker and tracker.is_full(lang)):
                    continue
                seen.add(title)
                text = self.get_article_text(title)
                if text:
                    for cat in self.get_article_categories(title)[:3]:
                        discovered_cats.add(cat)
                    results.append({
                        "text": text, "lang": lang,
                        "source": f"wikipedia_{lang}",
                        "url": self._article_url(title),
                    })
                    count += 1
                time.sleep(self._delay)
            print(f"{count} articles")
            time.sleep(self._delay * 2)

        # Phase 2 — expansion dynamique
        if discovered_cats and (tracker is None or not tracker.is_globally_full()):
            print(f"  {lang.upper()}/expansion: {len(discovered_cats)} catégories...")
            for cat in list(discovered_cats)[:15]:
                if tracker and tracker.is_globally_full():
                    break
                for title in self.get_category_articles(cat, max_articles=8):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "text": text, "lang": lang,
                            "source": f"wikipedia_{lang}:expanded:{cat}",
                            "url": self._article_url(title),
                        })
                    time.sleep(self._delay)
                time.sleep(self._delay * 2)

        # Phase 3 — fallback catégories de config
        if self._fallback_cats and (tracker is None or not tracker.is_globally_full()):
            print(f"  {lang.upper()}/fallback catégories...")
            for cat in self._fallback_cats:
                if tracker and tracker.is_globally_full():
                    break
                for title in self.get_category_articles(cat, self._max_per_query):
                    if title in seen:
                        continue
                    seen.add(title)
                    text = self.get_article_text(title)
                    if text:
                        results.append({
                            "text": text, "lang": lang,
                            "source": f"wikipedia_{lang}:cat:{cat}",
                            "url": self._article_url(title),
                        })
                    time.sleep(self._delay)
                time.sleep(self._delay * 2)

        print(f"  {lang.upper()} total: {len(results)} articles ({len(seen)} uniques)")
        return results
