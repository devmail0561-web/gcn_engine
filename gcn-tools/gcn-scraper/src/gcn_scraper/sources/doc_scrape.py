"""Scraper de documentation technique — Python/Rust/JS/TS/Java/Go/C#."""
from __future__ import annotations
import time
import trafilatura
from ._net import retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config


class DocScraper:
    """Scrape des pages de documentation via trafilatura.

    URLs lues depuis config/sources.yaml — aucun hardcoding.
    Supporte tous les langages configurés sous doc.urls.
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        import requests
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        cfg = get_source_config("doc")
        self._delay = float(cfg.get("rate_limit_delay", 1.0))

    def scrape_url(self, url: str, source_label: str) -> dict | None:
        """Scrape une URL et extrait le texte avec trafilatura."""
        label = url.rstrip("/").split("/")[-1][:40] or url[:40]
        print(f"  {source_label}: {label}...", end=" ", flush=True)
        resp, self.session = retry_get(
            self.session, url, {},
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            print("échec réseau")
            return None
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            print(f"OK ({len(text)} chars)")
            return {
                "text": text[:5000],
                "lang": "code",
                "source": source_label,
                "url": url,
            }
        print("vide")
        return None

    def scrape(self, languages: list[str] | None = None, seed: int | None = None, tracker=None) -> list[dict]:
        """Scrape la documentation des langages sélectionnés (None = tous).

        Liste d'URLs fixe (44 pages) : l'ordre est mélangé si seed, mais le
        périmètre reste identique — élargir le corpus = ajouter des URLs en config.
        """
        from ..diversity import shuffled as _shuffled
        cfg = get_source_config("doc")
        urls_by_lang: dict[str, list[str]] = cfg.get("urls", {})
        if languages is not None:
            urls_by_lang = {k: v for k, v in urls_by_lang.items() if k in languages}
        results: list[dict] = []
        items: list[tuple[str, str]] = [
            (lang, url) for lang, urls in urls_by_lang.items() for url in urls
        ]
        if seed is not None:
            items = _shuffled(items, seed, "doc")
        for lang, url in items:
            if tracker is not None and tracker.is_globally_full():
                break
            result = self.scrape_url(url, f"{lang}_doc")
            if result:
                results.append(result)
            time.sleep(self._delay)
        return results
