"""Scraper de flux RSS FR+EN — journaux riches en cause/concession/condition."""
from __future__ import annotations
import re
import time
import xml.etree.ElementTree as ET
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config


def _load_rss_sources() -> dict[str, list[tuple[str, str]]]:
    """Charge les URLs RSS depuis la config — aucune URL hardcodée."""
    cfg = get_source_config("news_rss")
    feeds_cfg = cfg.get("feeds", {})
    return {
        lang: [(f["name"], f["url"]) for f in feeds]
        for lang, feeds in feeds_cfg.items()
    }


_RSS_SOURCES: dict[str, list[tuple[str, str]]] | None = None


def _get_rss_sources() -> dict[str, list[tuple[str, str]]]:
    global _RSS_SOURCES
    if _RSS_SOURCES is None:
        try:
            _RSS_SOURCES = _load_rss_sources()
        except Exception as e:
            import warnings as _w
            _w.warn(
                f"news_rss: impossible de charger sources.yaml ({e}) — source désactivée.",
                UserWarning,
            )
            _RSS_SOURCES = {}
    return _RSS_SOURCES

_TAG_RE = re.compile(r'<[^>]+>')
_SPACE_RE = re.compile(r'\s+')


def _clean(html: str) -> str:
    return _SPACE_RE.sub(' ', _TAG_RE.sub('', html)).strip()


class NewsRSSScraper:
    """Scrappe des flux RSS de presse FR+EN."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._delay = float(get_source_config("news_rss").get("rate_limit_delay", 1.0))

    def _parse_feed(self, xml_text: str, lang: str, source_name: str, url: str) -> list[dict]:
        texts = []
        try:
            root = ET.fromstring(xml_text)
            # RSS 2.0
            for item in root.iter("item"):
                desc = item.find("description")
                title = item.find("title")
                link = item.find("link")
                text = None
                if desc is not None and desc.text:
                    text = _clean(desc.text)
                elif title is not None and title.text:
                    text = title.text.strip()
                if text and len(text) > 50:
                    texts.append({
                        "text": text,
                        "source": f"news:{source_name}",
                        "lang": lang,
                        "url": (link.text.strip() if link is not None and link.text else url),
                    })
            # Atom
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                summary = entry.find("atom:summary", ns)
                title = entry.find("atom:title", ns)
                link_el = entry.find("atom:link", ns)
                item_url = link_el.get("href", url) if link_el is not None else url
                text = None
                if summary is not None and summary.text:
                    text = _clean(summary.text)
                elif title is not None and title.text:
                    text = title.text.strip()
                if text and len(text) > 50:
                    texts.append({
                        "text": text,
                        "source": f"news:{source_name}",
                        "lang": lang,
                        "url": item_url,
                    })
        except ET.ParseError:
            pass
        return texts

    def _fetch_article_text(self, url: str) -> str | None:
        """Scrape l'article lié — description RSS en fallback appelant."""
        resp, self.session = _retry_get(
            self.session, url, {}, user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            return None
        try:
            text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
            return text if text and len(text) > 200 else None
        except Exception:
            return None

    def scrape_feed(self, name: str, url: str, lang: str) -> list[dict]:
        resp, self.session = _retry_get(
            self.session, url, {}, user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            print(f"  {name}: toutes tentatives échouées, skip")
            return []
        raw_items = self._parse_feed(resp.text, lang, name, url)
        # Principe : scraper le SITE lié (l'article), pas la sortie du flux.
        items = []
        for it in raw_items:
            link = it.get("url", "")
            text = self._fetch_article_text(link) if link and link != url else None
            time.sleep(self._delay)
            if not text:
                text = it.get("text", "")
            if text and len(text) > 50:
                it["text"] = text
                items.append(it)
        print(f"  {name}: {len(items)} items (articles scrapés)")
        return items

    def scrape(self, langs: list[str] | None = None, seed: int | None = None) -> list[dict]:
        """Scrape tous les flux pour les langues demandées (ordre mélangé si seed)."""
        from ..diversity import shuffled as _shuffled
        if langs is None:
            langs = ["fr", "en"]
        results = []
        consecutive_failures = 0
        for lang in langs:
            feeds = _get_rss_sources().get(lang, [])
            if seed is not None:
                feeds = _shuffled(feeds, seed, f"news-{lang}")
            for name, url in feeds:
                items = self.scrape_feed(name, url, lang)
                if not items:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print("  news: 5 flux en échec de suite, arrêt (anti-blocage).")
                        return results
                    continue
                consecutive_failures = 0
                results.extend(items)
                time.sleep(self._delay)
        return results
