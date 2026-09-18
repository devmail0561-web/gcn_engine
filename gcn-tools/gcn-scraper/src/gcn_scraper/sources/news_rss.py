"""Scraper de flux RSS FR+EN — journaux riches en cause/concession/condition."""
from __future__ import annotations
import re
import time
import xml.etree.ElementTree as ET
import requests
from ._net import retry_get as _retry_get
from ..config.loader import get_source_config


def _load_rss_sources() -> dict[str, list[tuple[str, str]]]:
    """Charge les URLs RSS depuis la config — aucune URL hardcodée."""
    cfg = get_source_config("news_rss")
    feeds_cfg = cfg.get("feeds", {})
    return {
        lang: [(f["name"], f["url"]) for f in feeds]
        for lang, feeds in feeds_cfg.items()
    }


RSS_SOURCES: dict[str, list[tuple[str, str]]] = _load_rss_sources()

_TAG_RE = re.compile(r'<[^>]+>')
_SPACE_RE = re.compile(r'\s+')


def _clean(html: str) -> str:
    return _SPACE_RE.sub(' ', _TAG_RE.sub('', html)).strip()


class NewsRSSScraper:
    """Scrappe des flux RSS de presse FR+EN."""

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

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

    def scrape_feed(self, name: str, url: str, lang: str) -> list[dict]:
        resp = _retry_get(self.session, url, {}, user_agent=self.user_agent)
        if resp is None:
            print(f"  {name}: toutes tentatives échouées, skip")
            return []
        items = self._parse_feed(resp.text, lang, name, url)
        print(f"  {name}: {len(items)} items")
        return items

    def scrape(self, langs: list[str] | None = None) -> list[dict]:
        """Scrape tous les flux pour les langues demandées."""
        if langs is None:
            langs = ["fr", "en"]
        results = []
        for lang in langs:
            for name, url in RSS_SOURCES.get(lang, []):
                results.extend(self.scrape_feed(name, url, lang))
                time.sleep(1)
        return results
