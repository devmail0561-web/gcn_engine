"""Scraper de flux RSS FR+EN — journaux riches en cause/concession/condition."""
from __future__ import annotations
import re
import time
import xml.etree.ElementTree as ET
import requests

RSS_SOURCES: dict[str, list[tuple[str, str]]] = {
    "fr": [
        ("lemonde_science",  "https://www.lemonde.fr/sciences/rss_full.xml"),
        ("lemonde_planete",  "https://www.lemonde.fr/planete/rss_full.xml"),
        ("lemonde_economie", "https://www.lemonde.fr/economie/rss_full.xml"),
        ("lemonde_politique","https://www.lemonde.fr/politique/rss_full.xml"),
        ("lefigaro_sante",   "https://www.lefigaro.fr/rss/figaro_sante.xml"),
        ("lefigaro_sciences","https://www.lefigaro.fr/rss/figaro_sciences.xml"),
        ("liberation_societe","https://www.liberation.fr/arc/outboundfeeds/rss/?outputType=xml"),
    ],
    "en": [
        ("guardian_science",     "https://www.theguardian.com/science/rss"),
        ("guardian_environment", "https://www.theguardian.com/environment/rss"),
        ("guardian_technology",  "https://www.theguardian.com/technology/rss"),
        ("guardian_world",       "https://www.theguardian.com/world/rss"),
        ("bbc_science",          "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("bbc_technology",       "http://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("reuters_science",      "https://feeds.reuters.com/reuters/scienceNews"),
        ("reuters_tech",         "https://feeds.reuters.com/reuters/technologyNews"),
    ],
}

_TAG_RE = re.compile(r'<[^>]+>')
_SPACE_RE = re.compile(r'\s+')


def _clean(html: str) -> str:
    return _SPACE_RE.sub(' ', _TAG_RE.sub('', html)).strip()


class NewsRSSScraper:
    """Scrappe des flux RSS de presse FR+EN."""

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
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
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  {name}: HTTP {resp.status_code}")
                return []
            items = self._parse_feed(resp.text, lang, name, url)
            print(f"  {name}: {len(items)} items")
            return items
        except Exception as e:
            print(f"  {name}: erreur {e}")
            return []

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
