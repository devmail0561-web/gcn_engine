# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Scraper de flux RSS FR+EN — journaux riches en cause/concession/condition."""
from __future__ import annotations
import re
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]


def _load_rss_sources() -> dict[str, list[tuple[str, str]]]:
    """Charge les URLs RSS depuis la config — aucune URL hardcodée."""
    import warnings as _w
    cfg = get_source_config("news_rss")
    feeds_cfg = cfg.get("feeds", {})
    out: dict[str, list[tuple[str, str]]] = {}
    for lang, feeds in feeds_cfg.items():
        pairs = []
        for f in feeds or []:
            if isinstance(f, dict) and f.get("name") and f.get("url"):
                pairs.append((f["name"], f["url"]))
            else:
                _w.warn(
                    f"news_rss: entrée flux invalide ignorée (lang={lang}) : {f!r}",
                    UserWarning,
                )
        out[lang] = pairs
    return out


def _get_rss_sources() -> dict[str, list[tuple[str, str]]]:
    """Charge les sources RSS depuis la config — pas de cache global (reload_config-safe)."""
    try:
        return _load_rss_sources()
    except Exception as e:
        import warnings as _w
        _w.warn(
            f"news_rss: impossible de charger sources.yaml ({e}) — source désactivée.",
            UserWarning,
        )
        return {}

_TAG_RE = re.compile(r'<[^>]+>')
_SPACE_RE = re.compile(r'\s+')
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"


def _el_text(el) -> str:
    """Texte complet d'un élément (enfants XML inclus via itertext)."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _atom_link(entry, ns, fallback: str) -> str:
    """Préfère le lien rel=alternate (article), pas rel=self/edit."""
    links = entry.findall("atom:link", ns)
    for link_el in links:
        rel = (link_el.get("rel") or "alternate").lower()
        href = (link_el.get("href") or "").strip()
        if href and rel == "alternate":
            return href
    for link_el in links:
        href = (link_el.get("href") or "").strip()
        if href:
            return href
    return fallback


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
            _ATOM_ROOT = "{http://www.w3.org/2005/Atom}feed"
            is_atom = root.tag == _ATOM_ROOT
            # RSS 2.0 — ignoré pour un flux Atom pur (évite double-parse).
            for idx, item in enumerate(root.iter("item") if not is_atom else []):
                desc = item.find("description")
                title = item.find("title")
                link = item.find("link")
                encoded = item.find(_CONTENT_NS)
                text = None
                if encoded is not None and _el_text(encoded):
                    text = _clean(_el_text(encoded))
                elif desc is not None and _el_text(desc):
                    text = _clean(_el_text(desc))
                elif _el_text(title):
                    text = _el_text(title)
                link_url = link.text.strip() if link is not None and link.text and link.text.strip() else ""
                if text and len(text) > 50:
                    texts.append({
                        "text": text,
                        "source": f"news:{source_name}",
                        "lang": lang,
                        "url": link_url or f"{url}#item-{idx}",
                    })
            # Atom — <content> avant <summary>.
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for idx, entry in enumerate(root.findall(".//atom:entry", ns)):
                summary = entry.find("atom:summary", ns)
                title = entry.find("atom:title", ns)
                content = entry.find("atom:content", ns)
                item_url = _atom_link(entry, ns, f"{url}#entry-{idx}")
                text = None
                if content is not None and _el_text(content):
                    text = _clean(_el_text(content))
                elif summary is not None and _el_text(summary):
                    text = _clean(_el_text(summary))
                elif _el_text(title):
                    text = _el_text(title)
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

    def scrape(self, langs: list[str] | None = None, seed: int | None = None, tracker=None) -> list[dict]:
        """Scrape tous les flux pour les langues demandées (ordre mélangé si seed)."""
        from ..diversity import shuffled as _shuffled
        if langs is None:
            langs = ["fr", "en"]
        results = []
        for lang in langs:
            if tracker is not None and tracker.is_globally_full():
                break
            consecutive_failures = 0  # par langue : un FR en panne n'annule pas EN
            feeds = _get_rss_sources().get(lang, [])
            if seed is not None:
                feeds = _shuffled(feeds, seed, f"news-{lang}")
            for name, url in feeds:
                if tracker is not None and tracker.is_globally_full():
                    break
                items = self.scrape_feed(name, url, lang)
                if not items:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print(f"  news/{lang}: 5 flux en échec de suite, langue suivante (anti-blocage).")
                        break
                    continue
                consecutive_failures = 0
                results.extend(items)
                time.sleep(self._delay)
        return results
