"""Scraper arXiv via l'API Atom/REST — gratuit, sans token."""
from __future__ import annotations
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]


class ArXivScraper:
    @property
    def BASE_URL(self) -> str:
        # audit-2 Fix 9 : HTTPS (le http:// fuit les requêtes en clair + redirect).
        return get_source_config("arxiv").get("api_url", "https://export.arxiv.org/api/query")

    QUERIES: dict[str, list[str]] = {
        "cause": [
            "causal inference", "causality machine learning",
            "cause and effect", "causal mechanism",
        ],
        "enable": [
            "enables learning", "machine learning enables",
            "AI enables applications",
        ],
        "prevent": [
            "prevents overfitting regularization",
            "safety constraints neural network",
            "robust training prevents",
        ],
        "condition": [
            "conditional probability estimation",
            "conditioned on context",
            "if-then formal reasoning",
        ],
        "concession": [
            "although results show limitation",
            "despite limitations accuracy",
            "even though performance",
        ],
        "sequence": [
            "step by step learning",
            "sequential process optimization",
            "pipeline architecture training",
        ],
        "motivation": [
            "motivated by reducing error",
            "in order to improve generalization",
            "with the goal of alignment",
        ],
        "opposition": [
            "in contrast to baseline",
            "unlike previous approaches",
            "however we show different",
        ],
        "data_dependency": [
            "depends on dataset size",
            "data-driven approach model",
            "relies on training data distribution",
        ],
        "control_dependency": [
            "algorithm controls training loop",
            "orchestrates training pipeline",
            "manages distributed workflow",
        ],
    }

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._delay = float(get_source_config("arxiv").get("rate_limit_delay", 3.0))

    def _search_page(self, query: str, start: int, per_page: int) -> list[dict]:
        """Une page de résultats Atom (sans fetch des pages abs)."""
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": min(per_page, 200),
        }
        resp, self.session = _retry_get(self.session, self.BASE_URL, params,
                                         user_agent=self.user_agent,
                                         min_interval=self._delay, base_delay=1.0)
        if resp is None:
            print(f"  arXiv '{query}': toutes tentatives échouées, skip")
            return []
        entries = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                link_el = entry.find("atom:id", ns)
                url = link_el.text.strip() if link_el is not None and link_el.text else ""
                summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""
                title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
                entries.append({"title": title, "summary": summary, "url": url})
        except ET.ParseError as e:
            print(f"  arXiv '{query}': flux Atom invalide ({e}), skip")
        return entries

    def search(self, query: str, max_results: int = 100, start: int = 0) -> list[dict]:
        """Recherche des papers arXiv avec retry réseau + vraie pagination."""
        results = []
        offset = max(0, start)
        remaining = max(0, max_results)
        while remaining > 0:
            per_page = min(remaining, 200)
            page = self._search_page(query, offset, per_page)
            if not page:
                break
            for entry in page:
                # Principe : scraper le SITE lié (page abs), pas la sortie du moteur.
                text = self._fetch_abs_text(entry["url"])
                time.sleep(self._delay)
                if not text:
                    text = entry["summary"]
                if text and len(text) > 50:
                    results.append({
                        "title": entry["title"],
                        "text": text,
                        "url": entry["url"],
                        "source": f"arxiv:{query}",
                        "lang": "en",
                    })
            if len(page) < per_page:
                break  # dernière page
            offset += len(page)
            remaining -= len(page)
        return results[:max_results]

    def _fetch_abs_text(self, url: str) -> str | None:
        """Scrape la page abs arXiv — summary API en fallback appelant."""
        if not url:
            return None
        resp2, self.session = _retry_get(
            self.session, url, {},
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp2 is None:
            return None
        try:
            text = trafilatura.extract(resp2.text, include_comments=False, include_tables=False)
            return text if text and len(text) > 200 else None
        except Exception:
            return None

    def scrape(self, max_per_query: int = 100, seed: int | None = None, tracker=None) -> list[dict]:
        """Scrape tous les topics, respecte le rate limit arXiv (3s entre requêtes)."""
        from ..diversity import shuffled as _shuffled, page_offset as _offset
        topics = list(self.QUERIES.items())
        if seed is not None:
            topics = _shuffled(topics, seed, "arxiv-topics")
        start = _offset(seed, "arxiv", min(max_per_query, 200)) if seed is not None else 0
        if start:
            print(f"  arXiv: offset pagination {start} (seed={seed})")
        results = []
        consecutive_failures = 0
        for relation_type, queries in topics:
            if tracker is not None and tracker.is_globally_full():
                break
            queries = _shuffled(queries, seed, f"arxiv-{relation_type}") if seed is not None else queries
            for q in queries:
                if tracker is not None and tracker.is_globally_full():
                    break
                print(f"  arXiv [{relation_type}]: '{q}'...", end=" ", flush=True)
                papers = self.search(q, max_results=max_per_query, start=start)
                if not papers:
                    consecutive_failures += 1
                    print("0 abstract (échec/rate-limit)")
                    if consecutive_failures >= 5:
                        print("  arXiv: 5 échecs consécutifs, arrêt (cooldown actif).")
                        return results
                    continue
                consecutive_failures = 0
                results.extend(papers)
                print(f"{len(papers)} abstracts")
                time.sleep(self._delay)
        return results
