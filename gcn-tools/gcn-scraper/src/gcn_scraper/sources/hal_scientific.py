"""Scraper HAL (archives ouvertes) — queries ciblées FR+EN, pagination."""
from __future__ import annotations
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config


class HALScraper:
    """Extrait des résumés de papiers scientifiques FR+EN depuis HAL."""

    @property
    def BASE_URL(self) -> str:
        return get_source_config("hal").get("api_url", "https://api.archives-ouvertes.fr/search/")

    QUERIES_FR: list[str] = [
        "causalité", "relation causale", "cause effet",
        "condition nécessaire", "si alors", "bien que résultats",
        "malgré contrainte", "séquence étapes protocole",
        "afin d'améliorer", "en revanche comparaison",
        "dépend des données", "algorithme contrôle",
    ]

    QUERIES_EN: list[str] = [
        "causal inference", "cause effect", "causal mechanism",
        "conditional probability", "if then implication",
        "although despite results", "sequence pipeline steps",
        "in order to improve", "in contrast to baseline",
        "depends on data", "algorithm controls training",
        "enables learning", "prevents overfitting",
    ]

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._delay = float(get_source_config("hal").get("rate_limit_delay", 1.0))

    def search_papers(self, query: str, max_results: int = 100, start: int = 0) -> list[dict]:
        """Recherche des papiers sur HAL avec retry réseau."""
        params = {
            "q": query,
            "wt": "json",
            "rows": min(max_results, 100),
            "start": start,
            "fl": "docid,label_s,abstract_s,language_s,uri_s",
        }
        resp, self.session = _retry_get(self.session, self.BASE_URL, params,  # noqa: B009
                                         user_agent=self.user_agent,
                                         min_interval=self._delay, base_delay=1.0)
        if resp is None:
            print(f"  HAL '{query}': toutes tentatives échouées, skip")
            return []
        try:
            return resp.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  HAL parse error: {e}")
            return []

    def _fetch_paper_text(self, url: str) -> str | None:
        """Scrape la page du papier (hal.science) — abstract en fallback appelant."""
        resp, self.session = _retry_get(
            self.session, url, {},
            user_agent=self.user_agent,
            min_interval=self._delay, base_delay=1.0, max_retries=2,
        )
        if resp is None:
            return None
        try:
            text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
            return text if text and len(text) > 200 else None
        except Exception:
            return None

    def _doc_to_item(self, doc: dict, query: str) -> dict | None:
        """Scrape la page du papier ; l'abstract API sert de fallback."""
        url = doc.get("uri_s", f"https://hal.science/{doc.get('docid', '')}")
        langs = doc.get("language_s", [])
        lang = langs[0] if langs else "fr"
        if lang.lower().startswith("en"):
            lang = "en"
        elif lang.lower().startswith("fr"):
            lang = "fr"
        title = doc.get("label_s", [""])[0] if doc.get("label_s") else ""
        # Principe : scraper le SITE lié, pas la sortie du moteur.
        text = self._fetch_paper_text(url) if url else None
        time.sleep(self._delay)
        if not text:
            for abstract in doc.get("abstract_s", []):
                if abstract and len(abstract) > 50:
                    text = abstract
                    break
        if text and len(text) > 50:
            return {
                "title": title,
                "text": text,
                "url": url,
                "source": f"hal:{query}",
                "lang": lang,
            }
        return None

    def scrape(self, max_per_query: int = 100, seed: int | None = None) -> list[dict]:
        """Scrape toutes les queries FR+EN (ordre mélangé + offset rotatif si seed)."""
        from ..diversity import shuffled as _shuffled, page_offset as _offset
        queries = _shuffled(self.QUERIES_FR + self.QUERIES_EN, seed, "hal") if seed is not None else list(self.QUERIES_FR + self.QUERIES_EN)
        start = _offset(seed, "hal", min(max_per_query, 100)) if seed is not None else 0
        if start:
            print(f"  HAL: offset pagination {start} (seed={seed})")
        all_papers = []
        consecutive_failures = 0
        for query in queries:
            docs = self.search_papers(query, max_per_query, start=start)
            if not docs:
                consecutive_failures += 1
                print(f"  HAL '{query}': 0 papier (échec/rate-limit)")
                if consecutive_failures >= 5:
                    print("  HAL: 5 échecs consécutifs, arrêt (cooldown actif).")
                    break
                continue
            consecutive_failures = 0
            for doc in docs:
                item = self._doc_to_item(doc, query)
                if item:
                    all_papers.append(item)
            print(f"  HAL '{query}': {len(docs)} papiers")
            time.sleep(self._delay)
        return all_papers
