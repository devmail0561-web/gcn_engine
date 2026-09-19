"""Scraper HAL (archives ouvertes) — queries ciblées FR+EN, pagination."""
from __future__ import annotations
import time
import requests
import trafilatura
from ._net import retry_get as _retry_get, DEFAULT_USER_AGENT
from ..config.loader import get_source_config


_QUERIES_FR_DEFAULT: list[str] = [
    "causalité", "relation causale", "cause effet",
    "condition nécessaire", "si alors", "bien que résultats",
    "malgré contrainte", "séquence étapes protocole",
    "afin d'améliorer", "en revanche comparaison",
    "dépend des données", "algorithme contrôle",
]

_QUERIES_EN_DEFAULT: list[str] = [
    "causal inference", "cause effect", "causal mechanism",
    "conditional probability", "if then implication",
    "although despite results", "sequence pipeline steps",
    "in order to improve", "in contrast to baseline",
    "depends on data", "algorithm controls training",
    "enables learning", "prevents overfitting",
]


class HALScraper:
    """Extrait des résumés de papiers scientifiques FR+EN depuis HAL."""

    @property
    def BASE_URL(self) -> str:
        return get_source_config("hal").get("api_url", "https://api.archives-ouvertes.fr/search/")

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        cfg = get_source_config("hal")
        self._delay = float(cfg.get("rate_limit_delay", 1.0))
        # Requêtes lues depuis sources.yaml — fallback sur les valeurs par défaut.
        self.QUERIES_FR: list[str] = cfg.get("queries_fr", _QUERIES_FR_DEFAULT)
        self.QUERIES_EN: list[str] = cfg.get("queries_en", _QUERIES_EN_DEFAULT)

    def search_papers(self, query: str, max_results: int = 100, start: int = 0) -> list[dict]:
        """Recherche des papiers sur HAL avec retry réseau + vraie pagination."""
        docs: list[dict] = []
        offset = max(0, start)
        remaining = max(0, max_results)
        while remaining > 0:
            rows = min(remaining, 100)
            params = {
                "q": query,
                "wt": "json",
                "rows": rows,
                "start": offset,
                "fl": "docid,label_s,abstract_s,language_s,uri_s",
            }
            resp, self.session = _retry_get(self.session, self.BASE_URL, params,  # noqa: B009
                                             user_agent=self.user_agent,
                                             min_interval=self._delay, base_delay=1.0)
            if resp is None:
                print(f"  HAL '{query}': toutes tentatives échouées, skip")
                break
            try:
                page = resp.json().get("response", {}).get("docs", [])
            except Exception as e:
                print(f"  HAL parse error: {e}")
                break
            if not page:
                break
            docs.extend(page)
            if len(page) < rows:
                break  # dernière page
            offset += len(page)
            remaining -= len(page)
        return docs[:max_results]

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
        docid = doc.get("docid", "")
        url = doc.get("uri_s") or (f"https://hal.science/{docid}" if docid else "")
        langs = doc.get("language_s", [])
        if isinstance(langs, str):  # l'API renvoie parfois un scalaire (Audit 3 M5)
            langs = [langs]
        lang = langs[0] if langs else "fr"
        if lang.lower().startswith("en"):
            lang = "en"
        elif lang.lower().startswith("fr"):
            lang = "fr"
        titles = doc.get("label_s", [""])
        if isinstance(titles, str):
            titles = [titles]
        title = titles[0] if titles else ""
        # Principe : scraper le SITE lié, pas la sortie du moteur.
        text = self._fetch_paper_text(url) if url else None
        time.sleep(self._delay)
        if not text:
            abstracts = doc.get("abstract_s", [])
            if isinstance(abstracts, str):
                abstracts = [abstracts]
            for abstract in abstracts:
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

    def scrape(self, max_per_query: int = 100, seed: int | None = None, tracker=None) -> list[dict]:
        """Scrape toutes les queries FR+EN (ordre mélangé + offset rotatif si seed)."""
        from ..diversity import shuffled as _shuffled, page_offset as _offset
        queries = _shuffled(self.QUERIES_FR + self.QUERIES_EN, seed, "hal") if seed is not None else list(self.QUERIES_FR + self.QUERIES_EN)
        start = _offset(seed, "hal", min(max_per_query, 100)) if seed is not None else 0
        if start:
            print(f"  HAL: offset pagination {start} (seed={seed})")
        all_papers = []
        consecutive_failures = 0
        for query in queries:
            if tracker is not None and tracker.is_globally_full():
                break
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
