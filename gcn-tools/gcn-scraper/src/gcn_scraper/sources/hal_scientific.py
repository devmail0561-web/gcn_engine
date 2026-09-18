"""Scraper HAL (archives ouvertes) — queries ciblées FR+EN, pagination."""
from __future__ import annotations
import time
import requests
from ._net import retry_get as _retry_get
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

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

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
                                         user_agent=self.user_agent)
        if resp is None:
            print(f"  HAL '{query}': toutes tentatives échouées, skip")
            return []
        try:
            return resp.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  HAL parse error: {e}")
            return []

    def _doc_to_item(self, doc: dict, query: str) -> dict | None:
        """Extrait l'abstract d'un doc HAL."""
        for abstract in doc.get("abstract_s", []):
            if abstract and len(abstract) > 50:
                langs = doc.get("language_s", [])
                lang = langs[0] if langs else "fr"
                if lang.lower().startswith("en"):
                    lang = "en"
                elif lang.lower().startswith("fr"):
                    lang = "fr"
                return {
                    "title": doc.get("label_s", [""])[0] if doc.get("label_s") else "",
                    "text": abstract,
                    "url": doc.get("uri_s", f"https://hal.science/{doc.get('docid', '')}"),
                    "source": f"hal:{query}",
                    "lang": lang,
                }
        return None

    def scrape(self, max_per_query: int = 100) -> list[dict]:
        """Scrape toutes les queries FR+EN."""
        all_papers = []
        for query in self.QUERIES_FR + self.QUERIES_EN:
            docs = self.search_papers(query, max_per_query)
            for doc in docs:
                item = self._doc_to_item(doc, query)
                if item:
                    all_papers.append(item)
            print(f"  HAL '{query}': {len(docs)} papiers")
            time.sleep(1)
        return all_papers
