"""Scraper HAL (archives ouvertes scientifiques) via REST API."""

import time
import requests


class HALScraper:
    """Extrait des resumes de papiers scientifiques francais."""

    BASE_URL = "https://api.archives-ouvertes.fr/search/"

    def __init__(self, user_agent: str = "GCN-Dataset/1.0 (research)"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def search_papers(self, query: str, max_results: int = 100) -> list[dict]:
        """Recherche des papiers sur HAL."""
        params = {
            "q": query,
            "wt": "json",
            "rows": min(max_results, 100),
            "fl": "docid,label_s,abstract_s,authFullName_s",
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  Erreur HAL: {e}")
            return []

    def extract_text(self, paper: dict) -> str | None:
        """Extrait l'abstract d'un papier."""
        for abstract in paper.get("abstract_s", []):
            if abstract and len(abstract) > 50:
                return abstract
        return None

    def scrape(self, queries: list[str], max_per_query: int = 50) -> list[dict]:
        """Scrape plusieurs requetes."""
        all_papers = []
        for query in queries:
            docs = self.search_papers(query, max_per_query)
            for doc in docs:
                text = self.extract_text(doc)
                if text:
                    all_papers.append({
                        "title": doc.get("label_s", [""])[0] if doc.get("label_s") else "",
                        "text": text,
                        "url": f"https://hal.science/{doc.get('docid', '')}",
                        "source": f"hal:{query}",
                    })
            print(f"  {query}: {len(docs)} papiers")
            time.sleep(1)
        return all_papers
