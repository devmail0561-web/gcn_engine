"""Scraper arXiv via l'API Atom/REST — gratuit, sans token."""
from __future__ import annotations
import time
import xml.etree.ElementTree as ET
import requests


class ArXivScraper:
    BASE_URL = "http://export.arxiv.org/api/query"

    # Queries ciblées par type de relation
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

    def __init__(self, user_agent: str = "GCN-Dataset/2.0 (research)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def search(self, query: str, max_results: int = 100, start: int = 0) -> list[dict]:
        """Recherche des papers et retourne leurs abstracts."""
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": min(max_results, 200),
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  arXiv erreur: {e}")
            return []

        results = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                link_el = entry.find("atom:id", ns)
                if summary_el is not None and summary_el.text:
                    results.append({
                        "title": title_el.text.strip() if title_el is not None else "",
                        "text": summary_el.text.strip().replace("\n", " "),
                        "url": link_el.text.strip() if link_el is not None else "",
                        "source": f"arxiv:{query}",
                        "lang": "en",
                    })
        except ET.ParseError:
            pass
        return results

    def scrape(self, max_per_query: int = 100) -> list[dict]:
        """Scrape tous les topics, respecte le rate limit arXiv (3s entre requêtes)."""
        results = []
        for relation_type, queries in self.QUERIES.items():
            for q in queries:
                print(f"  arXiv [{relation_type}]: '{q}'...", end=" ", flush=True)
                papers = self.search(q, max_results=max_per_query)
                for p in papers:
                    p["relation_hint"] = relation_type
                results.extend(papers)
                print(f"{len(papers)} abstracts")
                time.sleep(3)
        return results
